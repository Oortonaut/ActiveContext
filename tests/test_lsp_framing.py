"""Tests for LSP message framing."""

from __future__ import annotations

import asyncio
import pytest

from activecontext.transport.lsp.framing import (
    LSPFramingError,
    parse_header,
    read_message,
    write_message,
)


class TestParseHeader:
    """Tests for parse_header function."""

    def test_basic_content_length(self) -> None:
        """Parse simple Content-Length header."""
        header = b"Content-Length: 42"
        result = parse_header(header)
        assert result == {"Content-Length": "42"}

    def test_content_length_with_content_type(self) -> None:
        """Parse Content-Length with optional Content-Type."""
        header = b"Content-Length: 100\r\nContent-Type: application/json"
        result = parse_header(header)
        assert result == {
            "Content-Length": "100",
            "Content-Type": "application/json",
        }

    def test_whitespace_handling(self) -> None:
        """Header values can have leading/trailing whitespace."""
        header = b"Content-Length:   42  "
        result = parse_header(header)
        assert result["Content-Length"] == "42"

    def test_missing_content_length_raises(self) -> None:
        """Missing Content-Length header raises error."""
        header = b"Content-Type: application/json"
        with pytest.raises(LSPFramingError, match="Missing required Content-Length"):
            parse_header(header)

    def test_empty_header_raises(self) -> None:
        """Empty header block raises error."""
        with pytest.raises(LSPFramingError, match="Empty header block"):
            parse_header(b"")

    def test_invalid_content_length_raises(self) -> None:
        """Non-integer Content-Length raises error."""
        header = b"Content-Length: abc"
        with pytest.raises(LSPFramingError, match="Invalid Content-Length"):
            parse_header(header)

    def test_negative_content_length_raises(self) -> None:
        """Negative Content-Length raises error."""
        header = b"Content-Length: -5"
        with pytest.raises(LSPFramingError, match="Negative Content-Length"):
            parse_header(header)

    def test_malformed_header_line_raises(self) -> None:
        """Header line without colon raises error."""
        header = b"Content-Length 42"
        with pytest.raises(LSPFramingError, match="no colon"):
            parse_header(header)

    def test_zero_content_length_valid(self) -> None:
        """Zero Content-Length is valid."""
        header = b"Content-Length: 0"
        result = parse_header(header)
        assert result["Content-Length"] == "0"


class TestReadMessage:
    """Tests for read_message function."""

    @pytest.fixture
    def make_reader(self) -> asyncio.StreamReader:
        """Create a StreamReader with test data."""

        def _make_reader(data: bytes) -> asyncio.StreamReader:
            reader = asyncio.StreamReader()
            reader.feed_data(data)
            reader.feed_eof()
            return reader

        return _make_reader

    @pytest.mark.asyncio
    async def test_read_simple_message(self, make_reader) -> None:
        """Read a simple JSON-RPC message."""
        body = b'{"jsonrpc":"2.0","id":1,"method":"test"}'
        data = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        reader = make_reader(data)

        msg = await read_message(reader)

        assert msg == {"jsonrpc": "2.0", "id": 1, "method": "test"}

    @pytest.mark.asyncio
    async def test_read_returns_none_on_eof(self, make_reader) -> None:
        """EOF at message boundary returns None."""
        reader = make_reader(b"")

        msg = await read_message(reader)

        assert msg is None

    @pytest.mark.asyncio
    async def test_read_multiple_messages(self, make_reader) -> None:
        """Read multiple sequential messages."""
        body1 = b'{"jsonrpc":"2.0","id":1,"method":"foo"}'
        body2 = b'{"jsonrpc":"2.0","id":2,"method":"bar"}'
        data = (
            f"Content-Length: {len(body1)}\r\n\r\n".encode()
            + body1
            + f"Content-Length: {len(body2)}\r\n\r\n".encode()
            + body2
        )
        reader = make_reader(data)

        msg1 = await read_message(reader)
        msg2 = await read_message(reader)
        msg3 = await read_message(reader)

        assert msg1["method"] == "foo"
        assert msg2["method"] == "bar"
        assert msg3 is None

    @pytest.mark.asyncio
    async def test_read_with_content_type_header(self, make_reader) -> None:
        """Optional Content-Type header is accepted."""
        body = b'{"jsonrpc":"2.0","id":1}'
        data = (
            f"Content-Length: {len(body)}\r\n"
            f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
            f"\r\n"
        ).encode() + body
        reader = make_reader(data)

        msg = await read_message(reader)

        assert msg == {"jsonrpc": "2.0", "id": 1}

    @pytest.mark.asyncio
    async def test_read_unicode_content(self, make_reader) -> None:
        """Unicode content is properly decoded."""
        body = '{"message":"Hello, \u4e16\u754c!"}'.encode("utf-8")
        data = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        reader = make_reader(data)

        msg = await read_message(reader)

        assert msg == {"message": "Hello, \u4e16\u754c!"}

    @pytest.mark.asyncio
    async def test_read_incomplete_body_raises(self, make_reader) -> None:
        """Incomplete message body raises error."""
        data = b'Content-Length: 100\r\n\r\n{"partial":'
        reader = make_reader(data)

        with pytest.raises(LSPFramingError, match="Incomplete message body"):
            await read_message(reader)

    @pytest.mark.asyncio
    async def test_read_invalid_json_raises(self, make_reader) -> None:
        """Invalid JSON content raises error."""
        body = b"not valid json"
        data = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        reader = make_reader(data)

        with pytest.raises(LSPFramingError, match="Invalid JSON"):
            await read_message(reader)

    @pytest.mark.asyncio
    async def test_read_non_object_json_raises(self, make_reader) -> None:
        """JSON that isn't an object raises error."""
        body = b"[1, 2, 3]"
        data = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        reader = make_reader(data)

        with pytest.raises(LSPFramingError, match="must be an object"):
            await read_message(reader)

    @pytest.mark.asyncio
    async def test_read_message_size_limit(self, make_reader) -> None:
        """Messages exceeding size limit are rejected."""
        body = b'{"data":"x"}'
        data = f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        reader = make_reader(data)

        with pytest.raises(LSPFramingError, match="exceeds maximum"):
            await read_message(reader, max_message_size=5)


class TestWriteMessage:
    """Tests for write_message function."""

    @pytest.mark.asyncio
    async def test_write_simple_message(self) -> None:
        """Write a simple JSON-RPC message."""
        reader = asyncio.StreamReader()
        # Create a mock transport that captures written data
        written_data = bytearray()

        class MockTransport:
            def get_extra_info(self, name: str, default=None):
                return default

            def is_closing(self) -> bool:
                return False

            def write(self, data: bytes) -> None:
                written_data.extend(data)

            def close(self) -> None:
                pass

        transport = MockTransport()
        protocol = asyncio.StreamReaderProtocol(reader)
        writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())

        msg = {"jsonrpc": "2.0", "id": 1, "result": {"success": True}}
        await write_message(writer, msg, drain=False)

        # Verify format
        result = bytes(written_data)
        assert result.startswith(b"Content-Length: ")
        assert b"\r\n\r\n" in result

        # Extract and verify body
        header_end = result.index(b"\r\n\r\n") + 4
        body = result[header_end:]
        import json

        parsed = json.loads(body.decode("utf-8"))
        assert parsed == msg

    @pytest.mark.asyncio
    async def test_write_unicode_message(self) -> None:
        """Unicode content is properly encoded."""
        reader = asyncio.StreamReader()
        written_data = bytearray()

        class MockTransport:
            def get_extra_info(self, name: str, default=None):
                return default

            def is_closing(self) -> bool:
                return False

            def write(self, data: bytes) -> None:
                written_data.extend(data)

            def close(self) -> None:
                pass

        transport = MockTransport()
        protocol = asyncio.StreamReaderProtocol(reader)
        writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())

        msg = {"message": "Hello, \u4e16\u754c!"}
        await write_message(writer, msg, drain=False)

        result = bytes(written_data)
        header_end = result.index(b"\r\n\r\n") + 4
        body = result[header_end:]
        import json

        parsed = json.loads(body.decode("utf-8"))
        assert parsed["message"] == "Hello, \u4e16\u754c!"

    @pytest.mark.asyncio
    async def test_content_length_matches_body(self) -> None:
        """Content-Length header matches actual body size."""
        reader = asyncio.StreamReader()
        written_data = bytearray()

        class MockTransport:
            def get_extra_info(self, name: str, default=None):
                return default

            def is_closing(self) -> bool:
                return False

            def write(self, data: bytes) -> None:
                written_data.extend(data)

            def close(self) -> None:
                pass

        transport = MockTransport()
        protocol = asyncio.StreamReaderProtocol(reader)
        writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())

        msg = {"jsonrpc": "2.0", "id": 42}
        await write_message(writer, msg, drain=False)

        result = bytes(written_data)
        # Parse Content-Length
        header_end = result.index(b"\r\n\r\n")
        headers = result[:header_end].decode("ascii")
        content_length = int(headers.split(":")[1].strip())

        # Verify body length
        body = result[header_end + 4 :]
        assert len(body) == content_length

    @pytest.mark.asyncio
    async def test_write_non_serializable_raises(self) -> None:
        """Non-serializable content raises error."""
        reader = asyncio.StreamReader()

        class MockTransport:
            def get_extra_info(self, name: str, default=None):
                return default

            def is_closing(self) -> bool:
                return False

            def write(self, data: bytes) -> None:
                pass

            def close(self) -> None:
                pass

        transport = MockTransport()
        protocol = asyncio.StreamReaderProtocol(reader)
        writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())

        msg = {"callback": lambda x: x}  # Functions aren't JSON-serializable

        with pytest.raises(LSPFramingError, match="cannot be serialized"):
            await write_message(writer, msg, drain=False)


class TestRoundTrip:
    """End-to-end read/write tests."""

    @pytest.mark.asyncio
    async def test_write_then_read(self) -> None:
        """Message survives write + read cycle."""
        original = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.py",
                    "languageId": "python",
                    "version": 1,
                    "text": "print('hello')",
                }
            },
        }

        # Write to buffer
        reader = asyncio.StreamReader()
        written_data = bytearray()

        class MockTransport:
            def get_extra_info(self, name: str, default=None):
                return default

            def is_closing(self) -> bool:
                return False

            def write(self, data: bytes) -> None:
                written_data.extend(data)

            def close(self) -> None:
                pass

        transport = MockTransport()
        protocol = asyncio.StreamReaderProtocol(reader)
        writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())

        await write_message(writer, original, drain=False)

        # Read back
        reader2 = asyncio.StreamReader()
        reader2.feed_data(bytes(written_data))
        reader2.feed_eof()

        result = await read_message(reader2)

        assert result == original
