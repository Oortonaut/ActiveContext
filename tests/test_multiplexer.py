"""Tests for ACP/LSP multiplexed transport."""

from __future__ import annotations

import asyncio
import json

import pytest

from activecontext.transport.multiplexer import (
    MultiplexedTransport,
    ProtocolMode,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_acp_message(msg: dict) -> bytes:
    """Encode a dict as a newline-delimited JSON line (ACP format)."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def _make_lsp_message(msg: dict) -> bytes:
    """Encode a dict as an LSP Content-Length framed message."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _make_reader_writer():
    """Return (reader, writer, written_data) backed by an in-memory buffer."""
    reader = asyncio.StreamReader()
    written = bytearray()

    class _Transport:
        def get_extra_info(self, name, default=None):
            return default

        def is_closing(self):
            return False

        def write(self, data: bytes):
            written.extend(data)

        def close(self):
            pass

    transport = _Transport()
    protocol = asyncio.StreamReaderProtocol(reader)
    writer = asyncio.StreamWriter(transport, protocol, reader, asyncio.get_event_loop())
    return reader, writer, written


# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------


class TestDetectProtocol:
    """Tests for MultiplexedTransport.detect_protocol."""

    def test_detect_acp_from_json_line(self) -> None:
        """ACP messages start with '{' and are detected correctly."""
        transport = MultiplexedTransport()
        assert transport.detect_protocol(b'{"jsonrpc":"2.0"}') == ProtocolMode.ACP

    def test_detect_lsp_from_content_length_header(self) -> None:
        """LSP messages start with 'Content-Length:' and are detected correctly."""
        transport = MultiplexedTransport()
        assert transport.detect_protocol(b"Content-Length: 42\r\n") == ProtocolMode.LSP

    def test_detect_acp_with_leading_whitespace(self) -> None:
        """Leading whitespace is stripped before detection."""
        transport = MultiplexedTransport()
        assert transport.detect_protocol(b"  {") == ProtocolMode.ACP

    def test_detect_lsp_with_leading_whitespace(self) -> None:
        """Leading whitespace is stripped before detection."""
        transport = MultiplexedTransport()
        assert transport.detect_protocol(b"  Content-Length: 10") == ProtocolMode.LSP

    def test_invalid_data_raises_value_error(self) -> None:
        """Data that matches neither protocol raises ValueError."""
        transport = MultiplexedTransport()
        with pytest.raises(ValueError, match="Cannot detect protocol"):
            transport.detect_protocol(b"HELLO WORLD")

    def test_empty_data_raises_value_error(self) -> None:
        """Empty/whitespace-only data raises ValueError."""
        transport = MultiplexedTransport()
        with pytest.raises(ValueError, match="Cannot detect protocol"):
            transport.detect_protocol(b"   ")


# ---------------------------------------------------------------------------
# ACP-only mode
# ---------------------------------------------------------------------------


class TestACPOnlyMode:
    """In ACP mode, all messages are routed to the ACP handler."""

    @pytest.mark.asyncio
    async def test_acp_mode_routes_to_acp_handler(self) -> None:
        """ACP-only mode routes a JSON line to on_acp_message."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_acp_message({"jsonrpc": "2.0", "method": "test"}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)

        # Give the read loop time to process.
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 1
        assert received[0]["method"] == "test"

    @pytest.mark.asyncio
    async def test_acp_mode_multiple_messages(self) -> None:
        """ACP-only mode correctly reads multiple sequential JSON lines."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_acp_message({"id": 1}))
        reader.feed_data(_make_acp_message({"id": 2}))
        reader.feed_data(_make_acp_message({"id": 3}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 3
        assert [m["id"] for m in received] == [1, 2, 3]


# ---------------------------------------------------------------------------
# LSP-only mode
# ---------------------------------------------------------------------------


class TestLSPOnlyMode:
    """In LSP mode, all messages are routed to the LSP handler."""

    @pytest.mark.asyncio
    async def test_lsp_mode_routes_to_lsp_handler(self) -> None:
        """LSP-only mode routes a Content-Length framed message to on_lsp_message."""
        received: list[dict] = []

        async def on_lsp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_lsp_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.LSP,
            on_lsp_message=on_lsp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 1
        assert received[0]["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_lsp_mode_multiple_messages(self) -> None:
        """LSP-only mode reads multiple sequential framed messages."""
        received: list[dict] = []

        async def on_lsp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_lsp_message({"id": 1, "method": "foo"}))
        reader.feed_data(_make_lsp_message({"id": 2, "method": "bar"}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.LSP,
            on_lsp_message=on_lsp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 2
        assert received[0]["method"] == "foo"
        assert received[1]["method"] == "bar"


# ---------------------------------------------------------------------------
# Multiplexed mode
# ---------------------------------------------------------------------------


class TestMultiplexedMode:
    """In MULTIPLEXED mode, messages are routed based on content."""

    @pytest.mark.asyncio
    async def test_acp_message_routed_correctly(self) -> None:
        """ACP messages in multiplexed mode go to the ACP handler."""
        acp_received: list[dict] = []
        lsp_received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            acp_received.append(msg)

        async def on_lsp(msg: dict) -> None:
            lsp_received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_acp_message({"jsonrpc": "2.0", "method": "acp/test"}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.MULTIPLEXED,
            on_acp_message=on_acp,
            on_lsp_message=on_lsp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(acp_received) == 1
        assert len(lsp_received) == 0
        assert acp_received[0]["method"] == "acp/test"

    @pytest.mark.asyncio
    async def test_lsp_message_routed_correctly(self) -> None:
        """LSP messages in multiplexed mode go to the LSP handler."""
        acp_received: list[dict] = []
        lsp_received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            acp_received.append(msg)

        async def on_lsp(msg: dict) -> None:
            lsp_received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_lsp_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.MULTIPLEXED,
            on_acp_message=on_acp,
            on_lsp_message=on_lsp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(acp_received) == 0
        assert len(lsp_received) == 1
        assert lsp_received[0]["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_mixed_interleaved_messages(self) -> None:
        """Interleaved ACP and LSP messages are routed to correct handlers."""
        acp_received: list[dict] = []
        lsp_received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            acp_received.append(msg)

        async def on_lsp(msg: dict) -> None:
            lsp_received.append(msg)

        reader, writer, _ = _make_reader_writer()
        # ACP, LSP, ACP, LSP
        reader.feed_data(_make_acp_message({"type": "acp", "seq": 1}))
        reader.feed_data(_make_lsp_message({"jsonrpc": "2.0", "id": 1, "method": "lsp1"}))
        reader.feed_data(_make_acp_message({"type": "acp", "seq": 2}))
        reader.feed_data(_make_lsp_message({"jsonrpc": "2.0", "id": 2, "method": "lsp2"}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.MULTIPLEXED,
            on_acp_message=on_acp,
            on_lsp_message=on_lsp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.2)
        await transport.stop()

        assert len(acp_received) == 2
        assert len(lsp_received) == 2
        assert acp_received[0]["seq"] == 1
        assert acp_received[1]["seq"] == 2
        assert lsp_received[0]["method"] == "lsp1"
        assert lsp_received[1]["method"] == "lsp2"


# ---------------------------------------------------------------------------
# Invalid data handling
# ---------------------------------------------------------------------------


class TestInvalidData:
    """Invalid or unexpected data should not crash the transport."""

    @pytest.mark.asyncio
    async def test_invalid_json_skipped(self) -> None:
        """Malformed JSON lines are skipped without crashing."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(b"not valid json\n")
        reader.feed_data(_make_acp_message({"ok": True}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        # The valid message should still be received.
        assert len(received) == 1
        assert received[0]["ok"] is True

    @pytest.mark.asyncio
    async def test_non_object_json_skipped(self) -> None:
        """JSON arrays or scalars are skipped."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(b"[1, 2, 3]\n")
        reader.feed_data(_make_acp_message({"ok": True}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 1
        assert received[0]["ok"] is True

    @pytest.mark.asyncio
    async def test_unrecognised_data_skipped_in_multiplexed(self) -> None:
        """In multiplexed mode, data matching neither protocol is skipped."""
        acp_received: list[dict] = []
        lsp_received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            acp_received.append(msg)

        async def on_lsp(msg: dict) -> None:
            lsp_received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(b"GARBAGE DATA\n")
        reader.feed_data(_make_acp_message({"ok": True}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.MULTIPLEXED,
            on_acp_message=on_acp,
            on_lsp_message=on_lsp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(acp_received) == 1
        assert len(lsp_received) == 0

    @pytest.mark.asyncio
    async def test_blank_lines_skipped(self) -> None:
        """Blank lines between messages are silently skipped."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(b"\n\n")
        reader.feed_data(_make_acp_message({"ok": True}))
        reader.feed_data(b"\n")
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 1


# ---------------------------------------------------------------------------
# EOF handling
# ---------------------------------------------------------------------------


class TestEOFHandling:
    """The read loop exits cleanly on EOF."""

    @pytest.mark.asyncio
    async def test_immediate_eof(self) -> None:
        """Immediate EOF causes the read loop to exit without error."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_eof_after_messages(self) -> None:
        """Read loop exits after processing messages then hitting EOF."""
        received: list[dict] = []

        async def on_acp(msg: dict) -> None:
            received.append(msg)

        reader, writer, _ = _make_reader_writer()
        reader.feed_data(_make_acp_message({"id": 1}))
        reader.feed_data(_make_acp_message({"id": 2}))
        reader.feed_eof()

        transport = MultiplexedTransport(
            mode=ProtocolMode.ACP,
            on_acp_message=on_acp,
        )
        await transport.start(reader, writer)
        await asyncio.sleep(0.1)
        await transport.stop()

        assert len(received) == 2


# ---------------------------------------------------------------------------
# Send methods
# ---------------------------------------------------------------------------


class TestSendMethods:
    """Tests for send_acp and send_lsp."""

    @pytest.mark.asyncio
    async def test_send_acp_writes_newline_json(self) -> None:
        """send_acp writes newline-delimited JSON."""
        reader, writer, written = _make_reader_writer()
        reader.feed_eof()

        transport = MultiplexedTransport()
        await transport.start(reader, writer)

        msg = {"jsonrpc": "2.0", "method": "test"}
        await transport.send_acp(msg)
        await transport.stop()

        output = bytes(written).decode("utf-8")
        # Must end with newline.
        assert output.endswith("\n")
        # Must be valid JSON.
        parsed = json.loads(output.strip())
        assert parsed["method"] == "test"

    @pytest.mark.asyncio
    async def test_send_lsp_writes_content_length_framed(self) -> None:
        """send_lsp writes Content-Length framed message."""
        reader, writer, written = _make_reader_writer()
        reader.feed_eof()

        transport = MultiplexedTransport()
        await transport.start(reader, writer)

        msg = {"jsonrpc": "2.0", "id": 1, "result": {}}
        await transport.send_lsp(msg)
        await transport.stop()

        output = bytes(written)
        assert output.startswith(b"Content-Length: ")
        assert b"\r\n\r\n" in output

        # Extract and verify body.
        header_end = output.index(b"\r\n\r\n") + 4
        body = output[header_end:]
        parsed = json.loads(body.decode("utf-8"))
        assert parsed == msg

    @pytest.mark.asyncio
    async def test_send_before_start_raises(self) -> None:
        """Sending before start() raises RuntimeError."""
        transport = MultiplexedTransport()

        with pytest.raises(RuntimeError, match="Transport not started"):
            await transport.send_acp({"test": True})

        with pytest.raises(RuntimeError, match="Transport not started"):
            await transport.send_lsp({"test": True})


# ---------------------------------------------------------------------------
# Stop / cancellation
# ---------------------------------------------------------------------------


class TestStopBehaviour:
    """Transport stop and cancellation tests."""

    @pytest.mark.asyncio
    async def test_stop_cancels_read_loop(self) -> None:
        """Calling stop() cancels the read loop task."""
        reader, writer, _ = _make_reader_writer()
        # Don't feed EOF -- the reader will block indefinitely.

        transport = MultiplexedTransport(mode=ProtocolMode.ACP)
        await transport.start(reader, writer)

        # The read task should be running.
        assert transport._read_task is not None
        assert not transport._read_task.done()

        await transport.stop()

        # After stop, the task reference should be cleared.
        assert transport._read_task is None

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self) -> None:
        """Calling stop() twice does not raise."""
        reader, writer, _ = _make_reader_writer()
        reader.feed_eof()

        transport = MultiplexedTransport()
        await transport.start(reader, writer)
        await asyncio.sleep(0.05)

        await transport.stop()
        await transport.stop()  # Should not raise.
