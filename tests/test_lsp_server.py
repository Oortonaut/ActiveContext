"""Tests for LSP server lifecycle and request dispatch."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from activecontext.transport.lsp.server import LSPServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lsp_frame(msg: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message with Content-Length framing."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _feed_messages(reader: asyncio.StreamReader, *messages: dict[str, Any]) -> None:
    """Feed one or more LSP-framed messages into a StreamReader."""
    for msg in messages:
        reader.feed_data(_make_lsp_frame(msg))
    reader.feed_eof()


class MockTransport:
    """Minimal mock transport that captures written bytes."""

    def __init__(self) -> None:
        self.data = bytearray()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default

    def is_closing(self) -> bool:
        return False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    def close(self) -> None:
        pass


def _make_mock_writer() -> tuple[asyncio.StreamWriter, MockTransport]:
    """Create a StreamWriter backed by MockTransport."""
    mock_transport = MockTransport()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    writer = asyncio.StreamWriter(mock_transport, protocol, reader, asyncio.get_event_loop())
    return writer, mock_transport


def _parse_responses(transport: MockTransport) -> list[dict[str, Any]]:
    """Parse all LSP-framed responses from a MockTransport's captured data."""
    data = bytes(transport.data)
    responses: list[dict[str, Any]] = []

    while data:
        # Find header/body separator
        sep_idx = data.find(b"\r\n\r\n")
        if sep_idx == -1:
            break

        # Parse Content-Length from header
        header_text = data[:sep_idx].decode("ascii")
        content_length = None
        for line in header_text.split("\r\n"):
            if line.startswith("Content-Length:"):
                content_length = int(line.split(":")[1].strip())
                break

        if content_length is None:
            break

        # Extract body
        body_start = sep_idx + 4
        body_end = body_start + content_length
        if body_end > len(data):
            break

        body = data[body_start:body_end]
        responses.append(json.loads(body.decode("utf-8")))
        data = data[body_end:]

    return responses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitializeHandshake:
    """Test initialize request/response handshake."""

    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self) -> None:
        """Server responds to initialize with capabilities and serverInfo."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"capabilities": {}},
            },
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        assert len(responses) == 1

        resp = responses[0]
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "capabilities" in resp["result"]
        assert resp["result"]["capabilities"]["textDocumentSync"] == 1
        assert resp["result"]["capabilities"]["workspace"]["workspaceFolders"]["supported"] is True
        assert resp["result"]["serverInfo"]["name"] == "activecontext"
        assert resp["result"]["serverInfo"]["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_initialize_includes_custom_capabilities(self) -> None:
        """Custom capabilities set via set_capabilities appear in response."""
        server = LSPServer()
        server.set_capabilities({"completionProvider": {"triggerCharacters": ["."]}})

        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            },
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        result = responses[0]["result"]
        assert result["capabilities"]["completionProvider"] == {"triggerCharacters": ["."]}
        # Default capabilities still present
        assert result["capabilities"]["textDocumentSync"] == 1


class TestInitializedNotification:
    """Test initialized notification handling."""

    @pytest.mark.asyncio
    async def test_initialized_sets_flag(self) -> None:
        """Server marks itself as initialized after receiving notification."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        assert not server.initialized

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            },
            {"jsonrpc": "2.0", "method": "initialized"},
        )

        await server.serve(reader, writer)

        assert server.initialized


class TestShutdownExit:
    """Test shutdown/exit lifecycle."""

    @pytest.mark.asyncio
    async def test_shutdown_responds_with_null(self) -> None:
        """Shutdown request receives null result response."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            },
            {"jsonrpc": "2.0", "method": "initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        # Response 0 = initialize, Response 1 = shutdown
        assert len(responses) == 2
        shutdown_resp = responses[1]
        assert shutdown_resp["id"] == 2
        assert shutdown_resp["result"] is None

    @pytest.mark.asyncio
    async def test_exit_stops_server_loop(self) -> None:
        """Exit notification causes serve() to return."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            },
            {"jsonrpc": "2.0", "method": "initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
            {"jsonrpc": "2.0", "method": "exit"},
        )

        await server.serve(reader, writer)

        assert server.shutdown_requested

    @pytest.mark.asyncio
    async def test_full_lifecycle(self) -> None:
        """Complete initialize -> initialized -> shutdown -> exit cycle."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
            {"jsonrpc": "2.0", "method": "exit"},
        )

        await server.serve(reader, writer)

        assert server.initialized
        assert server.shutdown_requested

        responses = _parse_responses(transport)
        assert len(responses) == 2
        assert responses[0]["id"] == 1  # initialize response
        assert responses[1]["id"] == 2  # shutdown response


class TestCustomHandlers:
    """Test custom handler registration and dispatch."""

    @pytest.mark.asyncio
    async def test_registered_handler_called(self) -> None:
        """A registered handler receives params and its result is sent back."""
        server = LSPServer()
        received_params: list[dict[str, Any]] = []

        async def handle_completion(params: dict[str, Any]) -> dict[str, Any]:
            received_params.append(params)
            return {"items": [{"label": "hello"}]}

        server.on("textDocument/completion", handle_completion)

        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "textDocument/completion",
                "params": {
                    "textDocument": {"uri": "file:///test.py"},
                    "position": {"line": 0, "character": 0},
                },
            },
        )

        await server.serve(reader, writer)

        # Handler was called with correct params
        assert len(received_params) == 1
        assert received_params[0]["textDocument"]["uri"] == "file:///test.py"

        # Response sent back
        responses = _parse_responses(transport)
        assert len(responses) == 1
        assert responses[0]["id"] == 1
        assert responses[0]["result"] == {"items": [{"label": "hello"}]}

    @pytest.mark.asyncio
    async def test_handler_for_notification_no_response(self) -> None:
        """Handler for a notification (no id) is called but no response is sent."""
        server = LSPServer()
        called = False

        async def handle_did_open(params: dict[str, Any]) -> None:
            nonlocal called
            called = True

        server.on("textDocument/didOpen", handle_did_open)

        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {"textDocument": {"uri": "file:///test.py"}},
            },
        )

        await server.serve(reader, writer)

        assert called
        # No response for notifications
        responses = _parse_responses(transport)
        assert len(responses) == 0

    @pytest.mark.asyncio
    async def test_handler_with_empty_params(self) -> None:
        """Handler receives empty dict when params are missing from message."""
        server = LSPServer()
        received: list[dict[str, Any]] = []

        async def handle_custom(params: dict[str, Any]) -> str:
            received.append(params)
            return "ok"

        server.on("custom/method", handle_custom)

        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "custom/method"},
        )

        await server.serve(reader, writer)

        assert received == [{}]


class TestUnknownMethod:
    """Test handling of unknown/unregistered methods."""

    @pytest.mark.asyncio
    async def test_unknown_request_returns_error(self) -> None:
        """Unknown method with id returns MethodNotFound error."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "nonexistent/method",
            },
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        assert len(responses) == 1
        resp = responses[0]
        assert resp["id"] == 42
        assert "error" in resp
        assert resp["error"]["code"] == -32601
        assert "nonexistent/method" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_unknown_notification_ignored(self) -> None:
        """Unknown notification (no id) is silently ignored."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "method": "unknown/notification"},
        )

        await server.serve(reader, writer)

        # No response or error
        responses = _parse_responses(transport)
        assert len(responses) == 0


class TestCancelRequest:
    """Test $/cancelRequest handling."""

    @pytest.mark.asyncio
    async def test_cancel_request_does_not_error(self) -> None:
        """$/cancelRequest is accepted without error (best effort)."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "$/cancelRequest",
                "params": {"id": 5},
            },
        )

        await server.serve(reader, writer)

        # No response expected for cancel notification
        responses = _parse_responses(transport)
        assert len(responses) == 0


class TestSendNotification:
    """Test server-initiated notifications."""

    @pytest.mark.asyncio
    async def test_send_notification_with_params(self) -> None:
        """Server can send notification to client during message handling."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        async def handle_test(params: dict[str, Any]) -> str:
            # Send a notification as a side effect
            await server.send_notification(
                "window/logMessage",
                {"type": 3, "message": "Processing request"},
            )
            return "done"

        server.on("custom/test", handle_test)

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "custom/test", "params": {}},
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        assert len(responses) == 2

        # First response is the notification
        notif = responses[0]
        assert notif["method"] == "window/logMessage"
        assert notif["params"]["type"] == 3
        assert notif["params"]["message"] == "Processing request"
        assert "id" not in notif

        # Second response is the handler result
        result = responses[1]
        assert result["id"] == 1
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_send_notification_without_params(self) -> None:
        """Notification without params omits the params field."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        async def handle_test(params: dict[str, Any]) -> None:
            await server.send_notification("custom/event")
            return None

        server.on("custom/trigger", handle_test)

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "custom/trigger", "params": {}},
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        # Notification + response
        notif = responses[0]
        assert notif["method"] == "custom/event"
        assert "params" not in notif

    @pytest.mark.asyncio
    async def test_send_notification_without_writer_raises(self) -> None:
        """Sending notification when not serving raises AssertionError."""
        server = LSPServer()
        with pytest.raises(AssertionError, match="no writer"):
            await server.send_notification("test/method")


class TestMultipleRequests:
    """Test multiple requests in sequence."""

    @pytest.mark.asyncio
    async def test_sequential_requests(self) -> None:
        """Multiple requests are processed in order with correct responses."""
        server = LSPServer()
        call_count = 0

        async def handle_ping(params: dict[str, Any]) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"pong": call_count}

        server.on("custom/ping", handle_ping)

        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "custom/ping", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "custom/ping", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "custom/ping", "params": {}},
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        assert len(responses) == 3
        assert responses[0] == {"jsonrpc": "2.0", "id": 1, "result": {"pong": 1}}
        assert responses[1] == {"jsonrpc": "2.0", "id": 2, "result": {"pong": 2}}
        assert responses[2] == {"jsonrpc": "2.0", "id": 3, "result": {"pong": 3}}

    @pytest.mark.asyncio
    async def test_mixed_requests_and_notifications(self) -> None:
        """Interleaved requests and notifications are handled correctly."""
        server = LSPServer()
        events: list[str] = []

        async def handle_request(params: dict[str, Any]) -> str:
            events.append("request")
            return "ok"

        async def handle_notify(params: dict[str, Any]) -> None:
            events.append("notify")

        server.on("custom/request", handle_request)
        server.on("custom/notify", handle_notify)

        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "custom/request", "params": {}},
            {"jsonrpc": "2.0", "method": "custom/notify", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "custom/request", "params": {}},
        )

        await server.serve(reader, writer)

        assert events == ["request", "notify", "request"]

        # Only requests (with id) get responses
        responses = _parse_responses(transport)
        assert len(responses) == 2
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2


class TestEOFHandling:
    """Test server behavior on EOF."""

    @pytest.mark.asyncio
    async def test_eof_stops_server(self) -> None:
        """Server stops cleanly when reader reaches EOF."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        # Feed EOF immediately
        reader.feed_eof()

        await server.serve(reader, writer)

        # Server exited cleanly without shutdown_requested
        assert not server.shutdown_requested

    @pytest.mark.asyncio
    async def test_eof_after_messages(self) -> None:
        """Server processes messages then stops on EOF."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        )

        await server.serve(reader, writer)

        responses = _parse_responses(transport)
        assert len(responses) == 1
        assert responses[0]["id"] == 1
