"""Tests for WebSocketTransport.

Verifies:
- Basic WebSocket transport lifecycle (start/stop)
- Request/response handling
- Notification handling
- Server-initiated requests
- Error handling and recovery
- Connection loss and reconnection (if enabled)
- aiohttp dependency handling
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from activecontext.plugins.transport import JsonRpcError, PluginTransportError
from activecontext.plugins.wire import ErrorCodes


class MockWebSocketResponse:
    """Mock aiohttp WebSocketResponse."""

    def __init__(self) -> None:
        self.closed = False
        self.messages: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[bytes] = []
        self._exception: Exception | None = None

    async def send_bytes(self, data: bytes) -> None:
        """Mock send_bytes."""
        self.sent.append(data)

    async def close(self) -> None:
        """Mock close."""
        self.closed = True
        # Signal end of message stream
        await self.messages.put(None)

    def exception(self) -> Exception | None:
        """Mock exception."""
        return self._exception

    def __aiter__(self) -> MockWebSocketResponse:
        """Make async iterable."""
        return self

    async def __anext__(self) -> Any:
        """Return next message."""
        msg = await self.messages.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    async def push_text(self, text: str) -> None:
        """Push a text message to the queue."""
        msg = MagicMock()
        msg.type = 1  # WSMsgType.TEXT
        msg.data = text
        await self.messages.put(msg)

    async def push_binary(self, data: bytes) -> None:
        """Push a binary message to the queue."""
        msg = MagicMock()
        msg.type = 2  # WSMsgType.BINARY
        msg.data = data
        await self.messages.put(msg)

    async def push_close(self) -> None:
        """Push a close message to the queue."""
        msg = MagicMock()
        msg.type = 8  # WSMsgType.CLOSE
        await self.messages.put(msg)

    async def push_error(self, exc: Exception) -> None:
        """Push an error message to the queue."""
        self._exception = exc
        msg = MagicMock()
        msg.type = 9  # WSMsgType.ERROR
        await self.messages.put(msg)


class MockClientSession:
    """Mock aiohttp ClientSession."""

    def __init__(self) -> None:
        self.closed = False
        self.ws: MockWebSocketResponse | None = None

    async def ws_connect(self, url: str) -> MockWebSocketResponse:
        """Mock ws_connect."""
        self.ws = MockWebSocketResponse()
        return self.ws

    async def close(self) -> None:
        """Mock close."""
        self.closed = True


def _jsonrpc_response(result: Any, id: int | str) -> str:
    """Build a JSON-RPC response."""
    msg = {"jsonrpc": "2.0", "result": result, "id": id}
    return json.dumps(msg)


def _jsonrpc_error(code: int, message: str, id: int | str, data: Any = None) -> str:
    """Build a JSON-RPC error response."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    msg = {"jsonrpc": "2.0", "error": error, "id": id}
    return json.dumps(msg)


def _jsonrpc_notification(method: str, params: dict[str, Any] | None = None) -> str:
    """Build a JSON-RPC notification."""
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def _jsonrpc_request(method: str, id: int | str, params: dict[str, Any] | None = None) -> str:
    """Build a JSON-RPC request."""
    msg = {"jsonrpc": "2.0", "method": method, "id": id}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


class TestWebSocketTransportBasics:
    """Basic WebSocketTransport lifecycle tests."""

    @pytest.mark.asyncio
    async def test_import_without_aiohttp_raises_error(self) -> None:
        """Attempting to start WebSocketTransport without aiohttp raises helpful error."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        transport = WebSocketTransport(url="ws://localhost:8080")

        with (
            patch.dict("sys.modules", {"aiohttp": None}),
            patch("builtins.__import__", side_effect=ImportError("No module named 'aiohttp'")),
            pytest.raises(PluginTransportError, match="aiohttp.*pip install aiohttp"),
        ):
            await transport.start()

    @pytest.mark.asyncio
    async def test_start_creates_websocket_connection(self) -> None:
        """start() creates a WebSocket connection."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080/cap")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

        assert transport.is_running
        assert session.ws is not None
        assert not session.ws.closed

        await transport.stop()

    @pytest.mark.asyncio
    async def test_start_twice_raises_error(self) -> None:
        """Calling start() twice raises PluginTransportError."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()
            with pytest.raises(PluginTransportError, match="already started"):
                await transport.start()

        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_websocket_and_session(self) -> None:
        """stop() closes the WebSocket and HTTP session."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()
            await transport.stop()

        assert session.closed
        assert session.ws is not None
        assert session.ws.closed
        assert not transport.is_running

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """Calling stop() multiple times is safe."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()
            await transport.stop()
            await transport.stop()  # Should not raise

        assert not transport.is_running


class TestWebSocketTransportRequestResponse:
    """Request/response handling tests."""

    @pytest.mark.asyncio
    async def test_send_request_returns_result(self) -> None:
        """send_request() sends a request and returns the result."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Send request in background
            async def respond() -> None:
                await asyncio.sleep(0.01)
                assert session.ws is not None
                await session.ws.push_text(_jsonrpc_response({"status": "ok"}, 1))

            asyncio.create_task(respond())

            result = await transport.send_request("test_method", {"arg": "value"})
            assert result == {"status": "ok"}

            # Check sent data
            assert len(session.ws.sent) == 1
            sent_msg = json.loads(session.ws.sent[0].decode("utf-8"))
            assert sent_msg["method"] == "test_method"
            assert sent_msg["params"] == {"arg": "value"}
            assert sent_msg["id"] == 1

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_with_error_response_raises(self) -> None:
        """send_request() raises JsonRpcError on error response."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            async def respond_error() -> None:
                await asyncio.sleep(0.01)
                assert session.ws is not None
                await session.ws.push_text(
                    _jsonrpc_error(
                        ErrorCodes.INVALID_PARAMS,
                        "Bad arguments",
                        1,
                        data={"field": "arg"},
                    )
                )

            asyncio.create_task(respond_error())

            with pytest.raises(JsonRpcError) as exc_info:
                await transport.send_request("bad_method", {"arg": "bad"})

            assert exc_info.value.code == ErrorCodes.INVALID_PARAMS
            assert "Bad arguments" in str(exc_info.value)
            assert exc_info.value.data == {"field": "arg"}

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_timeout(self) -> None:
        """send_request() raises TimeoutError if no response."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Don't send a response
            with pytest.raises(asyncio.TimeoutError):
                await transport.send_request("slow_method", timeout=0.05)

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_when_not_running_raises(self) -> None:
        """send_request() before start() raises PluginTransportError."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        transport = WebSocketTransport(url="ws://localhost:8080")

        with pytest.raises(PluginTransportError, match="not running"):
            await transport.send_request("method")


class TestWebSocketTransportNotifications:
    """Notification handling tests."""

    @pytest.mark.asyncio
    async def test_send_notification(self) -> None:
        """send_notification() sends a fire-and-forget notification."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            await transport.send_notification("notify_event", {"data": "value"})

            # Check sent data
            assert len(session.ws.sent) == 1
            sent_msg = json.loads(session.ws.sent[0].decode("utf-8"))
            assert sent_msg["method"] == "notify_event"
            assert sent_msg["params"] == {"data": "value"}
            assert "id" not in sent_msg  # Notifications have no id

        await transport.stop()

    @pytest.mark.asyncio
    async def test_receive_notification_calls_callback(self) -> None:
        """Incoming notifications invoke the callback."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        notifications: list[tuple[str, dict[str, Any]]] = []

        def on_notification(method: str, params: dict[str, Any]) -> None:
            notifications.append((method, params))

        transport = WebSocketTransport(
            url="ws://localhost:8080",
            on_notification=on_notification,
        )

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Push a notification from the server
            await session.ws.push_text(
                _jsonrpc_notification("server_event", {"event": "data_changed"})
            )

            # Give the reader loop time to process
            await asyncio.sleep(0.02)

            assert len(notifications) == 1
            assert notifications[0] == ("server_event", {"event": "data_changed"})

        await transport.stop()

    @pytest.mark.asyncio
    async def test_receive_notification_without_callback_logs(self) -> None:
        """Incoming notifications without callback are logged but don't crash."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Push a notification from the server
            await session.ws.push_text(_jsonrpc_notification("unhandled_event", {}))

            # Give the reader loop time to process
            await asyncio.sleep(0.02)

            # Should not crash

        await transport.stop()


class TestWebSocketTransportServerRequests:
    """Server-initiated request tests."""

    @pytest.mark.asyncio
    async def test_receive_server_request_calls_callback_with_id(self) -> None:
        """Server requests are routed through notification callback with _request_id."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        requests: list[tuple[str, dict[str, Any]]] = []

        def on_notification(method: str, params: dict[str, Any]) -> None:
            requests.append((method, params))

        transport = WebSocketTransport(
            url="ws://localhost:8080",
            on_notification=on_notification,
        )

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Push a server-initiated request
            await session.ws.push_text(_jsonrpc_request("host_api_call", 999, {"arg": "value"}))

            # Give the reader loop time to process
            await asyncio.sleep(0.02)

            assert len(requests) == 1
            method, params = requests[0]
            assert method == "host_api_call"
            assert params["_request_id"] == 999
            assert params["arg"] == "value"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_response_to_server_request(self) -> None:
        """send_response() can respond to server-initiated requests."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            await transport.send_response({"status": "handled"}, request_id=999)

            # Check sent data
            assert len(session.ws.sent) == 1
            sent_msg = json.loads(session.ws.sent[0].decode("utf-8"))
            assert sent_msg["result"] == {"status": "handled"}
            assert sent_msg["id"] == 999

        await transport.stop()

    @pytest.mark.asyncio
    async def test_send_error_response_to_server_request(self) -> None:
        """send_error_response() can respond with errors to server requests."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            await transport.send_error_response(
                code=ErrorCodes.METHOD_NOT_FOUND,
                message="Unknown host API method",
                request_id=999,
                data={"method": "unknown"},
            )

            # Check sent data
            assert len(session.ws.sent) == 1
            sent_msg = json.loads(session.ws.sent[0].decode("utf-8"))
            assert sent_msg["error"]["code"] == ErrorCodes.METHOD_NOT_FOUND
            assert sent_msg["error"]["message"] == "Unknown host API method"
            assert sent_msg["error"]["data"] == {"method": "unknown"}
            assert sent_msg["id"] == 999

        await transport.stop()


class TestWebSocketTransportErrorHandling:
    """Error handling and edge case tests."""

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_crash(self) -> None:
        """Invalid JSON frames are logged but don't crash the transport."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Push invalid JSON
            await session.ws.push_text("{invalid json")

            # Give the reader loop time to process
            await asyncio.sleep(0.02)

            # Transport should still be running
            assert transport.is_running

        await transport.stop()

    @pytest.mark.asyncio
    async def test_connection_close_fails_pending_requests(self) -> None:
        """Connection loss fails all pending requests."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Start a request but close connection before response
            async def close_connection() -> None:
                await asyncio.sleep(0.02)
                await session.ws.push_close()

            asyncio.create_task(close_connection())

            with pytest.raises(PluginTransportError, match="connection lost"):
                await transport.send_request("method", timeout=1.0)

        await transport.stop()

    @pytest.mark.asyncio
    async def test_binary_frames_are_handled(self) -> None:
        """Binary WebSocket frames are decoded and handled."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Send request and respond with binary frame
            async def respond_binary() -> None:
                await asyncio.sleep(0.01)
                response = _jsonrpc_response({"binary": "ok"}, 1)
                await session.ws.push_binary(response.encode("utf-8"))

            asyncio.create_task(respond_binary())

            result = await transport.send_request("test")
            assert result == {"binary": "ok"}

        await transport.stop()

    @pytest.mark.asyncio
    async def test_unrecognized_message_format_is_logged(self) -> None:
        """Messages without proper JSON-RPC structure are logged but don't crash."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Push a message with unrecognized format
            await session.ws.push_text('{"foo": "bar"}')

            # Give the reader loop time to process
            await asyncio.sleep(0.02)

            # Transport should still be running
            assert transport.is_running

        await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_requests(self) -> None:
        """stop() cancels all pending requests with an error."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        session = MockClientSession()
        transport = WebSocketTransport(url="ws://localhost:8080")

        with patch("aiohttp.ClientSession", return_value=session):
            await transport.start()

            # Start a request but stop transport before response
            async def stop_transport() -> None:
                await asyncio.sleep(0.02)
                await transport.stop()

            asyncio.create_task(stop_transport())

            with pytest.raises(PluginTransportError, match="shutting down"):
                await transport.send_request("method", timeout=1.0)


class TestWebSocketTransportReconnection:
    """Reconnection handling tests."""

    @pytest.mark.asyncio
    async def test_reconnect_disabled_by_default(self) -> None:
        """By default, reconnection is disabled."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        transport = WebSocketTransport(url="ws://localhost:8080")
        assert not transport._reconnect

    @pytest.mark.asyncio
    async def test_reconnect_enabled_with_parameter(self) -> None:
        """reconnect=True enables automatic reconnection."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        transport = WebSocketTransport(url="ws://localhost:8080", reconnect=True)
        assert transport._reconnect

    @pytest.mark.asyncio
    async def test_reconnect_parameters_are_stored(self) -> None:
        """Reconnection parameters are stored correctly."""
        from activecontext.plugins.ws_transport import WebSocketTransport

        transport = WebSocketTransport(
            url="ws://localhost:8080",
            reconnect=True,
            reconnect_max_delay=120.0,
            reconnect_base_delay=2.0,
        )
        assert transport._reconnect_max_delay == 120.0
        assert transport._reconnect_base_delay == 2.0
