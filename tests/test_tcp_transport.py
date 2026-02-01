"""Tests for TcpTransport.

Verifies:
- Length-prefixed framing (4-byte big-endian + JSON payload)
- Protocol compliance with CAPTransport
- Request/response, notifications, bidirectional requests
- Error handling (connection failures, malformed frames, etc.)
- Transport factory integration
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

import pytest

from activecontext.plugins.cap_transport import CAPTransport
from activecontext.plugins.tcp_transport import TcpTransport
from activecontext.plugins.transport import JsonRpcError, PluginTransportError
from activecontext.plugins.transport_factory import TransportConfig, create_transport

# ---------------------------------------------------------------------------
# Test Server
# ---------------------------------------------------------------------------


class MockTcpServer:
    """Mock TCP server for testing TcpTransport."""

    def __init__(self) -> None:
        self.server: asyncio.Server | None = None
        self.clients: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        self.requests: list[dict[str, Any]] = []
        self.port: int = 0

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Start the mock server on a random available port."""
        self.server = await asyncio.start_server(self._handle_client, host, port)
        # Get the actual port assigned
        assert self.server.sockets is not None
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop the server and close all client connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        for _, writer in self.clients:
            if not writer.is_closing():
                writer.close()
                await writer.wait_closed()

        self.clients.clear()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a client connection."""
        self.clients.append((reader, writer))

        try:
            while True:
                # Read length prefix
                length_bytes = await reader.readexactly(4)
                payload_length = struct.unpack("!I", length_bytes)[0]

                if payload_length == 0:
                    continue

                # Read payload
                payload = await reader.readexactly(payload_length)
                msg = json.loads(payload.decode("utf-8"))
                self.requests.append(msg)

                # Auto-respond to requests
                if "id" in msg:
                    await self.send_response(writer, {"echo": msg.get("method")}, msg["id"])

        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            if not writer.is_closing():
                writer.close()

    async def send_response(
        self, writer: asyncio.StreamWriter, result: Any, request_id: int | str
    ) -> None:
        """Send a JSON-RPC response to the client."""
        msg = {"jsonrpc": "2.0", "result": result, "id": request_id}
        payload = json.dumps(msg).encode("utf-8")
        frame = struct.pack("!I", len(payload)) + payload
        writer.write(frame)
        await writer.drain()

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Send a notification to all connected clients."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        payload = json.dumps(msg).encode("utf-8")
        frame = struct.pack("!I", len(payload)) + payload

        for _, writer in self.clients:
            if not writer.is_closing():
                writer.write(frame)
                await writer.drain()

    async def send_error(self, code: int, message: str, request_id: int | str) -> None:
        """Send an error response to the client."""
        msg = {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id,
        }
        payload = json.dumps(msg).encode("utf-8")
        frame = struct.pack("!I", len(payload)) + payload

        for _, writer in self.clients:
            if not writer.is_closing():
                writer.write(frame)
                await writer.drain()


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestTcpTransportProtocol:
    """Verify TcpTransport satisfies CAPTransport protocol."""

    def test_tcp_transport_satisfies_protocol(self) -> None:
        """TcpTransport is a CAPTransport."""
        transport = TcpTransport(host="127.0.0.1", port=9000)
        assert isinstance(transport, CAPTransport)

    def test_has_required_methods(self) -> None:
        """TcpTransport has all required CAPTransport methods."""
        transport = TcpTransport(host="127.0.0.1", port=9000)
        assert hasattr(transport, "is_running")
        assert hasattr(transport, "start")
        assert hasattr(transport, "stop")
        assert hasattr(transport, "send_request")
        assert hasattr(transport, "send_notification")
        assert hasattr(transport, "send_response")
        assert hasattr(transport, "send_error_response")


# ---------------------------------------------------------------------------
# Connection lifecycle tests
# ---------------------------------------------------------------------------


class TestTcpTransportLifecycle:
    """Test connection lifecycle: start, stop, error handling."""

    @pytest.mark.asyncio
    async def test_start_connects_to_server(self) -> None:
        """start() establishes a TCP connection."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        assert not transport.is_running

        await transport.start()
        assert transport.is_running

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_start_fails_when_server_unreachable(self) -> None:
        """start() raises PluginTransportError when server is not available."""
        transport = TcpTransport(host="127.0.0.1", port=9999)

        with pytest.raises(PluginTransportError, match="Failed to connect"):
            await transport.start()

    @pytest.mark.asyncio
    async def test_start_twice_raises_error(self) -> None:
        """Calling start() twice raises PluginTransportError."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        with pytest.raises(PluginTransportError, match="already started"):
            await transport.start()

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_connection(self) -> None:
        """stop() closes the TCP connection and cleans up resources."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()
        assert transport.is_running

        await transport.stop()
        assert not transport.is_running

        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        """Calling stop() multiple times is safe."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        await transport.stop()
        await transport.stop()  # Should not raise

        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_requests(self) -> None:
        """stop() cancels all pending requests with an error."""
        # Create a server that doesn't respond
        server_started = asyncio.Event()

        async def silent_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            server_started.set()
            try:
                while True:
                    length_bytes = await reader.readexactly(4)
                    payload_length = struct.unpack("!I", length_bytes)[0]
                    await reader.readexactly(payload_length)
                    # Don't respond - just read and discard
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                if not writer.is_closing():
                    writer.close()

        server = await asyncio.start_server(silent_handler, "127.0.0.1", 0)
        assert server.sockets is not None
        port = server.sockets[0].getsockname()[1]

        transport = TcpTransport(host="127.0.0.1", port=port)
        await transport.start()

        # Wait for server to accept connection
        await asyncio.wait_for(server_started.wait(), timeout=1.0)

        # Send a request but server won't respond
        async def send_request() -> None:
            with pytest.raises(PluginTransportError, match="shutting down"):
                await transport.send_request("test_method", timeout=10.0)

        request_task = asyncio.create_task(send_request())

        # Give the request time to be sent
        await asyncio.sleep(0.1)

        # Stop transport before response arrives
        await transport.stop()

        # Wait for the request task to complete (it should have raised)
        await asyncio.wait_for(request_task, timeout=1.0)

        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Framing tests
# ---------------------------------------------------------------------------


class TestTcpTransportFraming:
    """Test length-prefixed framing protocol."""

    @pytest.mark.asyncio
    async def test_sends_length_prefixed_frames(self) -> None:
        """Outgoing messages are prefixed with 4-byte big-endian length."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        await transport.send_notification("test_notification", {"key": "value"})

        # Give server time to receive
        await asyncio.sleep(0.1)

        assert len(server.requests) == 1
        assert server.requests[0]["method"] == "test_notification"
        assert server.requests[0]["params"] == {"key": "value"}

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_reads_length_prefixed_frames(self) -> None:
        """Incoming messages are read with length-prefix parsing."""
        server = MockTcpServer()
        await server.start()

        notifications: list[tuple[str, dict[str, Any]]] = []

        def on_notification(method: str, params: dict[str, Any]) -> None:
            notifications.append((method, params))

        transport = TcpTransport(
            host="127.0.0.1", port=server.port, on_notification=on_notification
        )
        await transport.start()

        # Server sends a notification
        await server.send_notification("server_event", {"status": "ready"})

        # Give transport time to receive
        await asyncio.sleep(0.1)

        assert len(notifications) == 1
        assert notifications[0] == ("server_event", {"status": "ready"})

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_rejects_oversized_frames(self, caplog: Any) -> None:
        """Frames exceeding _MAX_FRAME_SIZE are rejected and logged."""
        import logging

        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        # Send a frame claiming to be 20MB (over the 16MB limit)
        _, writer = server.clients[0]
        malicious_length = 20 * 1024 * 1024
        writer.write(struct.pack("!I", malicious_length))
        await writer.drain()

        # Give transport time to process and log the error
        await asyncio.sleep(0.2)

        # Verify error was logged
        assert any(
            "frame too large" in record.message
            for record in caplog.records
            if record.levelno == logging.ERROR
        )

        await transport.stop()
        await server.stop()


# ---------------------------------------------------------------------------
# Request/response tests
# ---------------------------------------------------------------------------


class TestTcpTransportRequests:
    """Test request/response patterns."""

    @pytest.mark.asyncio
    async def test_send_request_returns_result(self) -> None:
        """send_request() waits for and returns the server's result."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        result = await transport.send_request("echo_test")
        assert result == {"echo": "echo_test"}

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_request_with_params(self) -> None:
        """send_request() sends parameters to the server."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        await transport.send_request("test_method", {"arg": "value"})

        await asyncio.sleep(0.1)

        assert len(server.requests) == 1
        assert server.requests[0]["params"] == {"arg": "value"}

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_request_timeout(self) -> None:
        """send_request() raises TimeoutError if no response arrives."""

        # Create a server that doesn't respond
        async def silent_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                while True:
                    length_bytes = await reader.readexactly(4)
                    payload_length = struct.unpack("!I", length_bytes)[0]
                    await reader.readexactly(payload_length)
                    # Don't respond - just consume
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                if not writer.is_closing():
                    writer.close()

        server = await asyncio.start_server(silent_handler, "127.0.0.1", 0)
        assert server.sockets is not None
        port = server.sockets[0].getsockname()[1]

        transport = TcpTransport(host="127.0.0.1", port=port)
        await transport.start()

        with pytest.raises(asyncio.TimeoutError):
            await transport.send_request("test_method", timeout=0.1)

        await transport.stop()
        server.close()
        await server.wait_closed()

    @pytest.mark.asyncio
    async def test_send_request_when_not_running_raises_error(self) -> None:
        """send_request() raises PluginTransportError if transport not running."""
        transport = TcpTransport(host="127.0.0.1", port=9000)

        with pytest.raises(PluginTransportError, match="not running"):
            await transport.send_request("test_method")

    @pytest.mark.asyncio
    async def test_handles_error_response(self) -> None:
        """Receiving a JSON-RPC error raises JsonRpcError."""

        async def error_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                length_bytes = await reader.readexactly(4)
                payload_length = struct.unpack("!I", length_bytes)[0]
                payload = await reader.readexactly(payload_length)
                msg = json.loads(payload.decode("utf-8"))

                # Send error response
                error_msg = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": "Method not found"},
                    "id": msg["id"],
                }
                error_payload = json.dumps(error_msg).encode("utf-8")
                frame = struct.pack("!I", len(error_payload)) + error_payload
                writer.write(frame)
                await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                if not writer.is_closing():
                    writer.close()

        server = await asyncio.start_server(error_handler, "127.0.0.1", 0)
        assert server.sockets is not None
        port = server.sockets[0].getsockname()[1]

        transport = TcpTransport(host="127.0.0.1", port=port)
        await transport.start()

        with pytest.raises(JsonRpcError, match="Method not found"):
            await transport.send_request("unknown_method")

        await transport.stop()
        server.close()
        await server.wait_closed()


# ---------------------------------------------------------------------------
# Notification tests
# ---------------------------------------------------------------------------


class TestTcpTransportNotifications:
    """Test fire-and-forget notifications."""

    @pytest.mark.asyncio
    async def test_send_notification_no_response(self) -> None:
        """send_notification() sends a message without waiting for response."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        await transport.send_notification("event", {"data": "test"})

        await asyncio.sleep(0.1)

        assert len(server.requests) == 1
        assert server.requests[0]["method"] == "event"
        assert "id" not in server.requests[0]  # No id for notifications

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_notification_when_not_running_raises_error(self) -> None:
        """send_notification() raises PluginTransportError if not running."""
        transport = TcpTransport(host="127.0.0.1", port=9000)

        with pytest.raises(PluginTransportError, match="not running"):
            await transport.send_notification("event")

    @pytest.mark.asyncio
    async def test_receives_server_notifications(self) -> None:
        """Transport invokes on_notification callback for server notifications."""
        server = MockTcpServer()
        await server.start()

        notifications: list[tuple[str, dict[str, Any]]] = []

        def on_notification(method: str, params: dict[str, Any]) -> None:
            notifications.append((method, params))

        transport = TcpTransport(
            host="127.0.0.1", port=server.port, on_notification=on_notification
        )
        await transport.start()

        await server.send_notification("test_event", {"key": "value"})

        await asyncio.sleep(0.1)

        assert len(notifications) == 1
        assert notifications[0] == ("test_event", {"key": "value"})

        await transport.stop()
        await server.stop()


# ---------------------------------------------------------------------------
# Bidirectional request tests
# ---------------------------------------------------------------------------


class TestTcpTransportBidirectional:
    """Test server-initiated requests (host API calls)."""

    @pytest.mark.asyncio
    async def test_handles_server_request(self) -> None:
        """Server-initiated requests are routed through on_notification."""
        server = MockTcpServer()
        await server.start()

        requests: list[tuple[str, dict[str, Any]]] = []

        def on_notification(method: str, params: dict[str, Any]) -> None:
            requests.append((method, params))

        transport = TcpTransport(
            host="127.0.0.1", port=server.port, on_notification=on_notification
        )
        await transport.start()

        # Server sends a request (has both method and id)
        _, writer = server.clients[0]
        msg = {"jsonrpc": "2.0", "method": "host_api_call", "id": 123, "params": {}}
        payload = json.dumps(msg).encode("utf-8")
        frame = struct.pack("!I", len(payload)) + payload
        writer.write(frame)
        await writer.drain()

        await asyncio.sleep(0.1)

        assert len(requests) == 1
        assert requests[0][0] == "host_api_call"
        # The _request_id is injected for response handling
        assert requests[0][1]["_request_id"] == 123

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_response_to_server_request(self) -> None:
        """send_response() sends a response to a server request."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        # Send a response
        await transport.send_response({"status": "ok"}, request_id=42)

        # Server should receive it (we can't easily verify without more
        # complex server logic, but ensure no errors)

        await transport.stop()
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_error_response_to_server_request(self) -> None:
        """send_error_response() sends an error to a server request."""
        server = MockTcpServer()
        await server.start()

        transport = TcpTransport(host="127.0.0.1", port=server.port)
        await transport.start()

        # Send an error response
        await transport.send_error_response(
            code=-32600,
            message="Invalid request",
            request_id=42,
            data={"detail": "Missing field"},
        )

        await transport.stop()
        await server.stop()


# ---------------------------------------------------------------------------
# Transport factory integration tests
# ---------------------------------------------------------------------------


class TestTcpTransportFactory:
    """Test integration with transport_factory."""

    def test_create_transport_tcp(self) -> None:
        """create_transport() instantiates TcpTransport for type='tcp'."""
        config = TransportConfig(type="tcp", address="localhost:9000")
        transport = create_transport(config)

        assert isinstance(transport, TcpTransport)

    def test_create_transport_tcp_missing_address_raises_error(self) -> None:
        """create_transport() raises ValueError if TCP address is missing."""
        config = TransportConfig(type="tcp")

        with pytest.raises(ValueError, match="requires 'address'"):
            create_transport(config)

    def test_create_transport_tcp_invalid_address_format_raises_error(
        self,
    ) -> None:
        """create_transport() raises ValueError if address lacks port."""
        config = TransportConfig(type="tcp", address="localhost")

        with pytest.raises(ValueError, match="must be 'host:port'"):
            create_transport(config)

    def test_create_transport_tcp_invalid_port_raises_error(self) -> None:
        """create_transport() raises ValueError if port is not numeric."""
        config = TransportConfig(type="tcp", address="localhost:abc")

        with pytest.raises(ValueError, match="Invalid TCP port"):
            create_transport(config)

    def test_tcp_in_list_available_transports(self) -> None:
        """list_available_transports() includes 'tcp'."""
        from activecontext.plugins.transport_factory import (
            list_available_transports,
        )

        available = list_available_transports()
        assert "tcp" in available
