"""Tests for GrpcTransport.

Tests the gRPC bidirectional streaming transport for CAP protocol.
Uses mocks to avoid requiring a running gRPC server.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from activecontext.plugins.transport import JsonRpcError, PluginTransportError


class TestGrpcTransportImportError:
    """Test GrpcTransport behavior when grpcio is not installed."""

    @pytest.mark.asyncio
    async def test_start_fails_without_grpcio(self) -> None:
        """Test that start() raises PluginTransportError if grpcio is missing."""
        # Import should succeed (lazy import)
        from activecontext.plugins.grpc_transport import GrpcTransport

        transport = GrpcTransport(target="localhost:50051")

        # Mock the import to simulate missing grpcio
        with patch.dict("sys.modules", {"grpc.aio": None}):
            # Clear the cached _grpc attribute
            transport._grpc = None

            # Patch __import__ to raise ImportError for grpc.aio
            def mock_import(name, *args, **kwargs):
                if name == "grpc.aio":
                    raise ImportError("No module named 'grpc'")
                return __import__(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=mock_import),
                pytest.raises(
                    PluginTransportError,
                    match="gRPC transport requires the 'grpcio' package",
                ),
            ):
                await transport.start()


@pytest.fixture
def mock_grpc():
    """Fixture that mocks the grpc.aio module."""
    mock_module = MagicMock()

    # Mock ChannelConnectivity states
    mock_module.ChannelConnectivity = MagicMock()
    mock_module.ChannelConnectivity.TRANSIENT_FAILURE = "TRANSIENT_FAILURE"
    mock_module.ChannelConnectivity.SHUTDOWN = "SHUTDOWN"
    mock_module.ChannelConnectivity.READY = "READY"

    # Mock channel methods
    mock_channel = AsyncMock()
    mock_channel.get_state = Mock(return_value="READY")
    mock_channel.close = AsyncMock()

    # Mock stream with proper async iterator
    mock_stream = MagicMock()
    mock_stream.write = AsyncMock()
    mock_stream.done_writing = AsyncMock()

    # Create an async iterator class for the stream
    class EmptyAsyncIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            # Block indefinitely, simulating a stream waiting for messages
            await asyncio.Event().wait()
            raise StopAsyncIteration

    # Mock __aiter__ to return the iterator (lambda gets self from mock)
    mock_stream.__aiter__ = Mock(return_value=EmptyAsyncIterator())

    # Mock channel factory
    mock_module.insecure_channel = Mock(return_value=mock_channel)
    mock_module.secure_channel = Mock(return_value=mock_channel)

    # Mock stream_stream on channel
    mock_channel.stream_stream = Mock(return_value=mock_stream)

    return {
        "module": mock_module,
        "channel": mock_channel,
        "stream": mock_stream,
    }


@pytest.fixture
async def grpc_transport(mock_grpc):
    """Fixture that provides a GrpcTransport with mocked grpcio."""
    from activecontext.plugins.grpc_transport import GrpcTransport

    transport = GrpcTransport(target="localhost:50051")

    # Inject the mock module
    with patch("grpc.aio", mock_grpc["module"]):
        yield transport, mock_grpc

        # Cleanup
        if transport.is_running:
            await transport.stop()


class TestGrpcTransportLifecycle:
    """Test GrpcTransport connection lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_channel_and_stream(self, grpc_transport) -> None:
        """Test that start() creates a gRPC channel and stream."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

        assert transport.is_running
        assert transport._channel is not None
        assert transport._stream is not None
        mocks["module"].insecure_channel.assert_called_once_with("localhost:50051", options=[])
        mocks["channel"].stream_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_credentials_creates_secure_channel(self, mock_grpc) -> None:
        """Test that start() with credentials creates a secure channel."""
        from activecontext.plugins.grpc_transport import GrpcTransport

        mock_creds = MagicMock()
        transport = GrpcTransport(target="server:443", credentials=mock_creds)

        with patch("grpc.aio", mock_grpc["module"]):
            await transport.start()

        assert transport.is_running
        mock_grpc["module"].secure_channel.assert_called_once_with(
            "server:443", mock_creds, options=[]
        )

    @pytest.mark.asyncio
    async def test_start_twice_raises_error(self, grpc_transport) -> None:
        """Test that calling start() twice raises PluginTransportError."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            with pytest.raises(PluginTransportError, match="already started"):
                await transport.start()

    @pytest.mark.asyncio
    async def test_stop_closes_channel_and_stream(self, grpc_transport) -> None:
        """Test that stop() closes the channel and stream."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()
            await transport.stop()

        assert not transport.is_running
        mocks["stream"].done_writing.assert_awaited_once()
        mocks["channel"].close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_requests(self, grpc_transport) -> None:
        """Test that stop() fails all pending requests."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            # Create a pending request
            future = asyncio.get_event_loop().create_future()
            transport._pending[1] = future

            await transport.stop()

            assert future.done()
            with pytest.raises(PluginTransportError, match="shutting down"):
                future.result()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, grpc_transport) -> None:
        """Test that calling stop() multiple times is safe."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()
            await transport.stop()
            await transport.stop()  # Should not raise

        assert not transport.is_running


class TestGrpcTransportMessaging:
    """Test GrpcTransport message sending and receiving."""

    @pytest.mark.asyncio
    async def test_send_request_writes_envelope(self, grpc_transport) -> None:
        """Test that send_request() writes a CAPEnvelope to the stream."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            # Create a task that will complete the request
            async def complete_request():
                await asyncio.sleep(0.01)
                # Simulate a response
                future = transport._pending.get(1)
                if future:
                    future.set_result({"status": "ok"})

            asyncio.create_task(complete_request())

            result = await transport.send_request("test/method", {"arg": "value"})

        assert result == {"status": "ok"}
        # Check that write was called
        assert mocks["stream"].write.await_count >= 1

    @pytest.mark.asyncio
    async def test_send_request_timeout(self, grpc_transport) -> None:
        """Test that send_request() times out if no response."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            with pytest.raises(asyncio.TimeoutError):
                await transport.send_request("test/method", timeout=0.1)

            # Pending request should be cleaned up
            assert 1 not in transport._pending

    @pytest.mark.asyncio
    async def test_send_request_fails_when_not_running(self, grpc_transport) -> None:
        """Test that send_request() raises if transport not running."""
        transport, mocks = grpc_transport

        with pytest.raises(PluginTransportError, match="not running"):
            await transport.send_request("test/method")

    @pytest.mark.asyncio
    async def test_send_notification_writes_envelope(self, grpc_transport) -> None:
        """Test that send_notification() writes to the stream."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()
            await transport.send_notification("test/notify", {"data": "value"})

        # Notification should be written
        assert mocks["stream"].write.await_count >= 1

    @pytest.mark.asyncio
    async def test_send_response_writes_envelope(self, grpc_transport) -> None:
        """Test that send_response() writes a response envelope."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()
            await transport.send_response({"result": "ok"}, request_id=123)

        assert mocks["stream"].write.await_count >= 1

    @pytest.mark.asyncio
    async def test_send_error_response_writes_envelope(self, grpc_transport) -> None:
        """Test that send_error_response() writes an error envelope."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()
            await transport.send_error_response(
                code=-32600,
                message="Invalid request",
                request_id=456,
                data={"detail": "missing field"},
            )

        assert mocks["stream"].write.await_count >= 1


class TestGrpcTransportDispatch:
    """Test GrpcTransport message dispatching."""

    @pytest.mark.asyncio
    async def test_dispatch_response_resolves_future(self, grpc_transport) -> None:
        """Test that a response envelope resolves the matching pending future."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            # Create a pending request
            future = asyncio.get_event_loop().create_future()
            transport._pending[1] = future

            # Dispatch a response
            transport._dispatch({"jsonrpc": "2.0", "id": 1, "result": {"data": "ok"}})

            assert future.done()
            assert future.result() == {"data": "ok"}

    @pytest.mark.asyncio
    async def test_dispatch_error_response_raises_exception(self, grpc_transport) -> None:
        """Test that an error response sets exception on the future."""
        transport, mocks = grpc_transport

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            # Create a pending request
            future = asyncio.get_event_loop().create_future()
            transport._pending[2] = future

            # Dispatch an error response
            transport._dispatch(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": -32600, "message": "Invalid request"},
                }
            )

            assert future.done()
            with pytest.raises(JsonRpcError, match="Invalid request"):
                future.result()

    @pytest.mark.asyncio
    async def test_dispatch_notification_invokes_callback(self, grpc_transport) -> None:
        """Test that a notification envelope invokes the callback."""
        transport, mocks = grpc_transport
        callback_invoked = False
        received_method = None
        received_params = None

        def on_notification(method: str, params: dict[str, Any]) -> None:
            nonlocal callback_invoked, received_method, received_params
            callback_invoked = True
            received_method = method
            received_params = params

        transport._on_notification = on_notification

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            # Dispatch a notification
            transport._dispatch(
                {
                    "jsonrpc": "2.0",
                    "method": "node/dirty",
                    "params": {"node_id": "test_node"},
                }
            )

        assert callback_invoked
        assert received_method == "node/dirty"
        assert received_params == {"node_id": "test_node"}

    @pytest.mark.asyncio
    async def test_dispatch_server_request_invokes_callback_with_request_id(
        self, grpc_transport
    ) -> None:
        """Test that a server request adds _request_id to params."""
        transport, mocks = grpc_transport
        received_params = None

        def on_notification(method: str, params: dict[str, Any]) -> None:
            nonlocal received_params
            received_params = params

        transport._on_notification = on_notification

        with patch("grpc.aio", mocks["module"]):
            await transport.start()

            # Dispatch a server request (has both method and id)
            transport._dispatch(
                {
                    "jsonrpc": "2.0",
                    "method": "host/create_node",
                    "id": 999,
                    "params": {"node_type": "text"},
                }
            )

        assert received_params is not None
        assert received_params["_request_id"] == 999
        assert received_params["node_type"] == "text"


class TestGrpcTransportReconnection:
    """Test GrpcTransport reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_after_connection_lost(self, mock_grpc) -> None:
        """Test that transport reconnects after connection loss."""
        from activecontext.plugins.grpc_transport import GrpcTransport

        transport = GrpcTransport(
            target="localhost:50051",
            reconnect=True,
            reconnect_base_delay=0.05,
        )

        # Create a stream that fails after 1 message
        messages = [{"jsonrpc": "2.0", "id": 1, "result": "ok"}]
        mock_stream = AsyncMock()
        mock_stream.write = AsyncMock()
        mock_stream.done_writing = AsyncMock()

        async def stream_iterator():
            for msg in messages:
                yield msg
            # Simulate connection lost
            raise Exception("Connection lost")

        mock_stream.__aiter__ = Mock(return_value=stream_iterator())

        mock_grpc["channel"].stream_stream = Mock(return_value=mock_stream)

        with patch("grpc.aio", mock_grpc["module"]):
            await transport.start()

            # Wait for reconnection attempt
            await asyncio.sleep(0.2)

            # Should have attempted reconnection
            # (check that stream_stream was called multiple times)
            assert mock_grpc["channel"].stream_stream.call_count >= 2

            await transport.stop()

    @pytest.mark.asyncio
    async def test_no_reconnect_when_disabled(self, mock_grpc) -> None:
        """Test that reconnection doesn't happen when disabled."""
        from activecontext.plugins.grpc_transport import GrpcTransport

        transport = GrpcTransport(
            target="localhost:50051",
            reconnect=False,  # Disabled
        )

        # Create a stream that fails immediately
        mock_stream = AsyncMock()
        mock_stream.write = AsyncMock()
        mock_stream.done_writing = AsyncMock()

        async def stream_iterator():
            raise Exception("Connection lost")
            yield  # Never reached

        mock_stream.__aiter__ = Mock(return_value=stream_iterator())

        mock_grpc["channel"].stream_stream = Mock(return_value=mock_stream)

        with patch("grpc.aio", mock_grpc["module"]):
            await transport.start()

            # Wait a bit
            await asyncio.sleep(0.1)

            # Should NOT have reconnected
            assert mock_grpc["channel"].stream_stream.call_count == 1

            await transport.stop()


class TestGrpcTransportHealthCheck:
    """Test GrpcTransport health checking."""

    @pytest.mark.asyncio
    async def test_health_check_detects_transient_failure(self, mock_grpc) -> None:
        """Test that health check detects TRANSIENT_FAILURE state."""
        from activecontext.plugins.grpc_transport import GrpcTransport

        transport = GrpcTransport(
            target="localhost:50051",
            health_check_interval=0.05,
        )

        # Initially healthy, then fails
        state_sequence = ["READY", "TRANSIENT_FAILURE"]
        state_index = [0]

        def get_state(try_to_connect=False):
            idx = state_index[0]
            if idx < len(state_sequence):
                state = state_sequence[idx]
                state_index[0] += 1
                return state
            return "TRANSIENT_FAILURE"

        mock_grpc["channel"].get_state = Mock(side_effect=get_state)

        with patch("grpc.aio", mock_grpc["module"]):
            await transport.start()

            # Wait for health check to detect failure
            await asyncio.sleep(0.15)

            # Reader task should have been cancelled
            if transport._reader_task:
                assert transport._reader_task.cancelled() or transport._reader_task.done()

            await transport.stop()
