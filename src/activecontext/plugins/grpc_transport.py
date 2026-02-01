"""CAP Transport -- gRPC implementation (GrpcTransport).

Bidirectional streaming gRPC transport for CAP protocol.
Uses the CAPBidirectional.Channel RPC defined in docs/cap.proto.

This transport connects to a gRPC server and exchanges CAPEnvelope
messages over a bidirectional stream. Each envelope wraps a CAP
message (request, response, or notification) with method/id fields
for correlation.

Features:
- Bidirectional streaming (host and server can send concurrently)
- Automatic reconnection with exponential backoff
- Health checking / keepalive
- Optional TLS support (via gRPC channel credentials)

Requires ``grpcio`` (lazy-imported at start() time):
    pip install grpcio
    uv add grpcio

Optional: ``grpcio-tools`` for .proto compilation (not needed at runtime).

Threading model:
- One asyncio reader task reads from the gRPC stream
- Incoming messages are dispatched: responses to pending futures,
  notifications to the registered callback
- Writes go to the gRPC stream (serialized under a lock)
- Optional reconnection task for connection recovery
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from activecontext.plugins.serialization import CAPSerializer, JsonSerializer
from activecontext.plugins.transport import NotificationCallback, PluginTransportError
from activecontext.plugins.wire import ErrorCodes

logger = logging.getLogger(__name__)

__all__ = ["GrpcTransport"]


class GrpcTransport:
    """JSON-RPC 2.0 transport over gRPC bidirectional streaming.

    Connects to a gRPC server and exchanges CAPEnvelope messages
    over the CAPBidirectional.Channel RPC. Each envelope contains
    the jsonrpc version, method, id, and the actual message payload.

    This implementation uses JSON serialization inside CAPEnvelope
    messages. For native protobuf (no JSON wrapper), use a dedicated
    ProtobufSerializer (future work).

    Usage:
        transport = GrpcTransport(target="localhost:50051")
        await transport.start()
        result = await transport.send_request("initialize", {...})
        await transport.send_notification("shutdown", {})
        await transport.stop()
    """

    def __init__(
        self,
        target: str,
        on_notification: NotificationCallback | None = None,
        serializer: CAPSerializer | None = None,
        *,
        credentials: Any = None,
        options: list[tuple[str, Any]] | None = None,
        reconnect: bool = False,
        reconnect_max_delay: float = 60.0,
        reconnect_base_delay: float = 1.0,
        health_check_interval: float = 30.0,
    ) -> None:
        """Initialize transport configuration.

        Args:
            target: gRPC target string (e.g. "localhost:50051" or "dns:///server:443").
            on_notification: Callback for server -> host notifications.
            serializer: Wire format serializer. Defaults to JsonSerializer.
                Note: grpcio handles framing; serializer only encodes the JSON payload.
            credentials: gRPC ChannelCredentials for TLS (None = insecure channel).
            options: gRPC channel options (e.g., [("grpc.keepalive_time_ms", 10000)]).
            reconnect: Whether to automatically reconnect on connection loss.
            reconnect_max_delay: Maximum delay between reconnection attempts (seconds).
            reconnect_base_delay: Base delay for exponential backoff (seconds).
            health_check_interval: Seconds between health checks (0 = disabled).
        """
        self._target = target
        self._on_notification = on_notification
        self._serializer: CAPSerializer = serializer or JsonSerializer()

        # gRPC configuration
        self._credentials = credentials
        self._options = options or []

        # Reconnection settings
        self._reconnect = reconnect
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_base_delay = reconnect_base_delay
        self._health_check_interval = health_check_interval

        # Runtime state
        self._grpc: Any = None  # Lazily imported grpc module
        self._channel: Any = None  # grpc.aio.Channel
        self._stream: Any = None  # Bidirectional stream
        self._reader_task: asyncio.Task[None] | None = None
        self._health_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._started = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        """Whether the transport is currently connected and operational."""
        return self._started and not self._stopping and self._stream is not None

    async def start(self) -> None:
        """Open a gRPC channel and start the bidirectional stream.

        Raises:
            PluginTransportError: If the connection fails or grpcio is
                not installed.
        """
        if self._started:
            raise PluginTransportError("Transport already started")

        # Lazy import of grpcio
        try:
            import grpc.aio as grpc_aio  # type: ignore[import-untyped]

            self._grpc = grpc_aio
        except ImportError:
            raise PluginTransportError(
                "gRPC transport requires the 'grpcio' package. "
                "Install it with: pip install grpcio (or: uv add grpcio)"
            ) from None

        try:
            # Create gRPC channel
            if self._credentials:
                self._channel = self._grpc.secure_channel(
                    self._target,
                    self._credentials,
                    options=self._options,
                )
            else:
                self._channel = self._grpc.insecure_channel(
                    self._target,
                    options=self._options,
                )

            # Start bidirectional stream
            # Note: We don't have the generated stub, so we use the low-level
            # stream API. For production, generate stubs from cap.proto.
            # For now, we'll create the stream manually using the channel.
            self._stream = self._channel.stream_stream(
                "/cap.v1.CAPBidirectional/Channel",
                request_serializer=self._serialize_envelope,
                response_deserializer=self._deserialize_envelope,
            )

        except Exception as e:
            # Clean up partial state
            if self._channel:
                await self._channel.close()
            self._channel = None
            self._stream = None
            raise PluginTransportError(
                f"Failed to connect to gRPC server at {self._target}: {e}"
            ) from e

        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop(), name="cap-grpc-transport-reader")

        if self._health_check_interval > 0:
            self._health_task = asyncio.create_task(
                self._health_check_loop(), name="cap-grpc-transport-health"
            )

        logger.info("CAP gRPC transport started: %s", self._target)

    async def stop(self) -> None:
        """Stop the transport and close the gRPC channel.

        Cancels pending requests with an error. Idempotent.
        """
        if not self._started or self._stopping:
            return

        self._stopping = True

        # Cancel all pending requests
        for future in self._pending.values():
            if not future.done():
                future.set_exception(PluginTransportError("Transport shutting down"))
        self._pending.clear()

        # Cancel reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task

        # Cancel health check task
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_task

        # Close stream
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.done_writing()
        self._stream = None

        # Close gRPC channel
        if self._channel is not None:
            await self._channel.close()
        self._channel = None

        self._started = False
        self._stopping = False
        logger.info("CAP gRPC transport stopped")

    async def send_request(self, method: str, params: Any = None, timeout: float = 30.0) -> Any:
        """Send a JSON-RPC request and wait for the response.

        Args:
            method: RPC method name.
            params: Parameters (dataclass or dict).
            timeout: Seconds to wait for response.

        Returns:
            The result field from the JSON-RPC response.

        Raises:
            JsonRpcError: If the server returns an error response.
            PluginTransportError: If the transport is not running.
            asyncio.TimeoutError: If no response within timeout.
        """
        if not self.is_running:
            raise PluginTransportError("Transport not running")

        request_id = self._next_id
        self._next_id += 1

        envelope = self._build_request_envelope(method, params, request_id)
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._write_envelope(envelope)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise
        except Exception:
            self._pending.pop(request_id, None)
            raise

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: RPC method name.
            params: Parameters (dataclass or dict).

        Raises:
            PluginTransportError: If the transport is not running.
        """
        if not self.is_running:
            raise PluginTransportError("Transport not running")

        envelope = self._build_notification_envelope(method, params)
        await self._write_envelope(envelope)

    async def send_response(self, result: Any, request_id: int | str) -> None:
        """Send a JSON-RPC response to a server request.

        Used by the connection layer to respond to host API calls.

        Args:
            result: Result value (dataclass or dict).
            request_id: The id from the server's request.
        """
        envelope = self._build_response_envelope(result, request_id)
        await self._write_envelope(envelope)

    async def send_error_response(
        self,
        code: int,
        message: str,
        request_id: int | str | None,
        data: Any = None,
    ) -> None:
        """Send a JSON-RPC error response to a server request.

        Args:
            code: Error code.
            message: Human-readable error message.
            request_id: The id from the server's request.
            data: Optional additional error data.
        """
        envelope = self._build_error_envelope(code, message, request_id, data)
        await self._write_envelope(envelope)

    def _build_request_envelope(
        self, method: str, params: Any, request_id: int | str
    ) -> dict[str, Any]:
        """Build a CAPEnvelope for a request."""
        # For now, we use a dict representation of the envelope
        # In production, use generated protobuf classes
        envelope = {
            "jsonrpc": "2.0",
            "method": method,
            "id": request_id,
        }
        if params is not None:
            envelope["params"] = params
        return envelope

    def _build_notification_envelope(self, method: str, params: Any) -> dict[str, Any]:
        """Build a CAPEnvelope for a notification."""
        envelope = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            envelope["params"] = params
        return envelope

    def _build_response_envelope(self, result: Any, request_id: int | str) -> dict[str, Any]:
        """Build a CAPEnvelope for a response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _build_error_envelope(
        self, code: int, message: str, request_id: int | str | None, data: Any
    ) -> dict[str, Any]:
        """Build a CAPEnvelope for an error response."""
        error = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error,
        }

    def _serialize_envelope(self, envelope: dict[str, Any]) -> bytes:
        """Serialize a CAPEnvelope to bytes for gRPC transmission.

        For now, we serialize the envelope as JSON. In production,
        use generated protobuf serialization.
        """
        import json

        return json.dumps(envelope).encode("utf-8")

    def _deserialize_envelope(self, data: bytes) -> dict[str, Any]:
        """Deserialize a CAPEnvelope from bytes received from gRPC.

        For now, we deserialize from JSON. In production, use
        generated protobuf deserialization.
        """
        import json

        result: dict[str, Any] = json.loads(data.decode("utf-8"))
        return result

    async def _write_envelope(self, envelope: dict[str, Any]) -> None:
        """Write a CAPEnvelope to the gRPC stream.

        Serializes under the write lock to prevent interleaving.
        """
        assert self._stream is not None

        async with self._write_lock:
            await self._stream.write(envelope)

    async def _read_loop(self) -> None:
        """Background task: read and decode CAPEnvelope messages from the stream."""
        assert self._stream is not None

        try:
            async for envelope in self._stream:
                if self._stopping:
                    break

                try:
                    self._dispatch(envelope)
                except Exception as e:
                    logger.warning("CAP gRPC: error dispatching message: %s", e)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("CAP gRPC reader error: %s", e)
        finally:
            # Connection lost: fail all pending requests
            if not self._stopping:
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(PluginTransportError("Server connection lost"))
                self._pending.clear()

                # Attempt reconnection if configured
                if self._reconnect:
                    asyncio.create_task(
                        self._reconnect_loop(),
                        name="cap-grpc-transport-reconnect",
                    )

    def _dispatch(self, envelope: dict[str, Any]) -> None:
        """Route an incoming CAPEnvelope to the correct handler."""
        if "id" in envelope and "method" not in envelope:
            # Response (has id, no method)
            self._handle_response(envelope)
        elif "method" in envelope and "id" not in envelope:
            # Notification (has method, no id)
            self._handle_notification(envelope)
        elif "method" in envelope and "id" in envelope:
            # Request from server (has both method and id)
            self._handle_server_request(envelope)
        else:
            logger.warning("CAP gRPC: unrecognized message: %s", envelope)

    def _handle_response(self, envelope: dict[str, Any]) -> None:
        """Handle a JSON-RPC response."""
        from activecontext.plugins.transport import JsonRpcError

        msg_id = envelope["id"]
        future = self._pending.pop(msg_id, None)
        if future is None:
            logger.warning("CAP gRPC: response for unknown id: %s", msg_id)
            return

        if "error" in envelope:
            err = envelope["error"]
            future.set_exception(
                JsonRpcError(
                    code=err.get("code", ErrorCodes.INTERNAL_ERROR),
                    message=err.get("message", "Unknown error"),
                    data=err.get("data"),
                )
            )
        else:
            future.set_result(envelope.get("result"))

    def _handle_notification(self, envelope: dict[str, Any]) -> None:
        """Handle a server -> host notification."""
        method = envelope["method"]
        params = envelope.get("params", {})
        if self._on_notification:
            try:
                self._on_notification(method, params)
            except Exception as e:
                logger.error("CAP gRPC notification handler error: %s", e)
        else:
            logger.debug("CAP gRPC: unhandled notification: %s", method)

    def _handle_server_request(self, envelope: dict[str, Any]) -> None:
        """Handle a server -> host request (host API call).

        These are bidirectional requests where the server calls the host.
        Routed through the notification callback with the full message
        so the connection layer can send a response.
        """
        method = envelope["method"]
        params = envelope.get("params", {})
        request_id = envelope["id"]

        if self._on_notification:
            try:
                self._on_notification(
                    method,
                    {"_request_id": request_id, **params},
                )
            except Exception as e:
                logger.error("CAP gRPC host API handler error: %s", e)
        else:
            logger.warning("CAP gRPC: unhandled host API request: %s", method)

    async def _health_check_loop(self) -> None:
        """Periodically check connection health.

        Uses gRPC channel state checking. If the channel enters
        TRANSIENT_FAILURE or SHUTDOWN state, triggers reconnection.
        """
        assert self._channel is not None
        grpc_aio = self._grpc

        try:
            while not self._stopping and self._started:
                await asyncio.sleep(self._health_check_interval)

                if self._stopping:
                    break

                # Check channel state
                state = self._channel.get_state(try_to_connect=False)
                if state in (
                    grpc_aio.ChannelConnectivity.TRANSIENT_FAILURE,
                    grpc_aio.ChannelConnectivity.SHUTDOWN,
                ):
                    logger.warning(
                        "CAP gRPC health check: channel in state %s, reconnecting",
                        state,
                    )
                    # Trigger reconnection by stopping reader
                    if self._reader_task and not self._reader_task.done():
                        self._reader_task.cancel()
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("CAP gRPC health check error: %s", e)

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff.

        Only runs if ``reconnect=True`` was set at construction time.
        Gives up if stop() is called during reconnection.
        """
        delay = self._reconnect_base_delay
        attempt = 0

        while not self._stopping and self._started:
            attempt += 1
            logger.info(
                "CAP gRPC reconnect attempt %d (delay=%.1fs)",
                attempt,
                delay,
            )
            await asyncio.sleep(delay)

            if self._stopping:
                break

            try:
                # Close old channel/stream
                if self._stream is not None:
                    with contextlib.suppress(Exception):
                        await self._stream.done_writing()
                    self._stream = None

                if self._channel is not None:
                    await self._channel.close()

                # Re-create channel and stream
                if self._credentials:
                    self._channel = self._grpc.secure_channel(
                        self._target,
                        self._credentials,
                        options=self._options,
                    )
                else:
                    self._channel = self._grpc.insecure_channel(
                        self._target,
                        options=self._options,
                    )

                self._stream = self._channel.stream_stream(
                    "/cap.v1.CAPBidirectional/Channel",
                    request_serializer=self._serialize_envelope,
                    response_deserializer=self._deserialize_envelope,
                )

                # Restart reader
                self._reader_task = asyncio.create_task(
                    self._read_loop(), name="cap-grpc-transport-reader"
                )

                logger.info(
                    "CAP gRPC reconnected to %s (attempt %d)",
                    self._target,
                    attempt,
                )
                return  # Success

            except Exception as e:
                logger.warning(
                    "CAP gRPC reconnect attempt %d failed: %s",
                    attempt,
                    e,
                )
                # Close partial state
                if self._stream is not None:
                    with contextlib.suppress(Exception):
                        await self._stream.done_writing()
                    self._stream = None

                if self._channel is not None:
                    with contextlib.suppress(Exception):
                        await self._channel.close()
                    self._channel = None

                # Exponential backoff with cap
                delay = min(delay * 2, self._reconnect_max_delay)

        logger.info("CAP gRPC reconnection stopped")
