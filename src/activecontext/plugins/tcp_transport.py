"""CAP Transport -- TCP implementation with length-prefixed framing.

JSON-RPC 2.0 over raw TCP with 4-byte big-endian length prefix.
No external dependencies beyond stdlib (asyncio).

Framing protocol:
    [4 bytes: big-endian uint32 payload length][N bytes: JSON payload]

Each frame is a single JSON-RPC 2.0 message. The length prefix
specifies the exact byte count of the JSON payload that follows.

Threading model mirrors StdioTransport:
- One asyncio reader task reads length-prefixed frames from the socket
- Incoming messages are dispatched: responses to pending futures,
  notifications to the registered callback
- Writes prepend the length prefix under a lock to prevent interleaving
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
from typing import Any

from activecontext.plugins.serialization import CAPSerializer, JsonSerializer
from activecontext.plugins.transport import (
    JsonRpcError,
    NotificationCallback,
    PluginTransportError,
)
from activecontext.plugins.wire import ErrorCodes

logger = logging.getLogger(__name__)

# 4-byte big-endian unsigned int for length prefix
_LENGTH_FMT = "!I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FMT)

# Maximum frame size: 16 MiB (sanity guard against corrupt length prefixes)
_MAX_FRAME_SIZE = 16 * 1024 * 1024

__all__ = [
    "TcpTransport",
]


class TcpTransport:
    """JSON-RPC 2.0 transport over TCP with length-prefixed framing.

    Connects to a TCP server and exchanges length-prefixed JSON-RPC
    messages. Each message is framed as:

        [4 bytes: big-endian uint32 length][N bytes: JSON payload]

    This is the simplest network transport -- no HTTP overhead, no
    WebSocket framing, just raw TCP with minimal framing.

    Usage:
        transport = TcpTransport(host="127.0.0.1", port=9000)
        await transport.start()
        result = await transport.send_request("initialize", {...})
        await transport.send_notification("shutdown", {})
        await transport.stop()
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_notification: NotificationCallback | None = None,
        serializer: CAPSerializer | None = None,
    ) -> None:
        """Initialize transport configuration.

        Args:
            host: TCP host to connect to.
            port: TCP port to connect to.
            on_notification: Callback for server -> host notifications.
            serializer: Wire format serializer. Defaults to JsonSerializer.
        """
        self._host = host
        self._port = port
        self._on_notification = on_notification
        self._serializer: CAPSerializer = serializer or JsonSerializer()

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._started = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        """Whether the transport is currently connected and operational."""
        return (
            self._started
            and not self._stopping
            and self._writer is not None
            and not self._writer.is_closing()
        )

    async def start(self) -> None:
        """Connect to the TCP server and start the reader task.

        Raises:
            PluginTransportError: If the connection fails.
        """
        if self._started:
            raise PluginTransportError("Transport already started")

        try:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
        except (OSError, ConnectionRefusedError) as e:
            raise PluginTransportError(
                f"Failed to connect to {self._host}:{self._port}: {e}"
            ) from e

        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop(), name="cap-tcp-reader")
        logger.info("CAP TCP transport connected: %s:%s", self._host, self._port)

    async def stop(self) -> None:
        """Stop the transport and close the TCP connection.

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

        # Close the TCP connection
        if self._writer and not self._writer.is_closing():
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, ConnectionError):
                pass  # Already closed or broken

        self._started = False
        self._stopping = False
        logger.info("CAP TCP transport stopped")

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

        data = self._serializer.encode_request(method, params, id=request_id)
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._write_frame(data)
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

        data = self._serializer.encode_notification(method, params)
        await self._write_frame(data)

    async def send_response(self, result: Any, request_id: int | str) -> None:
        """Send a JSON-RPC response to a server request.

        Used by the connection layer to respond to host API calls.

        Args:
            result: Result value (dataclass or dict).
            request_id: The id from the server's request.
        """
        data = self._serializer.encode_response(result, id=request_id)
        await self._write_frame(data)

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
        encoded = self._serializer.encode_error(code, message, id=request_id, data=data)
        await self._write_frame(encoded)

    async def _write_frame(self, payload: bytes) -> None:
        """Write a length-prefixed frame to the TCP socket.

        Prepends a 4-byte big-endian length prefix to the payload
        and writes under the write lock to prevent interleaving.

        Args:
            payload: Serialized JSON-RPC message bytes.
        """
        assert self._writer is not None

        frame = struct.pack(_LENGTH_FMT, len(payload)) + payload

        async with self._write_lock:
            self._writer.write(frame)
            await self._writer.drain()

    async def _read_loop(self) -> None:
        """Background task: read length-prefixed frames from the socket.

        Reads a 4-byte length prefix, then reads exactly that many bytes
        of payload. Decodes and dispatches each message.
        """
        assert self._reader is not None

        try:
            while not self._stopping:
                # Read 4-byte length prefix
                length_bytes = await self._read_exactly(_LENGTH_SIZE)
                if length_bytes is None:
                    # EOF -- connection closed
                    break

                (payload_length,) = struct.unpack(_LENGTH_FMT, length_bytes)

                # Sanity check: reject absurdly large frames
                if payload_length > _MAX_FRAME_SIZE:
                    logger.error(
                        "CAP TCP: frame too large (%d bytes), closing",
                        payload_length,
                    )
                    break

                if payload_length == 0:
                    continue

                # Read the payload
                payload = await self._read_exactly(payload_length)
                if payload is None:
                    # EOF mid-frame -- connection closed
                    break

                try:
                    msg = self._serializer.decode(payload)
                except (ValueError, Exception) as e:
                    logger.warning("CAP TCP: invalid message from server: %s", e)
                    continue

                self._dispatch(msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("CAP TCP reader error: %s", e)
        finally:
            # EOF or error: fail all pending requests
            if not self._stopping:
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(PluginTransportError("Server connection lost"))
                self._pending.clear()

    async def _read_exactly(self, n: int) -> bytes | None:
        """Read exactly n bytes from the stream.

        Returns None on EOF (when fewer bytes are available than requested).

        Args:
            n: Number of bytes to read.

        Returns:
            Exactly n bytes, or None if EOF was reached.
        """
        assert self._reader is not None

        try:
            data = await self._reader.readexactly(n)
        except asyncio.IncompleteReadError:
            return None
        except (ConnectionError, OSError):
            return None
        return data

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route an incoming message to the correct handler."""
        if "id" in msg and "method" not in msg:
            # Response (has id, no method)
            self._handle_response(msg)
        elif "method" in msg and "id" not in msg:
            # Notification (has method, no id)
            self._handle_notification(msg)
        elif "method" in msg and "id" in msg:
            # Request from server (has both method and id)
            self._handle_server_request(msg)
        else:
            logger.warning("CAP TCP: unrecognized message: %s", msg)

    def _handle_response(self, msg: dict[str, Any]) -> None:
        """Handle a JSON-RPC response."""
        msg_id = msg["id"]
        future = self._pending.pop(msg_id, None)
        if future is None:
            logger.warning("CAP TCP: response for unknown id: %s", msg_id)
            return

        if "error" in msg:
            err = msg["error"]
            future.set_exception(
                JsonRpcError(
                    code=err.get("code", ErrorCodes.INTERNAL_ERROR),
                    message=err.get("message", "Unknown error"),
                    data=err.get("data"),
                )
            )
        else:
            future.set_result(msg.get("result"))

    def _handle_notification(self, msg: dict[str, Any]) -> None:
        """Handle a server -> host notification."""
        method = msg["method"]
        params = msg.get("params", {})
        if self._on_notification:
            try:
                self._on_notification(method, params)
            except Exception as e:
                logger.error("CAP TCP notification handler error: %s", e)
        else:
            logger.debug("CAP TCP: unhandled notification: %s", method)

    def _handle_server_request(self, msg: dict[str, Any]) -> None:
        """Handle a server -> host request (host API call).

        Routed through the notification callback with the full message
        so the connection layer can send a response.
        """
        method = msg["method"]
        params = msg.get("params", {})
        request_id = msg["id"]

        if self._on_notification:
            try:
                self._on_notification(
                    method,
                    {"_request_id": request_id, **params},
                )
            except Exception as e:
                logger.error("CAP TCP host API handler error: %s", e)
        else:
            logger.warning("CAP TCP: unhandled host API request: %s", method)
