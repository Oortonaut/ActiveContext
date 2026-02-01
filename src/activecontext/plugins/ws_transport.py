"""CAP Transport -- WebSocket implementation (WebSocketTransport).

JSON-RPC 2.0 over WebSocket frames. This transport connects to a remote
CAP plugin server via a WebSocket URL (ws:// or wss://).

Useful for:
- Browser-based plugin UIs
- Electron apps
- Remote plugins over network
- Environments where stdio isn't available

Requires ``aiohttp`` (lazy-imported at start() time):
    pip install aiohttp
    uv add aiohttp

Threading model:
- One asyncio reader task reads WebSocket text frames
- Incoming messages are dispatched: responses to pending futures,
  notifications to the registered callback
- Writes go directly to the WebSocket (serialized under a lock)
- Optional reconnection with exponential backoff
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

__all__ = ["WebSocketTransport"]


class WebSocketTransport:
    """JSON-RPC 2.0 transport over WebSocket.

    Connects to a remote CAP plugin server via WebSocket. Each message
    is a single JSON frame (text). Supports request/response,
    fire-and-forget notifications, and server-initiated requests.

    Usage:
        transport = WebSocketTransport(url="ws://localhost:8080/cap")
        await transport.start()
        result = await transport.send_request("initialize", {...})
        await transport.send_notification("shutdown", {})
        await transport.stop()
    """

    def __init__(
        self,
        url: str,
        on_notification: NotificationCallback | None = None,
        serializer: CAPSerializer | None = None,
        *,
        reconnect: bool = False,
        reconnect_max_delay: float = 60.0,
        reconnect_base_delay: float = 1.0,
    ) -> None:
        """Initialize transport configuration.

        Args:
            url: WebSocket URL to connect to (ws:// or wss://).
            on_notification: Callback for server -> host notifications.
            serializer: Wire format serializer. Defaults to JsonSerializer.
            reconnect: Whether to automatically reconnect on connection loss.
            reconnect_max_delay: Maximum delay between reconnection attempts (seconds).
            reconnect_base_delay: Base delay for exponential backoff (seconds).
        """
        self._url = url
        self._on_notification = on_notification
        self._serializer: CAPSerializer = serializer or JsonSerializer()

        # Reconnection settings
        self._reconnect = reconnect
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_base_delay = reconnect_base_delay

        # Runtime state
        self._session: Any = None  # aiohttp.ClientSession
        self._ws: Any = None  # aiohttp.ClientWebSocketResponse
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._started = False
        self._stopping = False
        self._aiohttp: Any = None  # Lazily imported module

    @property
    def is_running(self) -> bool:
        """Whether the transport is currently connected and operational."""
        return self._started and not self._stopping and self._ws is not None and not self._ws.closed

    async def start(self) -> None:
        """Open a WebSocket connection to the server and start the reader.

        Raises:
            PluginTransportError: If the connection fails or aiohttp is
                not installed.
        """
        if self._started:
            raise PluginTransportError("Transport already started")

        # Lazy import of aiohttp
        try:
            import aiohttp

            self._aiohttp = aiohttp
        except ImportError:
            raise PluginTransportError(
                "WebSocket transport requires the 'aiohttp' package. "
                "Install it with: pip install aiohttp (or: uv add aiohttp)"
            ) from None

        try:
            self._session = self._aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self._url)
        except Exception as e:
            # Clean up partial state
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None
            self._ws = None
            raise PluginTransportError(
                f"Failed to connect to WebSocket server at {self._url}: {e}"
            ) from e

        self._started = True
        self._reader_task = asyncio.create_task(self._read_loop(), name="cap-ws-transport-reader")
        logger.info("CAP WebSocket transport started: %s", self._url)

    async def stop(self) -> None:
        """Stop the transport and close the WebSocket connection.

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

        # Close WebSocket connection
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        # Close HTTP session
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

        self._started = False
        self._stopping = False
        logger.info("CAP WebSocket transport stopped")

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

    async def _write_frame(self, data: bytes) -> None:
        """Write serialized bytes as a WebSocket text frame.

        Sends under the write lock to prevent interleaving.
        """
        assert self._ws is not None

        async with self._write_lock:
            # Send as text frame (JSON is text)
            await self._ws.send_bytes(data)

    async def _read_loop(self) -> None:
        """Background task: read and decode WebSocket frames."""
        assert self._ws is not None
        aiohttp = self._aiohttp
        assert aiohttp is not None

        try:
            async for msg in self._ws:
                if self._stopping:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._handle_frame(msg.data.encode("utf-8"))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self._handle_frame(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error("CAP WebSocket error: %s", self._ws.exception())
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("CAP WebSocket reader error: %s", e)
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
                        name="cap-ws-transport-reconnect",
                    )

    def _handle_frame(self, data: bytes) -> None:
        """Decode a WebSocket frame and dispatch it."""
        try:
            msg = self._serializer.decode(data)
        except (ValueError, Exception) as e:
            logger.warning("CAP WebSocket: invalid message: %s", e)
            return

        self._dispatch(msg)

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
            logger.warning("CAP WebSocket: unrecognized message: %s", msg)

    def _handle_response(self, msg: dict[str, Any]) -> None:
        """Handle a JSON-RPC response."""
        from activecontext.plugins.transport import JsonRpcError

        msg_id = msg["id"]
        future = self._pending.pop(msg_id, None)
        if future is None:
            logger.warning("CAP WebSocket: response for unknown id: %s", msg_id)
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
                logger.error("CAP WebSocket notification handler error: %s", e)
        else:
            logger.debug("CAP WebSocket: unhandled notification: %s", method)

    def _handle_server_request(self, msg: dict[str, Any]) -> None:
        """Handle a server -> host request (host API call).

        These are bidirectional requests where the server calls the host.
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
                logger.error("CAP WebSocket host API handler error: %s", e)
        else:
            logger.warning("CAP WebSocket: unhandled host API request: %s", method)

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
                "CAP WebSocket reconnect attempt %d (delay=%.1fs)",
                attempt,
                delay,
            )
            await asyncio.sleep(delay)

            if self._stopping:
                break

            try:
                # Re-create session and connection
                if self._session is None or self._session.closed:
                    self._session = self._aiohttp.ClientSession()
                self._ws = await self._session.ws_connect(self._url)

                # Restart reader
                self._reader_task = asyncio.create_task(
                    self._read_loop(), name="cap-ws-transport-reader"
                )

                logger.info(
                    "CAP WebSocket reconnected to %s (attempt %d)",
                    self._url,
                    attempt,
                )
                return  # Success

            except Exception as e:
                logger.warning(
                    "CAP WebSocket reconnect attempt %d failed: %s",
                    attempt,
                    e,
                )
                # Close partial state
                if self._ws is not None and not self._ws.closed:
                    with contextlib.suppress(Exception):
                        await self._ws.close()
                self._ws = None

                # Exponential backoff with cap
                delay = min(delay * 2, self._reconnect_max_delay)

        logger.info("CAP WebSocket reconnection stopped")
