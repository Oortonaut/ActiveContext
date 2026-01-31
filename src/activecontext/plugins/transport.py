"""CAP Transport — stdio implementation (StdioTransport).

JSON-RPC 2.0 over stdio pipes. This is the default CAP transport
implementation; see cap_transport.py for the abstract protocol.

Low-level bidirectional transport for communication with plugin servers.
Each message is a single JSON object followed by a newline.

The transport is bidirectional:
- Host → Server: requests (with id) and notifications (no id)
- Server → Host: responses (matching id) and push notifications

Threading model:
- One asyncio reader task reads from the server's stdout
- Incoming messages are dispatched: responses to pending futures,
  notifications to the registered callback
- Writes go directly to the server's stdin (serialized under a lock)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable

from activecontext.plugins.wire import (
    ErrorCodes,
    to_jsonrpc_notification,
    to_jsonrpc_request,
)

logger = logging.getLogger(__name__)

# Type for the notification callback: (method, params) -> None
NotificationCallback = Callable[[str, dict[str, Any]], None]


class PluginTransportError(Exception):
    """Error in the plugin transport layer."""


class JsonRpcError(PluginTransportError):
    """JSON-RPC error response from the server."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        self.code = code
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


class PluginTransport:
    """JSON-RPC 2.0 transport over stdio pipes.

    Manages a subprocess with stdin/stdout pipes for bidirectional
    JSON-RPC communication. Supports both request/response (with id)
    and fire-and-forget notifications.

    Usage:
        transport = PluginTransport(command=["python", "-m", "my_plugin"])
        await transport.start()
        result = await transport.send_request("initialize", {...})
        await transport.send_notification("shutdown", {})
        await transport.stop()
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        on_notification: NotificationCallback | None = None,
    ) -> None:
        """Initialize transport configuration.

        Args:
            command: Command and arguments to spawn the plugin server.
            env: Additional environment variables (merged with os.environ).
            cwd: Working directory for the subprocess.
            on_notification: Callback for server → host notifications.
        """
        self._command = command
        self._env = env
        self._cwd = cwd
        self._on_notification = on_notification

        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future[Any]] = {}
        self._write_lock = asyncio.Lock()
        self._started = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        """Whether the transport subprocess is running."""
        return (
            self._started
            and not self._stopping
            and self._process is not None
            and self._process.returncode is None
        )

    async def start(self) -> None:
        """Spawn the plugin server subprocess and start the reader.

        Raises:
            PluginTransportError: If the subprocess fails to start.
        """
        if self._started:
            raise PluginTransportError("Transport already started")

        # Merge environment
        merged_env = dict(os.environ)
        if self._env:
            merged_env.update(_expand_env_vars(self._env))

        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
                cwd=self._cwd,
            )
        except (OSError, FileNotFoundError) as e:
            raise PluginTransportError(
                f"Failed to start plugin server: {self._command[0]}: {e}"
            ) from e

        self._started = True
        self._reader_task = asyncio.create_task(
            self._read_loop(), name="cap-transport-reader"
        )
        logger.info(
            "CAP transport started: %s (pid=%s)",
            " ".join(self._command),
            self._process.pid,
        )

    async def stop(self) -> None:
        """Stop the transport and terminate the subprocess.

        Cancels pending requests with an error. Waits for the
        subprocess to exit gracefully, then kills if needed.
        """
        if not self._started or self._stopping:
            return

        self._stopping = True

        # Cancel all pending requests
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    PluginTransportError("Transport shutting down")
                )
        self._pending.clear()

        # Cancel reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        # Terminate subprocess
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Already dead

        self._started = False
        self._stopping = False
        logger.info("CAP transport stopped")

    async def send_request(
        self, method: str, params: Any = None, timeout: float = 30.0
    ) -> Any:
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

        msg = to_jsonrpc_request(method, params, id=request_id)
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            await self._write(msg)
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

        msg = to_jsonrpc_notification(method, params)
        await self._write(msg)

    async def _write(self, msg: dict[str, Any]) -> None:
        """Write a JSON-RPC message to stdin."""
        assert self._process is not None
        assert self._process.stdin is not None

        data = json.dumps(msg, separators=(",", ":")) + "\n"
        async with self._write_lock:
            self._process.stdin.write(data.encode("utf-8"))
            await self._process.stdin.drain()

    async def _read_loop(self) -> None:
        """Background task: read JSON-RPC messages from stdout."""
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            while not self._stopping:
                line = await self._process.stdout.readline()
                if not line:
                    # EOF — subprocess exited
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError as e:
                    logger.warning("CAP: invalid JSON from server: %s", e)
                    continue

                self._dispatch(msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("CAP reader error: %s", e)
        finally:
            # EOF or error: fail all pending requests
            if not self._stopping:
                for future in self._pending.values():
                    if not future.done():
                        future.set_exception(
                            PluginTransportError("Server connection lost")
                        )
                self._pending.clear()

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
            # This is a host API call — route to notification callback
            # which the connection layer handles as a reverse request
            self._handle_server_request(msg)
        else:
            logger.warning("CAP: unrecognized message: %s", msg)

    def _handle_response(self, msg: dict[str, Any]) -> None:
        """Handle a JSON-RPC response."""
        msg_id = msg["id"]
        future = self._pending.pop(msg_id, None)
        if future is None:
            logger.warning("CAP: response for unknown id: %s", msg_id)
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
        """Handle a server → host notification."""
        method = msg["method"]
        params = msg.get("params", {})
        if self._on_notification:
            try:
                self._on_notification(method, params)
            except Exception as e:
                logger.error("CAP notification handler error: %s", e)
        else:
            logger.debug("CAP: unhandled notification: %s", method)

    def _handle_server_request(self, msg: dict[str, Any]) -> None:
        """Handle a server → host request (host API call).

        These are bidirectional requests where the server calls the host.
        Routed through the notification callback with the full message
        so the connection layer can send a response.
        """
        method = msg["method"]
        params = msg.get("params", {})
        request_id = msg["id"]

        if self._on_notification:
            # Pass the full message including id so the connection
            # layer can respond
            try:
                self._on_notification(
                    method,
                    {"_request_id": request_id, **params},
                )
            except Exception as e:
                logger.error("CAP host API handler error: %s", e)
        else:
            logger.warning("CAP: unhandled host API request: %s", method)

    async def send_response(self, result: Any, request_id: int | str) -> None:
        """Send a JSON-RPC response to a server request.

        Used by the connection layer to respond to host API calls.

        Args:
            result: Result value (dataclass or dict).
            request_id: The id from the server's request.
        """
        from activecontext.plugins.wire import to_jsonrpc_response

        msg = to_jsonrpc_response(result, id=request_id)
        await self._write(msg)

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
        from activecontext.plugins.wire import to_jsonrpc_error

        msg = to_jsonrpc_error(code, message, id=request_id, data=data)
        await self._write(msg)


# Backward-compatible alias: StdioTransport is the concrete name,
# PluginTransport remains importable for existing code.
StdioTransport = PluginTransport


def _expand_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Expand ${VAR} references in environment variable values."""
    result = {}
    for key, value in env.items():
        if isinstance(value, str) and "${" in value:
            # Expand all ${VAR} references
            import re

            def replacer(match: re.Match[str]) -> str:
                return os.environ.get(match.group(1), "")

            result[key] = re.sub(r"\$\{([^}]+)\}", replacer, value)
        else:
            result[key] = value
    return result
