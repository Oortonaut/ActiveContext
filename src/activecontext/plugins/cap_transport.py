"""CAP Transport -- abstract transport protocol.

Defines the interface that all CAP transport implementations must satisfy.
Transport-agnostic: stdio, gRPC, WebSocket, TCP all implement this.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# Re-export the callback type
from activecontext.plugins.transport import NotificationCallback

__all__ = [
    "CAPTransport",
    "NotificationCallback",
]


@runtime_checkable
class CAPTransport(Protocol):
    """Transport protocol for CAP communication.

    All transport implementations (stdio, gRPC, WebSocket, TCP)
    must satisfy this interface. PluginConnection programs against
    this protocol, never against concrete transports.
    """

    @property
    def is_running(self) -> bool:
        """Whether the transport is currently connected and operational."""
        ...

    async def start(self) -> None:
        """Start the transport connection.

        Raises:
            PluginTransportError: If connection fails.
        """
        ...

    async def stop(self) -> None:
        """Stop the transport and release resources.

        Cancels pending requests. Idempotent.
        """
        ...

    async def send_request(
        self, method: str, params: Any = None, timeout: float = 30.0
    ) -> Any:
        """Send a request and wait for response.

        Args:
            method: RPC method name.
            params: Parameters.
            timeout: Seconds to wait.

        Returns:
            Result from server.

        Raises:
            JsonRpcError: Server returned an error.
            PluginTransportError: Transport not running.
            asyncio.TimeoutError: No response within timeout.
        """
        ...

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Send a fire-and-forget notification.

        Raises:
            PluginTransportError: Transport not running.
        """
        ...

    async def send_response(self, result: Any, request_id: int | str) -> None:
        """Send a response to a server request (host API)."""
        ...

    async def send_error_response(
        self,
        code: int,
        message: str,
        request_id: int | str | None,
        data: Any = None,
    ) -> None:
        """Send an error response to a server request."""
        ...
