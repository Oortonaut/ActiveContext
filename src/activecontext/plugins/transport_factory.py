"""CAP Transport Factory -- create transports from configuration.

Maps transport type strings to concrete CAPTransport implementations.
Gracefully errors when a required transport library is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from activecontext.plugins.cap_transport import CAPTransport
from activecontext.plugins.grpc_transport import GrpcTransport
from activecontext.plugins.tcp_transport import TcpTransport
from activecontext.plugins.transport import PluginTransportError, StdioTransport
from activecontext.plugins.ws_transport import WebSocketTransport

logger = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Configuration for a CAP transport.

    Holds all parameters needed to instantiate any supported transport type.
    Use ``from_dict`` to construct from a YAML-style configuration dictionary.

    Attributes:
        type: Transport type identifier. One of "stdio", "grpc", "websocket", "tcp".
        command: Command and arguments for stdio transport.
        env: Environment variables for stdio transport.
        cwd: Working directory for stdio transport.
        address: host:port for gRPC and TCP transports.
        url: WebSocket URL (ws://...) for websocket transport.
        timeout: Connection/request timeout in seconds.
    """

    type: str = "stdio"

    # stdio options
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    # Network options (grpc, websocket, tcp)
    address: str = ""
    url: str = ""

    # Common options
    timeout: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransportConfig:
        """Create from a config dict (e.g. parsed YAML).

        The ``transport`` key maps to the ``type`` field. All other keys
        map directly to their corresponding fields.

        Args:
            data: Configuration dictionary.

        Returns:
            A populated TransportConfig.
        """
        return cls(
            type=data.get("transport", "stdio"),
            command=data.get("command", []),
            env=data.get("env", {}),
            cwd=data.get("cwd"),
            address=data.get("address", ""),
            url=data.get("url", ""),
            timeout=data.get("timeout", 30.0),
        )


def create_transport(
    config: TransportConfig,
    on_notification: Any = None,
) -> CAPTransport:
    """Create a CAPTransport from configuration.

    Supports multiple transport types:
    - ``stdio``: JSON-RPC over subprocess stdio pipes (always available)
    - ``tcp``: JSON-RPC over TCP with length-prefix framing (always available)
    - ``websocket``: JSON-RPC over WebSocket (requires aiohttp)
    - ``grpc``: Protobuf over gRPC/HTTP2 (not yet implemented)

    Args:
        config: Transport configuration.
        on_notification: Notification callback for the transport.

    Returns:
        A configured CAPTransport instance.

    Raises:
        PluginTransportError: If the transport type is unknown or the
            required implementation is not yet available.
        ValueError: If required config fields are missing for the
            selected transport type.
    """
    transport_type = config.type.lower()

    if transport_type == "stdio":
        if not config.command:
            raise ValueError("stdio transport requires 'command'")
        return StdioTransport(
            command=config.command,
            env=config.env or None,
            cwd=config.cwd,
            on_notification=on_notification,
        )

    if transport_type == "grpc":
        if not config.address:
            raise ValueError("gRPC transport requires 'address' (host:port)")
        return GrpcTransport(
            target=config.address,
            on_notification=on_notification,
        )

    if transport_type == "websocket":
        if not config.url:
            raise ValueError("WebSocket transport requires 'url' (ws://...)")
        return WebSocketTransport(
            url=config.url,
            on_notification=on_notification,
        )

    if transport_type == "tcp":
        if not config.address:
            raise ValueError("TCP transport requires 'address' (host:port)")
        host, _, port_str = config.address.partition(":")
        if not port_str:
            raise ValueError(f"TCP address must be 'host:port', got: {config.address}")
        try:
            port = int(port_str)
        except ValueError as e:
            raise ValueError(f"Invalid TCP port: {port_str}") from e
        return TcpTransport(
            host=host,
            port=port,
            on_notification=on_notification,
        )

    raise PluginTransportError(
        f"Unknown transport type: '{transport_type}'. Supported types: stdio, grpc, websocket, tcp"
    )


def list_available_transports() -> list[str]:
    """List transport types that are currently available.

    Returns transport type strings that can be successfully instantiated
    (given valid configuration). Currently only ``stdio`` is always
    available. Future transports will appear here once their backend
    libraries are installed.

    Returns:
        List of transport type strings that can be instantiated.
    """
    available = ["stdio"]  # Always available

    # Check for optional transport libraries
    try:
        import grpc  # type: ignore[import-untyped]  # noqa: F401

        available.append("grpc")
    except ImportError:
        pass

    try:
        import aiohttp  # noqa: F401

        available.append("websocket")
    except ImportError:
        pass

    # TCP is always available (stdlib asyncio)
    available.append("tcp")

    return available
