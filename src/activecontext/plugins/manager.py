"""CAP Plugin Manager -- server lifecycle and node orchestration.

Manages the lifecycle of plugin server connections and provides
the bridge between the context graph and remote plugin nodes.

Follows the delegate pattern used by ShellManager and MCPIntegration:
- Created by Timeline/Session
- Called during tick to sync remote nodes
- Provides DSL-facing functions
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from activecontext.plugins.cap_transport import CAPTransport
from activecontext.plugins.connection import PluginConnection
from activecontext.plugins.remote_node import RemoteNode
from activecontext.plugins.wire import Methods, RootInfo

logger = logging.getLogger(__name__)


@dataclass
class PluginServerConfig:
    """Configuration for a plugin server."""

    name: str
    command: list[str] | None = None  # For stdio (host-managed)
    address: str | None = None  # For network transports (external)
    transport: str = "stdio"  # "stdio", "grpc", "websocket", "tcp"
    env: dict[str, str] | None = None
    cwd: str | None = None
    auto_connect: bool = False


@dataclass
class PluginConnectionInfo:
    """Status info for a plugin connection."""

    name: str
    status: str
    server_name: str
    server_version: str
    node_types: list[str]
    transport: str


class PluginManager:
    """Manages plugin server connections and remote node lifecycle.

    Responsibilities:
    1. Connect/disconnect plugin servers
    2. Register node types from server schemas
    3. Create RemoteNode instances
    4. Sync dirty remote nodes on tick
    5. Handle host API calls from servers
    6. Track roots for server access
    """

    def __init__(
        self,
        registry: Any,  # NodeTypeRegistry - use Any to avoid circular import
        session_id: str = "",
        cwd: str = "",
        fire_event: Callable[..., None] | None = None,
    ) -> None:
        self._registry = registry
        self._session_id = session_id
        self._cwd = cwd
        self._fire_event = fire_event

        self._connections: dict[str, PluginConnection] = {}
        self._remote_nodes: dict[str, RemoteNode] = {}  # node_id -> RemoteNode
        self._node_to_server: dict[str, str] = {}  # node_id -> server_name
        self._roots: list[RootInfo] = []

    # --- Connection Lifecycle ---

    async def connect(
        self,
        name: str,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        transport: CAPTransport | None = None,
    ) -> PluginConnection:
        """Connect to a plugin server.

        Args:
            name: Server name for identification.
            command: Command to spawn (stdio transport).
            env: Additional environment variables.
            cwd: Working directory.
            transport: Pre-configured transport (for non-stdio).

        Returns:
            The connected PluginConnection.

        Raises:
            PluginTransportError: If connection fails.
            ValueError: If already connected or no command/transport given.
        """
        if name in self._connections:
            raise ValueError(f"Already connected to '{name}'")

        if command is None and transport is None:
            raise ValueError("Must provide either command (stdio) or transport")

        conn = PluginConnection(
            name=name,
            command=command or [],
            env=env,
            cwd=cwd,
            on_notification=self._on_notification,
            on_host_api=self._on_host_api,
            transport=transport,
        )

        await conn.connect(
            session_id=self._session_id,
            cwd=self._cwd,
            roots=self._roots,
        )

        self._connections[name] = conn

        # Register node types from server
        for descriptor in conn.descriptors:
            try:
                self._registry.register_plugin(descriptor)
            except ValueError as e:
                logger.warning(
                    "Failed to register node type '%s': %s",
                    descriptor.node_type,
                    e,
                )

        logger.info(
            "Plugin '%s' connected: %d node types registered",
            name,
            len(conn.node_types),
        )

        if self._fire_event:
            self._fire_event("plugin_connected", name=name)

        return conn

    async def disconnect(self, name: str) -> None:
        """Disconnect from a plugin server.

        Unregisters node types and marks remote nodes as stale.
        """
        conn = self._connections.pop(name, None)
        if conn is None:
            return

        # Unregister node types
        for descriptor in conn.descriptors:
            self._registry.unregister_plugin(descriptor.node_type)

        # Mark all remote nodes from this server as stale
        for node_id, server_name in list(self._node_to_server.items()):
            if server_name == name:
                node = self._remote_nodes.get(node_id)
                if node is not None:
                    node.clear_connection()

        await conn.disconnect()

        logger.info("Plugin '%s' disconnected", name)

        if self._fire_event:
            self._fire_event("plugin_disconnected", name=name)

    async def disconnect_all(self) -> None:
        """Disconnect from all plugin servers."""
        names = list(self._connections.keys())
        for name in names:
            await self.disconnect(name)

    # --- Node Creation ---

    async def create_node(
        self,
        server_name: str,
        node_type: str,
        *args: Any,
        node_id: str | None = None,
        **kwargs: Any,
    ) -> RemoteNode:
        """Create a remote node instance.

        Sends node/create to the server and returns a RemoteNode proxy.

        Args:
            server_name: Which plugin server to create the node on.
            node_type: Type of node to create.
            *args: Positional constructor arguments.
            node_id: Optional node ID (server generates if not given).
            **kwargs: Named constructor arguments.

        Returns:
            A RemoteNode proxy.

        Raises:
            ValueError: If server not connected.
            PluginTransportError: If RPC fails.
        """
        conn = self._connections.get(server_name)
        if conn is None:
            raise ValueError(f"Not connected to '{server_name}'")

        # Send create request
        result = await conn.send_request(
            Methods.NODE_CREATE,
            {
                "node_type": node_type,
                "args": list(args),
                "kwargs": kwargs,
                "node_id": node_id or "",
            },
        )

        remote_id = result.get("node_id", node_id or "")
        initial_state = result.get("initial_state", {})

        # Create proxy
        node = RemoteNode(
            _actual_node_type=node_type,
            _connection=conn,
            _remote_id=remote_id,
            _server_name=server_name,
            _cached_state=initial_state,
        )

        self._remote_nodes[remote_id] = node
        self._node_to_server[remote_id] = server_name

        return node

    async def destroy_node(self, node_id: str) -> None:
        """Destroy a remote node."""
        server_name = self._node_to_server.get(node_id)
        if server_name is None:
            return

        conn = self._connections.get(server_name)
        if conn is not None:
            try:
                await conn.send_request(
                    Methods.NODE_DESTROY,
                    {
                        "node_id": node_id,
                    },
                )
            except Exception as e:
                logger.warning("Failed to destroy remote node '%s': %s", node_id, e)

        self._remote_nodes.pop(node_id, None)
        self._node_to_server.pop(node_id, None)

    # --- Tick Synchronization ---

    async def process_pending_results(self) -> list[str]:
        """Sync all dirty remote nodes. Called during tick.

        Returns:
            List of node IDs that were synced.
        """
        synced: list[str] = []

        for node_id, node in list(self._remote_nodes.items()):
            if node._dirty or node._pending_calls:
                try:
                    await node.tick_async(self._cwd)
                    synced.append(node_id)
                except Exception as e:
                    logger.error("Sync failed for node '%s': %s", node_id, e)

        return synced

    # --- Query ---

    def list_connections(self) -> list[PluginConnectionInfo]:
        """List all active plugin connections."""
        result = []
        for name, conn in self._connections.items():
            result.append(
                PluginConnectionInfo(
                    name=name,
                    status=conn.status.value,
                    server_name=conn.server_name,
                    server_version=conn.server_version,
                    node_types=[nt.node_type for nt in conn.node_types],
                    transport="stdio",
                )
            )
        return result

    def get_connection(self, name: str) -> PluginConnection | None:
        """Get a connection by name."""
        return self._connections.get(name)

    def get_remote_node(self, node_id: str) -> RemoteNode | None:
        """Get a remote node by ID."""
        return self._remote_nodes.get(node_id)

    def set_roots(self, roots: list[RootInfo]) -> None:
        """Update the roots exposed to plugin servers."""
        self._roots = roots

    # --- Host API Handlers ---

    def _on_notification(self, method: str, params: dict[str, Any]) -> None:
        """Handle push notifications from plugin servers."""
        if method == Methods.NODE_DIRTY:
            node_id = params.get("node_id", "")
            node = self._remote_nodes.get(node_id)
            if node is not None:
                node.mark_dirty()

        if self._fire_event:
            self._fire_event("plugin_notification", method=method, params=params)

    def _on_host_api(self, method: str, params: dict[str, Any], request_id: int | str) -> Any:
        """Handle host API calls from plugin servers.

        These are reverse requests where the server calls back into the host.
        """
        if method == Methods.HOST_QUERY_ROOTS:
            return {"roots": [{"uri": r.uri, "name": r.name} for r in self._roots]}

        if method == Methods.HOST_RESOLVE_ROOT:
            uri = params.get("uri", "")
            for root in self._roots:
                if root.uri == uri:
                    return {"path": root.uri.replace("file://", "")}
            return {"path": "", "error": "Root not found"}

        logger.warning("Unhandled host API call: %s", method)
        return None
