"""MCP server integration manager.

Encapsulates MCP server connections, namespace bindings, and node management.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import MCPManagerNode, MCPServerNode
from activecontext.context.state import Expansion
from activecontext.mcp.client import MCPClientManager
from activecontext.mcp.roots import RootsManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from activecontext.config.schema import MCPConfig
    from activecontext.context.graph import ContextGraph

_log = logging.getLogger("activecontext.mcp.integration")

# Module-level CLI roots, set once at startup via set_cli_roots()
_cli_roots: list[tuple[str, str]] = []


def set_cli_roots(roots: list[tuple[str, str]]) -> None:
    """Store CLI --root entries for use by MCPIntegration instances.

    Called from __main__.py before any sessions are created.
    """
    global _cli_roots
    _cli_roots = list(roots)


def to_snake_identifier(name: str) -> str:
    """Convert an arbitrary server name to a valid Python snake_case identifier.

    Examples:
        >>> to_snake_identifier("rust-filesystem")
        'rust_filesystem'
        >>> to_snake_identifier("My Server")
        'my_server'
        >>> to_snake_identifier("CamelCase")
        'camel_case'
        >>> to_snake_identifier("123-start")
        'start'
    """
    # Insert underscore before uppercase transitions (CamelCase → camel_case)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Replace non-alphanumeric characters with underscores
    s = re.sub(r"[^a-zA-Z0-9]", "_", s)
    # Lowercase
    s = s.lower()
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores and leading digits
    s = s.strip("_")
    s = re.sub(r"^[0-9]+_?", "", s)
    return s or "server"


class MCPIntegration:
    """Manages MCP server connections and integration with the context graph.

    Responsibilities:
    - Connect/disconnect MCP servers
    - Manage MCPServerNode instances
    - Provide query interface (list, tools)
    """

    def __init__(
        self,
        *,
        mcp_config: MCPConfig | None = None,
        context_graph: ContextGraph,
        fire_event: Callable[[str, dict[str, Any]], str | None],
        cwd: str = ".",
    ):
        """Initialize MCP integration manager.

        Args:
            mcp_config: MCP configuration with server definitions
            context_graph: The session's context graph for adding nodes
            fire_event: Callback for MCP result events
            cwd: Working directory (auto-registered as "project" root)
        """
        self._mcp_client_manager = MCPClientManager(config=mcp_config)
        self._mcp_server_nodes: dict[str, MCPServerNode] = {}
        self._context_graph = context_graph
        self._fire_event = fire_event

        # Initialize roots manager with config roots, cwd, and CLI roots
        self._roots_manager = RootsManager()
        self._init_roots(mcp_config, cwd)
        self._mcp_client_manager.set_roots_manager(self._roots_manager)

    def _init_roots(
        self,
        mcp_config: MCPConfig | None,
        cwd: str,
    ) -> None:
        """Populate roots from config, cwd, and CLI args."""
        # Auto-register cwd as "project" root
        abs_cwd = os.path.abspath(cwd)
        self._roots_manager.add(abs_cwd, name="project")
        _log.info("Auto-registered cwd root: %s", abs_cwd)

        # Load roots from config
        if mcp_config:
            for root_cfg in mcp_config.roots:
                path = root_cfg.path
                if not os.path.isabs(path):
                    path = os.path.join(abs_cwd, path)
                path = os.path.abspath(path)
                self._roots_manager.add(path, name=root_cfg.name)
                _log.info("Registered config root '%s': %s", root_cfg.name, path)

        # Load extra roots from CLI --root (set via set_cli_roots())
        for name, path in _cli_roots:
            if not os.path.isabs(path):
                path = os.path.join(abs_cwd, path)
            path = os.path.abspath(path)
            self._roots_manager.add(path, name=name)
            _log.info("Registered CLI root '%s': %s", name, path)

    async def connect(
        self,
        name: str | None = None,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        expansion: Expansion = Expansion.ALL,
    ) -> MCPServerNode:
        """Connect to an MCP server.

        Args:
            name: Server name from config, or custom name for dynamic connection
            command: For stdio transport: command and args to spawn server
            url: For streamable-http transport: server URL
            env: Environment variables for the server process
            state: Initial rendering state

        Returns:
            MCPServerNode representing the connection

        Examples:
            # Connect to configured server
            fs = mcp_connect("filesystem")

            # Dynamic stdio connection
            gh = mcp_connect("github", command=["npx", "-y", "@mcp/server-github"])

            # Dynamic HTTP connection
            tools = mcp_connect("remote", url="http://localhost:8000/mcp")
        """
        from activecontext.config.schema import MCPConnectMode, MCPServerConfig

        # Build config if dynamic
        config: MCPServerConfig | None = None
        if command or url:
            config = MCPServerConfig(
                name=name or "dynamic",
                command=command,
                url=url,
                env=env or {},
                transport="stdio" if command else "streamable-http",
            )
            name = config.name
        elif name is None:
            raise ValueError("Must provide 'name' or 'command'/'url'")

        # Check if server is disabled (NEVER mode) when using config from file
        if config is None and self._mcp_client_manager.config:
            for server_config in self._mcp_client_manager.config.servers:
                if server_config.name == name:
                    if server_config.connect == MCPConnectMode.NEVER:
                        raise ValueError(f"MCP server '{name}' is disabled (connect=never)")
                    break

        # Connect via MCPClientManager
        connection = await self._mcp_client_manager.connect(name=name, config=config)

        # Create or update MCPServerNode
        if name in self._mcp_server_nodes:
            node = self._mcp_server_nodes[name]
        else:
            identifier = to_snake_identifier(name)
            node = MCPServerNode(
                node_id=identifier,
                server_name=name,
                default_expansion=expansion,
            )
            self._mcp_server_nodes[name] = node
            self._context_graph.add_node(node)

            # Wire up MCP result callback to fire events
            # Use lambda to match expected signature (returns None)
            node.set_on_result_callback(
                lambda event_name, data: (self._fire_event(event_name, data), None)[1]
            )

        # Update node from connection
        node.update_from_connection(connection)

        # Register with MCPManagerNode
        mcp_manager = self._context_graph.get_node("mcp_manager")
        if mcp_manager and isinstance(mcp_manager, MCPManagerNode):
            # Link in graph (server as child of manager)
            self._context_graph.link(node.node_id, mcp_manager.node_id)
            mcp_manager.register_server(node)

        # Attach ServerProxy to node (used for tool dispatch via NodeView)
        bindings = self._mcp_client_manager.generate_namespace_bindings()
        if name in bindings:
            proxy = bindings[name]
            proxy._mcp_node = node
            proxy.tool = node.tool
            proxy.tool_nodes = node.tool_nodes
            node._server_proxy = proxy

        return node

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            name: Name of the server to disconnect
        """
        await self._mcp_client_manager.disconnect(name)

        # Update node status and clean up tool children
        if name in self._mcp_server_nodes:
            node = self._mcp_server_nodes[name]

            # Remove tool nodes from graph
            for tool_name, tool_node_id in list(node._tool_nodes.items()):
                tool_node = self._context_graph.get_node(tool_node_id)
                if tool_node:
                    tool_node.mark_changed(f"Tool '{tool_name}' removed (server disconnected)")
                    self._context_graph.remove_node(tool_node_id)
            node._tool_nodes.clear()

            node.status = "disconnected"
            node.tools = []
            node.resources = []
            node.prompts = []
            node.mark_changed(f"MCP {name}: disconnected")

        # Unregister from MCPManagerNode
        mcp_manager = self._context_graph.get_node("mcp_manager")
        if mcp_manager and isinstance(mcp_manager, MCPManagerNode):
            mcp_manager.unregister_server(name)

    def list_connections(self) -> list[dict[str, Any]]:
        """List all MCP server connections and their status.

        Returns:
            List of connection info dicts with name, status, tool/resource counts.
        """
        return [
            {
                "name": conn.name,
                "status": conn.status.value,
                "tools": len(conn.tools),
                "resources": len(conn.resources),
                "prompts": len(conn.prompts),
            }
            for conn in self._mcp_client_manager.list_connections()
        ]

    def list_tools(self, server: str | None = None) -> list[dict[str, Any]]:
        """List available MCP tools, optionally filtered by server.

        Args:
            server: Optional server name to filter by

        Returns:
            List of tool info dicts with server, name, description.
        """
        tools = self._mcp_client_manager.get_all_tools()
        if server:
            tools = [t for t in tools if t.server_name == server]
        return [
            {
                "server": t.server_name,
                "name": t.name,
                "description": t.description,
            }
            for t in tools
        ]

    async def cleanup(self) -> None:
        """Disconnect from all MCP servers. Called during session cleanup."""
        await self._mcp_client_manager.disconnect_all()

    def add_root(self, path: str, name: str | None = None) -> dict[str, str | None]:
        """Add a filesystem root for MCP servers.

        Args:
            path: Filesystem path (absolute or relative to cwd).
            name: Optional display name.

        Returns:
            Dict with uri and name of the added root.
        """
        root = self._roots_manager.add(path, name=name)
        return {"uri": root.uri, "name": root.name}

    def remove_root(self, path: str) -> bool:
        """Remove a filesystem root.

        Args:
            path: Path or file:// URI to remove.

        Returns:
            True if removed, False if not found.
        """
        return self._roots_manager.remove(path)

    def list_roots(self) -> list[dict[str, str | None]]:
        """List all registered MCP roots.

        Returns:
            List of dicts with uri and name for each root.
        """
        return [{"uri": r.uri, "name": r.name} for r in self._roots_manager.list_roots()]

    @property
    def roots_manager(self) -> RootsManager:
        """Access the underlying RootsManager."""
        return self._roots_manager

    def generate_namespace_bindings(self) -> dict[str, Any]:
        """Generate initial namespace bindings for all configured servers.

        Returns:
            Dict of server_name -> ServerProxy for namespace setup.
        """
        return self._mcp_client_manager.generate_namespace_bindings()
