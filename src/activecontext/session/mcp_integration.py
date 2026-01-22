"""MCP server integration manager.

Encapsulates MCP server connections, namespace bindings, and node management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import MCPManagerNode, MCPServerNode
from activecontext.context.state import Expansion
from activecontext.mcp.client import MCPClientManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from activecontext.config.schema import MCPConfig
    from activecontext.context.graph import ContextGraph


class MCPIntegration:
    """Manages MCP server connections and integration with the context graph.

    Responsibilities:
    - Connect/disconnect MCP servers
    - Manage MCPServerNode instances
    - Update namespace bindings when servers connect/disconnect
    - Provide query interface (list, tools)
    """

    def __init__(
        self,
        *,
        mcp_config: MCPConfig | None = None,
        context_graph: ContextGraph,
        namespace: dict[str, Any],
        fire_event: Callable[[str, dict[str, Any]], str | None],
    ):
        """Initialize MCP integration manager.

        Args:
            mcp_config: MCP configuration with server definitions
            context_graph: The session's context graph for adding nodes
            namespace: The session's namespace dict for binding server proxies
            fire_event: Callback for MCP result events
        """
        self._mcp_client_manager = MCPClientManager(config=mcp_config)
        self._mcp_server_nodes: dict[str, MCPServerNode] = {}
        self._context_graph = context_graph
        self._namespace = namespace
        self._fire_event = fire_event

    async def connect(
        self,
        name: str | None = None,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        tokens: int = 1000,
        expansion: Expansion = Expansion.DETAILS,
    ) -> MCPServerNode:
        """Connect to an MCP server.

        Args:
            name: Server name from config, or custom name for dynamic connection
            command: For stdio transport: command and args to spawn server
            url: For streamable-http transport: server URL
            env: Environment variables for the server process
            tokens: Token budget for tool documentation
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
                        raise ValueError(
                            f"MCP server '{name}' is disabled (connect=never)"
                        )
                    break

        # Connect via MCPClientManager
        connection = await self._mcp_client_manager.connect(name=name, config=config)

        # Create or update MCPServerNode
        if name in self._mcp_server_nodes:
            node = self._mcp_server_nodes[name]
        else:
            node = MCPServerNode(
                server_name=name,
                tokens=tokens,
                expansion=expansion,
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

        # Update namespace with the server proxy
        bindings = self._mcp_client_manager.generate_namespace_bindings()
        if name in bindings:
            proxy = bindings[name]
            # Augment proxy with tool() method for accessing MCPToolNode children
            proxy._mcp_node = node
            proxy.tool = node.tool
            proxy.tool_nodes = node.tool_nodes
            self._namespace[name] = proxy

        # Add tool nodes to namespace with {server}_{tool} naming
        for tool_name, tool_node_id in node._tool_nodes.items():
            tool_node = self._context_graph.get_node(tool_node_id)
            if tool_node:
                # e.g., filesystem_read_file
                namespace_key = f"{name}_{tool_name}"
                self._namespace[namespace_key] = tool_node

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

            # Remove tool nodes from namespace and graph
            for tool_name, tool_node_id in list(node._tool_nodes.items()):
                # Remove from namespace
                namespace_key = f"{name}_{tool_name}"
                if namespace_key in self._namespace:
                    del self._namespace[namespace_key]
                # Remove from graph
                tool_node = self._context_graph.get_node(tool_node_id)
                if tool_node:
                    tool_node._mark_changed(
                        description=f"Tool '{tool_name}' removed (server disconnected)"
                    )
                    self._context_graph.remove_node(tool_node_id)
            node._tool_nodes.clear()

            node.status = "disconnected"
            node.tools = []
            node.resources = []
            node.prompts = []
            node._mark_changed(description=f"MCP {name}: disconnected")

        # Unregister from MCPManagerNode
        mcp_manager = self._context_graph.get_node("mcp_manager")
        if mcp_manager and isinstance(mcp_manager, MCPManagerNode):
            mcp_manager.unregister_server(name)

        # Remove from namespace
        if name in self._namespace:
            del self._namespace[name]

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

    def generate_namespace_bindings(self) -> dict[str, Any]:
        """Generate initial namespace bindings for all configured servers.

        Returns:
            Dict of server_name -> ServerProxy for namespace setup.
        """
        return self._mcp_client_manager.generate_namespace_bindings()
