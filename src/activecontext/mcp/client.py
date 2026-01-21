"""MCP client manager for connecting to MCP servers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp import types
from mcp.client.session import ClientSession

from activecontext.mcp.transport import create_transport
from activecontext.mcp.types import (
    MCPConnectionStatus,
    MCPPromptInfo,
    MCPResourceInfo,
    MCPToolInfo,
    MCPToolResult,
)

if TYPE_CHECKING:
    from activecontext.config.schema import MCPConfig, MCPServerConfig

_log = logging.getLogger("activecontext.mcp.client")


@dataclass
class MCPConnection:
    """Represents an active MCP server connection."""

    name: str
    config: MCPServerConfig
    session: ClientSession | None = None
    status: MCPConnectionStatus = MCPConnectionStatus.DISCONNECTED
    tools: list[MCPToolInfo] = field(default_factory=list)
    resources: list[MCPResourceInfo] = field(default_factory=list)
    prompts: list[MCPPromptInfo] = field(default_factory=list)
    error_message: str | None = None
    _transport_context: Any = None
    _read_stream: Any = None
    _write_stream: Any = None

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        self.status = MCPConnectionStatus.CONNECTING
        try:
            self._transport_context = await create_transport(self.config)
            streams = await self._transport_context.__aenter__()
            self._read_stream, self._write_stream = streams[0], streams[1]

            self.session = ClientSession(self._read_stream, self._write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            # Discover tools
            tools_result = await self.session.list_tools()
            self.tools = [
                MCPToolInfo(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema,
                    server_name=self.name,
                )
                for t in tools_result.tools
            ]

            # Discover resources
            try:
                resources_result = await self.session.list_resources()
                self.resources = [
                    MCPResourceInfo(
                        uri=str(r.uri),
                        name=r.name,
                        description=r.description,
                        mime_type=r.mimeType,
                        server_name=self.name,
                    )
                    for r in resources_result.resources
                ]
            except Exception:
                # Resources are optional
                self.resources = []

            # Discover prompts
            try:
                prompts_result = await self.session.list_prompts()
                self.prompts = [
                    MCPPromptInfo(
                        name=p.name,
                        description=p.description,
                        arguments=[
                            {"name": a.name, "description": a.description, "required": a.required}
                            for a in (p.arguments or [])
                        ],
                        server_name=self.name,
                    )
                    for p in prompts_result.prompts
                ]
            except Exception:
                # Prompts are optional
                self.prompts = []

            self.status = MCPConnectionStatus.CONNECTED
            _log.info(
                f"Connected to MCP server '{self.name}' with "
                f"{len(self.tools)} tools, {len(self.resources)} resources, "
                f"{len(self.prompts)} prompts"
            )

        except Exception as e:
            self.status = MCPConnectionStatus.ERROR
            self.error_message = str(e)
            _log.error(f"Failed to connect to MCP server '{self.name}': {e}")
            raise

    async def disconnect(self) -> None:
        """Close the connection."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except Exception as e:
                _log.warning(f"Error closing session for '{self.name}': {e}")
            self.session = None
        if self._transport_context:
            try:
                await self._transport_context.__aexit__(None, None, None)
            except Exception as e:
                _log.warning(f"Error closing transport for '{self.name}': {e}")
            self._transport_context = None
        self.status = MCPConnectionStatus.DISCONNECTED
        self.tools = []
        self.resources = []
        self.prompts = []
        _log.info(f"Disconnected from MCP server '{self.name}'")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        """Call a tool on this server."""
        if not self.session or self.status != MCPConnectionStatus.CONNECTED:
            return MCPToolResult(
                success=False,
                content=[],
                is_error=True,
                error_message=f"Server '{self.name}' is not connected",
            )

        try:
            result = await self.session.call_tool(tool_name, arguments)

            # Convert content blocks
            content = []
            for block in result.content:
                if isinstance(block, types.TextContent):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, types.ImageContent):
                    content.append({
                        "type": "image",
                        "data": block.data,
                        "mime_type": block.mimeType,
                    })
                elif isinstance(block, types.EmbeddedResource):
                    res = block.resource
                    content.append({
                        "type": "resource",
                        "uri": str(res.uri) if hasattr(res, "uri") else None,
                        "text": res.text if hasattr(res, "text") else None,
                    })

            return MCPToolResult(
                success=True,
                content=content,
                structured_content=result.structuredContent,
                is_error=result.isError or False,
            )

        except Exception as e:
            _log.error(f"Tool call failed: {self.name}.{tool_name}: {e}")
            return MCPToolResult(
                success=False,
                content=[],
                is_error=True,
                error_message=str(e),
            )

    async def read_resource(self, uri: str) -> MCPToolResult:
        """Read a resource from this server."""
        if not self.session or self.status != MCPConnectionStatus.CONNECTED:
            return MCPToolResult(
                success=False,
                content=[],
                is_error=True,
                error_message=f"Server '{self.name}' is not connected",
            )

        try:
            from pydantic import AnyUrl

            result = await self.session.read_resource(AnyUrl(uri))

            content = []
            for block in result.contents:
                if isinstance(block, types.TextContent):
                    content.append({"type": "text", "text": block.text})
                elif hasattr(block, "data"):
                    content.append({
                        "type": "blob",
                        "data": block.data,
                        "mime_type": getattr(block, "mimeType", None),
                    })

            return MCPToolResult(success=True, content=content)

        except Exception as e:
            return MCPToolResult(
                success=False,
                content=[],
                is_error=True,
                error_message=str(e),
            )


# Type alias for permission callback
MCPPermissionCallback = Callable[[str, str, dict[str, Any]], Awaitable[bool]]


@dataclass
class MCPClientManager:
    """Manages multiple MCP server connections for a session."""

    connections: dict[str, MCPConnection] = field(default_factory=dict)
    config: MCPConfig | None = None
    _permission_callback: MCPPermissionCallback | None = None

    def set_permission_callback(self, callback: MCPPermissionCallback) -> None:
        """Set callback for permission checks: (server_name, tool_name, args) -> allowed."""
        self._permission_callback = callback

    async def connect(
        self,
        name: str | None = None,
        config: MCPServerConfig | None = None,
    ) -> MCPConnection:
        """Connect to an MCP server by name (from config) or with explicit config.

        Args:
            name: Server name from config, or custom name for dynamic connection
            config: Explicit server configuration for dynamic connections

        Returns:
            MCPConnection representing the active connection

        Raises:
            ValueError: If server not found or invalid configuration
        """
        if config:
            # Dynamic connection with provided config
            if name is None:
                name = config.name
        elif name:
            # Look up in configured servers
            if self.config:
                for server_config in self.config.servers:
                    if server_config.name == name:
                        config = server_config
                        break
            if config is None:
                raise ValueError(f"MCP server '{name}' not found in config")
        else:
            raise ValueError("Must provide either 'name' or 'config'")

        # Check if already connected
        if name in self.connections:
            existing = self.connections[name]
            if existing.status == MCPConnectionStatus.CONNECTED:
                return existing
            # Reconnect if in error state
            await existing.disconnect()

        # Create and connect
        connection = MCPConnection(name=name, config=config)
        await connection.connect()
        self.connections[name] = connection
        return connection

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server."""
        if name in self.connections:
            await self.connections[name].disconnect()
            del self.connections[name]

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self.connections.keys()):
            await self.disconnect(name)

    def get_connection(self, name: str) -> MCPConnection | None:
        """Get an active connection by name."""
        return self.connections.get(name)

    def list_connections(self) -> list[MCPConnection]:
        """List all active connections."""
        return list(self.connections.values())

    def get_all_tools(self) -> list[MCPToolInfo]:
        """Get tools from all connected servers."""
        tools = []
        for conn in self.connections.values():
            if conn.status == MCPConnectionStatus.CONNECTED:
                tools.extend(conn.tools)
        return tools

    def generate_namespace_bindings(self) -> dict[str, Any]:
        """Generate namespace bindings for all connected servers.

        Returns dict like:
        {
            "filesystem": <ServerProxy with .read_file(), .write_file(), etc.>,
            "github": <ServerProxy with .search_repos(), etc.>,
        }
        """
        bindings = {}
        for conn in self.connections.values():
            if conn.status == MCPConnectionStatus.CONNECTED:
                bindings[conn.name] = ServerProxy(conn, self._permission_callback)
        return bindings


class ServerProxy:
    """Proxy object that exposes MCP tools as async methods.

    The LLM can call tools like:
        result = filesystem.read_file(path="/home/user/data.txt")
    """

    def __init__(
        self,
        connection: MCPConnection,
        permission_callback: MCPPermissionCallback | None = None,
    ):
        self._connection = connection
        self._permission_callback = permission_callback
        self._tool_methods: dict[str, Callable[..., Awaitable[MCPToolResult]]] = {}

        # Create method for each tool
        for tool in connection.tools:
            method = self._make_tool_method(tool)
            self._tool_methods[tool.name] = method
            setattr(self, tool.name, method)

    def _make_tool_method(
        self, tool: MCPToolInfo
    ) -> Callable[..., Awaitable[MCPToolResult]]:
        """Create an async method for calling a tool."""
        from activecontext.mcp.hooks import get_pre_call_hook
        from activecontext.mcp.permissions import MCPPermissionDenied

        async def tool_method(**kwargs: Any) -> MCPToolResult:
            # Permission check
            if self._permission_callback:
                allowed = await self._permission_callback(
                    self._connection.name,
                    tool.name,
                    kwargs,
                )
                if not allowed:
                    raise MCPPermissionDenied(
                        server_name=self._connection.name,
                        tool_name=tool.name,
                        arguments=kwargs,
                    )

            # Pre-call hook for UI feedback (e.g., "Calling rider.list_files...")
            hook = get_pre_call_hook()
            if hook:
                await hook(self._connection.name, tool.name, kwargs)

            return await self._connection.call_tool(tool.name, kwargs)

        tool_method.__doc__ = tool.description
        tool_method.__name__ = tool.name
        return tool_method

    def __repr__(self) -> str:
        tools = [t.name for t in self._connection.tools]
        return f"<MCPServer '{self._connection.name}' tools={tools}>"

    def __dir__(self) -> list[str]:
        return list(self._tool_methods.keys()) + ["_connection", "_permission_callback"]
