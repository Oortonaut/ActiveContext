"""MCP (Model Context Protocol) related nodes.

This module defines nodes for MCP server connections and tool documentation:
- MCPServerNode: Represents an MCP server connection with its available tools
- MCPToolNode: Individual tool from an MCP server with schema
- MCPManagerNode: Singleton manager that tracks all MCP server connections
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import (
    Expansion,
    NotificationLevel,
    TickFrequency,
)
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from activecontext.context.headers import TokenInfo


@trace_all_fields
@dataclass(kw_only=True)
class MCPToolNode(ContextNode):
    """Represents an individual tool from an MCP server.

    Child node of MCPServerNode, displaying tool name, description, and schema.
    Each tool node has independent state control for granular visibility.

    Attributes:
        tool_name: Name of the tool (e.g., "read_file")
        server_name: Parent server name for context
        description: Tool description
        input_schema: JSON Schema for tool parameters

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: Just tool_name
        - SUMMARY: tool_name: description (truncated)
        - DETAILS: Name, description, required params
        - ALL: Full JSON schema
    """

    tool_name: str = ""
    server_name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "has_schema": bool(self.input_schema.get("properties")),
            "expansion": self.default_expansion.value,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render description and parameters (no headings — header via Render())."""
        parts: list[str] = []

        # Description
        parts.append(f"{self.description}\n\n")

        props = self.input_schema.get("properties", {})
        if props:
            required = set(self.input_schema.get("required", []))
            parts.append("**Parameters:**\n")
            for param, param_schema in props.items():
                param_type = param_schema.get("type", "any")
                param_desc = param_schema.get("description", "")
                req_marker = " (required)" if param in required else ""
                parts.append(f"- `{param}` ({param_type}){req_marker}: {param_desc}\n")
            parts.append("\n")

        return "".join(parts)

    def render_digest(self) -> str:
        """Return tool name for display."""
        return f"{self.server_name}.{self.tool_name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCPToolNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "tool_name": self.tool_name,
                "server_name": self.server_name,
                "description": self.description,
                "input_schema": self.input_schema,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MCPToolNode:
        """Deserialize MCPToolNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "header")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            tool_name=data.get("tool_name", ""),
            server_name=data.get("server_name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
        )


@trace_all_fields
@dataclass(kw_only=True)
class MCPServerNode(ContextNode):
    """Represents an MCP server connection with its available tools.

    Renders tool documentation for the LLM to understand available capabilities.
    The LLM can call tools via server.tool_name(**kwargs) in the namespace.

    Attributes:
        server_name: Unique name identifying the MCP server
        status: Connection status (disconnected, connecting, connected, error)
        error_message: Error message if status is "error"
        tools: List of available tools with name, description, input_schema
        resources: List of available resources with uri, name, description
        prompts: List of available prompts with name, description, arguments

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: "MCP: server_name [OK] (X tools)"
        - SUMMARY: Server + tool names list
        - DETAILS: Tool names + brief descriptions
        - ALL: Full documentation with JSON schemas
    """

    server_name: str = ""
    status: str = "disconnected"  # disconnected, connecting, connected, error
    error_message: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)

    # Enable notifications for MCP server state changes
    notification_level: NotificationLevel = NotificationLevel.HOLD

    # Pending async tool calls: call_id -> (tool_name, started_at)
    pending_calls: dict[str, tuple[str, float]] = field(default_factory=dict)

    # Callback for firing events when calls complete
    # Set by Timeline: (event_name, data) -> None
    _on_result_callback: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # Tool child nodes: tool_name -> node_id
    _tool_nodes: dict[str, str] = field(default_factory=dict, repr=False)

    # Runtime reference to server proxy for tool calls (not serialized)
    _server_proxy: Any = field(default=None, init=False, repr=False, compare=False)

    def __getattr__(self, name: str) -> Any:
        """Delegate tool method access to the server proxy."""
        proxy = self.__dict__.get("_server_proxy")
        if proxy is not None:
            try:
                return getattr(proxy, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "server_name": self.server_name,
            "status": self.status,
            "tool_count": len(self.tools),
            "resource_count": len(self.resources),
            "prompt_count": len(self.prompts),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def tool(self, name: str) -> MCPToolNode | None:
        """Get a tool child node by name.

        Args:
            name: Tool name (e.g., "read_file")

        Returns:
            MCPToolNode if found, None otherwise
        """
        node_id = self._tool_nodes.get(name)
        if node_id and self._graph:
            node = self._graph.get_node(node_id)
            if isinstance(node, MCPToolNode):
                return node
        return None

    @property
    def tool_nodes(self) -> list[MCPToolNode]:
        """Get all tool child nodes."""
        if not self._graph:
            return []
        nodes: list[MCPToolNode] = []
        for node_id in self._tool_nodes.values():
            node = self._graph.get_node(node_id)
            if isinstance(node, MCPToolNode):
                nodes.append(node)
        return nodes

    def _render_status_message(self) -> str | None:
        """Return status message for error/disconnected states, or None if connected."""
        if self.status == "error" and self.error_message:
            return f"Error: {self.error_message}\n\n"
        if self.status != "connected":
            return f"Status: {self.status}\n"
        return None

    def render_content(self) -> str:
        """Render tool names, usage hint, resources, and prompts."""
        status = self._render_status_message()
        if status:
            return status

        parts: list[str] = []
        tool_names = [t.get("name", "?") for t in self.tools]
        parts.append(f"Tools: {', '.join(tool_names)}\n")

        # Usage hint
        parts.append("To call tools:\n")
        parts.append("```python/acrepl\n")
        parts.append(f"result = {self.server_name}.tool_name(arg=value)\n")
        parts.append("```\n\n")

        # Resources
        if self.resources:
            parts.append("Resources:\n")
            for res in self.resources:
                parts.append(f"- `{res.get('uri', '?')}`")
                if res.get("description"):
                    parts.append(f": {res['description']}")
                parts.append("\n")

        # Prompts
        if self.prompts:
            parts.append("\nPrompts:\n")
            for prompt in self.prompts:
                parts.append(f"- **{prompt.get('name', '?')}**")
                if prompt.get("description"):
                    parts.append(f": {prompt['description']}")
                parts.append("\n")

        return "".join(parts)

    def update_from_connection(self, connection: Any) -> None:
        """Update node state from an MCPConnection object.

        Creates/updates/removes MCPToolNode children based on tool changes.
        Generates traces for removed tools to maintain audit trail.
        """
        self.status = connection.status.value
        self.error_message = connection.error_message

        # Build incoming tool data
        incoming_tools: dict[str, dict[str, Any]] = {}
        for t in connection.tools:
            incoming_tools[t.name] = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }

        # Update tools list (kept for backward compat / Render fallback)
        self.tools = list(incoming_tools.values())

        # Diff tool child nodes if graph is available
        if self._graph:
            current_tool_names = set(self._tool_nodes.keys())
            incoming_tool_names = set(incoming_tools.keys())

            # Remove tools that no longer exist
            removed = current_tool_names - incoming_tool_names
            for tool_name in removed:
                node_id = self._tool_nodes.pop(tool_name)
                node = self._graph.get_node(node_id)
                if node:
                    # Generate trace before removal
                    node.mark_changed(f"Tool '{tool_name}' removed from {self.server_name}")
                    self._graph.remove_node(node_id)

            # Add new tools
            added = incoming_tool_names - current_tool_names
            for tool_name in added:
                tool_data = incoming_tools[tool_name]
                tool_node = MCPToolNode(
                    tool_name=tool_name,
                    server_name=self.server_name,
                    description=tool_data["description"],
                    input_schema=tool_data["input_schema"],
                    default_expansion=Expansion.HEADER,
                )
                self._graph.add_node(tool_node)
                self._graph.link(tool_node.node_id, self.node_id)
                self._tool_nodes[tool_name] = tool_node.node_id

            # Update existing tools if schema/description changed
            unchanged = current_tool_names & incoming_tool_names
            for tool_name in unchanged:
                node_id = self._tool_nodes[tool_name]
                node = self._graph.get_node(node_id)
                if isinstance(node, MCPToolNode):
                    tool_data = incoming_tools[tool_name]
                    changed = False
                    if node.description != tool_data["description"]:
                        node.description = tool_data["description"]
                        changed = True
                    if node.input_schema != tool_data["input_schema"]:
                        node.input_schema = tool_data["input_schema"]
                        changed = True
                    if changed:
                        node.mark_changed(f"Tool '{tool_name}' schema updated")

        # Update resources and prompts (no child nodes for these yet)
        self.resources = [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
            }
            for r in connection.resources
        ]
        self.prompts = [
            {
                "name": p.name,
                "description": p.description,
                "arguments": p.arguments,
            }
            for p in connection.prompts
        ]
        self.mark_changed(f"MCP {self.server_name}: {self.status}, {len(self.tools)} tools")

    def start_call(self, call_id: str, tool_name: str) -> None:
        """Register a pending async tool call.

        Args:
            call_id: Unique ID for this call
            tool_name: Name of the tool being called
        """
        self.pending_calls[call_id] = (tool_name, time.time())

    def complete_call(
        self,
        call_id: str,
        result: Any,
        error: str | None = None,
    ) -> None:
        """Complete an async tool call and fire event.

        Args:
            call_id: ID of the call to complete
            result: Tool result (if successful)
            error: Error message (if failed)
        """
        call_info = self.pending_calls.pop(call_id, None)
        if not call_info:
            return

        tool_name, started_at = call_info
        duration_ms = (time.time() - started_at) * 1000

        # Mark the node as changed
        self.mark_changed(
            f"MCP tool '{tool_name}' completed",
            content=str(result)[:200] if result else error,
        )

        # Fire event if callback is set
        if self._on_result_callback:
            self._on_result_callback(
                "mcp_result",
                {
                    "call_id": call_id,
                    "server_name": self.server_name,
                    "tool_name": tool_name,
                    "result": result,
                    "error": error,
                    "duration_ms": duration_ms,
                },
            )

    def get_pending_count(self) -> int:
        """Get the number of pending async calls."""
        return len(self.pending_calls)

    def set_on_result_callback(
        self, callback: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        """Set the callback for MCP result events.

        Args:
            callback: Function to call with (event_name, data) when a call completes
        """
        self._on_result_callback = callback

    def render_digest(self) -> str:
        """Return 'MCP: name [STATUS] N tools' format."""
        return f"MCP: {self.server_name} [{self.status.upper()}] {len(self.tools)} tools"

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCPServerNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "server_name": self.server_name,
                "status": self.status,
                "error_message": self.error_message,
                "tools": self.tools,
                "resources": self.resources,
                "prompts": self.prompts,
                "_tool_nodes": self._tool_nodes,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MCPServerNode:
        """Deserialize MCPServerNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            server_name=data.get("server_name", ""),
            status=data.get("status", "disconnected"),
            error_message=data.get("error_message"),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            prompts=data.get("prompts", []),
            _tool_nodes=data.get("_tool_nodes", {}),
        )
        return node


@trace_all_fields
@dataclass(kw_only=True)
class MCPManagerNode(ContextNode):
    """Singleton manager that tracks all MCP server connections.

    This node aggregates state from all MCPServerNode children and tracks
    connection state changes, tool changes, and resource changes as traces.

    The manager is created automatically and has a fixed node_id="mcp_manager".
    Multiple observer nodes can reference it via the context graph.

    Attributes:
        server_states: Dict mapping server name to last known status
        tool_counts: Dict mapping server name to tool count
        resource_counts: Dict mapping server name to resource count
        connection_events: Recent connection state changes (for rendering)

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: "MCP Manager: X servers (Y connected)"
        - SUMMARY: Server status list
        - DETAILS: Server status + tool/resource counts
        - ALL: Full details + recent connection events
    """

    # Track last known state for diff generation
    server_states: dict[str, str] = field(default_factory=dict)  # name -> status
    tool_counts: dict[str, int] = field(default_factory=dict)  # name -> count
    resource_counts: dict[str, int] = field(default_factory=dict)

    # Recent events for rendering
    connection_events: list[dict[str, Any]] = field(default_factory=list)
    max_events: int = 10

    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        total = len(self.server_states)
        connected = sum(1 for s in self.server_states.values() if s == "connected")
        return {
            "id": self.node_id,
            "type": self.node_type,
            "total_servers": total,
            "connected_servers": connected,
            "server_states": dict(self.server_states),
        }

    def render_content(self) -> str:
        """Return empty - details are in child MCPServerNode headers, events are traces."""
        return ""

    def on_child_changed(self, child: ContextNode, description: str = "") -> None:
        """Handle MCPServerNode changes - track state transitions."""
        if not isinstance(child, MCPServerNode):
            return

        name = child.server_name
        old_status = self.server_states.get(name)
        new_status = child.status
        changes: list[str] = []

        # Track state change
        if old_status != new_status:
            self.server_states[name] = new_status
            changes.append(f"MCP '{name}': {old_status or 'new'} -> {new_status}")

            # Record event
            import time as time_module

            self.connection_events.append(
                {
                    "time": time_module.strftime("%H:%M:%S"),
                    "server": name,
                    "message": f"{name}: {old_status or 'new'} -> {new_status}",
                }
            )
            if len(self.connection_events) > self.max_events:
                self.connection_events.pop(0)

        # Track tool/resource count changes
        old_tools = self.tool_counts.get(name, 0)
        new_tools = len(child.tools)
        if old_tools != new_tools:
            self.tool_counts[name] = new_tools
            changes.append(f"MCP '{name}' tools: {old_tools} -> {new_tools}")

        old_resources = self.resource_counts.get(name, 0)
        new_resources = len(child.resources)
        if old_resources != new_resources:
            self.resource_counts[name] = new_resources

        change_desc = "; ".join(changes) if changes else description
        self.mark_changed(change_desc)
        self.notify_parents(change_desc)

    def register_server(self, server_node: MCPServerNode) -> None:
        """Register a server node as a child of this manager."""
        self.server_states[server_node.server_name] = server_node.status
        self.tool_counts[server_node.server_name] = len(server_node.tools)
        self.resource_counts[server_node.server_name] = len(server_node.resources)

    def unregister_server(self, server_name: str) -> None:
        """Remove a server from tracking."""
        self.server_states.pop(server_name, None)
        self.tool_counts.pop(server_name, None)
        self.resource_counts.pop(server_name, None)

    def render_digest(self) -> str:
        """Return 'MCP Manager (N/M connected, X tools)' format."""
        total = len(self.server_states)
        connected = sum(1 for s in self.server_states.values() if s == "connected")
        total_tools = sum(self.tool_counts.values())
        return f"MCP Manager ({connected}/{total} connected, {total_tools} tools)"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts - content is always empty."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        # Title: digest line only
        total = len(self.server_states)
        connected = sum(1 for s in self.server_states.values() if s == "connected")
        total_tools = sum(self.tool_counts.values())
        title_text = f"MCP Manager ({connected}/{total} connected, {total_tools} tools)\n"
        title_tokens = count_tokens(title_text)

        return TokenInfo(title=title_tokens, content=0, index=self.index_tokens, detail=0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        d = super().to_dict()
        d.update(
            {
                "server_states": dict(self.server_states),
                "tool_counts": dict(self.tool_counts),
                "resource_counts": dict(self.resource_counts),
                "connection_events": list(self.connection_events),
                "max_events": self.max_events,
            }
        )
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> MCPManagerNode:
        """Deserialize from dict."""
        # Parse tick_frequency if present
        tick_freq = None
        if d.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(d["tick_frequency"])

        node = cls(
            node_id=d.get("node_id", "mcp_manager"),
            parent_ids=set(d.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                d.get("child_order") or d.get("children_ids") or []
            ),
            default_expansion=Expansion(d.get("expansion", "content")),
            mode=d.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=d.get("version", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            display_sequence=d.get("display_sequence"),
            originator=d.get("originator"),
            title=d.get("title", ""),
            server_states=d.get("server_states", {}),
            tool_counts=d.get("tool_counts", {}),
            resource_counts=d.get("resource_counts", {}),
            connection_events=d.get("connection_events", []),
            max_events=d.get("max_events", 10),
        )
        return node
