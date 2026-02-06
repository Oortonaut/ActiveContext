"""Plugin manager node for tracking plugin server connections.

This module defines:
- PluginManagerNode: Singleton manager that tracks plugin server connections
  and available plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import (
    Expansion,
    TickFrequency,
)
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class PluginManagerNode(ContextNode):
    """Singleton manager that tracks plugin server connections and available plugins.

    This node aggregates state from the PluginManager and provides visibility
    into loaded vs. available plugins, connection status, and plugin metadata.

    The manager is created automatically and has a fixed node_id="plugin_manager".
    Multiple observer nodes can reference it via the context graph.

    Attributes:
        plugin_states: Dict mapping plugin server name to connection status
        plugin_types: Dict mapping plugin server to list of provided node types
        builtin_count: Number of builtin node types
        loaded_count: Number of currently loaded plugin servers

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: "Plugins: X builtin, Y loaded"
        - SUMMARY: List of loaded plugin types with descriptions
        - DETAILS: Full plugin info including paths, versions
        - ALL: Full details + connection events
    """

    # Track plugin connection state
    plugin_states: dict[str, str] = field(default_factory=dict)  # name -> status
    plugin_types: dict[str, list[str]] = field(default_factory=dict)  # name -> node_types

    # Metadata
    builtin_count: int = 0
    loaded_count: int = 0

    # Recent events for rendering
    connection_events: list[dict[str, Any]] = field(default_factory=list)
    max_events: int = 10

    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "builtin_count": self.builtin_count,
            "loaded_count": self.loaded_count,
            "plugin_states": dict(self.plugin_states),
        }

    def render_content(self) -> str:
        """Render overview, plugin list, and events."""
        lines: list[str] = []

        # Overview
        lines.append("Overview:")
        lines.append(f"- Builtin types: {self.builtin_count}")
        lines.append(f"- Loaded plugin servers: {self.loaded_count}")

        # Plugin status list with details
        if self.plugin_states:
            lines.append("")
            lines.append("Loaded Plugins:")
            for name, status in sorted(self.plugin_states.items()):
                emoji = {
                    "connected": "[OK]",
                    "connecting": "[...]",
                    "error": "[ERR]",
                    "disconnected": "[--]",
                }.get(status, "[?]")
                types = self.plugin_types.get(name, [])
                lines.append(f"- **{name}** {emoji}")
                lines.append(f"  - Status: {status}")
                lines.append(f"  - Node types ({len(types)}): {', '.join(types)}")
        else:
            lines.append("")
            lines.append("No plugin servers loaded.")

        # Recent events
        if self.connection_events:
            lines.append("")
            lines.append("Recent Events:")
            for event in self.connection_events[-5:]:
                lines.append(f"- {event.get('time', '?')}: {event.get('message', '?')}")

        return "\n".join(lines)

    def update_plugin_state(
        self, name: str, status: str, node_types: list[str] | None = None
    ) -> None:
        """Update the state of a plugin connection.

        Args:
            name: Plugin server name
            status: Connection status (connected, connecting, error, disconnected)
            node_types: Optional list of node types provided by this plugin
        """
        old_status = self.plugin_states.get(name)
        self.plugin_states[name] = status

        if node_types is not None:
            self.plugin_types[name] = node_types

        # Update loaded count
        self.loaded_count = sum(1 for s in self.plugin_states.values() if s == "connected")

        # Record event if status changed
        if old_status != status:
            import time as time_module

            self.connection_events.append(
                {
                    "time": time_module.strftime("%H:%M:%S"),
                    "plugin": name,
                    "message": f"{name}: {old_status or 'new'} -> {status}",
                }
            )
            if len(self.connection_events) > self.max_events:
                self.connection_events.pop(0)

            self.mark_changed(f"Plugin '{name}': {old_status or 'new'} -> {status}")

    def unregister_plugin(self, name: str) -> None:
        """Remove a plugin from tracking."""
        self.plugin_states.pop(name, None)
        self.plugin_types.pop(name, None)
        self.loaded_count = sum(1 for s in self.plugin_states.values() if s == "connected")
        self.mark_changed(f"Plugin '{name}' unregistered")

    def render_digest(self) -> str:
        """Return 'Plugin Manager (N builtin, M loaded)' format."""
        return f"Plugin Manager ({self.builtin_count} builtin, {self.loaded_count} loaded)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        d = super().to_dict()
        d.update(
            {
                "plugin_states": dict(self.plugin_states),
                "plugin_types": {k: list(v) for k, v in self.plugin_types.items()},
                "builtin_count": self.builtin_count,
                "loaded_count": self.loaded_count,
                "connection_events": list(self.connection_events),
                "max_events": self.max_events,
            }
        )
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PluginManagerNode:
        """Deserialize from dict."""
        # Parse tick_frequency if present
        tick_freq = None
        if d.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(d["tick_frequency"])

        node = cls(
            node_id=d.get("node_id", "plugin_manager"),
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
            plugin_states=d.get("plugin_states", {}),
            plugin_types=d.get("plugin_types", {}),
            builtin_count=d.get("builtin_count", 0),
            loaded_count=d.get("loaded_count", 0),
            connection_events=d.get("connection_events", []),
            max_events=d.get("max_events", 10),
        )
        return node
