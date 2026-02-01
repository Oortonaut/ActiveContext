"""Plugin descriptors — registration metadata for node types.

A NodePluginDescriptor is the unit of registration. It tells the system
everything about a node type: where it comes from, its schema, and how
to construct instances.

Sources:
- builtin: Hardcoded ContextNode subclasses (TextNode, GroupNode, etc.)
- local_plugin: Python ContextNode subclasses loaded from plugin packages
- remote_plugin: Cross-language nodes accessed via CAP JSON-RPC
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from activecontext.plugins.wire import NodeTypeSchema

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode


class PluginSource(Enum):
    """Where a node type comes from."""

    BUILTIN = "builtin"
    """Hardcoded in the core (TextNode, GroupNode, etc.)."""

    LOCAL = "local_plugin"
    """Python ContextNode subclass from a plugin package."""

    REMOTE = "remote_plugin"
    """Cross-language node via CAP JSON-RPC server."""


@dataclass
class NodePluginDescriptor:
    """Registration metadata for a node type.

    This is the unit of registration in the plugin registry. It carries
    everything the system needs to:
    - Construct node instances (via class or RPC)
    - Generate DSL functions and documentation
    - Route tick/sync to the correct server
    - Serialize/deserialize node state

    For builtin and local plugins, node_cls is the Python class.
    For remote plugins, node_cls is None — the RemoteNode proxy is
    used instead, and the server_name identifies which CAP server
    provides this type.
    """

    node_type: str
    """Type identifier (e.g., "shell", "topic", "custom_lint")."""

    source: PluginSource
    """Where this node type comes from."""

    schema: NodeTypeSchema
    """Full type schema (constructor, properties, methods)."""

    node_cls: type[ContextNode] | None = None
    """Python class for builtin/local plugins. None for remote."""

    server_name: str | None = None
    """CAP server name for remote plugins. None for local."""

    namespace_entries: dict[str, Any] = field(default_factory=dict)
    """Additional DSL namespace entries this plugin provides.

    Maps name → callable. These are added to the Timeline namespace
    when the plugin is active. For example, a plugin might provide
    helper functions beyond the constructor.
    """

    def __post_init__(self) -> None:
        """Validate descriptor consistency."""
        if self.source == PluginSource.REMOTE and self.server_name is None:
            raise ValueError(f"Remote plugin '{self.node_type}' must specify server_name")
        if self.source != PluginSource.REMOTE and self.node_cls is None:
            raise ValueError(f"Local/builtin plugin '{self.node_type}' must specify node_cls")
