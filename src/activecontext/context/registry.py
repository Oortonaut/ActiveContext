"""Node type registry for managing context node types.

This module provides a centralized registry for all context node types,
enabling dynamic plugin loading and avoiding the if-elif chain in from_dict().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode


class NodeTypeRegistry:
    """Registry for context node type definitions.

    Manages built-in and plugin-provided node types. Provides lookup
    for deserialization and supports dynamic registration/unregistration.
    """

    # Set of node_type strings that are builtin (cannot be unregistered)
    _builtin_types: set[str]

    def __init__(self) -> None:
        """Initialize the registry with built-in types."""
        self._types: dict[str, type[ContextNode]] = {}
        self._builtin_types = set()
        self._load_builtin_types()

    def _load_builtin_types(self) -> None:
        """Load all built-in node types from nodes.py."""
        # Import here to avoid circular imports
        from activecontext.context.nodes import (
            AgentNode,
            ArtifactNode,
            GroupNode,
            LockNode,
            MCPManagerNode,
            MCPServerNode,
            MCPToolNode,
            MessageNode,
            SessionNode,
            ShellNode,
            TaskNode,
            TextNode,
            TopicNode,
            TraceNode,
            WorkNode,
        )

        builtin_types: list[type[ContextNode]] = [
            TextNode,
            GroupNode,
            TopicNode,
            ArtifactNode,
            ShellNode,
            LockNode,
            SessionNode,
            MessageNode,
            WorkNode,
            MCPServerNode,
            MCPToolNode,
            MCPManagerNode,
            AgentNode,
            TraceNode,
            TaskNode,
        ]

        for cls in builtin_types:
            # Create a temporary instance to get the node_type property value
            # We need to be careful here since dataclasses require field values
            # Instead, use the class name convention
            node_type = self._get_node_type_from_class(cls)
            self._types[node_type] = cls
            self._builtin_types.add(node_type)

    def _get_node_type_from_class(self, cls: type[ContextNode]) -> str:
        """Extract node_type string from a ContextNode subclass.

        Uses the class's node_type property by creating a minimal instance
        or inferring from class name.
        """
        # Map class names to their node_type strings
        # This avoids instantiation issues with dataclass required fields
        type_map = {
            "TextNode": "text",
            "GroupNode": "group",
            "TopicNode": "topic",
            "ArtifactNode": "artifact",
            "ShellNode": "shell",
            "LockNode": "lock",
            "SessionNode": "session",
            "MessageNode": "message",
            "WorkNode": "work",
            "MCPServerNode": "mcp_server",
            "MCPToolNode": "mcp_tool",
            "MCPManagerNode": "mcp_manager",
            "AgentNode": "agent",
            "TraceNode": "trace",
            "TaskNode": "task",
        }
        return type_map.get(cls.__name__, cls.__name__.lower().replace("node", ""))

    def get(self, node_type: str) -> type[ContextNode] | None:
        """Get a node class by type identifier.

        Args:
            node_type: The node type string (e.g., "text", "group")

        Returns:
            The node class, or None if not found
        """
        return self._types.get(node_type)

    def register(self, node_type: str, cls: type[ContextNode]) -> None:
        """Register a new node type.

        Args:
            node_type: The node type string identifier
            cls: The ContextNode subclass to register

        Raises:
            ValueError: If node_type already exists as a builtin type
        """
        if node_type in self._builtin_types:
            raise ValueError(
                f"Cannot override builtin node type: {node_type}"
            )
        self._types[node_type] = cls

    def unregister(self, node_type: str) -> bool:
        """Unregister a node type.

        Args:
            node_type: The node type string to remove

        Returns:
            True if removed, False if not found or is builtin
        """
        if node_type in self._builtin_types:
            return False
        if node_type in self._types:
            del self._types[node_type]
            return True
        return False

    def list_types(self) -> list[tuple[str, type[ContextNode]]]:
        """List all registered node types.

        Returns:
            List of (node_type, class) tuples
        """
        return list(self._types.items())

    def is_builtin(self, node_type: str) -> bool:
        """Check if a node type is a builtin type.

        Args:
            node_type: The node type string to check

        Returns:
            True if the type is builtin, False otherwise
        """
        return node_type in self._builtin_types

    def from_dict(self, data: dict[str, Any]) -> ContextNode:
        """Deserialize a node from a dictionary using the registry.

        Args:
            data: Dictionary containing serialized node data with 'node_type' key

        Returns:
            The deserialized ContextNode instance

        Raises:
            ValueError: If the node_type is unknown
        """
        node_type = data.get("node_type")
        cls = self._types.get(node_type)  # type: ignore[arg-type]
        if cls is None:
            raise ValueError(f"Unknown node type: {node_type}")
        # All ContextNode subclasses implement _from_dict as a classmethod
        # that returns an instance of that specific subclass
        from_dict_method = getattr(cls, "_from_dict")
        result: ContextNode = from_dict_method(data)
        return result


# Global registry instance
_registry: NodeTypeRegistry | None = None


def get_node_registry() -> NodeTypeRegistry:
    """Get the global node type registry.

    Returns:
        The singleton NodeTypeRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = NodeTypeRegistry()
    return _registry
