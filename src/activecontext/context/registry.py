"""Node type registry for managing context node types.

This module provides a centralized registry for all context node types,
enabling dynamic plugin loading and avoiding the if-elif chain in from_dict().

The registry supports three sources of node types:
- Builtin: Hardcoded ContextNode subclasses (protected from override)
- Local plugin: Python ContextNode subclasses from plugin packages
- Remote plugin: Cross-language nodes via CAP JSON-RPC servers
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode
    from activecontext.plugins.descriptor import NodePluginDescriptor


@dataclass
class PluginInfo:
    """Information about a registered plugin node type."""

    node_type: str
    """Type identifier (e.g., "text", "group", "custom_lint")."""

    name: str
    """Human-readable name."""

    description: str
    """Brief description of what this node type does."""

    version: str
    """Plugin version string."""

    author: str
    """Plugin author."""

    source_path: Path | None
    """Path to the plugin source (None for builtins or remote plugins)."""

    is_builtin: bool
    """True if this is a builtin node type."""

    is_loaded: bool
    """True if the plugin is currently loaded."""


class NodeTypeRegistry:
    """Registry for context node type definitions.

    Manages built-in and plugin-provided node types. Provides lookup
    for deserialization and supports dynamic registration/unregistration.

    Plugin descriptors carry rich metadata (schema, source, server name)
    alongside the basic type → class mapping. The existing get/register/
    unregister API continues to work for backward compatibility; the new
    plugin API adds descriptor-level operations.
    """

    # Set of node_type strings that are builtin (cannot be unregistered)
    _builtin_types: set[str]

    def __init__(self) -> None:
        """Initialize the registry with built-in types."""
        self._types: dict[str, type[ContextNode]] = {}
        self._descriptors: dict[str, NodePluginDescriptor] = {}
        self._builtin_types = set()
        self._load_builtin_types()

    def _load_builtin_types(self) -> None:
        """Load all built-in node types from nodes.py."""
        # Import here to avoid circular imports
        from activecontext.context.nodes import (
            AgentNode,
            ArtifactNode,
            GroupNode,
            HelpNode,
            LockNode,
            MCPManagerNode,
            MCPServerNode,
            MCPToolNode,
            MessageNode,
            MessageSegmentNode,
            PluginManagerNode,
            PtyNode,
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
            PtyNode,
            LockNode,
            SessionNode,
            MessageNode,
            MessageSegmentNode,
            WorkNode,
            MCPServerNode,
            MCPToolNode,
            MCPManagerNode,
            PluginManagerNode,
            AgentNode,
            TraceNode,
            TaskNode,
            HelpNode,
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
            "PtyNode": "pty",
            "LockNode": "lock",
            "SessionNode": "session",
            "MessageNode": "message",
            "MessageSegmentNode": "segment",
            "WorkNode": "work",
            "MCPServerNode": "mcp_server",
            "MCPToolNode": "mcp_tool",
            "MCPManagerNode": "mcp_manager",
            "PluginManagerNode": "plugin_manager",
            "AgentNode": "agent",
            "TraceNode": "trace",
            "TaskNode": "task",
            "HelpNode": "help",
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
            raise ValueError(f"Cannot override builtin node type: {node_type}")
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
        node_type: str | None = data.get("node_type")
        cls: type[ContextNode] | None = self._types.get(node_type)  # type: ignore[arg-type]
        if cls is None:
            raise ValueError(f"Unknown node type: {node_type}")
        # All ContextNode subclasses implement _from_dict as a classmethod
        # that returns an instance of that specific subclass
        from_dict_method = cls._from_dict  # type: ignore[attr-defined]
        result: ContextNode = from_dict_method(data)
        return result

    # ------------------------------------------------------------------
    # Plugin descriptor API
    # ------------------------------------------------------------------

    def register_plugin(self, descriptor: NodePluginDescriptor) -> None:
        """Register a node type from a plugin descriptor.

        Adds both the type class (if local/builtin) and the descriptor
        metadata. For remote plugins, only the descriptor is stored —
        the RemoteNode proxy class is used for instantiation.

        Args:
            descriptor: Plugin descriptor with schema and source info.

        Raises:
            ValueError: If node_type conflicts with a builtin type.
        """
        node_type = descriptor.node_type
        if node_type in self._builtin_types:
            raise ValueError(f"Cannot override builtin node type: {node_type}")
        if descriptor.node_cls is not None:
            self._types[node_type] = descriptor.node_cls
        self._descriptors[node_type] = descriptor

    def unregister_plugin(self, node_type: str) -> bool:
        """Unregister a plugin node type.

        Removes both the type class and descriptor. Builtin types
        cannot be unregistered.

        Args:
            node_type: The node type to remove.

        Returns:
            True if removed, False if not found or is builtin.
        """
        if node_type in self._builtin_types:
            return False
        removed = False
        if node_type in self._types:
            del self._types[node_type]
            removed = True
        if node_type in self._descriptors:
            del self._descriptors[node_type]
            removed = True
        return removed

    def get_descriptor(self, node_type: str) -> NodePluginDescriptor | None:
        """Get the plugin descriptor for a node type.

        Returns None for builtin types that were registered before
        the descriptor system existed (they have no descriptor).

        Args:
            node_type: The node type identifier.

        Returns:
            The descriptor, or None if not found.
        """
        return self._descriptors.get(node_type)

    def list_plugins(self) -> list[NodePluginDescriptor]:
        """List all registered plugin descriptors.

        Returns:
            List of all descriptors (excludes builtins without descriptors).
        """
        return list(self._descriptors.values())

    def is_plugin(self, node_type: str) -> bool:
        """Check if a node type was registered via the plugin system.

        Args:
            node_type: The node type to check.

        Returns:
            True if the type has a plugin descriptor.
        """
        return node_type in self._descriptors

    def is_remote(self, node_type: str) -> bool:
        """Check if a node type is provided by a remote CAP server.

        Args:
            node_type: The node type to check.

        Returns:
            True if the type is a remote plugin.
        """
        from activecontext.plugins.descriptor import PluginSource

        desc: NodePluginDescriptor | None = self._descriptors.get(node_type)
        return desc is not None and desc.source == PluginSource.REMOTE

    def get_server_name(self, node_type: str) -> str | None:
        """Get the CAP server name for a remote plugin node type.

        Args:
            node_type: The node type to look up.

        Returns:
            Server name string, or None if not a remote plugin.
        """
        desc: NodePluginDescriptor | None = self._descriptors.get(node_type)
        if desc is not None:
            return desc.server_name
        return None

    # ------------------------------------------------------------------
    # Plugin lifecycle management
    # ------------------------------------------------------------------

    def unload_plugin(
        self, node_type: str, force: bool = False, has_active_nodes: bool = False
    ) -> bool:
        """Unload a plugin node type.

        Args:
            node_type: The node type to unload.
            force: If True, unload even if there are active nodes.
            has_active_nodes: If True, indicates there are active nodes of this type.

        Returns:
            True if unloaded successfully, False otherwise.

        Raises:
            ValueError: If attempting to unload a builtin type.
            RuntimeError: If there are active nodes and force=False.
        """
        # Cannot unload builtin types
        if node_type in self._builtin_types:
            raise ValueError(f"Cannot unload builtin node type: {node_type}")

        # Check if type is even registered
        if node_type not in self._types and node_type not in self._descriptors:
            return False

        # Check for active nodes if not forcing
        if has_active_nodes and not force:
            raise RuntimeError(
                f"Cannot unload '{node_type}': active nodes exist. "
                f"Use force=True to unload anyway (nodes will become orphaned)."
            )

        # Unregister the plugin
        return self.unregister_plugin(node_type)

    def reload_plugin(self, node_type: str) -> bool:
        """Reload a plugin node type.

        This reloads the Python module for local plugins. For remote plugins,
        this is effectively a no-op (return True if registered).

        Args:
            node_type: The node type to reload.

        Returns:
            True if reloaded successfully, False if not found or is builtin.

        Raises:
            ValueError: If attempting to reload a builtin type.
        """
        # Cannot reload builtin types
        if node_type in self._builtin_types:
            raise ValueError(f"Cannot reload builtin node type: {node_type}")

        # Get the descriptor to check source
        desc: NodePluginDescriptor | None = self._descriptors.get(node_type)
        if desc is None:
            return False

        from activecontext.plugins.descriptor import PluginSource

        # For remote plugins, just verify still registered
        if desc.source == PluginSource.REMOTE:
            return node_type in self._types or node_type in self._descriptors

        # For local plugins, reload the module
        if desc.node_cls is not None:
            module = sys.modules.get(desc.node_cls.__module__)
            if module is not None:
                # Guard: never reload the core nodes module -- doing so
                # re-creates all class objects (breaking isinstance checks)
                # and resets module-level state like _file_watchers.
                if module.__name__ == "activecontext.context.nodes":
                    return True
                try:
                    importlib.reload(module)
                    # Re-register with potentially updated class
                    # Note: This assumes the class name hasn't changed
                    reloaded_cls = getattr(module, desc.node_cls.__name__, None)
                    if reloaded_cls is not None:
                        self._types[node_type] = reloaded_cls
                        # Update descriptor if it has changed
                        # (This is a simple approach; more sophisticated reload
                        # logic could rebuild the full descriptor)
                        return True
                except Exception:
                    return False

        return False

    def get_plugin_info(self, node_type: str) -> PluginInfo | None:
        """Get information about a registered plugin.

        Args:
            node_type: The node type to query.

        Returns:
            PluginInfo if the type is registered, None otherwise.
        """
        # Check if type exists
        cls: type[ContextNode] | None = self._types.get(node_type)
        desc: NodePluginDescriptor | None = self._descriptors.get(node_type)

        if cls is None and desc is None:
            return None

        is_builtin: bool = node_type in self._builtin_types
        is_loaded: bool = node_type in self._types

        # Extract metadata from descriptor or class
        if desc is not None:
            schema = desc.schema
            name = schema.node_type if schema else node_type
            description = schema.description if schema else ""
            version = "unknown"  # schema doesn't have version field
            author = "unknown"  # schema doesn't have author field

            # Get source path for local plugins
            source_path = None
            if desc.source.value == "local_plugin" and desc.node_cls is not None:
                try:
                    module = sys.modules.get(desc.node_cls.__module__)
                    if module and hasattr(module, "__file__") and module.__file__:
                        source_path = Path(module.__file__)
                except Exception:
                    pass
        else:
            # Fallback for types registered without descriptors
            name = node_type
            description = f"{node_type} node"
            version = "unknown"
            author = "unknown"
            source_path = None

            # Try to get source path from class
            if cls is not None:
                try:
                    module = sys.modules.get(cls.__module__)
                    if module and hasattr(module, "__file__") and module.__file__:
                        source_path = Path(module.__file__)
                except Exception:
                    pass

        return PluginInfo(
            node_type=node_type,
            name=name,
            description=description,
            version=version,
            author=author,
            source_path=source_path,
            is_builtin=is_builtin,
            is_loaded=is_loaded,
        )

    # ------------------------------------------------------------------
    # Plugin loading from filesystem
    # ------------------------------------------------------------------

    def load_from_module(self, module_path: str) -> str | None:
        """Load a single plugin module by Python module path.

        Validates that the module contains a ContextNode subclass with
        a valid node_type property and required methods.

        Args:
            module_path: Python module path (e.g., "mypackage.nodes.custom")

        Returns:
            The node_type string if loaded successfully, None otherwise.

        Raises:
            ValueError: If validation fails (no ContextNode subclass, duplicate
                       node_type, missing required methods).
            ImportError: If the module cannot be imported.
        """
        from types import ModuleType

        try:
            # Import the module
            module: ModuleType = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Failed to import plugin module '{module_path}': {e}") from e

        # Find ContextNode subclasses in the module
        from activecontext.context.nodes import ContextNode

        candidates: list[tuple[str, type[ContextNode]]] = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # Check if it's a class, subclass of ContextNode, and not ContextNode itself
            if (
                isinstance(attr, type)
                and issubclass(attr, ContextNode)
                and attr is not ContextNode
                and attr.__module__ == module_path  # Defined in this module
            ):
                candidates.append((attr_name, attr))

        if not candidates:
            raise ValueError(
                f"No ContextNode subclass found in module '{module_path}'. "
                "Plugin modules must define a class extending ContextNode."
            )

        if len(candidates) > 1:
            names = ", ".join(name for name, _ in candidates)
            raise ValueError(
                f"Multiple ContextNode subclasses found in '{module_path}': {names}. "
                "Plugin modules should define exactly one ContextNode subclass."
            )

        class_name, node_cls = candidates[0]

        # Extract node_type from the class
        # Priority: plugin_info dict > minimal instance > class name derivation
        node_type: str | None = None

        # First, try plugin_info dict (most reliable)
        plugin_info_dict = getattr(module, "plugin_info", None)
        if isinstance(plugin_info_dict, dict):
            node_type = plugin_info_dict.get("node_type")

        # If no plugin_info, try to create minimal instance to get property value
        if node_type is None:
            try:
                # Try to instantiate with empty/default args
                # This works for dataclasses with defaults
                from inspect import signature

                sig = signature(node_cls)
                # Build kwargs with defaults for all parameters
                kwargs: dict[str, Any] = {}
                for param_name, param in sig.parameters.items():
                    if param.default is not param.empty:
                        # Has default, skip
                        pass
                    elif param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        # Skip *args, **kwargs
                        pass
                    else:
                        # Required parameter - use None or empty string
                        if param.annotation is str:
                            kwargs[param_name] = ""
                        elif param.annotation is int:
                            kwargs[param_name] = 0
                        elif param.annotation is bool:
                            kwargs[param_name] = False
                        else:
                            kwargs[param_name] = None

                # Try to instantiate
                temp_instance = node_cls(**kwargs)
                # Access node_type property
                if hasattr(temp_instance, "node_type"):
                    node_type_value = temp_instance.node_type
                    if isinstance(node_type_value, str):
                        node_type = node_type_value
            except Exception:
                # Instantiation failed, fall back to class name derivation
                pass

        # Last resort: derive from class name
        if node_type is None:
            # Convert ClassName -> class_name (strip "Node" suffix)
            node_type = class_name.replace("Node", "").lower()
            if not node_type:
                node_type = class_name.lower()

        # Validate node_type is unique
        if node_type in self._builtin_types:
            raise ValueError(
                f"Cannot register plugin '{module_path}': node_type '{node_type}' "
                "conflicts with builtin type."
            )

        if node_type in self._types:
            raise ValueError(
                f"Cannot register plugin '{module_path}': node_type '{node_type}' "
                "is already registered."
            )

        # Validate required methods exist
        required_methods = ["_from_dict", "to_dict", "GetDigest"]
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(node_cls, method_name):
                missing_methods.append(method_name)

        if missing_methods:
            raise ValueError(
                f"Plugin class '{class_name}' in '{module_path}' is missing required methods: "
                f"{', '.join(missing_methods)}"
            )

        # Extract metadata from plugin_info if available
        plugin_info = getattr(module, "plugin_info", {})
        plugin_info.get("name", node_type)
        plugin_info.get("description", f"Custom {node_type} node")
        plugin_info.get("version", "unknown")
        plugin_info.get("author", "unknown")

        # Register the plugin
        self.register(node_type, node_cls)

        return node_type

    def load_from_directory(self, path: Path) -> list[str]:
        """Load all plugin modules from a directory.

        Scans the directory for Python files (*.py) and attempts to load
        each as a plugin module. Skips __init__.py and __pycache__.

        Args:
            path: Directory path containing plugin modules.

        Returns:
            List of successfully loaded node_type strings.

        Note:
            Individual plugin loading errors are caught and logged,
            but do not prevent other plugins from loading.
        """
        if not path.exists():
            return []

        if not path.is_dir():
            raise ValueError(f"Plugin path is not a directory: {path}")

        loaded_types: list[str] = []

        # Find all .py files (excluding __init__.py)
        for py_file in path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue

            # Construct module path
            # For user plugins: ~/.ac/plugins/nodes/my_plugin.py
            # Module path depends on whether this is in sys.path
            # Safest: add directory to sys.path temporarily and import by module name
            module_name = py_file.stem  # filename without .py

            # Add directory to sys.path if not already there
            dir_str = str(path.resolve())
            if dir_str not in sys.path:
                sys.path.insert(0, dir_str)

            try:
                node_type = self.load_from_module(module_name)
                if node_type:
                    loaded_types.append(node_type)
            except (ImportError, ValueError) as e:
                # Log error but continue with other plugins
                import logging

                _log = logging.getLogger(__name__)
                _log.warning(f"Failed to load plugin from {py_file}: {e}")
                continue

        return loaded_types

    def discover_plugins(self) -> list[PluginInfo]:
        """Discover available plugins without loading them.

        Scans standard plugin locations for plugin modules and extracts
        metadata without actually importing and registering them.

        Plugin locations (searched in order):
        1. User plugins: ~/.ac/plugins/nodes/
        2. Project plugins: {cwd}/.ac/plugins/nodes/

        Returns:
            List of PluginInfo for discovered plugins (is_loaded=False).
        """
        from pathlib import Path as PathLib

        discovered: list[PluginInfo] = []
        seen_types: set[str] = set()

        # Standard plugin directories
        import os

        home = PathLib.home()
        user_plugins_dir = home / ".ac" / "plugins" / "nodes"
        cwd = PathLib(os.getcwd())
        project_plugins_dir = cwd / ".ac" / "plugins" / "nodes"

        for plugin_dir in [user_plugins_dir, project_plugins_dir]:
            if not plugin_dir.exists() or not plugin_dir.is_dir():
                continue

            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                # Try to extract metadata without importing
                # Read plugin_info dict from file if present
                try:
                    content = py_file.read_text(encoding="utf-8")
                except Exception:
                    continue

                # Simple parsing: look for plugin_info = {...}
                # This is best-effort; full parsing would require ast module
                import re

                # Pattern to match plugin_info = { ... }
                pattern = r"plugin_info\s*=\s*\{([^}]+)\}"
                match = re.search(pattern, content, re.DOTALL)

                node_type = py_file.stem  # Default from filename
                name = node_type
                description = "Custom node plugin"
                version = "unknown"
                author = "unknown"

                if match:
                    # Try to extract fields from dict content
                    dict_content = match.group(1)
                    # Extract quoted strings for common fields
                    for field in ["node_type", "name", "description", "version", "author"]:
                        field_pattern = rf'["\']?{field}["\']?\s*:\s*["\']([^"\']+)["\']'
                        field_match = re.search(field_pattern, dict_content)
                        if field_match:
                            value = field_match.group(1)
                            if field == "node_type":
                                node_type = value
                            elif field == "name":
                                name = value
                            elif field == "description":
                                description = value
                            elif field == "version":
                                version = value
                            elif field == "author":
                                author = value

                # Skip duplicates
                if node_type in seen_types:
                    continue
                seen_types.add(node_type)

                # Check if already loaded
                is_loaded = node_type in self._types

                discovered.append(
                    PluginInfo(
                        node_type=node_type,
                        name=name,
                        description=description,
                        version=version,
                        author=author,
                        source_path=py_file,
                        is_builtin=False,
                        is_loaded=is_loaded,
                    )
                )

        return discovered


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
