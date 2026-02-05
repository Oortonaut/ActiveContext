"""Tests for plugin descriptor and registry extensions.

Verifies that:
1. NodePluginDescriptor validates consistency
2. Registry plugin API works alongside existing builtin API
3. Remote vs local plugin distinction is maintained
4. Builtin protection is preserved
"""

from __future__ import annotations

import pytest

from activecontext.context.nodes import ShellNode, TopicNode
from activecontext.context.registry import NodeTypeRegistry
from activecontext.plugins.descriptor import (
    NodePluginDescriptor,
    PluginSource,
)
from activecontext.plugins.wire import (
    ConstructorSchema,
    NodeTypeSchema,
    ParamSchema,
)

# ---------------------------------------------------------------------------
# Descriptor tests
# ---------------------------------------------------------------------------


class TestNodePluginDescriptor:
    """Descriptor validation and construction."""

    def test_local_plugin_descriptor(self) -> None:
        desc = NodePluginDescriptor(
            node_type="custom_shell",
            source=PluginSource.LOCAL,
            schema=NodeTypeSchema(node_type="custom_shell"),
            node_cls=ShellNode,
        )
        assert desc.node_type == "custom_shell"
        assert desc.source == PluginSource.LOCAL
        assert desc.node_cls is ShellNode
        assert desc.server_name is None

    def test_remote_plugin_descriptor(self) -> None:
        desc = NodePluginDescriptor(
            node_type="lint",
            source=PluginSource.REMOTE,
            schema=NodeTypeSchema(node_type="lint"),
            server_name="lint-server",
        )
        assert desc.node_type == "lint"
        assert desc.source == PluginSource.REMOTE
        assert desc.node_cls is None
        assert desc.server_name == "lint-server"

    def test_remote_without_server_name_raises(self) -> None:
        with pytest.raises(ValueError, match="must specify server_name"):
            NodePluginDescriptor(
                node_type="broken",
                source=PluginSource.REMOTE,
                schema=NodeTypeSchema(node_type="broken"),
                # Missing server_name
            )

    def test_local_without_node_cls_raises(self) -> None:
        with pytest.raises(ValueError, match="must specify node_cls"):
            NodePluginDescriptor(
                node_type="broken",
                source=PluginSource.LOCAL,
                schema=NodeTypeSchema(node_type="broken"),
                # Missing node_cls
            )

    def test_builtin_without_node_cls_raises(self) -> None:
        with pytest.raises(ValueError, match="must specify node_cls"):
            NodePluginDescriptor(
                node_type="broken",
                source=PluginSource.BUILTIN,
                schema=NodeTypeSchema(node_type="broken"),
            )

    def test_namespace_entries(self) -> None:
        desc = NodePluginDescriptor(
            node_type="custom",
            source=PluginSource.LOCAL,
            schema=NodeTypeSchema(node_type="custom"),
            node_cls=ShellNode,
            namespace_entries={"custom_helper": lambda: "help"},
        )
        assert "custom_helper" in desc.namespace_entries

    def test_plugin_source_values(self) -> None:
        assert PluginSource.BUILTIN.value == "builtin"
        assert PluginSource.LOCAL.value == "local_plugin"
        assert PluginSource.REMOTE.value == "remote_plugin"


# ---------------------------------------------------------------------------
# Registry extension tests
# ---------------------------------------------------------------------------


def _make_local_descriptor(
    node_type: str = "custom_test",
) -> NodePluginDescriptor:
    """Helper to create a local plugin descriptor."""
    return NodePluginDescriptor(
        node_type=node_type,
        source=PluginSource.LOCAL,
        schema=NodeTypeSchema(
            node_type=node_type,
            description="Test plugin",
            constructor=ConstructorSchema(
                named=[ParamSchema(name="value", type="str", default="")],
            ),
        ),
        node_cls=TopicNode,
    )


def _make_remote_descriptor(
    node_type: str = "remote_lint",
    server_name: str = "lint-server",
) -> NodePluginDescriptor:
    """Helper to create a remote plugin descriptor."""
    return NodePluginDescriptor(
        node_type=node_type,
        source=PluginSource.REMOTE,
        schema=NodeTypeSchema(node_type=node_type),
        server_name=server_name,
    )


class TestRegistryPluginAPI:
    """Extended registry with plugin descriptors."""

    def test_register_local_plugin(self) -> None:
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        assert reg.get("custom_test") is TopicNode
        assert reg.get_descriptor("custom_test") is desc
        assert reg.is_plugin("custom_test")

    def test_register_remote_plugin(self) -> None:
        reg = NodeTypeRegistry()
        desc = _make_remote_descriptor()
        reg.register_plugin(desc)

        # Remote plugins don't add to _types (no Python class)
        assert reg.get("remote_lint") is None
        assert reg.get_descriptor("remote_lint") is desc
        assert reg.is_plugin("remote_lint")
        assert reg.is_remote("remote_lint")
        assert reg.get_server_name("remote_lint") == "lint-server"

    def test_cannot_override_builtin(self) -> None:
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor(node_type="shell")
        with pytest.raises(ValueError, match="Cannot override builtin"):
            reg.register_plugin(desc)

    def test_unregister_plugin(self) -> None:
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        assert reg.unregister_plugin("custom_test") is True
        assert reg.get("custom_test") is None
        assert reg.get_descriptor("custom_test") is None
        assert not reg.is_plugin("custom_test")

    def test_unregister_builtin_fails(self) -> None:
        reg = NodeTypeRegistry()
        assert reg.unregister_plugin("shell") is False

    def test_unregister_nonexistent(self) -> None:
        reg = NodeTypeRegistry()
        assert reg.unregister_plugin("nonexistent") is False

    def test_list_plugins_empty_initially(self) -> None:
        reg = NodeTypeRegistry()
        # Builtins don't have descriptors
        assert reg.list_plugins() == []

    def test_list_plugins_with_registered(self) -> None:
        reg = NodeTypeRegistry()
        desc1 = _make_local_descriptor("plugin_a")
        desc2 = _make_remote_descriptor("plugin_b", "server-b")
        reg.register_plugin(desc1)
        reg.register_plugin(desc2)

        plugins = reg.list_plugins()
        assert len(plugins) == 2
        types = {d.node_type for d in plugins}
        assert types == {"plugin_a", "plugin_b"}

    def test_is_remote_for_local(self) -> None:
        reg = NodeTypeRegistry()
        reg.register_plugin(_make_local_descriptor())
        assert not reg.is_remote("custom_test")

    def test_is_remote_for_builtin(self) -> None:
        reg = NodeTypeRegistry()
        assert not reg.is_remote("shell")

    def test_get_server_name_for_local(self) -> None:
        reg = NodeTypeRegistry()
        reg.register_plugin(_make_local_descriptor())
        assert reg.get_server_name("custom_test") is None

    def test_get_server_name_nonexistent(self) -> None:
        reg = NodeTypeRegistry()
        assert reg.get_server_name("nonexistent") is None

    def test_get_descriptor_for_builtin(self) -> None:
        """Builtins registered before descriptor system have no descriptor."""
        reg = NodeTypeRegistry()
        assert reg.get_descriptor("shell") is None

    def test_existing_api_still_works(self) -> None:
        """Old register/unregister/get API is preserved."""
        reg = NodeTypeRegistry()

        # Existing API
        assert reg.get("shell") is ShellNode
        assert reg.is_builtin("shell")
        assert len(reg.list_types()) == 19  # all builtins (including MessageSegmentNode)

        # Old register still works
        reg.register("custom_old", TopicNode)
        assert reg.get("custom_old") is TopicNode

        # Old unregister still works
        assert reg.unregister("custom_old") is True

    def test_plugin_and_old_api_coexist(self) -> None:
        """Plugin API and old API don't interfere."""
        reg = NodeTypeRegistry()

        # Register via old API (no descriptor)
        reg.register("old_style", TopicNode)
        assert not reg.is_plugin("old_style")
        assert reg.get("old_style") is TopicNode

        # Register via plugin API (has descriptor)
        reg.register_plugin(_make_local_descriptor("new_style"))
        assert reg.is_plugin("new_style")
        assert reg.get("new_style") is TopicNode

    def test_unregister_remote_plugin(self) -> None:
        reg = NodeTypeRegistry()
        desc = _make_remote_descriptor()
        reg.register_plugin(desc)

        assert reg.unregister_plugin("remote_lint") is True
        assert not reg.is_plugin("remote_lint")
        assert not reg.is_remote("remote_lint")


# ---------------------------------------------------------------------------
# Plugin lifecycle management tests
# ---------------------------------------------------------------------------


class TestPluginLifecycleManagement:
    """Tests for unload_plugin, reload_plugin, and get_plugin_info."""

    def test_unload_plugin_local(self) -> None:
        """Test unloading a local plugin."""
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        assert reg.unload_plugin("custom_test", has_active_nodes=False) is True
        assert reg.get("custom_test") is None
        assert reg.get_descriptor("custom_test") is None

    def test_unload_plugin_with_active_nodes_without_force(self) -> None:
        """Test that unloading fails when active nodes exist without force."""
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        with pytest.raises(RuntimeError, match="active nodes exist"):
            reg.unload_plugin("custom_test", has_active_nodes=True, force=False)

    def test_unload_plugin_with_active_nodes_with_force(self) -> None:
        """Test that unloading succeeds with force even when active nodes exist."""
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        assert reg.unload_plugin("custom_test", has_active_nodes=True, force=True) is True
        assert reg.get("custom_test") is None

    def test_unload_builtin_raises(self) -> None:
        """Test that unloading a builtin type raises ValueError."""
        reg = NodeTypeRegistry()

        with pytest.raises(ValueError, match="Cannot unload builtin"):
            reg.unload_plugin("shell")

    def test_unload_nonexistent_returns_false(self) -> None:
        """Test that unloading nonexistent plugin returns False."""
        reg = NodeTypeRegistry()

        assert reg.unload_plugin("nonexistent", has_active_nodes=False) is False

    def test_reload_plugin_local(self) -> None:
        """Test reloading a local plugin."""
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        # Reload should succeed (though may not actually change anything in test)
        assert reg.reload_plugin("custom_test") is True
        assert reg.get("custom_test") is not None

    def test_reload_plugin_remote(self) -> None:
        """Test reloading a remote plugin (should verify still registered)."""
        reg = NodeTypeRegistry()
        desc = _make_remote_descriptor()
        reg.register_plugin(desc)

        # Remote plugins can't really reload, but should return True if registered
        assert reg.reload_plugin("remote_lint") is True

    def test_reload_builtin_raises(self) -> None:
        """Test that reloading a builtin type raises ValueError."""
        reg = NodeTypeRegistry()

        with pytest.raises(ValueError, match="Cannot reload builtin"):
            reg.reload_plugin("shell")

    def test_reload_nonexistent_returns_false(self) -> None:
        """Test that reloading nonexistent plugin returns False."""
        reg = NodeTypeRegistry()

        assert reg.reload_plugin("nonexistent") is False

    def test_get_plugin_info_builtin(self) -> None:
        """Test getting info for a builtin type."""
        reg = NodeTypeRegistry()

        info = reg.get_plugin_info("shell")
        assert info is not None
        assert info.node_type == "shell"
        assert info.is_builtin is True
        assert info.is_loaded is True

    def test_get_plugin_info_local_plugin(self) -> None:
        """Test getting info for a local plugin."""
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        info = reg.get_plugin_info("custom_test")
        assert info is not None
        assert info.node_type == "custom_test"
        assert info.name == "custom_test"
        assert info.description == "Test plugin"
        assert info.is_builtin is False
        assert info.is_loaded is True

    def test_get_plugin_info_remote_plugin(self) -> None:
        """Test getting info for a remote plugin."""
        reg = NodeTypeRegistry()
        desc = _make_remote_descriptor()
        reg.register_plugin(desc)

        info = reg.get_plugin_info("remote_lint")
        assert info is not None
        assert info.node_type == "remote_lint"
        assert info.is_builtin is False
        # Remote plugins aren't "loaded" in the Python sense
        assert info.is_loaded is False

    def test_get_plugin_info_nonexistent(self) -> None:
        """Test getting info for nonexistent plugin returns None."""
        reg = NodeTypeRegistry()

        assert reg.get_plugin_info("nonexistent") is None

    def test_get_plugin_info_after_unload(self) -> None:
        """Test that get_plugin_info returns None after unload."""
        reg = NodeTypeRegistry()
        desc = _make_local_descriptor()
        reg.register_plugin(desc)

        reg.unload_plugin("custom_test", has_active_nodes=False)
        assert reg.get_plugin_info("custom_test") is None


# ---------------------------------------------------------------------------
# Plugin loading from filesystem tests
# ---------------------------------------------------------------------------


class TestPluginLoading:
    """Tests for load_from_module, load_from_directory, and discover_plugins."""

    def test_load_from_module_with_valid_plugin(self, tmp_path):
        """Test loading a valid plugin module."""
        # Create a test plugin file
        plugin_code = '''
from activecontext.context.nodes import ContextNode
from dataclasses import dataclass

@dataclass
class CustomTestNode(ContextNode):
    """A custom test node."""
    value: str = ""

    @property
    def node_type(self) -> str:
        return "custom_test"

    @classmethod
    def _from_dict(cls, data):
        return cls(value=data.get("value", ""))

    def to_dict(self):
        return {"node_type": self.node_type, "value": self.value}

    def GetDigest(self):
        return {"type": self.node_type}

plugin_info = {
    "node_type": "custom_test",
    "name": "Custom Test Node",
    "description": "A test node for plugin loading",
    "version": "1.0.0",
    "author": "Test Author"
}
'''
        plugin_file = tmp_path / "custom_plugin.py"
        plugin_file.write_text(plugin_code)

        # Add tmp_path to sys.path for import
        import sys

        sys.path.insert(0, str(tmp_path))

        try:
            reg = NodeTypeRegistry()
            node_type = reg.load_from_module("custom_plugin")

            assert node_type == "custom_test"
            assert reg.get("custom_test") is not None
            assert reg.is_builtin("custom_test") is False
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_from_module_missing_module(self):
        """Test loading from non-existent module raises ImportError."""
        reg = NodeTypeRegistry()

        with pytest.raises(ImportError, match="Failed to import"):
            reg.load_from_module("nonexistent_module_xyz")

    def test_load_from_module_no_contextnode_subclass(self, tmp_path):
        """Test loading module without ContextNode subclass raises ValueError."""
        plugin_code = """
# No ContextNode subclass here
def some_function():
    pass
"""
        plugin_file = tmp_path / "invalid_plugin.py"
        plugin_file.write_text(plugin_code)

        import sys

        sys.path.insert(0, str(tmp_path))

        try:
            reg = NodeTypeRegistry()

            with pytest.raises(ValueError, match="No ContextNode subclass found"):
                reg.load_from_module("invalid_plugin")
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_from_module_multiple_classes(self, tmp_path):
        """Test loading module with multiple ContextNode subclasses raises ValueError."""
        plugin_code = """
from activecontext.context.nodes import ContextNode
from dataclasses import dataclass

@dataclass
class FirstNode(ContextNode):
    pass

@dataclass
class SecondNode(ContextNode):
    pass
"""
        plugin_file = tmp_path / "multi_plugin.py"
        plugin_file.write_text(plugin_code)

        import sys

        sys.path.insert(0, str(tmp_path))

        try:
            reg = NodeTypeRegistry()

            with pytest.raises(ValueError, match="Multiple ContextNode subclasses"):
                reg.load_from_module("multi_plugin")
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_from_module_duplicate_node_type(self, tmp_path):
        """Test loading module with duplicate node_type raises ValueError."""
        # First, register a plugin with node_type "duplicate"
        from activecontext.context.nodes import TopicNode

        reg = NodeTypeRegistry()
        reg.register("duplicate", TopicNode)

        # Now try to load another with same node_type
        plugin_code = """
from activecontext.context.nodes import ContextNode
from dataclasses import dataclass

@dataclass
class DuplicateNode(ContextNode):
    @property
    def node_type(self) -> str:
        return "duplicate"

    @classmethod
    def _from_dict(cls, data):
        return cls()

    def to_dict(self):
        return {"node_type": "duplicate"}

    def GetDigest(self):
        return {"type": "duplicate"}

plugin_info = {"node_type": "duplicate"}
"""
        plugin_file = tmp_path / "dup_plugin.py"
        plugin_file.write_text(plugin_code)

        import sys

        sys.path.insert(0, str(tmp_path))

        try:
            with pytest.raises(ValueError, match="already registered"):
                reg.load_from_module("dup_plugin")
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_from_module_conflicts_with_builtin(self, tmp_path):
        """Test loading module with builtin node_type raises ValueError."""
        plugin_code = """
from activecontext.context.nodes import ContextNode
from dataclasses import dataclass

@dataclass
class ShellClone(ContextNode):
    @property
    def node_type(self) -> str:
        return "shell"  # Builtin type

    @classmethod
    def _from_dict(cls, data):
        return cls()

    def to_dict(self):
        return {"node_type": "shell"}

    def GetDigest(self):
        return {"type": "shell"}

plugin_info = {"node_type": "shell"}
"""
        plugin_file = tmp_path / "builtin_conflict.py"
        plugin_file.write_text(plugin_code)

        import sys

        sys.path.insert(0, str(tmp_path))

        try:
            reg = NodeTypeRegistry()

            with pytest.raises(ValueError, match="conflicts with builtin"):
                reg.load_from_module("builtin_conflict")
        finally:
            sys.path.remove(str(tmp_path))

    def test_load_from_directory_empty(self, tmp_path):
        """Test loading from empty directory returns empty list."""
        reg = NodeTypeRegistry()
        loaded = reg.load_from_directory(tmp_path)

        assert loaded == []

    def test_load_from_directory_with_plugins(self, tmp_path):
        """Test loading multiple plugins from directory."""
        # Create two valid plugins
        plugin1 = """
from activecontext.context.nodes import ContextNode
from dataclasses import dataclass

@dataclass
class Plugin1Node(ContextNode):
    @property
    def node_type(self) -> str:
        return "plugin1"

    @classmethod
    def _from_dict(cls, data):
        return cls()

    def to_dict(self):
        return {"node_type": "plugin1"}

    def GetDigest(self):
        return {"type": "plugin1"}
"""

        plugin2 = """
from activecontext.context.nodes import ContextNode
from dataclasses import dataclass

@dataclass
class Plugin2Node(ContextNode):
    @property
    def node_type(self) -> str:
        return "plugin2"

    @classmethod
    def _from_dict(cls, data):
        return cls()

    def to_dict(self):
        return {"node_type": "plugin2"}

    def GetDigest(self):
        return {"type": "plugin2"}
"""

        (tmp_path / "plugin1.py").write_text(plugin1)
        (tmp_path / "plugin2.py").write_text(plugin2)
        # Add __init__.py (should be skipped)
        (tmp_path / "__init__.py").write_text("")

        reg = NodeTypeRegistry()
        loaded = reg.load_from_directory(tmp_path)

        assert len(loaded) == 2
        assert "plugin1" in loaded
        assert "plugin2" in loaded
        assert reg.get("plugin1") is not None
        assert reg.get("plugin2") is not None

    def test_load_from_directory_nonexistent(self):
        """Test loading from non-existent directory returns empty list."""
        from pathlib import Path

        reg = NodeTypeRegistry()

        nonexistent = Path("/nonexistent/path/xyz")
        loaded = reg.load_from_directory(nonexistent)

        assert loaded == []

    def test_load_from_directory_not_a_directory(self, tmp_path):
        """Test loading from file (not directory) raises ValueError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        reg = NodeTypeRegistry()

        with pytest.raises(ValueError, match="not a directory"):
            reg.load_from_directory(file_path)

    def test_discover_plugins_empty(self):
        """Test discover_plugins with no plugins returns empty list."""
        reg = NodeTypeRegistry()
        discovered = reg.discover_plugins()

        # Should return empty list if no plugins in standard locations
        assert isinstance(discovered, list)

    def test_discover_plugins_with_metadata(self, tmp_path, monkeypatch):
        """Test discover_plugins extracts metadata from plugin files."""
        # Mock the plugin directory to tmp_path
        monkeypatch.setattr("os.getcwd", lambda: str(tmp_path))

        plugin_dir = tmp_path / ".ac" / "plugins" / "nodes"
        plugin_dir.mkdir(parents=True)

        plugin_code = """
from activecontext.context.nodes import ContextNode

plugin_info = {
    "node_type": "discovered",
    "name": "Discovered Plugin",
    "description": "Test discovery",
    "version": "2.0.0",
    "author": "Discovery Author"
}

class DiscoveredNode(ContextNode):
    pass
"""
        (plugin_dir / "discovered.py").write_text(plugin_code)

        reg = NodeTypeRegistry()
        discovered = reg.discover_plugins()

        assert len(discovered) >= 1
        # Find our plugin
        our_plugin = next((p for p in discovered if p.node_type == "discovered"), None)
        assert our_plugin is not None
        assert our_plugin.name == "Discovered Plugin"
        assert our_plugin.description == "Test discovery"
        assert our_plugin.version == "2.0.0"
        assert our_plugin.author == "Discovery Author"
        assert our_plugin.is_loaded is False
        assert our_plugin.is_builtin is False
