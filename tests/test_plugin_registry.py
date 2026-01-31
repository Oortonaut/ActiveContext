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
        assert len(reg.list_types()) == 15  # all builtins

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
