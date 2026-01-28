"""Tests for NodeTypeRegistry."""

import pytest

from activecontext.context.nodes import (
    ContextNode,
    GroupNode,
    TextNode,
)
from activecontext.context.registry import NodeTypeRegistry, get_node_registry


class TestNodeTypeRegistryInit:
    """Tests for registry initialization."""

    def test_init_loads_builtin_types(self) -> None:
        """Test that __init__ loads all builtin types."""
        registry = NodeTypeRegistry()
        # Should have all the builtin types
        assert registry.get("text") is not None
        assert registry.get("group") is not None
        assert registry.get("topic") is not None
        assert registry.get("artifact") is not None
        assert registry.get("shell") is not None
        assert registry.get("lock") is not None
        assert registry.get("session") is not None
        assert registry.get("message") is not None
        assert registry.get("work") is not None
        assert registry.get("mcp_server") is not None
        assert registry.get("mcp_tool") is not None
        assert registry.get("mcp_manager") is not None
        assert registry.get("agent") is not None
        assert registry.get("trace") is not None
        assert registry.get("task") is not None

    def test_init_marks_builtins(self) -> None:
        """Test that builtin types are marked as builtin."""
        registry = NodeTypeRegistry()
        assert registry.is_builtin("text")
        assert registry.is_builtin("group")
        assert registry.is_builtin("session")


class TestNodeTypeRegistryGet:
    """Tests for get() method."""

    def test_get_returns_class(self) -> None:
        """Test that get() returns the correct class."""
        registry = NodeTypeRegistry()
        assert registry.get("text") is TextNode
        assert registry.get("group") is GroupNode

    def test_get_returns_none_for_unknown(self) -> None:
        """Test that get() returns None for unknown types."""
        registry = NodeTypeRegistry()
        assert registry.get("nonexistent") is None


class TestNodeTypeRegistryRegister:
    """Tests for register() method."""

    def test_register_new_type(self) -> None:
        """Test registering a new plugin type."""
        registry = NodeTypeRegistry()

        # Create a simple mock class that pretends to be a ContextNode subclass
        class MockNode(TextNode):
            @property
            def node_type(self) -> str:
                return "custom_mock"

        registry.register("custom_mock", MockNode)
        assert registry.get("custom_mock") is MockNode

    def test_register_fails_for_builtin(self) -> None:
        """Test that registering over a builtin type raises ValueError."""
        registry = NodeTypeRegistry()
        with pytest.raises(ValueError, match="Cannot override builtin node type"):
            registry.register("text", TextNode)


class TestNodeTypeRegistryUnregister:
    """Tests for unregister() method."""

    def test_unregister_plugin_type(self) -> None:
        """Test unregistering a plugin type."""
        registry = NodeTypeRegistry()

        class MockNode(TextNode):
            pass

        registry.register("custom_mock", MockNode)
        assert registry.get("custom_mock") is MockNode

        result = registry.unregister("custom_mock")
        assert result is True
        assert registry.get("custom_mock") is None

    def test_unregister_builtin_returns_false(self) -> None:
        """Test that unregistering a builtin type returns False."""
        registry = NodeTypeRegistry()
        result = registry.unregister("text")
        assert result is False
        # Builtin should still exist
        assert registry.get("text") is not None

    def test_unregister_nonexistent_returns_false(self) -> None:
        """Test that unregistering nonexistent type returns False."""
        registry = NodeTypeRegistry()
        result = registry.unregister("nonexistent")
        assert result is False


class TestNodeTypeRegistryListTypes:
    """Tests for list_types() method."""

    def test_list_types_returns_all(self) -> None:
        """Test that list_types() returns all registered types."""
        registry = NodeTypeRegistry()
        types = registry.list_types()
        # Should have at least the 15 builtin types
        assert len(types) >= 15
        # Should be tuples of (node_type, class)
        type_names = [t[0] for t in types]
        assert "text" in type_names
        assert "group" in type_names


class TestNodeTypeRegistryIsBuiltin:
    """Tests for is_builtin() method."""

    def test_is_builtin_for_builtin(self) -> None:
        """Test is_builtin returns True for builtin types."""
        registry = NodeTypeRegistry()
        assert registry.is_builtin("text") is True
        assert registry.is_builtin("shell") is True

    def test_is_builtin_for_plugin(self) -> None:
        """Test is_builtin returns False for plugin types."""
        registry = NodeTypeRegistry()

        class MockNode(TextNode):
            pass

        registry.register("custom_mock", MockNode)
        assert registry.is_builtin("custom_mock") is False

    def test_is_builtin_for_nonexistent(self) -> None:
        """Test is_builtin returns False for nonexistent types."""
        registry = NodeTypeRegistry()
        assert registry.is_builtin("nonexistent") is False


class TestNodeTypeRegistryFromDict:
    """Tests for from_dict() method."""

    def test_from_dict_creates_text_node(self) -> None:
        """Test that from_dict correctly deserializes a TextNode."""
        registry = NodeTypeRegistry()
        data = {
            "node_type": "text",
            "node_id": "test_text",
            "path": "test.py",
            "pos": "1:0",
            "tokens": 1000,
        }
        node = registry.from_dict(data)
        assert isinstance(node, TextNode)
        assert node.node_id == "test_text"
        assert node.path == "test.py"

    def test_from_dict_creates_group_node(self) -> None:
        """Test that from_dict correctly deserializes a GroupNode."""
        registry = NodeTypeRegistry()
        data = {
            "node_type": "group",
            "node_id": "test_group",
            "name": "Test Group",
            "tokens": 500,
        }
        node = registry.from_dict(data)
        assert isinstance(node, GroupNode)
        assert node.node_id == "test_group"

    def test_from_dict_raises_for_unknown(self) -> None:
        """Test that from_dict raises ValueError for unknown types."""
        registry = NodeTypeRegistry()
        data = {"node_type": "nonexistent", "node_id": "test"}
        with pytest.raises(ValueError, match="Unknown node type"):
            registry.from_dict(data)


class TestGetNodeRegistry:
    """Tests for get_node_registry() singleton function."""

    def test_returns_singleton(self) -> None:
        """Test that get_node_registry returns the same instance."""
        registry1 = get_node_registry()
        registry2 = get_node_registry()
        assert registry1 is registry2

    def test_returns_functional_registry(self) -> None:
        """Test that the singleton registry is functional."""
        registry = get_node_registry()
        assert registry.get("text") is TextNode


class TestContextNodeFromDictIntegration:
    """Integration tests for ContextNode.from_dict using registry."""

    def test_from_dict_uses_registry(self) -> None:
        """Test that ContextNode.from_dict properly delegates to registry."""
        data = {
            "node_type": "text",
            "node_id": "integration_test",
            "path": "test.py",
        }
        node = ContextNode.from_dict(data)
        assert isinstance(node, TextNode)
        assert node.node_id == "integration_test"

    def test_roundtrip_serialization(self) -> None:
        """Test that nodes can be serialized and deserialized correctly."""
        original = TextNode(
            node_id="roundtrip_test",
            path="example.py",
            pos="10:5",
        )
        data = original.to_dict()
        restored = ContextNode.from_dict(data)

        assert isinstance(restored, TextNode)
        assert restored.node_id == original.node_id
        assert restored.path == original.path
        assert restored.pos == original.pos
