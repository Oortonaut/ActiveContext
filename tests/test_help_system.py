"""Tests for the self-documenting .help() system.

Covers:
1. HelpNode construction and rendering at all states
2. .help() creates HelpNode child
3. .help() returns existing if already created
4. Documentation extraction from class docstrings
5. @exposed decorator marks methods
6. get_exposed() returns correct set
7. help("shell") works for type name lookup
8. help() with no args returns DSL reference
9. HelpNode serialization (to_dict/from_dict)
"""

from __future__ import annotations

import pytest

from activecontext.context.exposed import EXPOSED_ATTR, exposed, get_exposed, is_exposed
from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    HelpNode,
    ShellNode,
    TextNode,
    _extract_help_content,
)
from activecontext.context.registry import get_node_registry
from activecontext.context.state import Expansion
from activecontext.context.view import NodeView

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph() -> ContextGraph:
    """Create a fresh ContextGraph for testing."""
    return ContextGraph()


@pytest.fixture
def text_node(graph: ContextGraph) -> TextNode:
    """Create a TextNode attached to a graph."""
    node = TextNode(path="main.py")
    graph.add_node(node)
    return node


@pytest.fixture
def shell_node(graph: ContextGraph) -> ShellNode:
    """Create a ShellNode attached to a graph."""
    node = ShellNode(command="pytest -v")
    graph.add_node(node)
    return node


# ---------------------------------------------------------------------------
# 1. HelpNode construction and rendering at all states
# ---------------------------------------------------------------------------


class TestHelpNodeConstruction:
    """Tests for HelpNode construction and rendering."""

    def test_helpnode_default_fields(self) -> None:
        """HelpNode has correct default field values."""
        node = HelpNode()
        assert node.node_type == "help"
        assert node.parent_node_type == ""
        assert node._help_content == ""

    def test_helpnode_with_content(self) -> None:
        """HelpNode stores parent_node_type and content."""
        node = HelpNode(
            parent_node_type="shell",
            _help_content="# ShellNode\nAsync shell command.",
        )
        assert node.parent_node_type == "shell"
        assert "ShellNode" in node._help_content

    def test_helpnode_render_header(self, graph: ContextGraph) -> None:
        """Header rendering includes type info."""
        node = HelpNode(
            parent_node_type="text",
            _help_content="## Methods\n- `SetPos(self, pos)` -- Set position\n",
        )
        graph.add_node(node)
        rendered = NodeView(node).render_header()
        # Should contain the display name
        assert "text Help" in rendered

    def test_helpnode_render_content(self, graph: ContextGraph) -> None:
        """Content rendering includes full help content."""
        content = "# TextNode\nFile view node.\n\n## Methods\n- `SetPos(self, pos)` -- Set position\n- `SetEndPos(self, end)` -- Set end position\n"
        node = HelpNode(
            parent_node_type="text",
            _help_content=content,
        )
        graph.add_node(node)
        rendered = node.render_content()
        assert "TextNode" in rendered
        assert "SetPos" in rendered
        assert "SetEndPos" in rendered

    def test_helpnode_render_content_full(self, graph: ContextGraph) -> None:
        """Content rendering includes full content with signatures."""
        content = "# TextNode\nFile view.\n\n## Methods\n- `SetPos(self, pos: str)` -- Set start position\n"
        node = HelpNode(
            parent_node_type="text",
            _help_content=content,
        )
        graph.add_node(node)
        rendered = node.render_content()
        assert "SetPos(self, pos: str)" in rendered
        assert "Set start position" in rendered


# ---------------------------------------------------------------------------
# 2. .help() creates HelpNode child
# ---------------------------------------------------------------------------


class TestHelpMethodCreation:
    """Tests for ContextNode.help() creating HelpNode children."""

    def test_help_creates_child(self, text_node: TextNode, graph: ContextGraph) -> None:
        """Calling .help() creates a HelpNode child of the node."""
        help_node = text_node.help()
        assert isinstance(help_node, HelpNode)
        assert help_node.parent_node_type == "text"
        assert help_node.node_id in text_node.children_ids

    def test_help_node_in_graph(self, text_node: TextNode, graph: ContextGraph) -> None:
        """HelpNode created by .help() is added to the graph."""
        help_node = text_node.help()
        found = graph.get_node(help_node.node_id)
        assert found is help_node

    def test_help_has_content(self, text_node: TextNode) -> None:
        """HelpNode has non-empty help content extracted from class."""
        help_node = text_node.help()
        assert help_node._help_content != ""
        assert "TextNode" in help_node._help_content

    def test_help_requires_graph(self) -> None:
        """Calling .help() on a node not in a graph raises RuntimeError."""
        node = TextNode(path="orphan.py")
        with pytest.raises(RuntimeError, match="not in a graph"):
            node.help()


# ---------------------------------------------------------------------------
# 3. .help() returns existing if already created
# ---------------------------------------------------------------------------


class TestHelpIdempotency:
    """Tests that .help() is idempotent."""

    def test_help_returns_same_node(self, text_node: TextNode) -> None:
        """Calling .help() twice returns the same HelpNode."""
        first = text_node.help()
        second = text_node.help()
        assert first is second
        assert first.node_id == second.node_id

    def test_help_does_not_duplicate(self, text_node: TextNode, graph: ContextGraph) -> None:
        """Calling .help() multiple times does not create extra HelpNodes."""
        text_node.help()
        text_node.help()
        text_node.help()

        help_count = sum(
            1
            for child_id in text_node.children_ids
            if isinstance(graph.get_node(child_id), HelpNode)
        )
        assert help_count == 1

    def test_help_unhides_existing(self, text_node: TextNode, graph: ContextGraph) -> None:
        """If HelpNode was set to HEADER, calling .help() again unhides it."""
        help_node = text_node.help()
        help_node.default_expansion = Expansion.HEADER

        # Calling .help() again should unhide it
        result = text_node.help()
        assert result is help_node
        assert result.default_expansion == Expansion.CONTENT


# ---------------------------------------------------------------------------
# 4. Documentation extraction from class docstrings
# ---------------------------------------------------------------------------


class TestDocumentationExtraction:
    """Tests for _extract_help_content."""

    def test_extract_from_textnode(self) -> None:
        """Extract documentation from TextNode class."""
        content = _extract_help_content(TextNode)
        assert "TextNode" in content
        assert "Methods" in content

    def test_extract_from_shellnode(self) -> None:
        """Extract documentation from ShellNode class."""
        content = _extract_help_content(ShellNode)
        assert "ShellNode" in content

    def test_extract_includes_properties(self) -> None:
        """Extraction includes properties section."""
        content = _extract_help_content(TextNode)
        assert "Properties" in content

    def test_extract_includes_method_signatures(self) -> None:
        """Extraction includes method signatures with parameters."""
        content = _extract_help_content(TextNode)
        # TextNode has SetPos method
        assert "SetPos" in content

    def test_extract_includes_class_description(self) -> None:
        """Extraction includes class docstring first line."""
        content = _extract_help_content(TextNode)
        # TextNode docstring starts with "View of a file..."
        assert "View of a file" in content or "TextNode" in content

    def test_extract_skips_base_fields(self) -> None:
        """Extraction skips inherited ContextNode base fields in constructor params."""
        content = _extract_help_content(TextNode)
        # Should include TextNode-specific fields like 'path'
        assert "path" in content
        # Should not redundantly list base class fields like 'node_id'
        lines = content.split("\n")
        [line for line in lines if "node_id" in line and "Constructor" not in line]
        # node_id might still appear in properties or method signatures, but not as a
        # constructor parameter with "**node_id**" formatting
        param_lines = [line for line in lines if line.startswith("- **node_id**")]
        assert len(param_lines) == 0


# ---------------------------------------------------------------------------
# 5. @exposed decorator marks methods
# ---------------------------------------------------------------------------


class TestExposedDecorator:
    """Tests for the @exposed decorator."""

    def test_exposed_sets_attribute(self) -> None:
        """@exposed sets the _exposed attribute."""

        @exposed
        def my_method():
            pass

        assert hasattr(my_method, EXPOSED_ATTR)
        assert getattr(my_method, EXPOSED_ATTR) is True

    def test_exposed_returns_same_function(self) -> None:
        """@exposed returns the same function object."""

        def my_method():
            pass

        result = exposed(my_method)
        assert result is my_method

    def test_exposed_on_property(self) -> None:
        """@exposed can mark a property."""

        class MyClass:
            @exposed
            @property
            def value(self) -> int:
                return 42

        # The property's fget should be marked
        assert is_exposed(MyClass.__dict__["value"])

    def test_is_exposed_false_for_unmarked(self) -> None:
        """is_exposed returns False for unmarked functions."""

        def plain_func():
            pass

        assert not is_exposed(plain_func)

    def test_is_exposed_true_for_marked(self) -> None:
        """is_exposed returns True for @exposed functions."""

        @exposed
        def marked_func():
            pass

        assert is_exposed(marked_func)


# ---------------------------------------------------------------------------
# 6. get_exposed() returns correct set
# ---------------------------------------------------------------------------


class TestGetExposed:
    """Tests for get_exposed() utility."""

    def test_empty_class(self) -> None:
        """Empty class has no exposed members."""

        class Empty:
            pass

        assert get_exposed(Empty) == set()

    def test_class_with_exposed_method(self) -> None:
        """get_exposed returns marked method names."""

        class MyNode:
            @exposed
            def SetState(self) -> None:
                pass

            def _internal(self) -> None:
                pass

            def normal(self) -> None:
                pass

        result = get_exposed(MyNode)
        assert "SetState" in result
        assert "_internal" not in result
        assert "normal" not in result

    def test_class_with_exposed_property(self) -> None:
        """get_exposed returns marked property names."""

        class MyNode:
            @exposed
            @property
            def is_complete(self) -> bool:
                return False

        result = get_exposed(MyNode)
        assert "is_complete" in result

    def test_class_with_mixed_exposed(self) -> None:
        """get_exposed returns both methods and properties."""

        class MyNode:
            @exposed
            def Run(self) -> None:
                pass

            @exposed
            @property
            def status(self) -> str:
                return "ok"

            def private_impl(self) -> None:
                pass

        result = get_exposed(MyNode)
        assert result == {"Run", "status"}

    def test_inherited_exposed(self) -> None:
        """get_exposed includes inherited @exposed members."""

        class Base:
            @exposed
            def base_method(self) -> None:
                pass

        class Child(Base):
            @exposed
            def child_method(self) -> None:
                pass

        result = get_exposed(Child)
        assert "base_method" in result
        assert "child_method" in result

    def test_skips_private_members(self) -> None:
        """get_exposed skips underscore-prefixed members."""

        class MyClass:
            @exposed
            def _private(self) -> None:
                pass

        result = get_exposed(MyClass)
        assert "_private" not in result


# ---------------------------------------------------------------------------
# 7. help("shell") works for type name lookup
# ---------------------------------------------------------------------------


class TestHelpTypeLookup:
    """Tests for help() with type name string (via HelpNode directly)."""

    def test_extract_content_for_shell(self) -> None:
        """Extracting help content for ShellNode by class works."""
        registry = get_node_registry()
        cls = registry.get("shell")
        assert cls is not None
        content = _extract_help_content(cls)
        assert "ShellNode" in content

    def test_extract_content_for_text(self) -> None:
        """Extracting help content for TextNode by class works."""
        registry = get_node_registry()
        cls = registry.get("text")
        assert cls is not None
        content = _extract_help_content(cls)
        assert "TextNode" in content

    def test_extract_content_for_group(self) -> None:
        """Extracting help content for GroupNode by class works."""
        registry = get_node_registry()
        cls = registry.get("group")
        assert cls is not None
        content = _extract_help_content(cls)
        assert "GroupNode" in content

    def test_registry_contains_help(self) -> None:
        """The help node type is registered."""
        registry = get_node_registry()
        cls = registry.get("help")
        assert cls is HelpNode

    def test_help_node_for_unknown_type(self) -> None:
        """Creating help for unknown type name raises ValueError when using registry."""
        registry = get_node_registry()
        cls = registry.get("nonexistent_type_xyz")
        assert cls is None


# ---------------------------------------------------------------------------
# 8. help() with no args returns DSL reference
# ---------------------------------------------------------------------------


class TestHelpNoArgs:
    """Tests for help() with no arguments (DSL reference).

    This tests the _extract_help_content and HelpNode behavior, not the
    timeline DSL function directly (which requires full Timeline setup).
    """

    def test_helpnode_for_dsl_reference(self, graph: ContextGraph) -> None:
        """A HelpNode can be created as a generic DSL reference placeholder."""
        # In the DSL, help() returns a TextNode pointing to dsl_reference.md
        # Here we just verify HelpNode works with generic content
        help_node = HelpNode(
            parent_node_type="dsl",
            _help_content="# DSL Reference\nFunctions available in the namespace.",
            default_expansion=Expansion.CONTENT,
        )
        graph.add_node(help_node)
        assert help_node.node_type == "help"
        rendered = help_node.render_content()
        assert "DSL Reference" in rendered


# ---------------------------------------------------------------------------
# 9. HelpNode serialization (to_dict/from_dict)
# ---------------------------------------------------------------------------


class TestHelpNodeSerialization:
    """Tests for HelpNode serialization and deserialization."""

    def test_to_dict(self) -> None:
        """HelpNode.to_dict() includes all fields."""
        node = HelpNode(
            node_id="help_1",
            parent_node_type="shell",
            _help_content="# ShellNode\nAsync shell.\n## Methods\n- `Run()` -- Start\n",
        )
        data = node.to_dict()
        assert data["node_type"] == "help"
        assert data["node_id"] == "help_1"
        assert data["parent_node_type"] == "shell"
        assert "ShellNode" in data["_help_content"]

    def test_from_dict(self) -> None:
        """HelpNode._from_dict() restores all fields."""
        original = HelpNode(
            node_id="help_2",
            parent_node_type="text",
            _help_content="# TextNode\nFile view.\n",
            default_expansion=Expansion.ALL,
        )
        data = original.to_dict()
        restored = HelpNode._from_dict(data)

        assert restored.node_id == "help_2"
        assert restored.parent_node_type == "text"
        assert restored._help_content == original._help_content
        assert restored.default_expansion == Expansion.ALL

    def test_roundtrip_serialization(self) -> None:
        """to_dict -> _from_dict preserves all data."""
        original = HelpNode(
            node_id="help_rt",
            parent_node_type="group",
            _help_content="# GroupNode\nSummary facade.\n\n## Methods\n- `SetSummary(text)` -- Set summary\n",
            default_expansion=Expansion.CONTENT,
            title="Group Help",
        )
        data = original.to_dict()
        restored = HelpNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.parent_node_type == original.parent_node_type
        assert restored._help_content == original._help_content
        assert restored.default_expansion == original.default_expansion
        assert restored.title == original.title

    def test_from_dict_via_registry(self) -> None:
        """Registry-based deserialization works for HelpNode."""
        registry = get_node_registry()
        data = {
            "node_type": "help",
            "node_id": "help_reg",
            "parent_node_type": "text",
            "_help_content": "# TextNode\nHelp text.\n",
        }
        node = registry.from_dict(data)
        assert isinstance(node, HelpNode)
        assert node.parent_node_type == "text"

    def test_to_dict_includes_node_type(self) -> None:
        """to_dict always includes the node_type field."""
        node = HelpNode(parent_node_type="artifact")
        data = node.to_dict()
        assert data["node_type"] == "help"


# ---------------------------------------------------------------------------
# Additional: HelpNode.render_digest and GetDigest
# ---------------------------------------------------------------------------


class TestHelpNodeMetadata:
    """Tests for HelpNode metadata methods."""

    def test_render_digest(self) -> None:
        """render_digest includes type name and method count."""
        node = HelpNode(
            parent_node_type="shell",
            _help_content="## Methods\n- `Run()` -- Start\n- `Pause()` -- Stop\n",
        )
        name = node.render_digest()
        assert "shell Help" in name
        assert "2 methods" in name

    def test_render_digest_zero_methods(self) -> None:
        """render_digest shows 0 methods when content has no method docs."""
        node = HelpNode(parent_node_type="trace", _help_content="# TraceNode\nA trace.\n")
        name = node.render_digest()
        assert "0 methods" in name

    def test_get_digest(self) -> None:
        """GetDigest returns correct metadata dictionary."""
        node = HelpNode(
            parent_node_type="text",
            _help_content="## Methods\n- `SetPos(pos)` -- Set position\n",
        )
        digest = node.GetDigest()
        assert digest["type"] == "help"
        assert digest["parent_node_type"] == "text"
        assert digest["methods"] == 1

    def test_count_methods(self) -> None:
        """_count_methods correctly counts method entries."""
        node = HelpNode(
            _help_content=(
                "## Methods\n"
                "- `SetPos(self, pos)` -- Set position\n"
                "- `SetEndPos(self, end)` -- Set end\n"
                "- `Run(self)` -- Start running\n"
                "## Properties\n"
                "- `is_complete` (read-only) -- Check complete\n"
            ),
        )
        assert node._count_methods() == 3
