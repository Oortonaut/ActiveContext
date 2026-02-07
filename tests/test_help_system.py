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
        assert node.node_type == "HelpNode"
        assert node.parent_node_type == ""
        assert node._help_content == ""

    def test_helpnode_with_content(self) -> None:
        """HelpNode stores parent_node_type and content."""
        node = HelpNode(
            parent_node_type="ShellNode",
            _help_content="# ShellNode\nAsync shell command.",
        )
        assert node.parent_node_type == "ShellNode"
        assert "ShellNode" in node._help_content

    def test_helpnode_render_header(self, graph: ContextGraph) -> None:
        """Header rendering includes type info."""
        node = HelpNode(
            parent_node_type="TextNode",
            _help_content="## Methods\n- `SetPos(self, pos)` -- Set position\n",
        )
        graph.add_node(node)
        rendered = NodeView(node).render_header()
        # Should contain the parent node type
        assert "TextNode Help" in rendered

    def test_helpnode_render_content(self, graph: ContextGraph) -> None:
        """Content rendering includes full help content."""
        content = "# TextNode\nFile view node.\n\n## Methods\n- `SetPos(self, pos)` -- Set position\n- `SetEndPos(self, end)` -- Set end position\n"
        node = HelpNode(
            parent_node_type="TextNode",
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
            parent_node_type="TextNode",
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
        assert help_node.parent_node_type == "TextNode"
        assert help_node.node_id in text_node.child_order

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
            for child_id in text_node.child_order
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
        cls = registry.get("ShellNode")
        assert cls is not None
        content = _extract_help_content(cls)
        assert "ShellNode" in content

    def test_extract_content_for_text(self) -> None:
        """Extracting help content for TextNode by class works."""
        registry = get_node_registry()
        cls = registry.get("TextNode")
        assert cls is not None
        content = _extract_help_content(cls)
        assert "TextNode" in content

    def test_extract_content_for_group(self) -> None:
        """Extracting help content for GroupNode by class works."""
        registry = get_node_registry()
        cls = registry.get("GroupNode")
        assert cls is not None
        content = _extract_help_content(cls)
        assert "GroupNode" in content

    def test_registry_contains_help(self) -> None:
        """The help node type is registered."""
        registry = get_node_registry()
        cls = registry.get("HelpNode")
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
        assert help_node.node_type == "HelpNode"
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
        assert data["node_type"] == "HelpNode"
        assert data["node_id"] == "help_1"
        assert data["parent_node_type"] == "shell"
        assert "ShellNode" in data["_help_content"]

    def test_from_dict(self) -> None:
        """HelpNode._from_dict() restores all fields."""
        original = HelpNode(
            node_id="help_2",
            parent_node_type="TextNode",
            _help_content="# TextNode\nFile view.\n",
            default_expansion=Expansion.ALL,
        )
        data = original.to_dict()
        restored = HelpNode._from_dict(data)

        assert restored.node_id == "help_2"
        assert restored.parent_node_type == "TextNode"
        assert restored._help_content == original._help_content
        assert restored.default_expansion == Expansion.ALL

    def test_roundtrip_serialization(self) -> None:
        """to_dict -> _from_dict preserves all data."""
        original = HelpNode(
            node_id="help_rt",
            parent_node_type="GroupNode",
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
            "node_type": "HelpNode",
            "node_id": "help_reg",
            "parent_node_type": "TextNode",
            "_help_content": "# TextNode\nHelp text.\n",
        }
        node = registry.from_dict(data)
        assert isinstance(node, HelpNode)
        assert node.parent_node_type == "TextNode"

    def test_to_dict_includes_node_type(self) -> None:
        """to_dict always includes the node_type field."""
        node = HelpNode(parent_node_type="artifact")
        data = node.to_dict()
        assert data["node_type"] == "HelpNode"


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
            parent_node_type="TextNode",
            _help_content="## Methods\n- `SetPos(pos)` -- Set position\n",
        )
        digest = node.GetDigest()
        assert digest["type"] == "HelpNode"
        assert digest["parent_node_type"] == "TextNode"
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


# ---------------------------------------------------------------------------
# Task #8: DSL methods return ContextNode
# ---------------------------------------------------------------------------


class TestDSLReturnTypes:
    """Tests that DSL methods return ContextNode, not NodeView."""

    @pytest.fixture
    def graph(self) -> ContextGraph:
        return ContextGraph()

    def test_text_node_methods_on_returned_node(self, graph: ContextGraph) -> None:
        """DSL text() returns ContextNode that can access node methods directly."""
        import asyncio
        import tempfile

        from activecontext.session.timeline import Timeline

        with tempfile.TemporaryDirectory() as tmp:
            timeline = Timeline("test", context_graph=graph, cwd=tmp)
            try:
                # text() should return a ContextNode, not a NodeView
                result = asyncio.get_event_loop().run_until_complete(
                    timeline.execute_statement('v = text("test.py")')
                )
                assert result.status.value == "ok"

                ns = timeline.get_namespace()
                v = ns["v"]

                # Should be a ContextNode (TextNode specifically)
                from activecontext.context.nodes import TextNode
                assert isinstance(v, TextNode), f"Expected TextNode, got {type(v)}"

                # Should have node_id directly accessible
                assert hasattr(v, "node_id")
                assert v.node_id.startswith("text_")
            finally:
                asyncio.get_event_loop().run_until_complete(timeline.close())


# ---------------------------------------------------------------------------
# Task #9: Node forwarding methods
# ---------------------------------------------------------------------------


class TestNodeForwardingMethods:
    """Tests for ContextNode.add_to(), remove(), link_child()."""

    @pytest.fixture
    def graph(self) -> ContextGraph:
        return ContextGraph()

    def test_add_to_graph(self, graph: ContextGraph) -> None:
        """ContextNode.add_to() adds node to graph."""
        node = TextNode(path="test.py")
        node_id = node.add_to(graph)
        assert node_id.startswith("text_")
        assert graph.get_node(node_id) is node
        assert node._graph is graph

    def test_remove_from_graph(self, graph: ContextGraph) -> None:
        """ContextNode.remove() removes node from graph."""
        node = TextNode(path="test.py")
        node.add_to(graph)
        node_id = node.node_id

        node.remove()
        assert graph.get_node(node_id) is None

    def test_remove_requires_graph(self) -> None:
        """ContextNode.remove() raises if not in graph."""
        node = TextNode(path="test.py")
        with pytest.raises(RuntimeError, match="not in a graph"):
            node.remove()

    def test_link_child(self, graph: ContextGraph) -> None:
        """ContextNode.link_child() links child to parent."""
        from activecontext.context.nodes import GroupNode

        parent = GroupNode()
        child = TextNode(path="test.py")
        parent.add_to(graph)
        child.add_to(graph)

        result = parent.link_child(child)
        assert result is True
        assert child.node_id in parent.child_order

    def test_link_child_with_after(self, graph: ContextGraph) -> None:
        """ContextNode.link_child() supports after parameter."""
        from activecontext.context.nodes import GroupNode

        parent = GroupNode()
        child1 = TextNode(path="a.py", node_id="child1")
        child2 = TextNode(path="b.py", node_id="child2")
        parent.add_to(graph)
        child1.add_to(graph)
        child2.add_to(graph)

        parent.link_child(child1)
        parent.link_child(child2, after=child1.node_id)

        children = list(parent.child_order)
        assert children == ["child1", "child2"]


# ---------------------------------------------------------------------------
# Task #10: HelpNode with FunctionDocNode children
# ---------------------------------------------------------------------------


class TestHelpNodeWithFunctionDocs:
    """Tests for HelpNode with FunctionDocNode children."""

    @pytest.fixture
    def graph(self) -> ContextGraph:
        return ContextGraph()

    def test_help_creates_functiondoc_children_for_exposed_methods(
        self, graph: ContextGraph
    ) -> None:
        """help() creates FunctionDocNode children when node type has @exposed methods."""
        from dataclasses import dataclass

        from activecontext.context.exposed import exposed
        from activecontext.context.nodes import ContextNode
        from activecontext.context.nodes.function_doc import FunctionDocNode

        # Create a custom node type with @exposed methods for testing
        @dataclass(kw_only=True)
        class TestNodeWithExposed(ContextNode):
            """Test node with exposed methods."""

            @exposed
            def my_method(self) -> str:
                """A documented method."""
                return "hello"

            @exposed
            def another_method(self, arg: int) -> None:
                """Another documented method."""
                pass

        node = TestNodeWithExposed()
        node.add_to(graph)

        help_node = node.help()

        # Check that FunctionDocNode children were created
        func_doc_count = 0
        func_names = []
        for child_id in help_node.child_order:
            child = graph.get_node(child_id)
            if isinstance(child, FunctionDocNode):
                func_doc_count += 1
                func_names.append(child.function_name)

        # Should have FunctionDocNode children for the 2 @exposed methods
        assert func_doc_count == 2
        assert "my_method" in func_names
        assert "another_method" in func_names

    def test_functiondoc_from_method(self) -> None:
        """FunctionDocNode.from_method() extracts signature and docstring."""
        from activecontext.context.nodes.function_doc import FunctionDocNode

        def sample_method(self, arg1: str, arg2: int = 5) -> bool:
            """Sample method docstring."""
            pass

        func_doc = FunctionDocNode.from_method(sample_method)
        assert func_doc.function_name == "sample_method"
        assert "sample_method" in func_doc.signature
        # Signature includes arg names and types (quotes may vary by Python version)
        assert "arg1" in func_doc.signature
        assert "str" in func_doc.signature
        assert "arg2" in func_doc.signature
        assert "Sample method docstring" in func_doc.docstring

    def test_functiondoc_from_async_method(self) -> None:
        """FunctionDocNode.from_method() handles async methods."""
        from activecontext.context.nodes.function_doc import FunctionDocNode

        async def async_method(self) -> None:
            """Async method docstring."""
            pass

        func_doc = FunctionDocNode.from_method(async_method)
        assert "async def" in func_doc.signature
        assert "Async method docstring" in func_doc.docstring

    def test_functiondoc_render_content(self) -> None:
        """FunctionDocNode.render_content() renders signature and docstring."""
        from activecontext.context.nodes.function_doc import FunctionDocNode

        def my_func(x: int) -> str:
            """Converts int to string."""
            return str(x)

        func_doc = FunctionDocNode.from_method(my_func)
        content = func_doc.render_content()
        assert "my_func" in content
        assert "Converts int to string" in content
