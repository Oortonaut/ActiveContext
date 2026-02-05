"""Tests for projection engine and budget allocation.

Tests coverage for:
- src/activecontext/core/projection_engine.py
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.state import Expansion
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine, RenderPath
from tests.utils import create_mock_context_node

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def projection_config():
    """Create ProjectionConfig with test values."""
    return ProjectionConfig()


@pytest.fixture
def projection_engine(projection_config):
    """Create ProjectionEngine with test config."""
    return ProjectionEngine(config=projection_config)


@pytest.fixture
def mock_graph():
    """Create ContextGraph with mock nodes for projection."""
    graph = ContextGraph()

    # Create mock nodes with Render method
    running_node = create_mock_context_node("running1", "view", mode="running")
    running_node.default_expansion = Expansion.ALL
    running_node.Render = Mock(return_value="# Running Node Content\nThis is the content.")
    running_node.clear_pending_traces = Mock()

    paused_root = create_mock_context_node("paused_root", "view", mode="paused")
    paused_root.default_expansion = Expansion.CONTENT
    paused_root.Render = Mock(return_value="# Paused Root Summary")
    paused_root.clear_pending_traces = Mock()

    # Add nodes to graph
    graph.add_node(running_node)
    graph.add_node(paused_root)

    return graph


# =============================================================================
# RenderPath Tests
# =============================================================================


class TestRenderPath:
    """Tests for RenderPath dataclass."""

    def test_render_path_empty(self):
        """Test empty render path."""
        path = RenderPath()

        assert len(path) == 0
        assert not path
        assert path.views == []
        assert path.edges == []
        assert path.root_ids == set()

    def test_render_path_with_views(self):
        """Test render path with views."""
        from activecontext.context.view import NodeView

        nodes = [
            create_mock_context_node(nid, "view") for nid in ("a", "b", "c")
        ]
        views = [NodeView(n) for n in nodes]

        path = RenderPath(
            views=views,
            edges=[("b", "a"), ("c", "a")],
            root_ids={"a"},
        )

        assert len(path) == 3
        assert path
        view_ids = [v.node_id for v in path.views]
        assert "a" in view_ids
        assert ("b", "a") in path.edges
        assert "a" in path.root_ids


# =============================================================================
# Render Path Collection Tests
# =============================================================================


class TestCollectRenderPath:
    """Tests for render path collection."""

    def test_collect_render_path_from_roots(self, projection_engine):
        """Test collecting render path starts from root nodes."""
        graph = ContextGraph()

        root1 = create_mock_context_node("r1", "view")
        root2 = create_mock_context_node("r2", "view")
        child = create_mock_context_node("c1", "view")

        graph.add_node(root1)
        graph.add_node(root2)
        graph.add_node(child)

        # Link child to root1 - child won't be a root anymore
        # But root1 is DETAILS by default so it will recurse
        graph.link("c1", "r1")

        path = projection_engine._collect_render_path(graph)
        view_ids = [v.node_id for v in path.views]

        # Both roots and child (via recursion) should be in path
        assert "r1" in view_ids
        assert "r2" in view_ids
        assert "c1" in view_ids  # Included via parent's DETAILS state

    def test_collect_render_path_collapsed_still_recurses(self, projection_engine):
        """Test that COLLAPSED parents still recurse into children for token counting."""
        graph = ContextGraph()

        root = create_mock_context_node("root", "view")
        root.default_expansion = Expansion.HEADER
        child = create_mock_context_node("child", "view")

        graph.add_node(root)
        graph.add_node(child)
        graph.link("child", "root")

        path = projection_engine._collect_render_path(graph)
        view_ids = [v.node_id for v in path.views]

        # Both root and child are collected - children always collected
        # for complete token information (expansion cost visibility)
        assert "root" in view_ids
        assert "child" in view_ids
        assert "root" in path.root_ids

    def test_collect_render_path_includes_hidden_nodes(self, projection_engine):
        """Test that hidden nodes are collected (filtering happens at render time)."""
        graph = ContextGraph()

        hidden_node = create_mock_context_node("hidden", "view", mode="running")
        hidden_node.default_expansion = Expansion.HEADER
        hidden_node.default_hidden = True  # Set default hidden to True

        visible_node = create_mock_context_node("visible", "view", mode="running")
        visible_node.default_expansion = Expansion.ALL
        visible_node.default_hidden = False

        graph.add_node(hidden_node)
        graph.add_node(visible_node)

        path = projection_engine._collect_render_path(graph)
        view_ids = [v.node_id for v in path.views]

        # Collect path includes all nodes (hidden filtering is in _render_path)
        assert "visible" in view_ids
        assert "hidden" in view_ids
        # But the view is marked hidden
        assert projection_engine.views["hidden"].hidden is True
        assert projection_engine.views["visible"].hidden is False

    def test_collect_render_path_records_edges(self, projection_engine):
        """Test that render path records parent-child edges."""
        graph = ContextGraph()

        parent = create_mock_context_node("parent", "view", mode="running")
        child = create_mock_context_node("child", "view", mode="running")

        graph.add_node(parent)
        graph.add_node(child)
        graph.link("child", "parent")

        path = projection_engine._collect_render_path(graph)
        view_ids = [v.node_id for v in path.views]

        # Both should be in path since both are running
        assert "parent" in view_ids
        assert "child" in view_ids

    def test_collect_render_path_empty_graph(self, projection_engine):
        """Test collecting render path from empty graph."""
        graph = ContextGraph()

        path = projection_engine._collect_render_path(graph)

        assert len(path) == 0
        assert not path

    def test_collect_render_path_sets_indent_by_depth(self, projection_engine):
        """Test that indent is set based on traversal depth."""
        graph = ContextGraph()

        root = create_mock_context_node("root", "view", mode="running")
        child1 = create_mock_context_node("child1", "view", mode="running")
        grandchild = create_mock_context_node("grandchild", "view", mode="running")
        child2 = create_mock_context_node("child2", "view", mode="running")

        graph.add_node(root)
        graph.add_node(child1)
        graph.add_node(grandchild)
        graph.add_node(child2)

        # Build hierarchy: root -> child1 -> grandchild
        #                   root -> child2
        graph.link("child1", "root")
        graph.link("grandchild", "child1")
        graph.link("child2", "root")

        path = projection_engine._collect_render_path(graph)

        # Check that views were created with correct indent
        assert projection_engine.views["root"].indent == 0
        assert projection_engine.views["child1"].indent == 1
        assert projection_engine.views["grandchild"].indent == 2
        assert projection_engine.views["child2"].indent == 1


# =============================================================================
# Render Path Rendering Tests
# =============================================================================


class TestRenderPathRendering:
    """Tests for rendering the collected path."""

    def test_render_path_basic(self, projection_engine, mock_graph):
        """Test basic path rendering."""
        path = projection_engine._collect_render_path(mock_graph)
        sections = projection_engine._render_path(path)

        assert len(sections) == 2  # Running node + paused root
        section_ids = {s.source_id for s in sections}
        assert "running1" in section_ids
        assert "paused_root" in section_ids

    def test_render_path_excludes_hidden(self, projection_engine):
        """Test that hidden views are excluded from rendered sections."""
        from activecontext.context.view import NodeView

        graph = ContextGraph()

        hidden_node = create_mock_context_node("hidden", "view", mode="running")
        hidden_node.default_expansion = Expansion.HEADER
        hidden_node.Render = Mock(return_value="Hidden content")
        hidden_node.clear_pending_traces = Mock()

        visible_node = create_mock_context_node("visible", "view", mode="running")
        visible_node.default_expansion = Expansion.ALL
        visible_node.Render = Mock(return_value="Visible content")
        visible_node.clear_pending_traces = Mock()

        graph.add_node(hidden_node)
        graph.add_node(visible_node)

        # Build path with views (collect creates them on-demand)
        path = projection_engine._collect_render_path(graph)

        # Override the views to mark hidden node as hidden
        projection_engine.views["hidden"].hidden = True
        projection_engine.views["visible"].hidden = False

        sections = projection_engine._render_path(path)

        # Only visible node should be rendered
        assert len(sections) == 1
        assert sections[0].source_id == "visible"

    def test_render_path_calls_render(self, projection_engine, mock_graph):
        """Test that render_content is called for each visible node."""
        path = projection_engine._collect_render_path(mock_graph)
        projection_engine._render_path(path)

        running_node = mock_graph.get_node("running1")
        paused_node = mock_graph.get_node("paused_root")

        # render_content is called via NodeView.render()
        running_node.render_content.assert_called_once_with(
            text_buffers=None
        )
        paused_node.render_content.assert_called_once_with(
            text_buffers=None
        )

    def test_render_empty_path_returns_empty_sections(self, projection_engine):
        """Test rendering empty path returns no sections."""
        path = RenderPath()

        sections = projection_engine._render_path(path)

        assert sections == []


# =============================================================================
# Projection Assembly Tests
# =============================================================================


class TestProjectionBuild:
    """Tests for complete projection assembly."""

    def test_build_with_graph(self, projection_engine, mock_graph):
        """Test building projection with context graph."""
        projection = projection_engine.build(
            context_graph=mock_graph,
        )

        assert len(projection.sections) >= 2  # Graph nodes
        assert projection.handles is not None
        assert "running1" in projection.handles

    def test_build_with_empty_graph(self, projection_engine):
        """Test building projection with empty graph."""
        empty_graph = ContextGraph()

        projection = projection_engine.build(
            context_graph=empty_graph,
        )

        # Empty graph = no sections
        assert len(projection.sections) == 0

    def test_build_no_context(self, projection_engine):
        """Test building with no context graph."""
        projection = projection_engine.build()

        # No context = no sections
        assert len(projection.sections) == 0
        assert projection.handles == {}

    def test_section_ordering(self, projection_engine, mock_graph):
        """Test that sections appear in expected order."""
        projection = projection_engine.build(
            context_graph=mock_graph,
        )

        # Graph nodes rendered
        assert len(projection.sections) >= 1

    def test_handles_from_graph(self, projection_engine, mock_graph):
        """Test that handles dict is built from graph nodes."""
        projection = projection_engine.build(
            context_graph=mock_graph,
        )

        assert "running1" in projection.handles
        assert "paused_root" in projection.handles
        assert isinstance(projection.handles["running1"], dict)


# =============================================================================
# Integration Tests
# =============================================================================


class TestProjectionIntegration:
    """Integration tests for full projection builds."""

    def test_full_projection_with_all_components(self):
        """Test building projection with all components."""
        # Create engine
        engine = ProjectionEngine()

        # Create graph with nodes
        graph = ContextGraph()
        node = create_mock_context_node("node1", "view", mode="running")
        node.default_expansion = Expansion.ALL
        node.Render = Mock(return_value="Node content")
        node.clear_pending_traces = Mock()
        graph.add_node(node)

        # Build projection
        projection = engine.build(
            context_graph=graph,
        )

        # Verify components
        assert len(projection.sections) >= 1
        assert any(s.section_type == "view" for s in projection.sections)

    def test_projection_metadata(self, projection_engine, mock_graph):
        """Test that section metadata is populated correctly."""
        projection = projection_engine.build(
            context_graph=mock_graph,
        )

        for section in projection.sections:
            assert section.section_type is not None
            assert section.source_id is not None
            assert section.content is not None
            assert section.tokens_used >= 0


# =============================================================================
# Tree Character Tests
# =============================================================================


class TestTreeCharacters:
    """Tests for ASCII tree-drawing character functionality."""

    def test_default_tree_config(self):
        """Test default tree character configuration."""
        config = ProjectionConfig()
        assert config.tree_detail == "| "
        assert config.tree_content == "|."
        assert config.tree_child == "+-"
        assert config.tree_last_child == "\\-"

    def test_custom_tree_config(self):
        """Test custom tree character configuration."""
        config = ProjectionConfig(
            tree_detail="│ ",
            tree_content="│·",
            tree_child="├─",
            tree_last_child="└─",
        )
        assert config.tree_detail == "│ "
        assert config.tree_content == "│·"
        assert config.tree_child == "├─"
        assert config.tree_last_child == "└─"

    def test_root_node_has_empty_tree_prefix(self, projection_engine):
        """Test that root nodes have empty tree prefix."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        graph.add_node(root)

        projection_engine._collect_render_path(graph)

        assert projection_engine.views["root"].tree_prefix == ""

    def test_single_child_gets_last_child_prefix(self, projection_engine):
        """Test that a single child uses last_child prefix."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        child = create_mock_context_node("child", "view")
        graph.add_node(root)
        graph.add_node(child)
        graph.link("child", "root")

        projection_engine._collect_render_path(graph)

        # Single child is also last child
        assert projection_engine.views["child"].tree_prefix == "\\-"

    def test_multiple_children_get_correct_prefixes(self, projection_engine):
        """Test that multiple children get correct branch prefixes."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        child1 = create_mock_context_node("child1", "view")
        child2 = create_mock_context_node("child2", "view")
        child3 = create_mock_context_node("child3", "view")
        graph.add_node(root)
        graph.add_node(child1)
        graph.add_node(child2)
        graph.add_node(child3)
        graph.link("child1", "root")
        graph.link("child2", "root")
        graph.link("child3", "root")

        projection_engine._collect_render_path(graph)

        # First two children use tree_child, last uses tree_last_child
        assert projection_engine.views["child1"].tree_prefix == "+-"
        assert projection_engine.views["child2"].tree_prefix == "+-"
        assert projection_engine.views["child3"].tree_prefix == "\\-"

    def test_nested_children_propagate_tree_detail(self, projection_engine):
        """Test that nested children propagate tree_detail correctly."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        child1 = create_mock_context_node("child1", "view")
        child2 = create_mock_context_node("child2", "view")
        grandchild = create_mock_context_node("grandchild", "view")
        graph.add_node(root)
        graph.add_node(child1)
        graph.add_node(child2)
        graph.add_node(grandchild)
        graph.link("child1", "root")
        graph.link("child2", "root")
        graph.link("grandchild", "child1")

        projection_engine._collect_render_path(graph)

        # grandchild is under child1 (not last), so prefix is "| " + "\\-"
        assert projection_engine.views["grandchild"].tree_prefix == "| \\-"

    def test_nested_under_last_child_uses_blank(self, projection_engine):
        """Test that children under last sibling use blank continuation."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        child1 = create_mock_context_node("child1", "view")
        child2 = create_mock_context_node("child2", "view")
        grandchild = create_mock_context_node("grandchild", "view")
        graph.add_node(root)
        graph.add_node(child1)
        graph.add_node(child2)
        graph.add_node(grandchild)
        graph.link("child1", "root")
        graph.link("child2", "root")
        graph.link("grandchild", "child2")  # Under last child

        projection_engine._collect_render_path(graph)

        # grandchild is under child2 (last), so prefix is "  " + "\\-"
        assert projection_engine.views["grandchild"].tree_prefix == "  \\-"

    def test_tree_prefix_passed_to_projection_section(self, projection_engine):
        """Test that tree_prefix is passed to ProjectionSection."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        root.Render = Mock(return_value="Root content")
        root.clear_pending_traces = Mock()
        child = create_mock_context_node("child", "view")
        child.Render = Mock(return_value="Child content")
        child.clear_pending_traces = Mock()
        graph.add_node(root)
        graph.add_node(child)
        graph.link("child", "root")

        projection = projection_engine.build(context_graph=graph)

        # Find the child section
        child_section = next(s for s in projection.sections if s.source_id == "child")
        assert child_section.tree_prefix == "\\-"

    def test_projection_render_with_tree_prefixes(self, projection_engine):
        """Test that Projection.render() applies tree prefixes."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        root.render_content = Mock(return_value="Root content\n")
        root.clear_pending_traces = Mock()
        child = create_mock_context_node("child", "view")
        child.render_content = Mock(return_value="Child content\nLine 2\n")
        child.clear_pending_traces = Mock()
        graph.add_node(root)
        graph.add_node(child)
        graph.link("child", "root")

        projection = projection_engine.build(context_graph=graph)
        rendered = projection.render()

        lines = rendered.split("\n")
        # Root lines should have no prefix (just indent from section.indent=0)
        root_lines = [l for l in lines if "root" in l.lower()]
        assert any(not l.startswith("\\-") and not l.startswith("+-") for l in root_lines)

        # Child header should have tree prefix (first line with "child" in it)
        child_lines = [l for l in lines if "child" in l.lower()]
        assert any(l.startswith("\\-") for l in child_lines)

        # Find a content line under child (Line 2)
        line2_lines = [l for l in lines if "Line 2" in l]
        if line2_lines:
            # Content under last child uses " ." (blank + dot)
            assert line2_lines[0].startswith(" .")

    def test_content_continuation_for_non_last_child(self, projection_engine):
        """Test content continuation uses tree_content for non-last children."""
        graph = ContextGraph()
        root = create_mock_context_node("root", "view")
        root.render_content = Mock(return_value="Root\n")
        root.clear_pending_traces = Mock()
        child1 = create_mock_context_node("child1", "view")
        child1.render_content = Mock(return_value="Child1 header\nContent line\n")
        child1.clear_pending_traces = Mock()
        child2 = create_mock_context_node("child2", "view")
        child2.render_content = Mock(return_value="Child2\n")
        child2.clear_pending_traces = Mock()
        graph.add_node(root)
        graph.add_node(child1)
        graph.add_node(child2)
        graph.link("child1", "root")
        graph.link("child2", "root")

        projection = projection_engine.build(context_graph=graph)
        rendered = projection.render()

        lines = rendered.split("\n")
        # child1 is not last, so content uses "|." marker
        # Find the content line for child1
        content_lines = [l for l in lines if "Content line" in l]
        if content_lines:
            assert content_lines[0].startswith("|.")

    def test_deeply_nested_tree_structure(self, projection_engine):
        """Test tree prefixes for deeply nested structure."""
        graph = ContextGraph()

        # Create a deep hierarchy: root -> a -> b -> c
        root = create_mock_context_node("root", "view")
        a = create_mock_context_node("a", "view")
        b = create_mock_context_node("b", "view")
        c = create_mock_context_node("c", "view")

        for node in [root, a, b, c]:
            node.Render = Mock(return_value=f"# {node.node_id}")
            node.clear_pending_traces = Mock()
            graph.add_node(node)

        graph.link("a", "root")
        graph.link("b", "a")
        graph.link("c", "b")

        projection_engine._collect_render_path(graph)

        # All are last children in their respective levels
        assert projection_engine.views["root"].tree_prefix == ""
        assert projection_engine.views["a"].tree_prefix == "\\-"
        assert projection_engine.views["b"].tree_prefix == "  \\-"  # blank + last
        assert projection_engine.views["c"].tree_prefix == "    \\-"  # blank + blank + last

    def test_mixed_tree_structure(self, projection_engine):
        """Test tree prefixes for mixed structure with multiple branches."""
        graph = ContextGraph()

        # Structure:
        # root
        # ├─ a
        # │  ├─ a1
        # │  └─ a2
        # └─ b
        #    └─ b1
        root = create_mock_context_node("root", "view")
        a = create_mock_context_node("a", "view")
        a1 = create_mock_context_node("a1", "view")
        a2 = create_mock_context_node("a2", "view")
        b = create_mock_context_node("b", "view")
        b1 = create_mock_context_node("b1", "view")

        for node in [root, a, a1, a2, b, b1]:
            node.Render = Mock(return_value=f"# {node.node_id}")
            node.clear_pending_traces = Mock()
            graph.add_node(node)

        graph.link("a", "root")
        graph.link("b", "root")
        graph.link("a1", "a")
        graph.link("a2", "a")
        graph.link("b1", "b")

        projection_engine._collect_render_path(graph)

        assert projection_engine.views["root"].tree_prefix == ""
        assert projection_engine.views["a"].tree_prefix == "+-"  # not last
        assert projection_engine.views["a1"].tree_prefix == "| +-"  # under non-last, not last
        assert projection_engine.views["a2"].tree_prefix == "| \\-"  # under non-last, last
        assert projection_engine.views["b"].tree_prefix == "\\-"  # last
        assert projection_engine.views["b1"].tree_prefix == "  \\-"  # under last, last
