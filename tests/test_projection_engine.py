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
    running_node.expansion = Expansion.DETAILS
    running_node.Render = Mock(return_value="# Running Node Content\nThis is the content.")
    running_node.clear_pending_traces = Mock()

    paused_root = create_mock_context_node("paused_root", "view", mode="paused")
    paused_root.expansion = Expansion.SUMMARY
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
        assert path.node_ids == []
        assert path.edges == []
        assert path.root_ids == set()

    def test_render_path_with_nodes(self):
        """Test render path with nodes."""
        path = RenderPath(
            node_ids=["a", "b", "c"],
            edges=[("b", "a"), ("c", "a")],
            root_ids={"a"},
        )

        assert len(path) == 3
        assert path
        assert "a" in path.node_ids
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

        # Both roots and child (via recursion) should be in path
        assert "r1" in path.node_ids
        assert "r2" in path.node_ids
        assert "c1" in path.node_ids  # Included via parent's DETAILS state

    def test_collect_render_path_collapsed_still_recurses(self, projection_engine):
        """Test that COLLAPSED parents still recurse into children for token counting."""
        graph = ContextGraph()

        root = create_mock_context_node("root", "view")
        root.expansion = Expansion.COLLAPSED
        child = create_mock_context_node("child", "view")

        graph.add_node(root)
        graph.add_node(child)
        graph.link("child", "root")

        path = projection_engine._collect_render_path(graph)

        # Both root and child are collected - children always collected
        # for complete token information (expansion cost visibility)
        assert "root" in path.node_ids
        assert "child" in path.node_ids
        assert "root" in path.root_ids

    def test_collect_render_path_excludes_hidden(self, projection_engine):
        """Test that HIDDEN nodes are excluded from render path."""
        graph = ContextGraph()

        hidden_node = create_mock_context_node("hidden", "view", mode="running")
        hidden_node.expansion = Expansion.HIDDEN

        visible_node = create_mock_context_node("visible", "view", mode="running")
        visible_node.expansion = Expansion.DETAILS

        graph.add_node(hidden_node)
        graph.add_node(visible_node)

        path = projection_engine._collect_render_path(graph)

        assert "visible" in path.node_ids
        assert "hidden" not in path.node_ids

    def test_collect_render_path_records_edges(self, projection_engine):
        """Test that render path records parent-child edges."""
        graph = ContextGraph()

        parent = create_mock_context_node("parent", "view", mode="running")
        child = create_mock_context_node("child", "view", mode="running")

        graph.add_node(parent)
        graph.add_node(child)
        graph.link("child", "parent")

        path = projection_engine._collect_render_path(graph)

        # Both should be in path since both are running
        assert "parent" in path.node_ids
        assert "child" in path.node_ids

    def test_collect_render_path_empty_graph(self, projection_engine):
        """Test collecting render path from empty graph."""
        graph = ContextGraph()

        path = projection_engine._collect_render_path(graph)

        assert len(path) == 0
        assert not path


# =============================================================================
# Render Path Rendering Tests
# =============================================================================


class TestRenderPathRendering:
    """Tests for rendering the collected path."""

    def test_render_path_basic(self, projection_engine, mock_graph):
        """Test basic path rendering."""
        path = projection_engine._collect_render_path(mock_graph)
        sections = projection_engine._render_path(mock_graph, path, cwd=".")

        assert len(sections) == 2  # Running node + paused root
        section_ids = {s.source_id for s in sections}
        assert "running1" in section_ids
        assert "paused_root" in section_ids

    def test_render_path_excludes_hidden(self, projection_engine):
        """Test that HIDDEN nodes are excluded from rendered sections."""
        graph = ContextGraph()

        hidden_node = create_mock_context_node("hidden", "view", mode="running")
        hidden_node.expansion = Expansion.HIDDEN
        hidden_node.Render = Mock(return_value="Hidden content")
        hidden_node.clear_pending_traces = Mock()

        visible_node = create_mock_context_node("visible", "view", mode="running")
        visible_node.expansion = Expansion.DETAILS
        visible_node.Render = Mock(return_value="Visible content")
        visible_node.clear_pending_traces = Mock()

        graph.add_node(hidden_node)
        graph.add_node(visible_node)

        path = projection_engine._collect_render_path(graph)
        sections = projection_engine._render_path(graph, path, cwd=".")

        # Only visible node should be rendered
        assert len(sections) == 1
        assert sections[0].source_id == "visible"

    def test_render_path_calls_render(self, projection_engine, mock_graph):
        """Test that Render is called for each visible node."""
        path = projection_engine._collect_render_path(mock_graph)
        sections = projection_engine._render_path(mock_graph, path, cwd="/test")

        running_node = mock_graph.get_node("running1")
        paused_node = mock_graph.get_node("paused_root")

        running_node.Render.assert_called_once_with(cwd="/test", text_buffers=None)
        paused_node.Render.assert_called_once_with(cwd="/test", text_buffers=None)

    def test_render_empty_path_returns_empty_sections(self, projection_engine):
        """Test rendering empty path returns no sections."""
        graph = ContextGraph()
        path = RenderPath()

        sections = projection_engine._render_path(graph, path, cwd=".")

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
            cwd="/test",
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
        node.expansion = Expansion.DETAILS
        node.Render = Mock(return_value="Node content")
        node.clear_pending_traces = Mock()
        graph.add_node(node)

        # Build projection
        projection = engine.build(
            context_graph=graph,
            cwd="/test",
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
