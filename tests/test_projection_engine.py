"""Tests for projection engine and budget allocation.

Tests coverage for:
- src/activecontext/core/projection_engine.py
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.state import NodeState
from activecontext.core.llm.provider import Message, Role
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine
from tests.utils import create_mock_context_node, create_mock_message


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def projection_config():
    """Create ProjectionConfig with test values."""
    return ProjectionConfig(
        total_budget=1000,
        conversation_ratio=0.3,
        views_ratio=0.5,
        groups_ratio=0.2,
    )


@pytest.fixture
def projection_engine(projection_config):
    """Create ProjectionEngine with test config."""
    return ProjectionEngine(config=projection_config)


@pytest.fixture
def mock_messages():
    """Create list of mock Message objects."""
    return [
        create_mock_message("user", "Hello, how are you?", actor="user"),
        create_mock_message("assistant", "I'm doing great! How can I help?", actor="agent"),
        create_mock_message("user", "What's the weather like?", actor="user"),
        create_mock_message("assistant", "I don't have access to weather data.", actor="agent"),
    ]


@pytest.fixture
def mock_graph():
    """Create ContextGraph with mock nodes for projection."""
    graph = ContextGraph()

    # Create mock nodes with Render method
    running_node = create_mock_context_node("running1", "view", mode="running")
    running_node.state = NodeState.DETAILS
    running_node.Render = Mock(return_value="# Running Node Content\nThis is the content.")
    running_node.clear_pending_diffs = Mock()

    paused_root = create_mock_context_node("paused_root", "view", mode="paused")
    paused_root.state = NodeState.SUMMARY
    paused_root.Render = Mock(return_value="# Paused Root Summary")
    paused_root.clear_pending_diffs = Mock()

    # Add nodes to graph
    graph.add_node(running_node)
    graph.add_node(paused_root)

    return graph


# =============================================================================
# Budget Allocation Tests
# =============================================================================


class TestBudgetAllocation:
    """Tests for token budget allocation."""

    def test_default_ratios(self):
        """Test default budget ratios."""
        config = ProjectionConfig()

        assert config.total_budget == 8000
        assert config.conversation_ratio == 0.3
        assert config.views_ratio == 0.5
        assert config.groups_ratio == 0.2

    def test_custom_ratios(self):
        """Test custom budget ratios from config."""
        config = ProjectionConfig(
            total_budget=2000,
            conversation_ratio=0.25,
            views_ratio=0.55,
            groups_ratio=0.20,
        )

        engine = ProjectionEngine(config=config)

        assert engine.config.total_budget == 2000
        assert engine.config.conversation_ratio == 0.25
        assert engine.config.views_ratio == 0.55

    def test_config_from_app_config(self, monkeypatch):
        """Test loading config from app config."""
        # Mock app config
        mock_app_config = Mock()
        mock_app_config.projection = Mock()
        mock_app_config.projection.total_budget = 5000
        mock_app_config.projection.conversation_ratio = 0.2
        mock_app_config.projection.views_ratio = 0.6
        mock_app_config.projection.groups_ratio = 0.2

        # Patch at the source module where get_config is imported from
        monkeypatch.setattr(
            "activecontext.config.get_config",
            lambda: mock_app_config,
        )

        engine = ProjectionEngine()

        assert engine.config.total_budget == 5000
        assert engine.config.conversation_ratio == 0.2
        assert engine.config.views_ratio == 0.6

    def test_config_defaults_on_import_error(self):
        """Test that ProjectionEngine uses defaults when config not available."""
        # Create engine without app config (import will fail in test env)
        engine = ProjectionEngine()

        # Should fall back to defaults
        assert engine.config.total_budget == 8000


# =============================================================================
# Conversation Rendering Tests
# =============================================================================


class TestConversationRender:
    """Tests for conversation rendering."""

    def test_render_conversation_basic(self, projection_engine, mock_messages):
        """Test basic conversation rendering."""
        section = projection_engine._render_conversation(
            mock_messages, budget=500, show_actors=True
        )

        assert section is not None
        assert section.section_type == "conversation"
        assert section.source_id == "conversation"
        assert "## Conversation History" in section.content
        assert "USER" in section.content
        assert "ASSISTANT" in section.content
        assert "Hello, how are you?" in section.content

    def test_render_conversation_with_actors(self, projection_engine):
        """Test conversation rendering shows actors."""
        messages = [
            create_mock_message("user", "Test", actor="user"),
            create_mock_message("assistant", "Response", actor="agent"),
        ]

        section = projection_engine._render_conversation(
            messages, budget=500, show_actors=True
        )

        assert "(user)" in section.content
        assert "(agent)" in section.content

    def test_render_conversation_without_actors(self, projection_engine):
        """Test conversation rendering without actors."""
        messages = [
            create_mock_message("user", "Test", actor="user"),
            create_mock_message("assistant", "Response", actor="agent"),
        ]

        section = projection_engine._render_conversation(
            messages, budget=500, show_actors=False
        )

        assert "(user)" not in section.content
        assert "(agent)" not in section.content
        assert "USER" in section.content
        assert "ASSISTANT" in section.content

    def test_render_conversation_truncates_long_messages(self, projection_engine):
        """Test that very long messages are truncated."""
        long_message = "x" * 3000
        messages = [create_mock_message("user", long_message, actor="user")]

        section = projection_engine._render_conversation(messages, budget=1000)

        assert "..." in section.content
        assert len(section.content) < len(long_message)

    def test_render_conversation_budget_limit(self, projection_engine):
        """Test conversation respects budget limits."""
        # Create many messages with enough content to exceed a small budget
        messages = [
            create_mock_message("user", f"This is message number {i} with some extra content", actor="user")
            for i in range(20)
        ]

        # Use a small budget that can't fit all messages
        section = projection_engine._render_conversation(messages, budget=100)

        # Should summarize older messages
        assert "earlier messages omitted" in section.content
        assert section.tokens_used <= 100

    def test_render_empty_conversation(self, projection_engine):
        """Test rendering empty conversation returns None."""
        section = projection_engine._render_conversation([], budget=500)

        assert section is None

    def test_conversation_message_order(self, projection_engine):
        """Test messages appear in chronological order."""
        messages = [
            create_mock_message("user", "First", actor="user"),
            create_mock_message("assistant", "Second", actor="agent"),
            create_mock_message("user", "Third", actor="user"),
        ]

        section = projection_engine._render_conversation(messages, budget=1000)

        # Check that "First" appears before "Second" before "Third"
        first_idx = section.content.index("First")
        second_idx = section.content.index("Second")
        third_idx = section.content.index("Third")

        assert first_idx < second_idx < third_idx


# =============================================================================
# Graph Rendering Tests
# =============================================================================


class TestGraphRender:
    """Tests for graph-based context rendering."""

    def test_render_graph_basic(self, projection_engine, mock_graph):
        """Test basic graph rendering."""
        sections = projection_engine._render_graph(mock_graph, budget=1000, cwd=".")

        assert len(sections) == 2  # Running node + paused root
        section_ids = {s.source_id for s in sections}
        assert "running1" in section_ids
        assert "paused_root" in section_ids

    def test_render_graph_excludes_hidden(self, projection_engine):
        """Test that HIDDEN nodes are excluded from projection."""
        graph = ContextGraph()

        hidden_node = create_mock_context_node("hidden", "view", mode="running")
        hidden_node.state = NodeState.HIDDEN
        hidden_node.Render = Mock(return_value="Hidden content")
        hidden_node.clear_pending_diffs = Mock()

        visible_node = create_mock_context_node("visible", "view", mode="running")
        visible_node.state = NodeState.DETAILS
        visible_node.Render = Mock(return_value="Visible content")
        visible_node.clear_pending_diffs = Mock()

        graph.add_node(hidden_node)
        graph.add_node(visible_node)

        sections = projection_engine._render_graph(graph, budget=1000, cwd=".")

        # Only visible node should be rendered
        assert len(sections) == 1
        assert sections[0].source_id == "visible"

    def test_render_graph_calls_render_with_budget(self, projection_engine, mock_graph):
        """Test that Render is called with per-node budget."""
        sections = projection_engine._render_graph(mock_graph, budget=1000, cwd="/test")

        # Two visible nodes, so budget is split
        per_node_budget = 1000 // 2

        running_node = mock_graph.get_node("running1")
        paused_node = mock_graph.get_node("paused_root")

        running_node.Render.assert_called_once_with(tokens=per_node_budget, cwd="/test")
        paused_node.Render.assert_called_once_with(tokens=per_node_budget, cwd="/test")

    def test_render_graph_clears_pending_diffs(self, projection_engine, mock_graph):
        """Test that rendering clears pending diffs."""
        projection_engine._render_graph(mock_graph, budget=1000, cwd=".")

        for node in mock_graph:
            node.clear_pending_diffs.assert_called_once()

    def test_collect_visible_nodes_running(self, projection_engine):
        """Test collecting visible nodes includes running nodes."""
        graph = ContextGraph()

        running1 = create_mock_context_node("r1", "view", mode="running")
        running2 = create_mock_context_node("r2", "view", mode="running")
        paused = create_mock_context_node("p1", "view", mode="paused")

        graph.add_node(running1)
        graph.add_node(running2)
        graph.add_node(paused)

        # Link paused as child
        graph.link("p1", "r1")

        visible = projection_engine._collect_visible_nodes(graph)
        visible_ids = {n.node_id for n in visible}

        # Running nodes and paused root
        assert "r1" in visible_ids
        assert "r2" in visible_ids
        assert "p1" not in visible_ids  # Not a root

    def test_collect_visible_nodes_paused_roots(self, projection_engine):
        """Test that paused root nodes are visible."""
        graph = ContextGraph()

        paused_root = create_mock_context_node("root", "view", mode="paused")
        paused_child = create_mock_context_node("child", "view", mode="paused")

        graph.add_node(paused_root)
        graph.add_node(paused_child)
        graph.link("child", "root")

        visible = projection_engine._collect_visible_nodes(graph)
        visible_ids = {n.node_id for n in visible}

        # Only paused root is visible
        assert "root" in visible_ids
        assert "child" not in visible_ids

    def test_empty_graph_returns_empty_sections(self, projection_engine):
        """Test rendering empty graph returns no sections."""
        graph = ContextGraph()

        sections = projection_engine._render_graph(graph, budget=1000, cwd=".")

        assert sections == []


# =============================================================================
# Projection Assembly Tests
# =============================================================================


class TestProjectionBuild:
    """Tests for complete projection assembly."""

    def test_build_with_graph(self, projection_engine, mock_graph, mock_messages):
        """Test building projection with context graph."""
        projection = projection_engine.build(
            context_graph=mock_graph,
            conversation=mock_messages,
            cwd="/test",
        )

        assert projection.token_budget == 1000
        assert len(projection.sections) >= 2  # Conversation + nodes
        assert projection.handles is not None
        assert "running1" in projection.handles

    def test_build_with_empty_graph(self, projection_engine, mock_messages):
        """Test building projection with empty graph."""
        empty_graph = ContextGraph()

        projection = projection_engine.build(
            context_graph=empty_graph,
            conversation=mock_messages,
        )

        # Should still have conversation
        assert len(projection.sections) >= 1
        assert projection.sections[0].section_type == "conversation"

    def test_build_with_legacy_objects(self, projection_engine, mock_messages):
        """Test building projection with legacy context_objects."""
        # Create mock view
        mock_view = Mock()
        mock_view.Render = Mock(return_value="# View Content")
        mock_view.GetDigest = Mock(return_value={"type": "view", "path": "test.py"})
        mock_view.state = NodeState.DETAILS

        # Create mock group
        mock_group = Mock()
        mock_group.Render = Mock(return_value="# Group Summary")
        mock_group.GetDigest = Mock(return_value={"type": "group", "members": 2})
        mock_group.state = NodeState.SUMMARY

        context_objects = {
            "view1": mock_view,
            "group1": mock_group,
        }

        projection = projection_engine.build(
            context_objects=context_objects,
            conversation=mock_messages,
            cwd="/test",
        )

        assert len(projection.sections) >= 3  # Conversation + view + group
        section_types = {s.section_type for s in projection.sections}
        assert "conversation" in section_types
        assert "view" in section_types
        assert "group" in section_types

    def test_build_respects_budget_override(self, projection_engine, mock_messages):
        """Test that token_budget parameter overrides config."""
        projection = projection_engine.build(
            context_graph=ContextGraph(),
            conversation=mock_messages,
            token_budget=5000,
        )

        assert projection.token_budget == 5000

    def test_build_no_context(self, projection_engine, mock_messages):
        """Test building with no context objects or graph."""
        projection = projection_engine.build(
            conversation=mock_messages,
        )

        # Should only have conversation
        assert len(projection.sections) == 1
        assert projection.sections[0].section_type == "conversation"
        assert projection.handles == {}

    def test_section_ordering(self, projection_engine, mock_graph, mock_messages):
        """Test that sections appear in expected order."""
        projection = projection_engine.build(
            context_graph=mock_graph,
            conversation=mock_messages,
        )

        # Conversation should be first
        assert projection.sections[0].section_type == "conversation"

    def test_handles_from_graph(self, projection_engine, mock_graph):
        """Test that handles dict is built from graph nodes."""
        projection = projection_engine.build(
            context_graph=mock_graph,
            conversation=[],
        )

        assert "running1" in projection.handles
        assert "paused_root" in projection.handles
        assert isinstance(projection.handles["running1"], dict)


# =============================================================================
# Legacy Rendering Tests
# =============================================================================


class TestLegacyRendering:
    """Tests for legacy view/group rendering."""

    def test_render_views_basic(self, projection_engine):
        """Test basic view rendering."""
        view = Mock()
        view._cwd = None
        view.Render = Mock(return_value="View content")
        view.GetDigest = Mock(return_value={"type": "view"})
        view.state = NodeState.DETAILS

        views = {"view1": view}
        sections = projection_engine._render_views(views, budget=500, cwd="/test")

        assert len(sections) == 1
        assert sections[0].section_type == "view"
        assert sections[0].source_id == "view1"
        assert sections[0].content == "View content"

        # Should set cwd
        assert view._cwd == "/test"

    def test_render_views_without_render_method(self, projection_engine):
        """Test rendering views without Render method (fallback)."""
        view = Mock()
        view.GetDigest = Mock(return_value={"path": "test.py"})
        del view.Render  # Remove Render method

        views = {"view1": view}
        sections = projection_engine._render_views(views, budget=500, cwd="/test")

        assert len(sections) == 1
        assert "[View view1:" in sections[0].content

    def test_render_groups_basic(self, projection_engine):
        """Test basic group rendering."""
        group = Mock()
        group.Render = Mock(return_value="Group summary")
        group.GetDigest = Mock(return_value={"type": "group"})
        group.state = NodeState.SUMMARY

        groups = {"group1": group}
        sections = projection_engine._render_groups(groups, budget=200)

        assert len(sections) == 1
        assert sections[0].section_type == "group"
        assert sections[0].content == "Group summary"

    def test_render_groups_without_render(self, projection_engine):
        """Test rendering groups without Render method."""
        group = Mock()
        group.members = ["a", "b", "c"]
        group.GetDigest = Mock(return_value={})
        del group.Render

        groups = {"group1": group}
        sections = projection_engine._render_groups(groups, budget=200)

        assert "[Group group1: 3 members]" in sections[0].content

    def test_is_view(self, projection_engine):
        """Test _is_view helper."""
        view = Mock()
        view.GetDigest = Mock(return_value={"type": "view"})

        assert projection_engine._is_view(view) is True

        not_view = Mock()
        not_view.GetDigest = Mock(return_value={"type": "group"})

        assert projection_engine._is_view(not_view) is False

    def test_is_group(self, projection_engine):
        """Test _is_group helper."""
        group = Mock()
        group.GetDigest = Mock(return_value={"type": "group"})

        assert projection_engine._is_group(group) is True

        not_group = Mock()
        not_group.GetDigest = Mock(return_value={"type": "view"})

        assert projection_engine._is_group(not_group) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestProjectionIntegration:
    """Integration tests for full projection builds."""

    def test_full_projection_with_all_components(self):
        """Test building projection with all components."""
        # Create engine
        config = ProjectionConfig(total_budget=2000)
        engine = ProjectionEngine(config=config)

        # Create graph with nodes
        graph = ContextGraph()
        node = create_mock_context_node("node1", "view", mode="running")
        node.state = NodeState.DETAILS
        node.Render = Mock(return_value="Node content")
        node.clear_pending_diffs = Mock()
        graph.add_node(node)

        # Create messages
        messages = [
            create_mock_message("user", "Test message", actor="user"),
        ]

        # Build projection
        projection = engine.build(
            context_graph=graph,
            conversation=messages,
            cwd="/test",
            show_message_actors=True,
        )

        # Verify components
        assert projection.token_budget == 2000
        assert len(projection.sections) >= 2
        assert any(s.section_type == "conversation" for s in projection.sections)
        assert any(s.section_type == "view" for s in projection.sections)

    def test_projection_metadata(self, projection_engine, mock_graph):
        """Test that section metadata is populated correctly."""
        projection = projection_engine.build(
            context_graph=mock_graph,
            conversation=[],
        )

        for section in projection.sections:
            assert section.section_type is not None
            assert section.source_id is not None
            assert section.content is not None
            assert section.tokens_used >= 0
