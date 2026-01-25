"""Tests for ChoiceView.

Tests coverage for:
- src/activecontext/context/view.py (ChoiceView class)
- ChoiceView integration with projection engine
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.state import Expansion
from activecontext.context.view import ChoiceView, NodeView
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine
from tests.utils import create_mock_context_node


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_with_children():
    """Create ContextGraph with a parent node and 3 children."""
    graph = ContextGraph()

    # Create parent node
    parent = create_mock_context_node("parent", "group")
    parent.title = "Parent Group"
    parent.expansion = Expansion.ALL
    parent.children_ids = {"child-a", "child-b", "child-c"}
    parent.child_order = ["child-a", "child-b", "child-c"]
    parent.Render = Mock(return_value="# Parent Group")

    # Create child nodes
    child_a = create_mock_context_node("child-a", "text")
    child_a.title = "Option A"
    child_a.parent_ids = {"parent"}
    child_a.expansion = Expansion.ALL
    child_a.Render = Mock(return_value="Content A")

    child_b = create_mock_context_node("child-b", "text")
    child_b.title = "Option B"
    child_b.parent_ids = {"parent"}
    child_b.expansion = Expansion.ALL
    child_b.Render = Mock(return_value="Content B")

    child_c = create_mock_context_node("child-c", "text")
    child_c.title = "Option C"
    child_c.parent_ids = {"parent"}
    child_c.expansion = Expansion.ALL
    child_c.Render = Mock(return_value="Content C")

    # Add to graph
    graph.add_node(parent)
    graph.add_node(child_a)
    graph.add_node(child_b)
    graph.add_node(child_c)

    # Link children to parent
    graph.link("child-a", "parent")
    graph.link("child-b", "parent")
    graph.link("child-c", "parent")

    return graph


@pytest.fixture
def projection_engine():
    """Create ProjectionEngine with test config."""
    return ProjectionEngine(config=ProjectionConfig())


# =============================================================================
# ChoiceView Basic Tests
# =============================================================================


class TestChoiceViewBasic:
    """Tests for ChoiceView basic functionality."""

    def test_choice_view_creation(self, mock_graph_with_children):
        """Test creating a ChoiceView with initial selection."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b")

        assert choice.selected_id == "child-b"
        assert choice.node() is parent
        assert not choice.hide

    def test_choice_view_select_fluent(self, mock_graph_with_children):
        """Test fluent select method."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent)

        result = choice.select("child-a")

        assert result is choice  # Returns self
        assert choice.selected_id == "child-a"

    def test_choice_view_selected_id_setter(self, mock_graph_with_children):
        """Test setting selected_id via property."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-a")

        choice.selected_id = "child-c"

        assert choice.selected_id == "child-c"

    def test_choice_view_repr(self, mock_graph_with_children):
        """Test string representation."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b")

        repr_str = repr(choice)

        assert "ChoiceView" in repr_str
        assert "selected='child-b'" in repr_str


# =============================================================================
# apply_selection Tests
# =============================================================================


class TestApplySelection:
    """Tests for apply_selection behavior."""

    def test_apply_selection_hides_non_selected(self, mock_graph_with_children):
        """Test that non-selected children are hidden in ALL mode."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b", expand=Expansion.ALL)

        # Create views for all children
        views = {
            "parent": choice,
            "child-a": NodeView(mock_graph_with_children.get_node("child-a")),
            "child-b": NodeView(mock_graph_with_children.get_node("child-b")),
            "child-c": NodeView(mock_graph_with_children.get_node("child-c")),
        }

        choice.apply_selection(views)

        assert views["child-a"].hide is True
        assert views["child-b"].hide is False
        assert views["child-c"].hide is True

    def test_apply_selection_index_mode_no_changes(self, mock_graph_with_children):
        """Test that INDEX mode doesn't modify child views (renders headers itself)."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b", expand=Expansion.INDEX)

        views = {
            "parent": choice,
            "child-a": NodeView(
                mock_graph_with_children.get_node("child-a"), expand=Expansion.ALL
            ),
            "child-b": NodeView(
                mock_graph_with_children.get_node("child-b"), expand=Expansion.ALL
            ),
            "child-c": NodeView(
                mock_graph_with_children.get_node("child-c"), expand=Expansion.ALL
            ),
        }

        choice.apply_selection(views)

        # INDEX mode: no changes to children (ChoiceView renders headers itself)
        assert views["child-a"].hide is False
        assert views["child-b"].hide is False
        assert views["child-c"].hide is False
        assert views["child-a"].expand == Expansion.ALL
        assert views["child-b"].expand == Expansion.ALL
        assert views["child-c"].expand == Expansion.ALL

    def test_apply_selection_no_effect_in_header_mode(self, mock_graph_with_children):
        """Test that HEADER mode shows all children (not filtered)."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b", expand=Expansion.HEADER)

        views = {
            "parent": choice,
            "child-a": NodeView(mock_graph_with_children.get_node("child-a")),
            "child-b": NodeView(mock_graph_with_children.get_node("child-b")),
            "child-c": NodeView(mock_graph_with_children.get_node("child-c")),
        }

        choice.apply_selection(views)

        assert views["child-a"].hide is False
        assert views["child-b"].hide is False
        assert views["child-c"].hide is False

    def test_apply_selection_with_no_selection_hides_all(self, mock_graph_with_children):
        """Test that all children are hidden when no selection is made in ALL mode."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id=None, expand=Expansion.ALL)

        views = {
            "parent": choice,
            "child-a": NodeView(mock_graph_with_children.get_node("child-a")),
            "child-b": NodeView(mock_graph_with_children.get_node("child-b")),
            "child-c": NodeView(mock_graph_with_children.get_node("child-c")),
        }

        choice.apply_selection(views)

        # All children hidden when no selection in ALL mode
        assert views["child-a"].hide is True
        assert views["child-b"].hide is True
        assert views["child-c"].hide is True

    def test_apply_selection_missing_view(self, mock_graph_with_children):
        """Test that missing views are skipped gracefully."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b", expand=Expansion.ALL)

        # Only include some views (child-c missing)
        views = {
            "parent": choice,
            "child-a": NodeView(mock_graph_with_children.get_node("child-a")),
            "child-b": NodeView(mock_graph_with_children.get_node("child-b")),
        }

        # Should not raise, just skip missing views
        choice.apply_selection(views)

        assert views["child-a"].hide is True
        assert views["child-b"].hide is False


# =============================================================================
# get_options Tests
# =============================================================================


class TestGetOptions:
    """Tests for get_options method."""

    def test_get_options_returns_titles(self, mock_graph_with_children):
        """Test that get_options returns child titles in order."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent)

        options = choice.get_options()

        assert options == ["Option A", "Option B", "Option C"]

    def test_get_options_no_graph(self, mock_graph_with_children):
        """Test get_options when node has no graph reference."""
        parent = mock_graph_with_children.get_node("parent")
        parent._graph = None
        choice = ChoiceView(parent)

        options = choice.get_options()

        assert options == []


# =============================================================================
# render_brief Tests
# =============================================================================


class TestRenderIndex:
    """Tests for render_index method."""

    def test_render_index_returns_child_headers(self, mock_graph_with_children):
        """Test that render_index returns header lines for all children."""
        parent = mock_graph_with_children.get_node("parent")

        # Add render_header to children
        mock_graph_with_children.get_node("child-a").render_header = Mock(
            return_value="## Option A [10 tokens]"
        )
        mock_graph_with_children.get_node("child-b").render_header = Mock(
            return_value="## Option B [20 tokens]"
        )
        mock_graph_with_children.get_node("child-c").render_header = Mock(
            return_value="## Option C [15 tokens]"
        )

        choice = ChoiceView(parent)
        index = choice.render_index()

        # Should contain all child headers
        assert "Option A" in index
        assert "Option B" in index
        assert "Option C" in index
        assert "[10 tokens]" in index

    def test_render_index_no_graph(self, mock_graph_with_children):
        """Test render_index when node has no graph reference."""
        parent = mock_graph_with_children.get_node("parent")
        parent._graph = None
        choice = ChoiceView(parent)

        index = choice.render_index()

        assert index == ""


class TestRenderBrief:
    """Tests for render_brief method."""

    def test_render_brief_with_selection(self, mock_graph_with_children):
        """Test render_brief shows 'selected [A | B | C]' format."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id="child-b")

        brief = choice.render_brief()

        assert brief == "Option B [Option A | Option B | Option C]"

    def test_render_brief_no_selection_uses_first(self, mock_graph_with_children):
        """Test render_brief uses first option when none selected."""
        parent = mock_graph_with_children.get_node("parent")
        choice = ChoiceView(parent, selected_id=None)

        brief = choice.render_brief()

        assert brief == "Option A [Option A | Option B | Option C]"

    def test_render_brief_no_options(self, mock_graph_with_children):
        """Test render_brief with empty child list."""
        parent = mock_graph_with_children.get_node("parent")
        parent.child_order = []
        parent.children_ids = set()
        choice = ChoiceView(parent)

        brief = choice.render_brief()

        assert brief == "[No options]"


# =============================================================================
# Projection Engine Integration Tests
# =============================================================================


class TestProjectionEngineIntegration:
    """Tests for ChoiceView integration with ProjectionEngine."""

    def test_choice_view_filters_in_projection(
        self, mock_graph_with_children, projection_engine
    ):
        """Test that ChoiceView filtering works through projection engine."""
        parent = mock_graph_with_children.get_node("parent")

        # Create views with ChoiceView for parent
        choice = ChoiceView(parent, selected_id="child-b", expand=Expansion.ALL)
        views = {
            "parent": choice,
            "child-a": NodeView(
                mock_graph_with_children.get_node("child-a"), expand=Expansion.ALL
            ),
            "child-b": NodeView(
                mock_graph_with_children.get_node("child-b"), expand=Expansion.ALL
            ),
            "child-c": NodeView(
                mock_graph_with_children.get_node("child-c"), expand=Expansion.ALL
            ),
        }

        # Build projection
        projection = projection_engine.build(
            context_graph=mock_graph_with_children,
            views=views,
        )

        # After build, non-selected views should be hidden
        assert views["child-a"].hide is True
        assert views["child-b"].hide is False
        assert views["child-c"].hide is True

        # The projection should only include parent and selected child
        rendered_ids = [section.source_id for section in projection.sections]
        assert "parent" in rendered_ids
        assert "child-b" in rendered_ids
        assert "child-a" not in rendered_ids
        assert "child-c" not in rendered_ids

    def test_multiple_choice_views(self, mock_graph_with_children, projection_engine):
        """Test that multiple ChoiceViews can coexist."""
        parent = mock_graph_with_children.get_node("parent")

        # Create a second parent node with children
        parent2 = create_mock_context_node("parent2", "group")
        parent2.title = "Parent 2"
        parent2.expansion = Expansion.ALL
        parent2.children_ids = {"child-x", "child-y"}
        parent2.child_order = ["child-x", "child-y"]
        parent2.Render = Mock(return_value="# Parent 2")

        child_x = create_mock_context_node("child-x", "text")
        child_x.title = "Option X"
        child_x.parent_ids = {"parent2"}
        child_x.expansion = Expansion.ALL
        child_x.Render = Mock(return_value="Content X")

        child_y = create_mock_context_node("child-y", "text")
        child_y.title = "Option Y"
        child_y.parent_ids = {"parent2"}
        child_y.expansion = Expansion.ALL
        child_y.Render = Mock(return_value="Content Y")

        mock_graph_with_children.add_node(parent2)
        mock_graph_with_children.add_node(child_x)
        mock_graph_with_children.add_node(child_y)
        mock_graph_with_children.link("child-x", "parent2")
        mock_graph_with_children.link("child-y", "parent2")

        # Create views with two ChoiceViews
        choice1 = ChoiceView(parent, selected_id="child-b", expand=Expansion.ALL)
        choice2 = ChoiceView(parent2, selected_id="child-x", expand=Expansion.ALL)

        views = {
            "parent": choice1,
            "child-a": NodeView(
                mock_graph_with_children.get_node("child-a"), expand=Expansion.ALL
            ),
            "child-b": NodeView(
                mock_graph_with_children.get_node("child-b"), expand=Expansion.ALL
            ),
            "child-c": NodeView(
                mock_graph_with_children.get_node("child-c"), expand=Expansion.ALL
            ),
            "parent2": choice2,
            "child-x": NodeView(
                mock_graph_with_children.get_node("child-x"), expand=Expansion.ALL
            ),
            "child-y": NodeView(
                mock_graph_with_children.get_node("child-y"), expand=Expansion.ALL
            ),
        }

        # Build projection
        projection_engine.build(
            context_graph=mock_graph_with_children,
            views=views,
        )

        # First choice: child-b selected
        assert views["child-a"].hide is True
        assert views["child-b"].hide is False
        assert views["child-c"].hide is True

        # Second choice: child-x selected
        assert views["child-x"].hide is False
        assert views["child-y"].hide is True
