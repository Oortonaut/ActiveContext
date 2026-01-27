"""Tests for progression views (SequenceView, LoopView, StateView).

Tests coverage for:
- src/activecontext/context/view.py (SequenceView, LoopView, StateView classes)
- Progression views integration with projection engine
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.state import Expansion
from activecontext.context.view import LoopView, NodeView, SequenceView, StateView
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine
from tests.utils import create_mock_context_node

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_graph_with_steps():
    """Create ContextGraph with a parent node and 3 step children."""
    graph = ContextGraph()

    # Create parent node (group for sequence)
    parent = create_mock_context_node("parent", "group")
    parent.title = "Workflow"
    parent.expansion = Expansion.ALL
    parent.children_ids = {"step-1", "step-2", "step-3"}
    parent.child_order = ["step-1", "step-2", "step-3"]
    parent.Render = Mock(return_value="# Workflow")

    # Create step nodes
    step1 = create_mock_context_node("step-1", "text")
    step1.title = "Understand"
    step1.parent_ids = {"parent"}
    step1.expansion = Expansion.ALL
    step1.Render = Mock(return_value="Read and understand the code...")

    step2 = create_mock_context_node("step-2", "text")
    step2.title = "Analyze"
    step2.parent_ids = {"parent"}
    step2.expansion = Expansion.ALL
    step2.Render = Mock(return_value="Identify issues...")

    step3 = create_mock_context_node("step-3", "text")
    step3.title = "Suggest"
    step3.parent_ids = {"parent"}
    step3.expansion = Expansion.ALL
    step3.Render = Mock(return_value="Propose improvements...")

    # Add to graph
    graph.add_node(parent)
    graph.add_node(step1)
    graph.add_node(step2)
    graph.add_node(step3)

    # Link children to parent
    graph.link("step-1", "parent")
    graph.link("step-2", "parent")
    graph.link("step-3", "parent")

    return graph


@pytest.fixture
def mock_graph_with_loop_child():
    """Create ContextGraph with a single child node for looping."""
    graph = ContextGraph()

    child = create_mock_context_node("review", "text")
    child.title = "Review"
    child.expansion = Expansion.ALL
    child.Render = Mock(return_value="Review content based on feedback...")

    graph.add_node(child)
    return graph


@pytest.fixture
def mock_graph_with_states():
    """Create ContextGraph with state nodes."""
    graph = ContextGraph()

    # Create parent node (group for state machine)
    parent = create_mock_context_node("parent", "group")
    parent.title = "Task"
    parent.expansion = Expansion.ALL
    parent.children_ids = {"idle", "working", "done"}
    parent.child_order = ["idle", "working", "done"]
    parent.Render = Mock(return_value="# Task")

    # Create state nodes
    idle = create_mock_context_node("idle", "text")
    idle.title = "Idle"
    idle.parent_ids = {"parent"}
    idle.expansion = Expansion.ALL
    idle.Render = Mock(return_value="Waiting for work...")

    working = create_mock_context_node("working", "text")
    working.title = "Working"
    working.parent_ids = {"parent"}
    working.expansion = Expansion.ALL
    working.Render = Mock(return_value="In progress...")

    done = create_mock_context_node("done", "text")
    done.title = "Done"
    done.parent_ids = {"parent"}
    done.expansion = Expansion.ALL
    done.Render = Mock(return_value="Complete!")

    # Add to graph
    graph.add_node(parent)
    graph.add_node(idle)
    graph.add_node(working)
    graph.add_node(done)

    # Link children to parent
    graph.link("idle", "parent")
    graph.link("working", "parent")
    graph.link("done", "parent")

    return graph


@pytest.fixture
def projection_engine():
    """Create ProjectionEngine with test config."""
    return ProjectionEngine(config=ProjectionConfig())


# =============================================================================
# SequenceView Basic Tests
# =============================================================================


class TestSequenceViewBasic:
    """Tests for SequenceView basic functionality."""

    def test_sequence_view_creation(self, mock_graph_with_steps):
        """Test creating a SequenceView."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        assert seq.current_index == 0
        assert seq.progress == "0/3"
        assert not seq.is_complete
        assert seq.total_steps == 3
        assert seq.completed_steps == set()

    def test_sequence_view_initial_selection(self, mock_graph_with_steps):
        """Test that SequenceView selects first child by default."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        assert seq.selected_id == "step-1"

    def test_sequence_view_advance(self, mock_graph_with_steps):
        """Test advancing through sequence."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        # Advance to step 2
        result = seq.advance()

        assert result is seq  # Returns self for fluent API
        assert seq.current_index == 1
        assert seq.selected_id == "step-2"
        assert 0 in seq.completed_steps
        assert seq.progress == "1/3"

    def test_sequence_view_advance_to_end(self, mock_graph_with_steps):
        """Test advancing through entire sequence."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        seq.advance()  # 0 -> 1
        seq.advance()  # 1 -> 2
        seq.advance()  # 2 -> stays at 2 (last step), marks 2 complete

        assert seq.current_index == 2
        assert seq.is_complete
        assert seq.progress == "3/3"
        assert seq.completed_steps == {0, 1, 2}

    def test_sequence_view_back(self, mock_graph_with_steps):
        """Test going back in sequence."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        seq.advance()
        seq.advance()
        result = seq.back()

        assert result is seq
        assert seq.current_index == 1
        assert seq.selected_id == "step-2"
        # Going back doesn't remove completion
        assert 0 in seq.completed_steps
        assert 1 in seq.completed_steps

    def test_sequence_view_back_at_start(self, mock_graph_with_steps):
        """Test back() at start does nothing."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        seq.back()

        assert seq.current_index == 0
        assert seq.selected_id == "step-1"

    def test_sequence_view_mark_complete(self, mock_graph_with_steps):
        """Test marking current step complete without advancing."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        result = seq.mark_complete()

        assert result is seq
        assert seq.current_index == 0  # Didn't advance
        assert 0 in seq.completed_steps
        assert seq.progress == "1/3"

    def test_sequence_view_skip(self, mock_graph_with_steps):
        """Test skipping current step without completing."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        result = seq.skip()

        assert result is seq
        assert seq.current_index == 1
        assert seq.selected_id == "step-2"
        assert 0 not in seq.completed_steps  # Didn't mark complete

    def test_sequence_view_goto(self, mock_graph_with_steps):
        """Test jumping to specific step."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        result = seq.goto(2)

        assert result is seq
        assert seq.current_index == 2
        assert seq.selected_id == "step-3"

    def test_sequence_view_goto_invalid(self, mock_graph_with_steps):
        """Test goto with invalid index does nothing."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        seq.goto(99)

        assert seq.current_index == 0  # Unchanged

    def test_sequence_view_render_progress(self, mock_graph_with_steps):
        """Test progress rendering."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        seq.advance()  # Complete step 1

        progress = seq.render_progress()

        assert "## Workflow Progress [1/3]" in progress
        assert "[x]" in progress  # Completed step
        assert "[>]" in progress  # Current step
        assert "[ ]" in progress  # Pending step
        assert "← current" in progress

    def test_sequence_view_repr(self, mock_graph_with_steps):
        """Test string representation."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        repr_str = repr(seq)

        assert "SequenceView" in repr_str
        assert "progress=0/3" in repr_str


class TestSequenceViewPersistence:
    """Tests for SequenceView state persistence."""

    def test_sequence_view_persists_state(self, mock_graph_with_steps):
        """Test that state is persisted to node tags."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent)

        seq.advance()
        seq.advance()

        # State is persisted to node.tags, not view.tags
        assert parent.tags["_seq_index"] == 2
        assert set(parent.tags["_seq_completed"]) == {0, 1}

    def test_sequence_view_restores_from_tags(self, mock_graph_with_steps):
        """Test that state is restored from tags."""
        parent = mock_graph_with_steps.get_node("parent")

        # Create first view and advance
        seq1 = SequenceView(parent)
        seq1.advance()
        seq1.advance()

        # Create second view with same node (simulates session restore)
        seq2 = SequenceView(parent)

        # State should be restored from tags
        assert seq2.current_index == 2
        assert seq2.completed_steps == {0, 1}


# =============================================================================
# LoopView Basic Tests
# =============================================================================


class TestLoopViewBasic:
    """Tests for LoopView basic functionality."""

    def test_loop_view_creation(self, mock_graph_with_loop_child):
        """Test creating a LoopView."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=5)

        assert loop.iteration == 1
        assert loop.max_iterations == 5
        assert loop.state == {}
        assert not loop.is_done
        assert loop.iterations_remaining == 5

    def test_loop_view_unlimited(self, mock_graph_with_loop_child):
        """Test creating unlimited LoopView."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        assert loop.max_iterations is None
        assert loop.iterations_remaining is None
        assert not loop.is_done

    def test_loop_view_iterate(self, mock_graph_with_loop_child):
        """Test iterating with state updates."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=5)

        result = loop.iterate(feedback="Add error handling")

        assert result is loop
        assert loop.iteration == 2
        assert loop.state == {"feedback": "Add error handling"}
        assert loop.iterations_remaining == 4

    def test_loop_view_iterate_accumulates_state(self, mock_graph_with_loop_child):
        """Test that iterate accumulates state across iterations."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        loop.iterate(feedback="First issue")
        loop.iterate(feedback="Second issue", approved=False)
        loop.iterate(approved=True)

        assert loop.iteration == 4
        assert loop.state == {"feedback": "Second issue", "approved": True}

    def test_loop_view_update_state(self, mock_graph_with_loop_child):
        """Test updating state without incrementing iteration."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        loop.update_state(note="Initial note")

        assert loop.iteration == 1  # Didn't increment
        assert loop.state == {"note": "Initial note"}

    def test_loop_view_done(self, mock_graph_with_loop_child):
        """Test marking loop as done."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=5)

        result = loop.done()

        assert result is loop
        assert loop.is_done
        assert loop.iteration == 1  # Didn't change iteration

    def test_loop_view_max_iterations_reached(self, mock_graph_with_loop_child):
        """Test that is_done returns True when max iterations exceeded."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=3)

        loop.iterate()  # 1 -> 2
        loop.iterate()  # 2 -> 3
        loop.iterate()  # 3 -> 4

        assert loop.is_done  # 4 > max_iterations(3)

    def test_loop_view_iterate_when_done(self, mock_graph_with_loop_child):
        """Test that iterate is no-op when done."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        loop.done()
        loop.iterate(feedback="ignored")

        assert loop.iteration == 1
        assert loop.state == {}  # State not updated

    def test_loop_view_reset(self, mock_graph_with_loop_child):
        """Test resetting loop to initial state."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=5)

        loop.iterate(note="something")
        loop.iterate(note="more")
        loop.done()

        result = loop.reset()

        assert result is loop
        assert loop.iteration == 1
        assert loop.state == {}
        assert not loop.is_done

    def test_loop_view_render_header(self, mock_graph_with_loop_child):
        """Test header rendering."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=5)
        loop.iterate()

        header = loop.render_header()

        assert "## Review" in header
        assert "iteration 2/5" in header

    def test_loop_view_render_header_unlimited(self, mock_graph_with_loop_child):
        """Test header rendering with unlimited iterations."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        header = loop.render_header()

        assert "iteration 1" in header
        assert "/" not in header

    def test_loop_view_render_state(self, mock_graph_with_loop_child):
        """Test state rendering."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        loop.iterate(feedback="Fix issues", approved=False)

        state_str = loop.render_state()

        assert "**State:**" in state_str
        assert "feedback" in state_str
        assert "approved" in state_str

    def test_loop_view_render_state_empty(self, mock_graph_with_loop_child):
        """Test state rendering with empty state."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        state_str = loop.render_state()

        assert state_str == ""

    def test_loop_view_repr(self, mock_graph_with_loop_child):
        """Test string representation."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=5)

        repr_str = repr(loop)

        assert "LoopView" in repr_str
        assert "iteration=1/5" in repr_str


class TestLoopViewPersistence:
    """Tests for LoopView state persistence."""

    def test_loop_view_persists_state(self, mock_graph_with_loop_child):
        """Test that state is persisted to node tags."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child)

        loop.iterate(note="test")
        loop.iterate(result="ok")

        # State is persisted to node.tags, not view.tags
        assert child.tags["_loop_iteration"] == 3
        assert child.tags["_loop_state"] == {"note": "test", "result": "ok"}
        assert child.tags["_loop_done"] is False

    def test_loop_view_restores_from_tags(self, mock_graph_with_loop_child):
        """Test that state is restored from tags."""
        child = mock_graph_with_loop_child.get_node("review")

        # Create first view and iterate
        loop1 = LoopView(child)
        loop1.iterate(note="test")
        loop1.done()

        # Create second view with same node (simulates session restore)
        loop2 = LoopView(child)

        assert loop2.iteration == 2
        assert loop2.state == {"note": "test"}
        assert loop2.is_done


# =============================================================================
# StateView Basic Tests
# =============================================================================


class TestStateViewBasic:
    """Tests for StateView basic functionality."""

    def test_state_view_creation(self, mock_graph_with_states):
        """Test creating a StateView."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done", "idle"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        assert fsm.current_state == "idle"
        assert fsm.selected_id == "idle"
        assert fsm.state_history == []
        assert fsm.all_states == ["idle", "working", "done"]

    def test_state_view_default_initial(self, mock_graph_with_states):
        """Test that first state is default initial."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working"}

        fsm = StateView(parent, states=states)

        assert fsm.current_state == "idle"

    def test_state_view_valid_transitions(self, mock_graph_with_states):
        """Test getting valid transitions."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done", "idle"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        assert fsm.valid_transitions == ["working"]
        assert fsm.can_transition("working")
        assert not fsm.can_transition("done")

    def test_state_view_transition(self, mock_graph_with_states):
        """Test transitioning to new state."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done", "idle"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        result = fsm.transition("working")

        assert result is fsm
        assert fsm.current_state == "working"
        assert fsm.selected_id == "working"
        assert fsm.state_history == ["idle"]
        assert fsm.valid_transitions == ["done", "idle"]

    def test_state_view_invalid_transition(self, mock_graph_with_states):
        """Test that invalid transition raises error."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done", "idle"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        with pytest.raises(ValueError, match="Cannot transition"):
            fsm.transition("done")

    def test_state_view_unknown_state(self, mock_graph_with_states):
        """Test that unknown state raises error."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working"}
        transitions = {"idle": ["working", "invalid"]}

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        with pytest.raises(ValueError, match="Unknown state"):
            fsm.transition("invalid")

    def test_state_view_force_transition(self, mock_graph_with_states):
        """Test force transition bypasses rules."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {"idle": [], "working": [], "done": []}  # No transitions allowed

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        result = fsm.force_transition("done")

        assert result is fsm
        assert fsm.current_state == "done"
        assert fsm.state_history == ["idle"]

    def test_state_view_reset(self, mock_graph_with_states):
        """Test resetting to initial state."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        fsm.transition("working")
        fsm.transition("done")

        result = fsm.reset()

        assert result is fsm
        assert fsm.current_state == "idle"
        assert fsm.state_history == []

    def test_state_view_render_header(self, mock_graph_with_states):
        """Test header rendering."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done", "idle"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        header = fsm.render_header()

        assert "## Task: idle" in header
        assert "→ [working]" in header

    def test_state_view_render_header_terminal(self, mock_graph_with_states):
        """Test header rendering for terminal state."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "done": "done"}
        transitions = {"idle": ["done"], "done": []}

        fsm = StateView(parent, states=states, transitions=transitions, initial="done")

        header = fsm.render_header()

        assert "(terminal)" in header

    def test_state_view_repr(self, mock_graph_with_states):
        """Test string representation."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working"}
        transitions = {"idle": ["working"], "working": []}

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")
        fsm.transition("working")

        repr_str = repr(fsm)

        assert "StateView" in repr_str
        assert "state='working'" in repr_str
        assert "history=1" in repr_str


class TestStateViewPersistence:
    """Tests for StateView state persistence."""

    def test_state_view_persists_state(self, mock_graph_with_states):
        """Test that state is persisted to node tags."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done"],
            "done": [],
        }

        fsm = StateView(parent, states=states, transitions=transitions, initial="idle")

        fsm.transition("working")
        fsm.transition("done")

        # State is persisted to node.tags, not view.tags
        assert parent.tags["_state_current"] == "done"
        assert parent.tags["_state_history"] == ["idle", "working"]

    def test_state_view_restores_from_tags(self, mock_graph_with_states):
        """Test that state is restored from tags."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle", "working": "working", "done": "done"}
        transitions = {
            "idle": ["working"],
            "working": ["done"],
            "done": [],
        }

        # Create first view and transition
        fsm1 = StateView(parent, states=states, transitions=transitions, initial="idle")
        fsm1.transition("working")

        # Create second view with same node (simulates session restore)
        fsm2 = StateView(parent, states=states, transitions=transitions, initial="idle")

        assert fsm2.current_state == "working"
        assert fsm2.state_history == ["idle"]


# =============================================================================
# Apply Selection Tests
# =============================================================================


class TestSequenceViewApplySelection:
    """Tests for SequenceView selection behavior."""

    def test_apply_selection_hides_non_current(self, mock_graph_with_steps):
        """Test that non-current steps are hidden."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent, expand=Expansion.ALL)

        views = {
            "parent": seq,
            "step-1": NodeView(mock_graph_with_steps.get_node("step-1")),
            "step-2": NodeView(mock_graph_with_steps.get_node("step-2")),
            "step-3": NodeView(mock_graph_with_steps.get_node("step-3")),
        }

        seq.apply_selection(views)

        assert views["step-1"].hide is False  # Current
        assert views["step-2"].hide is True
        assert views["step-3"].hide is True

    def test_apply_selection_after_advance(self, mock_graph_with_steps):
        """Test selection updates after advancing."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent, expand=Expansion.ALL)

        views = {
            "parent": seq,
            "step-1": NodeView(mock_graph_with_steps.get_node("step-1")),
            "step-2": NodeView(mock_graph_with_steps.get_node("step-2")),
            "step-3": NodeView(mock_graph_with_steps.get_node("step-3")),
        }

        seq.advance()
        seq.apply_selection(views)

        assert views["step-1"].hide is True
        assert views["step-2"].hide is False  # Now current
        assert views["step-3"].hide is True


# =============================================================================
# Projection Engine Integration Tests
# =============================================================================


class TestProjectionEngineIntegration:
    """Tests for progression views integration with ProjectionEngine."""

    def test_sequence_view_in_projection(
        self, mock_graph_with_steps, projection_engine
    ):
        """Test that SequenceView filtering works through projection engine."""
        parent = mock_graph_with_steps.get_node("parent")
        seq = SequenceView(parent, expand=Expansion.ALL)

        views = {
            "parent": seq,
            "step-1": NodeView(mock_graph_with_steps.get_node("step-1"), expand=Expansion.ALL),
            "step-2": NodeView(mock_graph_with_steps.get_node("step-2"), expand=Expansion.ALL),
            "step-3": NodeView(mock_graph_with_steps.get_node("step-3"), expand=Expansion.ALL),
        }

        projection = projection_engine.build(
            context_graph=mock_graph_with_steps,
            views=views,
        )

        # Only parent and current step should be rendered
        rendered_ids = [section.source_id for section in projection.sections]
        assert "parent" in rendered_ids
        assert "step-1" in rendered_ids  # Current step
        assert "step-2" not in rendered_ids
        assert "step-3" not in rendered_ids


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sequence_view_empty_children(self, mock_graph_with_steps):
        """Test SequenceView with no children."""
        parent = mock_graph_with_steps.get_node("parent")
        parent.children_ids = set()
        parent.child_order = []

        seq = SequenceView(parent)

        assert seq.total_steps == 0
        assert seq.is_complete  # Empty sequence is complete
        assert seq.progress == "0/0"

    def test_loop_view_zero_max_iterations(self, mock_graph_with_loop_child):
        """Test LoopView with max_iterations=0."""
        child = mock_graph_with_loop_child.get_node("review")
        loop = LoopView(child, max_iterations=0)

        # Iteration 1 > max_iterations(0), so is_done immediately
        assert loop.is_done

    def test_state_view_empty_states(self, mock_graph_with_states):
        """Test StateView with no states."""
        parent = mock_graph_with_states.get_node("parent")

        fsm = StateView(parent, states={}, transitions={})

        assert fsm.current_state == ""
        assert fsm.all_states == []
        assert fsm.valid_transitions == []

    def test_state_view_can_transition_invalid_state(self, mock_graph_with_states):
        """Test can_transition with non-existent state."""
        parent = mock_graph_with_states.get_node("parent")
        states = {"idle": "idle"}
        transitions = {"idle": []}

        fsm = StateView(parent, states=states, transitions=transitions)

        assert not fsm.can_transition("nonexistent")
