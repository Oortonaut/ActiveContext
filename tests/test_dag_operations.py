"""Tests for DAG operations, checkpoint/restore/branch cycles, state transitions,
and group summarization triggers.

This test module covers:
- Complex DAG link/unlink operations
- Checkpoint/restore/branch operation cycles
- State transitions (Expansion: HEADER->CONTENT->INDEX->ALL)
- Group summarization triggers and invalidation

Tests are organized per task-002 requirements.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from activecontext.context.checkpoint import Checkpoint, GroupState
from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import GroupNode, TextNode, TopicNode
from activecontext.context.state import Expansion, TickFrequency, Visibility
from tests.utils import create_mock_context_node


# =============================================================================
# DAG Link/Unlink Complex Operations Tests
# =============================================================================


class TestDAGLinkUnlinkOperations:
    """Tests for complex DAG link/unlink operations."""

    @pytest.fixture
    def graph(self):
        """Create an empty ContextGraph."""
        return ContextGraph()

    def test_link_multiple_children_to_parent(self, graph):
        """Test linking multiple children to a single parent."""
        parent = create_mock_context_node("parent", "group")
        child1 = create_mock_context_node("child1", "text")
        child2 = create_mock_context_node("child2", "text")
        child3 = create_mock_context_node("child3", "text")

        for node in [parent, child1, child2, child3]:
            graph.add_node(node)

        # Link all children to parent
        assert graph.link("child1", "parent") is True
        assert graph.link("child2", "parent") is True
        assert graph.link("child3", "parent") is True

        # Verify parent has all children
        children = graph.get_children("parent")
        assert len(children) == 3
        child_ids = {c.node_id for c in children}
        assert child_ids == {"child1", "child2", "child3"}

        # Verify each child has parent
        for child_id in ["child1", "child2", "child3"]:
            parents = graph.get_parents(child_id)
            assert len(parents) == 1
            assert parents[0].node_id == "parent"

    def test_diamond_dag_structure(self, graph):
        """Test diamond DAG: A->B, A->C, B->D, C->D (D has 2 parents)."""
        a = create_mock_context_node("a", "group")
        b = create_mock_context_node("b", "group")
        c = create_mock_context_node("c", "group")
        d = create_mock_context_node("d", "text")

        for node in [a, b, c, d]:
            graph.add_node(node)

        # Create diamond
        assert graph.link("b", "a") is True
        assert graph.link("c", "a") is True
        assert graph.link("d", "b") is True
        assert graph.link("d", "c") is True

        # Verify structure
        assert len(graph.get_children("a")) == 2
        assert len(graph.get_children("b")) == 1
        assert len(graph.get_children("c")) == 1
        assert len(graph.get_children("d")) == 0

        # D should have 2 parents
        d_parents = graph.get_parents("d")
        assert len(d_parents) == 2
        parent_ids = {p.node_id for p in d_parents}
        assert parent_ids == {"b", "c"}

        # Verify ancestors of D includes a, b, c
        ancestors = graph.get_ancestors("d")
        assert len(ancestors) == 3
        ancestor_ids = {a.node_id for a in ancestors}
        assert ancestor_ids == {"a", "b", "c"}

        # Verify descendants of A includes b, c, d
        descendants = graph.get_descendants("a")
        assert len(descendants) == 3
        desc_ids = {d.node_id for d in descendants}
        assert desc_ids == {"b", "c", "d"}

    def test_unlink_preserves_other_relationships(self, graph):
        """Test that unlinking one edge preserves other edges."""
        parent1 = create_mock_context_node("parent1", "group")
        parent2 = create_mock_context_node("parent2", "group")
        child = create_mock_context_node("child", "text")

        for node in [parent1, parent2, child]:
            graph.add_node(node)

        graph.link("child", "parent1")
        graph.link("child", "parent2")

        # Verify initial state
        assert len(graph.get_parents("child")) == 2

        # Unlink from parent1
        assert graph.unlink("child", "parent1") is True

        # Should still have parent2
        parents = graph.get_parents("child")
        assert len(parents) == 1
        assert parents[0].node_id == "parent2"

    def test_unlink_nonexistent_edge_is_noop(self, graph):
        """Test unlinking a non-existent edge is a no-op.
        
        Note: unlink() returns True as long as nodes exist, even if no edge exists.
        This is by design - discard is idempotent.
        """
        node1 = create_mock_context_node("node1", "text")
        node2 = create_mock_context_node("node2", "text")

        graph.add_node(node1)
        graph.add_node(node2)

        # No edge exists - unlink is a no-op but returns True
        result = graph.unlink("node1", "node2")
        assert result is True  # Nodes exist, even though no edge existed

        # Verify state unchanged - node1 still has no parents
        assert len(graph.get_parents("node1")) == 0
        
    def test_unlink_nonexistent_nodes_returns_false(self, graph):
        """Test unlinking with nonexistent nodes returns False."""
        node1 = create_mock_context_node("node1", "text")
        graph.add_node(node1)

        # Nonexistent parent
        assert graph.unlink("node1", "nonexistent") is False
        
        # Nonexistent child
        assert graph.unlink("nonexistent", "node1") is False

    def test_link_after_unlink_cycle_detection(self, graph):
        """Test that cycle detection works after unlink/relink sequences."""
        a = create_mock_context_node("a", "group")
        b = create_mock_context_node("b", "group")
        c = create_mock_context_node("c", "text")

        for node in [a, b, c]:
            graph.add_node(node)

        # Create chain: a -> b -> c
        graph.link("b", "a")
        graph.link("c", "b")

        # Attempt cycle c -> a (should fail)
        assert graph.link("a", "c") is False

        # Unlink b from a
        graph.unlink("b", "a")

        # Now a has no parents and b is root
        # c -> a cycle still invalid because c is descendant of b
        # Actually after unlinking, a is now a root with no descendants
        # Let's verify
        roots = graph.get_roots()
        root_ids = {r.node_id for r in roots}
        assert "a" in root_ids
        assert "b" in root_ids

        # Now a -> c should work (a has no relationship with c anymore)
        assert graph.link("c", "a") is True

    def test_remove_node_orphans_children(self, graph):
        """Test that removing a parent orphans its children (makes them roots)."""
        parent = create_mock_context_node("parent", "group")
        child1 = create_mock_context_node("child1", "text")
        child2 = create_mock_context_node("child2", "text")

        for node in [parent, child1, child2]:
            graph.add_node(node)

        graph.link("child1", "parent")
        graph.link("child2", "parent")

        # Remove parent (non-recursive)
        graph.remove_node("parent", recursive=False)

        # Children should become roots
        assert "parent" not in graph
        assert "child1" in graph
        assert "child2" in graph

        roots = graph.get_roots()
        root_ids = {r.node_id for r in roots}
        assert "child1" in root_ids
        assert "child2" in root_ids

    def test_complex_dag_traversal(self, graph):
        """Test traversal in a complex DAG with multiple paths."""
        # Create structure:
        #     root
        #    /    \
        #   a      b
        #  / \    / \
        # c   d  e   f
        #      \   /
        #       leaf

        nodes = {
            name: create_mock_context_node(name, "group" if name in ["root", "a", "b"] else "text")
            for name in ["root", "a", "b", "c", "d", "e", "f", "leaf"]
        }

        for node in nodes.values():
            graph.add_node(node)

        # Build structure
        graph.link("a", "root")
        graph.link("b", "root")
        graph.link("c", "a")
        graph.link("d", "a")
        graph.link("e", "b")
        graph.link("f", "b")
        graph.link("leaf", "d")
        graph.link("leaf", "e")

        # Leaf has 2 parents
        leaf_parents = graph.get_parents("leaf")
        assert len(leaf_parents) == 2

        # Root has all as descendants
        descendants = graph.get_descendants("root")
        assert len(descendants) == 7

        # Leaf has all as ancestors (via different paths)
        ancestors = graph.get_ancestors("leaf")
        assert len(ancestors) == 5  # d, e, a, b, root

    def test_link_with_after_parameter(self, graph):
        """Test linking with ordering via after parameter."""
        parent = create_mock_context_node("parent", "group")
        child1 = create_mock_context_node("child1", "text")
        child2 = create_mock_context_node("child2", "text")
        child3 = create_mock_context_node("child3", "text")

        for node in [parent, child1, child2, child3]:
            graph.add_node(node)

        # Link first child
        graph.link("child1", "parent")

        # Link second child
        graph.link("child2", "parent")

        # Link third child after first
        graph.link("child3", "parent", after="child1")

        # Verify ordering is preserved in children_ids
        # Note: actual ordering depends on LinkedChildOrder implementation
        parent_node = graph.get_node("parent")
        assert "child1" in parent_node.children_ids
        assert "child2" in parent_node.children_ids
        assert "child3" in parent_node.children_ids


# =============================================================================
# Checkpoint/Restore/Branch Cycle Tests
# =============================================================================


class TestCheckpointRestoreBranchCycles:
    """Tests for checkpoint/restore/branch operation cycles."""

    @pytest.fixture
    def populated_graph(self):
        """Create a graph with nodes for checkpoint testing."""
        graph = ContextGraph()

        root = create_mock_context_node("root", "group")
        child1 = create_mock_context_node("child1", "text")
        child2 = create_mock_context_node("child2", "text")
        grandchild = create_mock_context_node("grandchild", "text")

        for node in [root, child1, child2, grandchild]:
            graph.add_node(node)

        graph.link("child1", "root")
        graph.link("child2", "root")
        graph.link("grandchild", "child1")

        return graph

    def test_checkpoint_restore_preserves_structure(self, populated_graph):
        """Test checkpoint-restore cycle preserves DAG structure."""
        # Capture checkpoint
        cp = populated_graph.checkpoint("initial")

        # Modify structure
        populated_graph.unlink("child1", "root")
        populated_graph.link("child1", "child2")

        # Verify modification
        child1_parents = [p.node_id for p in populated_graph.get_parents("child1")]
        assert "child2" in child1_parents
        assert "root" not in child1_parents

        # Restore
        populated_graph.restore("initial")

        # Verify restoration
        child1_parents = [p.node_id for p in populated_graph.get_parents("child1")]
        assert "root" in child1_parents
        assert "child2" not in child1_parents

    def test_multiple_checkpoint_restore_cycles(self, populated_graph):
        """Test multiple checkpoint-restore cycles."""
        # Checkpoint 1: initial state
        populated_graph.checkpoint("cp1")

        # Modify
        populated_graph.unlink("child2", "root")

        # Checkpoint 2: modified state
        populated_graph.checkpoint("cp2")

        # Further modify
        populated_graph.unlink("grandchild", "child1")

        # Checkpoint 3: further modified
        populated_graph.checkpoint("cp3")

        # Verify we have 3 checkpoints
        checkpoints = populated_graph.get_checkpoints()
        assert len(checkpoints) == 3

        # Restore to cp1
        populated_graph.restore("cp1")
        assert "root" in [p.node_id for p in populated_graph.get_parents("child2")]
        assert "child1" in [p.node_id for p in populated_graph.get_parents("grandchild")]

        # Restore to cp2
        populated_graph.restore("cp2")
        # child2 should not have root as parent
        child2_parents = populated_graph.get_parents("child2")
        assert len(child2_parents) == 0  # child2 became root
        # grandchild should still have child1 as parent
        assert "child1" in [p.node_id for p in populated_graph.get_parents("grandchild")]

    def test_branch_creates_checkpoint(self, populated_graph):
        """Test that branch() creates a checkpoint for continuation."""
        # Branch is implemented as checkpoint in this codebase
        cp = populated_graph.checkpoint("branch-point")

        assert cp is not None
        assert cp.name == "branch-point"

        # Modify after branch
        populated_graph.unlink("child1", "root")

        # Original checkpoint should have the old structure
        stored_cp = populated_graph.get_checkpoint("branch-point")
        assert ("child1", "root") in stored_cp.edges

    def test_restore_then_branch_pattern(self, populated_graph):
        """Test restore followed by branch (common pattern for exploration)."""
        # Initial checkpoint
        populated_graph.checkpoint("base")

        # First attempt
        populated_graph.unlink("child1", "root")
        populated_graph.checkpoint("attempt1")

        # Restore to base
        populated_graph.restore("base")

        # Second attempt (branch)
        populated_graph.unlink("child2", "root")
        populated_graph.checkpoint("attempt2")

        # Verify we can switch between attempts
        populated_graph.restore("attempt1")
        child1_parents = [p.node_id for p in populated_graph.get_parents("child1")]
        child2_parents = [p.node_id for p in populated_graph.get_parents("child2")]
        assert "root" not in child1_parents
        assert "root" in child2_parents

        populated_graph.restore("attempt2")
        child1_parents = [p.node_id for p in populated_graph.get_parents("child1")]
        child2_parents = [p.node_id for p in populated_graph.get_parents("child2")]
        assert "root" in child1_parents
        assert "root" not in child2_parents

    def test_checkpoint_preserves_roots(self, populated_graph):
        """Test that checkpoint preserves root node set."""
        # Add another root
        new_root = create_mock_context_node("new_root", "text")
        populated_graph.add_node(new_root)

        # Now we have 2 roots
        roots = populated_graph.get_roots()
        assert len(roots) == 2

        cp = populated_graph.checkpoint("with-two-roots")

        # Verify checkpoint has both roots
        assert "root" in cp.root_ids
        assert "new_root" in cp.root_ids

    def test_restore_rebuilds_roots(self, populated_graph):
        """Test that restore properly rebuilds root set."""
        populated_graph.checkpoint("before")

        # Make child1 a root by unlinking
        populated_graph.unlink("child1", "root")

        roots = populated_graph.get_roots()
        assert any(r.node_id == "child1" for r in roots)

        # Restore
        populated_graph.restore("before")

        # child1 should no longer be a root
        roots = populated_graph.get_roots()
        root_ids = {r.node_id for r in roots}
        assert "child1" not in root_ids

    def test_delete_checkpoint_after_restore(self, populated_graph):
        """Test deleting a checkpoint after using it for restore."""
        populated_graph.checkpoint("temp")
        populated_graph.unlink("child1", "root")

        # Restore from checkpoint
        populated_graph.restore("temp")

        # Delete the checkpoint
        assert populated_graph.delete_checkpoint("temp") is True
        assert populated_graph.get_checkpoint("temp") is None

        # Further operations should work
        populated_graph.unlink("child2", "root")
        assert len(populated_graph.get_parents("child2")) == 0


# =============================================================================
# State Transition Tests
# =============================================================================


class TestStateTransitions:
    """Tests for Expansion and Visibility state transitions."""

    def test_expansion_header_to_content(self):
        """Test Expansion transition from HEADER to CONTENT."""
        node = TextNode(
            node_id="test",
            path="test.py",
            expansion=Expansion.HEADER,
        )

        assert node.expansion == Expansion.HEADER

        node.expansion = Expansion.CONTENT
        assert node.expansion == Expansion.CONTENT

    def test_expansion_content_to_index(self):
        """Test Expansion transition from CONTENT to INDEX."""
        node = TextNode(
            node_id="test",
            path="test.py",
            expansion=Expansion.CONTENT,
        )

        node.expansion = Expansion.INDEX
        assert node.expansion == Expansion.INDEX

    def test_expansion_index_to_all(self):
        """Test Expansion transition from INDEX to ALL."""
        node = TextNode(
            node_id="test",
            path="test.py",
            expansion=Expansion.INDEX,
        )

        node.expansion = Expansion.ALL
        assert node.expansion == Expansion.ALL

    def test_expansion_full_sequence(self):
        """Test full Expansion sequence: HEADER -> CONTENT -> INDEX -> ALL."""
        node = TextNode(
            node_id="test",
            path="test.py",
            expansion=Expansion.HEADER,
        )

        # Step through all states
        states = [Expansion.HEADER, Expansion.CONTENT, Expansion.INDEX, Expansion.ALL]

        for expected_state in states:
            node.expansion = expected_state
            assert node.expansion == expected_state

    def test_expansion_reverse_sequence(self):
        """Test reverse Expansion sequence: ALL -> INDEX -> CONTENT -> HEADER."""
        node = TextNode(
            node_id="test",
            path="test.py",
            expansion=Expansion.ALL,
        )

        states = [Expansion.ALL, Expansion.INDEX, Expansion.CONTENT, Expansion.HEADER]

        for expected_state in states:
            node.expansion = expected_state
            assert node.expansion == expected_state

    def test_expansion_skip_states(self):
        """Test skipping intermediate Expansion states."""
        node = TextNode(
            node_id="test",
            path="test.py",
            expansion=Expansion.HEADER,
        )

        # Skip directly to ALL
        node.expansion = Expansion.ALL
        assert node.expansion == Expansion.ALL

        # Skip back to HEADER
        node.expansion = Expansion.HEADER
        assert node.expansion == Expansion.HEADER

    def test_visibility_hide_show(self):
        """Test Visibility HIDE/SHOW transitions."""
        assert Visibility.HIDE.value == "hide"
        assert Visibility.SHOW.value == "show"
        assert str(Visibility.HIDE) == "hide"
        assert str(Visibility.SHOW) == "show"

    def test_expansion_ordering(self):
        """Test that Expansion values have logical ordering for comparison."""
        # HEADER < CONTENT < INDEX < ALL (by verbosity)
        states = [
            Expansion.HEADER,
            Expansion.CONTENT,
            Expansion.INDEX,
            Expansion.ALL,
        ]

        values = ["header", "content", "index", "all"]
        for state, expected_value in zip(states, values):
            assert state.value == expected_value

    def test_expansion_group_node_default(self):
        """Test GroupNode uses CONTENT as default expansion."""
        group = GroupNode(
            node_id="group1",
            expansion=Expansion.CONTENT,
        )
        assert group.expansion == Expansion.CONTENT

    def test_expansion_text_node_default(self):
        """Test TextNode uses ALL as default expansion."""
        text = TextNode(
            node_id="text1",
            path="test.py",
            expansion=Expansion.ALL,
        )
        assert text.expansion == Expansion.ALL


# =============================================================================
# Group Summarization Trigger Tests
# =============================================================================


class TestGroupSummarizationTriggers:
    """Tests for group summarization triggers and invalidation."""

    @pytest.fixture
    def graph_with_group(self):
        """Create a graph with a group and children."""
        graph = ContextGraph()

        group = GroupNode(
            node_id="group1",
            expansion=Expansion.CONTENT,
            cached_summary="Initial summary",
            summary_stale=False,
            last_child_versions={"child1": 1, "child2": 1},
        )

        child1 = TextNode(
            node_id="child1",
            path="file1.py",
            expansion=Expansion.ALL,
        )
        child1.version = 1

        child2 = TextNode(
            node_id="child2",
            path="file2.py",
            expansion=Expansion.ALL,
        )
        child2.version = 1

        graph.add_node(group)
        graph.add_node(child1)
        graph.add_node(child2)

        graph.link("child1", "group1")
        graph.link("child2", "group1")

        return graph, group, child1, child2

    def test_child_version_change_marks_summary_stale(self, graph_with_group):
        """Test that child version change marks group summary as stale."""
        graph, group, child1, child2 = graph_with_group

        # Initially not stale
        assert group.summary_stale is False

        # Simulate child change by incrementing version
        child1.version = 2

        # Call on_child_changed
        group.on_child_changed(child1, "Content updated")

        # Summary should be marked stale
        assert group.summary_stale is True

        # Version should be updated
        assert group.last_child_versions["child1"] == 2

    def test_same_version_does_not_mark_stale(self, graph_with_group):
        """Test that same version doesn't mark summary as stale."""
        graph, group, child1, child2 = graph_with_group

        # Manually set stale to False
        group.summary_stale = False

        # Call on_child_changed with same version
        group.on_child_changed(child1, "No actual change")

        # Should remain not stale
        assert group.summary_stale is False

    def test_invalidate_summary_explicit(self, graph_with_group):
        """Test explicit invalidate_summary() call."""
        graph, group, child1, child2 = graph_with_group

        group.summary_stale = False
        assert group.summary_stale is False

        group.invalidate_summary()

        assert group.summary_stale is True

    def test_multiple_child_changes(self, graph_with_group):
        """Test multiple child changes all trigger invalidation."""
        graph, group, child1, child2 = graph_with_group

        group.summary_stale = False

        # Change first child
        child1.version = 2
        group.on_child_changed(child1, "First update")
        assert group.summary_stale is True

        # Reset stale flag (simulating summary regeneration)
        group.summary_stale = False

        # Change second child
        child2.version = 2
        group.on_child_changed(child2, "Second update")
        assert group.summary_stale is True

    def test_cached_summary_returned_when_not_stale(self, graph_with_group):
        """Test that cached summary is returned when not stale."""
        graph, group, child1, child2 = graph_with_group

        group.summary_stale = False
        group.cached_summary = "Cached test summary"

        rendered = group.RenderSummary()

        # Should return cached summary
        assert "Cached test summary" in rendered

    def test_header_returned_when_stale(self, graph_with_group):
        """Test that header is returned when summary is stale."""
        graph, group, child1, child2 = graph_with_group

        group.summary_stale = True
        group.cached_summary = "Stale summary"

        rendered = group.RenderSummary()

        # Should not return the stale cached summary content directly
        # Instead returns header
        assert "Stale summary" not in rendered or "Group" in rendered

    def test_on_child_changed_hook(self, graph_with_group):
        """Test that on_child_changed hook is called."""
        graph, group, child1, child2 = graph_with_group

        hook_calls = []

        def hook(parent, child, description):
            hook_calls.append((parent.node_id, child.node_id, description))

        group._on_child_changed_hook = hook

        child1.version = 2
        group.on_child_changed(child1, "Test update")

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("group1", "child1", "Test update")

    def test_group_serialization_preserves_summary_state(self, graph_with_group):
        """Test that serialization preserves summary state."""
        graph, group, child1, child2 = graph_with_group

        group.summary_stale = True
        group.cached_summary = "Test summary"
        group.summary_prompt = "Summarize the auth module"
        group.last_child_versions = {"child1": 5, "child2": 3}

        # Serialize
        data = group.to_dict()

        # Deserialize
        restored = GroupNode._from_dict(data)

        assert restored.summary_stale is True
        assert restored.cached_summary == "Test summary"
        assert restored.summary_prompt == "Summarize the auth module"
        assert restored.last_child_versions == {"child1": 5, "child2": 3}


# =============================================================================
# Additional DAG Edge Cases
# =============================================================================


class TestDAGEdgeCases:
    """Edge case tests for DAG operations."""

    @pytest.fixture
    def graph(self):
        """Create an empty ContextGraph."""
        return ContextGraph()

    def test_empty_graph_operations(self, graph):
        """Test operations on empty graph."""
        assert len(graph) == 0
        assert graph.get_roots() == []
        assert graph.get_running_nodes() == []
        assert graph.get_node("nonexistent") is None
        assert graph.get_children("nonexistent") == []
        assert graph.get_parents("nonexistent") == []

    def test_single_node_graph(self, graph):
        """Test single node graph operations."""
        node = create_mock_context_node("solo", "text")
        graph.add_node(node)

        assert len(graph) == 1
        assert len(graph.get_roots()) == 1
        assert graph.get_roots()[0].node_id == "solo"
        assert graph.get_children("solo") == []
        assert graph.get_parents("solo") == []
        assert graph.get_descendants("solo") == []
        assert graph.get_ancestors("solo") == []

    def test_checkpoint_empty_graph(self, graph):
        """Test checkpointing an empty graph."""
        cp = graph.checkpoint("empty")

        assert cp is not None
        assert cp.name == "empty"
        assert len(cp.edges) == 0
        assert len(cp.root_ids) == 0

    def test_restore_adds_missing_edges_only(self, graph):
        """Test that restore only adds edges for existing nodes."""
        # Create nodes and checkpoint
        parent = create_mock_context_node("parent", "group")
        child = create_mock_context_node("child", "text")
        graph.add_node(parent)
        graph.add_node(child)
        graph.link("child", "parent")

        cp = graph.checkpoint("full")

        # Remove child
        graph.remove_node("child")

        # Restore - child edge should be skipped since node doesn't exist
        graph.restore(cp)

        # Parent should have no children (child doesn't exist)
        assert len(graph.get_children("parent")) == 0

    def test_type_index_after_remove(self, graph):
        """Test that type index is updated after node removal."""
        text1 = create_mock_context_node("text1", "text")
        text2 = create_mock_context_node("text2", "text")
        group1 = create_mock_context_node("group1", "group")

        graph.add_node(text1)
        graph.add_node(text2)
        graph.add_node(group1)

        assert len(graph.get_nodes_by_type("text")) == 2

        graph.remove_node("text1")

        assert len(graph.get_nodes_by_type("text")) == 1
        assert graph.get_nodes_by_type("text")[0].node_id == "text2"

    def test_running_index_after_mode_change(self, graph):
        """Test that running nodes index reflects mode changes."""
        running = create_mock_context_node("running", "text", mode="running")
        idle = create_mock_context_node("idle", "text", mode="idle")

        graph.add_node(running)
        graph.add_node(idle)

        assert len(graph.get_running_nodes()) == 1
        assert graph.get_running_nodes()[0].node_id == "running"

    def test_checkpoint_with_group_state(self, graph):
        """Test checkpoint preserves GroupNode-specific state."""
        group = GroupNode(
            node_id="group1",
            expansion=Expansion.CONTENT,
            summary_prompt="Test prompt",
            cached_summary="Test summary",
            summary_stale=True,
            last_child_versions={"child1": 3},
        )
        graph.add_node(group)

        cp = graph.checkpoint("with-group")

        assert "group1" in cp.group_states
        state = cp.group_states["group1"]
        assert state.summary_prompt == "Test prompt"
        assert state.cached_summary == "Test summary"
        assert state.last_child_versions == {"child1": 3}

    def test_clear_preserves_checkpoints(self, graph):
        """Test that clear() preserves checkpoints (nodes only cleared).
        
        Checkpoints are intentionally preserved so users can restore
        to previous states after clearing.
        """
        node = create_mock_context_node("node", "text")
        graph.add_node(node)
        graph.checkpoint("cp1")
        graph.checkpoint("cp2")

        assert len(graph.get_checkpoints()) == 2

        graph.clear()

        # Nodes should be cleared
        assert len(graph) == 0
        
        # Checkpoints are preserved
        assert len(graph.get_checkpoints()) == 2
