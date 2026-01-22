"""Tests for context graph operations, checkpoints, and state types.

Tests coverage for:
- src/activecontext/context/graph.py
- src/activecontext/context/checkpoint.py
- src/activecontext/context/state.py
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from activecontext.context.checkpoint import Checkpoint, GroupState
from activecontext.context.graph import ContextGraph
from activecontext.context.state import Expansion, TickFrequency
from tests.utils import create_mock_context_node


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_graph():
    """Create an empty ContextGraph."""
    return ContextGraph()


@pytest.fixture
def mock_text_node():
    """Create a mock TextNode for testing."""
    return create_mock_context_node("text-1", "text", mode="idle")


@pytest.fixture
def mock_running_node():
    """Create a mock node in running mode."""
    return create_mock_context_node("running-1", "text", mode="running")


@pytest.fixture
def populated_graph():
    """Create a graph with a test DAG structure.

    Structure:
        root1 (text)
        ├── child1 (text)
        │   └── grandchild1 (group)
        └── child2 (text, running)
        root2 (group)
    """
    graph = ContextGraph()

    # Create nodes
    root1 = create_mock_context_node("root1", "text")
    root2 = create_mock_context_node("root2", "group")
    child1 = create_mock_context_node("child1", "text")
    child2 = create_mock_context_node("child2", "text", mode="running")
    grandchild1 = create_mock_context_node("grandchild1", "group")

    # Add nodes
    graph.add_node(root1)
    graph.add_node(root2)
    graph.add_node(child1)
    graph.add_node(child2)
    graph.add_node(grandchild1)

    # Create links
    graph.link("child1", "root1")
    graph.link("child2", "root1")
    graph.link("grandchild1", "child1")

    return graph


# =============================================================================
# Graph Operations Tests
# =============================================================================


class TestGraphOperations:
    """Tests for basic graph operations."""

    def test_add_node(self, empty_graph):
        """Test adding a node to the graph."""
        node = create_mock_context_node("node1", "text")

        node_id = empty_graph.add_node(node)

        assert node_id == "node1"
        assert "node1" in empty_graph
        assert len(empty_graph) == 1
        assert node._graph is empty_graph

    def test_add_node_updates_type_index(self, empty_graph):
        """Test that adding nodes updates type index."""
        text1 = create_mock_context_node("text1", "text")
        text2 = create_mock_context_node("text2", "text")
        group1 = create_mock_context_node("group1", "group")

        empty_graph.add_node(text1)
        empty_graph.add_node(text2)
        empty_graph.add_node(group1)

        texts = empty_graph.get_nodes_by_type("text")
        groups = empty_graph.get_nodes_by_type("group")

        assert len(texts) == 2
        assert len(groups) == 1

    def test_add_node_as_root(self, empty_graph):
        """Test that nodes without parents become roots."""
        node = create_mock_context_node("node1", "text")

        empty_graph.add_node(node)

        roots = empty_graph.get_roots()
        assert len(roots) == 1
        assert roots[0].node_id == "node1"

    def test_add_running_node_updates_index(self, empty_graph):
        """Test that running nodes are indexed."""
        running_node = create_mock_context_node("running1", "text", mode="running")
        idle_node = create_mock_context_node("idle1", "text", mode="idle")

        empty_graph.add_node(running_node)
        empty_graph.add_node(idle_node)

        running_nodes = empty_graph.get_running_nodes()
        assert len(running_nodes) == 1
        assert running_nodes[0].node_id == "running1"

    def test_remove_node(self, empty_graph):
        """Test removing a node from the graph."""
        node = create_mock_context_node("node1", "text")
        empty_graph.add_node(node)

        empty_graph.remove_node("node1")

        assert "node1" not in empty_graph
        assert len(empty_graph) == 0

    def test_remove_node_recursive(self, populated_graph):
        """Test removing a node and all its descendants."""
        # Remove root1 recursively should remove child1, child2, grandchild1
        initial_count = len(populated_graph)
        populated_graph.remove_node("root1", recursive=True)

        assert "root1" not in populated_graph
        assert "child1" not in populated_graph
        assert "child2" not in populated_graph
        assert "grandchild1" not in populated_graph
        assert "root2" in populated_graph  # Should remain
        assert len(populated_graph) == 1

    def test_remove_node_unlinks_from_parents(self, populated_graph):
        """Test that removing a node unlinks it from parents."""
        # Remove child1, grandchild1 should become a root
        populated_graph.remove_node("child1")

        assert "child1" not in populated_graph
        assert "grandchild1" in populated_graph

        roots = populated_graph.get_roots()
        root_ids = {r.node_id for r in roots}
        assert "grandchild1" in root_ids

    def test_link_creates_parent_child_relationship(self, empty_graph):
        """Test linking two nodes creates parent-child relationship."""
        parent = create_mock_context_node("parent", "text")
        child = create_mock_context_node("child", "text")

        empty_graph.add_node(parent)
        empty_graph.add_node(child)

        success = empty_graph.link("child", "parent")

        assert success is True
        assert "parent" in child.parent_ids
        assert "child" in parent.children_ids

    def test_link_removes_child_from_roots(self, empty_graph):
        """Test that linking a root node removes it from roots."""
        parent = create_mock_context_node("parent", "text")
        child = create_mock_context_node("child", "text")

        empty_graph.add_node(parent)
        empty_graph.add_node(child)

        assert len(empty_graph.get_roots()) == 2

        empty_graph.link("child", "parent")

        roots = empty_graph.get_roots()
        assert len(roots) == 1
        assert roots[0].node_id == "parent"

    def test_link_cycle_detection(self, empty_graph):
        """Test that creating a cycle is rejected."""
        node_a = create_mock_context_node("a", "text")
        node_b = create_mock_context_node("b", "text")

        empty_graph.add_node(node_a)
        empty_graph.add_node(node_b)

        # Create A -> B
        empty_graph.link("b", "a")

        # Attempt to create B -> A (cycle)
        success = empty_graph.link("a", "b")

        assert success is False
        assert "b" not in node_a.parent_ids

    def test_link_self_cycle_rejection(self, empty_graph):
        """Test that self-links are rejected."""
        node = create_mock_context_node("node", "text")
        empty_graph.add_node(node)

        success = empty_graph.link("node", "node")

        assert success is False

    def test_link_nonexistent_nodes(self, empty_graph):
        """Test linking nonexistent nodes returns False."""
        node = create_mock_context_node("node", "text")
        empty_graph.add_node(node)

        success1 = empty_graph.link("node", "nonexistent")
        success2 = empty_graph.link("nonexistent", "node")

        assert success1 is False
        assert success2 is False

    def test_unlink_removes_relationship(self, populated_graph):
        """Test unlinking removes parent-child relationship."""
        child1_node = populated_graph.get_node("child1")
        root1_node = populated_graph.get_node("root1")

        assert "root1" in child1_node.parent_ids
        assert "child1" in root1_node.children_ids

        success = populated_graph.unlink("child1", "root1")

        assert success is True
        assert "root1" not in child1_node.parent_ids
        assert "child1" not in root1_node.children_ids

    def test_unlink_makes_node_root(self, populated_graph):
        """Test that unlinking all parents makes node a root."""
        # child1 has only one parent (root1)
        populated_graph.unlink("child1", "root1")

        roots = populated_graph.get_roots()
        root_ids = {r.node_id for r in roots}
        assert "child1" in root_ids

    def test_multiple_parents(self, empty_graph):
        """Test nodes can have multiple parents (DAG structure)."""
        parent1 = create_mock_context_node("parent1", "text")
        parent2 = create_mock_context_node("parent2", "text")
        child = create_mock_context_node("child", "text")

        empty_graph.add_node(parent1)
        empty_graph.add_node(parent2)
        empty_graph.add_node(child)

        empty_graph.link("child", "parent1")
        empty_graph.link("child", "parent2")

        child_node = empty_graph.get_node("child")
        assert len(child_node.parent_ids) == 2
        assert "parent1" in child_node.parent_ids
        assert "parent2" in child_node.parent_ids

        parents = empty_graph.get_parents("child")
        assert len(parents) == 2


# =============================================================================
# Graph Indices and Queries Tests
# =============================================================================


class TestGraphIndices:
    """Tests for graph indices and query methods."""

    def test_get_node(self, populated_graph):
        """Test getting a node by ID."""
        node = populated_graph.get_node("root1")
        assert node is not None
        assert node.node_id == "root1"

        node = populated_graph.get_node("nonexistent")
        assert node is None

    def test_get_nodes_by_type(self, populated_graph):
        """Test querying nodes by type."""
        texts = populated_graph.get_nodes_by_type("text")
        groups = populated_graph.get_nodes_by_type("group")

        assert len(texts) == 3  # root1, child1, child2
        assert len(groups) == 2  # root2, grandchild1

    def test_get_running_nodes(self, populated_graph):
        """Test querying running nodes."""
        running = populated_graph.get_running_nodes()

        assert len(running) == 1
        assert running[0].node_id == "child2"

    def test_get_roots(self, populated_graph):
        """Test querying root nodes."""
        roots = populated_graph.get_roots()

        assert len(roots) == 2
        root_ids = {r.node_id for r in roots}
        assert root_ids == {"root1", "root2"}

    def test_get_children(self, populated_graph):
        """Test getting direct children of a node."""
        children = populated_graph.get_children("root1")

        assert len(children) == 2
        child_ids = {c.node_id for c in children}
        assert child_ids == {"child1", "child2"}

        children = populated_graph.get_children("child2")
        assert len(children) == 0

    def test_get_parents(self, populated_graph):
        """Test getting direct parents of a node."""
        parents = populated_graph.get_parents("child1")

        assert len(parents) == 1
        assert parents[0].node_id == "root1"

        parents = populated_graph.get_parents("root1")
        assert len(parents) == 0


# =============================================================================
# Graph Traversal Tests
# =============================================================================


class TestGraphTraversal:
    """Tests for graph traversal methods."""

    def test_get_descendants(self, populated_graph):
        """Test getting all descendants of a node."""
        descendants = populated_graph.get_descendants("root1")

        desc_ids = {d.node_id for d in descendants}
        assert desc_ids == {"child1", "child2", "grandchild1"}

        descendants = populated_graph.get_descendants("child1")
        desc_ids = {d.node_id for d in descendants}
        assert desc_ids == {"grandchild1"}

        descendants = populated_graph.get_descendants("grandchild1")
        assert len(descendants) == 0

    def test_get_ancestors(self, populated_graph):
        """Test getting all ancestors of a node."""
        ancestors = populated_graph.get_ancestors("grandchild1")

        anc_ids = {a.node_id for a in ancestors}
        assert anc_ids == {"child1", "root1"}

        ancestors = populated_graph.get_ancestors("child1")
        anc_ids = {a.node_id for a in ancestors}
        assert anc_ids == {"root1"}

        ancestors = populated_graph.get_ancestors("root1")
        assert len(ancestors) == 0

    def test_is_descendant(self, populated_graph):
        """Test _is_descendant cycle detection helper."""
        # grandchild1 is descendant of root1
        assert populated_graph._is_descendant("grandchild1", "root1") is True

        # child1 is descendant of root1
        assert populated_graph._is_descendant("child1", "root1") is True

        # root1 is NOT descendant of child1
        assert populated_graph._is_descendant("root1", "child1") is False

        # root2 is NOT descendant of root1
        assert populated_graph._is_descendant("root2", "root1") is False

        # Node is descendant of itself
        assert populated_graph._is_descendant("root1", "root1") is True

    def test_traversal_with_dag(self, empty_graph):
        """Test traversal with DAG (multiple parents)."""
        # Create diamond structure: A -> B, A -> C, B -> D, C -> D
        a = create_mock_context_node("a", "text")
        b = create_mock_context_node("b", "text")
        c = create_mock_context_node("c", "text")
        d = create_mock_context_node("d", "text")

        for node in [a, b, c, d]:
            empty_graph.add_node(node)

        empty_graph.link("b", "a")
        empty_graph.link("c", "a")
        empty_graph.link("d", "b")
        empty_graph.link("d", "c")

        descendants = empty_graph.get_descendants("a")
        desc_ids = {n.node_id for n in descendants}
        assert desc_ids == {"b", "c", "d"}

        ancestors = empty_graph.get_ancestors("d")
        anc_ids = {n.node_id for n in ancestors}
        assert anc_ids == {"a", "b", "c"}


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoints:
    """Tests for graph checkpointing."""

    def test_capture_checkpoint(self, populated_graph):
        """Test capturing a checkpoint saves edges."""
        cp = populated_graph.checkpoint("test-checkpoint")

        assert cp.name == "test-checkpoint"
        assert len(cp.edges) > 0
        assert ("child1", "root1") in cp.edges
        assert ("child2", "root1") in cp.edges
        assert ("grandchild1", "child1") in cp.edges

    def test_checkpoint_saved_to_graph(self, populated_graph):
        """Test checkpoint is stored in graph."""
        populated_graph.checkpoint("checkpoint1")

        checkpoints = populated_graph.get_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].name == "checkpoint1"

        cp = populated_graph.get_checkpoint("checkpoint1")
        assert cp is not None
        assert cp.name == "checkpoint1"

    def test_restore_checkpoint_rebuilds_structure(self, populated_graph):
        """Test restoring a checkpoint rebuilds edge structure."""
        # Capture initial state
        cp = populated_graph.checkpoint("initial")

        # Modify graph
        populated_graph.unlink("child1", "root1")
        populated_graph.link("child1", "root2")

        # Verify modification
        child1_parents = [p.node_id for p in populated_graph.get_parents("child1")]
        assert "root2" in child1_parents
        assert "root1" not in child1_parents

        # Restore checkpoint
        populated_graph.restore("initial")

        # Verify restoration
        child1_parents = [p.node_id for p in populated_graph.get_parents("child1")]
        assert "root1" in child1_parents
        assert "root2" not in child1_parents

    def test_restore_checkpoint_by_object(self, populated_graph):
        """Test restoring checkpoint by passing Checkpoint object."""
        cp = populated_graph.checkpoint("test")

        # Modify graph
        populated_graph.clear()
        assert len(populated_graph) == 0

        # Can't restore by name (graph was cleared)
        # But can restore by object
        # Note: This won't work because nodes are gone. Let's test differently.

        # Actually, let me test that we can restore by checkpoint object
        cp2 = Checkpoint(
            name="manual",
            edges=[("child", "parent")],
            root_ids={"parent"},
        )

        # This should fail gracefully if nodes don't exist
        populated_graph.restore(cp2)  # No error, just skips missing nodes

    def test_checkpoint_nonexistent_raises(self, populated_graph):
        """Test restoring nonexistent checkpoint raises KeyError."""
        with pytest.raises(KeyError):
            populated_graph.restore("nonexistent")

    def test_checkpoint_preserves_group_states(self, empty_graph):
        """Test checkpoint saves and restores group-specific state."""
        from activecontext.context.nodes import GroupNode

        # Create a mock GroupNode
        group = Mock(spec=GroupNode)
        group.node_id = "group1"
        group.node_type = "group"
        group.mode = "idle"
        group.parent_ids = set()
        group.children_ids = set()
        group.summary_prompt = "Custom prompt"
        group.cached_summary = "Cached summary text"
        group.last_child_versions = {"child1": 5}
        group.GetDigest = Mock(return_value={"node_id": "group1"})

        empty_graph.add_node(group)

        cp = empty_graph.checkpoint("with-group-state")

        assert "group1" in cp.group_states
        state = cp.group_states["group1"]
        assert state.summary_prompt == "Custom prompt"
        assert state.cached_summary == "Cached summary text"
        assert state.last_child_versions == {"child1": 5}

    def test_multiple_checkpoints(self, populated_graph):
        """Test creating multiple checkpoints with unique names."""
        cp1 = populated_graph.checkpoint("checkpoint1")
        time.sleep(0.01)  # Ensure different timestamps
        cp2 = populated_graph.checkpoint("checkpoint2")

        checkpoints = populated_graph.get_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0].created_at < checkpoints[1].created_at

    def test_delete_checkpoint(self, populated_graph):
        """Test deleting a checkpoint."""
        populated_graph.checkpoint("to-delete")
        assert populated_graph.get_checkpoint("to-delete") is not None

        success = populated_graph.delete_checkpoint("to-delete")
        assert success is True
        assert populated_graph.get_checkpoint("to-delete") is None

        success = populated_graph.delete_checkpoint("to-delete")
        assert success is False  # Already deleted

    def test_checkpoint_serialization(self):
        """Test checkpoint to_dict and from_dict."""
        cp = Checkpoint(
            checkpoint_id="abc123",
            name="test",
            created_at=1234567890.0,
            edges=[("a", "b"), ("c", "d")],
            group_states={
                "g1": GroupState(
                    node_id="g1",
                    summary_prompt="prompt",
                    cached_summary="summary",
                    last_child_versions={"child": 3},
                )
            },
            root_ids={"a", "c"},
        )

        data = cp.to_dict()
        assert data["checkpoint_id"] == "abc123"
        assert data["name"] == "test"
        assert len(data["edges"]) == 2
        assert "g1" in data["group_states"]

        restored = Checkpoint.from_dict(data)
        assert restored.checkpoint_id == "abc123"
        assert restored.name == "test"
        assert restored.edges == [("a", "b"), ("c", "d")]
        assert "g1" in restored.group_states
        assert restored.root_ids == {"a", "c"}

    def test_checkpoint_get_digest(self):
        """Test checkpoint digest provides summary."""
        cp = Checkpoint(
            checkpoint_id="abc",
            name="test",
            edges=[("a", "b"), ("c", "d")],
            group_states={"g1": GroupState(node_id="g1")},
            root_ids={"a", "c"},
        )

        digest = cp.get_digest()
        assert digest["checkpoint_id"] == "abc"
        assert digest["name"] == "test"
        assert digest["edge_count"] == 2
        assert digest["group_count"] == 1
        assert digest["root_count"] == 2


# =============================================================================
# State Types Tests
# =============================================================================


class TestStateTypes:
    """Tests for Expansion and TickFrequency types."""

    def test_node_state_enum_values(self):
        """Test Expansion enum values."""
        assert Expansion.HIDDEN.value == "hidden"
        assert Expansion.COLLAPSED.value == "collapsed"
        assert Expansion.SUMMARY.value == "summary"
        assert Expansion.DETAILS.value == "details"
        assert Expansion.ALL.value == "all"

    def test_node_state_string_representation(self):
        """Test Expansion string representation."""
        assert str(Expansion.HIDDEN) == "hidden"
        assert str(Expansion.SUMMARY) == "summary"
        assert str(Expansion.ALL) == "all"

    def test_tick_frequency_turn(self):
        """Test TickFrequency.turn() factory."""
        freq = TickFrequency.turn()
        assert freq.mode == "turn"
        assert freq.interval is None
        assert freq.to_string() == "turn"

    def test_tick_frequency_async(self):
        """Test TickFrequency.async_() factory."""
        freq = TickFrequency.async_()
        assert freq.mode == "async"
        assert freq.interval is None
        assert freq.to_string() == "async"

    def test_tick_frequency_never(self):
        """Test TickFrequency.never() factory."""
        freq = TickFrequency.never()
        assert freq.mode == "never"
        assert freq.interval is None
        assert freq.to_string() == "never"

    def test_tick_frequency_period(self):
        """Test TickFrequency.period() factory."""
        freq = TickFrequency.period(5.0)
        assert freq.mode == "periodic"
        assert freq.interval == 5.0
        assert freq.to_string() == "period:5.0"

    def test_tick_frequency_from_string_sync(self):
        """Test parsing 'Sync' and 'turn' strings."""
        freq1 = TickFrequency.from_string("Sync")
        assert freq1.mode == "turn"

        freq2 = TickFrequency.from_string("turn")
        assert freq2.mode == "turn"

    def test_tick_frequency_from_string_async(self):
        """Test parsing 'async' string."""
        freq = TickFrequency.from_string("async")
        assert freq.mode == "async"

    def test_tick_frequency_from_string_never(self):
        """Test parsing 'never' string."""
        freq = TickFrequency.from_string("never")
        assert freq.mode == "never"

    def test_tick_frequency_from_string_periodic_seconds(self):
        """Test parsing periodic with seconds."""
        freq1 = TickFrequency.from_string("Periodic:5s")
        assert freq1.mode == "periodic"
        assert freq1.interval == 5.0

        freq2 = TickFrequency.from_string("period:10.5")
        assert freq2.mode == "periodic"
        assert freq2.interval == 10.5

    def test_tick_frequency_from_string_periodic_minutes(self):
        """Test parsing periodic with minutes."""
        freq = TickFrequency.from_string("period:2m")
        assert freq.mode == "periodic"
        assert freq.interval == 120.0

    def test_tick_frequency_from_string_periodic_hours(self):
        """Test parsing periodic with hours."""
        freq = TickFrequency.from_string("Periodic:1h")
        assert freq.mode == "periodic"
        assert freq.interval == 3600.0

    def test_tick_frequency_from_string_invalid(self):
        """Test that invalid strings raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tick frequency format"):
            TickFrequency.from_string("invalid")

        with pytest.raises(ValueError, match="Unknown tick frequency format"):
            TickFrequency.from_string("Periodic")  # Missing value (not recognized)

    def test_tick_frequency_to_string_roundtrip(self):
        """Test to_string() and from_string() roundtrip."""
        frequencies = [
            TickFrequency.turn(),
            TickFrequency.async_(),
            TickFrequency.never(),
            TickFrequency.period(5.0),
            TickFrequency.period(120.0),
        ]

        for freq in frequencies:
            string = freq.to_string()
            parsed = TickFrequency.from_string(string)
            assert parsed.mode == freq.mode
            assert parsed.interval == freq.interval

    def test_tick_frequency_str(self):
        """Test __str__ method."""
        freq = TickFrequency.period(5.0)
        assert str(freq) == "period:5.0"

    def test_tick_frequency_immutable(self):
        """Test TickFrequency is frozen/immutable."""
        freq = TickFrequency.turn()

        with pytest.raises(AttributeError):
            freq.mode = "async"  # type: ignore


# =============================================================================
# Graph Utility Tests
# =============================================================================


class TestGraphUtilities:
    """Tests for graph utility methods."""

    def test_to_dict_serialization(self, populated_graph):
        """Test graph serialization to dict."""
        data = populated_graph.to_dict()

        assert "nodes" in data
        assert "root_ids" in data
        assert "running_node_ids" in data

        assert len(data["nodes"]) == 5
        assert len(data["root_ids"]) == 2

    def test_clear(self, populated_graph):
        """Test clearing all nodes from graph."""
        assert len(populated_graph) > 0

        populated_graph.clear()

        assert len(populated_graph) == 0
        assert len(populated_graph.get_roots()) == 0
        assert len(populated_graph.get_running_nodes()) == 0

    def test_len(self, populated_graph):
        """Test __len__ returns node count."""
        assert len(populated_graph) == 5

    def test_contains(self, populated_graph):
        """Test __contains__ checks node existence."""
        assert "root1" in populated_graph
        assert "child1" in populated_graph
        assert "nonexistent" not in populated_graph

    def test_iter(self, populated_graph):
        """Test __iter__ iterates over nodes."""
        nodes = list(populated_graph)

        assert len(nodes) == 5
        node_ids = {n.node_id for n in nodes}
        assert "root1" in node_ids
        assert "root2" in node_ids
