"""Context graph (DAG) for managing context nodes.

The ContextGraph manages a directed acyclic graph of context nodes,
supporting multiple parents per node and efficient traversal.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.checkpoint import Checkpoint, GroupState
from activecontext.context.state import Notification, NotificationLevel

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode


@dataclass
class _ChildOrderNode:
    """Node in the doubly-linked child order list."""

    node_id: str
    prev: _ChildOrderNode | None = None
    next: _ChildOrderNode | None = None


class LinkedChildOrder:
    """Doubly-linked list with O(1) insert-after and dict index.

    Maintains document ordering for child nodes with efficient operations:
    - append: O(1)
    - insert_after: O(1)
    - remove: O(1)
    - membership test: O(1)
    - iteration: O(n)
    """

    __slots__ = ("_head", "_tail", "_index")

    def __init__(self) -> None:
        self._head: _ChildOrderNode | None = None
        self._tail: _ChildOrderNode | None = None
        self._index: dict[str, _ChildOrderNode] = {}

    def append(self, node_id: str) -> None:
        """Append node_id to end. O(1)."""
        if node_id in self._index:
            return  # Already exists

        new_node = _ChildOrderNode(node_id=node_id, prev=self._tail)
        if self._tail:
            self._tail.next = new_node
        else:
            self._head = new_node
        self._tail = new_node
        self._index[node_id] = new_node

    def insert_after(self, after_id: str, node_id: str) -> None:
        """Insert node_id immediately after after_id. O(1).

        If after_id not found, appends to end.
        """
        if node_id in self._index:
            return  # Already exists

        after_node = self._index.get(after_id)
        if not after_node:
            self.append(node_id)
            return

        new_node = _ChildOrderNode(
            node_id=node_id,
            prev=after_node,
            next=after_node.next,
        )
        if after_node.next:
            after_node.next.prev = new_node
        else:
            self._tail = new_node
        after_node.next = new_node
        self._index[node_id] = new_node

    def insert_before(self, before_id: str, node_id: str) -> None:
        """Insert node_id immediately before before_id. O(1).

        If before_id not found, prepends to beginning.
        """
        if node_id in self._index:
            return  # Already exists

        before_node = self._index.get(before_id)
        if not before_node:
            # Prepend to beginning
            new_node = _ChildOrderNode(node_id=node_id, next=self._head)
            if self._head:
                self._head.prev = new_node
            else:
                self._tail = new_node
            self._head = new_node
            self._index[node_id] = new_node
            return

        new_node = _ChildOrderNode(
            node_id=node_id,
            prev=before_node.prev,
            next=before_node,
        )
        if before_node.prev:
            before_node.prev.next = new_node
        else:
            self._head = new_node
        before_node.prev = new_node
        self._index[node_id] = new_node

    def remove(self, node_id: str) -> bool:
        """Remove node. O(1). Returns True if removed."""
        node = self._index.pop(node_id, None)
        if not node:
            return False

        if node.prev:
            node.prev.next = node.next
        else:
            self._head = node.next

        if node.next:
            node.next.prev = node.prev
        else:
            self._tail = node.prev

        return True

    def __contains__(self, node_id: str) -> bool:
        """O(1) membership test."""
        return node_id in self._index

    def __iter__(self) -> Iterator[str]:
        """Iterate in order. O(n)."""
        current = self._head
        while current:
            yield current.node_id
            current = current.next

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return len(self._index) > 0

    def clear(self) -> None:
        """Remove all nodes. O(1)."""
        self._head = None
        self._tail = None
        self._index.clear()

    def to_list(self) -> list[str]:
        """Convert to list for serialization."""
        return list(self)

    @classmethod
    def from_list(cls, items: list[str]) -> LinkedChildOrder:
        """Create from list for deserialization."""
        instance = cls()
        for item in items:
            instance.append(item)
        return instance


@dataclass
class ContextGraph:
    """Directed acyclic graph of context nodes.

    Nodes can have multiple parents (DAG structure). The graph maintains
    indices for efficient queries by type and mode.

    Attributes:
        _nodes: All nodes by ID
        _root_ids: Nodes with no parents (entry points)
        _by_type: Index of node IDs by type
        _running_nodes: Index of nodes with mode="running"
    """

    _nodes: dict[str, ContextNode] = field(default_factory=dict)
    _root_ids: set[str] = field(default_factory=set)
    _by_type: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _running_nodes: set[str] = field(default_factory=set)
    _checkpoints: dict[str, Checkpoint] = field(default_factory=dict)

    # Per-type sequence counters for display IDs (e.g., text_1, message_13)
    _type_counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Root context node ID for document-ordered rendering
    _root_context_id: str | None = field(default=None)

    # Notification system for change alerts
    _seen_traces: set[str] = field(default_factory=set)  # For deduplication
    _pending_notifications: list[Notification] = field(default_factory=list)
    _has_wake: bool = False

    def add_node(self, node: ContextNode) -> str:
        """Add a node to the graph.

        Args:
            node: The node to add

        Returns:
            The node's ID
        """
        # Set graph reference on node
        node._graph = self

        # Assign display sequence if not already set
        if node.display_sequence is None:
            self._type_counters[node.node_type] += 1
            node.display_sequence = self._type_counters[node.node_type]

        # Store node
        self._nodes[node.node_id] = node

        # Update type index
        self._by_type[node.node_type].add(node.node_id)

        # Update running index
        if node.mode == "running":
            self._running_nodes.add(node.node_id)

        # If no parents, it's a root
        if not node.parent_ids:
            self._root_ids.add(node.node_id)

        return node.node_id

    def remove_node(self, node_id: str, recursive: bool = False) -> None:
        """Remove a node from the graph.

        Args:
            node_id: ID of node to remove
            recursive: If True, also remove all descendants
        """
        node = self._nodes.get(node_id)
        if not node:
            return

        if recursive:
            # Remove descendants first (depth-first)
            for child_id in list(node.children_ids):
                self.remove_node(child_id, recursive=True)

        # Unlink from parents
        for parent_id in list(node.parent_ids):
            self.unlink(node_id, parent_id)

        # Unlink children (they become roots or stay linked to other parents)
        for child_id in list(node.children_ids):
            self.unlink(child_id, node_id)

        # Remove from indices
        self._by_type[node.node_type].discard(node_id)
        self._running_nodes.discard(node_id)
        self._root_ids.discard(node_id)

        # Remove node
        del self._nodes[node_id]

    def link(
        self,
        child_id: str,
        parent_id: str,
        *,
        after: str | None = None,
        before: str | None = None,
    ) -> bool:
        """Link a child node to a parent node.

        Handles cycle detection, root tracking, and child_order maintenance.

        Args:
            child_id: ID of child node
            parent_id: ID of parent node
            after: If provided, insert in child_order immediately after this node_id.
            before: If provided, insert in child_order immediately before this node_id.
                   Takes precedence over after if both are provided.
                   If neither, append to end.

        Returns:
            True if link was created, False if nodes don't exist or would create cycle
        """
        child = self._nodes.get(child_id)
        parent = self._nodes.get(parent_id)

        if not child or not parent:
            return False

        # Check for cycle (parent cannot be descendant of child)
        if self._is_descendant(parent_id, child_id):
            return False

        # Create the link
        child.parent_ids.add(parent_id)
        parent.children_ids.add(child_id)

        # Maintain child_order for document ordering
        if parent.child_order is None:
            parent.child_order = LinkedChildOrder()

        if child_id not in parent.child_order:
            if before and before in parent.child_order:
                parent.child_order.insert_before(before, child_id)
            elif after and after in parent.child_order:
                parent.child_order.insert_after(after, child_id)
            else:
                parent.child_order.append(child_id)

        # Child is no longer a root
        self._root_ids.discard(child_id)

        return True

    def unlink(self, child_id: str, parent_id: str) -> bool:
        """Remove link between child and parent.

        Also removes from child_order if parent has it.

        Args:
            child_id: ID of child node
            parent_id: ID of parent node

        Returns:
            True if link was removed, False if nodes don't exist
        """
        child = self._nodes.get(child_id)
        parent = self._nodes.get(parent_id)

        if not child or not parent:
            return False

        # Remove link
        child.parent_ids.discard(parent_id)
        parent.children_ids.discard(child_id)

        # Remove from child_order if present
        if parent.child_order and child_id in parent.child_order:
            parent.child_order.remove(child_id)

        # If child has no more parents, it becomes a root
        if not child.parent_ids:
            self._root_ids.add(child_id)

        return True

    def get_node(self, node_id: str) -> ContextNode | None:
        """Get a node by ID.

        Args:
            node_id: Unique identifier of the node.
        """
        return self._nodes.get(node_id)

    def get_node_by_display_id(self, display_id: str) -> ContextNode | None:
        """Get a node by display_id (e.g., 'text_1').

        Used for namespace injection to allow direct node access.

        Args:
            display_id: Display identifier like 'text_1', 'group_2'.

        Returns:
            The node if found, None otherwise.
        """
        for node in self._nodes.values():
            if node.display_id == display_id:
                return node
        return None

    def get_children(self, node_id: str) -> list[ContextNode]:
        """Get direct children of a node.

        Args:
            node_id: Unique identifier of the parent node.
        """
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parents(self, node_id: str) -> list[ContextNode]:
        """Get direct parents of a node.

        Args:
            node_id: Unique identifier of the child node.
        """
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[pid] for pid in node.parent_ids if pid in self._nodes]

    def get_ancestors(self, node_id: str) -> list[ContextNode]:
        """Get all ancestors of a node (parents, grandparents, etc.).

        Args:
            node_id: Unique identifier of the starting node.
        """
        ancestors: list[ContextNode] = []
        visited: set[str] = set()

        def collect(nid: str) -> None:
            node = self._nodes.get(nid)
            if not node:
                return
            for parent_id in node.parent_ids:
                if parent_id not in visited:
                    visited.add(parent_id)
                    parent = self._nodes.get(parent_id)
                    if parent:
                        ancestors.append(parent)
                        collect(parent_id)

        collect(node_id)
        return ancestors

    def get_descendants(self, node_id: str) -> list[ContextNode]:
        """Get all descendants of a node (children, grandchildren, etc.).

        Args:
            node_id: Unique identifier of the starting node.
        """
        descendants: list[ContextNode] = []
        visited: set[str] = set()

        def collect(nid: str) -> None:
            node = self._nodes.get(nid)
            if not node:
                return
            for child_id in node.children_ids:
                if child_id not in visited:
                    visited.add(child_id)
                    child = self._nodes.get(child_id)
                    if child:
                        descendants.append(child)
                        collect(child_id)

        collect(node_id)
        return descendants

    def get_roots(self) -> list[ContextNode]:
        """Get all root nodes (nodes with no parents)."""
        return [self._nodes[nid] for nid in self._root_ids if nid in self._nodes]

    def get_root(self) -> ContextNode | None:
        """Get the root context node for document-ordered rendering.

        Returns:
            The root context GroupNode, or None if not set.
        """
        if self._root_context_id and self._root_context_id in self._nodes:
            return self._nodes[self._root_context_id]
        return None

    def set_root(self, node_id: str) -> None:
        """Set the root context node ID.

        Args:
            node_id: ID of the GroupNode to use as root context.
        """
        self._root_context_id = node_id

    def get_running_nodes(self) -> list[ContextNode]:
        """Get all nodes with mode='running'."""
        return [self._nodes[nid] for nid in self._running_nodes if nid in self._nodes]

    def get_nodes_by_type(self, node_type: str) -> list[ContextNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: Type identifier (e.g., "text", "group", "shell").
        """
        return [
            self._nodes[nid] for nid in self._by_type.get(node_type, set()) if nid in self._nodes
        ]

    def get_traces_for_node(self, node_id: str) -> list[ContextNode]:
        """Get all TraceNodes for a given node (now siblings, not children).

        Traces are linked to the same parent as the traced node, or to the
        node's trace_sink if it has no parent.

        Args:
            node_id: The node to get traces for

        Returns:
            List of TraceNodes, sorted by version (newest first)
        """
        node = self._nodes.get(node_id)
        if not node:
            return []

        traces: list[ContextNode] = []

        # Find traces among siblings (same parent)
        for parent_id in node.parent_ids:
            parent = self._nodes.get(parent_id)
            if parent:
                for child_id in parent.children_ids:
                    child = self._nodes.get(child_id)
                    if (
                        child
                        and child.node_type == "trace"
                        and getattr(child, "node", None) == node_id
                    ):
                        traces.append(child)

        # Also check trace_sink if set
        if node.trace_sink and node.trace_sink.node_id in self._nodes:
            for child_id in node.trace_sink.children_ids:
                child = self._nodes.get(child_id)
                if child and child.node_type == "trace" and getattr(child, "node", None) == node_id:
                    traces.append(child)

        # Sort by new_version descending (newest first)
        traces.sort(key=lambda t: getattr(t, "new_version", 0), reverse=True)
        return traces

    # -------------------------------------------------------------------------
    # Notification System
    # -------------------------------------------------------------------------

    def emit_notification(
        self,
        node_id: str,
        trace_id: str,
        header: str,
        level: NotificationLevel,
        originator: str | None = None,
    ) -> None:
        """Collect notification with deduplication.

        Called by nodes when their notification_level is HOLD or WAKE.
        Notifications are deduplicated by trace_id to ensure at-most-once
        delivery.

        Args:
            node_id: Source node that changed
            trace_id: Unique ID for deduplication (typically node_id:version)
            header: Brief description for the alert
            level: NotificationLevel (HOLD or WAKE)
            originator: Who/what caused the change (node ID, filename, or arbitrary string)
        """
        if trace_id in self._seen_traces:
            return  # Already processed this trace
        self._seen_traces.add(trace_id)

        notification = Notification(
            node_id=node_id,
            trace_id=trace_id,
            header=header,
            level=level.value,
            originator=originator,
        )
        self._pending_notifications.append(notification)

        if level == NotificationLevel.WAKE:
            self._has_wake = True

    def has_wake_notification(self) -> bool:
        """Check if there's a pending WAKE notification."""
        return self._has_wake

    def flush_notifications(self) -> list[Notification]:
        """Get and clear pending notifications.

        Called by Session.tick() to process notifications and update
        the Alerts group.

        Returns:
            List of pending notifications (may be empty)
        """
        notifications = self._pending_notifications
        self._pending_notifications = []
        self._has_wake = False
        self._seen_traces.clear()  # Reset for next batch
        return notifications

    def _is_descendant(self, node_id: str, potential_ancestor_id: str) -> bool:
        """Check if node_id is a descendant of potential_ancestor_id."""
        if node_id == potential_ancestor_id:
            return True

        visited: set[str] = set()

        def check(nid: str) -> bool:
            if nid in visited:
                return False
            visited.add(nid)

            node = self._nodes.get(nid)
            if not node:
                return False

            for parent_id in node.parent_ids:
                if parent_id == potential_ancestor_id:
                    return True
                if check(parent_id):
                    return True
            return False

        return check(node_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph for persistence.

        Returns full node data (not just digest) for reconstruction.
        """
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "root_ids": list(self._root_ids),
            "running_node_ids": list(self._running_nodes),
            "type_counters": dict(self._type_counters),
            "root_context_id": self._root_context_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextGraph:
        """Deserialize graph from dict.

        Args:
            data: Dict with "nodes", "root_ids", "running_node_ids", "type_counters"

        Returns:
            Reconstructed ContextGraph
        """
        from activecontext.context.nodes import ContextNode

        graph = cls()

        # Restore type counters first (before adding nodes)
        graph._type_counters = defaultdict(int, data.get("type_counters", {}))

        # Deserialize nodes
        for node_data in data.get("nodes", []):
            node = ContextNode.from_dict(node_data)
            # Add to internal structures without using add_node()
            # (which would reset parent_ids)
            node._graph = graph
            graph._nodes[node.node_id] = node
            graph._by_type[node.node_type].add(node.node_id)

        # Restore root_ids
        graph._root_ids = set(data.get("root_ids", []))

        # Restore running_nodes
        graph._running_nodes = set(data.get("running_node_ids", []))

        # Restore root context ID
        graph._root_context_id = data.get("root_context_id")

        return graph

    def clear(self) -> None:
        """Remove all nodes from the graph."""
        self._nodes.clear()
        self._root_ids.clear()
        self._by_type.clear()
        self._running_nodes.clear()
        self._root_context_id = None

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def checkpoint(self, name: str) -> Checkpoint:
        """Capture current edge structure as a named checkpoint.

        Checkpoints save the organizational structure (parent/child links)
        and group-specific state (summaries, prompts), allowing restoration
        of a particular view of the content nodes.

        Args:
            name: Human-readable name for the checkpoint

        Returns:
            The created Checkpoint
        """
        from activecontext.context.nodes import GroupNode

        # Capture all edges
        edges: list[tuple[str, str]] = [
            (node.node_id, parent_id)
            for node in self._nodes.values()
            for parent_id in node.parent_ids
        ]

        # Capture group-specific state
        group_states: dict[str, GroupState] = {}
        for node in self._nodes.values():
            if isinstance(node, GroupNode):
                group_states[node.node_id] = GroupState(
                    node_id=node.node_id,
                    summary_prompt=node.summary_prompt,
                    cached_summary=node.cached_summary,
                    last_child_versions=dict(node.last_child_versions),
                )

        cp = Checkpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            name=name,
            created_at=time.time(),
            edges=edges,
            group_states=group_states,
            root_ids=set(self._root_ids),
        )

        self._checkpoints[name] = cp
        return cp

    def restore(self, name_or_checkpoint: str | Checkpoint) -> None:
        """Restore edge structure from a checkpoint.

        This replaces the current parent/child links with those from the
        checkpoint. Content nodes are preserved; only the organizational
        structure changes.

        Args:
            name_or_checkpoint: Checkpoint name or Checkpoint object
        """
        from activecontext.context.nodes import GroupNode

        # Get checkpoint
        if isinstance(name_or_checkpoint, str):
            cp = self._checkpoints.get(name_or_checkpoint)
            if not cp:
                raise KeyError(f"Checkpoint not found: {name_or_checkpoint}")
        else:
            cp = name_or_checkpoint

        # Clear all current edges
        for node in self._nodes.values():
            node.parent_ids.clear()
            node.children_ids.clear()
            # Clear child_order to prevent stale entries (if initialized)
            if node.child_order is not None:
                node.child_order.clear()
        self._root_ids.clear()

        # Restore edges from checkpoint
        for child_id, parent_id in cp.edges:
            if child_id in self._nodes and parent_id in self._nodes:
                self.link(child_id, parent_id)

        # Restore root IDs for nodes that exist
        for root_id in cp.root_ids:
            if root_id in self._nodes:
                node = self._nodes[root_id]
                if not node.parent_ids:
                    self._root_ids.add(root_id)

        # Restore group states
        for node_id, state in cp.group_states.items():
            group_node = self._nodes.get(node_id)
            if group_node is not None and isinstance(group_node, GroupNode):
                group_node.summary_prompt = state.summary_prompt
                group_node.cached_summary = state.cached_summary
                group_node.last_child_versions = dict(state.last_child_versions)

    def get_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints.

        Returns:
            List of all checkpoints, ordered by creation time
        """
        return sorted(self._checkpoints.values(), key=lambda cp: cp.created_at)

    def get_checkpoint(self, name: str) -> Checkpoint | None:
        """Get a checkpoint by name.

        Args:
            name: The checkpoint name

        Returns:
            The Checkpoint, or None if not found
        """
        return self._checkpoints.get(name)

    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint by name.

        Args:
            name: The checkpoint name

        Returns:
            True if deleted, False if not found
        """
        return self._checkpoints.pop(name, None) is not None

    def __len__(self) -> int:
        """Return number of nodes in graph."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self._nodes

    def __iter__(self) -> Iterator[ContextNode]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())
