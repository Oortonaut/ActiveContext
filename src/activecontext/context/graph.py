"""Context graph (DAG) for managing context nodes.

The ContextGraph manages a directed acyclic graph of context nodes,
supporting multiple parents per node and efficient traversal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode


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

    def add_node(self, node: ContextNode) -> str:
        """Add a node to the graph.

        Args:
            node: The node to add

        Returns:
            The node's ID
        """
        # Set graph reference on node
        node._graph = self

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

    def link(self, child_id: str, parent_id: str) -> bool:
        """Link a child node to a parent node.

        Args:
            child_id: ID of child node
            parent_id: ID of parent node

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

        # Create link
        child.parent_ids.add(parent_id)
        parent.children_ids.add(child_id)

        # Child is no longer a root
        self._root_ids.discard(child_id)

        return True

    def unlink(self, child_id: str, parent_id: str) -> bool:
        """Remove link between child and parent.

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

        # If child has no more parents, it becomes a root
        if not child.parent_ids:
            self._root_ids.add(child_id)

        return True

    def get_node(self, node_id: str) -> ContextNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> list[ContextNode]:
        """Get direct children of a node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parents(self, node_id: str) -> list[ContextNode]:
        """Get direct parents of a node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[pid] for pid in node.parent_ids if pid in self._nodes]

    def get_ancestors(self, node_id: str) -> list[ContextNode]:
        """Get all ancestors of a node (parents, grandparents, etc.)."""
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
        """Get all descendants of a node (children, grandchildren, etc.)."""
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

    def get_running_nodes(self) -> list[ContextNode]:
        """Get all nodes with mode='running'."""
        return [self._nodes[nid] for nid in self._running_nodes if nid in self._nodes]

    def get_nodes_by_type(self, node_type: str) -> list[ContextNode]:
        """Get all nodes of a specific type."""
        return [self._nodes[nid] for nid in self._by_type.get(node_type, set()) if nid in self._nodes]

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
        """Serialize graph for persistence."""
        return {
            "nodes": {nid: node.GetDigest() for nid, node in self._nodes.items()},
            "root_ids": list(self._root_ids),
            "edges": [
                {"child": nid, "parent": pid}
                for nid, node in self._nodes.items()
                for pid in node.parent_ids
            ],
        }

    def clear(self) -> None:
        """Remove all nodes from the graph."""
        self._nodes.clear()
        self._root_ids.clear()
        self._by_type.clear()
        self._running_nodes.clear()

    def __len__(self) -> int:
        """Return number of nodes in graph."""
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        """Check if node exists in graph."""
        return node_id in self._nodes

    def __iter__(self) -> Iterator[ContextNode]:
        """Iterate over all nodes."""
        return iter(self._nodes.values())
