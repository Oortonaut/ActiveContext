"""File watcher registry for propagating external file changes to TextNodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph


@dataclass
class LineChange:
    """Represents a line-level change within a file.

    Attributes:
        line_no: 1-based line number where the change starts.
        num_removed: Number of lines removed starting at line_no.
        new_lines: Lines inserted at line_no (may be empty for pure deletions).
    """

    line_no: int
    num_removed: int
    new_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level file watcher registry
# ---------------------------------------------------------------------------
# Maps file path (str) -> set of TextNode IDs currently viewing that file.
# This provides a lightweight lookup for propagating external file changes
# to the correct TextNode instances without requiring a graph traversal.

_file_watchers: dict[str, set[str]] = {}


def register_file_watcher(file_path: str, node_id: str) -> None:
    """Register a node as watching a file.

    Args:
        file_path: The file path being watched.
        node_id: The TextNode ID to associate.
    """
    if file_path not in _file_watchers:
        _file_watchers[file_path] = set()
    _file_watchers[file_path].add(node_id)


def unregister_file_watcher(file_path: str, node_id: str) -> None:
    """Unregister a node from watching a file.

    If no nodes remain for a path, the path entry is removed.

    Args:
        file_path: The file path to stop watching.
        node_id: The TextNode ID to remove.
    """
    watchers = _file_watchers.get(file_path)
    if watchers is not None:
        watchers.discard(node_id)
        if not watchers:
            del _file_watchers[file_path]


def get_watchers(file_path: str) -> set[str]:
    """Get node IDs watching a file.

    Args:
        file_path: The file path to query.

    Returns:
        A *copy* of the set of node IDs (empty set if none).
    """
    return _file_watchers.get(file_path, set()).copy()


def on_file_change(
    file_path: str,
    changes: list[LineChange],
    *,
    graph: ContextGraph | None = None,
) -> list[str]:
    """Propagate line-level changes to all TextNodes watching a file.

    For each watching TextNode whose displayed range overlaps a change,
    ``replace_lines`` is called with coordinates adjusted to the node's
    local line space.

    Args:
        file_path: Path of the changed file.
        changes: Ordered list of ``LineChange`` descriptions.
        graph: Optional ``ContextGraph`` used to look up nodes by ID.
               When ``None``, only node IDs are returned without
               applying changes.

    Returns:
        List of node IDs that were notified / updated.
    """
    # Import here to avoid circular imports
    from activecontext.context.nodes.text import TextNode

    watcher_ids = get_watchers(file_path)
    notified: list[str] = []

    for node_id in watcher_ids:
        if graph is None:
            notified.append(node_id)
            continue

        node = graph.get_node(node_id)
        if not isinstance(node, TextNode):
            continue

        # Determine the node's displayed line range (1-based)
        try:
            node_start = int(node.pos.split(":")[0])
        except (ValueError, IndexError):
            node_start = 1

        if node.end_pos:
            try:
                node_end: int | None = int(node.end_pos.split(":")[0])
            except (ValueError, IndexError):
                node_end = None
        else:
            node_end = None

        for change in changes:
            change_end = (
                change.line_no + change.num_removed - 1 if change.num_removed else change.line_no
            )

            # Skip if change is entirely before the node's range
            if node_end is not None and change.line_no > node_end:
                continue
            # Skip if change is entirely after the node's range
            if change_end < node_start:
                continue

            # Map to node-local coordinates
            local_line = max(change.line_no - node_start + 1, 1)
            node.replace_lines(local_line, change.num_removed, list(change.new_lines))

        notified.append(node_id)

    return notified
