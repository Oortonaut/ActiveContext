"""Context node types for the context DAG.

This module defines the typed node hierarchy:
- ContextNode: Base class with common fields and notification
- TextNode: File content view (text)
- GroupNode: Summary facade over children
- TopicNode: Conversation segment
- ArtifactNode: Code/output artifact
- ShellNode: Async shell command execution
- MessageNode: Conversation message with ID for referencing
- MarkdownNode: Structured markdown with list parsing
- MarkdownListItemNode: Individual list items with nesting support
- FileSystemNode: Directory tree view with filtering
- ClockNode: Timer/countdown with tick-driven updates
- FunctionDocNode: Function signature and docstring extraction
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.agents.schema import AgentState
from activecontext.context.state import (
    Expansion,
    IOMode,
    NotificationLevel,
    TaskStatus,
    TickFrequency,
    WorkStatus,
)
from activecontext.context.traceable import trace_all_fields

if TYPE_CHECKING:
    pass  # Moved to runtime import below

import contextlib
import re as _re

from activecontext.context.graph import LinkedChildOrder
from activecontext.core.tokens import MediaType, detect_media_type

# Overhead tokens for the token counts display itself.
# The "(tokens: NNN / NN+NN+NN of NNN)" string occupies tokens in the header.
# This constant is added to header_tokens to account for that self-referential cost.
TOKEN_COUNTS_OVERHEAD = 12

# ---------------------------------------------------------------------------
# LineChange dataclass for file-level change propagation
# ---------------------------------------------------------------------------


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


class ShellStatus(Enum):
    """Status of a shell command execution."""

    PENDING = "pending"  # Created, not yet started
    RUNNING = "running"  # Subprocess is executing
    COMPLETED = "completed"  # Finished successfully (exit_code == 0)
    FAILED = "failed"  # Finished with error (exit_code != 0)
    TIMEOUT = "timeout"  # Killed due to timeout
    CANCELLED = "cancelled"  # Cancelled by user


class LockStatus(Enum):
    """Status of a file lock."""

    PENDING = "pending"  # Waiting to acquire lock
    ACQUIRED = "acquired"  # Lock held
    TIMEOUT = "timeout"  # Failed to acquire within timeout
    RELEASED = "released"  # Lock released
    ERROR = "error"  # Error during lock operation


class MessageRole(Enum):
    """Role of a message in the conversation history."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentRelation(Enum):
    """Relationship of an agent to the viewing agent."""

    SELF = "self"
    PARENT = "parent"
    CHILD = "child"
    PEER = "peer"


if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.context.headers import TokenInfo


# Type alias for hooks
# (parent_node, child_node, description)
OnChildChangedHook = Callable[["ContextNode", "ContextNode", str], None]


@trace_all_fields
@dataclass(kw_only=True)
class ContextNode(ABC):
    """Base class for all context DAG nodes.

    Attributes:
        node_id: Unique identifier (8-char UUID suffix)
        parent_ids: Set of parent node IDs (DAG allows multiple parents)
        child_order: Ordered children (LinkedChildOrder with O(1) add/remove/contains)
        state: Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL)
        mode: "paused" or "running" for tick processing
        tick_frequency: Tick frequency specification (turn, async, never, period)
        version: Incremented on change for trace detection
        created_at: Unix timestamp of creation
        updated_at: Unix timestamp of last update
        tags: Arbitrary metadata
        originator: Source of this node (node ID, filename, or arbitrary string)
        title: Human-readable title for display (empty = use default from render_digest)
        tracing: When True, state changes create TraceNode children
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: set[str] = field(default_factory=set)

    # Ordered children (doubly-linked list with O(1) add/remove/contains)
    child_order: LinkedChildOrder = field(default_factory=LinkedChildOrder, repr=False)

    # Rendering configuration
    default_expansion: Expansion = Expansion.ALL
    default_hidden: bool = False
    mode: str = "paused"
    tick_frequency: TickFrequency | None = None

    # Version tracking
    version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Originator: identifies the source of this node (node ID, filename, or arbitrary string)
    originator: str | None = None

    # Human-readable title for this node (used in headers and display)
    # If empty, render_digest() provides a default based on node type
    title: str = ""

    # Display sequence for uniform headers (e.g., text_1, message_13)
    # Assigned by ContextGraph.add_node() using per-type counters
    display_sequence: int | None = field(default=None)

    # Split architecture: optional reference to shared ContentData
    # When set, nodes can delegate content storage to ContentRegistry
    content_id: str | None = field(default=None)

    # Notification configuration
    # Controls how changes to this node are communicated to the agent
    notification_level: NotificationLevel = NotificationLevel.IGNORE
    is_subscription_point: bool = False  # If True, notifications stop here
    # Notification flags — set by _mark_changed on ancestors, cleared by view processing
    _notified: bool = field(default=False, init=False, repr=False)
    _wake_notified: bool = field(default=False, init=False, repr=False)

    # Tracing configuration
    # When True, state changes create TraceNode children for history
    tracing: bool = True

    # Trace sink for nodes without parents
    # When set, traces link to this node instead of being orphaned
    trace_sink: ContextNode | None = field(default=None, repr=False)

    # Graph reference (set by ContextGraph.add_node)
    _graph: ContextGraph | None = field(default=None, init=False, repr=False)

    # Optional hook for child change notifications
    _on_child_changed_hook: OnChildChangedHook | None = field(default=None, init=False, repr=False)

    # Trace merging state (for time-window based merging)
    _last_trace: TraceNode | None = field(default=None, init=False, repr=False)
    _last_trace_time: float = field(default=0.0, init=False, repr=False)

    # Cached children tokens (computed during projection collection)
    _cached_children_tokens: int = field(default=0, init=False, repr=False)

    @property
    @abstractmethod
    def node_type(self) -> str:
        """Return the node type identifier."""
        ...

    @property
    def header_tokens(self) -> int:
        """Tokens for the header line, including token counts overhead.

        Computed from get_token_breakdown().title (the metadata line)
        plus TOKEN_COUNTS_OVERHEAD for the token display string in the header.
        """
        from activecontext.context.headers import TOKEN_COUNTS_OVERHEAD

        return self.get_token_breakdown().title + TOKEN_COUNTS_OVERHEAD

    @property
    def content_tokens(self) -> int:
        """Tokens for this node's own content (excluding children).

        Delegates to get_token_breakdown().content + detail, which subclasses
        override to provide accurate counts.
        """
        breakdown = self.get_token_breakdown()
        return breakdown.content + breakdown.detail

    @property
    def index_tokens(self) -> int:
        """Sum of immediate children's header_tokens.

        Represents the cost of showing child headers in INDEX mode.
        Returns 0 for leaf nodes or when no graph reference.
        """
        if not self._graph:
            return 0
        total = 0
        for child_id in self.child_order:
            child = self._graph.get_node(child_id)
            if child:
                total += child.header_tokens
        return total

    @property
    def children_tokens(self) -> int:
        """Sum of children's total_tokens.

        This is computed during projection collection and cached.
        Returns 0 for leaf nodes or before collection.
        """
        return self._cached_children_tokens

    @property
    def all_tokens(self) -> int:
        """Total tokens: header + content + all recursive children.

        Provides complete token count for this subtree.
        """
        return self.header_tokens + self.content_tokens + self.children_tokens

    @property
    def total_tokens(self) -> int:
        """Alias for all_tokens. Total tokens for this subtree."""
        return self.all_tokens

    @abstractmethod
    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        ...

    @abstractmethod
    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for different visibility levels.

        Returns:
            TokenInfo with title, content, index, detail, and total token counts.
        """
        ...

    @abstractmethod
    def render_digest(self) -> str:
        """Render node metadata — the framework prepends the title line.

        Examples:
            TextNode: "main.py:1-50"
            ShellNode: "Shell: pytest [COMPLETED]"
            MessageNode: "User"
        """
        ...

    def Recompute(self) -> None:
        """Recompute this node's content. Called during tick for running nodes.

        Default implementation does nothing. Subclasses override to:
        1. Perform actual recomputation (reload file, update stats, etc.)
        2. Call _mark_changed() with a meaningful description if content changed

        Note: Don't generate traces here - only trace meaningful state changes.
        """
        pass

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render the content section — the actual content of this node.

        Subclasses override this to provide node-specific content.
        Base returns empty string (header-only nodes).
        """
        return ""

    def tick(self) -> None:
        """Synchronous state materialization point.

        Adapter for NodePlugin protocol. Delegates to Recompute().
        """
        self.Recompute()

    def get_digest(self) -> dict[str, Any]:
        """Return compact metadata for the handles dict.

        Adapter for NodePlugin protocol. Delegates to GetDigest().
        """
        return self.GetDigest()

    def mark_changed(
        self,
        description: str,
        *,
        content: str | None = None,
        originator: str | None = None,
        field: str = "",
        old: Any = "",
        new: Any = "",
    ) -> TraceNode | None:
        """Record a change, creating a trace and notifying ancestors.

        Called by @trace_all_fields for field changes, or manually for
        non-field changes (e.g., "content reloaded", "shell completed").

        Args:
            description: Human-readable description of the change
            content: Optional diff or detailed change content
            originator: Who/what caused the change (defaults to self.originator)
            field: Name of field that changed (e.g., "state", "content")
            old: Previous value (will be stringified)
            new: Current value (will be stringified)

        Returns:
            TraceNode if tracing is enabled, None otherwise
        """
        old_version = self.version
        self.version += 1
        self.updated_at = time.time()

        trace_node: TraceNode | None = None
        if self.tracing and self._graph and description:
            trace_node = self._create_trace(
                old_version=old_version,
                new_version=self.version,
                description=description,
                content=content,
                originator=originator,
                field_name=field,
                prev_value=str(old) if old != "" else "",
                curr_value=str(new) if new != "" else "",
            )

        self._mark_changed(trace_node, description, originator)
        return trace_node

    def _mark_changed(
        self,
        trace_node: TraceNode | None,
        description: str = "",
        originator: str | None = None,
    ) -> None:
        """Internal: deliver notification to ancestors and notify parents.

        Args:
            trace_node: The trace node (for notification dedup), or None.
            description: Human-readable description of the change.
            originator: Who/what caused the change.
        """
        # Set notification flags on ancestor nodes
        if self.notification_level != NotificationLevel.IGNORE and description:
            is_wake = self.notification_level == NotificationLevel.WAKE
            for ancestor in self.find_ancestors():
                ancestor._notified = True
                if is_wake:
                    ancestor._wake_notified = True

        self.notify_parents(description)

    # Time window for merging traces to the same node (seconds)
    TRACE_MERGE_WINDOW = 1.0

    def _create_trace(
        self,
        old_version: int,
        new_version: int,
        description: str,
        content: str | None = None,
        originator: str | None = None,
        field_name: str = "",
        prev_value: str = "",
        curr_value: str = "",
    ) -> TraceNode:
        """Create a TraceNode as a sibling of this node, with optional merging.

        Traces are linked to the same parent as this node, inserted immediately
        BEFORE this node in child_order. If this node has no parent but has a
        trace_sink set, the trace is linked to the trace_sink instead.

        If a recent trace exists (within TRACE_MERGE_WINDOW), the new change
        is merged into it rather than creating a new trace.

        Args:
            old_version: Version before the change
            new_version: Version after the change
            description: Human-readable change description
            content: Optional diff or detailed content
            originator: Who/what caused the change (defaults to self.originator)
            field_name: Name of field that changed
            prev_value: Previous value (stringified)
            curr_value: Current value (stringified)

        Returns:
            The created or merged TraceNode
        """
        if self._graph is None:
            raise RuntimeError("Node not attached to graph")

        now = time.time()

        # Check for existing trace to merge with (same node, within time window)
        if (
            self._last_trace is not None
            and self._last_trace.trace_target is self
            and (now - self._last_trace_time) < self.TRACE_MERGE_WINDOW
        ):
            # Merge into existing trace
            existing = self._last_trace
            if field_name == existing.field_name:
                # Same field - add to merged_values
                existing.merged_values.append(curr_value)
            else:
                # Different field - add as child trace
                child = TraceNode(
                    node=self.node_id,
                    node_display_id=self.node_id,
                    old_version=old_version,
                    new_version=new_version,
                    description=description,
                    content=content,
                    originator=originator or self.originator,
                    field_name=field_name,
                    prev_value=prev_value,
                    curr_value=curr_value,
                    default_expansion=Expansion.HEADER,
                )
                existing.child_traces.append(child)
            existing.new_version = new_version
            existing.updated_at = now
            self._last_trace_time = now
            return existing

        # Create new trace
        trace_node = TraceNode(
            node=self.node_id,
            node_display_id=self.node_id,
            old_version=old_version,
            new_version=new_version,
            description=description,
            content=content,
            originator=originator or self.originator,
            field_name=field_name,
            prev_value=prev_value,
            curr_value=curr_value,
            trace_target=self,  # Set for future merges
            default_expansion=Expansion.HEADER,
        )
        self._graph.add_node(trace_node)

        # Link as sibling (same parent) - insert BEFORE self so traces appear first
        if self.parent_ids:
            parent_id = next(iter(self.parent_ids))  # Primary parent
            self._graph.link(trace_node.node_id, parent_id, before=self.node_id)
        elif self.trace_sink is not None:
            self._graph.link(trace_node.node_id, self.trace_sink.node_id, before=self.node_id)
        # No orphan case - traces always have a parent or sink

        # Track for future merging
        self._last_trace = trace_node
        self._last_trace_time = now

        return trace_node

    def find_ancestors(self) -> list[ContextNode]:
        """Return all ancestor nodes (parents, grandparents, etc.).

        Uses a visited set to avoid cycles in the DAG.
        """
        if not self._graph:
            return []
        return self._graph.get_ancestors(self.node_id)

    def _format_notification_header(self, description: str) -> str:
        """Format brief notification header. Override in subclasses.

        Args:
            description: Human-readable description of the change.
        """
        return f"{self.node_id}: {description}"

    def notify_parents(self, description: str = "") -> None:
        """Notify all parent nodes of a change.

        Args:
            description: Human-readable description of the change.
        """
        if not self._graph:
            return

        for parent_id in self.parent_ids:
            parent = self._graph.get_node(parent_id)
            if parent:
                parent.on_child_changed(self, description)

    def on_child_changed(self, child: ContextNode, description: str = "") -> None:
        """Handle notification that a child has changed.

        Args:
            child: The child node that changed.
            description: Human-readable description of the change.

        Default implementation propagates upward. GroupNode overrides
        to invalidate summary and generate traces.
        """
        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, description)

        # Propagate upward
        self.notify_parents(description)

    def set_on_child_changed_hook(self, hook: OnChildChangedHook | None) -> None:
        """Register a hook for child change notifications.

        Args:
            hook: Callback function or None to unregister.
        """
        self._on_child_changed_hook = hook

    def add_child(self, child: ContextNode, *, after: str | None = None) -> bool:
        """Add a child node to this node.

        Delegates to graph.link() for proper cycle detection and root tracking.
        Both nodes must be in a graph.

        Args:
            child: The child node to add.
            after: If provided, insert in child_order immediately after this node_id.
                   If None, append to end.

        Returns:
            True if child was added, False if would create cycle.

        Raises:
            RuntimeError: If this node is not in a graph.
        """
        if not self._graph:
            raise RuntimeError(f"Cannot add_child: node {self.node_id} is not in a graph")

        return self._graph.link(child.node_id, self.node_id, after=after)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dict for persistence.

        Subclasses should override to include their specific fields.
        """
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "parent_ids": list(self.parent_ids),
            "child_order": self.child_order.to_list(),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "tick_frequency": self.tick_frequency.to_dict() if self.tick_frequency else None,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "originator": self.originator,
            "title": self.title,
            "content_id": self.content_id,
            "display_sequence": self.display_sequence,
            "notification_level": self.notification_level.value,
            "is_subscription_point": self.is_subscription_point,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextNode:
        """Deserialize node from dict.

        Args:
            data: Dictionary containing serialized node data.

        This is a factory method that uses the node type registry
        to dispatch to the appropriate subclass.
        """
        from activecontext.context.registry import get_node_registry

        return get_node_registry().from_dict(data)

    # Fluent API for mode control
    def Run(self, freq: TickFrequency | None = None) -> None:
        """Enable tick recomputation with given frequency.

        Args:
            freq: Tick frequency (defaults to turn() if not specified)

        Note: Mode change is auto-traced by @trace_all_fields.
        """
        self.mode = "running"  # Auto-traced
        self.tick_frequency = freq or TickFrequency.turn()
        if self._graph:
            self._graph._running_nodes.add(self.node_id)

    def Pause(self) -> None:
        """Disable tick recomputation.

        Note: Mode change is auto-traced by @trace_all_fields.
        """
        self.mode = "paused"  # Auto-traced
        if self._graph:
            self._graph._running_nodes.discard(self.node_id)

    def help(self) -> HelpNode:
        """Get or create a documentation node for this node type.

        Returns the existing HelpNode child if one has already been created,
        otherwise creates a new HelpNode, links it as a child, and returns it.

        The HelpNode extracts documentation from this node's class: docstrings,
        method signatures, @exposed members, and properties.

        Returns:
            HelpNode documenting this node type's API.

        Raises:
            RuntimeError: If this node is not attached to a graph.
        """
        if self._graph is None:
            raise RuntimeError(f"Cannot create help: node {self.node_id} is not in a graph")

        # Check if a HelpNode child already exists
        for child_id in self.child_order:
            child = self._graph.get_node(child_id)
            if isinstance(child, HelpNode) and child.parent_node_type == self.node_type:
                # Unhide if hidden
                if child.default_expansion == Expansion.HEADER:
                    child.default_expansion = Expansion.CONTENT
                return child

        # Create new HelpNode
        help_content = _extract_help_content(type(self))
        help_node = HelpNode(
            parent_node_type=self.node_type,
            _help_content=help_content,
            default_expansion=Expansion.CONTENT,
            tracing=False,
        )
        self._graph.add_node(help_node)
        self._graph.link(help_node.node_id, self.node_id)
        return help_node


@dataclass(kw_only=True)
class TextNode(ContextNode):
    """View of a file or file region as text.

    Attributes:
        path: File path relative to cwd
        pos: Start position as "line:col" (1-indexed)
        end_pos: End position as "line:col" (None = to end of file)
        media_type: Content media type (auto-detected from file extension)
        buffer_id: Optional reference to a shared TextBuffer in Session
        start_line: Start line when using TextBuffer (1-indexed)
        end_line: End line when using TextBuffer (1-indexed, inclusive)
        indent: Indentation level for rendering (e.g., for nested list items)
    """

    path: str = ""
    pos: str = "1:0"
    end_pos: str | None = None
    media_type: MediaType = field(default=MediaType.TEXT)

    # TextBuffer reference (optional, for shared line storage)
    buffer_id: str | None = None
    start_line: int = 1
    end_line: int | None = None

    # Indentation for markdown list processing
    indent: int = 0

    # Line rendering configuration
    line_prefix: str | None = "numbers"  # None = no line numbers, "numbers" = show
    line_divider: str = " | "  # Separator between line number and content

    # Summary caching (for LLM-generated summaries)
    cached_summary: str | None = None
    summary_stale: bool = True
    content_hash: str | None = None  # Hash of content for staleness detection

    def __post_init__(self) -> None:
        """Auto-detect media type from file extension and set originator."""
        if self.path and self.media_type == MediaType.TEXT:
            self.media_type = detect_media_type(self.path)
        # Auto-populate originator from path if not explicitly set
        if self.originator is None and self.path:
            self.originator = self.path
        # Auto-register with file watcher registry
        self.register_watcher()

    @property
    def node_type(self) -> str:
        return "text"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "path": self.path,
            "pos": self.pos,
            "end_pos": self.end_pos,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
            "media_type": self.media_type.value,
            "indent": self.indent,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render file content with line numbers.

        If a cached LLM summary exists, it is prepended before the file lines.

        Args:
            cwd: Working directory for resolving paths
            text_buffers: Optional dict of buffer_id -> TextBuffer for markdown nodes

        Returns:
            Rendered content string (without header — Render() prepends it)
        """
        import os

        output_parts: list[str] = []

        # Prepend cached summary if available
        if self.cached_summary and not self.summary_stale:
            output_parts.append(f"\n{self.cached_summary}\n")

        # Get lines either from buffer or from file
        lines: list[str] = []

        if self.buffer_id and text_buffers:
            # Use TextBuffer if available
            buffer = text_buffers.get(self.buffer_id)
            if buffer:
                # Get lines from buffer using start_line/end_line
                start_idx = max(0, self.start_line - 1)
                end_idx = self.end_line if self.end_line else len(buffer.lines)
                lines = buffer.lines[start_idx:end_idx]
        else:
            # Fall back to reading from file
            # Parse start position
            try:
                start_line = int(self.pos.split(":")[0])
            except (ValueError, IndexError):
                start_line = 1

            # Parse end position
            end_line: int | None = None
            if self.end_pos:
                with contextlib.suppress(ValueError, IndexError):
                    end_line = int(self.end_pos.split(":")[0])

            # Read file
            file_path = os.path.join(cwd, self.path)
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    file_lines = f.readlines()
            except FileNotFoundError:
                prefix = "".join(output_parts)
                return f"{prefix}[File not found: {self.path}]"
            except OSError as e:
                prefix = "".join(output_parts)
                return f"{prefix}[Error reading {self.path}: {e}]"

            # Apply line range
            start_idx = max(0, start_line - 1)
            end_idx = end_line if end_line else len(file_lines)
            lines = [line.rstrip("\n\r") for line in file_lines[start_idx:end_idx]]

        # Regular text rendering with line numbers
        # Calculate base line number
        if self.buffer_id:
            base_line = self.start_line
        else:
            try:
                base_line = int(self.pos.split(":")[0])
            except (ValueError, IndexError):
                base_line = 1

        for i, line in enumerate(lines):
            line_num = base_line + i
            # Ensure line doesn't have trailing newline for consistent formatting
            line_content = line.rstrip("\n\r") if isinstance(line, str) else line
            if self.line_prefix == "numbers":
                output_parts.append(f"{line_num:4d}{self.line_divider}{line_content}\n")
            else:
                output_parts.append(f"{line_content}\n")

        return "".join(output_parts)

    def SetPos(self, pos: str) -> TextNode:
        """Set start position.

        Args:
            pos: Position string in format "line:col" or "line".
        """
        old_pos = self.pos
        self.pos = pos
        if old_pos != pos:
            self.mark_changed(f"Position: {old_pos} → {pos}")
        return self

    def SetEndPos(self, end_pos: str | None) -> TextNode:
        """Set end position.

        Args:
            end_pos: End position string or None for end of file.
        """
        old_end = self.end_pos
        self.end_pos = end_pos
        if old_end != end_pos:
            old_str = old_end or "end"
            new_str = end_pos or "end"
            self.mark_changed(f"EndPos: {old_str} → {new_str}")
        return self

    def replace_lines(
        self,
        line_no: int,
        num_removed: int,
        new_lines: list[str],
    ) -> None:
        """Replace lines in the node's buffered content.

        Operates on the node's in-memory line content (via ``buffer_id`` or the
        internal ``_lines`` cache).  If the node has no in-memory lines yet, they
        are loaded from the associated file via ``render_content``'s file-reading
        path and cached in ``_lines``.

        Args:
            line_no: 1-based line number where replacement starts.
            num_removed: Number of lines to remove starting at *line_no*.
                         Use 0 for a pure insertion.
            new_lines: Lines to insert at *line_no*.  Pass an empty list for
                       a pure deletion.

        Raises:
            ValueError: If *line_no* is less than 1.
            IndexError: If *line_no* exceeds the current line count + 1
                        (i.e. you cannot skip past the end).
        """
        if line_no < 1:
            raise ValueError(f"line_no must be >= 1, got {line_no}")

        lines = self._get_lines_mut()
        # line_no is 1-based; convert to 0-based index
        idx = line_no - 1

        if idx > len(lines):
            raise IndexError(f"line_no {line_no} is beyond the end of content ({len(lines)} lines)")

        # Remove old lines and splice in new ones
        removed = lines[idx : idx + num_removed]
        lines[idx : idx + num_removed] = new_lines

        # Write back to the backing store
        self._set_lines(lines)

        # Build a human-readable change description
        n_ins = len(new_lines)
        n_del = len(removed)
        parts: list[str] = []
        if n_del:
            parts.append(f"-{n_del}")
        if n_ins:
            parts.append(f"+{n_ins}")
        desc = f"Lines {line_no}: {', '.join(parts)}" if parts else f"Lines {line_no}: no-op"

        self.mark_changed(desc)

    # -- internal helpers for replace_lines ----------------------------------

    def _get_lines_mut(self) -> list[str]:
        """Return a mutable list of lines for the node's current content.

        If the node uses a ``TextBuffer`` (``buffer_id`` is set) the buffer's
        line list is returned directly.  Otherwise the internal ``_lines``
        cache is returned (populated lazily from the file on disk).
        """
        # Fast path: use internal cache if already populated
        if hasattr(self, "_lines") and self._lines is not None:
            return self._lines

        # Buffer-backed path
        if self.buffer_id:
            # Caller is expected to pass text_buffers through the session;
            # for replace_lines we only operate on _lines.
            pass

        # Lazy init from nothing (no file read here — callers provide content
        # or a buffer supplies it).
        self._lines: list[str] = []
        return self._lines

    def _set_lines(self, lines: list[str]) -> None:
        """Persist the mutated lines back to the backing store."""
        self._lines = lines

    # -- file watcher integration -------------------------------------------

    def register_watcher(self) -> None:
        """Register this node as watching its file path.

        Called automatically from ``__post_init__`` when ``path`` is set.
        """
        if self.path:
            register_file_watcher(self.path, self.node_id)

    def unregister_watcher(self) -> None:
        """Unregister this node from the file watcher registry.

        Should be called during node cleanup / removal.
        """
        if self.path:
            unregister_file_watcher(self.path, self.node_id)

    def _format_notification_header(self, description: str) -> str:
        """Format header with line position info for text nodes."""
        return f"{self.node_id}: {description} (at {self.pos})"

    def _parse_start_line(self) -> int:
        """Extract start line number from pos string."""
        try:
            return int(self.pos.split(":")[0])
        except (ValueError, IndexError):
            return 1

    def _parse_end_line(self) -> int | None:
        """Extract end line number from end_pos string."""
        if self.end_pos:
            try:
                return int(self.end_pos.split(":")[0])
            except (ValueError, IndexError):
                pass
        return None

    def render_digest(self) -> str:
        """Return title if set, otherwise 'path:start-end' format."""
        if self.title:
            return self.title

        start_line = self._parse_start_line()

        if self.end_pos:
            end_line = self._parse_end_line()
            if end_line is not None:
                return f"{self.path}:{start_line}-{end_line}"

        # Build line range caption
        line_range = ""
        start = self.start_line if self.buffer_id else self._parse_start_line()
        end = self.end_line if self.buffer_id else self._parse_end_line()
        if start and end:
            line_range = f"(lines {start}-{end})"
        elif start and start > 1:
            line_range = f"(line {start})"

        return f"{line_range}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: just metadata line
        collapsed_text = f"[{self.path}: lines, pending traces]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: cached summary if present
        summary_tokens = 0
        if self.cached_summary:
            summary_tokens = count_tokens(self.cached_summary)

        # Detail: estimate from line count (~10 tokens/line with line numbers)
        detail_tokens = 0
        if self.end_line and self.start_line:
            line_count = max(0, self.end_line - self.start_line + 1)
            detail_tokens = line_count * 10

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize TextNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "path": self.path,
                "pos": self.pos,
                "end_pos": self.end_pos,
                "media_type": self.media_type.value,
                "indent": self.indent,
                "cached_summary": self.cached_summary,
                "summary_stale": self.summary_stale,
                "content_hash": self.content_hash,
                "start_line": self.start_line,
                "end_line": self.end_line,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TextNode:
        """Deserialize TextNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        # Parse media_type, default to TEXT
        media_type_str = data.get("media_type", "text")
        try:
            media_type = MediaType(media_type_str)
        except ValueError:
            media_type = MediaType.TEXT


        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            path=data.get("path", ""),
            pos=data.get("pos", "1:0"),
            end_pos=data.get("end_pos"),
            media_type=media_type,
            indent=data.get("indent", 0),
            cached_summary=data.get("cached_summary"),
            summary_stale=data.get("summary_stale", True),
            content_hash=data.get("content_hash"),
            start_line=data.get("start_line", 1),
            end_line=data.get("end_line"),
        )
        return node


@dataclass(kw_only=True)
class GroupNode(ContextNode):
    """Summary facade over child nodes.

    Attributes:
        child_order: Ordered list of child node IDs (document order)
        summary_prompt: Custom prompt for LLM summarization
        cached_summary: Cached LLM-generated summary
        summary_stale: Whether summary needs regeneration
        last_child_versions: Version tracking for trace detection
    """

    summary_prompt: str | None = None
    cached_summary: str | None = None
    summary_stale: bool = True
    last_child_versions: dict[str, int] = field(default_factory=dict)

    @property
    def node_type(self) -> str:
        return "group"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "member_count": len(self.child_order),
            "child_order": self.child_order.to_list(),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
            "summary_stale": self.summary_stale,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render cached summary or empty string.

        Children are rendered by the projection engine, not here.
        """
        if self.cached_summary and not self.summary_stale:
            return self.cached_summary
        return ""

    def on_child_changed(self, child: ContextNode, description: str = "") -> None:
        """Handle child change: track version, mark summary stale, propagate."""
        old_version = self.last_child_versions.get(child.node_id, 0)
        new_version = child.version

        if new_version != old_version:
            self.summary_stale = True
            self.last_child_versions[child.node_id] = new_version

        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, description)

        # Propagate upward
        self.notify_parents(description)

    def invalidate_summary(self) -> None:
        """Mark summary as needing regeneration."""
        self.summary_stale = True

    def render_digest(self) -> str:
        """Return 'Group (N members)' format."""
        return f"Group ({len(self.child_order)} members)"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Use child_order for iteration
        ordered_children = self.child_order

        # Collapsed: member count line
        collapsed_text = f"[Group: {len(ordered_children)} members]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: cached summary if present
        summary_tokens = 0
        if self.cached_summary:
            summary_tokens = count_tokens(self.cached_summary)

        # Detail: children total (recursive)
        child_total = 0
        if self._graph:
            for child_id in ordered_children:
                child = self._graph.get_node(child_id)
                if child:
                    child_info = child.get_token_breakdown()
                    child_total += child_info.title + child_info.content + child_info.detail

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=0,  # Group has no detail of its own
            total=collapsed_tokens + summary_tokens + child_total if child_total else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize GroupNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "summary_prompt": self.summary_prompt,
                "cached_summary": self.cached_summary,
                "summary_stale": self.summary_stale,
                "last_child_versions": self.last_child_versions,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> GroupNode:
        """Deserialize GroupNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])


        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "content")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            summary_prompt=data.get("summary_prompt"),
            cached_summary=data.get("cached_summary"),
            summary_stale=data.get("summary_stale", True),
            last_child_versions=data.get("last_child_versions", {}),
        )
        return node


@dataclass(kw_only=True)
class TopicNode(ContextNode):
    """Represents a conversation topic/thread.

    Attributes:
        title: Short title for the topic
        message_indices: Indices into session._message_history
        status: "active", "resolved", or "deferred"
    """

    title: str = ""
    message_indices: list[int] = field(default_factory=list)
    status: str = "active"

    @property
    def node_type(self) -> str:
        return "topic"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "title": self.title,
            "message_count": len(self.message_indices),
            "status": self.status,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render message range and artifact count."""
        parts: list[str] = []
        if self.message_indices:
            parts.append(f"Messages: {self.message_indices[0]}-{self.message_indices[-1]}\n")
        if self.child_order:
            parts.append(f"Contains {len(self.child_order)} artifacts\n")
        return "".join(parts)

    def set_status(self, status: str) -> TopicNode:
        """Set topic status.

        Args:
            status: New status string (e.g., "active", "resolved", "pending").
        """
        old_status = self.status
        self.status = status
        if old_status != status:
            self.mark_changed(f"Topic status: {old_status} → {status}")
        return self

    def render_digest(self) -> str:
        """Return 'Topic: title' format."""
        return f"Topic: {self.title}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: topic metadata
        collapsed_text = f"[Topic: {self.title} [{self.status}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Topics don't have summary vs detail distinction
        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize TopicNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "title": self.title,
                "message_indices": self.message_indices,
                "status": self.status,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TopicNode:
        """Deserialize TopicNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            message_indices=data.get("message_indices", []),
            status=data.get("status", "active"),
        )
        return node


@dataclass(kw_only=True)
class ArtifactNode(ContextNode):
    """Represents a generated artifact (code, output, error, file).

    Attributes:
        artifact_type: "code", "output", "error", or "file"
        content: The artifact content
        language: Programming language (for code artifacts)
        source_statement_id: ID of statement that created this artifact
    """

    artifact_type: str = "code"
    content: str = ""
    language: str | None = None
    source_statement_id: str | None = None

    @property
    def node_type(self) -> str:
        return "artifact"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "artifact_type": self.artifact_type,
            "language": self.language,
            "content_length": len(self.content),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full artifact content."""
        return self.content

    def set_content(self, content: str) -> ArtifactNode:
        """Update artifact content.

        Args:
            content: New content string to replace existing content.
        """
        old_content = self.content
        self.content = content
        self.mark_changed(
            f"Content updated ({len(old_content)} → {len(content)} chars)",
        )
        return self

    def render_digest(self) -> str:
        """Return 'TYPE:language' format."""
        lang_suffix = f":{self.language}" if self.language else ""
        return f"{self.artifact_type.upper()}{lang_suffix}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: artifact metadata
        lang_info = f":{self.language}" if self.language else ""
        collapsed_text = f"[{self.artifact_type}{lang_info}: {len(self.content)} chars]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: full content
        detail_tokens = count_tokens(self.content)

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ArtifactNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "artifact_type": self.artifact_type,
                "content": self.content,
                "language": self.language,
                "source_statement_id": self.source_statement_id,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ArtifactNode:
        """Deserialize ArtifactNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            artifact_type=data.get("artifact_type", "code"),
            content=data.get("content", ""),
            language=data.get("language"),
            source_statement_id=data.get("source_statement_id"),
        )
        return node


@dataclass(kw_only=True)
class ShellNode(ContextNode):
    """Represents an async shell command execution.

    The shell() DSL function creates a ShellNode and starts the subprocess
    in the background. The node's status changes as the command progresses,
    and change notifications propagate up the DAG.

    Attributes:
        command: The command being executed (e.g., "pytest")
        args: Command arguments (e.g., ["-v", "tests/"])
        shell_status: Current execution status (PENDING, RUNNING, COMPLETED, etc.)
        exit_code: Process exit code (None until completed)
        output: Combined stdout/stderr output
        truncated: Whether output was truncated
        signal: Signal name if killed (e.g., "SIGKILL")
        duration_ms: Execution duration in milliseconds
        started_at_exec: When execution actually started (vs node creation)
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    shell_status: ShellStatus = ShellStatus.PENDING
    exit_code: int | None = None
    output: str = ""
    truncated: bool = False
    signal: str | None = None
    duration_ms: float = 0.0
    started_at_exec: float | None = None

    @property
    def node_type(self) -> str:
        return "shell"

    @property
    def is_complete(self) -> bool:
        """True if shell command has finished (success, failure, timeout, or cancelled)."""
        return self.shell_status in (
            ShellStatus.COMPLETED,
            ShellStatus.FAILED,
            ShellStatus.TIMEOUT,
            ShellStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        """True if shell command completed successfully."""
        return self.shell_status == ShellStatus.COMPLETED and self.exit_code == 0

    @property
    def full_command(self) -> str:
        """Full command string with arguments."""
        if self.args:
            return f"{self.command} {' '.join(self.args)}"
        return self.command

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "command": self.full_command,
            "status": self.shell_status.value,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full output with timing details."""
        result = self.output
        if result and not result.endswith("\n"):
            result += "\n"

        result += f"--- Duration: {self.duration_ms:.0f}ms"
        if self.truncated:
            result += " (output was truncated)"
        if self.signal:
            result += f", killed by {self.signal}"
        result += " ---\n"

        return result

    def set_running(self) -> ShellNode:
        """Mark as running (called when subprocess starts)."""
        old_status = self.shell_status
        self.shell_status = ShellStatus.RUNNING
        self.started_at_exec = time.time()
        self.mark_changed(f"Shell: {old_status.value} → running")
        return self

    def set_completed(
        self,
        exit_code: int,
        output: str,
        duration_ms: float,
        truncated: bool = False,
        signal: str | None = None,
    ) -> ShellNode:
        """Mark as completed with result (called when subprocess finishes).

        Args:
            exit_code: Process exit code (0 = success).
            output: Captured stdout/stderr output.
            duration_ms: Execution time in milliseconds.
            truncated: Whether output was truncated.
            signal: Signal name if killed (e.g., "SIGTERM").
        """
        self.exit_code = exit_code
        self.output = output
        self.duration_ms = duration_ms
        self.truncated = truncated
        self.signal = signal

        if signal:
            self.shell_status = ShellStatus.CANCELLED
        elif exit_code == 0:
            self.shell_status = ShellStatus.COMPLETED
        else:
            self.shell_status = ShellStatus.FAILED

        self.mark_changed(
            f"Shell '{self.command}' {self.shell_status.value} (exit={exit_code})",
            content=output[:500] if output else None,
        )
        return self

    def set_timeout(self, output: str, duration_ms: float) -> ShellNode:
        """Mark as timed out.

        Args:
            output: Partial output captured before timeout.
            duration_ms: Time elapsed before timeout in milliseconds.
        """
        self.shell_status = ShellStatus.TIMEOUT
        self.output = output
        self.duration_ms = duration_ms
        self.exit_code = -1
        self.mark_changed(
            f"Shell '{self.command}' timed out after {duration_ms:.0f}ms",
        )
        return self

    def set_cancelled(self) -> ShellNode:
        """Mark as cancelled by user."""
        self.shell_status = ShellStatus.CANCELLED
        self.mark_changed(
            f"Shell '{self.command}' cancelled",
        )
        return self

    def render_digest(self) -> str:
        """Return 'Shell: command [STATUS]' format."""
        cmd_display = (
            self.full_command[:40] + "..." if len(self.full_command) > 40 else self.full_command
        )
        return f"Shell: {cmd_display} [{self.shell_status.value.upper()}]"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: command and status
        collapsed_text = f"[Shell: {self.full_command} [{self.shell_status.value}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: output content
        detail_tokens = count_tokens(self.output) if self.output else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ShellNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "command": self.command,
                "args": self.args,
                "shell_status": self.shell_status.value,
                "exit_code": self.exit_code,
                "output": self.output,
                "truncated": self.truncated,
                "signal": self.signal,
                "duration_ms": self.duration_ms,
                "started_at_exec": self.started_at_exec,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ShellNode:
        """Deserialize ShellNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            command=data.get("command", ""),
            args=data.get("args", []),
            shell_status=ShellStatus(data.get("shell_status", "pending")),
            exit_code=data.get("exit_code"),
            output=data.get("output", ""),
            truncated=data.get("truncated", False),
            signal=data.get("signal"),
            duration_ms=data.get("duration_ms", 0.0),
            started_at_exec=data.get("started_at_exec"),
        )
        return node


class PtyStatus(Enum):
    """Status of an interactive PTY session."""

    PENDING = "pending"  # Created, not yet spawned
    RUNNING = "running"  # PTY process is alive
    EXITED = "exited"  # Process exited normally
    KILLED = "killed"  # Force-killed or signalled
    ERROR = "error"  # Failed to spawn or internal error


# Max lines / bytes kept in the PTY scrollback ring buffer.
_PTY_MAX_LINES = 500
_PTY_MAX_BYTES = 100_000

# Regex that matches a single ANSI/VT100 escape sequence.
_ANSI_RE = _re.compile(
    r"\x1b(?:\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]|\][^\x07]*\x07|[\(\)][AB012])"
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


@trace_all_fields
@dataclass(kw_only=True)
class PtyNode(ContextNode):
    """Represents a long-lived interactive PTY session.

    The ``pty()`` DSL function creates a PtyNode and spawns the process in
    the background.  Output arrives incrementally via ``append_output()``
    (called by PtyManager at tick boundaries after Nagle batching).

    The scrollback is a ring buffer capped at ``_PTY_MAX_LINES`` lines and
    ``_PTY_MAX_BYTES`` total bytes.  ``render_content()`` returns the most
    recent ~50 lines with ANSI escapes stripped so the projection stays
    compact and token-friendly.

    Attributes:
        command: The command (e.g., ``"gdb"``, ``"python"``)
        args: Command arguments
        pty_status: Current lifecycle status
        exit_code: Process exit code (None until exited)
        signal: Signal name if killed (e.g., ``"SIGTERM"``)
        input_history: Lines the agent has sent via ``pty_send``
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    pty_status: PtyStatus = PtyStatus.PENDING
    exit_code: int | None = None
    signal: str | None = None

    # Input tracking — records what the agent has sent
    input_history: list[str] = field(default_factory=list)

    # Ring-buffer scrollback (not serialized; rebuilt from output on load)
    _scrollback_lines: list[str] = field(default_factory=list, init=False, repr=False)
    _scrollback_bytes: int = field(default=0, init=False, repr=False)
    _total_line_count: int = field(default=0, init=False, repr=False)

    # Raw output accumulator (serialized for session persistence)
    _raw_output: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        # PTY nodes default to CONTENT expansion (scrollback only, not ALL)
        if self.default_expansion == Expansion.ALL:
            object.__setattr__(self, "default_expansion", Expansion.CONTENT)

    # -- Node identity --------------------------------------------------------

    @property
    def node_type(self) -> str:
        return "pty"

    @property
    def is_complete(self) -> bool:
        """True when the PTY session has ended."""
        return self.pty_status in (PtyStatus.EXITED, PtyStatus.KILLED, PtyStatus.ERROR)

    @property
    def full_command(self) -> str:
        if self.args:
            return f"{self.command} {' '.join(self.args)}"
        return self.command

    # -- Output management (ring buffer) --------------------------------------

    def append_output(self, text: str) -> None:
        """Append new output text to the scrollback ring buffer.

        Splits on newlines, trims oldest lines when limits are exceeded,
        and calls ``mark_changed`` once per batch.
        """
        if not text:
            return

        self._raw_output += text

        new_lines = text.split("\n")
        # If the last element is empty it means text ended with \\n;
        # don't add an extra blank line.
        if new_lines and new_lines[-1] == "":
            new_lines.pop()

        for line in new_lines:
            self._scrollback_lines.append(line)
            self._scrollback_bytes += len(line) + 1  # +1 for implicit newline
            self._total_line_count += 1

        # Trim ring buffer
        while (
            len(self._scrollback_lines) > _PTY_MAX_LINES
            or self._scrollback_bytes > _PTY_MAX_BYTES
        ) and self._scrollback_lines:
            dropped = self._scrollback_lines.pop(0)
            self._scrollback_bytes -= len(dropped) + 1

        self.mark_changed(
            f"PTY output ({len(new_lines)} lines)",
            content=text[:500],
        )

    # -- Input tracking -------------------------------------------------------

    def record_input(self, text: str) -> None:
        """Record an input line sent by the agent."""
        self.input_history.append(text)

    # -- Lifecycle transitions ------------------------------------------------

    def set_running(self) -> PtyNode:
        """PENDING → RUNNING (called when backend.spawn() succeeds)."""
        self.pty_status = PtyStatus.RUNNING
        self.mark_changed(f"PTY '{self.command}' running")
        return self

    def set_exited(self, code: int, signal_name: str | None = None) -> PtyNode:
        """Mark the session as exited.

        Args:
            code: Process exit code.
            signal_name: Signal name if killed externally.
        """
        self.exit_code = code
        self.signal = signal_name
        if signal_name:
            self.pty_status = PtyStatus.KILLED
        elif code == 0:
            self.pty_status = PtyStatus.EXITED
        else:
            self.pty_status = PtyStatus.EXITED

        self.mark_changed(
            f"PTY '{self.command}' {self.pty_status.value} (exit={code})",
        )
        return self

    def set_error(self, message: str) -> PtyNode:
        """Mark the session as failed with an error message."""
        self.pty_status = PtyStatus.ERROR
        self.append_output(f"\n[ERROR] {message}\n")
        self.mark_changed(f"PTY '{self.command}' error: {message}")
        return self

    # -- Rendering ------------------------------------------------------------

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "command": self.full_command,
            "status": self.pty_status.value,
            "exit_code": self.exit_code,
            "lines": self._total_line_count,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_digest(self) -> str:
        cmd_display = (
            self.full_command[:40] + "..." if len(self.full_command) > 40 else self.full_command
        )
        return f"PTY: {cmd_display} [{self.pty_status.value.upper()}]"

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render the most recent scrollback lines (ANSI-stripped)."""
        # Show last 50 lines of scrollback
        tail = self._scrollback_lines[-50:]
        cleaned = [_strip_ansi(line) for line in tail]
        content = "\n".join(cleaned)
        if content and not content.endswith("\n"):
            content += "\n"

        # Append status footer
        if self.is_complete:
            content += f"--- exit_code={self.exit_code}"
            if self.signal:
                content += f", signal={self.signal}"
            content += " ---\n"
        elif self.pty_status == PtyStatus.RUNNING:
            hidden = self._total_line_count - len(tail)
            if hidden > 0:
                content += f"--- {hidden} earlier lines omitted ---\n"

        return content

    def get_token_breakdown(self) -> TokenInfo:
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        collapsed_text = f"[PTY: {self.full_command} [{self.pty_status.value}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: last 10 lines
        summary_lines = self._scrollback_lines[-10:]
        summary_text = "\n".join(_strip_ansi(line) for line in summary_lines)
        summary_tokens = count_tokens(summary_text) if summary_text else 0

        # Detail: remaining scrollback beyond summary
        detail_lines = self._scrollback_lines[:-10] if len(self._scrollback_lines) > 10 else []
        detail_text = "\n".join(_strip_ansi(line) for line in detail_lines[-40:])
        detail_tokens = count_tokens(detail_text) if detail_text else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=detail_tokens,
        )

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "command": self.command,
                "args": self.args,
                "pty_status": self.pty_status.value,
                "exit_code": self.exit_code,
                "signal": self.signal,
                "input_history": self.input_history,
                "raw_output": self._raw_output,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> PtyNode:
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "content")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            command=data.get("command", ""),
            args=data.get("args", []),
            pty_status=PtyStatus(data.get("pty_status", "pending")),
            exit_code=data.get("exit_code"),
            signal=data.get("signal"),
            input_history=data.get("input_history", []),
        )
        # Rebuild scrollback from persisted raw output
        raw = data.get("raw_output", "")
        if raw:
            # Bypass mark_changed during deserialization
            node._raw_output = raw
            lines = raw.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            node._scrollback_lines = lines[-_PTY_MAX_LINES:]
            node._scrollback_bytes = sum(len(ln) + 1 for ln in node._scrollback_lines)
            node._total_line_count = len(lines)
        return node


@dataclass(kw_only=True)
class LockNode(ContextNode):
    """Represents an async file lock acquisition.

    The lock_file() DSL function creates a LockNode and starts the lock
    acquisition in the background. The node's status changes as the lock
    is acquired or times out, and change notifications propagate up the DAG.

    Attributes:
        lockfile: Path to the lock file
        lock_status: Current lock status (PENDING, ACQUIRED, TIMEOUT, etc.)
        timeout: Maximum time to wait for lock acquisition (seconds)
        error_message: Error details if lock failed
        acquired_at: When the lock was acquired
        holder_pid: PID holding the lock (this process when acquired)
    """

    lockfile: str = ""
    lock_status: LockStatus = LockStatus.PENDING
    timeout: float = 30.0
    error_message: str | None = None
    acquired_at: float | None = None
    holder_pid: int | None = None

    @property
    def node_type(self) -> str:
        return "lock"

    @property
    def is_complete(self) -> bool:
        """True if lock operation has finished (acquired, timeout, released, or error)."""
        return self.lock_status in (
            LockStatus.ACQUIRED,
            LockStatus.TIMEOUT,
            LockStatus.RELEASED,
            LockStatus.ERROR,
        )

    @property
    def is_held(self) -> bool:
        """True if lock is currently held by this process."""
        return self.lock_status == LockStatus.ACQUIRED

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "lockfile": self.lockfile,
            "status": self.lock_status.value,
            "timeout": self.timeout,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render timeout, holder, error, and acquired_at."""
        parts: list[str] = []
        parts.append(f"Timeout: {self.timeout}s\n")

        if self.holder_pid:
            parts.append(f"Holder PID: {self.holder_pid}\n")

        if self.error_message:
            parts.append(f"Error: {self.error_message}\n")

        if self.acquired_at:
            parts.append(f"Acquired at: {self.acquired_at:.3f}\n")

        return "".join(parts)

    def set_acquired(self, pid: int) -> LockNode:
        """Mark lock as acquired."""
        self.lock_status = LockStatus.ACQUIRED
        self.acquired_at = time.time()
        self.holder_pid = pid
        self.mark_changed(
            f"Lock '{self.lockfile}' acquired by PID {pid}",
        )
        return self

    def set_timeout(self) -> LockNode:
        """Mark lock acquisition as timed out."""
        self.lock_status = LockStatus.TIMEOUT
        self.error_message = f"Timed out after {self.timeout}s"
        self.mark_changed(
            f"Lock '{self.lockfile}' timed out",
        )
        return self

    def set_released(self) -> LockNode:
        """Mark lock as released."""
        self.lock_status = LockStatus.RELEASED
        self.mark_changed(
            f"Lock '{self.lockfile}' released",
        )
        return self

    def set_error(self, message: str) -> LockNode:
        """Mark lock operation as failed with error."""
        self.lock_status = LockStatus.ERROR
        self.error_message = message
        self.mark_changed(
            f"Lock '{self.lockfile}' error: {message}",
        )
        return self

    def render_digest(self) -> str:
        """Return 'Lock: file [STATUS]' format."""
        return f"Lock: {self.lockfile} [{self.lock_status.value.upper()}]"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: lock info
        collapsed_text = f"[Lock: {self.lockfile} [{self.lock_status.value}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: error message if present
        detail_tokens = count_tokens(self.error_message) if self.error_message else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize LockNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "lockfile": self.lockfile,
                "lock_status": self.lock_status.value,
                "timeout": self.timeout,
                "error_message": self.error_message,
                "acquired_at": self.acquired_at,
                "holder_pid": self.holder_pid,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LockNode:
        """Deserialize LockNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "header")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            lockfile=data.get("lockfile", ""),
            lock_status=LockStatus(data.get("lock_status", "pending")),
            timeout=data.get("timeout", 30.0),
            error_message=data.get("error_message"),
            acquired_at=data.get("acquired_at"),
            holder_pid=data.get("holder_pid"),
        )
        return node


@dataclass(kw_only=True)
class SessionNode(ContextNode):
    """Represents session-level metadata for agent situational awareness.

    This node is auto-created as a root node in each session and placed early
    in the context projection. It provides the LLM agent with visibility into:
    - Recent token usage trends (to adjust verbosity)
    - Execution timing statistics (to notice slow operations)
    - Context graph composition (what's currently loaded)
    - Cumulative session statistics

    The node updates automatically each tick with TickFrequency.turn().

    Attributes:
        token_history: Token counts for last N turns
        token_min: Minimum tokens in history
        token_max: Maximum tokens in history
        token_avg: Average tokens in history
        turn_durations_ms: Execution time for last N turns
        time_min_ms: Minimum turn duration
        time_max_ms: Maximum turn duration
        total_statements_executed: Cumulative statement count
        total_tokens_consumed: Cumulative token usage
        session_start_time: Unix timestamp when session started
        node_count_by_type: Current graph composition by node type
        running_node_count: Number of nodes in "running" mode
        graph_depth: Maximum depth of the context graph
        recent_actions: Short descriptions of last N actions
    """

    # Token tracking (rolling window)
    token_history: list[int] = field(default_factory=list)
    token_min: int = 0
    token_max: int = 0
    token_avg: float = 0.0

    # Timing records (rolling window)
    turn_durations_ms: list[float] = field(default_factory=list)
    time_min_ms: float = 0.0
    time_max_ms: float = 0.0

    # Cumulative stats
    total_statements_executed: int = 0
    total_tokens_consumed: int = 0
    session_start_time: float = field(default_factory=time.time)
    turn_count: int = 0

    # Context graph snapshot
    node_count_by_type: dict[str, int] = field(default_factory=dict)
    running_node_count: int = 0
    graph_depth: int = 0

    # Recent actions (short descriptions)
    recent_actions: list[str] = field(default_factory=list)

    # Configuration
    history_depth: int = 10  # How many turns to track

    @property
    def node_type(self) -> str:
        return "session"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens_consumed,
            "total_statements": self.total_statements_executed,
            "running_nodes": self.running_node_count,
            "graph_depth": self.graph_depth,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render turn info, token info, graph stats, and recent actions."""
        parts: list[str] = []

        # Turn and timing info
        if self.turn_durations_ms:
            last_duration = self.turn_durations_ms[-1]
            parts.append(
                f"Turn: {self.turn_count} | Duration: {last_duration:.0f}ms "
                f"(avg: {sum(self.turn_durations_ms) / len(self.turn_durations_ms):.0f}ms, "
                f"range: {self.time_min_ms:.0f}-{self.time_max_ms:.0f}ms)\n"
            )
        else:
            parts.append(f"Turn: {self.turn_count}\n")

        # Token info
        if self.token_history:
            last_tokens = self.token_history[-1]
            parts.append(
                f"Tokens: {last_tokens:,} this turn "
                f"(avg: {self.token_avg:,.0f}, range: {self.token_min:,}-{self.token_max:,})\n"
            )
        parts.append(
            f"Total: {self.total_tokens_consumed:,} tokens across {self.turn_count} turns\n"
        )

        # Context graph summary
        if self.node_count_by_type:
            parts.append("\nContext Graph:\n")
            for node_type, count in sorted(self.node_count_by_type.items()):
                running_info = ""
                if node_type == "text" and self.running_node_count > 0:
                    running_info = f" ({self.running_node_count} running)"
                parts.append(f"  {node_type}s: {count}{running_info}\n")
            if self.graph_depth > 0:
                parts.append(f"  depth: {self.graph_depth}\n")

        # Recent actions
        if self.recent_actions:
            parts.append("\nRecent Actions:\n")
            for i, action in enumerate(self.recent_actions[-5:]):
                turn_idx = self.turn_count - (len(self.recent_actions[-5:]) - i - 1)
                parts.append(f"  [T{turn_idx}] {action}\n")

        return "".join(parts)

    def record_turn(
        self,
        tokens_used: int,
        duration_ms: float,
        action_description: str | None = None,
    ) -> SessionNode:
        """Record statistics for a completed turn.

        Args:
            tokens_used: Tokens consumed this turn
            duration_ms: Turn execution time in milliseconds
            action_description: Optional short description of the turn's main action
        """
        self.turn_count += 1
        self.total_tokens_consumed += tokens_used

        # Update token history (rolling window)
        self.token_history.append(tokens_used)
        if len(self.token_history) > self.history_depth:
            self.token_history = self.token_history[-self.history_depth :]

        # Update token stats
        if self.token_history:
            self.token_min = min(self.token_history)
            self.token_max = max(self.token_history)
            self.token_avg = sum(self.token_history) / len(self.token_history)

        # Update timing history (rolling window)
        self.turn_durations_ms.append(duration_ms)
        if len(self.turn_durations_ms) > self.history_depth:
            self.turn_durations_ms = self.turn_durations_ms[-self.history_depth :]

        # Update timing stats
        if self.turn_durations_ms:
            self.time_min_ms = min(self.turn_durations_ms)
            self.time_max_ms = max(self.turn_durations_ms)

        # Record action
        if action_description:
            self.recent_actions.append(action_description)
            if len(self.recent_actions) > self.history_depth:
                self.recent_actions = self.recent_actions[-self.history_depth :]

        desc = f"Turn {self.turn_count}: {tokens_used} tokens"
        if action_description:
            desc += f" - {action_description[:50]}"
        self.mark_changed(desc)
        return self

    def record_statement(self) -> SessionNode:
        """Record that a statement was executed."""
        self.total_statements_executed += 1
        return self

    def update_graph_stats(self) -> SessionNode:
        """Update context graph statistics from the attached graph."""
        if not self._graph:
            return self

        # Count nodes by type
        type_counts: dict[str, int] = {}
        running_count = 0
        max_depth = 0

        for node_id in self._graph._nodes:
            node = self._graph.get_node(node_id)
            if node:
                node_type = node.node_type
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
                if node.mode == "running":
                    running_count += 1

        # Calculate graph depth via BFS from roots
        root_nodes = self._graph.get_roots()
        if root_nodes:
            visited: set[str] = set()
            queue: deque[tuple[str, int]] = deque((r.node_id, 1) for r in root_nodes)
            while queue:
                nid, depth = queue.popleft()
                if nid in visited:
                    continue
                visited.add(nid)
                max_depth = max(max_depth, depth)
                node = self._graph.get_node(nid)
                if node:
                    for child_id in node.child_order:
                        if child_id not in visited:
                            queue.append((child_id, depth + 1))

        self.node_count_by_type = type_counts
        self.running_node_count = running_count
        self.graph_depth = max_depth
        return self

    def Recompute(self) -> None:
        """Recompute graph statistics on tick."""
        self.update_graph_stats()
        super().Recompute()

    def render_digest(self) -> str:
        """Return 'Session' format."""
        return "Session"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: session metadata line
        collapsed_text = (
            f"[Session: Turn {self.turn_count} | {self.total_tokens_consumed:,} tokens]\n"
        )
        collapsed_tokens = count_tokens(collapsed_text)

        # Session node has statistics as detail
        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize SessionNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "token_history": self.token_history,
                "token_min": self.token_min,
                "token_max": self.token_max,
                "token_avg": self.token_avg,
                "turn_durations_ms": self.turn_durations_ms,
                "time_min_ms": self.time_min_ms,
                "time_max_ms": self.time_max_ms,
                "total_statements_executed": self.total_statements_executed,
                "total_tokens_consumed": self.total_tokens_consumed,
                "session_start_time": self.session_start_time,
                "turn_count": self.turn_count,
                "node_count_by_type": self.node_count_by_type,
                "running_node_count": self.running_node_count,
                "graph_depth": self.graph_depth,
                "recent_actions": self.recent_actions,
                "history_depth": self.history_depth,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SessionNode:
        """Deserialize SessionNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "running"),  # Default to running for session node
            tick_frequency=tick_freq or TickFrequency.turn(),
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            token_history=data.get("token_history", []),
            token_min=data.get("token_min", 0),
            token_max=data.get("token_max", 0),
            token_avg=data.get("token_avg", 0.0),
            turn_durations_ms=data.get("turn_durations_ms", []),
            time_min_ms=data.get("time_min_ms", 0.0),
            time_max_ms=data.get("time_max_ms", 0.0),
            total_statements_executed=data.get("total_statements_executed", 0),
            total_tokens_consumed=data.get("total_tokens_consumed", 0),
            session_start_time=data.get("session_start_time", time.time()),
            turn_count=data.get("turn_count", 0),
            node_count_by_type=data.get("node_count_by_type", {}),
            running_node_count=data.get("running_node_count", 0),
            graph_depth=data.get("graph_depth", 0),
            recent_actions=data.get("recent_actions", []),
            history_depth=data.get("history_depth", 10),
        )
        return node


@dataclass(kw_only=True)
class MessageNode(ContextNode):
    """Represents a message in the conversation history.

    MessageNodes are automatically created when messages are added to the
    conversation. They enable:
    - ID-based referencing of messages (e.g., [msg:abc123])
    - Proper role alternation for LLM pretraining compatibility
    - Block merging of adjacent same-role content

    Attributes:
        role: Message role ("user", "assistant", "tool_call", "tool_result")
        content: The message content
        originator: (inherited) Who produced this message (e.g., "user", "agent", "tool:grep")
        tool_name: Tool name for tool_call/tool_result messages
        tool_args: Tool arguments (for tool_call messages)
        content_type: Content type ("text", "image", "audio", etc.)
        mime_type: MIME type (e.g., "image/png", "audio/wav") or None for text
    """

    role: MessageRole = MessageRole.USER
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    content_type: str = "text"  # "text", "image", "audio", etc.
    mime_type: str | None = None  # MIME type (e.g., "image/png")
    processed: bool = False  # Whether this message has been processed by the agent loop

    @property
    def node_type(self) -> str:
        return "message"

    @property
    def effective_role(self) -> str:
        """Return the role for LLM alternation (USER or ASSISTANT)."""
        return "USER" if self.originator == "user" else "ASSISTANT"

    @property
    def display_label(self) -> str:
        """Return the human-friendly label for this message.

        Mapping:
        - originator="user" → configured user name (default "User")
        - originator="agent" → "Agent"
        - originator="agent:plan" → "Agent (Plan)"
        - originator="agent:{name}" → "Child: {name}"
        - originator="tool:{name}" with role=tool_call → "Tool Call: {name}"
        - originator="tool:{name}" with role=tool_result → "Tool Result"
        """
        if not self.originator:
            return "Unknown"

        if self.originator == "user":
            return "User"  # Will be overridden by config at render time

        if self.originator == "agent":
            return "Agent"

        if self.originator == "agent:plan":
            return "Agent (Plan)"

        if self.originator.startswith("agent:"):
            subagent_name = self.originator[6:]  # Remove "agent:" prefix
            return f"Child: {subagent_name}"

        if self.originator.startswith("tool:"):
            tool_name = self.originator[5:]  # Remove "tool:" prefix
            if self.role == MessageRole.TOOL_CALL:
                return f"Tool Call: {tool_name}"
            elif self.role == MessageRole.TOOL_RESULT:
                return "Tool Result"
            return f"Tool: {tool_name}"

        return self.originator

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "role": self.role.value,
            "originator": self.originator,
            "effective_role": self.effective_role,
            "content_length": len(self.content),
            "tool_name": self.tool_name,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
            "content_type": self.content_type,
            "mime_type": self.mime_type,
        }

    def _get_formatted_content(self) -> str:
        """Get formatted content based on message type."""
        if self.role == MessageRole.TOOL_CALL:
            return self._format_tool_call()
        elif self.role == MessageRole.TOOL_RESULT:
            return self._format_tool_result()
        return self.content

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full message content."""
        content = self._get_formatted_content()
        return content + "\n"

    def _format_tool_call(self) -> str:
        """Format a tool call message."""
        parts = [f"[Tool: {self.tool_name or 'unknown'}]"]
        if self.tool_args:
            args_str = ", ".join(f'{k}="{v}"' for k, v in self.tool_args.items())
            parts.append(f" {args_str}")
        return "".join(parts)

    def _format_tool_result(self) -> str:
        """Format a tool result message."""
        return f"[Result] {self.content}"

    def set_content(self, content: str) -> MessageNode:
        """Update message content."""
        old_len = len(self.content)
        self.content = content
        self.mark_changed(
            f"Message content updated ({old_len} → {len(content)} chars)",
        )
        return self

    def render_digest(self) -> str:
        """Return 'Role #N' format using display_sequence."""
        seq = self.display_sequence or 0
        # Use actual role for display (user, assistant, tool_call, tool_result)
        role_display = self.role.value.replace("_", " ").title()
        return f"{role_display} #{seq}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: role and char count
        collapsed_text = f"[{self.role.value.upper()}: {len(self.content)} chars]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: full message content
        detail_tokens = count_tokens(self.content)

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MessageNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "role": self.role.value,
                "content": self.content,
                "tool_name": self.tool_name,
                "tool_args": self.tool_args,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MessageNode:
        """Deserialize MessageNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        originator = data.get("originator")

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=originator,
            title=data.get("title", ""),
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args", {}),
        )
        return node


@dataclass(kw_only=True)
class WorkNode(ContextNode):
    """Represents this agent's work coordination entry.

    This node shows what files the agent is working on and any conflicts
    with other agents working on the same project. It integrates with the
    ScratchpadManager to provide visibility into multi-agent coordination.

    Attributes:
        intent: Human-readable description of current work
        work_status: Current work status (active, paused, done)
        files: List of file paths being accessed with mode (read/write)
        dependencies: Files needed but not modified
        conflicts: Detected conflicts with other agents
        agent_id: This agent's unique ID
    """

    intent: str = ""
    work_status: WorkStatus = WorkStatus.ACTIVE
    files: list[dict[str, str]] = field(default_factory=list)  # [{path, mode}]
    dependencies: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(
        default_factory=list
    )  # [{agent_id, file, their_mode, their_intent}]
    agent_id: str = ""

    @property
    def node_type(self) -> str:
        return "work"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "intent": self.intent,
            "status": self.work_status.value,
            "file_count": len(self.files),
            "conflict_count": len(self.conflicts),
            "agent_id": self.agent_id,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render agent, files, dependencies, and conflicts."""
        parts: list[str] = []
        parts.append(f"Agent: {self.agent_id}\n")

        # Show files being worked on
        if self.files:
            parts.append("\nFiles:\n")
            for f in self.files:
                mode_indicator = "[W]" if f.get("mode") == "write" else "[R]"
                parts.append(f"  {mode_indicator} {f.get('path', '')}\n")

        # Show dependencies
        if self.dependencies:
            parts.append("\nDependencies:\n")
            for dep in self.dependencies:
                parts.append(f"  [R] {dep}\n")

        # Show conflicts
        if self.conflicts:
            parts.append("\n--- CONFLICTS ---\n")
            for c in self.conflicts:
                parts.append(
                    f"  Agent {c.get('agent_id', '?')}: {c.get('file', '?')} "
                    f"[{c.get('their_mode', '?')}] - {c.get('their_intent', '?')}\n"
                )

        return "".join(parts)

    def set_intent(self, intent: str) -> WorkNode:
        """Update work intent."""
        old_intent = self.intent
        self.intent = intent
        if old_intent != intent:
            self.mark_changed(f"Intent: {old_intent[:30]}... → {intent[:30]}...")
        return self

    def set_files(self, files: list[dict[str, str]]) -> WorkNode:
        """Update files being worked on."""
        old_count = len(self.files)
        self.files = files
        if old_count != len(files):
            self.mark_changed(f"Files: {old_count} → {len(files)}")
        return self

    def set_conflicts(self, conflicts: list[dict[str, str]]) -> WorkNode:
        """Update detected conflicts."""
        old_count = len(self.conflicts)
        self.conflicts = conflicts
        if len(conflicts) != old_count:
            self.mark_changed(
                f"Work conflicts: {old_count} → {len(conflicts)}",
            )
        return self

    def render_digest(self) -> str:
        """Return 'Work: intent [status]' format."""
        intent_display = self.intent[:30] + "..." if len(self.intent) > 30 else self.intent
        return f"Work: {intent_display} [{self.work_status.value}]"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: work metadata
        nf, nc = len(self.files), len(self.conflicts)
        collapsed_text = f"[Work: {self.intent} [{self.work_status.value}] {nf}f {nc}c]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: file list and conflict details
        detail_parts: list[str] = []
        for f in self.files:
            detail_parts.append(f"  {f.get('path', '?')} ({f.get('access', '?')})")
        for c in self.conflicts:
            detail_parts.append(f"  CONFLICT: {c}")
        detail_tokens = count_tokens("\n".join(detail_parts)) if detail_parts else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize WorkNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "intent": self.intent,
                "work_status": self.work_status.value,
                "files": self.files,
                "dependencies": self.dependencies,
                "conflicts": self.conflicts,
                "agent_id": self.agent_id,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> WorkNode:
        """Deserialize WorkNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            intent=data.get("intent", ""),
            work_status=WorkStatus(data.get("work_status", "active")),
            files=data.get("files", []),
            dependencies=data.get("dependencies", []),
            conflicts=data.get("conflicts", []),
            agent_id=data.get("agent_id", ""),
        )
        return node


@dataclass(kw_only=True)
class MCPServerNode(ContextNode):
    """Represents an MCP server connection with its available tools.

    Renders tool documentation for the LLM to understand available capabilities.
    The LLM can call tools via server.tool_name(**kwargs) in the namespace.

    Attributes:
        server_name: Unique name identifying the MCP server
        status: Connection status (disconnected, connecting, connected, error)
        error_message: Error message if status is "error"
        tools: List of available tools with name, description, input_schema
        resources: List of available resources with uri, name, description
        prompts: List of available prompts with name, description, arguments

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: "MCP: server_name [OK] (X tools)"
        - SUMMARY: Server + tool names list
        - DETAILS: Tool names + brief descriptions
        - ALL: Full documentation with JSON schemas
    """

    server_name: str = ""
    status: str = "disconnected"  # disconnected, connecting, connected, error
    error_message: str | None = None
    tools: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    prompts: list[dict[str, Any]] = field(default_factory=list)

    # Enable notifications for MCP server state changes
    notification_level: NotificationLevel = NotificationLevel.HOLD

    # Pending async tool calls: call_id -> (tool_name, started_at)
    pending_calls: dict[str, tuple[str, float]] = field(default_factory=dict)

    # Callback for firing events when calls complete
    # Set by Timeline: (event_name, data) -> None
    _on_result_callback: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # Tool child nodes: tool_name -> node_id
    _tool_nodes: dict[str, str] = field(default_factory=dict, repr=False)

    # Runtime reference to server proxy for tool calls (not serialized)
    _server_proxy: Any = field(default=None, init=False, repr=False, compare=False)

    def __getattr__(self, name: str) -> Any:
        """Delegate tool method access to the server proxy."""
        proxy = self.__dict__.get("_server_proxy")
        if proxy is not None:
            try:
                return getattr(proxy, name)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def node_type(self) -> str:
        return "mcp_server"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "server_name": self.server_name,
            "status": self.status,
            "tool_count": len(self.tools),
            "resource_count": len(self.resources),
            "prompt_count": len(self.prompts),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def tool(self, name: str) -> MCPToolNode | None:
        """Get a tool child node by name.

        Args:
            name: Tool name (e.g., "read_file")

        Returns:
            MCPToolNode if found, None otherwise
        """
        node_id = self._tool_nodes.get(name)
        if node_id and self._graph:
            node = self._graph.get_node(node_id)
            if isinstance(node, MCPToolNode):
                return node
        return None

    @property
    def tool_nodes(self) -> list[MCPToolNode]:
        """Get all tool child nodes."""
        if not self._graph:
            return []
        nodes = []
        for node_id in self._tool_nodes.values():
            node = self._graph.get_node(node_id)
            if isinstance(node, MCPToolNode):
                nodes.append(node)
        return nodes

    def _render_status_message(self) -> str | None:
        """Return status message for error/disconnected states, or None if connected."""
        if self.status == "error" and self.error_message:
            return f"Error: {self.error_message}\n\n"
        if self.status != "connected":
            return f"Status: {self.status}\n"
        return None

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render tool names, usage hint, resources, and prompts."""
        status = self._render_status_message()
        if status:
            return status

        parts: list[str] = []
        tool_names = [t.get("name", "?") for t in self.tools]
        parts.append(f"Tools: {', '.join(tool_names)}\n")

        # Usage hint
        parts.append("To call tools:\n")
        parts.append("```python/acrepl\n")
        parts.append(f"result = {self.server_name}.tool_name(arg=value)\n")
        parts.append("```\n\n")

        # Resources
        if self.resources:
            parts.append("Resources:\n")
            for res in self.resources:
                parts.append(f"- `{res.get('uri', '?')}`")
                if res.get("description"):
                    parts.append(f": {res['description']}")
                parts.append("\n")

        # Prompts
        if self.prompts:
            parts.append("\nPrompts:\n")
            for prompt in self.prompts:
                parts.append(f"- **{prompt.get('name', '?')}**")
                if prompt.get("description"):
                    parts.append(f": {prompt['description']}")
                parts.append("\n")

        return "".join(parts)

    def update_from_connection(self, connection: Any) -> None:
        """Update node state from an MCPConnection object.

        Creates/updates/removes MCPToolNode children based on tool changes.
        Generates traces for removed tools to maintain audit trail.
        """
        self.status = connection.status.value
        self.error_message = connection.error_message

        # Build incoming tool data
        incoming_tools: dict[str, dict[str, Any]] = {}
        for t in connection.tools:
            incoming_tools[t.name] = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }

        # Update tools list (kept for backward compat / Render fallback)
        self.tools = list(incoming_tools.values())

        # Diff tool child nodes if graph is available
        if self._graph:
            current_tool_names = set(self._tool_nodes.keys())
            incoming_tool_names = set(incoming_tools.keys())

            # Remove tools that no longer exist
            removed = current_tool_names - incoming_tool_names
            for tool_name in removed:
                node_id = self._tool_nodes.pop(tool_name)
                node = self._graph.get_node(node_id)
                if node:
                    # Generate trace before removal
                    node.mark_changed(f"Tool '{tool_name}' removed from {self.server_name}")
                    self._graph.remove_node(node_id)

            # Add new tools
            added = incoming_tool_names - current_tool_names
            for tool_name in added:
                tool_data = incoming_tools[tool_name]
                tool_node = MCPToolNode(
                    tool_name=tool_name,
                    server_name=self.server_name,
                    description=tool_data["description"],
                    input_schema=tool_data["input_schema"],
                    default_expansion=Expansion.HEADER,
                )
                self._graph.add_node(tool_node)
                self._graph.link(tool_node.node_id, self.node_id)
                self._tool_nodes[tool_name] = tool_node.node_id

            # Update existing tools if schema/description changed
            unchanged = current_tool_names & incoming_tool_names
            for tool_name in unchanged:
                node_id = self._tool_nodes[tool_name]
                node = self._graph.get_node(node_id)
                if isinstance(node, MCPToolNode):
                    tool_data = incoming_tools[tool_name]
                    changed = False
                    if node.description != tool_data["description"]:
                        node.description = tool_data["description"]
                        changed = True
                    if node.input_schema != tool_data["input_schema"]:
                        node.input_schema = tool_data["input_schema"]
                        changed = True
                    if changed:
                        node.mark_changed(f"Tool '{tool_name}' schema updated")

        # Update resources and prompts (no child nodes for these yet)
        self.resources = [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
            }
            for r in connection.resources
        ]
        self.prompts = [
            {
                "name": p.name,
                "description": p.description,
                "arguments": p.arguments,
            }
            for p in connection.prompts
        ]
        self.mark_changed(f"MCP {self.server_name}: {self.status}, {len(self.tools)} tools")

    def start_call(self, call_id: str, tool_name: str) -> None:
        """Register a pending async tool call.

        Args:
            call_id: Unique ID for this call
            tool_name: Name of the tool being called
        """
        self.pending_calls[call_id] = (tool_name, time.time())

    def complete_call(
        self,
        call_id: str,
        result: Any,
        error: str | None = None,
    ) -> None:
        """Complete an async tool call and fire event.

        Args:
            call_id: ID of the call to complete
            result: Tool result (if successful)
            error: Error message (if failed)
        """
        call_info = self.pending_calls.pop(call_id, None)
        if not call_info:
            return

        tool_name, started_at = call_info
        duration_ms = (time.time() - started_at) * 1000

        # Mark the node as changed
        self.mark_changed(
            f"MCP tool '{tool_name}' completed",
            content=str(result)[:200] if result else error,
        )

        # Fire event if callback is set
        if self._on_result_callback:
            self._on_result_callback(
                "mcp_result",
                {
                    "call_id": call_id,
                    "server_name": self.server_name,
                    "tool_name": tool_name,
                    "result": result,
                    "error": error,
                    "duration_ms": duration_ms,
                },
            )

    def get_pending_count(self) -> int:
        """Get the number of pending async calls."""
        return len(self.pending_calls)

    def set_on_result_callback(
        self, callback: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        """Set the callback for MCP result events.

        Args:
            callback: Function to call with (event_name, data) when a call completes
        """
        self._on_result_callback = callback

    def render_digest(self) -> str:
        """Return 'MCP: name [status]' format."""
        return f"MCP: {self.server_name} [{self.status.upper()}]"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: server info
        collapsed_text = f"[MCP: {self.server_name} [{self.status}] {len(self.tools)} tools]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: tool names
        summary_text = ", ".join(t.get("name", "?") for t in self.tools)
        summary_tokens = count_tokens(summary_text) if summary_text else 0

        # Detail: full tool documentation (descriptions + schemas)
        detail_parts: list[str] = []
        for t in self.tools:
            name = t.get("name", "?")
            desc = t.get("description", "")
            detail_parts.append(f"  {name}: {desc}")
            schema = t.get("inputSchema") or t.get("input_schema")
            if schema:
                import json

                detail_parts.append(f"    {json.dumps(schema)}")
        detail_tokens = count_tokens("\n".join(detail_parts)) if detail_parts else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCPServerNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "server_name": self.server_name,
                "status": self.status,
                "error_message": self.error_message,
                "tools": self.tools,
                "resources": self.resources,
                "prompts": self.prompts,
                "_tool_nodes": self._tool_nodes,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MCPServerNode:
        """Deserialize MCPServerNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])


        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            server_name=data.get("server_name", ""),
            status=data.get("status", "disconnected"),
            error_message=data.get("error_message"),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            prompts=data.get("prompts", []),
            _tool_nodes=data.get("_tool_nodes", {}),
        )
        return node


@dataclass(kw_only=True)
class MCPToolNode(ContextNode):
    """Represents an individual tool from an MCP server.

    Child node of MCPServerNode, displaying tool name, description, and schema.
    Each tool node has independent state control for granular visibility.

    Attributes:
        tool_name: Name of the tool (e.g., "read_file")
        server_name: Parent server name for context
        description: Tool description
        input_schema: JSON Schema for tool parameters

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: Just tool_name
        - SUMMARY: tool_name: description (truncated)
        - DETAILS: Name, description, required params
        - ALL: Full JSON schema
    """

    tool_name: str = ""
    server_name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> str:
        return "mcp_tool"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "has_schema": bool(self.input_schema.get("properties")),
            "expansion": self.default_expansion.value,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render description and parameters (no headings — header via Render())."""
        parts: list[str] = []

        # Description
        parts.append(f"{self.description}\n\n")

        props = self.input_schema.get("properties", {})
        if props:
            required = set(self.input_schema.get("required", []))
            parts.append("**Parameters:**\n")
            for param, param_schema in props.items():
                param_type = param_schema.get("type", "any")
                param_desc = param_schema.get("description", "")
                req_marker = " (required)" if param in required else ""
                parts.append(f"- `{param}` ({param_type}){req_marker}: {param_desc}\n")
            parts.append("\n")

        return "".join(parts)

    def render_digest(self) -> str:
        """Return tool name for display."""
        return f"{self.server_name}.{self.tool_name}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        collapsed_tokens = count_tokens(f"`{self.tool_name}`\n")

        desc = self.description[:80] if len(self.description) > 80 else self.description
        summary_tokens = count_tokens(f"**{self.tool_name}**: {desc}\n")

        # Detail: full description + input schema
        detail_parts: list[str] = [self.description]
        if self.input_schema:
            import json

            detail_parts.append(json.dumps(self.input_schema))
        detail_tokens = count_tokens("\n".join(detail_parts))

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCPToolNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "tool_name": self.tool_name,
                "server_name": self.server_name,
                "description": self.description,
                "input_schema": self.input_schema,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MCPToolNode:
        """Deserialize MCPToolNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "header")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            tool_name=data.get("tool_name", ""),
            server_name=data.get("server_name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
        )


@dataclass(kw_only=True)
class MCPManagerNode(ContextNode):
    """Singleton manager that tracks all MCP server connections.

    This node aggregates state from all MCPServerNode children and tracks
    connection state changes, tool changes, and resource changes as traces.

    The manager is created automatically and has a fixed node_id="mcp_manager".
    Multiple observer nodes can reference it via the context graph.

    Attributes:
        server_states: Dict mapping server name to last known status
        tool_counts: Dict mapping server name to tool count
        resource_counts: Dict mapping server name to resource count
        connection_events: Recent connection state changes (for rendering)

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: "MCP Manager: X servers (Y connected)"
        - SUMMARY: Server status list
        - DETAILS: Server status + tool/resource counts
        - ALL: Full details + recent connection events
    """

    # Track last known state for diff generation
    server_states: dict[str, str] = field(default_factory=dict)  # name -> status
    tool_counts: dict[str, int] = field(default_factory=dict)  # name -> count
    resource_counts: dict[str, int] = field(default_factory=dict)

    # Recent events for rendering
    connection_events: list[dict[str, Any]] = field(default_factory=list)
    max_events: int = 10

    @property
    def node_type(self) -> str:
        return "mcp_manager"

    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        total = len(self.server_states)
        connected = sum(1 for s in self.server_states.values() if s == "connected")
        return {
            "id": self.node_id,
            "type": self.node_type,
            "total_servers": total,
            "connected_servers": connected,
            "server_states": dict(self.server_states),
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render server status, capabilities, and events."""
        lines: list[str] = []

        # Server status list
        if self.server_states:
            lines.append("Server Status:")
            for name, status in sorted(self.server_states.items()):
                emoji = {
                    "connected": "[OK]",
                    "connecting": "[...]",
                    "error": "[ERR]",
                    "disconnected": "[--]",
                }.get(status, "[?]")
                lines.append(f"- {name} {emoji}")
        else:
            lines.append("No MCP servers configured.")

        # Capabilities
        if self.tool_counts:
            lines.append("")
            lines.append("Capabilities:")
            for name in sorted(self.server_states.keys()):
                tools = self.tool_counts.get(name, 0)
                resources = self.resource_counts.get(name, 0)
                lines.append(f"- {name}: {tools} tools, {resources} resources")

        # Recent events
        if self.connection_events:
            lines.append("")
            lines.append("Recent Events:")
            for event in self.connection_events[-5:]:
                lines.append(f"- {event.get('time', '?')}: {event.get('message', '?')}")

        return "\n".join(lines)

    def on_child_changed(self, child: ContextNode, description: str = "") -> None:
        """Handle MCPServerNode changes - track state transitions."""
        if not isinstance(child, MCPServerNode):
            return

        name = child.server_name
        old_status = self.server_states.get(name)
        new_status = child.status
        changes: list[str] = []

        # Track state change
        if old_status != new_status:
            self.server_states[name] = new_status
            changes.append(f"MCP '{name}': {old_status or 'new'} -> {new_status}")

            # Record event
            import time as time_module

            self.connection_events.append(
                {
                    "time": time_module.strftime("%H:%M:%S"),
                    "server": name,
                    "message": f"{name}: {old_status or 'new'} -> {new_status}",
                }
            )
            if len(self.connection_events) > self.max_events:
                self.connection_events.pop(0)

        # Track tool/resource count changes
        old_tools = self.tool_counts.get(name, 0)
        new_tools = len(child.tools)
        if old_tools != new_tools:
            self.tool_counts[name] = new_tools
            changes.append(f"MCP '{name}' tools: {old_tools} -> {new_tools}")

        old_resources = self.resource_counts.get(name, 0)
        new_resources = len(child.resources)
        if old_resources != new_resources:
            self.resource_counts[name] = new_resources

        change_desc = "; ".join(changes) if changes else description
        self.mark_changed(change_desc)
        self.notify_parents(change_desc)

    def register_server(self, server_node: MCPServerNode) -> None:
        """Register a server node as a child of this manager."""
        self.server_states[server_node.server_name] = server_node.status
        self.tool_counts[server_node.server_name] = len(server_node.tools)
        self.resource_counts[server_node.server_name] = len(server_node.resources)

    def unregister_server(self, server_name: str) -> None:
        """Remove a server from tracking."""
        self.server_states.pop(server_name, None)
        self.tool_counts.pop(server_name, None)
        self.resource_counts.pop(server_name, None)

    def render_digest(self) -> str:
        """Return 'MCP Manager' format."""
        return f"MCP Manager ({len(self.server_states)} servers)"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: server count
        total_tools = sum(self.tool_counts.values())
        collapsed_text = f"[MCP Manager: {len(self.server_states)} servers, {total_tools} tools]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: server names and statuses
        summary_lines = [f"{name}:{status}" for name, status in self.server_states.items()]
        summary_tokens = count_tokens(" ".join(summary_lines)) if summary_lines else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        d = super().to_dict()
        d.update(
            {
                "server_states": dict(self.server_states),
                "tool_counts": dict(self.tool_counts),
                "resource_counts": dict(self.resource_counts),
                "connection_events": list(self.connection_events),
                "max_events": self.max_events,
            }
        )
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> MCPManagerNode:
        """Deserialize from dict."""
        # Parse tick_frequency if present
        tick_freq = None
        if d.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(d["tick_frequency"])

        node = cls(
            node_id=d.get("node_id", "mcp_manager"),
            parent_ids=set(d.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                d.get("child_order") or d.get("children_ids") or []
            ),
            default_expansion=Expansion(d.get("expansion", "content")),
            mode=d.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=d.get("version", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            display_sequence=d.get("display_sequence"),
            originator=d.get("originator"),
            title=d.get("title", ""),
            server_states=d.get("server_states", {}),
            tool_counts=d.get("tool_counts", {}),
            resource_counts=d.get("resource_counts", {}),
            connection_events=d.get("connection_events", []),
            max_events=d.get("max_events", 10),
        )
        return node


@dataclass(kw_only=True)
class PluginManagerNode(ContextNode):
    """Singleton manager that tracks plugin server connections and available plugins.

    This node aggregates state from the PluginManager and provides visibility
    into loaded vs. available plugins, connection status, and plugin metadata.

    The manager is created automatically and has a fixed node_id="plugin_manager".
    Multiple observer nodes can reference it via the context graph.

    Attributes:
        plugin_states: Dict mapping plugin server name to connection status
        plugin_types: Dict mapping plugin server to list of provided node types
        builtin_count: Number of builtin node types
        loaded_count: Number of currently loaded plugin servers

    Rendering states:
        - HIDDEN: Not shown in projection
        - COLLAPSED: "Plugins: X builtin, Y loaded"
        - SUMMARY: List of loaded plugin types with descriptions
        - DETAILS: Full plugin info including paths, versions
        - ALL: Full details + connection events
    """

    # Track plugin connection state
    plugin_states: dict[str, str] = field(default_factory=dict)  # name -> status
    plugin_types: dict[str, list[str]] = field(default_factory=dict)  # name -> node_types

    # Metadata
    builtin_count: int = 0
    loaded_count: int = 0

    # Recent events for rendering
    connection_events: list[dict[str, Any]] = field(default_factory=list)
    max_events: int = 10

    @property
    def node_type(self) -> str:
        return "plugin_manager"

    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        return {
            "id": self.node_id,
            "type": self.node_type,
            "builtin_count": self.builtin_count,
            "loaded_count": self.loaded_count,
            "plugin_states": dict(self.plugin_states),
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render overview, plugin list, and events."""
        lines: list[str] = []

        # Overview
        lines.append("Overview:")
        lines.append(f"- Builtin types: {self.builtin_count}")
        lines.append(f"- Loaded plugin servers: {self.loaded_count}")

        # Plugin status list with details
        if self.plugin_states:
            lines.append("")
            lines.append("Loaded Plugins:")
            for name, status in sorted(self.plugin_states.items()):
                emoji = {
                    "connected": "[OK]",
                    "connecting": "[...]",
                    "error": "[ERR]",
                    "disconnected": "[--]",
                }.get(status, "[?]")
                types = self.plugin_types.get(name, [])
                lines.append(f"- **{name}** {emoji}")
                lines.append(f"  - Status: {status}")
                lines.append(f"  - Node types ({len(types)}): {', '.join(types)}")
        else:
            lines.append("")
            lines.append("No plugin servers loaded.")

        # Recent events
        if self.connection_events:
            lines.append("")
            lines.append("Recent Events:")
            for event in self.connection_events[-5:]:
                lines.append(f"- {event.get('time', '?')}: {event.get('message', '?')}")

        return "\n".join(lines)

    def update_plugin_state(
        self, name: str, status: str, node_types: list[str] | None = None
    ) -> None:
        """Update the state of a plugin connection.

        Args:
            name: Plugin server name
            status: Connection status (connected, connecting, error, disconnected)
            node_types: Optional list of node types provided by this plugin
        """
        old_status = self.plugin_states.get(name)
        self.plugin_states[name] = status

        if node_types is not None:
            self.plugin_types[name] = node_types

        # Update loaded count
        self.loaded_count = sum(1 for s in self.plugin_states.values() if s == "connected")

        # Record event if status changed
        if old_status != status:
            import time as time_module

            self.connection_events.append(
                {
                    "time": time_module.strftime("%H:%M:%S"),
                    "plugin": name,
                    "message": f"{name}: {old_status or 'new'} -> {status}",
                }
            )
            if len(self.connection_events) > self.max_events:
                self.connection_events.pop(0)

            self.mark_changed(f"Plugin '{name}': {old_status or 'new'} -> {status}")

    def unregister_plugin(self, name: str) -> None:
        """Remove a plugin from tracking."""
        self.plugin_states.pop(name, None)
        self.plugin_types.pop(name, None)
        self.loaded_count = sum(1 for s in self.plugin_states.values() if s == "connected")
        self.mark_changed(f"Plugin '{name}' unregistered")

    def render_digest(self) -> str:
        """Return 'Plugin Manager' format."""
        return f"Plugin Manager ({self.builtin_count} builtin, {self.loaded_count} loaded)"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: counts
        collapsed_text = (
            f"[Plugin Manager: {self.builtin_count} builtin, {self.loaded_count} loaded]\n"
        )
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: plugin names and types
        summary_lines = [f"{name}: {', '.join(types)}" for name, types in self.plugin_types.items()]
        summary_tokens = count_tokens(" ".join(summary_lines)) if summary_lines else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        d = super().to_dict()
        d.update(
            {
                "plugin_states": dict(self.plugin_states),
                "plugin_types": {k: list(v) for k, v in self.plugin_types.items()},
                "builtin_count": self.builtin_count,
                "loaded_count": self.loaded_count,
                "connection_events": list(self.connection_events),
                "max_events": self.max_events,
            }
        )
        return d

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PluginManagerNode:
        """Deserialize from dict."""
        # Parse tick_frequency if present
        tick_freq = None
        if d.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(d["tick_frequency"])

        node = cls(
            node_id=d.get("node_id", "plugin_manager"),
            parent_ids=set(d.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                d.get("child_order") or d.get("children_ids") or []
            ),
            default_expansion=Expansion(d.get("expansion", "content")),
            mode=d.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=d.get("version", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            display_sequence=d.get("display_sequence"),
            originator=d.get("originator"),
            title=d.get("title", ""),
            plugin_states=d.get("plugin_states", {}),
            plugin_types=d.get("plugin_types", {}),
            builtin_count=d.get("builtin_count", 0),
            loaded_count=d.get("loaded_count", 0),
            connection_events=d.get("connection_events", []),
            max_events=d.get("max_events", 10),
        )
        return node


@dataclass(kw_only=True)
class AgentNode(ContextNode):
    """Represents an agent in the context (self or another agent).

    Used for agent awareness - showing the agent its own identity and
    information about other agents (parent, children, peers).

    Attributes:
        agent_id: The agent's ID
        agent_type: Type of agent (explorer, summarizer, etc.)
        relation: Relationship to viewing agent ("self", "parent", "child", "peer")
        task: The agent's task description
        agent_state: Current state (spawned, running, waiting, etc.)
        session_id: Underlying session ID
        message_count: Number of pending messages for this agent
    """

    agent_id: str = ""
    agent_type: str = "default"
    relation: AgentRelation = AgentRelation.SELF
    task: str = ""
    agent_state: AgentState = AgentState.RUNNING
    session_id: str = ""
    message_count: int = 0

    @property
    def node_type(self) -> str:
        return "agent"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "relation": self.relation.value,
            "task": self.task,
            "agent_state": self.agent_state.value,
            "message_count": self.message_count,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render type, state, task, and messages."""
        parts: list[str] = []
        parts.append(f"  Type: {self.agent_type}\n")
        parts.append(f"  State: {self.agent_state.value}\n")

        if self.task:
            parts.append(f"  Task: {self.task}\n")

        if self.message_count > 0:
            parts.append(f"  Messages: {self.message_count} pending\n")

        return "".join(parts)

    def update_state(self, agent_state: AgentState) -> AgentNode:
        """Update the agent's state."""
        old_state = self.agent_state
        self.agent_state = agent_state
        if old_state != agent_state:
            self.mark_changed(f"Agent state: {old_state.value} → {agent_state.value}")
        return self

    def update_message_count(self, count: int) -> AgentNode:
        """Update pending message count."""
        old_count = self.message_count
        self.message_count = count
        if old_count != count:
            self.mark_changed(f"Messages: {old_count} → {count}")
        return self

    def render_digest(self) -> str:
        """Return 'Agent: id [state]' format."""
        return f"Agent: {self.agent_id} [{self.agent_state.value.upper()}]"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: agent info
        state = self.agent_state.value
        collapsed_text = f"[Agent: {self.agent_id} [{state}] {self.message_count}m]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: task and session info
        detail_tokens = count_tokens(self.task) if self.task else 0

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize AgentNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "relation": self.relation.value,
                "task": self.task,
                "agent_state": self.agent_state.value,
                "session_id": self.session_id,
                "message_count": self.message_count,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AgentNode:
        """Deserialize AgentNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", "default"),
            relation=AgentRelation(data.get("relation", "self")),
            task=data.get("task", ""),
            agent_state=AgentState(data.get("agent_state", "running")),
            session_id=data.get("session_id", ""),
            message_count=data.get("message_count", 0),
        )
        return node


@dataclass(kw_only=True)
class TraceNode(ContextNode):
    """A change trace as a first-class DAG node.

    TraceNodes are created as children of the node being traced, recording
    state changes for history and debugging. The `originator` field (inherited
    from ContextNode) identifies the cause: agent, file watcher, async process, etc.

    Attributes:
        node: The node_id of the traced node (also accessible via parent link)
        node_display_id: Display ID of traced node (e.g., "text_1") for rendering
        old_version: Version before the change
        new_version: Version after the change
        description: Human-readable change description
        content: Optional diff or detailed change content
    """

    node: str = ""  # node_id of the traced node
    node_display_id: str = ""  # Display ID of traced node for rendering
    old_version: int = 0
    new_version: int = 0
    description: str = ""
    content: str | None = None

    # Field-level change tracking
    field_name: str = ""  # Field that changed (e.g., "state", "content")
    prev_value: str = ""  # Previous value (stringified)
    curr_value: str = ""  # Current value (stringified)

    # Trace merging support
    merged_values: list[str] = field(
        default_factory=list
    )  # For same field: list of additional curr values
    trace_target: ContextNode | None = field(
        default=None, repr=False
    )  # Target node for merge lookup
    child_traces: list[TraceNode] = field(default_factory=list)  # Child traces when merged

    @property
    def node_type(self) -> str:
        return "trace"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "node": self.node,
            "versions": f"v{self.old_version}->v{self.new_version}",
            "description": self.description,
            "originator": self.originator,
            "expansion": self.default_expansion.value,
        }

    # Formatting constants
    MAX_HEADER_LENGTH = 80
    MAX_VALUES_SHOWN = 5

    def render_digest(self) -> str:
        """Return display name with smart formatting.

        Formats:
        - Single field: "state: collapsed -> details"
        - Multiple fields: "state: a -> b, tokens: 100 -> 200"
        - Merged values: "state: collapsed -> { details, all, hidden }"
        - Many fields: "Fields Changed: state, tokens, content..."
        - Many values: "state: a -> { b, c, d, (35 others) }"

        """
        if not self.child_traces:
            # Single change (possibly with merged values for same field)
            if self.merged_values:
                return f"{self.field_name}: {self.prev_value} -> {self._format_merged_values()}"
            return f"{self.field_name}: {self.prev_value} -> {self.curr_value}"

        # Merged from multiple fields
        all_changes = [(self.field_name, self.prev_value, self.curr_value)]
        all_changes += [(t.field_name, t.prev_value, t.curr_value) for t in self.child_traces]

        # Try compact: "field: a -> b, other: c -> d"
        compact = ", ".join(f"{f}: {p} -> {c}" for f, p, c in all_changes)
        if len(compact) <= self.MAX_HEADER_LENGTH:
            return compact

        # Field list: "Fields Changed: state, tokens, content..."
        fields = list(dict.fromkeys(f for f, _, _ in all_changes))  # unique, preserve order
        if len(fields) <= 5:
            return f"Fields Changed: {', '.join(fields)}"
        return f"Fields Changed: {len(fields)}"

    def _format_merged_values(self) -> str:
        """Format merged values as { val1, val2, ... } or { val1, val2, (N others) }."""
        values = [self.curr_value] + self.merged_values
        if len(values) <= self.MAX_VALUES_SHOWN:
            return f"{{ {', '.join(values)} }}"
        shown = values[: self.MAX_VALUES_SHOWN]
        remaining = len(values) - self.MAX_VALUES_SHOWN
        return f"{{ {', '.join(shown)}, ({remaining} others) }}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: just the description (shown in header)
        collapsed_tokens = count_tokens(self.description) + 5

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=0,
        )

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render originator and content."""
        parts: list[str] = []

        if self.originator:
            parts.append(f"  Originator: {self.originator}\n")

        if self.content:
            parts.append("  ---\n")
            for line in self.content.split("\n"):
                parts.append(f"  {line}\n")

        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        d = {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "parent_ids": list(self.parent_ids),
            "child_order": self.child_order.to_list(),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "display_sequence": self.display_sequence,
            "originator": self.originator,
            "node": self.node,
            "node_display_id": self.node_display_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "description": self.description,
            "content": self.content,
            # Field-level tracking
            "field_name": self.field_name,
            "prev_value": self.prev_value,
            "curr_value": self.curr_value,
            "merged_values": self.merged_values,
            # trace_target is not serialized (runtime reference)
            "child_traces": [t.to_dict() for t in self.child_traces],
        }
        if self.tick_frequency:
            d["tick_frequency"] = self.tick_frequency.to_dict()
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TraceNode:
        """Deserialize from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data.get("node_id", str(uuid.uuid4())[-8:]),
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "header")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            node=data.get("node", ""),
            node_display_id=data.get("node_display_id", ""),
            old_version=data.get("old_version", 0),
            new_version=data.get("new_version", 0),
            description=data.get("description", ""),
            content=data.get("content"),
            # Field-level tracking
            field_name=data.get("field_name", ""),
            prev_value=data.get("prev_value", ""),
            curr_value=data.get("curr_value", ""),
            merged_values=data.get("merged_values", []),
            # trace_target not deserialized (runtime reference)
            child_traces=[cls._from_dict(t) for t in data.get("child_traces", [])],
        )


@dataclass(kw_only=True)
class TaskNode(ContextNode):
    """Represents a task in the context graph.

    TaskNodes track the status and metadata of concurrent tasks
    within a session. They provide visibility into running agents,
    blocking conversations, and streaming tasks.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (e.g., "agent", "mcp_menu", "shell_session")
        io_mode: I/O mode ("sync", "async", "streaming")
        status: Current task status (pending, running, paused, done, failed)
        created_at: When the task was created
        started_at: When the task started running (if applicable)
        completed_at: When the task completed (if applicable)
        metadata: Additional task-specific metadata
    """

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    task_type: str = "script"
    io_mode: IOMode = IOMode.ASYNC
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=lambda: time.time())
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> str:
        return "task"

    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this task."""
        duration = None
        if self.started_at:
            if self.completed_at:
                duration = self.completed_at - self.started_at
            else:
                duration = time.time() - self.started_at

        return {
            "id": self.node_id,
            "type": self.node_type,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "io_mode": self.io_mode.value,
            "status": self.status.value,
            "duration": duration,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full task info with timing and metadata."""
        lines = [f"Task: {self.task_id}"]
        lines.append(f"  Type: {self.task_type}")
        lines.append(f"  Status: {self.status.value}")
        lines.append(f"  I/O Mode: {self.io_mode.value}")

        # Timing info
        if self.created_at:
            from datetime import datetime

            created = datetime.fromtimestamp(self.created_at)
            lines.append(f"  Created: {created.isoformat()}")

        if self.started_at:
            from datetime import datetime

            started = datetime.fromtimestamp(self.started_at)
            lines.append(f"  Started: {started.isoformat()}")

        if self.completed_at:
            from datetime import datetime

            completed = datetime.fromtimestamp(self.completed_at)
            lines.append(f"  Completed: {completed.isoformat()}")

            if self.started_at:
                duration = self.completed_at - self.started_at
                lines.append(f"  Duration: {duration:.2f}s")

        if self.metadata:
            lines.append("  Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def update_status(self, status: TaskStatus) -> None:
        """Update task status with timing."""
        old_status = self.status
        self.status = status

        if status == TaskStatus.RUNNING and not self.started_at:
            self.started_at = time.time()
        elif status in (TaskStatus.DONE, TaskStatus.FAILED) and not self.completed_at:
            self.completed_at = time.time()

        self.mark_changed(f"status: {old_status.value} -> {status.value}")

    def render_digest(self) -> str:
        """Display name for the task."""
        return f"Task[{self.task_type}:{self.task_id[:8]}]"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: task status line
        collapsed_text = f"[Task: {self.task_type} | {self.status.value}]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Content includes basic info
        summary_text = self.render_content()
        summary_tokens = count_tokens(summary_text)

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "io_mode": self.io_mode.value,
                "status": self.status.value,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "metadata": self.metadata,
            }
        )
        return base

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TaskNode:
        """Deserialize from dictionary."""
        tick_freq_data = data.get("tick_frequency")
        tick_frequency = TickFrequency.from_dict(tick_freq_data) if tick_freq_data else None

        return cls(
            node_id=data.get("node_id", ""),
            default_expansion=Expansion(data.get("expansion", Expansion.CONTENT.value)),
            mode=data.get("mode", "running"),
            tick_frequency=tick_frequency,
            task_id=data.get("task_id", ""),
            task_type=data.get("task_type", "script"),
            io_mode=IOMode(data.get("io_mode", "async")),
            status=TaskStatus(data.get("status", "pending")),
            created_at=data.get("created_at", 0.0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Help system
# ---------------------------------------------------------------------------


def _extract_help_content(cls: type) -> str:
    """Extract documentation from a ContextNode subclass.

    Parses class docstrings, method signatures, @exposed members, and
    properties to produce structured help content.

    For plugin node types, documentation is extracted in priority order:
    1. Explicit plugin_info dict in the module
    2. Class docstring
    3. Method docstrings for Render*, GetDigest, etc.

    Args:
        cls: The ContextNode subclass to document.

    Returns:
        Formatted help text string.
    """
    import inspect
    import sys

    from activecontext.context.exposed import get_exposed

    lines: list[str] = []

    # Try to get plugin_info dict from the class's module (for plugin nodes)
    plugin_info: dict[str, str] | None = None
    try:
        module = sys.modules.get(cls.__module__)
        if module is not None:
            plugin_info_dict = getattr(module, "plugin_info", None)
            if isinstance(plugin_info_dict, dict):
                plugin_info = plugin_info_dict
    except Exception:
        pass

    # Type name and description from plugin_info or docstring
    if plugin_info:
        # Use plugin_info as priority source
        plugin_name = plugin_info.get("name", cls.__name__)
        description = plugin_info.get("description")

        # If description is missing from plugin_info, fall back to docstring
        if not description:
            cls_doc = inspect.getdoc(cls) or ""
            description = cls_doc.split("\n")[0] if cls_doc else "(no description)"

        lines.append(f"# {plugin_name}")
        lines.append(f"{description}")
        lines.append("")

        # Add plugin metadata if available
        version = plugin_info.get("version")
        author = plugin_info.get("author")
        if version or author:
            lines.append("## Plugin Information")
            if version:
                lines.append(f"- **Version**: {version}")
            if author:
                lines.append(f"- **Author**: {author}")
            lines.append("")
    else:
        # Fallback to class docstring
        cls_doc = inspect.getdoc(cls) or ""
        first_line = cls_doc.split("\n")[0] if cls_doc else "(no description)"
        lines.append(f"# {cls.__name__}")
        lines.append(f"{first_line}")
        lines.append("")

    # Constructor parameters (from dataclass fields)
    try:
        from dataclasses import MISSING
        from dataclasses import fields as dc_fields

        cls_fields = dc_fields(cls)
        if cls_fields:
            lines.append("## Constructor Parameters")
            for f in cls_fields:
                if f.name.startswith("_"):
                    continue
                # Skip inherited ContextNode base fields for brevity
                if f.name in {
                    "node_id",
                    "parent_ids",
                    "child_order",
                    "expansion",
                    "mode",
                    "tick_frequency",
                    "version",
                    "created_at",
                    "updated_at",
                    "originator",
                    "title",
                    "display_sequence",
                    "content_id",
                    "notification_level",
                    "is_subscription_point",
                    "tracing",
                    "trace_sink",
                }:
                    continue
                type_str = str(f.type) if f.type else "Any"
                default_str = ""
                if f.default is not MISSING:
                    default_str = f" = {f.default!r}"
                elif f.default_factory is not MISSING:
                    default_str = " = ..."
                lines.append(f"- **{f.name}**: `{type_str}`{default_str}")
            lines.append("")
    except TypeError:
        pass

    # Exposed members
    exposed_names = get_exposed(cls)

    # Public methods (including exposed)
    methods: list[tuple[str, str, str]] = []  # (name, signature, docstring)
    properties: list[tuple[str, str, bool]] = []  # (name, docstring, writable)

    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue

        # Check via class __dict__ to find descriptors properly
        member = None
        for klass in cls.__mro__:
            if name in klass.__dict__:
                member = klass.__dict__[name]
                break
        if member is None:
            member = getattr(cls, name, None)
        if member is None:
            continue

        # Properties
        if isinstance(member, property):
            doc = inspect.getdoc(member.fget) if member.fget else ""
            first_doc = (doc or "").split("\n")[0]
            writable = member.fset is not None
            properties.append((name, first_doc, writable))
            continue

        # Regular methods
        if callable(member) and not isinstance(member, type):
            try:
                sig = inspect.signature(member)
                sig_str = str(sig)
            except (ValueError, TypeError):
                sig_str = "(...)"
            doc = inspect.getdoc(member) or ""
            first_doc = doc.split("\n")[0]
            is_exp = name in exposed_names
            marker = " *" if is_exp else ""
            methods.append((name, sig_str, first_doc + marker))

    if methods:
        lines.append("## Methods")
        for name, sig_str_display, doc in methods:
            lines.append(f"- `{name}{sig_str_display}` -- {doc}")
        lines.append("")

    if properties:
        lines.append("## Properties")
        for name, doc, writable in properties:
            rw = "read/write" if writable else "read-only"
            lines.append(f"- `{name}` ({rw}) -- {doc}")
        lines.append("")

    if exposed_names:
        lines.append("## Agent-Facing API (@exposed)")
        for name in sorted(exposed_names):
            lines.append(f"- {name}")
        lines.append("")

    return "\n".join(lines)


@dataclass(kw_only=True)
class HelpNode(ContextNode):
    """Documentation node generated by .help().

    HelpNode renders API documentation for a specific node type. It is
    created as a child of the node being documented and provides three
    rendering levels: collapsed (brief header), summary (overview +
    method list), and detail (full documentation with signatures).

    Attributes:
        parent_node_type: The node_type string of the documented node type.
    """

    parent_node_type: str = ""
    _help_content: str = field(default="", repr=False)

    @property
    def node_type(self) -> str:
        return "help"

    def _count_methods(self) -> int:
        """Count documented methods from help content.

        Matches lines in the Methods section that look like function signatures:
        ``- `name(`` pattern (backtick-name-open-paren).
        """
        import re

        count = 0
        in_methods = False
        for line in self._help_content.split("\n"):
            if line.startswith("## Methods"):
                in_methods = True
                continue
            if line.startswith("## ") and in_methods:
                in_methods = False
                continue
            if in_methods and re.match(r"^- `\w+\(", line):
                count += 1
        return count

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "parent_node_type": self.parent_node_type,
            "methods": self._count_methods(),
            "expansion": self.default_expansion.value,
        }

    def render_digest(self) -> str:
        """Return display name with method count."""
        methods = self._count_methods()
        return f"{self.parent_node_type} Help -- {methods} methods"

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full documentation."""
        parts: list[str] = []
        for line in self._help_content.split("\n"):
            parts.append(f"  {line}\n")
        return "".join(parts)

    def _build_summary_text(self) -> str:
        """Build summary text from help content without calling Render methods.

        Avoids the recursion: get_token_breakdown -> render_content -> render_header
        -> get_token_breakdown.
        """
        parts: list[str] = []
        in_methods = False
        method_names: list[str] = []
        for line in self._help_content.split("\n"):
            if line.startswith("# "):
                parts.append(line[2:])
            elif line and not line.startswith("#") and not line.startswith("-"):
                if not in_methods:
                    parts.append(line)
            elif line.startswith("## Methods"):
                in_methods = True
            elif in_methods and line.startswith("- `"):
                name = line[3:].split("(")[0].split("`")[0]
                method_names.append(name)
            elif line.startswith("## ") and in_methods:
                in_methods = False
        if method_names:
            parts.append(f"Methods: {', '.join(method_names)}")
        return "\n".join(parts)

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        collapsed_text = f"[{self.parent_node_type} Help]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Use _build_summary_text to avoid recursion
        summary_tokens = count_tokens(self._build_summary_text())
        detail_tokens = count_tokens(self._help_content)

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize HelpNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "parent_node_type": self.parent_node_type,
                "_help_content": self._help_content,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> HelpNode:
        """Deserialize HelpNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "content")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            parent_node_type=data.get("parent_node_type", ""),
            _help_content=data.get("_help_content", ""),
        )


@dataclass(kw_only=True)
class MarkdownListItemNode(ContextNode):
    """A single markdown list item that can contain nested items.

    Attributes:
        content: The text content of this list item (without list marker)
        is_ordered: True for numbered lists (1. 2. 3.), False for bullets (- *)
        indent_level: Indentation level (0 = root, 1 = nested once, etc.)
        marker: The original list marker (e.g., "-", "*", "1.", "2.")
    """

    content: str = ""
    is_ordered: bool = False
    indent_level: int = 0
    marker: str = "-"

    @property
    def node_type(self) -> str:
        return "markdown_list_item"

    def GetDigest(self) -> dict[str, Any]:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return {
            "id": self.node_id,
            "type": self.node_type,
            "content_preview": preview,
            "is_ordered": self.is_ordered,
            "indent_level": self.indent_level,
            "expansion": self.default_expansion.value,
            "children_count": len(self.child_order),
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full list item with proper indentation."""
        indent = "  " * self.indent_level
        return f"{indent}{self.marker} {self.content}\n"

    def render_digest(self) -> str:
        """Return list item type indicator."""
        return f"{'OL' if self.is_ordered else 'UL'}-{self.indent_level}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: just metadata
        collapsed_text = f"[List item: {len(self.content)} chars]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: full content with marker
        indent = "  " * self.indent_level
        detail_text = f"{indent}{self.marker} {self.content}\n"
        detail_tokens = count_tokens(detail_text)

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MarkdownListItemNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "content": self.content,
                "is_ordered": self.is_ordered,
                "indent_level": self.indent_level,
                "marker": self.marker,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MarkdownListItemNode:
        """Deserialize MarkdownListItemNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            content=data.get("content", ""),
            is_ordered=data.get("is_ordered", False),
            indent_level=data.get("indent_level", 0),
            marker=data.get("marker", "-"),
        )


@dataclass(kw_only=True)
class MarkdownNode(ContextNode):
    """Structured markdown with automatic list parsing.

    MarkdownNode parses markdown content and creates child MarkdownListItemNode
    instances for each list item. Nested lists become nested child nodes.

    Attributes:
        content: The full markdown content
        buffer_id: Optional reference to a text buffer for live editing
        auto_parse: If True, automatically parse lists on content changes
    """

    content: str = ""
    buffer_id: str | None = None
    auto_parse: bool = True

    @property
    def node_type(self) -> str:
        return "markdown"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "content_length": len(self.content),
            "buffer_id": self.buffer_id,
            "expansion": self.default_expansion.value,
            "children_count": len(self.child_order),
        }

    def _parse_lists(self) -> list[tuple[str, bool, int, str]]:
        """Parse markdown content for list items.

        Returns:
            List of tuples: (content, is_ordered, indent_level, marker)
        """
        import re

        items: list[tuple[str, bool, int, str]] = []
        lines = self.content.split("\n")

        # Regex patterns for list items
        unordered_pattern = re.compile(r"^(\s*)([*+-])\s+(.*)$")
        ordered_pattern = re.compile(r"^(\s*)(\d+\.)\s+(.*)$")

        for line in lines:
            # Try unordered list
            match = unordered_pattern.match(line)
            if match:
                indent = len(match.group(1))
                marker = match.group(2)
                content = match.group(3)
                indent_level = indent // 2  # 2 spaces per indent level
                items.append((content, False, indent_level, marker))
                continue

            # Try ordered list
            match = ordered_pattern.match(line)
            if match:
                indent = len(match.group(1))
                marker = match.group(2)
                content = match.group(3)
                indent_level = indent // 2
                items.append((content, True, indent_level, marker))
                continue

        return items

    def parse_and_create_children(self) -> MarkdownNode:
        """Parse lists and create child nodes for each item.

        This method parses the markdown content, creates MarkdownListItemNode
        children, and establishes parent-child relationships based on indentation.

        Returns:
            Self for chaining
        """
        if not self._graph:
            # Can't create children without graph reference
            return self

        # Parse list items
        items = self._parse_lists()
        if not items:
            return self

        # Remove existing list item children (recursively)
        def remove_list_items(parent_id: str) -> None:
            """Recursively remove all MarkdownListItemNode children."""
            if self._graph is None:
                return
            parent = self._graph.get_node(parent_id)
            if not parent:
                return
            children_to_remove = [
                child_id
                for child_id in parent.child_order
                if isinstance(self._graph.get_node(child_id), MarkdownListItemNode)
            ]
            for child_id in children_to_remove:
                # Recursively remove nested items first
                remove_list_items(child_id)
                # Then unlink from parent
                self._graph.unlink(child_id, parent_id)

        remove_list_items(self.node_id)

        # Create new list item nodes
        stack: list[tuple[int, MarkdownListItemNode]] = []  # (indent_level, node)

        for content, is_ordered, indent_level, marker in items:
            item_node = MarkdownListItemNode(
                content=content,
                is_ordered=is_ordered,
                indent_level=indent_level,
                marker=marker,
                originator=self.node_id,
            )
            self._graph.add_node(item_node)

            # Find parent based on indent level
            parent_node: ContextNode = self

            # Pop stack until we find the right parent level
            while stack and stack[-1][0] >= indent_level:
                stack.pop()

            if stack:
                # Parent is the last item with lower indent
                parent_node = stack[-1][1]

            # Link to parent
            self._graph.link(item_node.node_id, parent_node.node_id)

            # Add to stack for potential children
            stack.append((indent_level, item_node))

        self.mark_changed(f"Parsed {len(items)} list items")
        return self

    def set_content(self, content: str) -> MarkdownNode:
        """Update markdown content and re-parse if auto_parse is enabled.

        Args:
            content: New markdown content

        Returns:
            Self for chaining
        """
        old_content = self.content
        self.content = content

        if self.auto_parse:
            self.parse_and_create_children()

        self.mark_changed(f"Content updated ({len(old_content)} → {len(content)} chars)")
        return self

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render full markdown content."""
        return self.content

    def render_digest(self) -> str:
        """Return markdown document indicator."""
        if self.buffer_id:
            return f"MD:{self.buffer_id}"
        return "MARKDOWN"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: metadata only
        collapsed_text = f"[Markdown: {len(self.content)} chars, {len(self.child_order)} items]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: full content
        detail_tokens = count_tokens(self.content)

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MarkdownNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "content": self.content,
                "buffer_id": self.buffer_id,
                "auto_parse": self.auto_parse,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MarkdownNode:
        """Deserialize MarkdownNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            content=data.get("content", ""),
            buffer_id=data.get("buffer_id"),
            auto_parse=data.get("auto_parse", True),
        )


@dataclass(kw_only=True)
class FileSystemNode(ContextNode):
    """Directory tree view with filtering and expand/collapse support.

    Attributes:
        root_path: Root directory path to display
        pattern: Glob pattern for filtering (e.g., "*.py", "**/*.md")
        max_depth: Maximum directory depth to display (None = unlimited)
        show_hidden: Whether to show hidden files/directories
        expanded_paths: Set of expanded directory paths
    """

    root_path: str = "."
    pattern: str | None = None
    max_depth: int | None = None
    show_hidden: bool = False
    expanded_paths: set[str] = field(default_factory=set)
    _cached_tree: str = field(default="", init=False, repr=False)
    _last_scan: float = field(default=0.0, init=False, repr=False)

    @property
    def node_type(self) -> str:
        return "filesystem"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "root_path": self.root_path,
            "pattern": self.pattern,
            "max_depth": self.max_depth,
            "expansion": self.default_expansion.value,
        }

    def _scan_directory(self) -> str:
        """Scan directory and build tree representation."""
        try:
            root = Path(self.root_path).resolve()
            if not root.exists():
                return f"[Directory not found: {self.root_path}]\n"

            lines: list[str] = []
            self._build_tree(root, "", lines, 0)
            return "\\n".join(lines)
        except Exception as e:
            return f"[Error scanning directory: {e}]\n"

    def _build_tree(
        self,
        path: Path,
        prefix: str,
        lines: list[str],
        depth: int,
    ) -> None:
        """Recursively build tree structure."""
        if self.max_depth is not None and depth > self.max_depth:
            return

        # Filter hidden files
        if not self.show_hidden and path.name.startswith("."):
            return

        # Apply pattern filter - skip non-matching files (but still recurse into dirs)
        if self.pattern and not path.match(self.pattern) and not path.is_dir():
            return

        # Add current item
        is_dir = path.is_dir()
        marker = "📁" if is_dir else "📄"
        is_expanded = str(path) in self.expanded_paths

        if is_dir:
            marker = "📂" if is_expanded else "📁"

        lines.append(f"{prefix}{marker} {path.name}")

        # Recurse into directories if expanded
        if is_dir and (is_expanded or depth == 0):
            try:
                children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
                for i, child in enumerate(children):
                    is_last = i == len(children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    self._build_tree(child, new_prefix, lines, depth + 1)
            except PermissionError:
                lines.append(f"{prefix}    [Permission denied]")

    def toggle_path(self, path: str) -> FileSystemNode:
        """Toggle expansion state of a directory path."""
        if path in self.expanded_paths:
            self.expanded_paths.remove(path)
        else:
            self.expanded_paths.add(path)
        self._last_scan = 0  # Force rescan
        self.mark_changed(f"Toggled {path}")
        return self

    def Recompute(self) -> None:
        """Recompute directory tree on tick."""
        current_time = time.time()
        # Rescan every 5 seconds or on expansion change
        if current_time - self._last_scan > 5.0:
            self._cached_tree = self._scan_directory()
            self._last_scan = current_time

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render root path and directory tree."""
        if not self._cached_tree:
            self._cached_tree = self._scan_directory()
        return f"Root: {self.root_path}\n{self._cached_tree}"

    def render_digest(self) -> str:
        """Return filesystem node indicator."""
        return f"FS:{Path(self.root_path).name}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        collapsed_text = f"[FileSystem: {self.root_path}]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        if not self._cached_tree:
            self._cached_tree = self._scan_directory()
        detail_tokens = count_tokens(self._cached_tree)

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize FileSystemNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "root_path": self.root_path,
                "pattern": self.pattern,
                "max_depth": self.max_depth,
                "show_hidden": self.show_hidden,
                "expanded_paths": list(self.expanded_paths),
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FileSystemNode:
        """Deserialize FileSystemNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            root_path=data.get("root_path", "."),
            pattern=data.get("pattern"),
            max_depth=data.get("max_depth"),
            show_hidden=data.get("show_hidden", False),
            expanded_paths=set(data.get("expanded_paths", [])),
        )


@dataclass(kw_only=True)
class ClockNode(ContextNode):
    """Timer/countdown node with tick-driven updates.

    Attributes:
        start_time: Unix timestamp when timer started
        duration_seconds: Duration for countdown (None = stopwatch mode)
        is_running: Whether the timer is currently running
        elapsed_seconds: Cached elapsed time
    """

    start_time: float = field(default_factory=time.time)
    duration_seconds: float | None = None
    is_running: bool = True
    elapsed_seconds: float = 0.0

    @property
    def node_type(self) -> str:
        return "clock"

    def GetDigest(self) -> dict[str, Any]:
        remaining = self.get_remaining()
        return {
            "id": self.node_id,
            "type": self.node_type,
            "mode": "countdown" if self.duration_seconds else "stopwatch",
            "elapsed": f"{self.elapsed_seconds:.1f}s",
            "remaining": f"{remaining:.1f}s" if remaining is not None else None,
            "is_running": self.is_running,
            "expansion": self.default_expansion.value,
        }

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if not self.is_running:
            return self.elapsed_seconds
        return time.time() - self.start_time + self.elapsed_seconds

    def get_remaining(self) -> float | None:
        """Get remaining time in countdown mode (None in stopwatch mode)."""
        if self.duration_seconds is None:
            return None
        remaining = self.duration_seconds - self.get_elapsed()
        return max(0.0, remaining)

    def is_complete(self) -> bool:
        """Check if countdown has completed."""
        if self.duration_seconds is None:
            return False
        return self.get_elapsed() >= self.duration_seconds

    def start(self) -> ClockNode:
        """Start or resume the timer."""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
            self.mark_changed("Timer started")
        return self

    def pause(self) -> ClockNode:
        """Pause the timer."""
        if self.is_running:
            self.elapsed_seconds = self.get_elapsed()
            self.is_running = False
            self.mark_changed("Timer paused")
        return self

    def reset(self) -> ClockNode:
        """Reset the timer to zero."""
        self.start_time = time.time()
        self.elapsed_seconds = 0.0
        self.is_running = False
        self.mark_changed("Timer reset")
        return self

    def Recompute(self) -> None:
        """Update elapsed time on tick."""
        if self.is_running and self.is_complete():
            self.pause()
            self.mark_changed("Countdown completed")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render time and status."""
        elapsed = self.get_elapsed()
        status = "⏸" if not self.is_running else "▶"

        if self.duration_seconds:
            remaining = self.get_remaining()
            elapsed_str = self._format_time(elapsed)
            duration_str = self._format_time(self.duration_seconds)
            remaining_str = self._format_time(remaining or 0)
            return f"{status} {elapsed_str} / {duration_str} (remaining: {remaining_str})\n"
        else:
            return f"{status} {self._format_time(elapsed)}\n"

    def render_digest(self) -> str:
        """Return clock type indicator."""
        if self.duration_seconds:
            return f"COUNTDOWN:{self._format_time(self.duration_seconds)}"
        return "STOPWATCH"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        collapsed_text = f"[Clock: {self.get_elapsed():.1f}s]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Build summary text without calling render_content to avoid recursion
        elapsed = self.get_elapsed()
        elapsed_str = self._format_time(elapsed)
        summary_tokens = count_tokens(elapsed_str)

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ClockNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "start_time": self.start_time,
                "duration_seconds": self.duration_seconds,
                "is_running": self.is_running,
                "elapsed_seconds": self.elapsed_seconds,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ClockNode:
        """Deserialize ClockNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            start_time=data.get("start_time", time.time()),
            duration_seconds=data.get("duration_seconds"),
            is_running=data.get("is_running", True),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )


@dataclass(kw_only=True)
class FunctionDocNode(ContextNode):
    """Extract and display function signatures and docstrings.

    Attributes:
        file_path: Path to Python file
        function_name: Name of function to document
        signature: Extracted function signature
        docstring: Extracted docstring
        source_lines: Optional full source code
    """

    file_path: str = ""
    function_name: str = ""
    signature: str = ""
    docstring: str = ""
    source_lines: str = ""

    @property
    def node_type(self) -> str:
        return "function_doc"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "has_docstring": bool(self.docstring),
            "expansion": self.default_expansion.value,
        }

    def extract_function_info(self) -> FunctionDocNode:
        """Extract function signature and docstring from file."""
        try:
            import ast

            with open(self.file_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=self.file_path)

            # Find the function
            for node in ast.walk(tree):
                is_func = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                if is_func and node.name == self.function_name:
                    # Extract signature
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    returns = ""
                    if node.returns:
                        returns = f" -> {ast.unparse(node.returns)}"

                    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                    arg_list = ", ".join(args)
                    self.signature = f"{async_prefix}def {node.name}({arg_list}){returns}"

                    # Extract docstring
                    self.docstring = ast.get_docstring(node) or ""

                    # Extract source
                    self.source_lines = ast.unparse(node)

                    self.mark_changed(f"Extracted {self.function_name}")
                    return self

            self.docstring = f"[Function {self.function_name} not found in {self.file_path}]"
        except Exception as e:
            self.docstring = f"[Error extracting function: {e}]"

        return self

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render content: signature + docstring."""
        if not self.signature:
            self.extract_function_info()

        parts = [f"{self.signature}\n"]
        if self.docstring:
            parts.append(f'"""\n{self.docstring}\n"""\n')
        return "".join(parts)

    def render_digest(self) -> str:
        """Return function doc indicator."""
        return f"DOC:{self.function_name}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        collapsed_text = f"[FunctionDoc: {self.function_name}]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        summary_text = f"{self.signature}\n"
        summary_tokens = count_tokens(summary_text)

        detail_text = f"{self.signature}\n{self.docstring}\n"
        detail_tokens = count_tokens(detail_text)

        return TokenInfo(
            title=collapsed_tokens,
            content=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize FunctionDocNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "file_path": self.file_path,
                "function_name": self.function_name,
                "signature": self.signature,
                "docstring": self.docstring,
                "source_lines": self.source_lines,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FunctionDocNode:
        """Deserialize FunctionDocNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            file_path=data.get("file_path", ""),
            function_name=data.get("function_name", ""),
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            source_lines=data.get("source_lines", ""),
        )
