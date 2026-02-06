"""Base context node classes.

This module defines the core node hierarchy:
- ContextNode: Base class with common fields and notification
- SimpleNode: Simple node with string content
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import (
    Expansion,
    NotificationLevel,
    TickFrequency,
)
from activecontext.context.traceable import trace_all_fields

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.context.headers import TokenInfo
    from activecontext.context.nodes.help import HelpNode
    from activecontext.context.nodes.trace import TraceNode

# Overhead tokens for the token counts display itself.
# The "(tokens: NNN / NN+NN+NN of NNN)" string occupies tokens in the header.
# This constant is added to header_tokens to account for that self-referential cost.
TOKEN_COUNTS_OVERHEAD = 12

# Type alias for hooks
# (parent_node, child_node, description)
OnChildChangedHook = Callable[["ContextNode", "ContextNode", str], None]


@trace_all_fields
@dataclass(kw_only=True)
class ContextNode:
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

    # Graph reference: None during __init__, always set by ContextGraph.add_node() before use
    _graph: ContextGraph | None = field(default=None, init=False, repr=False)

    # Optional hook for child change notifications
    _on_child_changed_hook: OnChildChangedHook | None = field(default=None, init=False, repr=False)

    # Trace merging state (for time-window based merging)
    _last_trace: TraceNode | None = field(default=None, init=False, repr=False)
    _last_trace_time: float = field(default=0.0, init=False, repr=False)

    # Cached children tokens (computed during projection collection)
    _cached_children_tokens: int = field(default=0, init=False, repr=False)

    @property
    def node_type(self) -> str:
        """Return the node type identifier (class name).

        Returns the Python class name (e.g., "TextNode", "GroupNode").
        For display-friendly short names (e.g., "text"), use display_type.
        """
        return type(self).__name__

    @property
    def display_type(self) -> str:
        """Return display-friendly type name for IDs and headers.

        Derives from class name: TextNode -> text, GroupNode -> group.
        """
        name = type(self).__name__
        # Remove "Node" suffix and lowercase
        if name.endswith("Node"):
            return name[:-4].lower()
        return name.lower()

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
    def detail_tokens(self) -> int:
        """Sum of children's tokens beyond their headers.

        Represents additional tokens when expanding from INDEX to ALL.
        For each child: content + index + detail (everything except title).
        Returns 0 for leaf nodes or when no graph reference.
        """
        if not self._graph:
            return 0
        total = 0
        for child_id in self.child_order:
            child = self._graph.get_node(child_id)
            if child:
                info = child.get_token_breakdown()
                total += info.content + info.index + info.detail
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

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for different visibility levels.

        Uses render_digest() for title tokens and render_content() for content tokens.
        Subclasses should override render_digest() and render_content() instead of this.

        Returns:
            TokenInfo with title, content, index, detail, and total token counts.
        """
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        title_tokens = count_tokens(self.render_digest())
        content_tokens = count_tokens(self.render_content() or "")

        return TokenInfo(
            title=title_tokens,
            content=content_tokens,
            index=self.index_tokens,
            detail=self.detail_tokens,
        )

    def render_digest(self) -> str:
        """Render node metadata — the framework prepends the title line.

        Default returns "[node_type] title". Subclasses can override to add
        type-specific info via super().render_digest() + extra.

        Examples:
            TextNode: "[TextNode] main.py" + " (lines 1-50)"
            ShellNode: "[ShellNode] pytest" + " [COMPLETED]"
        """
        return ""

    def Recompute(self) -> None:
        """Recompute this node's content. Called during tick for running nodes.

        Default implementation does nothing. Subclasses override to:
        1. Perform actual recomputation (reload file, update stats, etc.)
        2. Call _mark_changed() with a meaningful description if content changed

        Note: Don't generate traces here - only trace meaningful state changes.
        """
        pass

    def render_content(self) -> str:
        """Render the content section — the actual content of this node.

        Subclasses override this to provide node-specific content.
        Base returns empty string (header-only nodes).

        Note: Nodes that need TextBuffer access (like TextNode) should
        use TextBuffer.get_by_id() to look up their buffer.
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

    def GetDigest(self) -> dict[str, Any]:
        """Return compact metadata for the handles dict.

        Override in subclasses to provide type-specific digest fields.
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
        }

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

        # Notify graph of change (for view-based notification routing)
        if self._graph:
            self._graph.notify_change(self, trace_node)

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
        # Import here to avoid circular imports
        from activecontext.context.nodes.trace import TraceNode

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

        result: bool = self._graph.link(child.node_id, self.node_id, after=after)
        return result

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

        # Import here to avoid circular imports
        from activecontext.context.nodes.help import HelpNode, _extract_help_content

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
class SimpleNode(ContextNode):
    """Simple node with string content."""

    content: str = ""

    @property
    def node_type(self) -> str:
        return "SimpleNode"

    def render_content(self) -> str:
        return self.content

    def append(self, content: str, sep: str = "\n\n") -> None:
        if self.content:
            self.content += sep
        self.content += content

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["content"] = self.content
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SimpleNode:
        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "content")),
            mode=data.get("mode", "paused"),
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            content=data.get("content", ""),
        )
