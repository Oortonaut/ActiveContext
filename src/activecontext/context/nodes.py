"""Context node types for the context DAG.

This module defines the typed node hierarchy:
- ContextNode: Base class with common fields and notification
- TextNode: File content view (text)
- GroupNode: Summary facade over children
- TopicNode: Conversation segment
- ArtifactNode: Code/output artifact
- ShellNode: Async shell command execution
- MessageNode: Conversation message with ID for referencing
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from activecontext.context.state import NodeState, NotificationLevel, TickFrequency

if TYPE_CHECKING:
    from activecontext.context.graph import LinkedChildOrder
from activecontext.core.tokens import MediaType, detect_media_type, tokens_to_chars


class ShellStatus(Enum):
    """Status of a shell command execution."""

    PENDING = "pending"      # Created, not yet started
    RUNNING = "running"      # Subprocess is executing
    COMPLETED = "completed"  # Finished successfully (exit_code == 0)
    FAILED = "failed"        # Finished with error (exit_code != 0)
    TIMEOUT = "timeout"      # Killed due to timeout
    CANCELLED = "cancelled"  # Cancelled by user


class LockStatus(Enum):
    """Status of a file lock."""

    PENDING = "pending"      # Waiting to acquire lock
    ACQUIRED = "acquired"    # Lock held
    TIMEOUT = "timeout"      # Failed to acquire within timeout
    RELEASED = "released"    # Lock released
    ERROR = "error"          # Error during lock operation


if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.context.headers import TokenInfo


# Type alias for hooks
# (parent_node, child_node, description)
OnChildChangedHook = Callable[["ContextNode", "ContextNode", str], None]


@dataclass
class ContextNode(ABC):
    """Base class for all context DAG nodes.

    Attributes:
        node_id: Unique identifier (8-char UUID suffix)
        parent_ids: Set of parent node IDs (DAG allows multiple parents)
        children_ids: Set of child node IDs
        tokens: Token budget for rendering
        state: Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL)
        mode: "paused" or "running" for tick processing
        tick_frequency: Tick frequency specification (turn, async, never, period)
        version: Incremented on change for trace detection
        created_at: Unix timestamp of creation
        updated_at: Unix timestamp of last update
        tags: Arbitrary metadata
        tracing: When True, state changes create TraceNode children
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: set[str] = field(default_factory=set)
    children_ids: set[str] = field(default_factory=set)

    # Ordered children for projection rendering (lazily initialized by graph.link())
    # None = no children linked yet, LinkedChildOrder = has/had children
    child_order: LinkedChildOrder | None = field(default=None, repr=False)

    # Rendering configuration
    tokens: int = 1000
    state: NodeState = NodeState.DETAILS
    mode: str = "paused"
    tick_frequency: TickFrequency | None = None

    # Version tracking
    version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Metadata
    tags: dict[str, Any] = field(default_factory=dict)

    # Originator: identifies the source of this node (node ID, filename, or arbitrary string)
    originator: str | None = None

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

    # Tracing configuration
    # When True, state changes create TraceNode children for history
    tracing: bool = True

    # Trace sink for nodes without parents
    # When set, traces link to this node instead of being orphaned
    trace_sink: ContextNode | None = field(default=None, repr=False)

    # Graph reference (set by ContextGraph.add_node)
    _graph: ContextGraph | None = field(default=None, repr=False)

    # Optional hook for child change notifications
    _on_child_changed_hook: OnChildChangedHook | None = field(default=None, repr=False)

    @property
    @abstractmethod
    def node_type(self) -> str:
        """Return the node type identifier."""
        ...

    @property
    def display_id(self) -> str:
        """Return short display ID for uniform headers.

        Format: "{node_type}_{sequence}" e.g., "text_1", "message_13"
        Used by LLM to uniquely reference this node. Valid Python identifier.
        """
        seq = self.display_sequence or 0
        return f"{self.node_type}_{seq}"

    @abstractmethod
    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        ...

    def Render(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render this node's content based on current state.

        Dispatches to RenderCollapsed, RenderSummary, or RenderDetail based
        on the node's state. Subclasses should override those methods instead.

        Args:
            cwd: Working directory for file access
            text_buffers: Optional dict of buffer_id -> TextBuffer for markdown nodes
        """
        if self.state == NodeState.HIDDEN:
            return ""
        elif self.state == NodeState.COLLAPSED:
            return self.RenderCollapsed(cwd=cwd, text_buffers=text_buffers)
        elif self.state == NodeState.SUMMARY:
            return self.RenderSummary(cwd=cwd, text_buffers=text_buffers)
        elif self.state == NodeState.DETAILS:
            return self.RenderDetail(include_summary=False, cwd=cwd, text_buffers=text_buffers)
        else:  # ALL
            return self.RenderDetail(include_summary=True, cwd=cwd, text_buffers=text_buffers)

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render minimal collapsed view.

        Default: just the header. Subclasses override for node-specific content.
        """
        return self.render_header(cwd=cwd)

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary view with key information.

        Default: same as collapsed. Subclasses override to add summary content.
        """
        return self.RenderCollapsed(cwd=cwd, text_buffers=text_buffers)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detailed view.

        Args:
            include_summary: If True, include summary content (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Optional dict of buffer_id -> TextBuffer.

        Default: same as summary. Subclasses override for detailed content.
        Child nodes are rendered by the projection engine, not here.
        """
        return self.RenderSummary(cwd=cwd, text_buffers=text_buffers)

    @abstractmethod
    def get_display_name(self) -> str:
        """Return human-readable name for uniform header.

        Examples:
            TextNode: "main.py:1-50"
            MessageNode: "User #13"
            ShellNode: "Shell: pytest [COMPLETED]"
        """
        ...

    @abstractmethod
    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for different visibility levels.

        Args:
            cwd: Working directory for relative path resolution.

        Returns:
            TokenInfo with collapsed, summary, and detail token counts.
        """
        ...

    def render_header(self, cwd: str = ".") -> str:
        """Render uniform header for this node based on current state.

        Args:
            cwd: Working directory for relative path resolution.

        Format varies by state:
            COLLAPSED: #### [type.N] name collapsed (tokens: visible/hidden)
            SUMMARY:   ### [type.N] name summary wake (tokens: collapsed+summary/hidden)
            ALL:       ### [type.N] name all (tokens: collapsed+summary+detail)
        """
        from .headers import render_header

        token_info = self.get_token_breakdown(cwd)
        return render_header(
            self.display_id,
            self.get_display_name(),
            self.state,
            token_info,
            notification_level=self.notification_level.value,
        )

    def Recompute(self) -> None:
        """Recompute this node's content. Called during tick for running nodes.

        Default implementation does nothing. Subclasses override to:
        1. Perform actual recomputation (reload file, update stats, etc.)
        2. Call _mark_changed() with a meaningful description if content changed

        Note: Don't generate traces here - only trace meaningful state changes.
        """
        pass

    def _mark_changed(
        self,
        description: str = "",
        content: str | None = None,
        originator: str | None = None,
    ) -> TraceNode | None:
        """Mark this node as changed, optionally creating a trace.

        Args:
            description: Human-readable description of the change
            content: Optional diff or detailed change content
            originator: Who/what caused the change (defaults to self.originator)

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
            )

        # Generate notification if level is not IGNORE
        # Pass trace_node so notification can use its node_id as trace_id
        if self.notification_level != NotificationLevel.IGNORE and description:
            self._emit_notification(description, originator, trace_node=trace_node)

        self.notify_parents(description)
        return trace_node

    def _create_trace(
        self,
        old_version: int,
        new_version: int,
        description: str,
        content: str | None = None,
        originator: str | None = None,
    ) -> TraceNode:
        """Create a TraceNode as a sibling of this node.

        Traces are linked to the same parent as this node, inserted immediately
        after this node in child_order. If this node has no parent but has a
        trace_sink set, the trace is linked to the trace_sink instead.

        Args:
            old_version: Version before the change
            new_version: Version after the change
            description: Human-readable change description
            content: Optional diff or detailed content
            originator: Who/what caused the change (defaults to self.originator)

        Returns:
            The created TraceNode
        """
        if self._graph is None:
            raise RuntimeError("Node not attached to graph")

        trace_node = TraceNode(
            node=self.node_id,
            node_display_id=self.display_id,
            old_version=old_version,
            new_version=new_version,
            description=description,
            content=content,
            originator=originator or self.originator,
            state=NodeState.COLLAPSED,
            tokens=50,  # Traces are compact by default
        )
        self._graph.add_node(trace_node)

        # Link as sibling (same parent) instead of child
        if self.parent_ids:
            # Has parent - link trace to same parent, insert after self
            parent_id = next(iter(self.parent_ids))  # Primary parent
            self._graph.link(trace_node.node_id, parent_id, after=self.node_id)
        elif self.trace_sink is not None:
            # No parent but has trace_sink - link to sink
            self._graph.link(trace_node.node_id, self.trace_sink.node_id)
        # else: orphan trace (no parent, no sink) - just exists in graph

        return trace_node

    def _emit_notification(
        self,
        description: str,
        originator: str | None = None,
        trace_node: TraceNode | None = None,
    ) -> None:
        """Emit notification - collected by graph's notification system.

        Args:
            description: Human-readable description of the change.
            originator: Who/what caused the change.
            trace_node: Optional TraceNode to use for trace_id (uses its node_id).
        """
        if self._graph and hasattr(self._graph, "emit_notification"):
            header = self._format_notification_header(description)
            # Use trace node_id if available, else fall back to synthetic ID
            trace_id = trace_node.node_id if trace_node else f"{self.node_id}:{self.version}"
            self._graph.emit_notification(
                node_id=self.node_id,
                trace_id=trace_id,
                header=header,
                level=self.notification_level,
            )

    def _format_notification_header(self, description: str) -> str:
        """Format brief notification header. Override in subclasses.

        Args:
            description: Human-readable description of the change.
        """
        return f"{self.display_id}: {description}"

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
        # Convert LinkedChildOrder to list for serialization
        child_order_list: list[str] | None = None
        if self.child_order is not None:
            if hasattr(self.child_order, "to_list"):
                child_order_list = self.child_order.to_list()
            else:
                child_order_list = list(self.child_order)

        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "parent_ids": list(self.parent_ids),
            "children_ids": list(self.children_ids),
            "child_order": child_order_list,
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "tick_frequency": self.tick_frequency.to_dict() if self.tick_frequency else None,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "originator": self.originator,
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

        This is a factory method that dispatches to the appropriate subclass.
        """
        node_type = data.get("node_type")
        if node_type == "text":
            return TextNode._from_dict(data)
        elif node_type == "group":
            return GroupNode._from_dict(data)
        elif node_type == "topic":
            return TopicNode._from_dict(data)
        elif node_type == "artifact":
            return ArtifactNode._from_dict(data)
        elif node_type == "shell":
            return ShellNode._from_dict(data)
        elif node_type == "lock":
            return LockNode._from_dict(data)
        elif node_type == "session":
            return SessionNode._from_dict(data)
        elif node_type == "message":
            return MessageNode._from_dict(data)
        elif node_type == "work":
            return WorkNode._from_dict(data)
        elif node_type == "mcp_server":
            return MCPServerNode._from_dict(data)
        elif node_type == "mcp_manager":
            return MCPManagerNode._from_dict(data)
        elif node_type == "mcp_tool":
            return MCPToolNode._from_dict(data)
        # Note: MarkdownNode removed - markdown content now uses TextNode
        elif node_type == "agent":
            return AgentNode._from_dict(data)
        elif node_type == "trace":
            return TraceNode._from_dict(data)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    # Fluent API for mode control
    def Run(self, freq: TickFrequency | None = None) -> ContextNode:
        """Enable tick recomputation with given frequency.

        Args:
            freq: Tick frequency (defaults to turn() if not specified)
        """
        old_mode = self.mode
        self.mode = "running"
        self.tick_frequency = freq or TickFrequency.turn()
        if self._graph:
            self._graph._running_nodes.add(self.node_id)
        if old_mode != "running":
            self._mark_changed(description=f"Mode: {old_mode} → running")
        return self

    def Pause(self) -> ContextNode:
        """Disable tick recomputation."""
        old_mode = self.mode
        self.mode = "paused"
        if self._graph:
            self._graph._running_nodes.discard(self.node_id)
        if old_mode != "paused":
            self._mark_changed(description=f"Mode: {old_mode} → paused")
        return self

    def SetTokens(self, n: int) -> ContextNode:
        """Set token budget.

        Args:
            n: New token budget for rendering.
        """
        old_tokens = self.tokens
        self.tokens = n
        if old_tokens != n:
            self._mark_changed(description=f"Tokens: {old_tokens} → {n}")
        return self

    def SetState(self, s: NodeState) -> ContextNode:
        """Set rendering state.

        Args:
            s: New node state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL).

        Generates a trace if tracing is enabled and state actually changed.
        """
        old_state = self.state
        if old_state != s:
            self.state = s
            self._mark_changed(
                description=f"State: {old_state.value} → {s.value}",
            )
        return self

    def SetNotify(self, level: NotificationLevel) -> ContextNode:
        """Set notification level (fluent API).

        Args:
            level: NotificationLevel.IGNORE, HOLD, or WAKE
        """
        old_level = self.notification_level
        self.notification_level = level
        if old_level != level:
            self._mark_changed(description=f"Notify: {old_level.value} → {level.value}")
        return self


@dataclass
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
    """

    path: str = ""
    pos: str = "1:0"
    end_pos: str | None = None
    media_type: MediaType = field(default=MediaType.TEXT)

    # TextBuffer reference (optional, for shared line storage)
    buffer_id: str | None = None
    start_line: int = 1
    end_line: int | None = None

    def __post_init__(self) -> None:
        """Auto-detect media type from file extension and set originator."""
        if self.path and self.media_type == MediaType.TEXT:
            self.media_type = detect_media_type(self.path)
        # Auto-populate originator from path if not explicitly set
        if self.originator is None and self.path:
            self.originator = self.path

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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
            "media_type": self.media_type.value,
        }

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render file content with line numbers.

        Args:
            include_summary: If True, include full content (for ALL state).
            cwd: Working directory for resolving paths
            text_buffers: Optional dict of buffer_id -> TextBuffer for markdown nodes

        Returns:
            Rendered content string
        """
        import os

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
                try:
                    end_line = int(self.end_pos.split(":")[0])
                except (ValueError, IndexError):
                    pass

            # Read file
            file_path = os.path.join(cwd, self.path)
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    file_lines = f.readlines()
            except FileNotFoundError:
                return f"[File not found: {self.path}]"
            except OSError as e:
                return f"[Error reading {self.path}: {e}]"

            # Apply line range
            start_idx = max(0, start_line - 1)
            end_idx = end_line if end_line else len(file_lines)
            lines = [l.rstrip("\n\r") for l in file_lines[start_idx:end_idx]]

        # Check if this is a markdown heading section
        is_markdown_heading = (
            self.media_type == MediaType.MARKDOWN
            and "heading" in self.tags
            and "level" in self.tags
        )

        # Build output
        output_parts: list[str] = []

        if is_markdown_heading:
            # Render markdown with heading annotation
            heading = self.tags["heading"]
            level = self.tags["level"]
            prefix = "#" * level

            # Format: ## Heading {#text_1}
            output_parts.append(f"{prefix} {heading} {{#{self.display_id}}}\n")

            # Render remaining lines (skip the heading line itself)
            for i, line in enumerate(lines):
                # Skip the first line if it's the heading
                if i == 0 and line.strip().startswith("#"):
                    continue
                output_parts.append(f"{line}\n")
        else:
            # Regular text rendering with line numbers
            output_parts.append(self.render_header(cwd=cwd))

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
                output_parts.append(f"{line_num:4d} | {line_content}\n")

        return "".join(output_parts)

    def SetPos(self, pos: str) -> TextNode:
        """Set start position.

        Args:
            pos: Position string in format "line:col" or "line".
        """
        old_pos = self.pos
        self.pos = pos
        if old_pos != pos:
            self._mark_changed(description=f"Position: {old_pos} → {pos}")
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
            self._mark_changed(description=f"EndPos: {old_str} → {new_str}")
        return self

    def _format_notification_header(self, description: str) -> str:
        """Format header with line position info for text nodes."""
        return f"{self.display_id}: {description} (at {self.pos})"

    def get_display_name(self) -> str:
        """Return 'path:start-end' format."""
        try:
            start_line = int(self.pos.split(":")[0])
        except (ValueError, IndexError):
            start_line = 1

        if self.end_pos:
            try:
                end_line = int(self.end_pos.split(":")[0])
                return f"{self.path}:{start_line}-{end_line}"
            except (ValueError, IndexError):
                pass

        return f"{self.path}:{start_line}"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: just metadata line
        collapsed_text = f"[{self.path}: lines, pending traces]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary and Detail are same for TextNode - file content
        # Estimate based on token budget
        detail_tokens = self.tokens

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,  # TextNode has no summary state distinct from detail
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize TextNode to dict."""
        data = super().to_dict()
        data.update({
            "path": self.path,
            "pos": self.pos,
            "end_pos": self.end_pos,
            "media_type": self.media_type.value,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 1000),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            path=data.get("path", ""),
            pos=data.get("pos", "1:0"),
            end_pos=data.get("end_pos"),
            media_type=media_type,
        )
        return node


@dataclass
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
            "member_count": len(self.children_ids),
            "child_order": self.child_order,
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
            "summary_stale": self.summary_stale,
        }

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: cached summary if available, otherwise header.

        Args:
            cwd: Working directory for file access
            text_buffers: Not used by GroupNode but required for interface
        """
        # Check for valid child order
        ordered_children = self.child_order if self.child_order else list(self.children_ids)
        if not ordered_children:
            return "[Empty group]"

        if self.cached_summary and not self.summary_stale:
            return self.cached_summary

        # Summary is stale or missing - just show header
        return self.render_header(cwd=cwd)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: header only (children rendered by projection engine).

        Args:
            include_summary: If True, include summary content (for ALL state).
            cwd: Working directory for file access
            text_buffers: Not used by GroupNode but required for interface
        """
        # Check for valid child order
        ordered_children = self.child_order if self.child_order else list(self.children_ids)
        if not ordered_children:
            return "[Empty group]"

        # Group itself just renders header - children are rendered by projection engine
        return self.render_header(cwd=cwd)

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

    def get_display_name(self) -> str:
        """Return 'Group (N members)' format."""
        return f"Group ({len(self.children_ids)} members)"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Use child_order for iteration
        ordered_children = self.child_order if self.child_order else list(self.children_ids)

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
                    child_info = child.get_token_breakdown(cwd)
                    child_total += child_info.collapsed + child_info.summary + child_info.detail

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=summary_tokens,
            detail=0,  # Group has no detail of its own
            total=collapsed_tokens + summary_tokens + child_total if child_total else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize GroupNode to dict."""
        data = super().to_dict()
        data.update({
            "summary_prompt": self.summary_prompt,
            "cached_summary": self.cached_summary,
            "summary_stale": self.summary_stale,
            "last_child_versions": self.last_child_versions,
        })
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
            children_ids=set(data.get("children_ids", [])),
            child_order=data.get("child_order"),
            tokens=data.get("tokens", 500),
            state=NodeState(data.get("state", "summary")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            summary_prompt=data.get("summary_prompt"),
            cached_summary=data.get("cached_summary"),
            summary_stale=data.get("summary_stale", True),
            last_child_versions=data.get("last_child_versions", {}),
        )
        return node


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render collapsed: just the header."""
        return self.render_header(cwd=cwd)

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + message range."""
        parts: list[str] = [self.render_header(cwd=cwd)]
        if self.message_indices:
            parts.append(f"Messages: {self.message_indices[0]}-{self.message_indices[-1]}\n")
        return "".join(parts)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: header + message info.

        Child nodes (artifacts) are rendered by the projection engine.
        """
        parts: list[str] = [self.render_header(cwd=cwd)]
        if self.message_indices:
            parts.append(f"Messages: {self.message_indices[0]}-{self.message_indices[-1]}\n")
        if include_summary and self.children_ids:
            parts.append(f"Contains {len(self.children_ids)} artifacts\n")
        return "".join(parts)

    def set_status(self, status: str) -> TopicNode:
        """Set topic status.

        Args:
            status: New status string (e.g., "active", "resolved", "pending").
        """
        old_status = self.status
        self.status = status
        if old_status != status:
            self._mark_changed(description=f"Topic status: {old_status} → {status}")
        return self

    def get_display_name(self) -> str:
        """Return 'Topic: title' format."""
        return f"Topic: {self.title}"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: topic metadata
        collapsed_text = f"[Topic: {self.title} [{self.status}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Topics don't have summary vs detail distinction
        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=self.tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize TopicNode to dict."""
        data = super().to_dict()
        data.update({
            "title": self.title,
            "message_indices": self.message_indices,
            "status": self.status,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 1000),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            message_indices=data.get("message_indices", []),
            status=data.get("status", "active"),
        )
        return node


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render collapsed: just the header."""
        return self.render_header(cwd=cwd)

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + first line of content."""
        header = self.render_header(cwd=cwd)
        first_line = self.content.split("\n")[0][:80]
        if len(self.content) > len(first_line):
            first_line += "..."
        return header + first_line + "\n"

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: header + full content."""
        return self.render_header(cwd=cwd) + self.content

    def set_content(self, content: str) -> ArtifactNode:
        """Update artifact content.

        Args:
            content: New content string to replace existing content.
        """
        old_content = self.content
        self.content = content
        self._mark_changed(
            description=f"Content updated ({len(old_content)} → {len(content)} chars)",
        )
        return self

    def get_display_name(self) -> str:
        """Return 'TYPE:language' format."""
        lang_suffix = f":{self.language}" if self.language else ""
        return f"{self.artifact_type.upper()}{lang_suffix}"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
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
            collapsed=collapsed_tokens,
            summary=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ArtifactNode to dict."""
        data = super().to_dict()
        data.update({
            "artifact_type": self.artifact_type,
            "content": self.content,
            "language": self.language,
            "source_statement_id": self.source_statement_id,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 500),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            artifact_type=data.get("artifact_type", "code"),
            content=data.get("content", ""),
            language=data.get("language"),
            source_statement_id=data.get("source_statement_id"),
        )
        return node


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render collapsed: just the header."""
        return self.render_header(cwd=cwd)

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + truncated output."""
        header = self.render_header(cwd=cwd)
        summary_budget = min(self.tokens * 4, 500)
        output_preview = self.output[:summary_budget]
        if len(self.output) > summary_budget:
            output_preview += f"\n... [{len(self.output) - summary_budget} more chars]"
        return header + output_preview + "\n"

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: header + full output.

        When include_summary=True (ALL state), includes timing details.
        """
        char_budget = self.tokens * 4
        header = self.render_header(cwd=cwd)

        output_text = self.output
        if len(output_text) + len(header) > char_budget:
            available = char_budget - len(header) - 30
            output_text = output_text[:available] + f"\n... [truncated, {len(self.output)} total chars]"

        result = header + output_text
        if not result.endswith("\n"):
            result += "\n"

        # ALL state: include timing details
        if include_summary:
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
        self._mark_changed(description=f"Shell: {old_status.value} → running")
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

        self._mark_changed(
            description=f"Shell '{self.command}' {self.shell_status.value} (exit={exit_code})",
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
        self._mark_changed(
            description=f"Shell '{self.command}' timed out after {duration_ms:.0f}ms",
        )
        return self

    def set_cancelled(self) -> ShellNode:
        """Mark as cancelled by user."""
        self.shell_status = ShellStatus.CANCELLED
        self._mark_changed(
            description=f"Shell '{self.command}' cancelled",
        )
        return self

    def get_display_name(self) -> str:
        """Return 'Shell: command [STATUS]' format."""
        cmd_display = self.full_command[:40] + "..." if len(self.full_command) > 40 else self.full_command
        return f"Shell: {cmd_display} [{self.shell_status.value.upper()}]"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: command and status
        collapsed_text = f"[Shell: {self.full_command} [{self.shell_status.value}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: output content
        detail_tokens = count_tokens(self.output) if self.output else 0

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ShellNode to dict."""
        data = super().to_dict()
        data.update({
            "command": self.command,
            "args": self.args,
            "shell_status": self.shell_status.value,
            "exit_code": self.exit_code,
            "output": self.output,
            "truncated": self.truncated,
            "signal": self.signal,
            "duration_ms": self.duration_ms,
            "started_at_exec": self.started_at_exec,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 2000),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
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


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + error message if present."""
        header = self.render_header(cwd=cwd)
        if self.error_message:
            return header + f"Error: {self.error_message}\n"
        return header

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: header + timeout + holder + error + acquired_at.

        Args:
            include_summary: If True, include acquired_at (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Not used by LockNode but required for interface.
        """
        parts: list[str] = [self.render_header(cwd=cwd)]
        parts.append(f"Timeout: {self.timeout}s\n")

        if self.holder_pid:
            parts.append(f"Holder PID: {self.holder_pid}\n")

        if self.error_message:
            parts.append(f"Error: {self.error_message}\n")

        if include_summary and self.acquired_at:
            parts.append(f"Acquired at: {self.acquired_at:.3f}\n")

        return "".join(parts)

    def set_acquired(self, pid: int) -> LockNode:
        """Mark lock as acquired."""
        self.lock_status = LockStatus.ACQUIRED
        self.acquired_at = time.time()
        self.holder_pid = pid
        self._mark_changed(
            description=f"Lock '{self.lockfile}' acquired by PID {pid}",
        )
        return self

    def set_timeout(self) -> LockNode:
        """Mark lock acquisition as timed out."""
        self.lock_status = LockStatus.TIMEOUT
        self.error_message = f"Timed out after {self.timeout}s"
        self._mark_changed(
            description=f"Lock '{self.lockfile}' timed out",
        )
        return self

    def set_released(self) -> LockNode:
        """Mark lock as released."""
        self.lock_status = LockStatus.RELEASED
        self._mark_changed(
            description=f"Lock '{self.lockfile}' released",
        )
        return self

    def set_error(self, message: str) -> LockNode:
        """Mark lock operation as failed with error."""
        self.lock_status = LockStatus.ERROR
        self.error_message = message
        self._mark_changed(
            description=f"Lock '{self.lockfile}' error: {message}",
        )
        return self

    def get_display_name(self) -> str:
        """Return 'Lock: file [STATUS]' format."""
        return f"Lock: {self.lockfile} [{self.lock_status.value.upper()}]"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: lock info
        collapsed_text = f"[Lock: {self.lockfile} [{self.lock_status.value}]]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: error message if present
        detail_tokens = count_tokens(self.error_message) if self.error_message else 0

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize LockNode to dict."""
        data = super().to_dict()
        data.update({
            "lockfile": self.lockfile,
            "lock_status": self.lock_status.value,
            "timeout": self.timeout,
            "error_message": self.error_message,
            "acquired_at": self.acquired_at,
            "holder_pid": self.holder_pid,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 200),
            state=NodeState(data.get("state", "collapsed")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            lockfile=data.get("lockfile", ""),
            lock_status=LockStatus(data.get("lock_status", "pending")),
            timeout=data.get("timeout", 30.0),
            error_message=data.get("error_message"),
            acquired_at=data.get("acquired_at"),
            holder_pid=data.get("holder_pid"),
        )
        return node


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + turn info + token info."""
        parts: list[str] = [self.render_header(cwd=cwd)]

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
        parts.append(f"Total: {self.total_tokens_consumed:,} tokens across {self.turn_count} turns\n")

        return "".join(parts)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: summary + context graph + recent actions.

        Args:
            include_summary: If True, include recent actions (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Not used by SessionNode but required for interface.
        """
        parts: list[str] = [self.RenderSummary(cwd=cwd, text_buffers=text_buffers)]

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

        # Recent actions (ALL only via include_summary)
        if include_summary and self.recent_actions:
            parts.append("\nRecent Actions:\n")
            for i, action in enumerate(self.recent_actions[-5:]):  # Last 5 actions
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
        self._mark_changed(description=desc)
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
                    for child_id in node.children_ids:
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

    def get_display_name(self) -> str:
        """Return 'Session' format."""
        return "Session"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: session metadata line
        collapsed_text = f"[Session: Turn {self.turn_count} | {self.total_tokens_consumed:,} tokens]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Session node has statistics as detail
        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=self.tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize SessionNode to dict."""
        data = super().to_dict()
        data.update({
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
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 500),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "running"),  # Default to running for session node
            tick_frequency=tick_freq or TickFrequency.turn(),
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
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



@dataclass
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
    """

    role: str = "user"  # "user", "assistant", "tool_call", "tool_result"
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)

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
            if self.role == "tool_call":
                return f"Tool Call: {tool_name}"
            elif self.role == "tool_result":
                return "Tool Result"
            return f"Tool: {tool_name}"

        return self.originator

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "role": self.role,
            "originator": self.originator,
            "effective_role": self.effective_role,
            "content_length": len(self.content),
            "tool_name": self.tool_name,
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def _get_formatted_content(self, char_budget: int) -> str:
        """Get formatted content based on message type."""
        if self.role == "tool_call":
            return self._format_tool_call()
        elif self.role == "tool_result":
            return self._format_tool_result(char_budget)
        return self.content

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render collapsed: just the header."""
        return self.render_header(cwd=cwd)

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: truncated content preview."""
        char_budget = self.tokens * 4
        content = self._get_formatted_content(char_budget)
        preview_len = min(200, len(content))
        preview = content[:preview_len]
        if len(content) > preview_len:
            preview += "..."
        return preview + "\n"

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: full content within budget.

        Note: Block merging happens in ProjectionEngine._render_messages()
        which groups adjacent same-role messages before rendering.
        """
        char_budget = self.tokens * 4
        content = self._get_formatted_content(char_budget)

        if len(content) > char_budget:
            content = content[: char_budget - 20] + "\n... [truncated]"

        return content + "\n"

    def _format_tool_call(self) -> str:
        """Format a tool call message."""
        parts = [f"[Tool: {self.tool_name or 'unknown'}]"]
        if self.tool_args:
            args_str = ", ".join(f'{k}="{v}"' for k, v in self.tool_args.items())
            parts.append(f" {args_str}")
        return "".join(parts)

    def _format_tool_result(self, char_budget: int) -> str:
        """Format a tool result message."""
        result = self.content
        if len(result) > char_budget - 20:
            result = result[: char_budget - 40] + "\n... [truncated]"
        return f"[Result] {result}"

    def set_content(self, content: str) -> MessageNode:
        """Update message content."""
        old_len = len(self.content)
        self.content = content
        self._mark_changed(
            description=f"Message content updated ({old_len} → {len(content)} chars)",
        )
        return self

    def get_display_name(self) -> str:
        """Return 'Role #N' format using display_sequence."""
        seq = self.display_sequence or 0
        # Use actual role for display (user, assistant, tool_call, tool_result)
        role_display = self.role.replace("_", " ").title()
        return f"{role_display} #{seq}"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: role and char count
        collapsed_text = f"[{self.role.upper()}: {len(self.content)} chars]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: full message content
        detail_tokens = count_tokens(self.content)

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MessageNode to dict."""
        data = super().to_dict()
        data.update({
            "role": self.role,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 500),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=originator,
            role=data.get("role", "user"),
            content=data.get("content", ""),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args", {}),
        )
        return node

@dataclass
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
    work_status: str = "active"  # active, paused, done
    files: list[dict[str, str]] = field(default_factory=list)  # [{path, mode}]
    dependencies: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)  # [{agent_id, file, their_mode, their_intent}]
    agent_id: str = ""

    @property
    def node_type(self) -> str:
        return "work"

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "intent": self.intent,
            "status": self.work_status,
            "file_count": len(self.files),
            "conflict_count": len(self.conflicts),
            "agent_id": self.agent_id,
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + agent + files + conflicts."""
        parts: list[str] = [self.render_header(cwd=cwd)]
        parts.append(f"Agent: {self.agent_id}\n")

        # Show files being worked on
        if self.files:
            parts.append("\nFiles:\n")
            for f in self.files:
                mode_indicator = "[W]" if f.get("mode") == "write" else "[R]"
                parts.append(f"  {mode_indicator} {f.get('path', '')}\n")

        # Show conflicts (always important)
        if self.conflicts:
            parts.append("\n--- CONFLICTS ---\n")
            for c in self.conflicts:
                parts.append(
                    f"  Agent {c.get('agent_id', '?')}: {c.get('file', '?')} "
                    f"[{c.get('their_mode', '?')}] - {c.get('their_intent', '?')}\n"
                )

        return "".join(parts)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: summary + dependencies.

        Args:
            include_summary: If True, include all detail (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Not used by WorkNode but required for interface.
        """
        parts: list[str] = [self.render_header(cwd=cwd)]
        parts.append(f"Agent: {self.agent_id}\n")

        # Show files being worked on
        if self.files:
            parts.append("\nFiles:\n")
            for f in self.files:
                mode_indicator = "[W]" if f.get("mode") == "write" else "[R]"
                parts.append(f"  {mode_indicator} {f.get('path', '')}\n")

        # Show dependencies (DETAILS and ALL)
        if self.dependencies:
            parts.append("\nDependencies:\n")
            for dep in self.dependencies:
                parts.append(f"  [R] {dep}\n")

        # Show conflicts (always important)
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
            self._mark_changed(description=f"Intent: {old_intent[:30]}... → {intent[:30]}...")
        return self

    def set_files(self, files: list[dict[str, str]]) -> WorkNode:
        """Update files being worked on."""
        old_count = len(self.files)
        self.files = files
        if old_count != len(files):
            self._mark_changed(description=f"Files: {old_count} → {len(files)}")
        return self

    def set_conflicts(self, conflicts: list[dict[str, str]]) -> WorkNode:
        """Update detected conflicts."""
        old_count = len(self.conflicts)
        self.conflicts = conflicts
        if len(conflicts) != old_count:
            self._mark_changed(
                description=f"Work conflicts: {old_count} → {len(conflicts)}",
            )
        return self

    def get_display_name(self) -> str:
        """Return 'Work: intent [status]' format."""
        intent_display = self.intent[:30] + "..." if len(self.intent) > 30 else self.intent
        return f"Work: {intent_display} [{self.work_status}]"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: work metadata
        collapsed_text = f"[Work: {self.intent} [{self.work_status}] {len(self.files)}f {len(self.conflicts)}c]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: file list and conflict details
        detail_tokens = self.tokens

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize WorkNode to dict."""
        data = super().to_dict()
        data.update({
            "intent": self.intent,
            "work_status": self.work_status,
            "files": self.files,
            "dependencies": self.dependencies,
            "conflicts": self.conflicts,
            "agent_id": self.agent_id,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 200),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            intent=data.get("intent", ""),
            work_status=data.get("work_status", "active"),
            files=data.get("files", []),
            dependencies=data.get("dependencies", []),
            conflicts=data.get("conflicts", []),
            agent_id=data.get("agent_id", ""),
        )
        return node


@dataclass
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
        default=None, repr=False, compare=False
    )

    # Tool child nodes: tool_name -> node_id
    _tool_nodes: dict[str, str] = field(default_factory=dict, repr=False)

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
            "tokens": self.tokens,
            "state": self.state.value,
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

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render collapsed view: header + status if not connected."""
        header = self.render_header(cwd=cwd)
        status = self._render_status_message()
        if status:
            return header + status
        return header

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + tool names + resource names."""
        header = self.render_header(cwd=cwd)
        status = self._render_status_message()
        if status:
            return header + status

        parts: list[str] = [header]
        tool_names = [t.get("name", "?") for t in self.tools]
        parts.append(f"Tools: {', '.join(tool_names)}\n")
        if self.resources:
            resource_names = [r.get("name", "?") for r in self.resources]
            parts.append(f"Resources: {', '.join(resource_names)}\n")
        return "".join(parts)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail view: header + resources/prompts.

        Tool details come from child MCPToolNode nodes (rendered by projection).
        When include_summary=True (ALL state), includes summary info.
        """
        header = self.render_header(cwd=cwd)
        status = self._render_status_message()
        if status:
            return header + status

        parts: list[str] = [header]

        # Include summary info for ALL state
        if include_summary:
            tool_names = [t.get("name", "?") for t in self.tools]
            parts.append(f"Tools: {', '.join(tool_names)}\n")

        # Usage hint - show code block wrapper so agent knows how to call
        parts.append("To call tools:\n")
        parts.append("```python/acrepl\n")
        parts.append(f"result = {self.server_name}.tool_name(arg=value)\n")
        parts.append("```\n\n")

        # Resources (server-level, not per-tool)
        if self.resources:
            parts.append("### Resources\n\n")
            for res in self.resources:
                parts.append(f"- `{res.get('uri', '?')}`")
                if res.get("description"):
                    parts.append(f": {res['description']}")
                parts.append("\n")

        # Prompts (server-level)
        if self.prompts:
            parts.append("\n### Prompts\n\n")
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
                    node._mark_changed(
                        description=f"Tool '{tool_name}' removed from {self.server_name}"
                    )
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
                    tokens=200,
                    state=NodeState.COLLAPSED,
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
                        node._mark_changed(
                            description=f"Tool '{tool_name}' schema updated"
                        )

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
        self._mark_changed(
            description=f"MCP {self.server_name}: {self.status}, {len(self.tools)} tools"
        )

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
        self._mark_changed(
            description=f"MCP tool '{tool_name}' completed",
            content=str(result)[:200] if result else error,
        )

        # Fire event if callback is set
        if self._on_result_callback:
            self._on_result_callback("mcp_result", {
                "call_id": call_id,
                "server_name": self.server_name,
                "tool_name": tool_name,
                "result": result,
                "error": error,
                "duration_ms": duration_ms,
            })

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

    def get_display_name(self) -> str:
        """Return 'MCP: name [status]' format."""
        return f"MCP: {self.server_name} [{self.status.upper()}]"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: server info
        collapsed_text = f"[MCP: {self.server_name} [{self.status}] {len(self.tools)} tools]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Summary: tool names
        summary_text = ", ".join(t.get("name", "?") for t in self.tools)
        summary_tokens = count_tokens(summary_text) if summary_text else 0

        # Detail: full tool documentation
        detail_tokens = self.tokens

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCPServerNode to dict."""
        data = super().to_dict()
        data.update({
            "server_name": self.server_name,
            "status": self.status,
            "error_message": self.error_message,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts,
            "_tool_nodes": self._tool_nodes,
        })
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
            children_ids=set(data.get("children_ids", [])),
            child_order=data.get("child_order"),
            tokens=data.get("tokens", 1000),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            server_name=data.get("server_name", ""),
            status=data.get("status", "disconnected"),
            error_message=data.get("error_message"),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            prompts=data.get("prompts", []),
            _tool_nodes=data.get("_tool_nodes", {}),
        )
        return node


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
            "version": self.version,
        }

    def RenderCollapsed(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render collapsed: just tool name."""
        return f"`{self.tool_name}`\n"

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: name + truncated description."""
        desc = self.description
        if len(desc) > 80:
            desc = desc[:80] + "..."
        return f"**{self.tool_name}**: {desc}\n"

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detailed tool documentation.

        The description is treated as markdown and rendered as-is.
        """
        if include_summary:
            # ALL: full schema with server prefix
            parts = [f"### `{self.server_name}.{self.tool_name}()`\n\n"]
        else:
            # DETAILS: just tool name header
            parts = [f"### `{self.tool_name}`\n\n"]

        # Description treated as markdown
        parts.append(f"{self.description}\n\n")

        props = self.input_schema.get("properties", {})
        if props:
            required = set(self.input_schema.get("required", []))
            if include_summary:
                # ALL: full parameter details
                parts.append("**Parameters:**\n")
                for param, param_schema in props.items():
                    param_type = param_schema.get("type", "any")
                    param_desc = param_schema.get("description", "")
                    req_marker = " (required)" if param in required else ""
                    parts.append(f"- `{param}` ({param_type}){req_marker}: {param_desc}\n")
                parts.append("\n")
            else:
                # DETAILS: compact parameter list
                parts.append("**Parameters:** ")
                param_strs = []
                for param in props:
                    req = "*" if param in required else ""
                    param_strs.append(f"`{param}`{req}")
                parts.append(", ".join(param_strs))
                parts.append("\n")

        return "".join(parts)

    def get_display_name(self) -> str:
        """Return tool name for display."""
        return f"{self.server_name}.{self.tool_name}"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        collapsed_tokens = count_tokens(f"`{self.tool_name}`\n")

        desc = self.description[:80] if len(self.description) > 80 else self.description
        summary_tokens = count_tokens(f"**{self.tool_name}**: {desc}\n")

        detail_tokens = self.tokens

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=summary_tokens,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MCPToolNode to dict."""
        data = super().to_dict()
        data.update({
            "tool_name": self.tool_name,
            "server_name": self.server_name,
            "description": self.description,
            "input_schema": self.input_schema,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 200),
            state=NodeState(data.get("state", "collapsed")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            tool_name=data.get("tool_name", ""),
            server_name=data.get("server_name", ""),
            description=data.get("description", ""),
            input_schema=data.get("input_schema", {}),
        )


@dataclass
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

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + server status list."""
        lines = [self.render_header(cwd=cwd).rstrip()]

        # Server status list
        if self.server_states:
            lines.append("### Server Status")
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

        return "\n".join(lines)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: summary + capabilities + events.

        Args:
            include_summary: If True, include recent events (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Not used but required for interface.
        """
        lines = [self.render_header(cwd=cwd).rstrip()]

        # Server status list
        if self.server_states:
            lines.append("### Server Status")
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

        # Capabilities (DETAILS and ALL)
        if self.tool_counts:
            lines.append("")
            lines.append("### Capabilities")
            for name in sorted(self.server_states.keys()):
                tools = self.tool_counts.get(name, 0)
                resources = self.resource_counts.get(name, 0)
                lines.append(f"- {name}: {tools} tools, {resources} resources")

        # Recent events (ALL only via include_summary)
        if include_summary and self.connection_events:
            lines.append("")
            lines.append("### Recent Events")
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
        self._mark_changed(description=change_desc)
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

    def get_display_name(self) -> str:
        """Return 'MCP Manager' format."""
        return f"MCP Manager ({len(self.server_states)} servers)"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
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
            collapsed=collapsed_tokens,
            summary=summary_tokens,
            detail=self.tokens,
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
            children_ids=set(d.get("children_ids", [])),
            tokens=d.get("tokens", 300),
            state=NodeState(d.get("state", "summary")),
            mode=d.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=d.get("version", 0),
            created_at=d.get("created_at", 0.0),
            updated_at=d.get("updated_at", 0.0),
            tags=d.get("tags", {}),
            display_sequence=d.get("display_sequence"),
            originator=d.get("originator"),
            server_states=d.get("server_states", {}),
            tool_counts=d.get("tool_counts", {}),
            resource_counts=d.get("resource_counts", {}),
            connection_events=d.get("connection_events", []),
            max_events=d.get("max_events", 10),
        )
        return node





@dataclass
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
    relation: str = "self"  # self, parent, child, peer
    task: str = ""
    agent_state: str = "running"
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
            "relation": self.relation,
            "task": self.task,
            "agent_state": self.agent_state,
            "message_count": self.message_count,
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def RenderSummary(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render summary: header + type + state."""
        parts = [self.render_header(cwd=cwd)]
        parts.append(f"  Type: {self.agent_type}\n")
        parts.append(f"  State: {self.agent_state}\n")
        return "".join(parts)

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: summary + task + messages.

        Args:
            include_summary: If True, include all detail (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Not used but required for interface.
        """
        parts = [self.render_header(cwd=cwd)]
        parts.append(f"  Type: {self.agent_type}\n")
        parts.append(f"  State: {self.agent_state}\n")

        if self.task:
            parts.append(f"  Task: {self.task}\n")

        if self.message_count > 0:
            parts.append(f"  Messages: {self.message_count} pending\n")

        return "".join(parts)

    def update_state(self, agent_state: str) -> AgentNode:
        """Update the agent's state."""
        old_state = self.agent_state
        self.agent_state = agent_state
        if old_state != agent_state:
            self._mark_changed(description=f"Agent state: {old_state} → {agent_state}")
        return self

    def update_message_count(self, count: int) -> AgentNode:
        """Update pending message count."""
        old_count = self.message_count
        self.message_count = count
        if old_count != count:
            self._mark_changed(description=f"Messages: {old_count} → {count}")
        return self

    def get_display_name(self) -> str:
        """Return 'Agent: id [state]' format."""
        return f"Agent: {self.agent_id} [{self.agent_state.upper()}]"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: agent info
        collapsed_text = f"[Agent: {self.agent_id} [{self.agent_state}] {self.message_count}m]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: task and session info
        detail_tokens = count_tokens(self.task) if self.task else 0

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=detail_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize AgentNode to dict."""
        data = super().to_dict()
        data.update({
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "relation": self.relation,
            "task": self.task,
            "agent_state": self.agent_state,
            "session_id": self.session_id,
            "message_count": self.message_count,
        })
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 200),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", "default"),
            relation=data.get("relation", "self"),
            task=data.get("task", ""),
            agent_state=data.get("agent_state", "running"),
            session_id=data.get("session_id", ""),
            message_count=data.get("message_count", 0),
        )
        return node


@dataclass
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
            "tokens": self.tokens,
            "state": self.state.value,
        }

    def get_display_name(self) -> str:
        """Return display name: '{node_display_id} {description}'.

        Example: 'context_1 Recomputed' or 'text_5 State: collapsed → details'
        """
        return f"{self.node_display_id} {self.description}"

    def get_token_breakdown(self, cwd: str = ".") -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.core.tokens import count_tokens

        from .headers import TokenInfo

        # Collapsed: just the description (shown in header)
        collapsed_tokens = count_tokens(self.description) + 5

        return TokenInfo(
            collapsed=collapsed_tokens,
            summary=0,
            detail=self.tokens if self.content else 0,
        )

    def RenderDetail(
        self,
        include_summary: bool = False,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Render detail: header + originator + content.

        Args:
            include_summary: If True, include all content (for ALL state).
            cwd: Working directory for file access.
            text_buffers: Not used by TraceNode but required for interface.
        """
        # Header includes display_id and description via get_display_name()
        parts = [self.render_header(cwd=cwd)]

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
            "children_ids": list(self.children_ids),
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "display_sequence": self.display_sequence,
            "originator": self.originator,
            "node": self.node,
            "node_display_id": self.node_display_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "description": self.description,
            "content": self.content,
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
            children_ids=set(data.get("children_ids", [])),
            tokens=data.get("tokens", 200),
            state=NodeState(data.get("state", "collapsed")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            node=data.get("node", ""),
            node_display_id=data.get("node_display_id", ""),
            old_version=data.get("old_version", 0),
            new_version=data.get("new_version", 0),
            description=data.get("description", ""),
            content=data.get("content"),
        )
