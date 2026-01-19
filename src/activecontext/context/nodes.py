"""Context node types for the context DAG.

This module defines the typed node hierarchy:
- ContextNode: Base class with common fields and notification
- ViewNode: File content view
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

from activecontext.context.state import NodeState, TickFrequency
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


@dataclass
class Trace:
    """Represents a change between versions."""

    node_id: str
    old_version: int
    new_version: int
    description: str
    content: str | None = None  # Optional trace content


# Type alias for hooks
OnChildChangedHook = Callable[["ContextNode", "ContextNode", Trace | None], None]


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
        pending_traces: Accumulated traces until commit
        tags: Arbitrary metadata
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_ids: set[str] = field(default_factory=set)
    children_ids: set[str] = field(default_factory=set)

    # Rendering configuration
    tokens: int = 1000
    state: NodeState = NodeState.DETAILS
    mode: str = "paused"
    tick_frequency: TickFrequency | None = None

    # Version tracking
    version: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Trace accumulation
    pending_traces: list[Trace] = field(default_factory=list)

    # Metadata
    tags: dict[str, Any] = field(default_factory=dict)

    # Split architecture: optional reference to shared ContentData
    # When set, nodes can delegate content storage to ContentRegistry
    content_id: str | None = field(default=None)

    # Graph reference (set by ContextGraph.add_node)
    _graph: ContextGraph | None = field(default=None, repr=False)

    # Optional hook for child change notifications
    _on_child_changed_hook: OnChildChangedHook | None = field(default=None, repr=False)

    @property
    @abstractmethod
    def node_type(self) -> str:
        """Return the node type identifier."""
        ...

    @abstractmethod
    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this node."""
        ...

    @abstractmethod
    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render this node's content within token budget."""
        ...

    def Recompute(self) -> None:
        """Recompute this node's content. Called during tick for running nodes.

        Default implementation just bumps version and notifies parents.
        Subclasses should override to implement actual recomputation.
        """
        self._mark_changed()

    def _mark_changed(self, trace: Trace | None = None) -> None:
        """Mark this node as changed and notify parents."""
        self.version += 1
        self.updated_at = time.time()
        if trace:
            self.pending_traces.append(trace)
        self.notify_parents(trace)

    def notify_parents(self, trace: Trace | None = None) -> None:
        """Notify all parent nodes of a change."""
        if not self._graph:
            return

        for parent_id in self.parent_ids:
            parent = self._graph.get_node(parent_id)
            if parent:
                parent.on_child_changed(self, trace)

    def on_child_changed(self, child: ContextNode, trace: Trace | None = None) -> None:
        """Handle notification that a child has changed.

        Default implementation propagates upward. GroupNode overrides
        to invalidate summary and generate traces.
        """
        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, trace)

        # Propagate upward
        self.notify_parents(trace)

    def set_on_child_changed_hook(self, hook: OnChildChangedHook | None) -> None:
        """Register a hook for child change notifications."""
        self._on_child_changed_hook = hook

    def clear_pending_traces(self) -> None:
        """Clear accumulated diffs (called after projection render)."""
        self.pending_traces.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dict for persistence.

        Subclasses should override to include their specific fields.
        """
        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "parent_ids": list(self.parent_ids),
            "children_ids": list(self.children_ids),
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "tick_frequency": self.tick_frequency.to_dict() if self.tick_frequency else None,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextNode:
        """Deserialize node from dict.

        This is a factory method that dispatches to the appropriate subclass.
        """
        node_type = data.get("node_type")
        if node_type == "view":
            return ViewNode._from_dict(data)
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
        elif node_type == "markdown":
            return MarkdownNode._from_dict(data)
        elif node_type == "agent":
            return AgentNode._from_dict(data)
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    # Fluent API for mode control
    def Run(self, freq: TickFrequency | None = None) -> ContextNode:
        """Enable tick recomputation with given frequency.

        Args:
            freq: Tick frequency (defaults to turn() if not specified)
        """
        self.mode = "running"
        self.tick_frequency = freq or TickFrequency.turn()
        if self._graph:
            self._graph._running_nodes.add(self.node_id)
        return self

    def Pause(self) -> ContextNode:
        """Disable tick recomputation."""
        self.mode = "paused"
        if self._graph:
            self._graph._running_nodes.discard(self.node_id)
        return self

    def SetTokens(self, n: int) -> ContextNode:
        """Set token budget."""
        self.tokens = n
        return self

    def SetState(self, s: NodeState) -> ContextNode:
        """Set rendering state."""
        self.state = s
        return self


@dataclass
class ViewNode(ContextNode):
    """View of a file or file region.

    Attributes:
        path: File path relative to cwd
        pos: Start position as "line:col" (1-indexed)
        end_pos: End position as "line:col" (None = to end of file)
        media_type: Content media type (auto-detected from file extension)
    """

    path: str = ""
    pos: str = "1:0"
    end_pos: str | None = None
    media_type: MediaType = field(default=MediaType.TEXT)

    def __post_init__(self) -> None:
        """Auto-detect media type from file extension."""
        if self.path and self.media_type == MediaType.TEXT:
            self.media_type = detect_media_type(self.path)

    @property
    def node_type(self) -> str:
        return "view"

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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render file content based on rendering state."""
        import os

        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        effective_tokens = tokens or self.tokens
        char_budget = tokens_to_chars(effective_tokens, self.media_type)

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
                lines = f.readlines()
        except FileNotFoundError:
            return f"[File not found: {self.path}]"
        except OSError as e:
            return f"[Error reading {self.path}: {e}]"

        # Apply line range
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line else len(lines)
        selected_lines = lines[start_idx:end_idx]

        # COLLAPSED: only show metadata
        if self.state == NodeState.COLLAPSED:
            line_count = len(selected_lines)
            trace_count = len(self.pending_traces)
            return f"[{self.path}: {line_count} lines, {trace_count} pending traces]\n"

        # Build output with line numbers (for SUMMARY, DETAILS, ALL)
        output_parts: list[str] = []
        output_parts.append(f"=== {self.path} (lines {start_line}-{start_idx + len(selected_lines)}) ===\n")

        chars_used = len(output_parts[0])
        lines_included = 0

        for i, line in enumerate(selected_lines):
            line_num = start_idx + i + 1
            formatted = f"{line_num:4d} | {line}"

            if chars_used + len(formatted) > char_budget:
                remaining = len(selected_lines) - lines_included
                output_parts.append(f"... [{remaining} more lines]\n")
                break

            output_parts.append(formatted)
            chars_used += len(formatted)
            lines_included += 1

        content = "".join(output_parts)

        # At ALL state, append pending traces
        if self.state == NodeState.ALL and self.pending_traces:
            trace_section = "\n--- Pending Traces ---\n"
            for trace in self.pending_traces:
                trace_section += f"[v{trace.old_version}→v{trace.new_version}] {trace.description}\n"
                if trace.content:
                    trace_section += trace.content + "\n"
            content += trace_section

        return content

    def SetPos(self, pos: str) -> ViewNode:
        """Set start position."""
        self.pos = pos
        return self

    def SetEndPos(self, end_pos: str | None) -> ViewNode:
        """Set end position."""
        self.end_pos = end_pos
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize ViewNode to dict."""
        data = super().to_dict()
        data.update({
            "path": self.path,
            "pos": self.pos,
            "end_pos": self.end_pos,
            "media_type": self.media_type.value,
        })
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ViewNode:
        """Deserialize ViewNode from dict."""
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
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
            "summary_stale": self.summary_stale,
        }

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render group based on rendering state."""
        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        if not self.children_ids:
            return "[Empty group]"

        # COLLAPSED: only show metadata
        if self.state == NodeState.COLLAPSED:
            member_count = len(self.children_ids)
            trace_count = len(self.pending_traces)
            return f"[Group: {member_count} members, {trace_count} traces]\n"

        effective_tokens = tokens or self.tokens

        # SUMMARY: use cached summary if available, otherwise fall back to children
        if self.state == NodeState.SUMMARY:
            if self.cached_summary and not self.summary_stale:
                return self.cached_summary

            # Summary is stale or missing, fall back to rendering children
            parts: list[str] = [f"=== Group ({len(self.children_ids)} members) [summary stale] ===\n"]

            if self._graph:
                per_child_tokens = effective_tokens // len(self.children_ids)
                for child_id in self.children_ids:
                    child = self._graph.get_node(child_id)
                    if child:
                        child_content = child.Render(tokens=per_child_tokens, cwd=cwd)
                        parts.append(child_content)
                        parts.append("\n")

            return "".join(parts)

        # DETAILS or ALL: render children with their own settings
        detail_parts: list[str] = [f"=== Group ({len(self.children_ids)} members) ===\n"]

        if self._graph:
            per_child_tokens = effective_tokens // len(self.children_ids)
            for child_id in self.children_ids:
                child = self._graph.get_node(child_id)
                if child:
                    child_content = child.Render(tokens=per_child_tokens, cwd=cwd)
                    detail_parts.append(child_content)
                    detail_parts.append("\n")

        # At ALL state, include pending traces from children
        if self.state == NodeState.ALL and self.pending_traces:
            detail_parts.append("--- Group Traces ---\n")
            for trace in self.pending_traces:
                detail_parts.append(f"[{trace.node_id}] {trace.description}\n")

        return "".join(detail_parts)

    def on_child_changed(self, child: ContextNode, trace: Trace | None = None) -> None:
        """Handle child change: track version, generate trace, propagate."""
        old_version = self.last_child_versions.get(child.node_id, 0)
        new_version = child.version

        if new_version != old_version:
            # Generate trace for this change
            group_trace = Trace(
                node_id=child.node_id,
                old_version=old_version,
                new_version=new_version,
                description=f"{child.node_type} '{child.node_id}' changed",
                content=trace.content if trace else None,
            )
            self.pending_traces.append(group_trace)
            self.summary_stale = True
            self.last_child_versions[child.node_id] = new_version

        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, trace)

        # Propagate upward
        self.notify_parents(trace)

    def invalidate_summary(self) -> None:
        """Mark summary as needing regeneration."""
        self.summary_stale = True

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
            tokens=data.get("tokens", 500),
            state=NodeState(data.get("state", "summary")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render topic header and children."""
        parts: list[str] = [f"=== Topic: {self.title} ({self.status}) ===\n"]

        if self.message_indices:
            parts.append(f"Messages: {self.message_indices[0]}-{self.message_indices[-1]}\n")

        # Render children (artifacts)
        if self._graph and self.children_ids:
            effective_tokens = tokens or self.tokens
            per_child_tokens = effective_tokens // len(self.children_ids)

            for child_id in self.children_ids:
                child = self._graph.get_node(child_id)
                if child:
                    parts.append(child.Render(tokens=per_child_tokens, cwd=cwd))
                    parts.append("\n")

        return "".join(parts)

    def set_status(self, status: str) -> TopicNode:
        """Set topic status."""
        self.status = status
        self._mark_changed()
        return self

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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render artifact content."""
        effective_tokens = tokens or self.tokens
        char_budget = effective_tokens * 4

        # Header
        header = f"[{self.artifact_type.upper()}"
        if self.language:
            header += f": {self.language}"
        header += "]\n"

        # Truncate content if needed
        content = self.content
        if len(content) > char_budget - len(header):
            content = content[: char_budget - len(header) - 20] + "\n... [truncated]"

        return header + content

    def set_content(self, content: str) -> ArtifactNode:
        """Update artifact content."""
        old_content = self.content
        self.content = content

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Content updated ({len(old_content)} → {len(content)} chars)",
        )
        self._mark_changed(trace)
        return self

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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render shell command status and output based on rendering state."""
        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        effective_tokens = tokens or self.tokens
        char_budget = effective_tokens * 4

        # Build header based on status
        status_str = self.shell_status.value.upper()
        if self.shell_status == ShellStatus.RUNNING:
            elapsed = time.time() - (self.started_at_exec or self.created_at)
            status_str = f"RUNNING {elapsed:.1f}s"
        elif self.is_complete:
            status_str = f"{self.shell_status.value.upper()} exit={self.exit_code}"

        header = f"=== Shell: {self.full_command} [{status_str}] ===\n"

        # COLLAPSED: only show header
        if self.state == NodeState.COLLAPSED:
            line_count = self.output.count("\n") if self.output else 0
            return f"[Shell: {self.full_command} - {status_str}, {line_count} lines]\n"

        # SUMMARY: show header + truncated output
        if self.state == NodeState.SUMMARY:
            summary_budget = min(char_budget, 500)
            output_preview = self.output[:summary_budget]
            if len(self.output) > summary_budget:
                output_preview += f"\n... [{len(self.output) - summary_budget} more chars]"
            return header + output_preview + "\n"

        # DETAILS or ALL: show full output within budget
        output_text = self.output
        if len(output_text) + len(header) > char_budget:
            available = char_budget - len(header) - 30
            output_text = output_text[:available] + f"\n... [truncated, {len(self.output)} total chars]"

        result = header + output_text
        if not result.endswith("\n"):
            result += "\n"

        # At ALL state, include timing details
        if self.state == NodeState.ALL:
            result += f"--- Duration: {self.duration_ms:.0f}ms"
            if self.truncated:
                result += " (output was truncated)"
            if self.signal:
                result += f", killed by {self.signal}"
            result += " ---\n"

        return result

    def set_running(self) -> ShellNode:
        """Mark as running (called when subprocess starts)."""
        self.shell_status = ShellStatus.RUNNING
        self.started_at_exec = time.time()
        self._mark_changed()
        return self

    def set_completed(
        self,
        exit_code: int,
        output: str,
        duration_ms: float,
        truncated: bool = False,
        signal: str | None = None,
    ) -> ShellNode:
        """Mark as completed with result (called when subprocess finishes)."""
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

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Shell '{self.command}' {self.shell_status.value} (exit={exit_code})",
            content=output[:500] if output else None,
        )
        self._mark_changed(trace)
        return self

    def set_timeout(self, output: str, duration_ms: float) -> ShellNode:
        """Mark as timed out."""
        self.shell_status = ShellStatus.TIMEOUT
        self.output = output
        self.duration_ms = duration_ms
        self.exit_code = -1

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Shell '{self.command}' timed out after {duration_ms:.0f}ms",
        )
        self._mark_changed(trace)
        return self

    def set_cancelled(self) -> ShellNode:
        """Mark as cancelled by user."""
        self.shell_status = ShellStatus.CANCELLED
        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Shell '{self.command}' cancelled",
        )
        self._mark_changed(trace)
        return self

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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render lock status based on rendering state."""
        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        status_str = self.lock_status.value.upper()
        if self.lock_status == LockStatus.PENDING:
            elapsed = time.time() - self.created_at
            status_str = f"PENDING {elapsed:.1f}s"
        elif self.lock_status == LockStatus.ACQUIRED and self.acquired_at:
            held_for = time.time() - self.acquired_at
            status_str = f"ACQUIRED {held_for:.1f}s"

        # COLLAPSED: minimal info
        if self.state == NodeState.COLLAPSED:
            return f"[Lock: {self.lockfile} - {status_str}]\n"

        header = f"=== Lock: {self.lockfile} [{status_str}] ===\n"

        # SUMMARY: show header + basic info
        if self.state == NodeState.SUMMARY:
            if self.error_message:
                return header + f"Error: {self.error_message}\n"
            return header

        # DETAILS or ALL: show full details
        parts: list[str] = [header]
        parts.append(f"Timeout: {self.timeout}s\n")

        if self.holder_pid:
            parts.append(f"Holder PID: {self.holder_pid}\n")

        if self.error_message:
            parts.append(f"Error: {self.error_message}\n")

        if self.state == NodeState.ALL and self.acquired_at:
            parts.append(f"Acquired at: {self.acquired_at:.3f}\n")

        return "".join(parts)

    def set_acquired(self, pid: int) -> LockNode:
        """Mark lock as acquired."""
        self.lock_status = LockStatus.ACQUIRED
        self.acquired_at = time.time()
        self.holder_pid = pid

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Lock '{self.lockfile}' acquired by PID {pid}",
        )
        self._mark_changed(trace)
        return self

    def set_timeout(self) -> LockNode:
        """Mark lock acquisition as timed out."""
        self.lock_status = LockStatus.TIMEOUT
        self.error_message = f"Timed out after {self.timeout}s"

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Lock '{self.lockfile}' timed out",
        )
        self._mark_changed(trace)
        return self

    def set_released(self) -> LockNode:
        """Mark lock as released."""
        self.lock_status = LockStatus.RELEASED

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Lock '{self.lockfile}' released",
        )
        self._mark_changed(trace)
        return self

    def set_error(self, message: str) -> LockNode:
        """Mark lock operation as failed with error."""
        self.lock_status = LockStatus.ERROR
        self.error_message = message

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Lock '{self.lockfile}' error: {message}",
        )
        self._mark_changed(trace)
        return self

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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render session statistics for agent awareness."""
        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        # COLLAPSED: minimal info
        if self.state == NodeState.COLLAPSED:
            return f"[Session: turn {self.turn_count}, {self.total_tokens_consumed:,} tokens]\n"

        parts: list[str] = ["=== Session Context ===\n"]

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

        # Context graph summary (DETAILS and ALL)
        if self.state in (NodeState.DETAILS, NodeState.ALL) and self.node_count_by_type:
            parts.append("\nContext Graph:\n")
            for node_type, count in sorted(self.node_count_by_type.items()):
                running_info = ""
                if node_type == "view" and self.running_node_count > 0:
                    running_info = f" ({self.running_node_count} running)"
                parts.append(f"  {node_type}s: {count}{running_info}\n")
            if self.graph_depth > 0:
                parts.append(f"  depth: {self.graph_depth}\n")

        # Recent actions (ALL only)
        if self.state == NodeState.ALL and self.recent_actions:
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

        self._mark_changed()
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
        actor: Who produced this message (e.g., "user", "agent", "tool:grep")
        tool_name: Tool name for tool_call/tool_result messages
        tool_args: Tool arguments (for tool_call messages)
    """

    role: str = "user"  # "user", "assistant", "tool_call", "tool_result"
    content: str = ""
    actor: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    @property
    def node_type(self) -> str:
        return "message"

    @property
    def effective_role(self) -> str:
        """Return the role for LLM alternation (USER or ASSISTANT)."""
        return "USER" if self.actor == "user" else "ASSISTANT"

    @property
    def display_label(self) -> str:
        """Return the human-friendly label for this message.

        Mapping:
        - actor="user" → configured user name (default "User")
        - actor="agent" → "Agent"
        - actor="agent:plan" → "Agent (Plan)"
        - actor="agent:{name}" → "Child: {name}"
        - actor="tool:{name}" with role=tool_call → "Tool Call: {name}"
        - actor="tool:{name}" with role=tool_result → "Tool Result"
        """
        if not self.actor:
            return "Unknown"

        if self.actor == "user":
            return "User"  # Will be overridden by config at render time

        if self.actor == "agent":
            return "Agent"

        if self.actor == "agent:plan":
            return "Agent (Plan)"

        if self.actor.startswith("agent:"):
            subagent_name = self.actor[6:]  # Remove "agent:" prefix
            return f"Child: {subagent_name}"

        if self.actor.startswith("tool:"):
            tool_name = self.actor[5:]  # Remove "tool:" prefix
            if self.role == "tool_call":
                return f"Tool Call: {tool_name}"
            elif self.role == "tool_result":
                return "Tool Result"
            return f"Tool: {tool_name}"

        return self.actor

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "role": self.role,
            "actor": self.actor,
            "effective_role": self.effective_role,
            "content_length": len(self.content),
            "tool_name": self.tool_name,
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render message content based on rendering state.

        Note: This renders a single message. Block merging happens in
        ProjectionEngine._render_messages() which groups adjacent same-role
        messages before rendering.
        """
        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        effective_tokens = tokens or self.tokens
        char_budget = effective_tokens * 4

        # COLLAPSED: metadata only
        if self.state == NodeState.COLLAPSED:
            return f"[{self.role.upper()}: {len(self.content)} chars]\n"

        # Format content based on message type
        if self.role == "tool_call":
            # Format tool call with arguments
            content = self._format_tool_call()
        elif self.role == "tool_result":
            # Format tool result
            content = self._format_tool_result(char_budget)
        else:
            # Regular message content
            content = self.content

        # SUMMARY: truncate content
        if self.state == NodeState.SUMMARY:
            preview_len = min(200, len(content))
            preview = content[:preview_len]
            if len(content) > preview_len:
                preview += "..."
            return preview + "\n"

        # DETAILS/ALL: full content within budget
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

        trace = Trace(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Message content updated ({old_len} → {len(content)} chars)",
        )
        self._mark_changed(trace)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize MessageNode to dict."""
        data = super().to_dict()
        data.update({
            "role": self.role,
            "content": self.content,
            "actor": self.actor,
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
            role=data.get("role", "user"),
            content=data.get("content", ""),
            actor=data.get("actor"),
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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render work coordination status."""
        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        # COLLAPSED: minimal info
        if self.state == NodeState.COLLAPSED:
            conflict_info = f", {len(self.conflicts)} conflicts" if self.conflicts else ""
            return f"[Work: {self.intent[:40]}... ({len(self.files)} files{conflict_info})]\n"

        parts: list[str] = [f"=== Work: {self.intent} [{self.work_status}] ===\n"]
        parts.append(f"Agent: {self.agent_id}\n")

        # Show files being worked on
        if self.files:
            parts.append("\nFiles:\n")
            for f in self.files:
                mode_indicator = "[W]" if f.get("mode") == "write" else "[R]"
                parts.append(f"  {mode_indicator} {f.get('path', '')}\n")

        # Show dependencies
        if self.dependencies and self.state in (NodeState.DETAILS, NodeState.ALL):
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
        self.intent = intent
        self._mark_changed()
        return self

    def set_files(self, files: list[dict[str, str]]) -> WorkNode:
        """Update files being worked on."""
        self.files = files
        self._mark_changed()
        return self

    def set_conflicts(self, conflicts: list[dict[str, str]]) -> WorkNode:
        """Update detected conflicts."""
        old_count = len(self.conflicts)
        self.conflicts = conflicts
        if len(conflicts) != old_count:
            trace = Trace(
                node_id=self.node_id,
                old_version=self.version,
                new_version=self.version + 1,
                description=f"Work conflicts: {old_count} → {len(conflicts)}",
            )
            self._mark_changed(trace)
        return self

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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render MCP server as tool documentation for the LLM."""
        if self.state == NodeState.HIDDEN:
            return ""

        status_indicator = {
            "connected": "[OK]",
            "connecting": "[...]",
            "disconnected": "[-]",
            "error": "[ERR]",
        }.get(self.status, "[?]")

        # COLLAPSED: just status line
        if self.state == NodeState.COLLAPSED:
            return f"MCP: {self.server_name} {status_indicator} ({len(self.tools)} tools)\n"

        parts: list[str] = [f"## MCP Server: {self.server_name} {status_indicator}\n\n"]

        if self.status == "error" and self.error_message:
            parts.append(f"Error: {self.error_message}\n\n")
            return "".join(parts)

        if self.status != "connected":
            parts.append(f"Status: {self.status}\n")
            return "".join(parts)

        # SUMMARY: tool names only
        if self.state == NodeState.SUMMARY:
            tool_names = [t.get("name", "?") for t in self.tools]
            parts.append(f"Tools: {', '.join(tool_names)}\n")
            if self.resources:
                resource_names = [r.get("name", "?") for r in self.resources]
                parts.append(f"Resources: {', '.join(resource_names)}\n")
            return "".join(parts)

        # DETAILS: tool names + descriptions
        if self.state == NodeState.DETAILS:
            parts.append("### Tools\n\n")
            for tool in self.tools:
                desc = tool.get("description", "No description")
                if len(desc) > 100:
                    desc = desc[:100] + "..."
                parts.append(f"- **{tool.get('name', '?')}**: {desc}\n")

            if self.resources:
                parts.append("\n### Resources\n\n")
                for res in self.resources:
                    parts.append(f"- `{res.get('uri', '?')}`: {res.get('name', 'unnamed')}\n")

            parts.append(f"\nUsage: `{self.server_name}.tool_name(arg1=value1, ...)`\n")
            return "".join(parts)

        # ALL: full documentation with schemas
        parts.append("### Available Tools\n\n")
        for tool in self.tools:
            parts.append(f"#### `{self.server_name}.{tool.get('name', '?')}()`\n\n")
            parts.append(f"{tool.get('description', 'No description')}\n\n")

            schema = tool.get("input_schema", {})
            props = schema.get("properties", {})
            if props:
                parts.append("**Parameters:**\n")
                required = set(schema.get("required", []))
                for param, param_schema in props.items():
                    param_type = param_schema.get("type", "any")
                    param_desc = param_schema.get("description", "")
                    req_marker = " (required)" if param in required else ""
                    parts.append(f"- `{param}` ({param_type}){req_marker}: {param_desc}\n")
                parts.append("\n")

        if self.resources:
            parts.append("### Available Resources\n\n")
            for res in self.resources:
                parts.append(f"- `{res.get('uri', '?')}`")
                if res.get("description"):
                    parts.append(f": {res['description']}")
                parts.append("\n")

        if self.prompts:
            parts.append("\n### Available Prompts\n\n")
            for prompt in self.prompts:
                parts.append(f"- **{prompt.get('name', '?')}**")
                if prompt.get("description"):
                    parts.append(f": {prompt['description']}")
                parts.append("\n")

        return "".join(parts)

    def update_from_connection(self, connection: Any) -> None:
        """Update node state from an MCPConnection object."""
        self.status = connection.status.value
        self.error_message = connection.error_message
        self.tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in connection.tools
        ]
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
        self._mark_changed()

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
            tokens=data.get("tokens", 1000),
            state=NodeState(data.get("state", "details")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
            server_name=data.get("server_name", ""),
            status=data.get("status", "disconnected"),
            error_message=data.get("error_message"),
            tools=data.get("tools", []),
            resources=data.get("resources", []),
            prompts=data.get("prompts", []),
        )
        return node


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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render manager state based on node state."""
        if self.state == NodeState.HIDDEN:
            return ""

        total = len(self.server_states)
        connected = sum(1 for s in self.server_states.values() if s == "connected")

        if self.state == NodeState.COLLAPSED:
            return f"[MCP Manager: {total} servers ({connected} connected)]"

        lines = ["## MCP Manager", ""]

        # SUMMARY: Status list
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

        if self.state in (NodeState.DETAILS, NodeState.ALL) and self.tool_counts:
            # Add tool/resource counts
            lines.append("")
            lines.append("### Capabilities")
            for name in sorted(self.server_states.keys()):
                tools = self.tool_counts.get(name, 0)
                resources = self.resource_counts.get(name, 0)
                lines.append(f"- {name}: {tools} tools, {resources} resources")

        if self.state == NodeState.ALL and self.connection_events:
            # Add recent events
            lines.append("")
            lines.append("### Recent Events")
            for event in self.connection_events[-5:]:
                lines.append(f"- {event.get('time', '?')}: {event.get('message', '?')}")

        return "\n".join(lines)

    def on_child_changed(
        self, child: ContextNode, trace: Trace | None = None
    ) -> None:
        """Handle MCPServerNode changes - track state transitions and generate traces."""
        if not isinstance(child, MCPServerNode):
            return

        name = child.server_name
        old_status = self.server_states.get(name)
        new_status = child.status

        # Track state change
        if old_status != new_status:
            self.server_states[name] = new_status

            # Generate trace for status change
            event_trace = Trace(
                node_id=child.node_id,
                old_version=child.version - 1,
                new_version=child.version,
                description=f"MCP '{name}': {old_status or 'new'} -> {new_status}",
            )
            self.pending_traces.append(event_trace)

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
            tool_trace = Trace(
                node_id=child.node_id,
                old_version=child.version - 1,
                new_version=child.version,
                description=f"MCP '{name}' tools: {old_tools} -> {new_tools}",
            )
            self.pending_traces.append(tool_trace)

        old_resources = self.resource_counts.get(name, 0)
        new_resources = len(child.resources)
        if old_resources != new_resources:
            self.resource_counts[name] = new_resources

        self._mark_changed()
        self.notify_parents(trace)

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
            pending_traces=[],
            tags=d.get("tags", {}),
            server_states=d.get("server_states", {}),
            tool_counts=d.get("tool_counts", {}),
            resource_counts=d.get("resource_counts", {}),
            connection_events=d.get("connection_events", []),
            max_events=d.get("max_events", 10),
        )
        return node


@dataclass
class MarkdownNode(ContextNode):
    """Represents a section of a markdown document with hierarchical structure.

    Each heading in a markdown document creates a MarkdownNode child. The tree
    structure follows the heading hierarchy (# -> ## -> ### etc.).

    Attributes:
        path: File path to the markdown file (root node only)
        heading: The heading text (e.g., "Example Markdown Text")
        level: Heading level (0 = document root, 1 = #, 2 = ##, etc.)
        content: Text content at this level (between this heading and first child)
        child_order: Ordered list of child node IDs (document order)
        summary_tokens: Max tokens for summary (first paragraph truncated)

    Rendering states:
        - HIDDEN: Not rendered
        - COLLAPSED: "# Heading (X tokens)"
        - SUMMARY: Heading + first paragraph (or summary_tokens of content)
        - DETAILS: Full content, children replaced with "### Child (X tokens)"
        - ALL: Full content, children render according to their own states
    """

    path: str = ""  # Only set on root node
    heading: str = ""
    level: int = 0  # 0 = document, 1 = #, 2 = ##, etc.
    content: str = ""  # Content between this heading and first child
    child_order: list[str] = field(default_factory=list)  # Ordered children
    summary_tokens: int = 100  # Configurable summary length

    @property
    def node_type(self) -> str:
        return "markdown"

    def next_sibling(self) -> MarkdownNode | None:
        """Get the next sibling in document order."""
        if not self._graph or not self.parent_ids:
            return None
        # Get first parent (markdown nodes typically have single parent)
        parent_id = next(iter(self.parent_ids))
        parent = self._graph.get_node(parent_id)
        if not parent or not isinstance(parent, MarkdownNode):
            return None
        # Build index map for O(1) lookup instead of O(N) list.index()
        child_index = {cid: i for i, cid in enumerate(parent.child_order)}
        idx = child_index.get(self.node_id)
        if idx is not None and idx < len(parent.child_order) - 1:
            next_id = parent.child_order[idx + 1]
            sibling = self._graph.get_node(next_id)
            return sibling if isinstance(sibling, MarkdownNode) else None
        return None

    def prev_sibling(self) -> MarkdownNode | None:
        """Get the previous sibling in document order."""
        if not self._graph or not self.parent_ids:
            return None
        parent_id = next(iter(self.parent_ids))
        parent = self._graph.get_node(parent_id)
        if not parent or not isinstance(parent, MarkdownNode):
            return None
        # Build index map for O(1) lookup instead of O(N) list.index()
        child_index = {cid: i for i, cid in enumerate(parent.child_order)}
        idx = child_index.get(self.node_id)
        if idx is not None and idx > 0:
            prev_id = parent.child_order[idx - 1]
            sibling = self._graph.get_node(prev_id)
            return sibling if isinstance(sibling, MarkdownNode) else None
        return None

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "heading": self.heading,
            "level": self.level,
            "path": self.path,
            "child_count": len(self.child_order),
            "content_tokens": self._estimate_tokens(self.content),
            "tokens": self.tokens,
            "state": self.state.value,
            "mode": self.mode,
            "version": self.version,
        }

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def _get_summary(self) -> str:
        """Get summary text (first paragraph or truncated content)."""
        # Find first paragraph (text before double newline)
        if "\n\n" in self.content:
            first_para = self.content.split("\n\n")[0].strip()
        else:
            first_para = self.content.strip()

        # Truncate to summary_tokens
        char_budget = self.summary_tokens * 4
        if len(first_para) > char_budget:
            return first_para[:char_budget] + "..."
        return first_para

    def _total_tokens(self) -> int:
        """Calculate total tokens including all descendants."""
        total = self._estimate_tokens(self.content)
        if self._graph:
            for child_id in self.child_order:
                child = self._graph.get_node(child_id)
                if child and isinstance(child, MarkdownNode):
                    total += child._total_tokens()
        return total

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render markdown section based on state."""
        if self.state == NodeState.HIDDEN:
            return ""

        heading_prefix = "#" * self.level if self.level > 0 else ""
        heading_line = f"{heading_prefix} {self.heading}".strip() if self.heading else ""
        content_tokens = self._estimate_tokens(self.content)

        # COLLAPSED: just heading with token count
        if self.state == NodeState.COLLAPSED:
            if heading_line:
                return f"{heading_line} ({content_tokens} tokens)\n"
            return f"[Document: {content_tokens} tokens]\n"

        # SUMMARY: heading + first paragraph
        if self.state == NodeState.SUMMARY:
            parts: list[str] = []
            if heading_line:
                parts.append(heading_line + "\n\n")
            summary = self._get_summary()
            if summary:
                parts.append(summary + "\n")
            return "".join(parts)

        # DETAILS: full content, children replaced with collapsed placeholders
        if self.state == NodeState.DETAILS:
            parts = []
            if heading_line:
                parts.append(heading_line + "\n\n")
            if self.content:
                parts.append(self.content)
                if not self.content.endswith("\n"):
                    parts.append("\n")

            # Render children as collapsed placeholders
            if self._graph and self.child_order:
                parts.append("\n")
                for child_id in self.child_order:
                    child = self._graph.get_node(child_id)
                    if child and isinstance(child, MarkdownNode):
                        child_total = child._total_tokens()
                        child_prefix = "#" * child.level
                        parts.append(f"{child_prefix} {child.heading} ({child_total} tokens)\n")

            return "".join(parts)

        # ALL: full content, children render according to their own states
        effective_tokens = tokens or self.tokens
        parts = []
        if heading_line:
            parts.append(heading_line + "\n\n")
        if self.content:
            parts.append(self.content)
            if not self.content.endswith("\n"):
                parts.append("\n")

        # Render children (each uses its own state)
        if self._graph and self.child_order:
            parts.append("\n")
            child_budget = effective_tokens // max(len(self.child_order), 1)
            for child_id in self.child_order:
                child = self._graph.get_node(child_id)
                if child:
                    parts.append(child.Render(tokens=child_budget, cwd=cwd))

        return "".join(parts)

    @classmethod
    def from_markdown(
        cls,
        path: str,
        content: str | None = None,
        cwd: str = ".",
        summary_tokens: int = 100,
        tokens: int = 2000,
        state: NodeState = NodeState.DETAILS,
    ) -> tuple[MarkdownNode, list[MarkdownNode]]:
        """Parse markdown into a tree of MarkdownNodes.

        Args:
            path: File path
            content: Markdown content (if None, reads from path)
            cwd: Working directory for resolving path
            summary_tokens: Token limit for summaries
            tokens: Token budget for root node
            state: Initial rendering state

        Returns:
            Tuple of (root_node, all_nodes)

        Note:
            If there's <=1 h1 heading AND no preamble content, the document
            root is elided (popped off) and the first heading becomes the root.
            Heading levels are preserved as-is (# = 1, ## = 2, etc.).
        """
        import os
        import re

        if content is None:
            file_path = os.path.join(cwd, path)
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()

        # Parse headings
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        # Find all headings with positions
        headings: list[tuple[int, str, int, int]] = []
        for match in heading_pattern.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            start = match.start()
            headings.append((level, text, start, match.end()))

        # Create root document node
        all_nodes: list[MarkdownNode] = []

        if not headings:
            # No headings - just content
            root = cls(
                path=path,
                heading="",
                level=0,
                content=content.strip(),
                summary_tokens=summary_tokens,
                tokens=tokens,
                state=state,
            )
            return root, [root]

        # Build tree structure
        # Each node: (level, heading, content)
        nodes_data: list[tuple[int, str, str]] = []
        for i, (level, text, start, end) in enumerate(headings):
            # Content ends at next heading or end of file
            if i + 1 < len(headings):
                content_end = headings[i + 1][2]
            else:
                content_end = len(content)

            # Content is everything after heading line until next heading
            section_content = content[end:content_end].strip()
            nodes_data.append((level, text, section_content))

        # Get content before first heading
        preamble = content[: headings[0][2]].strip()

        # Count h1 headings
        h1_count = sum(1 for h in headings if h[0] == 1)

        # Decide whether to elide document root:
        # Pop off empty root if <=1 h1 heading AND no preamble content
        elide_root = h1_count <= 1 and not preamble

        if elide_root:
            # Use first heading as root (keep its original level)
            first_level, first_heading, first_content = nodes_data[0]
            root = cls(
                path=path,
                heading=first_heading,
                level=first_level,  # Keep original level
                content=first_content,
                summary_tokens=summary_tokens,
                tokens=tokens,
                state=state,
            )
            remaining_data = nodes_data[1:]
        else:
            # Keep document root with preamble
            root = cls(
                path=path,
                heading="",
                level=0,
                content=preamble,
                summary_tokens=summary_tokens,
                tokens=tokens,
                state=state,
            )
            remaining_data = nodes_data

        all_nodes.append(root)

        # Build remaining nodes with parent tracking
        # Stack of (node, level)
        stack: list[tuple[MarkdownNode, int]] = [(root, root.level)]

        for level, heading, section_content in remaining_data:
            node = cls(
                heading=heading,
                level=level,  # Keep original level
                content=section_content,
                summary_tokens=summary_tokens,
                state=state,
            )
            all_nodes.append(node)

            # Find parent - pop until we find a node with lower level
            while len(stack) > 1 and stack[-1][1] >= level:
                stack.pop()

            parent = stack[-1][0]
            parent.child_order.append(node.node_id)
            parent.children_ids.add(node.node_id)
            node.parent_ids.add(parent.node_id)

            stack.append((node, level))

        return root, all_nodes

    def to_dict(self) -> dict[str, Any]:
        """Serialize MarkdownNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "path": self.path,
                "heading": self.heading,
                "level": self.level,
                "content": self.content,
                "child_order": self.child_order,
                "summary_tokens": self.summary_tokens,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MarkdownNode:
        """Deserialize MarkdownNode from dict."""
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
            path=data.get("path", ""),
            heading=data.get("heading", ""),
            level=data.get("level", 0),
            content=data.get("content", ""),
            child_order=data.get("child_order", []),
            summary_tokens=data.get("summary_tokens", 100),
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

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render agent information based on state."""
        if self.state == NodeState.HIDDEN:
            return ""

        # Header with relation
        relation_label = self.relation.title()
        header = f"[{relation_label} Agent: {self.agent_id}]\n"

        # COLLAPSED: just header
        if self.state == NodeState.COLLAPSED:
            return header

        # SUMMARY/DETAILS/ALL: include more info
        parts = [header]
        parts.append(f"  Type: {self.agent_type}\n")
        parts.append(f"  State: {self.agent_state}\n")

        if self.task:
            parts.append(f"  Task: {self.task}\n")

        if self.message_count > 0:
            parts.append(f"  Messages: {self.message_count} pending\n")

        return "".join(parts)

    def update_state(self, agent_state: str) -> AgentNode:
        """Update the agent's state."""
        self.agent_state = agent_state
        self._mark_changed()
        return self

    def update_message_count(self, count: int) -> AgentNode:
        """Update pending message count."""
        self.message_count = count
        self._mark_changed()
        return self

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
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", "default"),
            relation=data.get("relation", "self"),
            task=data.get("task", ""),
            agent_state=data.get("agent_state", "running"),
            session_id=data.get("session_id", ""),
            message_count=data.get("message_count", 0),
        )
        return node
