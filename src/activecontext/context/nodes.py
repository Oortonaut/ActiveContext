"""Context node types for the context DAG.

This module defines the typed node hierarchy:
- ContextNode: Base class with common fields and notification
- ViewNode: File content view
- GroupNode: Summary facade over children
- TopicNode: Conversation segment
- ArtifactNode: Code/output artifact
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.state import NodeState, TickFrequency

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph


@dataclass
class Diff:
    """Represents a change between versions."""

    node_id: str
    old_version: int
    new_version: int
    description: str
    content: str | None = None  # Optional diff content


# Type alias for hooks
OnChildChangedHook = Callable[["ContextNode", "ContextNode", Diff | None], None]


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
        version: Incremented on change for diff detection
        created_at: Unix timestamp of creation
        updated_at: Unix timestamp of last update
        pending_diffs: Accumulated diffs until commit
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

    # Diff accumulation
    pending_diffs: list[Diff] = field(default_factory=list)

    # Metadata
    tags: dict[str, Any] = field(default_factory=dict)

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

    def _mark_changed(self, diff: Diff | None = None) -> None:
        """Mark this node as changed and notify parents."""
        self.version += 1
        self.updated_at = time.time()
        if diff:
            self.pending_diffs.append(diff)
        self.notify_parents(diff)

    def notify_parents(self, diff: Diff | None = None) -> None:
        """Notify all parent nodes of a change."""
        if not self._graph:
            return

        for parent_id in self.parent_ids:
            parent = self._graph.get_node(parent_id)
            if parent:
                parent.on_child_changed(self, diff)

    def on_child_changed(self, child: ContextNode, diff: Diff | None = None) -> None:
        """Handle notification that a child has changed.

        Default implementation propagates upward. GroupNode overrides
        to invalidate summary and generate diffs.
        """
        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, diff)

        # Propagate upward
        self.notify_parents(diff)

    def set_on_child_changed_hook(self, hook: OnChildChangedHook | None) -> None:
        """Register a hook for child change notifications."""
        self._on_child_changed_hook = hook

    def clear_pending_diffs(self) -> None:
        """Clear accumulated diffs (called after projection render)."""
        self.pending_diffs.clear()

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
    """

    path: str = ""
    pos: str = "1:0"
    end_pos: str | None = None

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
        }

    def Render(self, tokens: int | None = None, cwd: str = ".") -> str:
        """Render file content based on rendering state."""
        import os

        # HIDDEN: don't render anything
        if self.state == NodeState.HIDDEN:
            return ""

        effective_tokens = tokens or self.tokens
        char_budget = effective_tokens * 4

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
            diff_count = len(self.pending_diffs)
            return f"[{self.path}: {line_count} lines, {diff_count} pending changes]\n"

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

        # At ALL state, append pending diffs
        if self.state == NodeState.ALL and self.pending_diffs:
            diff_section = "\n--- Pending Changes ---\n"
            for diff in self.pending_diffs:
                diff_section += f"[v{diff.old_version}→v{diff.new_version}] {diff.description}\n"
                if diff.content:
                    diff_section += diff.content + "\n"
            content += diff_section

        return content

    def SetPos(self, pos: str) -> ViewNode:
        """Set start position."""
        self.pos = pos
        return self

    def SetEndPos(self, end_pos: str | None) -> ViewNode:
        """Set end position."""
        self.end_pos = end_pos
        return self


@dataclass
class GroupNode(ContextNode):
    """Summary facade over child nodes.

    Attributes:
        summary_prompt: Custom prompt for LLM summarization
        cached_summary: Cached LLM-generated summary
        summary_stale: Whether summary needs regeneration
        last_child_versions: Version tracking for diff detection
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
            diff_count = len(self.pending_diffs)
            return f"[Group: {member_count} members, {diff_count} changes]\n"

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
        parts: list[str] = [f"=== Group ({len(self.children_ids)} members) ===\n"]

        if self._graph:
            per_child_tokens = effective_tokens // len(self.children_ids)
            for child_id in self.children_ids:
                child = self._graph.get_node(child_id)
                if child:
                    child_content = child.Render(tokens=per_child_tokens, cwd=cwd)
                    parts.append(child_content)
                    parts.append("\n")

        # At ALL state, include pending diffs from children
        if self.state == NodeState.ALL and self.pending_diffs:
            parts.append("--- Group Changes ---\n")
            for diff in self.pending_diffs:
                parts.append(f"[{diff.node_id}] {diff.description}\n")

        return "".join(parts)

    def on_child_changed(self, child: ContextNode, diff: Diff | None = None) -> None:
        """Handle child change: track version, generate diff, propagate."""
        old_version = self.last_child_versions.get(child.node_id, 0)
        new_version = child.version

        if new_version != old_version:
            # Generate diff for this change
            group_diff = Diff(
                node_id=child.node_id,
                old_version=old_version,
                new_version=new_version,
                description=f"{child.node_type} '{child.node_id}' changed",
                content=diff.content if diff else None,
            )
            self.pending_diffs.append(group_diff)
            self.summary_stale = True
            self.last_child_versions[child.node_id] = new_version

        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, diff)

        # Propagate upward
        self.notify_parents(diff)

    def invalidate_summary(self) -> None:
        """Mark summary as needing regeneration."""
        self.summary_stale = True


@dataclass
class TopicNode(ContextNode):
    """Represents a conversation topic/thread.

    Attributes:
        title: Short title for the topic
        message_indices: Indices into session._conversation
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

        diff = Diff(
            node_id=self.node_id,
            old_version=self.version,
            new_version=self.version + 1,
            description=f"Content updated ({len(old_content)} → {len(content)} chars)",
        )
        self._mark_changed(diff)
        return self
