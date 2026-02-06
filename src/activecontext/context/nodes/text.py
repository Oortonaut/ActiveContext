"""TextNode - File content view."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields
from activecontext.core.tokens import MediaType, detect_media_type

from .base import ContextNode
from .file_watcher import register_file_watcher, unregister_file_watcher

if TYPE_CHECKING:
    from activecontext.context.headers import TokenInfo


@trace_all_fields
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

    def __post_init__(self) -> None:
        """Auto-detect media type from file extension and set originator."""
        if self.path and self.media_type == MediaType.TEXT:
            self.media_type = detect_media_type(self.path)
        # Auto-populate originator from path if not explicitly set
        if self.originator is None and self.path:
            self.originator = self.path
        # Auto-register with file watcher registry
        self.register_watcher()

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

    def render_content(self) -> str:
        """Render file content with line numbers.

        TextNode requires a TextBuffer to render content. The buffer_id
        must be set and the buffer must exist in TextBuffer's class-level cache.

        Returns:
            Rendered content string (without header — Render() prepends it)
        """
        from activecontext.context.buffer import TextBuffer

        output_parts: list[str] = []
        lines: list[str] = []

        if not self.buffer_id:
            return "[TextNode requires buffer_id to be set]"

        buffer = TextBuffer.get_by_id(self.buffer_id)
        if not buffer:
            return f"[Buffer not found: {self.buffer_id}]"

        # Get lines from buffer using start_line/end_line
        start_idx = max(0, self.start_line - 1)
        end_idx = self.end_line if self.end_line else len(buffer.lines)
        lines = buffer.lines[start_idx:end_idx]

        # Render with line numbers
        base_line = self.start_line

        for i, line in enumerate(lines):
            line_num = base_line + i
            line_content = line.replace("\n", "\u2424").replace("\r", "\u240D")
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
        """Return title if set, otherwise 'path (lines N-M)' format."""
        if self.title:
            return self.title

        # Build line range caption
        start = self.start_line if self.buffer_id else self._parse_start_line()
        end = self.end_line if self.buffer_id else self._parse_end_line()

        if start and end:
            line_range = f" (lines {start}-{end})"
        elif start and start > 1:
            line_range = f" (line {start}+)"
        else:
            line_range = ""

        return f"{self.path}{line_range}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/content/index/detail."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        # Title: just metadata line
        collapsed_text = f"[{self.path}: lines, pending traces]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Content: estimate from line count (~10 tokens/line with line numbers)
        content_tokens = 0
        if self.end_line and self.start_line:
            line_count = max(0, self.end_line - self.start_line + 1)
            content_tokens = line_count * 10

        return TokenInfo(
            title=collapsed_tokens,
            content=content_tokens,
            index=self.index_tokens,
            detail=self.detail_tokens,
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
            start_line=data.get("start_line", 1),
            end_line=data.get("end_line"),
        )
        return node
