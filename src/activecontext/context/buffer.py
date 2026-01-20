"""Text buffer for shared line storage.

TextBuffer holds file content as lines, allowing multiple TextNode instances
to reference the same underlying content with different line ranges.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TextBuffer:
    """Shared storage for file content.

    Attributes:
        buffer_id: Unique identifier for this buffer
        path: File path (relative to session cwd)
        lines: Actual content as list of lines (without trailing newlines)
        metadata: Optional metadata (encoding, modification time, etc.)
    """

    buffer_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    path: str = ""
    lines: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, cwd: str = ".") -> "TextBuffer":
        """Load a TextBuffer from a file.

        Args:
            path: Path to the file (can be relative or absolute)
            cwd: Working directory for relative paths

        Returns:
            TextBuffer with content loaded from file
        """
        file_path = Path(cwd) / path if not Path(path).is_absolute() else Path(path)

        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()

        return cls(
            path=str(path),
            lines=lines,
            metadata={
                "encoding": "utf-8",
                "line_count": len(lines),
            },
        )

    @classmethod
    def from_content(cls, content: str, path: str = "") -> "TextBuffer":
        """Create a TextBuffer from string content.

        Args:
            content: String content to store
            path: Optional path to associate with this buffer

        Returns:
            TextBuffer with content split into lines
        """
        lines = content.splitlines()
        return cls(
            path=path,
            lines=lines,
            metadata={
                "line_count": len(lines),
            },
        )

    def get_lines(self, start_line: int, end_line: int) -> list[str]:
        """Get a range of lines from the buffer.

        Args:
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive)

        Returns:
            List of lines in the specified range
        """
        # Convert to 0-indexed
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self.lines), end_line)
        return self.lines[start_idx:end_idx]

    def get_content(self, start_line: int = 1, end_line: int | None = None) -> str:
        """Get content as a string for the specified line range.

        Args:
            start_line: Starting line number (1-indexed, inclusive)
            end_line: Ending line number (1-indexed, inclusive), None for end of file

        Returns:
            Content as a single string with newlines
        """
        if end_line is None:
            end_line = len(self.lines)
        lines = self.get_lines(start_line, end_line)
        return "\n".join(lines)

    @property
    def line_count(self) -> int:
        """Total number of lines in the buffer."""
        return len(self.lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "buffer_id": self.buffer_id,
            "path": self.path,
            "lines": self.lines,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextBuffer":
        """Deserialize from dictionary."""
        return cls(
            buffer_id=data.get("buffer_id", str(uuid.uuid4())[:8]),
            path=data.get("path", ""),
            lines=data.get("lines", []),
            metadata=data.get("metadata", {}),
        )
