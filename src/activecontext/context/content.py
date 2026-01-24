"""Shared content data for the split architecture.

ContentData holds the actual content that can be shared across multiple
NodeViews. Each view can have different visibility (hide) and expansion
settings for the same underlying content.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4 + 1


@dataclass
class ContentData:
    """Shared content storage.

    Holds the actual content that can be viewed by multiple agents.
    The summary is shared - if one agent generates a summary, all see it.

    Attributes:
        content_id: Unique identifier
        content_type: Type of content ("file", "artifact", "shell", "markdown", etc.)
        raw_content: The actual content text
        summary: LLM-generated summary (shared across all viewers)
        source_info: Metadata about the source (path, line range, etc.)
        version: Incremented when content changes
        token_count: Cached token estimate for raw_content
        summary_tokens: Cached token estimate for summary
    """

    content_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content_type: str = "unknown"
    raw_content: str = ""
    summary: str | None = None
    source_info: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    token_count: int = 0
    summary_tokens: int = 0

    def __post_init__(self) -> None:
        """Update token counts after initialization."""
        if self.token_count == 0 and self.raw_content:
            self.token_count = _estimate_tokens(self.raw_content)
        if self.summary_tokens == 0 and self.summary:
            self.summary_tokens = _estimate_tokens(self.summary)

    def update_content(self, new_content: str) -> None:
        """Update the raw content and bump version."""
        self.raw_content = new_content
        self.token_count = _estimate_tokens(new_content)
        self.version += 1
        # Invalidate summary when content changes
        self.summary = None
        self.summary_tokens = 0

    def set_summary(self, summary: str) -> None:
        """Set the summary (shared across all viewers)."""
        self.summary = summary
        self.summary_tokens = _estimate_tokens(summary)

    def content_hash(self) -> str:
        """Get a hash of the content for change detection."""
        return hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content_id": self.content_id,
            "content_type": self.content_type,
            "raw_content": self.raw_content,
            "summary": self.summary,
            "source_info": self.source_info,
            "version": self.version,
            "token_count": self.token_count,
            "summary_tokens": self.summary_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentData:
        """Deserialize from dictionary."""
        return cls(
            content_id=data["content_id"],
            content_type=data.get("content_type", "unknown"),
            raw_content=data.get("raw_content", ""),
            summary=data.get("summary"),
            source_info=data.get("source_info", {}),
            version=data.get("version", 0),
            token_count=data.get("token_count", 0),
            summary_tokens=data.get("summary_tokens", 0),
        )


class ContentRegistry:
    """Registry for shared content data.

    Manages ContentData instances that can be referenced by multiple
    NodeViews with different visibility and expansion settings.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._content: dict[str, ContentData] = {}

    def register(self, content: ContentData) -> str:
        """Register content and return its ID.

        Args:
            content: ContentData to register

        Returns:
            The content_id
        """
        self._content[content.content_id] = content
        return content.content_id

    def get(self, content_id: str) -> ContentData | None:
        """Get content by ID.

        Args:
            content_id: Content ID to look up

        Returns:
            ContentData or None if not found
        """
        return self._content.get(content_id)

    def remove(self, content_id: str) -> bool:
        """Remove content from registry.

        Args:
            content_id: Content ID to remove

        Returns:
            True if removed, False if not found
        """
        if content_id in self._content:
            del self._content[content_id]
            return True
        return False

    def list_ids(self) -> list[str]:
        """List all content IDs."""
        return list(self._content.keys())

    def __len__(self) -> int:
        """Return number of registered content items."""
        return len(self._content)

    def __contains__(self, content_id: str) -> bool:
        """Check if content ID is registered."""
        return content_id in self._content


# Factory functions for creating ContentData from different sources


def content_from_file(
    path: str,
    content: str,
    start_line: int = 0,
    end_line: int | None = None,
) -> ContentData:
    """Create ContentData from a file.

    Args:
        path: File path
        content: File content
        start_line: Starting line (0-indexed)
        end_line: Ending line (None = end of file)

    Returns:
        ContentData instance
    """
    return ContentData(
        content_type="file",
        raw_content=content,
        source_info={
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
        },
    )


def content_from_artifact(
    artifact_type: str,
    content: str,
    language: str | None = None,
) -> ContentData:
    """Create ContentData from an artifact.

    Args:
        artifact_type: Type of artifact (code, output, error, etc.)
        content: Artifact content
        language: Programming language if applicable

    Returns:
        ContentData instance
    """
    return ContentData(
        content_type="artifact",
        raw_content=content,
        source_info={
            "artifact_type": artifact_type,
            "language": language,
        },
    )


def content_from_shell(
    command: str,
    output: str,
    exit_code: int | None = None,
) -> ContentData:
    """Create ContentData from shell command output.

    Args:
        command: Command that was executed
        output: Command output
        exit_code: Exit code if completed

    Returns:
        ContentData instance
    """
    return ContentData(
        content_type="shell",
        raw_content=output,
        source_info={
            "command": command,
            "exit_code": exit_code,
        },
    )


def content_from_markdown(
    title: str,
    content: str,
    source_path: str | None = None,
) -> ContentData:
    """Create ContentData from markdown.

    Args:
        title: Document/section title
        content: Markdown content
        source_path: Source file path if applicable

    Returns:
        ContentData instance
    """
    return ContentData(
        content_type="markdown",
        raw_content=content,
        source_info={
            "title": title,
            "source_path": source_path,
        },
    )
