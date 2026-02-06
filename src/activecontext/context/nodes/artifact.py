"""ArtifactNode - Code snippets, outputs, errors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
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

    def render_content(self) -> str:
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
        """Return 'TYPE:language (N chars)' format."""
        lang_suffix = f":{self.language}" if self.language else ""
        return f"{self.artifact_type.upper()}{lang_suffix} ({len(self.content)} chars)"

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
