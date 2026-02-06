"""TopicNode - Conversation topic/thread marker."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
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

    def render_content(self) -> str:
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
        """Return 'Topic: title [status]' format."""
        return f"Topic: {self.title} [{self.status}]"

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
