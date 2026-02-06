"""GroupNode - Summary facade over child nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode

if TYPE_CHECKING:
    from activecontext.context.headers import TokenInfo


@trace_all_fields
@dataclass(kw_only=True)
class GroupNode(ContextNode):
    """Summary facade over child nodes.

    Attributes:
        child_order: Ordered list of child node IDs (document order)
        summary_prompt: Custom prompt for LLM summarization
        last_child_versions: Version tracking for trace detection
    """

    summary_prompt: str | None = None
    last_child_versions: dict[str, int] = field(default_factory=dict)

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "member_count": len(self.child_order),
            "child_order": self.child_order.to_list(),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Empty — children are rendered by the projection engine."""
        return ""

    def on_child_changed(self, child: ContextNode, description: str = "") -> None:
        """Handle child change: track version and propagate."""
        self.last_child_versions[child.node_id] = child.version

        # Call hook if registered
        if self._on_child_changed_hook:
            self._on_child_changed_hook(self, child, description)

        # Propagate upward
        self.notify_parents(description)

    def render_digest(self) -> str:
        """Return 'Group (N members)' format."""
        return super().render_digest() + f" ({len(self.child_order)} members)"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        # Use child_order for iteration
        ordered_children = self.child_order

        # Collapsed: member count line
        collapsed_text = f"[Group: {len(ordered_children)} members]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        # Detail: children total (recursive)
        child_total = 0
        if self._graph:
            for child_id in ordered_children:
                child = self._graph.get_node(child_id)
                if child:
                    child_info = child.get_token_breakdown()
                    child_total += (
                        child_info.title + child_info.content +
                        child_info.index + child_info.detail
                    )

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            index=self.index_tokens,
            detail=0,
            total=collapsed_tokens + child_total if child_total else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize GroupNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "summary_prompt": self.summary_prompt,
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
            last_child_versions=data.get("last_child_versions", {}),
        )
        return node
