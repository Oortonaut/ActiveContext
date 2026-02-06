"""TraceNode - Change trace as a first-class DAG node."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency

from .base import ContextNode

if TYPE_CHECKING:
    from activecontext.context.headers import TokenInfo


@dataclass(kw_only=True)
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

    # Field-level change tracking
    field_name: str = ""  # Field that changed (e.g., "state", "content")
    prev_value: str = ""  # Previous value (stringified)
    curr_value: str = ""  # Current value (stringified)

    # Trace merging support
    merged_values: list[str] = field(
        default_factory=list
    )  # For same field: list of additional curr values
    trace_target: ContextNode | None = field(
        default=None, repr=False
    )  # Target node for merge lookup
    child_traces: list[TraceNode] = field(default_factory=list)  # Child traces when merged

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "node": self.node,
            "versions": f"v{self.old_version}->v{self.new_version}",
            "description": self.description,
            "originator": self.originator,
            "expansion": self.default_expansion.value,
        }

    # Formatting constants
    MAX_HEADER_LENGTH = 80
    MAX_VALUES_SHOWN = 5

    def render_digest(self) -> str:
        """Return display name with smart formatting.

        Formats:
        - Single field: "state: collapsed -> details"
        - Multiple fields: "state: a -> b, tokens: 100 -> 200"
        - Merged values: "state: collapsed -> { details, all, hidden }"
        - Many fields: "Fields Changed: state, tokens, content..."
        - Many values: "state: a -> { b, c, d, (35 others) }"

        """
        if not self.child_traces:
            # Single change (possibly with merged values for same field)
            if self.merged_values:
                return f"{self.field_name}: {self.prev_value} -> {self._format_merged_values()}"
            return f"{self.field_name}: {self.prev_value} -> {self.curr_value}"

        # Merged from multiple fields
        all_changes = [(self.field_name, self.prev_value, self.curr_value)]
        all_changes += [(t.field_name, t.prev_value, t.curr_value) for t in self.child_traces]

        # Try compact: "field: a -> b, other: c -> d"
        compact = ", ".join(f"{f}: {p} -> {c}" for f, p, c in all_changes)
        if len(compact) <= self.MAX_HEADER_LENGTH:
            return compact

        # Field list: "Fields Changed: state, tokens, content..."
        fields = list(dict.fromkeys(f for f, _, _ in all_changes))  # unique, preserve order
        if len(fields) <= 5:
            return f"Fields Changed: {', '.join(fields)}"
        return f"Fields Changed: {len(fields)}"

    def _format_merged_values(self) -> str:
        """Format merged values as { val1, val2, ... } or { val1, val2, (N others) }."""
        values = [self.curr_value] + self.merged_values
        if len(values) <= self.MAX_VALUES_SHOWN:
            return f"{{ {', '.join(values)} }}"
        shown = values[: self.MAX_VALUES_SHOWN]
        remaining = len(values) - self.MAX_VALUES_SHOWN
        return f"{{ {', '.join(shown)}, ({remaining} others) }}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts for collapsed/summary/detail."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        # Collapsed: just the description (shown in header)
        collapsed_tokens = count_tokens(self.description) + 5

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            index=self.index_tokens,
            detail=0,
        )

    def render_content(self) -> str:
        """Render originator and content."""
        parts: list[str] = []

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
            "child_order": self.child_order.to_list(),
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "display_sequence": self.display_sequence,
            "originator": self.originator,
            "node": self.node,
            "node_display_id": self.node_display_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "description": self.description,
            "content": self.content,
            # Field-level tracking
            "field_name": self.field_name,
            "prev_value": self.prev_value,
            "curr_value": self.curr_value,
            "merged_values": self.merged_values,
            # trace_target is not serialized (runtime reference)
            "child_traces": [t.to_dict() for t in self.child_traces],
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
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "header")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            node=data.get("node", ""),
            node_display_id=data.get("node_display_id", ""),
            old_version=data.get("old_version", 0),
            new_version=data.get("new_version", 0),
            description=data.get("description", ""),
            content=data.get("content"),
            # Field-level tracking
            field_name=data.get("field_name", ""),
            prev_value=data.get("prev_value", ""),
            curr_value=data.get("curr_value", ""),
            merged_values=data.get("merged_values", []),
            # trace_target not deserialized (runtime reference)
            child_traces=[cls._from_dict(t) for t in data.get("child_traces", [])],
        )
