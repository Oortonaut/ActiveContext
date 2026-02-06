"""MessageNode and MessageSegmentNode - Conversation message nodes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode
from .enums import MessageRole

if TYPE_CHECKING:
    from activecontext.context.headers import TokenInfo


@trace_all_fields
@dataclass(kw_only=True)
class MessageNode(ContextNode):
    """Metadata-only envelope for a conversation message.

    MessageNodes serve as hidden envelopes that anchor the message in
    the conversation history.  The actual visible content lives in
    ``MessageSegmentNode`` children created by ``Timeline.ingest_segments()``.

    The ``content`` field is retained for backward compatibility with
    ``_message_history`` and session serialization -- old sessions with
    content-bearing MessageNodes still load correctly.

    Attributes:
        role: Message role ("user", "assistant", "tool_call", "tool_result")
        content: Full response text (kept for history/serialization)
        originator: (inherited) Who produced this message
        tool_name: Tool name for tool_call/tool_result messages
        tool_args: Tool arguments (for tool_call messages)
        content_type: Content type ("text", "image", "audio", etc.)
        mime_type: MIME type (e.g., "image/png", "audio/wav") or None for text
    """

    role: MessageRole = MessageRole.USER
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    content_type: str = "text"  # "text", "image", "audio", etc.
    mime_type: str | None = None  # MIME type (e.g., "image/png")
    processed: bool = False  # Whether this message has been processed by the agent loop
    default_hidden: bool = True  # Hidden by default; segments carry visible content

    @property
    def effective_role(self) -> str:
        """Return the role for LLM alternation (USER or ASSISTANT)."""
        return "USER" if self.originator == "user" else "ASSISTANT"

    @property
    def display_label(self) -> str:
        """Return the human-friendly label for this message.

        Mapping:
        - originator="user" -> configured user name (default "User")
        - originator="agent" -> "Agent"
        - originator="agent:plan" -> "Agent (Plan)"
        - originator="agent:{name}" -> "Child: {name}"
        - originator="tool:{name}" with role=tool_call -> "Tool Call: {name}"
        - originator="tool:{name}" with role=tool_result -> "Tool Result"
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
            if self.role == MessageRole.TOOL_CALL:
                return f"Tool Call: {tool_name}"
            elif self.role == MessageRole.TOOL_RESULT:
                return "Tool Result"
            return f"Tool: {tool_name}"

        return self.originator

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "role": self.role.value,
            "originator": self.originator,
            "effective_role": self.effective_role,
            "content_length": len(self.content),
            "tool_name": self.tool_name,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
            "content_type": self.content_type,
            "mime_type": self.mime_type,
        }

    def _get_formatted_content(self) -> str:
        """Get formatted content based on message type."""
        if self.role == MessageRole.TOOL_CALL:
            return self._format_tool_call()
        elif self.role == MessageRole.TOOL_RESULT:
            return self._format_tool_result()
        return self.content

    def render_content(self) -> str:
        """Render minimal metadata line.

        The full message content is in child ``MessageSegmentNode`` objects.
        This envelope renders only role and originator for debugging.
        """
        label = self.display_label
        return f"[{self.role.value}: {label}]\n"

    def _format_tool_call(self) -> str:
        """Format a tool call message."""
        parts = [f"[Tool: {self.tool_name or 'unknown'}]"]
        if self.tool_args:
            args_str = ", ".join(f'{k}="{v}"' for k, v in self.tool_args.items())
            parts.append(f" {args_str}")
        return "".join(parts)

    def _format_tool_result(self) -> str:
        """Format a tool result message."""
        return f"[Result] {self.content}"

    def set_content(self, content: str) -> MessageNode:
        """Update message content."""
        old_len = len(self.content)
        self.content = content
        self.mark_changed(
            f"Message content updated ({old_len} -> {len(content)} chars)",
        )
        return self

    def render_digest(self) -> str:
        """Return 'Role #N' format using display_sequence."""
        seq = self.display_sequence or 0
        # Use actual role for display (user, assistant, tool_call, tool_result)
        role_display = self.role.value.replace("_", " ").title()
        return f"{role_display} #{seq}"

    def get_token_breakdown(self) -> TokenInfo:
        """Return token counts -- detail is 0 since content lives in segments."""
        from activecontext.context.headers import TokenInfo
        from activecontext.core.tokens import count_tokens

        # Collapsed: role metadata line
        collapsed_text = f"[{self.role.value.upper()}: {self.display_label}]\n"
        collapsed_tokens = count_tokens(collapsed_text)

        return TokenInfo(
            title=collapsed_tokens,
            content=0,
            index=self.index_tokens,
            detail=0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize MessageNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "role": self.role.value,
                "content": self.content,
                "tool_name": self.tool_name,
                "tool_args": self.tool_args,
            }
        )
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
            originator=originator,
            title=data.get("title", ""),
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args", {}),
        )
        return node


@trace_all_fields
@dataclass(kw_only=True)
class MessageSegmentNode(ContextNode):
    """A parsed segment of an LLM response, held as a context node.

    Each segment represents one structural piece of a parsed response:
    prose text, a fenced code block, or a blockquote.  Segments are
    created by ``Timeline.ingest_segments()`` and linked to a parent
    group or the root context.

    Attributes:
        kind: Structural type -- ``"prose"``, ``"fenced"``, or ``"quoted"``.
        content: The parsed text content.
        language: For fenced blocks, the language tag
            (e.g. ``"python/acrepl"``, ``"xml"``, ``"bash"``).
            Empty string for non-fenced segments.
        mime_type: Content format hint (e.g. ``"text/markdown"``).
        source_message_id: Back-reference to the parent ``MessageNode``
            that produced this segment (optional).
    """

    kind: str = "prose"
    content: str = ""
    language: str = ""
    mime_type: str = "text/markdown"
    source_message_id: str | None = None

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "kind": self.kind,
            "language": self.language,
            "content_length": len(self.content),
            "source_message_id": self.source_message_id,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render segment content.

        Fenced blocks are wrapped in triple-backtick markers with the
        language tag.  Prose and quoted segments return content as-is.
        """
        if self.kind == "fenced" and self.language:
            return f"```{self.language}\n{self.content}\n```\n"
        return self.content + "\n"

    def set_content(self, content: str) -> MessageSegmentNode:
        """Update segment content."""
        old_len = len(self.content)
        self.content = content
        self.mark_changed(
            f"Segment content updated ({old_len} -> {len(content)} chars)",
        )
        return self

    def render_digest(self) -> str:
        """Return a compact descriptor like ``prose (N chars)``, ``python (N chars)``, etc."""
        kind_display = self.language if (self.kind == "fenced" and self.language) else self.kind
        return f"{kind_display} ({len(self.content)} chars)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize MessageSegmentNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "kind": self.kind,
                "content": self.content,
                "language": self.language,
                "mime_type": self.mime_type,
                "source_message_id": self.source_message_id,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MessageSegmentNode:
        """Deserialize MessageSegmentNode from dict."""
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
            kind=data.get("kind", "prose"),
            content=data.get("content", ""),
            language=data.get("language", ""),
            mime_type=data.get("mime_type", "text/markdown"),
            source_message_id=data.get("source_message_id"),
        )
        return node
