"""Markdown node types for structured markdown content.

This module defines:
- MarkdownListItemNode: Individual list items with nesting support
- MarkdownNode: Structured markdown with automatic list parsing
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode

if TYPE_CHECKING:
    pass


@trace_all_fields
@dataclass(kw_only=True)
class MarkdownListItemNode(ContextNode):
    """A single markdown list item that can contain nested items.

    Attributes:
        content: The text content of this list item (without list marker)
        is_ordered: True for numbered lists (1. 2. 3.), False for bullets (- *)
        indent_level: Indentation level (0 = root, 1 = nested once, etc.)
        marker: The original list marker (e.g., "-", "*", "1.", "2.")
    """

    content: str = ""
    is_ordered: bool = False
    indent_level: int = 0
    marker: str = "-"

    def GetDigest(self) -> dict[str, Any]:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return {
            "id": self.node_id,
            "type": self.node_type,
            "content_preview": preview,
            "is_ordered": self.is_ordered,
            "expansion": self.default_expansion.value,
            "children_count": len(self.child_order),
        }

    def render_content(self) -> str:
        """Render list item content (indentation handled by projection)."""
        return f"{self.marker} {self.content}\n"

    def render_digest(self) -> str:
        """Return list item type indicator with char count."""
        return f"{'OL' if self.is_ordered else 'UL'} ({len(self.content)} chars)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize MarkdownListItemNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "content": self.content,
                "is_ordered": self.is_ordered,
                "indent_level": self.indent_level,
                "marker": self.marker,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MarkdownListItemNode:
        """Deserialize MarkdownListItemNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
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
            content=data.get("content", ""),
            is_ordered=data.get("is_ordered", False),
            indent_level=data.get("indent_level", 0),
            marker=data.get("marker", "-"),
        )


@trace_all_fields
@dataclass(kw_only=True)
class MarkdownNode(ContextNode):
    """Structured markdown with automatic list parsing.

    MarkdownNode parses markdown content and creates child MarkdownListItemNode
    instances for each list item. Nested lists become nested child nodes.

    Attributes:
        content: The full markdown content
        buffer_id: Optional reference to a text buffer for live editing
        auto_parse: If True, automatically parse lists on content changes
    """

    content: str = ""
    buffer_id: str | None = None
    auto_parse: bool = True

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "content_length": len(self.content),
            "buffer_id": self.buffer_id,
            "expansion": self.default_expansion.value,
            "children_count": len(self.child_order),
        }

    def _parse_lists(self) -> list[tuple[str, bool, int, str]]:
        """Parse markdown content for list items.

        Returns:
            List of tuples: (content, is_ordered, indent_level, marker)
        """
        import re

        items: list[tuple[str, bool, int, str]] = []
        lines = self.content.split("\n")

        # Regex patterns for list items
        unordered_pattern = re.compile(r"^(\s*)([*+-])\s+(.*)$")
        ordered_pattern = re.compile(r"^(\s*)(\d+\.)\s+(.*)$")

        for line in lines:
            # Try unordered list
            match = unordered_pattern.match(line)
            if match:
                indent = len(match.group(1))
                marker = match.group(2)
                content = match.group(3)
                indent_level = indent // 2  # 2 spaces per indent level
                items.append((content, False, indent_level, marker))
                continue

            # Try ordered list
            match = ordered_pattern.match(line)
            if match:
                indent = len(match.group(1))
                marker = match.group(2)
                content = match.group(3)
                indent_level = indent // 2
                items.append((content, True, indent_level, marker))
                continue

        return items

    def parse_and_create_children(self) -> MarkdownNode:
        """Parse lists and create child nodes for each item.

        This method parses the markdown content, creates MarkdownListItemNode
        children, and establishes parent-child relationships based on indentation.

        Returns:
            Self for chaining
        """
        if not self._graph:
            # Can't create children without graph reference
            return self

        # Parse list items
        items = self._parse_lists()
        if not items:
            return self

        # Remove existing list item children (recursively)
        def remove_list_items(parent_id: str) -> None:
            """Recursively remove all MarkdownListItemNode children."""
            if self._graph is None:
                return
            parent = self._graph.get_node(parent_id)
            if not parent:
                return
            children_to_remove = [
                child_id
                for child_id in parent.child_order
                if isinstance(self._graph.get_node(child_id), MarkdownListItemNode)
            ]
            for child_id in children_to_remove:
                # Recursively remove nested items first
                remove_list_items(child_id)
                # Then unlink from parent
                self._graph.unlink(child_id, parent_id)

        remove_list_items(self.node_id)

        # Create new list item nodes
        stack: list[tuple[int, MarkdownListItemNode]] = []  # (indent_level, node)

        for content, is_ordered, indent_level, marker in items:
            item_node = MarkdownListItemNode(
                content=content,
                is_ordered=is_ordered,
                indent_level=indent_level,
                marker=marker,
                originator=self.node_id,
            )
            self._graph.add_node(item_node)

            # Find parent based on indent level
            parent_node: ContextNode = self

            # Pop stack until we find the right parent level
            while stack and stack[-1][0] >= indent_level:
                stack.pop()

            if stack:
                # Parent is the last item with lower indent
                parent_node = stack[-1][1]

            # Link to parent
            self._graph.link(item_node.node_id, parent_node.node_id)

            # Add to stack for potential children
            stack.append((indent_level, item_node))

        self.mark_changed(f"Parsed {len(items)} list items")
        return self

    def set_content(self, content: str) -> MarkdownNode:
        """Update markdown content and re-parse if auto_parse is enabled.

        Args:
            content: New markdown content

        Returns:
            Self for chaining
        """
        old_content = self.content
        self.content = content

        if self.auto_parse:
            self.parse_and_create_children()

        self.mark_changed(f"Content updated ({len(old_content)} -> {len(content)} chars)")
        return self

    def render_content(self) -> str:
        """Render full markdown content."""
        return self.content

    def render_digest(self) -> str:
        """Return markdown document indicator with size info."""
        prefix = f"MD:{self.buffer_id}" if self.buffer_id else "MARKDOWN"
        return f"{prefix} ({len(self.content)} chars, {len(self.child_order)} items)"

    def to_dict(self) -> dict[str, Any]:
        """Serialize MarkdownNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "content": self.content,
                "buffer_id": self.buffer_id,
                "auto_parse": self.auto_parse,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MarkdownNode:
        """Deserialize MarkdownNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
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
            content=data.get("content", ""),
            buffer_id=data.get("buffer_id"),
            auto_parse=data.get("auto_parse", True),
        )
