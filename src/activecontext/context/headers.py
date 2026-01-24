"""Uniform header rendering for context nodes.

This module provides consistent header formatting for all node types,
making every node uniquely referenceable by the LLM.

Header Format (using markdown heading ID syntax):
- COLLAPSED: ### name {#type#N} (tokens: visible/hidden)
- SUMMARY:   ## name {#type#N} (tokens: collapsed+summary/hidden)
- ALL:       # name {#type#N} (tokens: collapsed+summary+detail)

Token Format Rules:
- `/` separates visible from hidden tokens
- `+` combines tokens at same visibility level
- `0` when no summary exists
- `of total` only for groups with recursive children
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Expansion


@dataclass
class TokenInfo:
    """Token counts for different visibility levels.

    Attributes:
        collapsed: Tokens rendered at COLLAPSED state (metadata only)
        summary: Additional tokens rendered at SUMMARY state
        detail: Additional tokens rendered at DETAILS/ALL state
        total: Total recursive tokens (for groups with children), None if no recursion
        is_bytes: If True, format as bytes instead of tokens
    """

    collapsed: int = 0
    summary: int = 0
    detail: int = 0
    total: int | None = None
    is_bytes: bool = False


def format_token_info(info: TokenInfo, state: Expansion) -> str:
    """Format token info string based on current visibility state.

    The slash `/` separates visible tokens from hidden tokens.
    The plus `+` combines tokens at the same visibility level.

    Args:
        info: Token breakdown for the node
        state: Current rendering state

    Returns:
        Formatted string like "(tokens: 18+74/340)" or "(bytes: 1024/4096 of ???)"

    Examples:
        COLLAPSED: (tokens: 18/340) - 18 visible, 340 hidden
        SUMMARY:   (tokens: 18+74/340) - 18+74 visible, 340 hidden
        ALL:       (tokens: 18+74+340) - all visible
    """
    from .state import Expansion

    unit = "bytes" if info.is_bytes else "tokens"

    if state == Expansion.HEADER:
        # Visible: header only
        # Hidden: content + detail
        visible = info.collapsed
        hidden = info.summary + info.detail
        result = f"{unit}: {visible}/{hidden}"

    elif state == Expansion.CONTENT:
        # Visible: header + content
        # Hidden: detail
        visible_str = (
            f"{info.collapsed}+{info.summary}"
            if info.summary > 0
            else f"{info.collapsed}+0"
        )
        hidden = info.detail
        result = f"{unit}: {visible_str}/{hidden}"

    elif state in (Expansion.INDEX, Expansion.ALL):
        # All visible
        if info.summary > 0:
            result = f"{unit}: {info.collapsed}+{info.summary}+{info.detail}"
        else:
            result = f"{unit}: {info.collapsed}+{info.detail}"

    else:
        return ""

    # Add total if present (for groups with recursive children)
    if info.total is not None:
        if info.is_bytes and info.total < 0:
            result += " of ???"
        else:
            result += f" of {info.total}"

    return f"({result})"


def render_header(
    display_id: str,
    name: str,
    state: Expansion,
    token_info: TokenInfo,
    notification_level: str | None = None,
) -> str:
    """Render a uniform header for a context node.

    Args:
        display_id: Short display ID like "text_1" or "message_13"
        name: Human-readable name like "main.py:1-50" or "User #13"
        state: Current rendering state
        token_info: Token breakdown for the node
        notification_level: Optional notification level (ignore/hold/wake)

    Returns:
        Formatted header string with appropriate heading level
        Uses markdown heading ID syntax: {#type#N}
        Includes state and notification level as brief descriptor

    Examples:
        COLLAPSED: "### main.py:1-50 {#text_1} collapsed (tokens: 18/340)\n"
        SUMMARY:   "## main.py:1-50 {#text_1} summary wake (tokens: 18+74/340)\n"
        DETAILS:   "# main.py:1-50 {#text_1} details hold (tokens: 18+74+340)\n"
        ALL:       "# main.py:1-50 {#text_1} all (tokens: 18+74+340)\n"
    """
    from .state import Expansion

    token_str = format_token_info(token_info, state)

    # Build brief: "summary wake" or just "summary" if notification is ignore/None
    brief = state.value
    if notification_level and notification_level != "ignore":
        brief = f"{brief} {notification_level}"

    if state == Expansion.HEADER:
        # Level 3 heading for header-only nodes
        return f"### {name} {{#{display_id}}} {brief} {token_str}\n"

    if state == Expansion.CONTENT:
        # Level 2 heading for content
        return f"## {name} {{#{display_id}}} {brief} {token_str}\n"

    # Level 1 heading for INDEX, ALL
    return f"# {name} {{#{display_id}}} {brief} {token_str}\n"
