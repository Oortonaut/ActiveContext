"""Uniform header rendering for context nodes.

This module provides consistent header formatting for all node types,
making every node uniquely referenceable by the LLM.

Header Format:
- COLLAPSED: #### [type.N] name (tokens: visible/hidden)
- SUMMARY:   ### [type.N] name (tokens: collapsed+summary/hidden)
- ALL:       ### [type.N] name (tokens: collapsed+summary+detail)

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
    from .state import NodeState


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


def format_token_info(info: TokenInfo, state: "NodeState") -> str:
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
    from .state import NodeState

    unit = "bytes" if info.is_bytes else "tokens"

    if state == NodeState.HIDDEN:
        return ""

    if state == NodeState.COLLAPSED:
        # Visible: collapsed only
        # Hidden: summary + detail
        visible = info.collapsed
        hidden = info.summary + info.detail
        result = f"{unit}: {visible}/{hidden}"

    elif state == NodeState.SUMMARY:
        # Visible: collapsed + summary
        # Hidden: detail
        if info.summary > 0:
            visible = f"{info.collapsed}+{info.summary}"
        else:
            visible = f"{info.collapsed}+0"
        hidden = info.detail
        result = f"{unit}: {visible}/{hidden}"

    elif state in (NodeState.DETAILS, NodeState.ALL):
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
    state: "NodeState",
    token_info: TokenInfo,
) -> str:
    """Render a uniform header for a context node.

    Args:
        display_id: Short display ID like "view.1" or "message.13"
        name: Human-readable name like "main.py:1-50" or "User #13"
        state: Current rendering state
        token_info: Token breakdown for the node

    Returns:
        Formatted header string with appropriate heading level

    Examples:
        COLLAPSED: "### [view.1] main.py:1-50 (tokens: 18/340)\n"
        SUMMARY:   "## [view.1] main.py:1-50 (tokens: 18+74/340)\n"
        DETAILS:   "# [view.1] main.py:1-50 (tokens: 18+74+340)\n"
        ALL:       "# [view.1] main.py:1-50 (tokens: 18+74+340)\n"
    """
    from .state import NodeState

    if state == NodeState.HIDDEN:
        return ""

    token_str = format_token_info(token_info, state)

    if state == NodeState.COLLAPSED:
        # Level 3 heading for collapsed nodes
        return f"### [{display_id}] {name} {token_str}\n"

    if state == NodeState.SUMMARY:
        # Level 2 heading for summary
        return f"## [{display_id}] {name} {token_str}\n"

    # Level 1 heading for DETAILS, ALL
    return f"# [{display_id}] {name} {token_str}\n"
