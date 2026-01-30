"""Uniform header rendering for context nodes.

This module provides consistent header formatting for all node types,
making every node uniquely referenceable by the LLM.

Header Format (using markdown heading ID syntax):
- HEADER:  ### name {#id} header (tokens: 18 / 18+74+120 of 340)
- CONTENT: ## name {#id} content (tokens: 92 / 18+74+120 of 340)
- INDEX:   # name {#id} index (tokens: 212 / 18+74+120 of 340)
- ALL:     # name {#id} all (tokens: 340 / 18+74+120 of 340)

Token Format:
- visible / header+content+index of all
- When index=0: visible / header+content of all
- visible is computed from expansion level
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import Expansion

# Overhead tokens for the token counts display itself.
# The "(tokens: NNN / NN+NN+NN of NNN)" string occupies tokens in the header.
# This constant is added to header_tokens to account for that self-referential cost.
TOKEN_COUNTS_OVERHEAD = 12


@dataclass
class TokenInfo:
    """Token counts for different visibility levels.

    Attributes:
        collapsed: Tokens rendered at COLLAPSED/HEADER state (metadata only)
        summary: Additional tokens rendered at CONTENT state
        detail: Additional tokens rendered at DETAILS/ALL state
        total: Total recursive tokens (for groups with children), None if no recursion
        is_bytes: If True, format as bytes instead of tokens
    """

    collapsed: int = 0
    summary: int = 0
    detail: int = 0
    total: int | None = None
    is_bytes: bool = False


def format_token_info(
    header: int,
    content: int,
    index: int,
    all_tokens: int,
    state: Expansion,
) -> str:
    """Format token info string based on current visibility state.

    New format: (tokens: visible / header+content+index of all)
    When index=0, omit: (tokens: visible / header+content of all)

    Args:
        header: Header line tokens
        content: Node's own content tokens
        index: Sum of children's header tokens
        all_tokens: Total recursive tokens
        state: Current rendering state

    Returns:
        Formatted string like "(tokens: 92 / 18+74+120 of 340)"

    Examples:
        HEADER:  (tokens: 18 / 18+74+120 of 340) — only header visible
        CONTENT: (tokens: 92 / 18+74+120 of 340) — header+content visible
        INDEX:   (tokens: 212 / 18+74+120 of 340) — header+content+index visible
        ALL:     (tokens: 340 / 18+74+120 of 340) — everything visible
    """
    from .state import Expansion

    # Compute visible tokens based on expansion
    if state == Expansion.HEADER:
        visible = header
    elif state == Expansion.CONTENT:
        visible = header + content
    elif state == Expansion.INDEX:
        visible = header + content + index
    else:  # ALL
        visible = all_tokens

    # Build breakdown: header+content+index (omit +0 when no index)
    if index > 0:
        breakdown = f"{header}+{content}+{index}"
    else:
        breakdown = f"{header}+{content}"

    return f"(tokens: {visible} / {breakdown} of {all_tokens})"


def render_header(
    display_id: str,
    name: str,
    state: Expansion,
    token_info: TokenInfo,
    notification_level: str | None = None,
    *,
    index_tokens: int = 0,
    all_tokens: int | None = None,
) -> str:
    """Render a uniform header for a context node.

    Args:
        display_id: Short display ID like "text_1" or "message_13"
        name: Human-readable name like "main.py:1-50" or "User #13"
        state: Current rendering state
        token_info: Token breakdown for the node's own content
        notification_level: Optional notification level (ignore/hold/wake)
        index_tokens: Sum of children's header tokens (from node.index_tokens)
        all_tokens: Total recursive tokens (from node.all_tokens), overrides computed

    Returns:
        Formatted header string with appropriate heading level
        Uses markdown heading ID syntax: {#type#N}
        Includes state and notification level as brief descriptor

    Examples:
        HEADER:  "### main.py:1-50 {#text_1} header (tokens: 18 / 18+74+120 of 340)\n"
        CONTENT: "## main.py:1-50 {#text_1} content (tokens: 92 / 18+74+120 of 340)\n"
        INDEX:   "# main.py:1-50 {#text_1} index (tokens: 212 / 18+74+120 of 340)\n"
        ALL:     "# main.py:1-50 {#text_1} all (tokens: 340 / 18+74+120 of 340)\n"
    """
    from .state import Expansion

    # Compute header and content from TokenInfo (node's own breakdown)
    header_toks = token_info.collapsed + TOKEN_COUNTS_OVERHEAD
    content_toks = token_info.summary + token_info.detail
    index_toks = index_tokens

    # all = total recursive; use explicit value or fallback to TokenInfo.total or sum
    if all_tokens is not None:
        all_toks = all_tokens
    elif token_info.total is not None:
        all_toks = token_info.total
    else:
        all_toks = header_toks + content_toks + index_toks

    token_str = format_token_info(header_toks, content_toks, index_toks, all_toks, state)

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
