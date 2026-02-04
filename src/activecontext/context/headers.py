"""Uniform header rendering for context nodes.

This module provides consistent header formatting for all node types,
making every node uniquely referenceable by the LLM.

Header Format:
  name [line_range] | {#display_id} state (tokens: ...)

Examples:
  system_prompt:1 | {#text_8} all (tokens: 111 / 21+90 of 111)
  ### Running Commands (lines 77-84) | {#system_prompt_8} all (tokens: 111)

Token Format:
- visible / header+content+index of all
- When index=0: visible / header+content of all
- visible is computed from expansion level
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import Expansion

# Overhead tokens for the token counts display itself.
# The "(tokens: NNN / NN+NN+NN of NNN)" string occupies tokens in the header.
# This constant is added to header_tokens to account for that self-referential cost.
TOKEN_COUNTS_OVERHEAD = 12


@dataclass
class TokenInfo:
    """Token counts for different visibility levels.

    Attributes:
        title: Tokens rendered at COLLAPSED/HEADER state (metadata only)
        content: Additional tokens rendered at CONTENT state
        detail: Additional tokens rendered at DETAILS/ALL state
        total: Total recursive tokens (for groups with children), None if no recursion
        is_bytes: If True, format as bytes instead of tokens
    """

    title: int = 0
    content: int = 0
    index: int = 0 # included in detail rendering
    detail: int = 0
    total: int | None = None
    is_bytes: bool = False

    def format_token_info(
        self,
        expansion: Expansion
    ) -> str:
        """Format token info string based on current visibility state.

        New format: (tokens: visible / header+content+index of all)
        When index=0, omit: (tokens: visible / header+content of all)

        Args:
            header: Header line tokens
            content: Node's own content tokens
            index: Sum of children's header tokens
            all_tokens: Total recursive tokens
            expansion: Current rendering state

        Returns:
            Formatted string like "(tokens: 92 / 18+74+120 of 340)"

        Examples:
            HEADER:  (tokens: 18 / 18+74+120 of 340) — only header visible
            CONTENT: (tokens: 92 / 18+74+120 of 340) — header+content visible
            INDEX:   (tokens: 212 / 18+74+120 of 340) — header+content+index visible
            ALL:     (tokens: 340 / 18+74+120 of 340) — everything visible
        """

        title: int = self.title + TOKEN_COUNTS_OVERHEAD
        breakdown: str = ""
        visible: int = 0

        if self.index == 0:  # Assuming no children
            # Compute visible tokens based on expansion
            if expansion == Expansion.HEADER:
                visible = title
                breakdown = f"{self.title}|{self.content}+{self.index}+{self.detail}"
            else:
                visible = title + self.content
                breakdown = f"{visible}"
        else:
            # Compute visible tokens based on expansion
            if expansion == Expansion.HEADER:
                visible = title
                breakdown = f"{visible}={title}|{self.content}+{self.index}+{self.detail}"
            elif expansion == Expansion.CONTENT:
                visible = title + self.content
                breakdown = f"{visible}={title}+{self.content}|{self.index}+{self.detail}"
            elif expansion == Expansion.INDEX:
                visible = title + self.content + self.index
                breakdown = f"{visible}={title}+{self.content}+{self.index}|{self.detail}"
            else:  # ALL
                # Don't include detail because that's included in its own reporting
                visible = title + self.content + self.index
                breakdown = f"{visible}={title}+{self.content}+{self.index}"

        if self.total is not None and self.total > 0:
            breakdown += f" of {self.total}"

        return breakdown
