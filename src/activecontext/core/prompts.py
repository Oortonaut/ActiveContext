"""Prompt templates and response parsing for code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from activecontext.prompts import FULL_SYSTEM_PROMPT, SYSTEM_PROMPT

if TYPE_CHECKING:
    from activecontext.session.protocols import Projection

# Re-export for backward compatibility
__all__ = ["SYSTEM_PROMPT", "FULL_SYSTEM_PROMPT", "ParsedResponse", "parse_response", "build_user_message"]


@dataclass
class ParsedResponse:
    """Parsed LLM response with prose and code blocks."""

    segments: list[tuple[str, str]]  # List of (type, content) - type is "prose" or "code"

    @property
    def prose_only(self) -> str:
        """Get just the prose content."""
        return "\n".join(content for typ, content in self.segments if typ == "prose")

    @property
    def code_blocks(self) -> list[str]:
        """Get just the code blocks."""
        return [content for typ, content in self.segments if typ == "code"]

    @property
    def has_code(self) -> bool:
        """Check if response contains any code."""
        return any(typ == "code" for typ, _ in self.segments)


def parse_response(text: str) -> ParsedResponse:
    """Parse an LLM response into prose and code segments.

    Only executes code blocks tagged with ```python/acrepl - this is explicit
    so the LLM can show code examples without executing them.

    Args:
        text: Raw LLM response text

    Returns:
        ParsedResponse with interleaved prose and code segments
    """
    segments: list[tuple[str, str]] = []

    # Only match ```python/acrepl blocks - explicit execution marker
    pattern = r"```python/acrepl\s*\n(.*?)```"

    last_end = 0
    for match in re.finditer(pattern, text, re.DOTALL):
        # Add prose before this code block
        prose = text[last_end : match.start()].strip()
        if prose:
            segments.append(("prose", prose))

        # Add the code block
        code = match.group(1).strip()
        if code:
            segments.append(("code", code))

        last_end = match.end()

    # Add any remaining prose after the last code block
    remaining = text[last_end:].strip()
    if remaining:
        segments.append(("prose", remaining))

    # If no segments were found, treat entire text as prose
    if not segments:
        segments.append(("prose", text.strip()))

    return ParsedResponse(segments=segments)


def build_user_message(prompt: str, projection: Projection | None = None) -> str:
    """Build the user message with optional context projection.

    Args:
        prompt: User's prompt text
        projection: Optional projection of current context state

    Returns:
        Formatted user message
    """
    parts = []

    if projection and projection.handles:
        parts.append("## Current Context\n")
        for handle_id, digest in projection.handles.items():
            obj_type = digest.get("type", "unknown")
            if obj_type == "view":
                parts.append(
                    f"- `{handle_id}`: view of {digest.get('path', '?')} "
                    f"(state={digest.get('state', 'details')}, tokens={digest.get('tokens', 0)})"
                )
            elif obj_type == "group":
                parts.append(
                    f"- `{handle_id}`: group with {digest.get('member_count', 0)} members"
                )
        parts.append("")

    parts.append(prompt)

    return "\n".join(parts)
