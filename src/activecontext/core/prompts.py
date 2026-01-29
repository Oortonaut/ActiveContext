"""Prompt templates and response parsing for code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
