"""Prompt templates and response parsing for code generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activecontext.session.protocols import Projection

SYSTEM_PROMPT = """\
You are an AI assistant that helps users work with code through a Python-based context system.

## Available Functions

You have access to these functions in your Python environment:

- `view(path, pos="0:0", tokens=2000, lod=0, mode="paused")` - Create a view of a file
  - `path`: File path to view
  - `pos`: Position as "line:col" or just "line"
  - `tokens`: Token budget for content
  - `lod`: Level of detail (0=raw, 1=structured, 2=summary, 3=diff)
  - `mode`: "paused" or "running" (running updates each turn)

- `group(*members, tokens=500, lod=1, mode="paused")` - Create a summary group
  - Groups multiple views into a single summarized context

- View methods: `.SetPos(pos)`, `.SetTokens(n)`, `.SetLod(k)`, `.Scroll(delta)`
  Also: `.Run()`, `.Pause()`, `.Refresh()`
- Group methods: `.SetTokens(n)`, `.SetLod(k)`, `.Run()`, `.Pause()`
- `ls()` - List all context handles
- `show(obj)` - Display a handle's content

## Response Format

You can respond with a mix of explanation and Python code blocks.
Code blocks will be executed in sequence.

Example:
```
I'll create a view of the main file to understand its structure.

```python
main = view("src/main.py", tokens=3000)
```

Now let me check the imports.

```python
main.SetPos("1:0").SetTokens(500)
```
```

## Guidelines

1. Use `view()` to examine files before making suggestions
2. Adjust `lod` to control detail level (higher = more summarized)
3. Use `group()` to organize related views
4. Code in ```python blocks will be executed automatically
5. You can mix prose explanations with code
"""


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

    Handles markdown code blocks with ```python or ``` fencing.

    Args:
        text: Raw LLM response text

    Returns:
        ParsedResponse with interleaved prose and code segments
    """
    segments: list[tuple[str, str]] = []

    # Pattern matches ```python or ``` code blocks
    # Using non-greedy match for content
    pattern = r"```(?:python|py)?\s*\n(.*?)```"

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
                    f"(lod={digest.get('lod', 0)}, tokens={digest.get('tokens', 0)})"
                )
            elif obj_type == "group":
                parts.append(
                    f"- `{handle_id}`: group with {digest.get('member_count', 0)} members"
                )
        parts.append("")

    parts.append(prompt)

    return "\n".join(parts)
