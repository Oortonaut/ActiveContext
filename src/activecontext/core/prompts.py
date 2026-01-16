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

### Context Management
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

### Shell Execution
- `shell(command, args=None, cwd=None, env=None, timeout=30)` - Execute a shell command
  - `command`: The command to execute (e.g., "pytest", "git", "npm")
  - `args`: List of arguments (e.g., ["tests/", "-v"])
  - `cwd`: Working directory (default: session cwd)
  - `env`: Additional environment variables (dict)
  - `timeout`: Timeout in seconds (default: 30, None for no limit)
  - Returns: `ShellResult` with `output`, `exit_code`, `success`, `status`

### Agent Control
- `done(message="")` - Signal task completion
  - Call this when you have finished the user's request
  - The message is your final response to the user
  - After calling done(), the agent loop stops

## Code Execution

Use ```python/acrepl blocks for code that should be executed:

```python/acrepl
main = view("src/main.py", tokens=3000)
```

Regular ```python blocks are for showing examples WITHOUT execution.
Only ```python/acrepl blocks run in the REPL.

## Alternative: XML Syntax

You can also use XML-style tags instead of Python syntax:

```xml
<!-- Object creation (name becomes variable) -->
<view name="v" path="src/main.py" tokens="3000"/>
<group name="g" tokens="500">
    <member ref="v"/>
</group>
<topic name="t" title="Feature X" tokens="1000"/>

<!-- Method calls (self refers to variable) -->
<SetLod self="v" level="1"/>
<SetTokens self="v" count="500"/>
<Run self="v" freq="Sync"/>

<!-- Utility functions -->
<ls/>
<show self="v"/>
<done message="Task complete"/>

<!-- Shell execution -->
<shell command="pytest" args="tests/,-v" timeout="60"/>
<shell command="git" args="status,--short"/>

<!-- DAG manipulation -->
<link child="v" parent="g"/>
<unlink child="v" parent="g"/>
```

XML tags are converted to Python before execution. Use whichever syntax you prefer.

## Guidelines

1. Use `view()` to examine files before making suggestions
2. Adjust `lod` to control detail level (higher = more summarized)
3. Use `group()` to organize related views
4. Use ```python/acrepl for executable code, ```python for examples
5. You can mix prose explanations with executable code
6. **Always call `done()` when you have completed the user's request**
   - Include a summary of what you did as the message
   - Example: `done("I've analyzed the file and found 3 issues...")`
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
                    f"(lod={digest.get('lod', 0)}, tokens={digest.get('tokens', 0)})"
                )
            elif obj_type == "group":
                parts.append(
                    f"- `{handle_id}`: group with {digest.get('member_count', 0)} members"
                )
        parts.append("")

    parts.append(prompt)

    return "\n".join(parts)
