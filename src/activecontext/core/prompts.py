"""Response parsing — structural block splitting for LLM output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# DSL tags recognized as XML commands in LLM responses.
# Keep in sync with session/xml_parser.py CONSTRUCTOR_TAGS and UTILITY_TAGS.
_XML_DSL_TAGS = frozenset(
    {
        # Constructor tags
        "view",
        "text",
        "group",
        "topic",
        "artifact",
        # Utility tags
        "ls",
        "show",
        "done",
        "link",
        "unlink",
        "shell",
        "wait",
        "wait_all",
        "wait_any",
    }
)

_XML_TAG_PATTERN = re.compile(
    r"<(" + "|".join(sorted(_XML_DSL_TAGS)) + r")[\s/>]"
)

# Fenced code block pattern: ```<language> ... ```
# MULTILINE so ^ matches line starts; DOTALL so . matches newlines.
_FENCED_BLOCK_PATTERN = re.compile(
    r"^```(\S*)[ \t]*\n(.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class Segment:
    """A parsed segment of an LLM response.

    Attributes:
        kind: Structural type — "prose", "fenced", "quoted", or "xml".
        content: The text content of this segment.
        mime_type: Content format hint (e.g. "text/markdown", "text/x-python").
        language: For fenced blocks, the language tag (e.g. "python/acrepl",
            "python", "bash"). Empty string for non-fenced segments.
    """

    kind: str
    content: str
    mime_type: str
    language: str = ""


@dataclass
class ParsedResponse:
    """Parsed LLM response with typed segments.

    This is a structural parse only — it splits the response into blocks
    without making execution decisions.  Consumers decide which segments
    are executable.
    """

    segments: list[Segment] = field(default_factory=list)

    @property
    def prose_only(self) -> str:
        """Get just the prose content (kind == "prose")."""
        return "\n".join(s.content for s in self.segments if s.kind == "prose")


def _lang_to_mime(language: str) -> str:
    """Map a fenced-block language tag to a mime type."""
    lang = language.lower()
    if lang.startswith("python"):
        return "text/x-python"
    if lang in ("json", "jsonc"):
        return "application/json"
    if lang in ("bash", "sh", "zsh", "shell"):
        return "text/x-shellscript"
    if lang in ("yaml", "yml"):
        return "text/yaml"
    if lang in ("xml",):
        return "application/xml"
    return "text/plain"


def _is_xml_command_line(line: str) -> bool:
    """Check if a line is a standalone DSL XML command.

    Matches self-closing tags like ``<view name="v" path="main.py"/>``
    and single-line open/close pairs like ``<done message="ok"/>``.
    Only matches known DSL tag names to avoid false positives with HTML.
    """
    stripped = line.strip()
    if not stripped or not _XML_TAG_PATTERN.match(stripped):
        return False
    # Must be self-closing (/>) or have a closing tag on the same line
    return stripped.endswith("/>") or bool(re.search(r"</\w+>\s*$", stripped))


def _split_prose(text: str) -> list[Segment]:
    """Split a prose block into sub-segments: prose, quoted, xml commands.

    Recognizes:
    - Markdown blockquotes (consecutive lines starting with ``>``)
    - DSL XML commands (``<view .../>``, ``<shell .../>`` etc.)
    """
    if not text.strip():
        return []

    segments: list[Segment] = []
    current_lines: list[str] = []
    current_kind: str | None = None

    def _flush(kind: str | None) -> None:
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                segments.append(
                    Segment(
                        kind=kind or "prose",
                        content=content,
                        mime_type="text/markdown",
                    )
                )

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith(">"):
            # Blockquote line
            if current_kind != "quoted":
                _flush(current_kind)
                current_lines = []
                current_kind = "quoted"
            current_lines.append(line)
        elif _is_xml_command_line(stripped):
            # XML command — flush accumulated lines and emit as code
            _flush(current_kind)
            current_lines = []
            current_kind = None
            segments.append(
                Segment(kind="xml", content=stripped, mime_type="application/xml")
            )
        else:
            # Regular prose line
            if current_kind == "quoted":
                _flush("quoted")
                current_lines = []
                current_kind = None
            if current_kind is None:
                current_kind = "prose"
            current_lines.append(line)

    _flush(current_kind)
    return segments


def parse_response(text: str) -> ParsedResponse:
    """Parse an LLM response into structural segments.

    This is a block splitter — it identifies the structure of the response
    without making execution decisions.  Segment kinds:

    - ``"fenced"`` — fenced code blocks (any language). The ``language``
      field contains the tag (e.g. ``"python/acrepl"``, ``"bash"``).
    - ``"xml"`` — DSL XML commands (``<view/>``, ``<shell/>``, etc.)
    - ``"quoted"`` — markdown blockquotes (lines starting with ``>``)
    - ``"prose"`` — everything else

    Consumers decide which segments are executable.  For ActiveContext,
    the convention is: ``language == "python/acrepl"`` fenced blocks and
    ``kind == "xml"`` segments.

    Args:
        text: Raw LLM response text

    Returns:
        ParsedResponse with typed, interleaved segments
    """
    segments: list[Segment] = []

    last_end = 0
    for match in _FENCED_BLOCK_PATTERN.finditer(text):
        # Process prose before this fenced block
        prose_text = text[last_end : match.start()]
        segments.extend(_split_prose(prose_text))

        # Add the fenced block
        language = match.group(1)
        content = match.group(2).strip()
        if content:
            segments.append(
                Segment(
                    kind="fenced",
                    content=content,
                    mime_type=_lang_to_mime(language),
                    language=language,
                )
            )

        last_end = match.end()

    # Process any remaining text after the last fenced block
    remaining = text[last_end:]
    segments.extend(_split_prose(remaining))

    # If no segments found, treat entire text as prose
    if not segments:
        segments.append(
            Segment(kind="prose", content=text.strip(), mime_type="text/markdown")
        )

    return ParsedResponse(segments=segments)
