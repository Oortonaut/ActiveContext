"""Markdown parser that extracts heading sections while ignoring code blocks.

This parser identifies markdown headings and their content ranges while properly
skipping fake headings inside:
- Fenced code blocks (``` or ~~~)
- Blockquotes (lines starting with >)
- Indented code blocks (4+ spaces at line start)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .buffer import TextBuffer


@dataclass
class HeadingSection:
    """A heading section in a markdown document.

    Attributes:
        level: Heading level (1-6 for #-######)
        title: The heading text
        start_line: Line number where the heading starts (1-indexed)
        end_line: Line number where the section ends (1-indexed, inclusive)
    """

    level: int
    title: str
    start_line: int
    end_line: int

    @property
    def line_range(self) -> tuple[int, int]:
        """Return (start_line, end_line) tuple."""
        return (self.start_line, self.end_line)


@dataclass
class ParseResult:
    """Result of parsing a markdown document.

    Attributes:
        sections: List of heading sections in document order
        preamble_end: Line number where preamble ends (0 if no preamble)
        total_lines: Total number of lines in the document
    """

    sections: list[HeadingSection] = field(default_factory=list)
    preamble_end: int = 0
    total_lines: int = 0


class MarkdownParser:
    """Parser that extracts heading sections from markdown content.

    The parser correctly handles:
    - Fenced code blocks (``` or ~~~) - headings inside are ignored
    - Blockquotes (> prefix) - headings inside are ignored
    - Indented code blocks (4+ spaces) - headings inside are ignored
    - ATX headings only (# style), not Setext (underline style)
    """

    # Match ATX-style headings: 1-6 # followed by space and text
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

    # Match fenced code block delimiter (``` or ~~~)
    FENCE_PATTERN = re.compile(r"^(`{3,}|~{3,})")

    # Match blockquote (starts with >)
    BLOCKQUOTE_PATTERN = re.compile(r"^\s*>")

    # Match indented code (4+ spaces or tab at start)
    INDENT_PATTERN = re.compile(r"^(    |\t)")

    def __init__(self) -> None:
        self._in_fenced_block = False
        self._fence_marker: str | None = None
        self._fence_length: int = 0  # Length of opening fence
        self._prev_was_blank = True  # For indented code detection

    def parse(self, content: str) -> ParseResult:
        """Parse markdown content and extract heading sections.

        Args:
            content: The markdown text to parse

        Returns:
            ParseResult with heading sections and metadata
        """
        lines = content.split("\n")
        result = ParseResult(total_lines=len(lines))

        # Reset state
        self._in_fenced_block = False
        self._fence_marker = None
        self._fence_length = 0
        self._prev_was_blank = True

        # Find all headings with their line numbers
        headings: list[tuple[int, int, str]] = []  # (line_num, level, title)

        for line_num, line in enumerate(lines, start=1):
            if self._is_heading(line):
                match = self.HEADING_PATTERN.match(line)
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    headings.append((line_num, level, title))

            # Track blank lines for indented code detection
            self._prev_was_blank = not line.strip()

        if not headings:
            # No headings - entire document is preamble
            return result

        # Set preamble end (content before first heading)
        first_heading_line = headings[0][0]
        if first_heading_line > 1:
            # Check if there's actual content before first heading
            preamble_content = "\n".join(lines[: first_heading_line - 1]).strip()
            if preamble_content:
                result.preamble_end = first_heading_line - 1

        # Create sections from headings
        for i, (line_num, level, title) in enumerate(headings):
            # Section ends at the line before next heading, or end of document
            if i + 1 < len(headings):
                end_line = headings[i + 1][0] - 1
            else:
                end_line = len(lines)

            section = HeadingSection(
                level=level,
                title=title,
                start_line=line_num,
                end_line=end_line,
            )
            result.sections.append(section)

        return result

    def parse_buffer(self, buffer: "TextBuffer") -> ParseResult:
        """Parse a TextBuffer and extract heading sections.

        Args:
            buffer: TextBuffer containing markdown lines

        Returns:
            ParseResult with heading sections and metadata
        """
        content = "\n".join(buffer.lines)
        return self.parse(content)

    def _is_heading(self, line: str) -> bool:
        """Check if a line is a valid markdown heading.

        Returns False for headings inside code blocks or blockquotes.
        """
        # Update fenced block state
        fence_match = self.FENCE_PATTERN.match(line)
        if fence_match:
            fence_str = fence_match.group(1)
            marker = fence_str[0]  # ` or ~
            fence_len = len(fence_str)

            if self._in_fenced_block:
                # Check if this closes the block
                # Must be same marker type AND at least as long
                if marker == self._fence_marker and fence_len >= self._fence_length:
                    self._in_fenced_block = False
                    self._fence_marker = None
                    self._fence_length = 0
            else:
                # Opening a new fenced block
                self._in_fenced_block = True
                self._fence_marker = marker
                self._fence_length = fence_len
            return False  # Fence line itself is not a heading

        # Skip if inside fenced code block
        if self._in_fenced_block:
            return False

        # Skip blockquotes
        if self.BLOCKQUOTE_PATTERN.match(line):
            return False

        # Skip indented code blocks
        # A line is indented code if it has 4+ leading spaces AND
        # the previous line was blank (or start of document)
        if self.INDENT_PATTERN.match(line) and self._prev_was_blank:
            return False

        # Check if it's a valid heading
        return bool(self.HEADING_PATTERN.match(line))


def parse_markdown(content: str) -> ParseResult:
    """Convenience function to parse markdown content.

    Args:
        content: The markdown text to parse

    Returns:
        ParseResult with heading sections and metadata
    """
    parser = MarkdownParser()
    return parser.parse(content)
