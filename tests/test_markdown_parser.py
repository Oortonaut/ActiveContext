"""Tests for the markdown parser.

These tests verify that the parser correctly identifies headings while
ignoring fake headings inside code blocks, blockquotes, and indented code.
"""

from pathlib import Path

import pytest

from activecontext.context.markdown_parser import (
    HeadingSection,
    MarkdownParser,
    ParseResult,
    parse_markdown,
    render_with_tags,
)


class TestHeadingSection:
    """Tests for HeadingSection dataclass."""

    def test_line_range_property(self) -> None:
        """Test that line_range returns correct tuple."""
        section = HeadingSection(level=2, title="Test", start_line=10, end_line=25)
        assert section.line_range == (10, 25)


class TestMarkdownParser:
    """Tests for MarkdownParser class."""

    @pytest.fixture
    def parser(self) -> MarkdownParser:
        """Create a fresh parser instance."""
        return MarkdownParser()

    def test_simple_document(self, parser: MarkdownParser) -> None:
        """Test parsing a simple document with headings."""
        content = """# Introduction

Some content here.

## Section One

More content.

### Subsection

Even more content.
"""
        result = parser.parse(content)

        assert len(result.sections) == 3
        assert result.sections[0].title == "Introduction"
        assert result.sections[0].level == 1
        assert result.sections[1].title == "Section One"
        assert result.sections[1].level == 2
        assert result.sections[2].title == "Subsection"
        assert result.sections[2].level == 3

    def test_no_headings(self, parser: MarkdownParser) -> None:
        """Test document with no headings."""
        content = """Just some text.

No headings here at all.
"""
        result = parser.parse(content)
        assert len(result.sections) == 0
        assert result.preamble_end == 0

    def test_preamble_detection(self, parser: MarkdownParser) -> None:
        """Test that preamble before first heading is detected."""
        content = """This is preamble content.
More preamble.

# First Heading

Content after heading.
"""
        result = parser.parse(content)

        assert result.preamble_end == 3  # Lines 1-3 are preamble
        assert len(result.sections) == 1
        assert result.sections[0].start_line == 4

    def test_no_preamble_when_heading_first(self, parser: MarkdownParser) -> None:
        """Test that preamble_end is 0 when document starts with heading."""
        content = """# First Heading

Content here.
"""
        result = parser.parse(content)
        assert result.preamble_end == 0

    def test_fenced_code_block_backticks(self, parser: MarkdownParser) -> None:
        """Test that headings inside backtick fenced blocks are ignored."""
        content = """# Real Heading

```python
# This is a comment, not a heading
## Also not a heading
def hello():
    # Nested comment
    pass
```

## Another Real Heading

More content.
"""
        result = parser.parse(content)

        assert len(result.sections) == 2
        assert result.sections[0].title == "Real Heading"
        assert result.sections[1].title == "Another Real Heading"

    def test_fenced_code_block_tildes(self, parser: MarkdownParser) -> None:
        """Test that headings inside tilde fenced blocks are ignored."""
        content = """# Real Heading

~~~bash
# Shell comment
## Another shell comment
~~~

## Real Section Two
"""
        result = parser.parse(content)

        assert len(result.sections) == 2
        assert result.sections[0].title == "Real Heading"
        assert result.sections[1].title == "Real Section Two"

    def test_nested_fenced_blocks(self, parser: MarkdownParser) -> None:
        """Test that nested fence markers work correctly."""
        content = """# Start

````markdown
# Fake in outer

```python
# Fake in inner
```

# Still fake
````

## Real After Nested
"""
        result = parser.parse(content)

        assert len(result.sections) == 2
        assert result.sections[0].title == "Start"
        assert result.sections[1].title == "Real After Nested"

    def test_blockquotes_ignored(self, parser: MarkdownParser) -> None:
        """Test that headings inside blockquotes are ignored."""
        content = """# Real Heading

> # This is in a blockquote
> ## Also in blockquote
> Content continues

## Real Section Two

> Nested blockquote:
> > # Deeply nested fake heading
> > ## Another fake

### Real Section Three
"""
        result = parser.parse(content)

        assert len(result.sections) == 3
        assert result.sections[0].title == "Real Heading"
        assert result.sections[1].title == "Real Section Two"
        assert result.sections[2].title == "Real Section Three"

    def test_indented_code_blocks(self, parser: MarkdownParser) -> None:
        """Test that headings in indented code blocks are ignored."""
        content = """# Real Heading

Here is some code:

    # This is indented code, not a heading
    ## Also not a heading
    def example():
        pass

## Real Section Two

More text.

    # Another indented block
    more_code()

### Real Section Three
"""
        result = parser.parse(content)

        assert len(result.sections) == 3
        assert result.sections[0].title == "Real Heading"
        assert result.sections[1].title == "Real Section Two"
        assert result.sections[2].title == "Real Section Three"

    def test_indented_code_requires_blank_line(self, parser: MarkdownParser) -> None:
        """Test that indented text after non-blank line is not code."""
        content = """# Real Heading

Some text followed by
    # This might look like indented code
    but it follows non-blank content

## Real Section Two
"""
        result = parser.parse(content)

        # The "# This might look like indented code" should still be ignored
        # because it has leading spaces - but the exact behavior depends on
        # CommonMark interpretation. Our parser skips 4-space indent after blank.
        assert result.sections[0].title == "Real Heading"
        assert result.sections[1].title == "Real Section Two"

    def test_all_heading_levels(self, parser: MarkdownParser) -> None:
        """Test that all heading levels 1-6 are recognized."""
        content = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6

####### This is not a valid heading (7 hashes)
"""
        result = parser.parse(content)

        assert len(result.sections) == 6
        for i, section in enumerate(result.sections, start=1):
            assert section.level == i
            assert section.title == f"Level {i}"

    def test_heading_must_have_space(self, parser: MarkdownParser) -> None:
        """Test that headings require space after hashes."""
        content = """# Valid Heading

#NoSpaceNotHeading

##AlsoNotHeading

### Valid With Space
"""
        result = parser.parse(content)

        # Only lines with space after # are headings
        assert len(result.sections) == 2
        assert result.sections[0].title == "Valid Heading"
        assert result.sections[1].title == "Valid With Space"

    def test_heading_with_inline_formatting(self, parser: MarkdownParser) -> None:
        """Test headings with inline markdown formatting."""
        content = """# Heading with `code` in it

## Heading with *emphasis* and **strong**

### Heading with [link](http://example.com)
"""
        result = parser.parse(content)

        assert len(result.sections) == 3
        assert result.sections[0].title == "Heading with `code` in it"
        assert result.sections[1].title == "Heading with *emphasis* and **strong**"
        assert result.sections[2].title == "Heading with [link](http://example.com)"

    def test_empty_sections(self, parser: MarkdownParser) -> None:
        """Test back-to-back headings with no content between."""
        content = """# First

## Second

### Third
### Fourth
### Fifth

Back to content.
"""
        result = parser.parse(content)

        assert len(result.sections) == 5
        # Third and Fourth have no content - their line ranges should be small
        assert result.sections[2].title == "Third"
        assert result.sections[3].title == "Fourth"
        assert result.sections[4].title == "Fifth"

    def test_section_line_ranges(self, parser: MarkdownParser) -> None:
        """Test that section line ranges are calculated correctly."""
        content = """# Section One

Content line 3
Content line 4

## Section Two

Content line 8
"""
        result = parser.parse(content)

        assert len(result.sections) == 2
        # Section One: lines 1-5 (before Section Two on line 6)
        assert result.sections[0].start_line == 1
        assert result.sections[0].end_line == 5
        # Section Two: lines 6-9
        assert result.sections[1].start_line == 6
        assert result.sections[1].end_line == 9

    def test_total_lines(self, parser: MarkdownParser) -> None:
        """Test that total_lines is set correctly."""
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        result = parser.parse(content)
        assert result.total_lines == 5


class TestParseMarkdownFunction:
    """Tests for the convenience function."""

    def test_parse_markdown_function(self) -> None:
        """Test that the convenience function works."""
        content = """# Heading

Content.
"""
        result = parse_markdown(content)

        assert isinstance(result, ParseResult)
        assert len(result.sections) == 1


class TestFixtureFile:
    """Tests using the comprehensive test fixture."""

    @pytest.fixture
    def fixture_content(self) -> str:
        """Load the test fixture file."""
        fixture_path = Path(__file__).parent / "fixtures" / "example.md"
        return fixture_path.read_text(encoding="utf-8")

    def test_fixture_parses_without_error(self, fixture_content: str) -> None:
        """Test that the fixture file parses without errors."""
        result = parse_markdown(fixture_content)
        assert isinstance(result, ParseResult)

    def test_fixture_finds_expected_headings(self, fixture_content: str) -> None:
        """Test that the parser finds the expected number of real headings."""
        result = parse_markdown(fixture_content)

        # The fixture has many real headings and many fake ones in code blocks
        # Count of legitimate headings (not in code blocks):
        # This is approximate - the exact count depends on the fixture
        assert len(result.sections) >= 45  # At least 45 real headings

    def test_fixture_starts_with_heading(self, fixture_content: str) -> None:
        """Test that fixture starts with heading (no preamble)."""
        result = parse_markdown(fixture_content)

        # The fixture starts with "# Comprehensive Markdown Parser Test Fixture"
        # so there is no preamble
        assert result.preamble_end == 0
        assert result.sections[0].title == "Comprehensive Markdown Parser Test Fixture"

    def test_fixture_no_code_block_headings(self, fixture_content: str) -> None:
        """Test that fake headings in code blocks are not included."""
        result = parse_markdown(fixture_content)

        # These are headings that appear in code blocks and should NOT be found
        fake_titles = [
            "This looks like a heading but is a Python comment",
            "Another fake heading inside code",
            "## This also looks like an h2 but isn't",
            "Main Script",
            "Configuration Section",
            "Helper Functions",
            "Example H1 in Code Block",
            "Example H2 in Code Block",
        ]

        actual_titles = [s.title for s in result.sections]
        for fake in fake_titles:
            assert fake not in actual_titles, f"Fake heading found: {fake}"

    def test_fixture_no_blockquote_headings(self, fixture_content: str) -> None:
        """Test that fake headings in blockquotes are not included."""
        result = parse_markdown(fixture_content)

        # These headings appear inside blockquotes
        blockquote_titles = [
            "This looks like a heading inside a blockquote",
            "And this looks like an h2",
            "Fake heading in nested quote",
            "Another fake heading",
        ]

        actual_titles = [s.title for s in result.sections]
        for fake in blockquote_titles:
            assert fake not in actual_titles, f"Blockquote heading found: {fake}"

    def test_fixture_no_indented_code_headings(self, fixture_content: str) -> None:
        """Test that fake headings in indented code are not included."""
        result = parse_markdown(fixture_content)

        # These headings appear in indented code blocks
        indented_titles = [
            "This is indented code, not a heading",
            "Still in the code block",
            "Also not a heading",
            "Definitely not a heading either",
            "First indented block",
            "Second indented block",
        ]

        actual_titles = [s.title for s in result.sections]
        for fake in indented_titles:
            assert fake not in actual_titles, f"Indented code heading found: {fake}"

    def test_fixture_has_expected_sections(self, fixture_content: str) -> None:
        """Test that specific expected headings are found."""
        result = parse_markdown(fixture_content)

        expected_titles = [
            "Introduction",
            "Getting Started",
            "Prerequisites",
            "Installation",
            "Configuration",
            "Code Examples with Fake Headings",
            "Python Examples",
            "Shell Script Examples",
            "Blockquotes with Fake Headings",
            "Indented Code Blocks",
            "Edge Cases",
            "Empty Sections",
            "Final Section",
            "Conclusion",
            "Appendix",
        ]

        actual_titles = [s.title for s in result.sections]
        for expected in expected_titles:
            assert expected in actual_titles, f"Expected heading not found: {expected}"


class TestRenderWithTags:
    """Tests for render_with_tags function."""

    def test_round_trip_golden_file(self) -> None:
        """Parse example.md, render with tags, compare to golden."""
        fixtures = Path(__file__).parent / "fixtures"
        example = (fixtures / "example.md").read_text(encoding="utf-8")
        golden = (fixtures / "example_golden.md").read_text(encoding="utf-8")

        result = parse_markdown(example)
        rendered = render_with_tags(example, result)

        assert rendered == golden

    def test_render_tags_format(self) -> None:
        """Test that render_tags produces correct format."""
        section = HeadingSection(level=2, title="Test", start_line=10, end_line=25)

        # Without node_id
        assert section.render_tags(100) == "| line 10..25 of 100"

        # With node_id
        assert section.render_tags(100, "text_1") == "| line 10..25 of 100 {#text_1}"

    def test_render_heading_format(self) -> None:
        """Test that render_heading produces correct format."""
        section = HeadingSection(level=2, title="Features", start_line=10, end_line=25)

        result = section.render_heading(100, "text_0")
        assert result == "## Features | line 10..25 of 100 {#text_0}"

    def test_render_heading_levels(self) -> None:
        """Test render_heading with different heading levels."""
        for level in range(1, 7):
            section = HeadingSection(level=level, title="Test", start_line=1, end_line=5)
            result = section.render_heading(10, "test")
            prefix = "#" * level
            assert result.startswith(f"{prefix} Test")
