# Comprehensive Markdown Parser Test Fixture

This preamble appears before any heading in the document. It should be captured
as content belonging to no section, or associated with the first heading depending
on parser implementation. This tests that parsers handle pre-heading content correctly.

Some additional preamble text with **bold** and *italic* formatting to ensure
the parser doesn't get confused by inline markup.

# Introduction

This is the introduction section. It contains normal paragraph text that should
be associated with the "Introduction" heading.

## Getting Started

This subsection explains how to get started with the project.

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- Node.js 18+
- Git

### Installation

Run the following command to install:

```bash
pip install example-package
```

## Configuration

Configuration is handled through YAML files.

### Basic Configuration

Here is a minimal configuration example:

```yaml
# This is a YAML comment, not a heading
server:
  host: localhost
  port: 8080
```

# Code Examples with Fake Headings

This section contains code blocks that include lines starting with `#` which
should NOT be parsed as headings.

## Python Examples

Here's a Python example with comments that look like headings:

```python
# This looks like a heading but is a Python comment
def hello():
    # Another fake heading inside code
    print("Hello, World!")

# ## This also looks like an h2 but isn't
class Example:
    # ### Nested fake heading
    pass
```

## Shell Script Examples

Shell scripts use `#` for comments extensively:

```bash
#!/bin/bash
# Main Script
# This entire block should be ignored by the heading parser

# Configuration Section
CONFIG_FILE="/etc/app/config.yaml"

# Helper Functions
function log() {
    echo "[LOG] $1"
}

## Double hash comment (still not a heading!)
### Triple hash comment
```

## Markdown in Fenced Blocks

Sometimes documentation shows markdown syntax itself:

```markdown
# Example H1 in Code Block
## Example H2 in Code Block

This entire fenced block contains markdown that should NOT be parsed.

### Another Fake Heading

- List item
- Another item
```

# Blockquotes with Fake Headings

Blockquotes can also contain lines that look like headings but should be skipped.

## Standard Blockquotes

> This is a normal blockquote.
> It spans multiple lines.

> # This looks like a heading inside a blockquote
> ## And this looks like an h2
> But these should not be parsed as real headings.

## Nested Blockquotes

> Outer quote
> > # Fake heading in nested quote
> > Content continues here
>
> Back to outer level
> ### Another fake heading

# Indented Code Blocks

Indented code (4+ spaces) also creates code blocks where headings should be ignored.

## Traditional Indented Code

The following is an indented code block:

    # This is indented code, not a heading
    def example():
        # Still in the code block
        pass

    ## Also not a heading
    ### Definitely not a heading either

Normal text resumes here after the indented block.

## Mixed Indentation

    # First indented block
    some_code()

Regular paragraph interrupts the code.

    # Second indented block
    more_code()

# Edge Cases

This section tests various edge cases that parsers might struggle with.

## Empty Sections

The following heading has no content before the next heading:

### Empty Section

### Another Empty Section

### Third Empty Section

Back to content now.

## Back-to-Back Headings

# H1 Immediately Followed
## By H2 Immediately Followed
### By H3 Immediately Followed
#### By H4

Finally some content after the heading cascade.

## Headings with Special Characters

### Heading with `code` in It

### Heading with *emphasis* and **strong**

### Heading with [link](http://example.com)

### Heading with "quotes" and 'apostrophes'

### Heading with <angle> brackets

### Heading with & ampersand and @ at-sign

### Heading: with colon

### Heading - with dash

### Heading (with parentheses)

### Heading [with brackets]

### Heading {with braces}

## Very Long Headings

### This is an extremely long heading that goes on and on and on to test how the parser handles headings that exceed typical length expectations and might cause issues with display or storage limitations in some systems

## Headings at Different Levels

# Level 1

## Level 2

### Level 3

#### Level 4

##### Level 5

###### Level 6

####### This is not a valid heading (7 hashes)

# Unicode and Internationalization

## Heading with Emoji Test

Content under emoji test heading.

## Heading with Accented Characters

## Section d'introduction en francais

## Uberschrift auf Deutsch

## Sekcja po Polsku z ogonkami

## Japanese Heading Test

Some content here.

# Final Section

## Whitespace Variations

###No space after hashes (invalid heading per CommonMark)

###    Extra spaces after hashes

   ### Leading spaces before heading (invalid)

	### Tab before heading (invalid)

## Horizontal Rules Near Headings

---

### Heading After HR

Content here.

### Heading Before HR

---

## Lists Near Headings

### Heading Followed by List

- Item 1
- Item 2
- Item 3

### Heading After List

Previous section had a list.

# Conclusion

This document serves as a comprehensive test fixture for markdown heading parsers.
It includes all the edge cases mentioned:

1. Multiple heading levels (h1 through h6)
2. Fenced code blocks with fake headings
3. Blockquotes with fake headings
4. Indented code blocks with fake headings
5. Normal content between sections
6. Preamble before the first heading
7. Empty sections
8. Back-to-back headings
9. Headings with special characters
10. Very long headings
11. Unicode and internationalization
12. Whitespace edge cases

## Expected Heading Count

A correct parser should find approximately 50+ legitimate headings in this document,
while ignoring all the fake headings inside code blocks, blockquotes, and indented code.

### Test Verification

Use this section to verify your parser's output against expected results.

# Appendix

## Appendix A: Reference

Additional reference material.

## Appendix B: Glossary

- **Heading**: A title or subtitle in a document
- **Fenced Code Block**: Code wrapped in triple backticks
- **Blockquote**: Text prefixed with `>`

### End of Document

This is the final content in the test fixture.
