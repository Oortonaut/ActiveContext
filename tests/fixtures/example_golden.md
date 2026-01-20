# Comprehensive Markdown Parser Test Fixture | line 1..9 of 323 {#text#0}

This preamble appears before any heading in the document. It should be captured
as content belonging to no section, or associated with the first heading depending
on parser implementation. This tests that parsers handle pre-heading content correctly.

Some additional preamble text with **bold** and *italic* formatting to ensure
the parser doesn't get confused by inline markup.

# Introduction | line 10..14 of 323 {#text#1}

This is the introduction section. It contains normal paragraph text that should
be associated with the "Introduction" heading.

## Getting Started | line 15..18 of 323 {#text#2}

This subsection explains how to get started with the project.

### Prerequisites | line 19..26 of 323 {#text#3}

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- Node.js 18+
- Git

### Installation | line 27..34 of 323 {#text#4}

Run the following command to install:

```bash
pip install example-package
```

## Configuration | line 35..38 of 323 {#text#5}

Configuration is handled through YAML files.

### Basic Configuration | line 39..49 of 323 {#text#6}

Here is a minimal configuration example:

```yaml
# This is a YAML comment, not a heading
server:
  host: localhost
  port: 8080
```

# Code Examples with Fake Headings | line 50..54 of 323 {#text#7}

This section contains code blocks that include lines starting with `#` which
should NOT be parsed as headings.

## Python Examples | line 55..70 of 323 {#text#8}

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

## Shell Script Examples | line 71..91 of 323 {#text#9}

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

## Markdown in Fenced Blocks | line 92..107 of 323 {#text#10}

Sometimes documentation shows markdown syntax itself:

```markdown
# Example H1 in Code Block
## Example H2 in Code Block

This entire fenced block contains markdown that should NOT be parsed.

### Another Fake Heading

- List item
- Another item
```

# Blockquotes with Fake Headings | line 108..111 of 323 {#text#11}

Blockquotes can also contain lines that look like headings but should be skipped.

## Standard Blockquotes | line 112..120 of 323 {#text#12}

> This is a normal blockquote.
> It spans multiple lines.

> # This looks like a heading inside a blockquote
> ## And this looks like an h2
> But these should not be parsed as real headings.

## Nested Blockquotes | line 121..129 of 323 {#text#13}

> Outer quote
> > # Fake heading in nested quote
> > Content continues here
>
> Back to outer level
> ### Another fake heading

# Indented Code Blocks | line 130..133 of 323 {#text#14}

Indented code (4+ spaces) also creates code blocks where headings should be ignored.

## Traditional Indented Code | line 134..147 of 323 {#text#15}

The following is an indented code block:

    # This is indented code, not a heading
    def example():
        # Still in the code block
        pass

    ## Also not a heading
    ### Definitely not a heading either

Normal text resumes here after the indented block.

## Mixed Indentation | line 148..157 of 323 {#text#16}

    # First indented block
    some_code()

Regular paragraph interrupts the code.

    # Second indented block
    more_code()

# Edge Cases | line 158..161 of 323 {#text#17}

This section tests various edge cases that parsers might struggle with.

## Empty Sections | line 162..165 of 323 {#text#18}

The following heading has no content before the next heading:

### Empty Section | line 166..167 of 323 {#text#19}

### Another Empty Section | line 168..169 of 323 {#text#20}

### Third Empty Section | line 170..173 of 323 {#text#21}

Back to content now.

## Back-to-Back Headings | line 174..175 of 323 {#text#22}

# H1 Immediately Followed | line 176..176 of 323 {#text#23}
## By H2 Immediately Followed | line 177..177 of 323 {#text#24}
### By H3 Immediately Followed | line 178..178 of 323 {#text#25}
#### By H4 | line 179..182 of 323 {#text#26}

Finally some content after the heading cascade.

## Headings with Special Characters | line 183..184 of 323 {#text#27}

### Heading with `code` in It | line 185..186 of 323 {#text#28}

### Heading with *emphasis* and **strong** | line 187..188 of 323 {#text#29}

### Heading with [link](http://example.com) | line 189..190 of 323 {#text#30}

### Heading with "quotes" and 'apostrophes' | line 191..192 of 323 {#text#31}

### Heading with <angle> brackets | line 193..194 of 323 {#text#32}

### Heading with & ampersand and @ at-sign | line 195..196 of 323 {#text#33}

### Heading: with colon | line 197..198 of 323 {#text#34}

### Heading - with dash | line 199..200 of 323 {#text#35}

### Heading (with parentheses) | line 201..202 of 323 {#text#36}

### Heading [with brackets] | line 203..204 of 323 {#text#37}

### Heading {with braces} | line 205..206 of 323 {#text#38}

## Very Long Headings | line 207..208 of 323 {#text#39}

### This is an extremely long heading that goes on and on and on to test how the parser handles headings that exceed typical length expectations and might cause issues with display or storage limitations in some systems | line 209..210 of 323 {#text#40}

## Headings at Different Levels | line 211..212 of 323 {#text#41}

# Level 1 | line 213..214 of 323 {#text#42}

## Level 2 | line 215..216 of 323 {#text#43}

### Level 3 | line 217..218 of 323 {#text#44}

#### Level 4 | line 219..220 of 323 {#text#45}

##### Level 5 | line 221..222 of 323 {#text#46}

###### Level 6 | line 223..226 of 323 {#text#47}

####### This is not a valid heading (7 hashes)

# Unicode and Internationalization | line 227..228 of 323 {#text#48}

## Heading with Emoji Test | line 229..232 of 323 {#text#49}

Content under emoji test heading.

## Heading with Accented Characters | line 233..234 of 323 {#text#50}

## Section d'introduction en francais | line 235..236 of 323 {#text#51}

## Uberschrift auf Deutsch | line 237..238 of 323 {#text#52}

## Sekcja po Polsku z ogonkami | line 239..240 of 323 {#text#53}

## Japanese Heading Test | line 241..244 of 323 {#text#54}

Some content here.

# Final Section | line 245..246 of 323 {#text#55}

## Whitespace Variations | line 247..250 of 323 {#text#56}

###No space after hashes (invalid heading per CommonMark)

### Extra spaces after hashes | line 251..256 of 323 {#text#57}

   ### Leading spaces before heading (invalid)

	### Tab before heading (invalid)

## Horizontal Rules Near Headings | line 257..260 of 323 {#text#58}

---

### Heading After HR | line 261..264 of 323 {#text#59}

Content here.

### Heading Before HR | line 265..268 of 323 {#text#60}

---

## Lists Near Headings | line 269..270 of 323 {#text#61}

### Heading Followed by List | line 271..276 of 323 {#text#62}

- Item 1
- Item 2
- Item 3

### Heading After List | line 277..280 of 323 {#text#63}

Previous section had a list.

# Conclusion | line 281..298 of 323 {#text#64}

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

## Expected Heading Count | line 299..303 of 323 {#text#65}

A correct parser should find approximately 50+ legitimate headings in this document,
while ignoring all the fake headings inside code blocks, blockquotes, and indented code.

### Test Verification | line 304..307 of 323 {#text#66}

Use this section to verify your parser's output against expected results.

# Appendix | line 308..309 of 323 {#text#67}

## Appendix A: Reference | line 310..313 of 323 {#text#68}

Additional reference material.

## Appendix B: Glossary | line 314..319 of 323 {#text#69}

- **Heading**: A title or subtitle in a document
- **Fenced Code Block**: Code wrapped in triple backticks
- **Blockquote**: Text prefixed with `>`

### End of Document | line 320..323 of 323 {#text#70}

This is the final content in the test fixture.
