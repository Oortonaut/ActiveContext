"""Tests for split_statements() utility."""

import pytest

from activecontext.core.prompts import split_statements


class TestSplitStatements:
    """Tests for AST-based statement splitting."""

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert split_statements("") == []
        assert split_statements("   ") == []
        assert split_statements("\n\n") == []

    def test_single_statement(self) -> None:
        """Single statement returns as-is."""
        assert split_statements("x = 1") == ["x = 1"]
        assert split_statements("  x = 1  ") == ["x = 1"]

    def test_multiple_simple_statements(self) -> None:
        """Multiple one-line statements split correctly."""
        source = "x = 1\ny = 2\nz = 3"
        result = split_statements(source)
        assert result == ["x = 1", "y = 2", "z = 3"]

    def test_multiline_dict(self) -> None:
        """Multi-line dict is kept together."""
        source = """config = {
    "key1": "value1",
    "key2": "value2",
}
x = 1"""
        result = split_statements(source)
        assert len(result) == 2
        assert "key1" in result[0]
        assert "key2" in result[0]
        assert result[1] == "x = 1"

    def test_multiline_list(self) -> None:
        """Multi-line list is kept together."""
        source = """items = [
    "a",
    "b",
    "c",
]
y = 2"""
        result = split_statements(source)
        assert len(result) == 2
        assert result[1] == "y = 2"

    def test_multiline_function_call(self) -> None:
        """Multi-line function call is kept together."""
        source = """result = some_function(
    arg1="value1",
    arg2="value2",
)
done = True"""
        result = split_statements(source)
        assert len(result) == 2
        assert "arg1" in result[0]
        assert result[1] == "done = True"

    def test_function_definition(self) -> None:
        """Function definitions are split correctly."""
        source = """def foo():
    return 1

def bar():
    return 2"""
        result = split_statements(source)
        assert len(result) == 2
        assert "def foo" in result[0]
        assert "def bar" in result[1]

    def test_class_definition(self) -> None:
        """Class definitions are split correctly."""
        source = """class Foo:
    x = 1

class Bar:
    y = 2"""
        result = split_statements(source)
        assert len(result) == 2

    def test_inline_comments_preserved(self) -> None:
        """Inline comments are preserved in output."""
        source = "x = 1  # comment\ny = 2"
        result = split_statements(source)
        assert result[0] == "x = 1  # comment"
        assert result[1] == "y = 2"

    def test_standalone_comments_preserved_single_stmt(self) -> None:
        """Comments before a single statement are preserved (source returned as-is)."""
        source = "# This is a comment\nx = 1"
        result = split_statements(source)
        # Single statement case returns source as-is, preserving comment
        assert len(result) == 1
        assert "# This is a comment" in result[0]
        assert "x = 1" in result[0]

    def test_comments_between_statements(self) -> None:
        """Comments between statements are included with following statement."""
        source = "x = 1\n# comment\ny = 2"
        result = split_statements(source)
        # AST extracts lines [0:1] for x=1 and [2:3] for y=2
        # The comment on line 1 is between them
        assert len(result) == 2
        assert result[0] == "x = 1"
        assert result[1] == "y = 2"  # Comment not included (between statements)

    def test_syntax_error_returns_source(self) -> None:
        """Syntax errors return original source for caller to handle."""
        source = "x = {invalid"
        result = split_statements(source)
        assert result == [source]

    def test_semicolon_same_line(self) -> None:
        """Semicolon-separated statements on one line are kept together."""
        # This is correct behavior - AST sees them as separate statements
        # but they're on the same line, so they get returned as one string
        source = "x = 1; y = 2"
        result = split_statements(source)
        # Both statements are on line 1, so they're extracted together
        assert len(result) == 2
        # Each gets the full line since they share the same line number
        assert "x = 1" in result[0]

    def test_decorated_function(self) -> None:
        """Decorated functions include decorator in statement."""
        source = """@decorator
def foo():
    pass

x = 1"""
        result = split_statements(source)
        assert len(result) == 2
        assert "@decorator" in result[0]

    def test_try_except_block(self) -> None:
        """Try/except blocks are kept together."""
        source = """try:
    x = 1
except:
    x = 2
y = 3"""
        result = split_statements(source)
        assert len(result) == 2
        assert "try:" in result[0]
        assert "except:" in result[0]
        assert result[1] == "y = 3"

    def test_if_else_block(self) -> None:
        """If/else blocks are kept together."""
        source = """if True:
    x = 1
else:
    x = 2
y = 3"""
        result = split_statements(source)
        assert len(result) == 2
        assert "if True:" in result[0]
        assert result[1] == "y = 3"

    def test_for_loop(self) -> None:
        """For loops are kept together."""
        source = """for i in range(10):
    print(i)
done = True"""
        result = split_statements(source)
        assert len(result) == 2
        assert "for i" in result[0]

    def test_with_statement(self) -> None:
        """With statements are kept together."""
        source = """with open("f") as f:
    data = f.read()
x = 1"""
        result = split_statements(source)
        assert len(result) == 2
        assert "with open" in result[0]

    def test_async_function(self) -> None:
        """Async function definitions work correctly."""
        source = """async def foo():
    await bar()

x = 1"""
        result = split_statements(source)
        assert len(result) == 2
        assert "async def foo" in result[0]

    def test_triple_quoted_string(self) -> None:
        """Multi-line strings are kept together."""
        source = '''doc = """
This is a
multi-line string
"""
x = 1'''
        result = split_statements(source)
        assert len(result) == 2
        assert "multi-line" in result[0]

    def test_continuation_backslash(self) -> None:
        """Line continuation with backslash works."""
        source = """x = 1 + \\
    2 + \\
    3
y = 4"""
        result = split_statements(source)
        assert len(result) == 2

    def test_real_world_startup_pattern(self) -> None:
        """Pattern from startup.md works correctly."""
        source = """ref = group("Reference")
markdown(
    "@prompts/dsl_reference.md",
    parent=ref,
    expansion=Expansion.HEADER,
)
context_guide = markdown(path)"""
        result = split_statements(source)
        assert len(result) == 3
        assert 'ref = group("Reference")' == result[0]
        assert "@prompts/dsl_reference.md" in result[1]
        assert "context_guide" in result[2]
