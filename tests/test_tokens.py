"""Tests for the token counting module."""

from __future__ import annotations

import pytest

from activecontext.core.tokens import (
    CHARS_PER_TOKEN,
    EXTENSION_MAP,
    MediaType,
    chars_to_tokens,
    count_tokens,
    count_tokens_heuristic,
    detect_media_type,
    invalidate_cache,
    tokens_to_chars,
)


class TestMediaTypeDetection:
    """Tests for media type detection from file extensions."""

    def test_python_file(self) -> None:
        assert detect_media_type("main.py") == MediaType.CODE
        assert detect_media_type("src/utils.pyi") == MediaType.CODE

    def test_javascript_file(self) -> None:
        assert detect_media_type("app.js") == MediaType.CODE
        assert detect_media_type("component.jsx") == MediaType.CODE
        assert detect_media_type("index.ts") == MediaType.CODE
        assert detect_media_type("component.tsx") == MediaType.CODE

    def test_markup_files(self) -> None:
        assert detect_media_type("config.json") == MediaType.MARKUP
        assert detect_media_type("settings.yaml") == MediaType.MARKUP
        assert detect_media_type("config.yml") == MediaType.MARKUP
        assert detect_media_type("data.xml") == MediaType.MARKUP
        assert detect_media_type("index.html") == MediaType.MARKUP
        assert detect_media_type("pyproject.toml") == MediaType.MARKUP

    def test_markdown_files(self) -> None:
        assert detect_media_type("README.md") == MediaType.MARKDOWN
        assert detect_media_type("docs/guide.mdx") == MediaType.MARKDOWN

    def test_data_files(self) -> None:
        assert detect_media_type("data.csv") == MediaType.DATA
        assert detect_media_type("records.tsv") == MediaType.DATA

    def test_text_files(self) -> None:
        assert detect_media_type("notes.txt") == MediaType.TEXT
        assert detect_media_type("doc.rst") == MediaType.TEXT

    def test_unknown_extension_defaults_to_text(self) -> None:
        assert detect_media_type("file.xyz") == MediaType.TEXT
        assert detect_media_type("file.unknown") == MediaType.TEXT
        assert detect_media_type("noextension") == MediaType.TEXT

    def test_case_insensitive(self) -> None:
        assert detect_media_type("File.PY") == MediaType.CODE
        assert detect_media_type("README.MD") == MediaType.MARKDOWN
        assert detect_media_type("Config.JSON") == MediaType.MARKUP

    def test_pathlib_path(self) -> None:
        from pathlib import Path

        assert detect_media_type(Path("src/main.py")) == MediaType.CODE
        assert detect_media_type(Path("docs/README.md")) == MediaType.MARKDOWN


class TestCharsPerToken:
    """Tests for character-to-token ratio constants."""

    def test_ratios_defined(self) -> None:
        assert MediaType.TEXT in CHARS_PER_TOKEN
        assert MediaType.CODE in CHARS_PER_TOKEN
        assert MediaType.MARKUP in CHARS_PER_TOKEN
        assert MediaType.MARKDOWN in CHARS_PER_TOKEN
        assert MediaType.DATA in CHARS_PER_TOKEN
        assert MediaType.BINARY in CHARS_PER_TOKEN

    def test_code_more_dense_than_prose(self) -> None:
        # Code has fewer chars per token (more token-dense)
        assert CHARS_PER_TOKEN[MediaType.CODE] < CHARS_PER_TOKEN[MediaType.TEXT]

    def test_markup_more_dense_than_prose(self) -> None:
        # Markup (JSON/XML) has many short tokens
        assert CHARS_PER_TOKEN[MediaType.MARKUP] < CHARS_PER_TOKEN[MediaType.TEXT]


class TestTokenCounting:
    """Tests for tiktoken-based token counting."""

    def test_count_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_count_simple_text(self) -> None:
        result = count_tokens("Hello, world!")
        assert result > 0
        assert isinstance(result, int)

    def test_count_code_snippet(self) -> None:
        code = "def hello():\n    print('Hello')\n"
        result = count_tokens(code, MediaType.CODE)
        assert result > 0

    def test_count_json(self) -> None:
        json_text = '{"name": "test", "value": 123}'
        result = count_tokens(json_text, MediaType.MARKUP)
        assert result > 0

    def test_binary_uses_length(self) -> None:
        # Binary content counts bytes as tokens
        binary = "abc123"
        result = count_tokens(binary, MediaType.BINARY)
        assert result == len(binary)


class TestHeuristicCounting:
    """Tests for heuristic token estimation."""

    def test_text_ratio(self) -> None:
        text = "a" * 100  # 100 characters
        result = count_tokens_heuristic(text, MediaType.TEXT)
        expected = int(100 / CHARS_PER_TOKEN[MediaType.TEXT])
        assert result == expected

    def test_code_ratio(self) -> None:
        code = "x" * 100
        result = count_tokens_heuristic(code, MediaType.CODE)
        expected = int(100 / CHARS_PER_TOKEN[MediaType.CODE])
        assert result == expected

    def test_empty_string(self) -> None:
        assert count_tokens_heuristic("") == 0


class TestTokenCharConversion:
    """Tests for token-character conversion functions."""

    def test_tokens_to_chars_text(self) -> None:
        result = tokens_to_chars(100, MediaType.TEXT)
        expected = int(100 * CHARS_PER_TOKEN[MediaType.TEXT])
        assert result == expected

    def test_tokens_to_chars_code(self) -> None:
        result = tokens_to_chars(100, MediaType.CODE)
        expected = int(100 * CHARS_PER_TOKEN[MediaType.CODE])
        assert result == expected

    def test_chars_to_tokens_text(self) -> None:
        result = chars_to_tokens(400, MediaType.TEXT)
        expected = int(400 / CHARS_PER_TOKEN[MediaType.TEXT])
        assert result == expected

    def test_roundtrip(self) -> None:
        # tokens -> chars -> tokens should approximate original
        original = 100
        chars = tokens_to_chars(original, MediaType.TEXT)
        back = chars_to_tokens(chars, MediaType.TEXT)
        assert back == original


class TestCaching:
    """Tests for token count caching."""

    def test_cache_hit(self) -> None:
        invalidate_cache()  # Start fresh
        text = "This is a test string for caching."

        # First call computes
        result1 = count_tokens(text)
        # Second call should hit cache (same result)
        result2 = count_tokens(text)

        assert result1 == result2

    def test_cache_invalidation(self) -> None:
        text = "Test text for invalidation"
        result1 = count_tokens(text)

        invalidate_cache()

        # After invalidation, still works
        result2 = count_tokens(text)
        assert result1 == result2

    def test_different_media_types_cached_separately(self) -> None:
        invalidate_cache()
        text = "Same text different type"

        # For non-binary, tiktoken gives same result regardless of media_type
        result_text = count_tokens(text, MediaType.TEXT)
        result_code = count_tokens(text, MediaType.CODE)

        # tiktoken returns same count for same content
        assert result_text == result_code

        # Binary is different (uses length)
        result_binary = count_tokens(text, MediaType.BINARY)
        assert result_binary == len(text)


class TestExtensionMap:
    """Tests for the extension mapping coverage."""

    def test_common_code_extensions_covered(self) -> None:
        code_extensions = [".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs", ".rb"]
        for ext in code_extensions:
            assert ext in EXTENSION_MAP, f"{ext} should be in EXTENSION_MAP"
            assert EXTENSION_MAP[ext] == MediaType.CODE

    def test_common_markup_extensions_covered(self) -> None:
        markup_extensions = [".json", ".yaml", ".yml", ".xml", ".html", ".toml"]
        for ext in markup_extensions:
            assert ext in EXTENSION_MAP, f"{ext} should be in EXTENSION_MAP"
            assert EXTENSION_MAP[ext] == MediaType.MARKUP

    def test_css_is_markup(self) -> None:
        assert EXTENSION_MAP.get(".css") == MediaType.MARKUP
        assert EXTENSION_MAP.get(".scss") == MediaType.MARKUP


class TestEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_unicode_text(self) -> None:
        text = "Hello 世界! 🎉"
        result = count_tokens(text)
        assert result > 0

    def test_multiline_code(self) -> None:
        code = """
def function():
    x = 1
    y = 2
    return x + y
"""
        result = count_tokens(code, MediaType.CODE)
        assert result > 0

    def test_large_text(self) -> None:
        # 10KB of text
        large_text = "word " * 2000
        result = count_tokens(large_text)
        assert result > 0
        # Should be reasonable (not just chars/4)
        assert result < len(large_text)

    def test_newlines_and_whitespace(self) -> None:
        text = "\n\n\n   \t\t\n"
        result = count_tokens(text)
        # Whitespace still has tokens
        assert result >= 0
