"""Media type-aware token counting with tiktoken."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import tiktoken


class MediaType(Enum):
    """Content media types with token estimation ratios."""

    TEXT = "text"  # Plain text, prose
    CODE = "code"  # Source code (Python, JS, etc.)
    MARKUP = "markup"  # HTML, XML, YAML, JSON
    MARKDOWN = "markdown"  # Markdown documents
    DATA = "data"  # CSV, TSV, structured data
    BINARY = "binary"  # Non-text content


# Chars per token by media type (used for budget→char conversion)
CHARS_PER_TOKEN: dict[MediaType, float] = {
    MediaType.TEXT: 4.0,  # Prose is ~4 chars/token
    MediaType.CODE: 3.5,  # Code is more token-dense
    MediaType.MARKUP: 3.0,  # XML/JSON has many short tokens
    MediaType.MARKDOWN: 4.0,  # Similar to prose
    MediaType.DATA: 2.5,  # Tabular data, many delimiters
    MediaType.BINARY: 1.0,  # Bytes = tokens (placeholder)
}


# File extension to media type mapping
EXTENSION_MAP: dict[str, MediaType] = {
    # Code
    ".py": MediaType.CODE,
    ".pyi": MediaType.CODE,
    ".js": MediaType.CODE,
    ".jsx": MediaType.CODE,
    ".ts": MediaType.CODE,
    ".tsx": MediaType.CODE,
    ".java": MediaType.CODE,
    ".c": MediaType.CODE,
    ".h": MediaType.CODE,
    ".cpp": MediaType.CODE,
    ".hpp": MediaType.CODE,
    ".go": MediaType.CODE,
    ".rs": MediaType.CODE,
    ".rb": MediaType.CODE,
    ".php": MediaType.CODE,
    ".swift": MediaType.CODE,
    ".kt": MediaType.CODE,
    ".scala": MediaType.CODE,
    ".cs": MediaType.CODE,
    ".sh": MediaType.CODE,
    ".bash": MediaType.CODE,
    ".zsh": MediaType.CODE,
    ".ps1": MediaType.CODE,
    ".sql": MediaType.CODE,
    # Markup
    ".json": MediaType.MARKUP,
    ".yaml": MediaType.MARKUP,
    ".yml": MediaType.MARKUP,
    ".xml": MediaType.MARKUP,
    ".html": MediaType.MARKUP,
    ".htm": MediaType.MARKUP,
    ".xhtml": MediaType.MARKUP,
    ".toml": MediaType.MARKUP,
    ".ini": MediaType.MARKUP,
    ".cfg": MediaType.MARKUP,
    ".conf": MediaType.MARKUP,
    ".css": MediaType.MARKUP,
    ".scss": MediaType.MARKUP,
    ".less": MediaType.MARKUP,
    # Markdown
    ".md": MediaType.MARKDOWN,
    ".mdx": MediaType.MARKDOWN,
    ".markdown": MediaType.MARKDOWN,
    # Data
    ".csv": MediaType.DATA,
    ".tsv": MediaType.DATA,
    # Text
    ".txt": MediaType.TEXT,
    ".rst": MediaType.TEXT,
    ".log": MediaType.TEXT,
}


def detect_media_type(path: str | Path) -> MediaType:
    """Detect media type from file extension.

    Args:
        path: File path (can be string or Path object)

    Returns:
        MediaType based on extension, defaults to TEXT for unknown extensions
    """
    ext = Path(path).suffix.lower()
    return EXTENSION_MAP.get(ext, MediaType.TEXT)


# Singleton encoder (loaded once on first use)
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Get cached tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("o200k_base")
    return _encoder


def _count_with_tiktoken(text: str, media_type: MediaType) -> int:
    """Count tokens using tiktoken, with media-type fallback for binary."""
    if media_type == MediaType.BINARY:
        return len(text)  # Bytes as tokens for binary content
    return len(_get_encoder().encode(text))


def _count_heuristic(text: str, media_type: MediaType) -> int:
    """Heuristic counter using per-media-type char ratios."""
    ratio = CHARS_PER_TOKEN.get(media_type, 4.0)
    return int(len(text) / ratio)


# Content hash -> token count cache
_token_cache: dict[tuple[int, MediaType], int] = {}


def count_tokens(text: str, media_type: MediaType = MediaType.TEXT) -> int:
    """Count tokens with caching (uses tiktoken).

    Args:
        text: The text content to count tokens for
        media_type: The media type of the content

    Returns:
        Number of tokens in the text
    """
    key = (hash(text), media_type)
    if key not in _token_cache:
        _token_cache[key] = _count_with_tiktoken(text, media_type)
    return _token_cache[key]


def count_tokens_heuristic(text: str, media_type: MediaType = MediaType.TEXT) -> int:
    """Count tokens using heuristic (for budget estimation without encoding).

    Args:
        text: The text content to estimate tokens for
        media_type: The media type of the content

    Returns:
        Estimated number of tokens based on character count and media type ratio
    """
    return _count_heuristic(text, media_type)


def invalidate_cache() -> None:
    """Clear token count cache.

    Call this when content is updated to ensure fresh counts.
    """
    _token_cache.clear()


def tokens_to_chars(tokens: int, media_type: MediaType = MediaType.TEXT) -> int:
    """Convert token budget to approximate character budget.

    Args:
        tokens: Number of tokens
        media_type: The media type for conversion ratio

    Returns:
        Approximate number of characters for the given token count
    """
    return int(tokens * CHARS_PER_TOKEN.get(media_type, 4.0))


def chars_to_tokens(chars: int, media_type: MediaType = MediaType.TEXT) -> int:
    """Convert character count to approximate token count.

    Args:
        chars: Number of characters
        media_type: The media type for conversion ratio

    Returns:
        Approximate number of tokens for the given character count
    """
    ratio = CHARS_PER_TOKEN.get(media_type, 4.0)
    return int(chars / ratio)
