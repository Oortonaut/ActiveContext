"""Terminal capability detection and graceful degradation.

Detects terminal features such as:
- Color support (24-bit, 256-color, 16-color, none)
- Unicode support
- Terminal size (columns, rows)
- Interactive TTY status

Provides graceful fallbacks when capabilities are limited.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum


class ColorSupport(Enum):
    """Level of color support available in the terminal."""

    NONE = "none"
    """No color support (monochrome)."""

    BASIC_16 = "16"
    """Basic 16-color ANSI support."""

    EXTENDED_256 = "256"
    """Extended 256-color support."""

    TRUECOLOR = "truecolor"
    """24-bit true color support (16 million colors)."""


@dataclass
class TerminalCapabilities:
    """Detected terminal capabilities.

    Attributes:
        is_tty: True if stdout is connected to a TTY.
        color_support: Level of color support available.
        unicode_support: True if terminal supports Unicode (UTF-8).
        columns: Terminal width in columns (or fallback value).
        rows: Terminal height in rows (or fallback value).
        term_program: Terminal program name (e.g., "vscode", "rider").
        term_version: Terminal program version string.
        platform: Operating system platform.
    """

    is_tty: bool
    color_support: ColorSupport
    unicode_support: bool
    columns: int
    rows: int
    term_program: str | None
    term_version: str | None
    platform: str

    @property
    def supports_color(self) -> bool:
        """True if terminal supports any color."""
        return self.color_support != ColorSupport.NONE

    @property
    def supports_unicode(self) -> bool:
        """True if terminal supports Unicode."""
        return self.unicode_support

    @property
    def is_rider(self) -> bool:
        """True if running in JetBrains Rider terminal."""
        return self.term_program == "rider" or (
            self.term_program is not None and "jetbrains" in self.term_program.lower()
        )

    @property
    def is_vscode(self) -> bool:
        """True if running in VS Code terminal."""
        return self.term_program == "vscode"

    def __repr__(self) -> str:
        """Concise repr for debugging."""
        return (
            f"<TerminalCapabilities tty={self.is_tty} "
            f"color={self.color_support.value} "
            f"unicode={self.unicode_support} "
            f"size={self.columns}x{self.rows}>"
        )


def detect_capabilities() -> TerminalCapabilities:
    """Detect current terminal capabilities.

    Returns:
        TerminalCapabilities with detected values and fallbacks.
    """
    # Check if stdout is a TTY
    is_tty = sys.stdout.isatty()

    # Detect color support
    color_support = _detect_color_support(is_tty)

    # Detect Unicode support
    unicode_support = _detect_unicode_support()

    # Get terminal size
    size = shutil.get_terminal_size(fallback=(80, 24))
    columns = size.columns
    rows = size.lines

    # Detect terminal program
    term_program = os.environ.get("TERM_PROGRAM")
    term_version = os.environ.get("TERM_PROGRAM_VERSION")

    # Get platform
    platform = sys.platform

    return TerminalCapabilities(
        is_tty=is_tty,
        color_support=color_support,
        unicode_support=unicode_support,
        columns=columns,
        rows=rows,
        term_program=term_program,
        term_version=term_version,
        platform=platform,
    )


def _detect_color_support(is_tty: bool) -> ColorSupport:
    """Detect color support level.

    Args:
        is_tty: Whether stdout is a TTY.

    Returns:
        Detected color support level.
    """
    # No color if not a TTY or explicitly disabled
    if not is_tty:
        return ColorSupport.NONE

    if os.environ.get("NO_COLOR"):
        return ColorSupport.NONE

    # Check for explicit color term
    colorterm = os.environ.get("COLORTERM", "").lower()
    if "truecolor" in colorterm or "24bit" in colorterm:
        return ColorSupport.TRUECOLOR

    # Check TERM variable
    term = os.environ.get("TERM", "").lower()

    # Dumb terminals have no color
    if term == "dumb":
        return ColorSupport.NONE

    if "256color" in term:
        return ColorSupport.EXTENDED_256
    elif "color" in term or term in ("xterm", "screen", "vt100"):
        return ColorSupport.BASIC_16

    # Windows-specific detection
    if sys.platform == "win32":
        # Windows 10+ supports ANSI colors
        try:
            # Check Windows version
            version = sys.getwindowsversion()
            if version.major >= 10:
                return ColorSupport.TRUECOLOR
        except Exception:
            pass

    # Conservative fallback
    return ColorSupport.NONE


def _detect_unicode_support() -> bool:
    """Detect if terminal supports Unicode.

    Returns:
        True if Unicode is supported.
    """
    # Check encoding
    encoding = sys.stdout.encoding or sys.getdefaultencoding()
    if encoding.lower() in ("utf-8", "utf8"):
        return True

    # Windows-specific check
    if sys.platform == "win32":
        # Windows Terminal and newer consoles support Unicode
        if os.environ.get("WT_SESSION"):  # Windows Terminal
            return True
        # Check for UTF-8 code page
        try:
            import ctypes

            cp = ctypes.windll.kernel32.GetConsoleOutputCP()
            if cp == 65001:  # UTF-8 code page
                return True
        except Exception:
            pass

    return False


# Global cached capabilities
_cached_capabilities: TerminalCapabilities | None = None


def get_capabilities(force_detect: bool = False) -> TerminalCapabilities:
    """Get terminal capabilities (cached).

    Args:
        force_detect: If True, force re-detection instead of using cache.

    Returns:
        Terminal capabilities.
    """
    global _cached_capabilities
    if _cached_capabilities is None or force_detect:
        _cached_capabilities = detect_capabilities()
    return _cached_capabilities


def format_with_fallback(
    text: str,
    *,
    bold: bool = False,
    color: str | None = None,
    capabilities: TerminalCapabilities | None = None,
) -> str:
    """Format text with ANSI codes if supported, plain text otherwise.

    Args:
        text: Text to format.
        bold: If True, make text bold.
        color: Color name ("red", "green", "blue", "yellow", "cyan", "magenta").
        capabilities: Terminal capabilities (auto-detected if None).

    Returns:
        Formatted text (with ANSI codes if supported, plain otherwise).
    """
    caps = capabilities or get_capabilities()

    # No formatting if color not supported
    if not caps.supports_color:
        return text

    # Build ANSI escape codes
    codes = []
    if bold:
        codes.append("1")
    if color:
        color_map = {
            "red": "31",
            "green": "32",
            "yellow": "33",
            "blue": "34",
            "magenta": "35",
            "cyan": "36",
        }
        if color in color_map:
            codes.append(color_map[color])

    if not codes:
        return text

    # Apply formatting
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def get_unicode_char(char_name: str, fallback: str = "?") -> str:
    """Get Unicode character with fallback for limited terminals.

    Args:
        char_name: Character name ("check", "cross", "arrow", "dot").
        fallback: Fallback ASCII character.

    Returns:
        Unicode character or fallback.
    """
    caps = get_capabilities()
    if not caps.supports_unicode:
        return fallback

    char_map = {
        "check": "✓",
        "cross": "✗",
        "arrow": "→",
        "dot": "•",
        "ellipsis": "…",
        "info": "ℹ",
        "warning": "⚠",
        "error": "✗",
    }

    return char_map.get(char_name, fallback)
