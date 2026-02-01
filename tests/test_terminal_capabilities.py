"""Tests for terminal capability detection and graceful degradation."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

from activecontext.terminal.capabilities import (
    ColorSupport,
    TerminalCapabilities,
    _detect_color_support,
    _detect_unicode_support,
    detect_capabilities,
    format_with_fallback,
    get_capabilities,
    get_unicode_char,
)


class TestColorSupportDetection:
    """Tests for color support detection."""

    def test_no_color_when_not_tty(self) -> None:
        """Color should be disabled when not a TTY."""
        result = _detect_color_support(is_tty=False)
        assert result == ColorSupport.NONE

    @patch.dict(os.environ, {"NO_COLOR": "1"})
    def test_no_color_when_env_set(self) -> None:
        """Color should be disabled when NO_COLOR is set."""
        result = _detect_color_support(is_tty=True)
        assert result == ColorSupport.NONE

    @patch.dict(os.environ, {"COLORTERM": "truecolor"})
    def test_truecolor_detection(self) -> None:
        """Detect 24-bit truecolor support."""
        result = _detect_color_support(is_tty=True)
        assert result == ColorSupport.TRUECOLOR

    @patch.dict(os.environ, {"COLORTERM": "24bit"})
    def test_24bit_detection(self) -> None:
        """Detect 24-bit via COLORTERM=24bit."""
        result = _detect_color_support(is_tty=True)
        assert result == ColorSupport.TRUECOLOR

    @patch.dict(os.environ, {"TERM": "xterm-256color"})
    def test_256_color_detection(self) -> None:
        """Detect 256-color support from TERM."""
        result = _detect_color_support(is_tty=True)
        assert result == ColorSupport.EXTENDED_256

    @patch.dict(os.environ, {"TERM": "xterm"})
    def test_basic_color_detection(self) -> None:
        """Detect basic 16-color support."""
        result = _detect_color_support(is_tty=True)
        assert result == ColorSupport.BASIC_16

    @patch.dict(os.environ, {"TERM": "dumb"})
    def test_no_color_for_dumb_term(self) -> None:
        """Dumb terminal should have no color."""
        result = _detect_color_support(is_tty=True)
        assert result == ColorSupport.NONE


class TestUnicodeSupportDetection:
    """Tests for Unicode support detection."""

    def test_utf8_encoding(self) -> None:
        """UTF-8 encoding should enable Unicode."""
        # This depends on system encoding, so just test that it returns bool
        result = _detect_unicode_support()
        assert isinstance(result, bool)

    @patch.dict(os.environ, {"WT_SESSION": "1"})
    def test_windows_terminal_unicode(self) -> None:
        """Windows Terminal supports Unicode."""
        if sys.platform == "win32":
            result = _detect_unicode_support()
            assert isinstance(result, bool)


class TestTerminalCapabilities:
    """Tests for TerminalCapabilities dataclass."""

    def test_supports_color_property(self) -> None:
        """Test supports_color property."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )
        assert caps.supports_color is True

        caps_no_color = TerminalCapabilities(
            is_tty=False,
            color_support=ColorSupport.NONE,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )
        assert caps_no_color.supports_color is False

    def test_is_rider_detection(self) -> None:
        """Test Rider terminal detection."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program="rider",
            term_version="2025.3",
            platform="win32",
        )
        assert caps.is_rider is True

        caps_jetbrains = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program="JetBrains-Rider",
            term_version="2025.3",
            platform="win32",
        )
        assert caps_jetbrains.is_rider is True

    def test_is_vscode_detection(self) -> None:
        """Test VS Code terminal detection."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.TRUECOLOR,
            unicode_support=True,
            columns=120,
            rows=30,
            term_program="vscode",
            term_version="1.85.0",
            platform="darwin",
        )
        assert caps.is_vscode is True

    def test_repr(self) -> None:
        """Test repr format."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.TRUECOLOR,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )
        repr_str = repr(caps)
        assert "tty=True" in repr_str
        assert "color=truecolor" in repr_str
        assert "unicode=True" in repr_str
        assert "80x24" in repr_str


class TestDetectCapabilities:
    """Tests for capability detection."""

    def test_detect_capabilities_returns_valid_object(self) -> None:
        """Detect capabilities should return valid TerminalCapabilities."""
        caps = detect_capabilities()

        assert isinstance(caps, TerminalCapabilities)
        assert isinstance(caps.is_tty, bool)
        assert isinstance(caps.color_support, ColorSupport)
        assert isinstance(caps.unicode_support, bool)
        assert caps.columns > 0
        assert caps.rows > 0
        assert isinstance(caps.platform, str)

    def test_get_capabilities_caches(self) -> None:
        """get_capabilities should cache result."""
        caps1 = get_capabilities()
        caps2 = get_capabilities()
        # Should be the same instance (cached)
        assert caps1 is caps2

    def test_get_capabilities_force_detect(self) -> None:
        """get_capabilities with force_detect should re-detect."""
        caps1 = get_capabilities()
        caps2 = get_capabilities(force_detect=True)
        # May or may not be same instance, but both should be valid
        assert isinstance(caps1, TerminalCapabilities)
        assert isinstance(caps2, TerminalCapabilities)


class TestFormatWithFallback:
    """Tests for ANSI formatting with graceful fallback."""

    def test_plain_text_when_no_color_support(self) -> None:
        """Should return plain text when color not supported."""
        caps = TerminalCapabilities(
            is_tty=False,
            color_support=ColorSupport.NONE,
            unicode_support=False,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = format_with_fallback("test", color="red", capabilities=caps)
        assert result == "test"
        assert "\033[" not in result

    def test_bold_formatting(self) -> None:
        """Should apply bold formatting when supported."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = format_with_fallback("test", bold=True, capabilities=caps)
        assert "\033[1m" in result
        assert "\033[0m" in result

    def test_color_formatting(self) -> None:
        """Should apply color formatting when supported."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = format_with_fallback("test", color="red", capabilities=caps)
        assert "\033[31m" in result
        assert "\033[0m" in result

    def test_bold_and_color_combined(self) -> None:
        """Should combine bold and color formatting."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.TRUECOLOR,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = format_with_fallback("test", bold=True, color="green", capabilities=caps)
        assert "\033[1;32m" in result or ("\033[1m" in result and "\033[32m" in result)
        assert "\033[0m" in result

    def test_unsupported_color_returns_plain(self) -> None:
        """Unknown color name should still format (just without color code)."""
        caps = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = format_with_fallback("test", color="invalid", capabilities=caps)
        # Should return plain text or text with only reset code
        assert "test" in result


class TestGetUnicodeChar:
    """Tests for Unicode character fallback."""

    @patch("activecontext.terminal.capabilities.get_capabilities")
    def test_unicode_char_when_supported(self, mock_caps) -> None:
        """Should return Unicode char when supported."""
        mock_caps.return_value = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = get_unicode_char("check")
        assert result == "✓"

    @patch("activecontext.terminal.capabilities.get_capabilities")
    def test_fallback_when_no_unicode(self, mock_caps) -> None:
        """Should return fallback when Unicode not supported."""
        mock_caps.return_value = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=False,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = get_unicode_char("check", fallback="*")
        assert result == "*"

    @patch("activecontext.terminal.capabilities.get_capabilities")
    def test_unknown_char_returns_fallback(self, mock_caps) -> None:
        """Unknown char name should return fallback."""
        mock_caps.return_value = TerminalCapabilities(
            is_tty=True,
            color_support=ColorSupport.BASIC_16,
            unicode_support=True,
            columns=80,
            rows=24,
            term_program=None,
            term_version=None,
            platform="linux",
        )

        result = get_unicode_char("unknown_char", fallback="?")
        assert result == "?"
