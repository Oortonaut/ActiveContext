"""Tests for PtyNode context node."""

from __future__ import annotations

import time

import pytest

from activecontext.context.nodes import (
    PtyNode,
    PtyStatus,
    _strip_ansi,
)
from activecontext.context.state import Expansion


# ---------------------------------------------------------------------------
# ANSI stripping
# ---------------------------------------------------------------------------


class TestStripAnsi:
    def test_plain_text_unchanged(self):
        assert _strip_ansi("hello world") == "hello world"

    def test_strip_sgr(self):
        assert _strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strip_cursor_movement(self):
        assert _strip_ansi("\x1b[2J\x1b[H") == ""

    def test_strip_osc_title(self):
        assert _strip_ansi("\x1b]0;window title\x07") == ""

    def test_strip_mixed(self):
        raw = "\x1b[?25l\x1b[32mOK\x1b[0m done\x1b[?25h"
        assert _strip_ansi(raw) == "OK done"


# ---------------------------------------------------------------------------
# Construction and identity
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_expansion_is_content(self):
        node = PtyNode(command="python")
        assert node.default_expansion == Expansion.CONTENT

    def test_node_type(self):
        node = PtyNode(command="gdb")
        assert node.node_type == "PtyNode"

    def test_full_command_no_args(self):
        node = PtyNode(command="python")
        assert node.full_command == "python"

    def test_full_command_with_args(self):
        node = PtyNode(command="gdb", args=["--quiet", "./a.out"])
        assert node.full_command == "gdb --quiet ./a.out"

    def test_is_complete_false_when_pending(self):
        node = PtyNode(command="gdb")
        assert not node.is_complete

    def test_is_complete_false_when_running(self):
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        assert not node.is_complete


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_set_running(self):
        node = PtyNode(command="gdb")
        result = node.set_running()
        assert result is node
        assert node.pty_status == PtyStatus.RUNNING

    def test_set_exited_success(self):
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        node.set_exited(0)
        assert node.pty_status == PtyStatus.EXITED
        assert node.exit_code == 0
        assert node.is_complete

    def test_set_exited_failure(self):
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        node.set_exited(1)
        assert node.pty_status == PtyStatus.EXITED
        assert node.exit_code == 1
        assert node.is_complete

    def test_set_exited_with_signal(self):
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        node.set_exited(-9, signal_name="SIGKILL")
        assert node.pty_status == PtyStatus.KILLED
        assert node.signal == "SIGKILL"
        assert node.is_complete

    def test_set_error(self):
        node = PtyNode(command="gdb")
        node.set_error("spawn failed")
        assert node.pty_status == PtyStatus.ERROR
        assert node.is_complete
        assert "spawn failed" in node._raw_output


# ---------------------------------------------------------------------------
# Scrollback ring buffer
# ---------------------------------------------------------------------------


class TestScrollback:
    def test_append_single_line(self):
        node = PtyNode(command="test")
        node.append_output("hello\n")
        assert node._scrollback_lines == ["hello"]
        assert node._total_line_count == 1

    def test_append_multiple_lines(self):
        node = PtyNode(command="test")
        node.append_output("line1\nline2\nline3\n")
        assert node._scrollback_lines == ["line1", "line2", "line3"]
        assert node._total_line_count == 3

    def test_append_incremental(self):
        node = PtyNode(command="test")
        node.append_output("first\n")
        node.append_output("second\n")
        assert node._scrollback_lines == ["first", "second"]
        assert node._total_line_count == 2

    def test_append_without_trailing_newline(self):
        node = PtyNode(command="test")
        node.append_output("no newline")
        assert node._scrollback_lines == ["no newline"]

    def test_empty_append_is_noop(self):
        node = PtyNode(command="test")
        node.append_output("")
        assert node._scrollback_lines == []
        assert node._total_line_count == 0

    def test_ring_buffer_line_limit(self):
        node = PtyNode(command="test")
        # Add more than _PTY_MAX_LINES
        for i in range(600):
            node.append_output(f"line{i}\n")
        assert len(node._scrollback_lines) <= 500
        assert node._total_line_count == 600
        # Oldest lines should be dropped
        assert node._scrollback_lines[0] == "line100"
        assert node._scrollback_lines[-1] == "line599"

    def test_ring_buffer_byte_limit(self):
        node = PtyNode(command="test")
        # Add lines that exceed _PTY_MAX_BYTES (100KB)
        big_line = "x" * 5000  # 5KB per line
        for i in range(30):
            node.append_output(f"{big_line}\n")
        # Should trim to stay under 100KB
        total_bytes = sum(len(l) + 1 for l in node._scrollback_lines)
        assert total_bytes <= 100_000 + 5001  # allow one line of slack

    def test_raw_output_accumulates(self):
        node = PtyNode(command="test")
        node.append_output("hello\n")
        node.append_output("world\n")
        assert node._raw_output == "hello\nworld\n"


# ---------------------------------------------------------------------------
# Input tracking
# ---------------------------------------------------------------------------


class TestInputHistory:
    def test_record_input(self):
        node = PtyNode(command="gdb")
        node.record_input("break main\n")
        node.record_input("run\n")
        assert node.input_history == ["break main\n", "run\n"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_render_digest_pending(self):
        node = PtyNode(command="python", args=["-i"])
        digest = node.render_digest()
        assert "PTY:" in digest
        assert "python -i" in digest
        assert "PENDING" in digest

    def test_render_digest_running(self):
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        assert "RUNNING" in node.render_digest()

    def test_render_digest_truncates_long_command(self):
        node = PtyNode(command="python", args=["-c", "x" * 100])
        digest = node.render_digest()
        assert "..." in digest

    def test_render_content_shows_scrollback(self):
        node = PtyNode(command="test", pty_status=PtyStatus.RUNNING)
        node.append_output("line1\nline2\nline3\n")
        content = node.render_content()
        assert "line1" in content
        assert "line2" in content
        assert "line3" in content

    def test_render_content_strips_ansi(self):
        node = PtyNode(command="test", pty_status=PtyStatus.RUNNING)
        node.append_output("\x1b[31mcolored\x1b[0m\n")
        content = node.render_content()
        assert "colored" in content
        assert "\x1b" not in content

    def test_render_content_shows_last_50_lines(self):
        node = PtyNode(command="test", pty_status=PtyStatus.RUNNING)
        for i in range(100):
            node.append_output(f"line{i}\n")
        content = node.render_content()
        assert "line99" in content
        assert "line50" in content
        assert "line49" not in content

    def test_render_content_shows_omitted_count(self):
        node = PtyNode(command="test", pty_status=PtyStatus.RUNNING)
        for i in range(100):
            node.append_output(f"line{i}\n")
        content = node.render_content()
        assert "earlier lines omitted" in content

    def test_render_content_shows_exit_footer(self):
        node = PtyNode(command="test", pty_status=PtyStatus.EXITED, exit_code=0)
        node.append_output("done\n")
        content = node.render_content()
        assert "exit_code=0" in content

    def test_render_content_shows_signal_in_footer(self):
        node = PtyNode(
            command="test",
            pty_status=PtyStatus.KILLED,
            exit_code=-9,
            signal="SIGKILL",
        )
        content = node.render_content()
        assert "signal=SIGKILL" in content

    def test_get_digest_dict(self):
        node = PtyNode(command="gdb", args=["./a.out"], pty_status=PtyStatus.RUNNING)
        d = node.GetDigest()
        assert d["type"] == "PtyNode"
        assert d["command"] == "gdb ./a.out"
        assert d["status"] == "running"


# ---------------------------------------------------------------------------
# Token breakdown
# ---------------------------------------------------------------------------


class TestTokenBreakdown:
    def test_empty_node(self):
        node = PtyNode(command="gdb")
        info = node.get_token_breakdown()
        assert info.title > 0
        assert info.content == 0
        assert info.detail == 0

    def test_with_output(self):
        node = PtyNode(command="gdb")
        for i in range(20):
            node.append_output(f"line {i}\n")
        info = node.get_token_breakdown()
        assert info.title > 0
        assert info.content > 0  # last 10 lines
        assert info.detail > 0  # remaining lines


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_round_trip(self):
        node = PtyNode(
            command="gdb",
            args=["./a.out"],
            pty_status=PtyStatus.RUNNING,
            exit_code=None,
        )
        node.append_output("(gdb) \n")
        node.record_input("break main\n")

        data = node.to_dict()
        assert data["node_type"] == "PtyNode"
        assert data["command"] == "gdb"
        assert data["args"] == ["./a.out"]
        assert data["pty_status"] == "running"
        assert data["input_history"] == ["break main\n"]
        assert "(gdb)" in data["raw_output"]

    def test_from_dict_restores_scrollback(self):
        node = PtyNode(command="python", args=["-i"])
        node.append_output(">>> print(1)\n1\n>>> \n")
        data = node.to_dict()

        restored = PtyNode._from_dict(data)
        assert restored.command == "python"
        assert restored.args == ["-i"]
        assert restored._scrollback_lines == [">>> print(1)", "1", ">>> "]
        assert restored._total_line_count == 3

    def test_from_dict_default_expansion(self):
        data = {
            "node_type": "PtyNode",
            "node_id": "test1234",
            "command": "gdb",
        }
        node = PtyNode._from_dict(data)
        assert node.default_expansion == Expansion.CONTENT

    def test_from_dict_preserves_status(self):
        data = {
            "node_type": "PtyNode",
            "node_id": "test1234",
            "command": "gdb",
            "pty_status": "exited",
            "exit_code": 0,
        }
        node = PtyNode._from_dict(data)
        assert node.pty_status == PtyStatus.EXITED
        assert node.exit_code == 0

    def test_registry_round_trip(self):
        """Verify that the node registry can deserialize PtyNode."""
        from activecontext.context.registry import get_node_registry

        registry = get_node_registry()
        cls = registry.get("PtyNode")
        assert cls is PtyNode

        node = PtyNode(command="ssh", args=["host"])
        node.append_output("Last login: ...\n$ \n")
        data = node.to_dict()

        restored = registry.from_dict(data)
        assert isinstance(restored, PtyNode)
        assert restored.command == "ssh"
        assert "Last login" in restored._raw_output
