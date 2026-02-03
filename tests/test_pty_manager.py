"""Tests for PtyManager — spawn, output batching, tick processing."""

from __future__ import annotations

import asyncio
import sys

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import PtyNode, PtyStatus
from activecontext.terminal.pty_support import is_pty_supported

pytestmark = pytest.mark.skipif(
    not is_pty_supported(),
    reason="PTY not supported on this platform",
)


def _make_graph() -> ContextGraph:
    return ContextGraph()


def _echo_command_with_delay() -> tuple[str, list[str]]:
    """Command that prints 'hello' and stays alive briefly."""
    script = "import time; print('hello'); time.sleep(1)"
    if sys.platform == "win32":
        return "python", ["-c", script]
    return "python3", ["-c", script]


def _long_running_command() -> tuple[str, list[str]]:
    """Command that sleeps for 60 seconds."""
    if sys.platform == "win32":
        return "python", ["-c", "import time; time.sleep(60)"]
    return "sleep", ["60"]


# ---------------------------------------------------------------------------
# Spawn and lifecycle
# ---------------------------------------------------------------------------


class TestSpawn:
    @pytest.mark.asyncio
    async def test_spawn_returns_pty_node(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")
        cmd, args = _echo_command_with_delay()
        node = mgr.spawn(cmd, args)

        assert isinstance(node, PtyNode)
        assert node.command == cmd
        assert node.args == args
        assert graph.get_node(node.node_id) is node

        # Let it run briefly and clean up
        await asyncio.sleep(0.2)
        mgr.close(node.node_id)

    @pytest.mark.asyncio
    async def test_spawn_transitions_to_running(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")
        cmd, args = _echo_command_with_delay()
        node = mgr.spawn(cmd, args)

        # Wait for spawn to complete
        for _ in range(50):
            if node.pty_status == PtyStatus.RUNNING:
                break
            await asyncio.sleep(0.05)

        assert node.pty_status == PtyStatus.RUNNING
        mgr.close(node.node_id)


# ---------------------------------------------------------------------------
# Output batching
# ---------------------------------------------------------------------------


class TestOutputBatching:
    @pytest.mark.asyncio
    async def test_output_appears_after_tick(self):
        """Output should appear in node after process_pending_output()."""
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(
            context_graph=graph,
            cwd=".",
            flush_interval=0.02,  # Fast flush for tests
            flush_threshold=10,
        )
        cmd, args = _echo_command_with_delay()
        node = mgr.spawn(cmd, args)

        # Wait for output to arrive and be flushed by NagleBuffer
        for _ in range(100):
            await asyncio.sleep(0.05)
            updated = mgr.process_pending_output()
            if node._raw_output and "hello" in node._raw_output:
                break

        assert "hello" in node._raw_output, (
            f"Expected 'hello' in output, got: {node._raw_output!r}"
        )
        mgr.close(node.node_id)


# ---------------------------------------------------------------------------
# Exit handling
# ---------------------------------------------------------------------------


class TestExitHandling:
    @pytest.mark.asyncio
    async def test_node_reaches_exited_after_process_ends(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(
            context_graph=graph,
            cwd=".",
            flush_interval=0.02,
        )

        # Use a short-lived command
        if sys.platform == "win32":
            cmd, args = "python", ["-c", "print('done')"]
        else:
            cmd, args = "python3", ["-c", "print('done')"]

        node = mgr.spawn(cmd, args)

        # Wait for process to exit and sentinel to be processed
        for _ in range(100):
            await asyncio.sleep(0.05)
            mgr.process_pending_output()
            if node.is_complete:
                break

        assert node.is_complete
        assert node.pty_status in (PtyStatus.EXITED, PtyStatus.KILLED)


# ---------------------------------------------------------------------------
# Close / close_all
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_kills_and_marks_node(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")
        cmd, args = _long_running_command()
        node = mgr.spawn(cmd, args)

        # Wait for RUNNING
        for _ in range(50):
            if node.pty_status == PtyStatus.RUNNING:
                break
            await asyncio.sleep(0.05)

        mgr.close(node.node_id)
        assert node.is_complete

    @pytest.mark.asyncio
    async def test_close_all(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")

        cmd, args = _long_running_command()
        n1 = mgr.spawn(cmd, args)
        n2 = mgr.spawn(cmd, args)

        await asyncio.sleep(0.3)
        mgr.close_all()

        assert n1.is_complete
        assert n2.is_complete


# ---------------------------------------------------------------------------
# Send input
# ---------------------------------------------------------------------------


class TestSendInput:
    @pytest.mark.asyncio
    async def test_send_input_records_on_node(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")
        cmd, args = _long_running_command()
        node = mgr.spawn(cmd, args)

        # Wait for RUNNING
        for _ in range(50):
            if node.pty_status == PtyStatus.RUNNING:
                break
            await asyncio.sleep(0.05)

        result = mgr.send_input(node.node_id, "hello\n")
        assert result is True
        assert "hello\n" in node.input_history

        mgr.close(node.node_id)

    @pytest.mark.asyncio
    async def test_send_input_returns_false_for_unknown(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")
        assert mgr.send_input("nonexistent", "hello") is False


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class TestResize:
    @pytest.mark.asyncio
    async def test_resize_does_not_crash(self):
        from activecontext.session.pty_manager import PtyManager

        graph = _make_graph()
        mgr = PtyManager(context_graph=graph, cwd=".")
        cmd, args = _long_running_command()
        node = mgr.spawn(cmd, args)

        await asyncio.sleep(0.3)
        mgr.resize(node.node_id, 120, 40)  # Should not raise
        mgr.close(node.node_id)
