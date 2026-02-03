"""Tests for PTY integration in Timeline and Session."""

from __future__ import annotations

import asyncio
import contextlib
import sys

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import PtyNode, PtyStatus
from activecontext.session.protocols import WaitCondition, WaitMode
from activecontext.terminal.pty_support import is_pty_supported

pytestmark = pytest.mark.skipif(
    not is_pty_supported(),
    reason="PTY not supported on this platform",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def timeline(tmp_path):
    """Create a Timeline with PTY support and clean up after test."""
    from activecontext.session.timeline import Timeline

    tl = Timeline(session_id="test", context_graph=ContextGraph(), cwd=str(tmp_path))
    yield tl
    # Clean up PTY and shell tasks
    tl._pty_manager.close_all()
    for task in tl._shell_manager._shell_tasks.values():
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await tl.close()


# ---------------------------------------------------------------------------
# DSL function registration
# ---------------------------------------------------------------------------


class TestDSLRegistration:
    """Verify PTY DSL functions are available in the namespace."""

    @pytest.mark.asyncio
    async def test_pty_function_exists(self, timeline):
        ns = timeline.get_namespace()
        assert "pty" in ns
        assert callable(ns["pty"])

    @pytest.mark.asyncio
    async def test_pty_send_function_exists(self, timeline):
        ns = timeline.get_namespace()
        assert "pty_send" in ns
        assert callable(ns["pty_send"])

    @pytest.mark.asyncio
    async def test_pty_close_function_exists(self, timeline):
        ns = timeline.get_namespace()
        assert "pty_close" in ns
        assert callable(ns["pty_close"])

    @pytest.mark.asyncio
    async def test_wait_for_output_function_exists(self, timeline):
        ns = timeline.get_namespace()
        assert "wait_for_output" in ns
        assert callable(ns["wait_for_output"])


# ---------------------------------------------------------------------------
# pty() via Timeline execution
# ---------------------------------------------------------------------------


class TestPtySpawn:
    @pytest.mark.asyncio
    async def test_pty_spawn_via_execute(self, timeline):
        """Spawn a PTY via timeline.execute_statement."""
        if sys.platform == "win32":
            stmt = 'p = pty("python", args=["-c", "import time; time.sleep(1)"])'
        else:
            stmt = 'p = pty("python3", args=["-c", "import time; time.sleep(1)"])'

        result = await timeline.execute_statement(stmt)
        assert result.status.value == "ok"

        ns = timeline.get_namespace()
        assert "p" in ns
        node = ns["p"]
        assert isinstance(node, PtyNode)

        # Wait briefly for spawn then close
        await asyncio.sleep(0.3)
        timeline._pty_manager.close(node.node_id)

    @pytest.mark.asyncio
    async def test_pty_spawn_added_to_graph(self, timeline):
        """PTY node should be in the context graph."""
        if sys.platform == "win32":
            stmt = 'p = pty("python", args=["-c", "import time; time.sleep(1)"])'
        else:
            stmt = 'p = pty("python3", args=["-c", "import time; time.sleep(1)"])'

        await timeline.execute_statement(stmt)
        ns = timeline.get_namespace()
        node = ns["p"]

        graph_node = timeline.context_graph.get_node(node.node_id)
        assert graph_node is node

        await asyncio.sleep(0.2)
        timeline._pty_manager.close(node.node_id)


# ---------------------------------------------------------------------------
# pty_send() via Timeline
# ---------------------------------------------------------------------------


class TestPtySend:
    @pytest.mark.asyncio
    async def test_pty_send_via_execute(self, timeline):
        """pty_send() should record input on the node."""
        if sys.platform == "win32":
            spawn = 'p = pty("python", args=["-c", "import time; time.sleep(5)"])'
        else:
            spawn = 'p = pty("python3", args=["-c", "import time; time.sleep(5)"])'

        await timeline.execute_statement(spawn)
        ns = timeline.get_namespace()
        node = ns["p"]

        # Wait for RUNNING
        for _ in range(50):
            if node.pty_status == PtyStatus.RUNNING:
                break
            await asyncio.sleep(0.05)

        result = await timeline.execute_statement('pty_send(p, "hello\\n")')
        assert result.status.value == "ok"
        assert "hello\n" in node.input_history

        timeline._pty_manager.close(node.node_id)

    @pytest.mark.asyncio
    async def test_pty_send_with_node_id(self, timeline):
        """pty_send() should accept a node_id string."""
        if sys.platform == "win32":
            spawn = 'p = pty("python", args=["-c", "import time; time.sleep(5)"])'
        else:
            spawn = 'p = pty("python3", args=["-c", "import time; time.sleep(5)"])'

        await timeline.execute_statement(spawn)
        ns = timeline.get_namespace()
        node = ns["p"]

        for _ in range(50):
            if node.pty_status == PtyStatus.RUNNING:
                break
            await asyncio.sleep(0.05)

        result = await timeline.execute_statement(f'pty_send("{node.node_id}", "test\\n")')
        assert result.status.value == "ok"
        assert "test\n" in node.input_history

        timeline._pty_manager.close(node.node_id)


# ---------------------------------------------------------------------------
# pty_close() via Timeline
# ---------------------------------------------------------------------------


class TestPtyClose:
    @pytest.mark.asyncio
    async def test_pty_close_via_execute(self, timeline):
        """pty_close() should terminate the PTY."""
        if sys.platform == "win32":
            spawn = 'p = pty("python", args=["-c", "import time; time.sleep(60)"])'
        else:
            spawn = 'p = pty("sleep", args=["60"])'

        await timeline.execute_statement(spawn)
        ns = timeline.get_namespace()
        node = ns["p"]

        await asyncio.sleep(0.3)

        result = await timeline.execute_statement("pty_close(p)")
        assert result.status.value == "ok"
        assert node.is_complete


# ---------------------------------------------------------------------------
# wait_for_output() via Timeline
# ---------------------------------------------------------------------------


class TestWaitForOutput:
    @pytest.mark.asyncio
    async def test_wait_for_output_sets_condition(self, timeline):
        """wait_for_output() should set a WaitCondition with output_pattern."""
        if sys.platform == "win32":
            spawn = 'p = pty("python", args=["-c", "import time; time.sleep(5)"])'
        else:
            spawn = 'p = pty("python3", args=["-c", "import time; time.sleep(5)"])'

        await timeline.execute_statement(spawn)
        ns = timeline.get_namespace()
        node = ns["p"]

        result = await timeline.execute_statement(
            'wait_for_output(p, ">>> ", timeout=5)'
        )
        assert result.status.value == "ok"

        # Should have set a wait condition
        assert timeline.is_waiting()
        condition = timeline.get_wait_condition()
        assert condition is not None
        assert condition.output_pattern == ">>> "
        assert condition.timeout == 5.0
        assert node.node_id in condition.node_ids

        # Clean up
        timeline.clear_wait_condition()
        timeline._pty_manager.close(node.node_id)

    @pytest.mark.asyncio
    async def test_wait_for_output_ends_turn(self, timeline):
        """wait_for_output() should set done_called to end the turn."""
        if sys.platform == "win32":
            spawn = 'p = pty("python", args=["-c", "import time; time.sleep(5)"])'
        else:
            spawn = 'p = pty("python3", args=["-c", "import time; time.sleep(5)"])'

        await timeline.execute_statement(spawn)
        result = await timeline.execute_statement(
            'wait_for_output(p, "prompt> ")'
        )
        assert result.status.value == "ok"
        assert timeline._done_called

        # Clean up
        timeline.clear_wait_condition()
        ns = timeline.get_namespace()
        timeline._pty_manager.close(ns["p"].node_id)


# ---------------------------------------------------------------------------
# check_wait_condition with output_pattern
# ---------------------------------------------------------------------------


class TestCheckWaitConditionPty:
    def test_output_pattern_not_yet_matched(self):
        """check_wait_condition returns False when pattern not in scrollback."""
        from activecontext.session.timeline import Timeline

        graph = ContextGraph()
        tl = Timeline(session_id="test", context_graph=graph, cwd=".")
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        graph.add_node(node)
        node.append_output("Starting program...\n")

        tl._wait_condition = WaitCondition(
            node_ids=[node.node_id],
            mode=WaitMode.SINGLE,
            wake_prompt="Pattern matched.",
            output_pattern="(gdb) ",
        )

        satisfied, prompt = tl.check_wait_condition()
        assert not satisfied
        assert prompt is None

    def test_output_pattern_matched(self):
        """check_wait_condition returns True when pattern is in scrollback."""
        from activecontext.session.timeline import Timeline

        graph = ContextGraph()
        tl = Timeline(session_id="test", context_graph=graph, cwd=".")
        node = PtyNode(command="gdb", pty_status=PtyStatus.RUNNING)
        graph.add_node(node)
        node.append_output("Breakpoint 1 at main\n(gdb) \n")

        tl._wait_condition = WaitCondition(
            node_ids=[node.node_id],
            mode=WaitMode.SINGLE,
            wake_prompt="GDB is ready.",
            output_pattern="(gdb) ",
        )

        satisfied, prompt = tl.check_wait_condition()
        assert satisfied
        assert prompt == "GDB is ready."

    def test_output_pattern_pty_exited(self):
        """check_wait_condition returns failure when PTY exits without match."""
        from activecontext.session.timeline import Timeline

        graph = ContextGraph()
        tl = Timeline(session_id="test", context_graph=graph, cwd=".")
        node = PtyNode(command="gdb", pty_status=PtyStatus.EXITED, exit_code=1)
        graph.add_node(node)
        node.append_output("Error: file not found\n")

        tl._wait_condition = WaitCondition(
            node_ids=[node.node_id],
            mode=WaitMode.SINGLE,
            wake_prompt="Matched.",
            output_pattern="(gdb) ",
        )

        satisfied, prompt = tl.check_wait_condition()
        assert satisfied
        assert "exited" in prompt.lower()

    def test_pty_node_in_standard_wait(self):
        """PtyNode should work with standard wait() (no output_pattern)."""
        from activecontext.session.timeline import Timeline

        graph = ContextGraph()
        tl = Timeline(session_id="test", context_graph=graph, cwd=".")
        node = PtyNode(command="gdb", pty_status=PtyStatus.EXITED, exit_code=0)
        graph.add_node(node)

        tl._wait_condition = WaitCondition(
            node_ids=[node.node_id],
            mode=WaitMode.SINGLE,
            wake_prompt="PTY done.",
        )

        satisfied, prompt = tl.check_wait_condition()
        assert satisfied


# ---------------------------------------------------------------------------
# process_pending_pty_output via Timeline
# ---------------------------------------------------------------------------


class TestProcessPendingPtyOutput:
    def test_delegates_to_pty_manager(self):
        """process_pending_pty_output should delegate to PtyManager."""
        from activecontext.session.timeline import Timeline

        graph = ContextGraph()
        tl = Timeline(session_id="test", context_graph=graph, cwd=".")

        # Nothing pending — should return empty
        result = tl.process_pending_pty_output()
        assert result == []


# ---------------------------------------------------------------------------
# Timeline.close() includes PTY cleanup
# ---------------------------------------------------------------------------


class TestTimelineClose:
    @pytest.mark.asyncio
    async def test_close_cleans_up_ptys(self, timeline):
        """close() should terminate all PTY sessions."""
        if sys.platform == "win32":
            spawn = 'p = pty("python", args=["-c", "import time; time.sleep(60)"])'
        else:
            spawn = 'p = pty("sleep", args=["60"])'

        await timeline.execute_statement(spawn)
        ns = timeline.get_namespace()
        node = ns["p"]

        await asyncio.sleep(0.3)

        # Close the timeline — should clean up PTYs
        await timeline.close()

        assert node.is_complete
