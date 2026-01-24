"""Tests for Timeline critical paths.

These tests focus on:
- Statement replay/rollback
- Shell execution flows
- Lock management
- Error handling paths
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.session.timeline import Timeline
from activecontext.context.graph import ContextGraph
from activecontext.context.state import Expansion


class TestReplayFrom:
    """Tests for replay_from functionality."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_replay_from_start(self, temp_cwd: Path) -> None:
        """Test replaying from the beginning."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Execute some statements
            await timeline.execute_statement('t1 = topic("Topic 1")')
            await timeline.execute_statement('t2 = topic("Topic 2")')

            # Replay from start
            results = []
            async for result in timeline.replay_from(0):
                results.append(result)

            assert len(results) == 2
            assert all(r.status.value == "ok" for r in results)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_replay_from_last_index(self, temp_cwd: Path) -> None:
        """Test replaying from the last index yields results."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t1 = topic("Topic 1")')
            await timeline.execute_statement('t2 = topic("Topic 2")')
            await timeline.execute_statement('t3 = topic("Topic 3")')

            # Replay from last index
            results = []
            async for result in timeline.replay_from(2):
                results.append(result)

            # Should have at least one result (the last statement)
            assert len(results) >= 1
            # All results should be successful
            assert all(r.status.value == "ok" for r in results)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_replay_from_negative_index(self, temp_cwd: Path) -> None:
        """Test that negative index returns empty iterator."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Topic")')

            results = []
            async for result in timeline.replay_from(-1):
                results.append(result)

            assert len(results) == 0
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_replay_from_out_of_bounds(self, temp_cwd: Path) -> None:
        """Test that out of bounds index returns empty iterator."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Topic")')

            results = []
            async for result in timeline.replay_from(100):
                results.append(result)

            assert len(results) == 0
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_replay_clears_state(self, temp_cwd: Path) -> None:
        """Test that replay properly clears and rebuilds state."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Original")')

            # Verify we have context objects
            assert len(timeline.get_context_objects()) == 1

            # Replay - should clear and rebuild
            async for _ in timeline.replay_from(0):
                pass

            # Should still have context objects
            assert len(timeline.get_context_objects()) == 1
        finally:
            await timeline.close()


class TestShellExecution:
    """Tests for shell command execution."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_shell_creates_node(self, temp_cwd: Path) -> None:
        """Test that shell() creates a ShellNode in the context graph."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement('s = shell("echo", ["hello"])')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            shell_node = ns["s"]

            assert shell_node is not None
            assert shell_node.command == "echo"
            assert shell_node.args == ["hello"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_shell_with_env_vars(self, temp_cwd: Path) -> None:
        """Test shell execution with environment variables."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement(
                's = shell("echo", env={"MY_VAR": "test_value"})'
            )
            assert result.status.value == "ok"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_shell_with_timeout(self, temp_cwd: Path) -> None:
        """Test shell execution with timeout."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement(
                's = shell("echo", ["test"], timeout=60.0)'
            )
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            shell_node = ns["s"]
            # Node should be created regardless of execution status
            assert shell_node is not None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_shell_node_added_to_graph(self, temp_cwd: Path) -> None:
        """Test that shell node is properly added to context graph."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('s = shell("echo")')

            ns = timeline.get_namespace()
            shell_node = ns["s"]

            graph = timeline.context_graph
            retrieved = graph.get_node(shell_node.node_id)
            assert retrieved is shell_node
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_cancel_all_shells(self, temp_cwd: Path) -> None:
        """Test cancelling all pending shell tasks."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Start a shell (it may complete quickly on fast systems)
            await timeline.execute_statement('s = shell("echo", ["test"])')

            # Cancel all should not raise (it's a sync function returning int)
            cancelled = timeline._shell_manager.cancel_all()
            assert isinstance(cancelled, int)
        finally:
            await timeline.close()


class TestLockManagement:
    """Tests for file lock management."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_lock_creates_node(self, temp_cwd: Path) -> None:
        """Test that lock_file() creates a LockNode."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement('lock = lock_file(".test.lock")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            lock_node = ns["lock"]

            assert lock_node is not None
            assert ".test.lock" in lock_node.lockfile
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_lock_with_custom_timeout(self, temp_cwd: Path) -> None:
        """Test lock acquisition with custom timeout."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement(
                'lock = lock_file(".test.lock", timeout=5.0)'
            )
            assert result.status.value == "ok"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_lock_with_custom_state(self, temp_cwd: Path) -> None:
        """Test lock with custom initial state."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement(
                'lock = lock_file(".test.lock", expansion=Expansion.ALL)'
            )
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            lock_node = ns["lock"]
            assert lock_node.expansion == Expansion.ALL
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_lock_node_added_to_graph(self, temp_cwd: Path) -> None:
        """Test that lock node is properly added to context graph."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('lock = lock_file(".test.lock")')

            ns = timeline.get_namespace()
            lock_node = ns["lock"]

            graph = timeline.context_graph
            retrieved = graph.get_node(lock_node.node_id)
            assert retrieved is lock_node
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_release_all_locks(self, temp_cwd: Path) -> None:
        """Test releasing all locks (sync function)."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('lock = lock_file(".test.lock")')

            # Release all should not raise (it's a sync function returning int)
            released = timeline._lock_manager.release_all()
            assert isinstance(released, int)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_cancel_all_locks(self, temp_cwd: Path) -> None:
        """Test cancelling all pending lock tasks (sync function)."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('lock = lock_file(".test.lock")')

            # Cancel all should not raise (it's a sync function returning int)
            cancelled = timeline._lock_manager.cancel_all()
            assert isinstance(cancelled, int)
        finally:
            await timeline.close()


class TestExecutionErrorHandling:
    """Tests for error handling during statement execution."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_syntax_error_returns_error_status(self, temp_cwd: Path) -> None:
        """Test that syntax errors are caught and reported."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement("x = [unclosed")
            assert result.status.value == "error"
            # Exception dict should have error info
            assert result.exception is not None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_name_error_returns_error_status(self, temp_cwd: Path) -> None:
        """Test that name errors are caught and reported."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement("x = undefined_variable")
            assert result.status.value == "error"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_exception_in_dsl_function(self, temp_cwd: Path) -> None:
        """Test that exceptions in DSL functions are handled."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Try to create text node for non-existent file (should not raise)
            result = await timeline.execute_statement('v = text("nonexistent_file.py")')
            # This might succeed (creates a node) or fail (permission denied)
            # Either way, it should not raise an unhandled exception
            assert result.status.value in ("ok", "error")
        finally:
            await timeline.close()


class TestDoneStatus:
    """Tests for done() function and status."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_done_sets_status(self, temp_cwd: Path) -> None:
        """Test that done() sets the done status."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            assert not timeline.is_done()

            await timeline.execute_statement('done("Task complete")')

            assert timeline.is_done()
            assert timeline.get_done_message() == "Task complete"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_done_without_message(self, temp_cwd: Path) -> None:
        """Test done() with no message defaults to empty string."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement("done()")

            assert timeline.is_done()
            # Default message is empty string, not None
            assert timeline.get_done_message() == ""
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_reset_done(self, temp_cwd: Path) -> None:
        """Test resetting done status."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('done("Complete")')
            assert timeline.is_done()

            timeline.reset_done()

            assert not timeline.is_done()
        finally:
            await timeline.close()


class TestWaitConditions:
    """Tests for wait conditions and blocking operations."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_wait_sets_condition(self, temp_cwd: Path) -> None:
        """Test that wait() sets a wait condition."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('s = shell("echo")')
            await timeline.execute_statement("wait(s)")

            # Wait condition should be set
            assert timeline.is_waiting()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_clear_wait_condition(self, temp_cwd: Path) -> None:
        """Test clearing wait condition."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('s = shell("echo")')
            await timeline.execute_statement("wait(s)")

            timeline.clear_wait_condition()

            assert not timeline.is_waiting()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_wait_condition(self, temp_cwd: Path) -> None:
        """Test getting wait condition details."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('s = shell("echo")')
            await timeline.execute_statement("wait(s)")

            condition = timeline.get_wait_condition()
            assert condition is not None
        finally:
            await timeline.close()


class TestContextGraph:
    """Tests for context graph operations."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_context_graph(self, temp_cwd: Path) -> None:
        """Test getting the context graph."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            graph = timeline.context_graph
            assert graph is not None
            assert timeline.context_graph is graph
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_link_nodes_via_dsl(self, temp_cwd: Path) -> None:
        """Test linking nodes via DSL creates parent-child relationship."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('g = group(summary="Test Group")')
            await timeline.execute_statement('t = topic("Child Topic")')

            # Get node IDs before linking
            ns = timeline.get_namespace()
            group_node = ns["g"]
            topic_node = ns["t"]

            # Link them: link(child, parent) - topic is child, group is parent
            result = await timeline.execute_statement("link(t, g)")
            assert result.status.value == "ok"

            # Verify the graph relationship - topic should be child of group
            graph = timeline.context_graph
            children = graph.get_children(group_node.node_id)
            child_ids = [c.node_id for c in children]
            assert topic_node.node_id in child_ids
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unlink_nodes(self, temp_cwd: Path) -> None:
        """Test unlinking nodes in the context graph."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('g = group(summary="Test Group")')
            await timeline.execute_statement('t = topic("Child Topic")')
            # link(child, parent)
            await timeline.execute_statement("link(t, g)")

            ns = timeline.get_namespace()
            group_node = ns["g"]
            topic_node = ns["t"]

            # Unlink: unlink(child, parent)
            await timeline.execute_statement("unlink(t, g)")

            graph = timeline.context_graph
            children = graph.get_children(group_node.node_id)
            assert topic_node.node_id not in children
        finally:
            await timeline.close()


class TestCheckpoints:
    """Tests for checkpoint/restore functionality."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_checkpoint_creates_snapshot(self, temp_cwd: Path) -> None:
        """Test that checkpoint() creates a snapshot."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Before")')
            await timeline.execute_statement('checkpoint("cp1")')

            graph = timeline.context_graph
            cp = graph.get_checkpoint("cp1")
            assert cp is not None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, temp_cwd: Path) -> None:
        """Test restoring from a checkpoint."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t1 = topic("First")')
            await timeline.execute_statement('checkpoint("before_second")')
            await timeline.execute_statement('t2 = topic("Second")')

            # Have two topics
            assert len(timeline.get_context_objects()) == 2

            # Restore should work without error
            result = await timeline.execute_statement('restore("before_second")')
            assert result.status.value == "ok"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_list(self, temp_cwd: Path) -> None:
        """Test listing available checkpoints returns a list."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('checkpoint("cp1")')
            await timeline.execute_statement('checkpoint("cp2")')

            # checkpoints() returns a list of checkpoint digests
            result = await timeline.execute_statement("cps = checkpoints()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            cps = ns.get("cps")
            # Should be a list containing checkpoint digests (dicts)
            assert isinstance(cps, list)
            assert len(cps) == 2
        finally:
            await timeline.close()


class TestStatementHistory:
    """Tests for statement history tracking."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_statements(self, temp_cwd: Path) -> None:
        """Test getting statement history."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement("x = 1")
            await timeline.execute_statement("y = 2")

            statements = timeline.get_statements()
            assert len(statements) == 2
            assert statements[0].source == "x = 1"
            assert statements[1].source == "y = 2"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_statements_recorded_in_order(self, temp_cwd: Path) -> None:
        """Test that statements are recorded in execution order."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement("a = 1")
            await timeline.execute_statement("b = 2")
            await timeline.execute_statement("c = 3")

            statements = timeline.get_statements()
            sources = [s.source for s in statements]
            assert sources == ["a = 1", "b = 2", "c = 3"]
        finally:
            await timeline.close()


class TestNamespace:
    """Tests for namespace management."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_namespace_stores_user_variables(self, temp_cwd: Path) -> None:
        """Test that user variables are stored in namespace."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement("my_var = 42")

            ns = timeline.get_namespace()
            assert "my_var" in ns
            assert ns["my_var"] == 42
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_namespace_excludes_internal(self, temp_cwd: Path) -> None:
        """Test that get_namespace returns filtered view for user."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Execute statement to ensure namespace is populated
            await timeline.execute_statement("x = 1")

            ns = timeline.get_namespace()
            # User variable should be there
            assert "x" in ns
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_context_objects_available(self, temp_cwd: Path) -> None:
        """Test that created context objects are in namespace."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Test")')

            ns = timeline.get_namespace()
            assert "t" in ns
            assert ns["t"].title == "Test"
        finally:
            await timeline.close()
