"""Integration tests for crosscutting concerns.

These tests verify end-to-end workflows across multiple components.
"""

import asyncio
from pathlib import Path

import pytest

from activecontext import ActiveContext
from activecontext.context.state import NodeState
from activecontext.session.timeline import Timeline


class TestCheckpointRestore:
    """Test checkpoint and restore functionality via Timeline DSL."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self, temp_cwd: Path) -> None:
        """Test creating a checkpoint and restoring edge structure."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create some views
            await timeline.execute_statement('v1 = text("file1.py")')
            await timeline.execute_statement('v2 = text("file2.py")')

            # Create a group linking them
            await timeline.execute_statement("g = group(v1, v2)")

            ns = timeline.get_namespace()
            v1_id = ns["v1"].node_id

            # Verify v1 is linked to g
            assert ns["g"].node_id in ns["v1"].parent_ids

            # Create checkpoint
            result = await timeline.execute_statement('checkpoint("before_changes")')
            assert result.status.value == "ok"

            # Unlink v1 from the group
            await timeline.execute_statement("unlink(v1, g)")

            # Verify v1 is no longer linked to g
            assert ns["g"].node_id not in ns["v1"].parent_ids

            # Restore to checkpoint
            result = await timeline.execute_statement('restore("before_changes")')
            assert result.status.value == "ok"

            # v1 should be linked to g again after restore
            assert ns["g"].node_id in ns["v1"].parent_ids
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, temp_cwd: Path) -> None:
        """Test listing available checkpoints."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create multiple checkpoints
            await timeline.execute_statement('checkpoint("cp1")')
            await timeline.execute_statement('checkpoint("cp2")')

            # List checkpoints
            result = await timeline.execute_statement("checkpoints()")
            assert result.status.value == "ok"
            assert "cp1" in result.stdout
            assert "cp2" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_branch_from_checkpoint(self, temp_cwd: Path) -> None:
        """Test branching (creating checkpoint of current state)."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create a view and checkpoint
            await timeline.execute_statement('v1 = text("original.py")')
            await timeline.execute_statement('checkpoint("base")')

            # Make more changes
            await timeline.execute_statement('v2 = text("extra.py")')

            # Create branch (saves current state as new checkpoint)
            result = await timeline.execute_statement('branch("experiment")')
            assert result.status.value == "ok"

            # Both checkpoints should exist
            result = await timeline.execute_statement("checkpoints()")
            assert "base" in result.stdout
            assert "experiment" in result.stdout
        finally:
            await timeline.close()


class TestWaitFunctionality:
    """Test wait() and related async control flow."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_wait_for_shell(self, temp_cwd: Path) -> None:
        """Test shell command completion and wait condition setup."""
        import sys

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create a quick shell command
            if sys.platform == "win32":
                cmd = 'shell("cmd", ["/c", "echo", "test"])'
            else:
                cmd = 'shell("echo", ["test"])'

            result = await timeline.execute_statement(f"s = {cmd}")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            shell_node = ns["s"]

            # Wait for the actual shell task to complete (async)
            shell_task = timeline._shell_tasks.get(shell_node.node_id)
            if shell_task:
                await asyncio.wait_for(shell_task, timeout=10.0)

            # Process results to update shell node
            timeline.process_pending_shell_results()

            # Shell should be complete
            assert shell_node.is_complete
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_wait_returns_immediately_for_complete(self, temp_cwd: Path) -> None:
        """Test that wait() returns immediately for completed nodes."""
        import sys

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create and wait for shell
            if sys.platform == "win32":
                cmd = 'shell("cmd", ["/c", "echo", "done"])'
            else:
                cmd = 'shell("echo", ["done"])'

            await timeline.execute_statement(f"s = {cmd}")

            # Wait for it to complete
            await asyncio.sleep(0.5)
            timeline.process_pending_shell_results()

            # Wait should return immediately since already complete
            result = await timeline.execute_statement("wait(s)")
            assert result.status.value == "ok"
        finally:
            await timeline.close()


class TestDoneFunction:
    """Test done() agent control function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_done_sets_flag(self, temp_cwd: Path) -> None:
        """Test that done() sets the completion flag."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            assert not timeline.is_done()

            result = await timeline.execute_statement('done("Task completed")')
            assert result.status.value == "ok"

            assert timeline.is_done()
            assert timeline.get_done_message() == "Task completed"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_done_without_message(self, temp_cwd: Path) -> None:
        """Test done() without a message."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("done()")
            assert result.status.value == "ok"

            assert timeline.is_done()
            assert timeline.get_done_message() == ""
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_reset_done(self, temp_cwd: Path) -> None:
        """Test resetting the done flag."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('done("First")')
            assert timeline.is_done()

            timeline.reset_done()
            assert not timeline.is_done()
            assert timeline.get_done_message() is None
        finally:
            await timeline.close()


class TestLinkUnlink:
    """Test link() and unlink() graph manipulation."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_link_view_to_group(self, temp_cwd: Path) -> None:
        """Test linking a view to a group."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create a group and a view
            await timeline.execute_statement("g = group()")
            await timeline.execute_statement('v = text("test.py")')

            # Link view to group
            result = await timeline.execute_statement("link(v, g)")
            assert result.status.value == "ok"

            # Check parent-child relationship using node's parent_ids
            ns = timeline.get_namespace()
            assert ns["g"].node_id in ns["v"].parent_ids
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unlink_removes_connection(self, temp_cwd: Path) -> None:
        """Test unlinking removes the connection."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create linked nodes
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("g = group(v)")

            ns = timeline.get_namespace()

            # Verify initially linked
            assert ns["g"].node_id in ns["v"].parent_ids

            # Unlink
            result = await timeline.execute_statement("unlink(v, g)")
            assert result.status.value == "ok"

            # Verify unlinked
            assert ns["g"].node_id not in ns["v"].parent_ids
        finally:
            await timeline.close()


class TestTopicAndArtifact:
    """Test topic() and artifact() node creation."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_create_topic(self, temp_cwd: Path) -> None:
        """Test creating a topic node."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('t = topic("Authentication")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert "t" in ns

            digest = ns["t"].GetDigest()
            assert digest["type"] == "topic"
            assert digest["title"] == "Authentication"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_create_artifact(self, temp_cwd: Path) -> None:
        """Test creating an artifact node."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            code = "def hello(): pass"
            result = await timeline.execute_statement(
                f'a = artifact("code", content={code!r}, language="python")'
            )
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert "a" in ns

            digest = ns["a"].GetDigest()
            assert digest["type"] == "artifact"
            assert digest["artifact_type"] == "code"
        finally:
            await timeline.close()


class TestSessionLifecycleIntegration:
    """Test full session lifecycle through ActiveContext."""

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self) -> None:
        """Test that multiple sessions are isolated from each other."""
        async with ActiveContext() as ctx:
            session1 = await ctx.create_session(cwd="/tmp")
            session2 = await ctx.create_session(cwd="/tmp")

            # Set different values in each session
            await session1.execute("x = 100")
            await session2.execute("x = 200")

            # Values should be isolated
            ns1 = session1.get_namespace()
            ns2 = session2.get_namespace()

            assert ns1["x"] == 100
            assert ns2["x"] == 200

    @pytest.mark.asyncio
    async def test_session_context_objects_accumulate(self) -> None:
        """Test that context objects accumulate across statements."""
        async with ActiveContext() as ctx:
            session = await ctx.create_session(cwd="/tmp")

            initial_count = len(session.get_context_objects())

            await session.execute('v1 = text("a.py")')
            assert len(session.get_context_objects()) == initial_count + 1

            await session.execute('v2 = text("b.py")')
            assert len(session.get_context_objects()) == initial_count + 2

            await session.execute("g = group(v1, v2)")
            assert len(session.get_context_objects()) == initial_count + 3

    @pytest.mark.asyncio
    async def test_projection_updates_after_execute(self) -> None:
        """Test that projection updates after each execution."""
        async with ActiveContext() as ctx:
            session = await ctx.create_session(cwd="/tmp")

            # Get initial projection
            proj1 = session.get_projection().render()

            # Create a view
            await session.execute('v = text("test.py", tokens=500)')

            # Projection should have changed
            proj2 = session.get_projection().render()
            
            # New projection should contain view info
            assert "test.py" in proj2


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self, temp_cwd: Path) -> None:
        """Test that syntax errors are captured."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("if True print('x')")  # Missing colon
            assert result.status.value == "error"
            assert result.exception is not None
            assert "SyntaxError" in result.exception["type"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_runtime_error_captured(self, temp_cwd: Path) -> None:
        """Test that runtime errors are captured."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("1 / 0")
            assert result.status.value == "error"
            assert result.exception is not None
            assert "ZeroDivisionError" in result.exception["type"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_undefined_variable_error(self, temp_cwd: Path) -> None:
        """Test that undefined variable errors are captured."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("x = undefined_var + 1")
            assert result.status.value == "error"
            assert result.exception is not None
            assert "NameError" in result.exception["type"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_restore_nonexistent_checkpoint_error(self, temp_cwd: Path) -> None:
        """Test that restoring nonexistent checkpoint raises error."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('restore("nonexistent")')
            assert result.status.value == "error"
            assert result.exception is not None
        finally:
            await timeline.close()
