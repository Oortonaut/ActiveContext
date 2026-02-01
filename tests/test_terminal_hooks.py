"""Tests for terminal hook system."""

from __future__ import annotations

import pytest

from activecontext.terminal.hooks import (
    HookManager,
    HookPhase,
    PostCommandContext,
    PreCommandContext,
    StateChangeContext,
    TickBoundaryContext,
    get_hook_manager,
    log_command_hook,
    log_result_hook,
)


class TestPreCommandContext:
    """Tests for PreCommandContext."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ctx = PreCommandContext(
            command="pytest",
            args=["-xvs"],
            cwd="/project",
            env={"DEBUG": "1"},
            timeout=30.0,
            node_id="shell_1",
        )

        d = ctx.to_dict()
        assert d["command"] == "pytest"
        assert d["args"] == ["-xvs"]
        assert d["cwd"] == "/project"
        assert d["env"] == {"DEBUG": "1"}
        assert d["timeout"] == 30.0
        assert d["node_id"] == "shell_1"


class TestPostCommandContext:
    """Tests for PostCommandContext."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ctx = PostCommandContext(
            command="pytest -xvs",
            exit_code=0,
            output="All tests passed",
            truncated=False,
            status="ok",
            duration_ms=1234.5,
            node_id="shell_1",
        )

        d = ctx.to_dict()
        assert d["command"] == "pytest -xvs"
        assert d["exit_code"] == 0
        assert d["output"] == "All tests passed"
        assert d["truncated"] is False
        assert d["status"] == "ok"
        assert d["duration_ms"] == 1234.5
        assert d["node_id"] == "shell_1"


class TestStateChangeContext:
    """Tests for StateChangeContext."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ctx = StateChangeContext(
            node_id="shell_1",
            old_state="queued",
            new_state="running",
            timestamp=1234567890.0,
        )

        d = ctx.to_dict()
        assert d["node_id"] == "shell_1"
        assert d["old_state"] == "queued"
        assert d["new_state"] == "running"
        assert d["timestamp"] == 1234567890.0


class TestTickBoundaryContext:
    """Tests for TickBoundaryContext."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        ctx = TickBoundaryContext(
            phase="before",
            tick_count=42,
            session_id="session_123",
        )

        d = ctx.to_dict()
        assert d["phase"] == "before"
        assert d["tick_count"] == 42
        assert d["session_id"] == "session_123"


class TestHookManager:
    """Tests for HookManager."""

    def test_register_and_unregister(self) -> None:
        """Test hook registration and unregistration."""
        manager = HookManager()

        def test_hook(ctx):
            return ctx

        # Register
        manager.register(HookPhase.PRE_COMMAND, test_hook)
        assert manager.count_hooks(HookPhase.PRE_COMMAND) == 1

        # Unregister
        assert manager.unregister(HookPhase.PRE_COMMAND, test_hook) is True
        assert manager.count_hooks(HookPhase.PRE_COMMAND) == 0

        # Unregister non-existent
        assert manager.unregister(HookPhase.PRE_COMMAND, test_hook) is False

    def test_invalid_phase_raises(self) -> None:
        """Test that invalid phase raises ValueError."""
        manager = HookManager()

        with pytest.raises(ValueError, match="Invalid hook phase"):
            manager.register("invalid_phase", lambda x: x)  # type: ignore[arg-type]

    def test_run_pre_command_hooks(self) -> None:
        """Test running pre-command hooks."""
        manager = HookManager()
        call_order = []

        def hook1(ctx):
            call_order.append("hook1")
            return ctx

        def hook2(ctx):
            call_order.append("hook2")
            ctx.timeout = 60.0  # Modify context
            return ctx

        manager.register(HookPhase.PRE_COMMAND, hook1)
        manager.register(HookPhase.PRE_COMMAND, hook2)

        ctx = PreCommandContext(
            command="test",
            args=[],
            cwd=None,
            env=None,
            timeout=30.0,
            node_id=None,
        )

        result = manager.run_pre_command(ctx)
        assert result is not None
        assert result.timeout == 60.0
        assert call_order == ["hook1", "hook2"]

    def test_pre_command_hook_can_cancel(self) -> None:
        """Test that pre-command hook can cancel execution."""
        manager = HookManager()

        def blocking_hook(ctx):
            return None  # Cancel execution

        manager.register(HookPhase.PRE_COMMAND, blocking_hook)

        ctx = PreCommandContext(
            command="test",
            args=[],
            cwd=None,
            env=None,
            timeout=30.0,
            node_id=None,
        )

        result = manager.run_pre_command(ctx)
        assert result is None

    def test_run_post_command_hooks(self) -> None:
        """Test running post-command hooks."""
        manager = HookManager()

        def modify_output(ctx):
            ctx.output = ctx.output + " [modified]"
            return ctx

        manager.register(HookPhase.POST_COMMAND, modify_output)

        ctx = PostCommandContext(
            command="test",
            exit_code=0,
            output="original",
            truncated=False,
            status="ok",
            duration_ms=100.0,
            node_id=None,
        )

        result = manager.run_post_command(ctx)
        assert result.output == "original [modified]"

    def test_run_state_change_hooks(self) -> None:
        """Test running state change hooks."""
        manager = HookManager()
        called = []

        def track_change(ctx):
            called.append((ctx.old_state, ctx.new_state))

        manager.register(HookPhase.STATE_CHANGE, track_change)

        ctx = StateChangeContext(
            node_id="test",
            old_state="idle",
            new_state="running",
            timestamp=0.0,
        )

        manager.run_state_change(ctx)
        assert called == [("idle", "running")]

    def test_state_change_hook_errors_ignored(self) -> None:
        """Test that state change hook errors don't break execution."""
        manager = HookManager()

        def failing_hook(ctx):
            raise RuntimeError("Hook failed")

        manager.register(HookPhase.STATE_CHANGE, failing_hook)

        ctx = StateChangeContext(
            node_id="test",
            old_state="idle",
            new_state="running",
            timestamp=0.0,
        )

        # Should not raise
        manager.run_state_change(ctx)

    def test_run_tick_boundary_hooks(self) -> None:
        """Test running tick boundary hooks."""
        manager = HookManager()
        ticks = []

        def track_tick(ctx):
            ticks.append((ctx.phase, ctx.tick_count))

        manager.register(HookPhase.TICK_BOUNDARY, track_tick)

        ctx = TickBoundaryContext(
            phase="before",
            tick_count=1,
            session_id="test",
        )

        manager.run_tick_boundary(ctx)
        assert ticks == [("before", 1)]

    def test_tick_boundary_hook_errors_ignored(self) -> None:
        """Test that tick boundary hook errors don't break execution."""
        manager = HookManager()

        def failing_hook(ctx):
            raise RuntimeError("Hook failed")

        manager.register(HookPhase.TICK_BOUNDARY, failing_hook)

        ctx = TickBoundaryContext(
            phase="before",
            tick_count=1,
            session_id="test",
        )

        # Should not raise
        manager.run_tick_boundary(ctx)

    def test_clear_hooks(self) -> None:
        """Test clearing hooks."""
        manager = HookManager()

        def hook1(ctx):
            return ctx

        def hook2(ctx):
            return ctx

        manager.register(HookPhase.PRE_COMMAND, hook1)
        manager.register(HookPhase.POST_COMMAND, hook2)

        # Clear specific phase
        manager.clear(HookPhase.PRE_COMMAND)
        assert manager.count_hooks(HookPhase.PRE_COMMAND) == 0
        assert manager.count_hooks(HookPhase.POST_COMMAND) == 1

        # Clear all
        manager.clear()
        assert manager.count_hooks(HookPhase.POST_COMMAND) == 0

    def test_count_hooks(self) -> None:
        """Test counting hooks."""
        manager = HookManager()

        def hook(ctx):
            return ctx

        assert manager.count_hooks(HookPhase.PRE_COMMAND) == 0

        manager.register(HookPhase.PRE_COMMAND, hook)
        assert manager.count_hooks(HookPhase.PRE_COMMAND) == 1

        manager.register(HookPhase.PRE_COMMAND, hook)
        assert manager.count_hooks(HookPhase.PRE_COMMAND) == 2


class TestGlobalHookManager:
    """Tests for global hook manager."""

    def test_get_hook_manager_singleton(self) -> None:
        """Test that get_hook_manager returns singleton."""
        manager1 = get_hook_manager()
        manager2 = get_hook_manager()
        assert manager1 is manager2


class TestBuiltinHooks:
    """Tests for built-in hook functions."""

    def test_log_command_hook(self) -> None:
        """Test log_command_hook returns unmodified context."""
        ctx = PreCommandContext(
            command="test",
            args=["arg"],
            cwd=None,
            env=None,
            timeout=30.0,
            node_id=None,
        )

        result = log_command_hook(ctx)
        assert result is ctx

    def test_log_result_hook(self) -> None:
        """Test log_result_hook returns unmodified context."""
        ctx = PostCommandContext(
            command="test",
            exit_code=0,
            output="output",
            truncated=False,
            status="ok",
            duration_ms=100.0,
            node_id=None,
        )

        result = log_result_hook(ctx)
        assert result is ctx
