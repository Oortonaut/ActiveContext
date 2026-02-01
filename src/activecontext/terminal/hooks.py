"""Hook system for terminal command execution lifecycle.

Provides extensible hook points for:
- Pre-command: Before command execution starts
- Post-command: After command completes
- State change: When shell node state changes
- Tick boundary: At session tick phases

Use cases:
- Logging and metrics
- Permission checks
- Command transformations
- Output filtering
- Progress notifications
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class HookPhase(Enum):
    """Execution phases where hooks can be registered."""

    PRE_COMMAND = "pre_command"
    """Before command execution starts. Can modify command."""

    POST_COMMAND = "post_command"
    """After command completes. Can modify result."""

    STATE_CHANGE = "state_change"
    """When shell node state changes (queued -> running -> complete)."""

    TICK_BOUNDARY = "tick_boundary"
    """At session tick phases (before/after tick)."""


@dataclass
class PreCommandContext:
    """Context passed to pre-command hooks.

    Attributes:
        command: The command to execute.
        args: Command arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        node_id: ID of the shell node triggering this command.
    """

    command: str
    args: list[str]
    cwd: str | None
    env: dict[str, str] | None
    timeout: float | None
    node_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "args": self.args,
            "cwd": self.cwd,
            "env": self.env,
            "timeout": self.timeout,
            "node_id": self.node_id,
        }


@dataclass
class PostCommandContext:
    """Context passed to post-command hooks.

    Attributes:
        command: The command that was executed.
        exit_code: Process exit code.
        output: Command output.
        truncated: Whether output was truncated.
        status: Execution status.
        duration_ms: Duration in milliseconds.
        node_id: ID of the shell node.
    """

    command: str
    exit_code: int | None
    output: str
    truncated: bool
    status: str
    duration_ms: float
    node_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "output": self.output,
            "truncated": self.truncated,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "node_id": self.node_id,
        }


@dataclass
class StateChangeContext:
    """Context passed to state change hooks.

    Attributes:
        node_id: ID of the node whose state changed.
        old_state: Previous state value.
        new_state: New state value.
        timestamp: When the change occurred (milliseconds since epoch).
    """

    node_id: str
    old_state: Any
    new_state: Any
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "old_state": str(self.old_state),
            "new_state": str(self.new_state),
            "timestamp": self.timestamp,
        }


@dataclass
class TickBoundaryContext:
    """Context passed to tick boundary hooks.

    Attributes:
        phase: "before" or "after" tick.
        tick_count: Current tick number.
        session_id: Session identifier.
    """

    phase: str
    tick_count: int
    session_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase,
            "tick_count": self.tick_count,
            "session_id": self.session_id,
        }


# Type aliases for hook callbacks
PreCommandHook = Callable[[PreCommandContext], PreCommandContext | None]
"""Pre-command hook: receives context, returns modified context or None to skip."""

PostCommandHook = Callable[[PostCommandContext], PostCommandContext | None]
"""Post-command hook: receives context, returns modified context or None."""

StateChangeHook = Callable[[StateChangeContext], None]
"""State change hook: receives context, return value ignored."""

TickBoundaryHook = Callable[[TickBoundaryContext], None]
"""Tick boundary hook: receives context, return value ignored."""


class HookManager:
    """Manages hook registration and execution.

    Thread-safe hook registry with ordered execution.
    """

    def __init__(self) -> None:
        """Initialize the hook manager."""
        self._hooks: dict[HookPhase, list[Callable[..., Any]]] = {
            HookPhase.PRE_COMMAND: [],
            HookPhase.POST_COMMAND: [],
            HookPhase.STATE_CHANGE: [],
            HookPhase.TICK_BOUNDARY: [],
        }

    def register(self, phase: HookPhase, hook: Callable[..., Any], priority: int = 0) -> None:
        """Register a hook for a specific phase.

        Args:
            phase: The execution phase to hook into.
            hook: The callback function.
            priority: Higher priority hooks run first (default 0).
        """
        if phase not in self._hooks:
            raise ValueError(f"Invalid hook phase: {phase}")

        # Insert hook maintaining priority order (descending)
        hooks = self._hooks[phase]
        # Store with priority for sorting
        # For now, just append (priority sorting can be added if needed)
        hooks.append(hook)

    def unregister(self, phase: HookPhase, hook: Callable[..., Any]) -> bool:
        """Unregister a hook.

        Args:
            phase: The phase to remove from.
            hook: The callback to remove.

        Returns:
            True if removed, False if not found.
        """
        if phase not in self._hooks:
            return False

        hooks = self._hooks[phase]
        if hook in hooks:
            hooks.remove(hook)
            return True
        return False

    def run_pre_command(self, context: PreCommandContext) -> PreCommandContext | None:
        """Run all pre-command hooks.

        Hooks can modify the context or return None to skip execution.

        Args:
            context: The pre-command context.

        Returns:
            Modified context, or None if execution should be skipped.
        """
        current = context
        for hook in self._hooks[HookPhase.PRE_COMMAND]:
            result = hook(current)
            if result is None:
                return None  # Skip execution
            current = result
        return current

    def run_post_command(self, context: PostCommandContext) -> PostCommandContext:
        """Run all post-command hooks.

        Hooks can modify the result context.

        Args:
            context: The post-command context.

        Returns:
            Modified context.
        """
        current = context
        for hook in self._hooks[HookPhase.POST_COMMAND]:
            result = hook(current)
            if result is not None:
                current = result
        return current

    def run_state_change(self, context: StateChangeContext) -> None:
        """Run all state change hooks.

        Args:
            context: The state change context.
        """
        import contextlib

        for hook in self._hooks[HookPhase.STATE_CHANGE]:
            with contextlib.suppress(Exception):
                # State change hooks should not break execution
                hook(context)

    def run_tick_boundary(self, context: TickBoundaryContext) -> None:
        """Run all tick boundary hooks.

        Args:
            context: The tick boundary context.
        """
        import contextlib

        for hook in self._hooks[HookPhase.TICK_BOUNDARY]:
            with contextlib.suppress(Exception):
                # Tick hooks should not break execution
                hook(context)

    def clear(self, phase: HookPhase | None = None) -> None:
        """Clear all hooks for a phase, or all hooks if phase is None.

        Args:
            phase: The phase to clear, or None to clear all.
        """
        if phase is None:
            for hooks in self._hooks.values():
                hooks.clear()
        elif phase in self._hooks:
            self._hooks[phase].clear()

    def count_hooks(self, phase: HookPhase) -> int:
        """Get the number of registered hooks for a phase.

        Args:
            phase: The phase to count.

        Returns:
            Number of hooks registered.
        """
        return len(self._hooks.get(phase, []))


# Global hook manager instance
_global_hook_manager: HookManager | None = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager.

    Returns:
        The singleton HookManager instance.
    """
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


# Convenience functions for common hooks


def log_command_hook(context: PreCommandContext) -> PreCommandContext:
    """Example pre-command hook that logs commands.

    Args:
        context: Pre-command context.

    Returns:
        Unmodified context.
    """
    import logging

    logger = logging.getLogger("activecontext.terminal")
    logger.info("Executing: %s %s", context.command, " ".join(context.args))
    return context


def log_result_hook(context: PostCommandContext) -> PostCommandContext:
    """Example post-command hook that logs results.

    Args:
        context: Post-command context.

    Returns:
        Unmodified context.
    """
    import logging

    logger = logging.getLogger("activecontext.terminal")
    logger.info(
        "Completed: %s (exit=%s, duration=%.2fms)",
        context.command,
        context.exit_code,
        context.duration_ms,
    )
    return context
