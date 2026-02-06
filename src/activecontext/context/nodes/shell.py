"""ShellNode - Async shell command execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode
from .enums import ShellStatus


@trace_all_fields
@dataclass(kw_only=True)
class ShellNode(ContextNode):
    """Represents an async shell command execution.

    The shell() DSL function creates a ShellNode and starts the subprocess
    in the background. The node's status changes as the command progresses,
    and change notifications propagate up the DAG.

    Attributes:
        command: The command being executed (e.g., "pytest")
        args: Command arguments (e.g., ["-v", "tests/"])
        shell_status: Current execution status (PENDING, RUNNING, COMPLETED, etc.)
        exit_code: Process exit code (None until completed)
        output: Combined stdout/stderr output
        truncated: Whether output was truncated
        signal: Signal name if killed (e.g., "SIGKILL")
        duration_ms: Execution duration in milliseconds
        started_at_exec: When execution actually started (vs node creation)
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    shell_status: ShellStatus = ShellStatus.PENDING
    exit_code: int | None = None
    output: str = ""
    truncated: bool = False
    signal: str | None = None
    duration_ms: float = 0.0
    started_at_exec: float | None = None

    @property
    def is_complete(self) -> bool:
        """True if shell command has finished (success, failure, timeout, or cancelled)."""
        return self.shell_status in (
            ShellStatus.COMPLETED,
            ShellStatus.FAILED,
            ShellStatus.TIMEOUT,
            ShellStatus.CANCELLED,
        )

    @property
    def is_success(self) -> bool:
        """True if shell command completed successfully."""
        return self.shell_status == ShellStatus.COMPLETED and self.exit_code == 0

    @property
    def error(self) -> str | None:
        """Error message if command failed, None otherwise."""
        if self.shell_status == ShellStatus.FAILED:
            return f"Exit code {self.exit_code}"
        if self.shell_status == ShellStatus.TIMEOUT:
            return "Command timed out"
        if self.shell_status == ShellStatus.CANCELLED:
            return "Command cancelled"
        return None

    @property
    def full_command(self) -> str:
        """Full command string with arguments."""
        if self.args:
            return f"{self.command} {' '.join(self.args)}"
        return self.command

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "command": self.full_command,
            "status": self.shell_status.value,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render full output with timing details."""
        result = self.output
        if result and not result.endswith("\n"):
            result += "\n"

        result += f"--- Duration: {self.duration_ms:.0f}ms"
        if self.truncated:
            result += " (output was truncated)"
        if self.signal:
            result += f", killed by {self.signal}"
        result += " ---\n"

        return result

    def set_running(self) -> ShellNode:
        """Mark as running (called when subprocess starts)."""
        old_status = self.shell_status
        self.shell_status = ShellStatus.RUNNING
        self.started_at_exec = time.time()
        self.mark_changed(f"Shell: {old_status.value} → running")
        return self

    def set_completed(
        self,
        exit_code: int,
        output: str,
        duration_ms: float,
        truncated: bool = False,
        signal: str | None = None,
    ) -> ShellNode:
        """Mark as completed with result (called when subprocess finishes).

        Args:
            exit_code: Process exit code (0 = success).
            output: Captured stdout/stderr output.
            duration_ms: Execution time in milliseconds.
            truncated: Whether output was truncated.
            signal: Signal name if killed (e.g., "SIGTERM").
        """
        self.exit_code = exit_code
        self.output = output
        self.duration_ms = duration_ms
        self.truncated = truncated
        self.signal = signal

        if signal:
            self.shell_status = ShellStatus.CANCELLED
        elif exit_code == 0:
            self.shell_status = ShellStatus.COMPLETED
        else:
            self.shell_status = ShellStatus.FAILED

        self.mark_changed(
            f"Shell '{self.command}' {self.shell_status.value} (exit={exit_code})",
            content=output[:500] if output else None,
        )
        return self

    def set_timeout(self, output: str, duration_ms: float) -> ShellNode:
        """Mark as timed out.

        Args:
            output: Partial output captured before timeout.
            duration_ms: Time elapsed before timeout in milliseconds.
        """
        self.shell_status = ShellStatus.TIMEOUT
        self.output = output
        self.duration_ms = duration_ms
        self.exit_code = -1
        self.mark_changed(
            f"Shell '{self.command}' timed out after {duration_ms:.0f}ms",
        )
        return self

    def set_cancelled(self) -> ShellNode:
        """Mark as cancelled by user."""
        self.shell_status = ShellStatus.CANCELLED
        self.mark_changed(
            f"Shell '{self.command}' cancelled",
        )
        return self

    def render_digest(self) -> str:
        """Return 'Shell: command [STATUS]' format."""
        cmd_display = (
            self.full_command[:40] + "..." if len(self.full_command) > 40 else self.full_command
        )
        return f"Shell: {cmd_display} [{self.shell_status.value.upper()}]"

    def get_wake_data(self) -> dict[str, Any]:
        """Return data dict for wake prompt template formatting."""
        return {
            "node_id": self.node_id,
            "command": self.full_command,
            "exit_code": self.exit_code,
            "output": self.output[:500] if self.output else "",
            "status": self.shell_status.value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize ShellNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "command": self.command,
                "args": self.args,
                "shell_status": self.shell_status.value,
                "exit_code": self.exit_code,
                "output": self.output,
                "truncated": self.truncated,
                "signal": self.signal,
                "duration_ms": self.duration_ms,
                "started_at_exec": self.started_at_exec,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ShellNode:
        """Deserialize ShellNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            command=data.get("command", ""),
            args=data.get("args", []),
            shell_status=ShellStatus(data.get("shell_status", "pending")),
            exit_code=data.get("exit_code"),
            output=data.get("output", ""),
            truncated=data.get("truncated", False),
            signal=data.get("signal"),
            duration_ms=data.get("duration_ms", 0.0),
            started_at_exec=data.get("started_at_exec"),
        )
        return node
