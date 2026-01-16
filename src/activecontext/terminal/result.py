"""Shell execution result dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ShellResult:
    """Result of a shell command execution.

    Attributes:
        command: The command that was executed (including args).
        exit_code: Process exit code (0 = success), or None if killed/timeout.
        output: Combined stdout/stderr output (may be truncated).
        truncated: True if output was truncated due to output_limit.
        status: Execution status - "ok", "error", "timeout", or "killed".
        signal: Signal name if process was killed by signal (e.g., "SIGKILL").
        duration_ms: Execution duration in milliseconds.
    """

    command: str
    exit_code: int | None
    output: str
    truncated: bool
    status: str  # "ok", "error", "timeout", "killed"
    signal: str | None
    duration_ms: float

    @property
    def success(self) -> bool:
        """True if command completed with exit code 0."""
        return self.exit_code == 0

    def __repr__(self) -> str:
        """Concise repr for display in REPL."""
        if self.success:
            lines = self.output.count("\n") + 1 if self.output else 0
            return f"<ShellResult ok, {lines} lines>"
        else:
            return f"<ShellResult {self.status}, exit={self.exit_code}>"
