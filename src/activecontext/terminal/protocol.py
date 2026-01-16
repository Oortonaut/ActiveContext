"""Terminal executor protocol for shell command execution."""

from __future__ import annotations

from typing import Protocol

from activecontext.terminal.result import ShellResult


class TerminalExecutor(Protocol):
    """Protocol for executing shell commands.

    Implementations:
    - SubprocessTerminalExecutor: Local subprocess execution
    - ACPTerminalExecutor: IDE terminal via ACP protocol
    """

    async def execute(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = 30.0,
        output_limit: int = 50000,
    ) -> ShellResult:
        """Execute a shell command.

        Args:
            command: The command to execute (e.g., "pytest", "git").
            args: Optional list of arguments (e.g., ["tests/", "-v"]).
            cwd: Working directory. If None, uses executor's default.
            env: Additional environment variables to set.
            timeout: Timeout in seconds. None means no timeout.
            output_limit: Maximum characters of output to capture.

        Returns:
            ShellResult with exit code, output, and status.
        """
        ...
