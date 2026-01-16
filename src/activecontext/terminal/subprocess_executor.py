"""Subprocess-based terminal executor for local shell execution."""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING

from activecontext.terminal.result import ShellResult

if TYPE_CHECKING:
    pass


class SubprocessTerminalExecutor:
    """Execute shell commands using asyncio subprocess.

    This is the fallback executor used when ACP terminal is not available
    (e.g., when using the Direct transport API).
    """

    def __init__(self, default_cwd: str = ".") -> None:
        """Initialize the subprocess executor.

        Args:
            default_cwd: Default working directory for commands.
        """
        self._default_cwd = default_cwd

    async def execute(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = 30.0,
        output_limit: int = 50000,
    ) -> ShellResult:
        """Execute a shell command using asyncio subprocess.

        Args:
            command: The command to execute.
            args: Optional list of arguments.
            cwd: Working directory. Uses default_cwd if None.
            env: Additional environment variables.
            timeout: Timeout in seconds. None for no timeout.
            output_limit: Maximum characters of output to capture.

        Returns:
            ShellResult with execution details.
        """
        start_time = time.perf_counter()

        # Build command list
        cmd_list = [command]
        if args:
            cmd_list.extend(args)

        # Build full command string for result
        full_command = " ".join(cmd_list)

        # Use provided cwd or default
        working_dir = cwd or self._default_cwd

        # Build environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                cwd=working_dir,
                env=process_env,
            )

            # Wait for completion with optional timeout
            try:
                if timeout is not None:
                    stdout_data, _ = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout,
                    )
                else:
                    stdout_data, _ = await process.communicate()

                # Process completed
                duration_ms = (time.perf_counter() - start_time) * 1000
                output = stdout_data.decode("utf-8", errors="replace")

                # Check if output needs truncation
                truncated = len(output) > output_limit
                if truncated:
                    output = output[:output_limit] + "\n... (output truncated)"

                # Determine status
                exit_code = process.returncode
                status = "ok" if exit_code == 0 else "error"

                return ShellResult(
                    command=full_command,
                    exit_code=exit_code,
                    output=output,
                    truncated=truncated,
                    status=status,
                    signal=None,
                    duration_ms=duration_ms,
                )

            except asyncio.TimeoutError:
                # Kill the process on timeout
                duration_ms = (time.perf_counter() - start_time) * 1000

                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass  # Process already gone

                return ShellResult(
                    command=full_command,
                    exit_code=None,
                    output=f"Command timed out after {timeout}s",
                    truncated=False,
                    status="timeout",
                    signal="SIGKILL",
                    duration_ms=duration_ms,
                )

        except FileNotFoundError:
            # Command not found
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult(
                command=full_command,
                exit_code=127,  # Standard "command not found" exit code
                output=f"Command not found: {command}",
                truncated=False,
                status="error",
                signal=None,
                duration_ms=duration_ms,
            )

        except PermissionError:
            # Permission denied
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult(
                command=full_command,
                exit_code=126,  # Standard "permission denied" exit code
                output=f"Permission denied: {command}",
                truncated=False,
                status="error",
                signal=None,
                duration_ms=duration_ms,
            )

        except OSError as e:
            # Other OS errors
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ShellResult(
                command=full_command,
                exit_code=1,
                output=f"OS error: {e}",
                truncated=False,
                status="error",
                signal=None,
                duration_ms=duration_ms,
            )
