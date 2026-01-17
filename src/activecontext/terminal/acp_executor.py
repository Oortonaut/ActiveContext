"""ACP-based terminal executor for IDE terminal execution."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

from activecontext.terminal.result import ShellResult

if TYPE_CHECKING:
    from acp.interfaces import Client


class ACPTerminalExecutor:
    """Execute shell commands using ACP terminal/* methods.

    This executor runs commands in the IDE's integrated terminal,
    providing a better user experience when running through ACP.
    """

    def __init__(
        self,
        client: Client,
        session_id: str,
        default_cwd: str = ".",
    ) -> None:
        """Initialize the ACP terminal executor.

        Args:
            client: ACP client connection for terminal methods.
            session_id: Session ID for terminal association.
            default_cwd: Default working directory for commands.
        """
        self._client = client
        self._session_id = session_id
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
        """Execute a shell command using ACP terminal.

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

        # Build full command string
        cmd_list = [command]
        if args:
            cmd_list.extend(args)
        full_command = " ".join(cmd_list)

        # Use provided cwd or default
        working_dir = cwd or self._default_cwd

        terminal_id: str | None = None

        try:
            # Create terminal and run command
            terminal_response = await self._client.create_terminal(
                session_id=self._session_id,
                command=full_command,
                cwd=working_dir,
                env=env,  # type: ignore[arg-type]
            )
            terminal_id = terminal_response.terminal_id

            # Wait for completion with optional timeout
            try:
                if timeout is not None:
                    exit_response = await asyncio.wait_for(
                        self._client.wait_for_terminal_exit(terminal_id),  # type: ignore[call-arg]
                        timeout=timeout,
                    )
                else:
                    exit_response = await self._client.wait_for_terminal_exit(
                        terminal_id  # type: ignore[call-arg]
                    )

                # Get terminal output
                output_response = await self._client.terminal_output(terminal_id)  # type: ignore[call-arg]
                output = output_response.output

                # Check if output needs truncation
                truncated = len(output) > output_limit
                if truncated:
                    output = output[:output_limit] + "\n... (output truncated)"

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Determine status
                exit_code = exit_response.exit_code
                signal_name = getattr(exit_response, "signal", None)

                if signal_name:
                    status = "killed"
                elif exit_code == 0:
                    status = "ok"
                else:
                    status = "error"

                return ShellResult(
                    command=full_command,
                    exit_code=exit_code,
                    output=output,
                    truncated=truncated,
                    status=status,
                    signal=signal_name,
                    duration_ms=duration_ms,
                )

            except asyncio.TimeoutError:
                # Kill the terminal on timeout
                duration_ms = (time.perf_counter() - start_time) * 1000

                with contextlib.suppress(Exception):
                    await self._client.kill_terminal(terminal_id)  # type: ignore[call-arg]

                return ShellResult(
                    command=full_command,
                    exit_code=None,
                    output=f"Command timed out after {timeout}s",
                    truncated=False,
                    status="timeout",
                    signal="SIGKILL",
                    duration_ms=duration_ms,
                )

        except Exception as e:
            # Handle ACP errors
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Check for common error patterns
            error_msg = str(e).lower()
            if "not found" in error_msg or "no such" in error_msg:
                exit_code = 127
            elif "permission" in error_msg:
                exit_code = 126
            else:
                exit_code = 1

            return ShellResult(
                command=full_command,
                exit_code=exit_code,
                output=f"Terminal error: {e}",
                truncated=False,
                status="error",
                signal=None,
                duration_ms=duration_ms,
            )

        finally:
            # Always try to release the terminal
            if terminal_id:
                with contextlib.suppress(Exception):
                    await self._client.release_terminal(terminal_id)  # type: ignore[call-arg]
