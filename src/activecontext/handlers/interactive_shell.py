"""Interactive shell handler for conversation delegation.

Provides an interactive shell session through the ConversationTransport protocol,
allowing users to interact with bash/cmd shells in real-time.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from typing import TYPE_CHECKING, Any

from activecontext.protocols.conversation import InputType

if TYPE_CHECKING:
    from activecontext.protocols.conversation import ConversationTransport


class InteractiveShellHandler:
    """Handler for interactive shell sessions.

    Connects user directly to a shell running in the agent's workspace.
    All stdin/stdout flows through the conversation transport.

    Example:
        >>> handler = InteractiveShellHandler()  # Uses default shell
        >>> result = await session.delegate_conversation(
        ...     handler,
        ...     originator="shell:bash",
        ... )
        >>> print(f"Shell exited with code {result['exit_code']}")

    Platform Behavior:
        - Windows: Uses cmd.exe by default
        - Unix: Uses /bin/bash by default (falls back to /bin/sh)

    Note:
        This implementation uses pipe-based I/O rather than PTY, which means
        some interactive features (readline, ANSI codes) may not work fully.
        For full terminal emulation, consider using a PTY library.
    """

    # Default shells by platform
    _DEFAULT_SHELLS = {
        "win32": ["cmd.exe", "/Q"],  # /Q disables echo
        "darwin": ["/bin/bash", "-i"],  # -i for interactive
        "linux": ["/bin/bash", "-i"],
    }
    _FALLBACK_SHELL = ["/bin/sh"]

    def __init__(
        self,
        shell: str | None = None,
        args: tuple[str, ...] | list[str] = (),
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Initialize the interactive shell handler.

        Args:
            shell: Shell executable path. If None, uses platform default.
            args: Additional arguments for the shell.
            cwd: Working directory for the shell. Defaults to current directory.
            env: Additional environment variables to set.
        """
        self.shell = shell
        self.args = list(args) if args else []
        self.cwd = cwd or os.getcwd()
        self.env = env

    def _get_shell_command(self) -> list[str]:
        """Get the shell command list for the current platform.

        Returns:
            List containing shell executable and its arguments.
        """
        if self.shell:
            return [self.shell, *self.args]

        # Get platform default
        platform = sys.platform
        if platform.startswith("linux"):
            platform = "linux"

        default = self._DEFAULT_SHELLS.get(platform, self._FALLBACK_SHELL)
        return [*default, *self.args]

    async def handle(self, transport: ConversationTransport) -> dict[str, Any]:
        """Handle the interactive shell session.

        Args:
            transport: Communication channel to user.

        Returns:
            Dict containing:
                - exit_code: Process exit code (int or None if killed)
                - shell: Shell command that was run
                - reason: How the session ended ("exit", "cancelled", "error")
        """
        shell_cmd = self._get_shell_command()
        shell_display = " ".join(shell_cmd)

        await transport.send_output(f"Starting interactive shell: {shell_display}\n")
        await transport.send_output("Type 'exit' or press Ctrl+C to end the session.\n")
        await transport.send_output("-" * 40 + "\n")

        # Build environment
        process_env = os.environ.copy()
        if self.env:
            process_env.update(self.env)

        # Disable Python buffering for subprocess
        process_env["PYTHONUNBUFFERED"] = "1"

        try:
            # Create subprocess with pipes
            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
                cwd=self.cwd,
                env=process_env,
            )

            # Track completion
            exit_code: int | None = None
            reason = "exit"

            # Create output streaming task
            async def stream_output() -> None:
                """Stream shell output to the transport."""
                assert process.stdout is not None
                try:
                    while True:
                        # Read in chunks for responsiveness
                        chunk = await process.stdout.read(1024)
                        if not chunk:
                            break
                        text = chunk.decode("utf-8", errors="replace")
                        await transport.send_output(text, append=True)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass  # Output stream closed

            # Start output streaming
            output_task = asyncio.create_task(stream_output())

            try:
                # Main input loop
                while process.returncode is None:
                    # Check if cancelled
                    if transport.check_cancelled():
                        reason = "cancelled"
                        break

                    # Request input from user (blocks until response)
                    try:
                        user_input = await transport.request_input(
                            "",  # Empty prompt - shell provides its own
                            input_type=InputType.TEXT,
                        )
                    except asyncio.CancelledError:
                        reason = "cancelled"
                        break

                    # Check if process already exited while waiting
                    if process.returncode is not None:
                        break

                    # Send input to shell
                    if process.stdin is not None:
                        try:
                            process.stdin.write((user_input + "\n").encode("utf-8"))
                            await process.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            # Process closed stdin
                            break

                    # Give shell time to process and output
                    await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                reason = "cancelled"

            finally:
                # Stop output streaming
                output_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await output_task

                # Terminate process if still running
                if process.returncode is None:
                    try:
                        # Try graceful termination first
                        if sys.platform == "win32":
                            process.terminate()
                        else:
                            process.send_signal(signal.SIGTERM)

                        # Wait briefly for clean exit
                        try:
                            await asyncio.wait_for(process.wait(), timeout=2.0)
                        except asyncio.TimeoutError:
                            # Force kill
                            process.kill()
                            await process.wait()
                    except ProcessLookupError:
                        pass  # Process already gone

                exit_code = process.returncode

        except FileNotFoundError:
            await transport.send_output(f"\nError: Shell not found: {shell_cmd[0]}\n")
            return {
                "exit_code": 127,
                "shell": shell_display,
                "reason": "error",
            }
        except PermissionError:
            await transport.send_output(
                f"\nError: Permission denied: {shell_cmd[0]}\n"
            )
            return {
                "exit_code": 126,
                "shell": shell_display,
                "reason": "error",
            }
        except OSError as e:
            await transport.send_output(f"\nError starting shell: {e}\n")
            return {
                "exit_code": 1,
                "shell": shell_display,
                "reason": "error",
            }

        # Send completion message
        await transport.send_output("-" * 40 + "\n")
        if reason == "cancelled":
            await transport.send_output("Shell session cancelled.\n")
        else:
            await transport.send_output(f"Shell exited with code {exit_code}.\n")

        return {
            "exit_code": exit_code,
            "shell": shell_display,
            "reason": reason,
        }
