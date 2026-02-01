"""PTY (pseudo-terminal) support for interactive commands.

Provides PTY allocation for commands that require a TTY, such as:
- Interactive prompts (ssh, password inputs)
- REPL environments (python, node, ipython)
- Full-screen terminal applications (vim, nano, htop)

Platform support:
- Unix/Linux: Uses built-in pty module
- Windows: Not natively supported (use Windows Terminal or ConPTY API)
- Fallback: Regular subprocess without PTY
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activecontext.terminal.result import ShellResult


@dataclass
class PTYConfig:
    """Configuration for PTY allocation.

    Attributes:
        enabled: Whether to use PTY (auto-detected if None).
        columns: Terminal width in columns.
        rows: Terminal height in rows.
        echo: Whether to echo input.
        raw_mode: Whether to use raw mode (disable line buffering).
    """

    enabled: bool | None = None
    columns: int = 80
    rows: int = 24
    echo: bool = True
    raw_mode: bool = False


def is_pty_supported() -> bool:
    """Check if PTY is supported on this platform.

    Returns:
        True if PTY allocation is supported.
    """
    # Unix/Linux have native pty support
    if sys.platform in ("linux", "darwin"):
        try:
            import pty  # noqa: F401

            return True
        except ImportError:
            return False

    # Windows 10+ has ConPTY, but requires special setup
    if sys.platform == "win32":
        # Check for Windows 10 version 1809 or later
        try:
            version = sys.getwindowsversion()  # type: ignore[attr-defined]
            # ConPTY requires Windows 10 build 17763+
            return version.major >= 10 and version.build >= 17763
        except Exception:
            return False

    return False


async def execute_with_pty(
    command: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    config: PTYConfig | None = None,
) -> ShellResult:
    """Execute a command with PTY allocation.

    Note: This is a platform-dependent feature. On platforms without PTY
    support, this falls back to regular subprocess execution.

    Args:
        command: Command to execute.
        args: Command arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        config: PTY configuration.

    Returns:
        ShellResult with execution details.

    Raises:
        NotImplementedError: If PTY is not supported on this platform.
    """

    cfg = config or PTYConfig()

    # Auto-detect PTY support if not explicitly enabled/disabled
    if cfg.enabled is None:
        cfg.enabled = is_pty_supported()

    if not cfg.enabled:
        # Fallback to regular subprocess
        from activecontext.terminal.subprocess_executor import (
            SubprocessTerminalExecutor,
        )

        executor = SubprocessTerminalExecutor(default_cwd=cwd or ".")
        return await executor.execute(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    # Platform-specific PTY execution
    if sys.platform in ("linux", "darwin"):
        return await _execute_unix_pty(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
            config=cfg,
        )
    elif sys.platform == "win32":
        return await _execute_windows_pty(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
            config=cfg,
        )
    else:
        raise NotImplementedError(f"PTY execution not implemented for platform: {sys.platform}")


async def _execute_unix_pty(
    command: str,
    args: list[str] | None,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: float | None,
    config: PTYConfig,
) -> ShellResult:
    """Execute command with Unix PTY.

    Args:
        command: Command to execute.
        args: Command arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        config: PTY configuration.

    Returns:
        ShellResult.
    """
    import os
    import pty
    import time

    from activecontext.terminal.result import ShellResult

    start_time = time.perf_counter()

    # Build command list
    cmd_list = [command]
    if args:
        cmd_list.extend(args)
    full_command = " ".join(cmd_list)

    # Build environment
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    # Set terminal size
    process_env["COLUMNS"] = str(config.columns)
    process_env["LINES"] = str(config.rows)

    try:
        # Create PTY
        master_fd, slave_fd = pty.openpty()

        # Set terminal size on the PTY
        try:
            import fcntl
            import struct
            import termios

            winsize = struct.pack("HHHH", config.rows, config.columns, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass  # Size setting is best-effort

        # Create subprocess with PTY
        process = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=cwd,
            env=process_env,
        )

        # Close slave_fd in parent (child has its copy)
        os.close(slave_fd)

        # Read output from master
        output_parts: list[bytes] = []
        output_size = 0
        max_output = 50000  # 50KB limit

        try:
            # Set non-blocking
            import fcntl

            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            # Wait for process with timeout
            if timeout:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=timeout,
                )
            else:
                await process.wait()

            # Read any remaining output
            while output_size < max_output:
                try:
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    output_parts.append(chunk)
                    output_size += len(chunk)
                except (BlockingIOError, OSError):
                    break

        except asyncio.TimeoutError:
            # Kill process on timeout
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

            os.close(master_fd)
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ShellResult(
                command=full_command,
                exit_code=None,
                output=f"Command timed out after {timeout}s",
                truncated=False,
                status="timeout",
                signal="SIGKILL",
                duration_ms=duration_ms,
            )

        finally:
            os.close(master_fd)

        # Decode output
        output = b"".join(output_parts).decode("utf-8", errors="replace")
        truncated = output_size >= max_output

        duration_ms = (time.perf_counter() - start_time) * 1000
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

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ShellResult(
            command=full_command,
            exit_code=1,
            output=f"PTY error: {e}",
            truncated=False,
            status="error",
            signal=None,
            duration_ms=duration_ms,
        )


async def _execute_windows_pty(
    command: str,
    args: list[str] | None,
    cwd: str | None,
    env: dict[str, str] | None,
    timeout: float | None,
    config: PTYConfig,
) -> ShellResult:
    """Execute command with Windows ConPTY.

    Note: This is a placeholder. Full ConPTY implementation requires
    ctypes bindings to the Windows Pseudo Console API.

    Args:
        command: Command to execute.
        args: Command arguments.
        cwd: Working directory.
        env: Environment variables.
        timeout: Timeout in seconds.
        config: PTY configuration.

    Returns:
        ShellResult.
    """
    # For now, fallback to regular subprocess on Windows
    # Full ConPTY implementation would use:
    # - CreatePseudoConsole()
    # - ResizePseudoConsole()
    # - ClosePseudoConsole()
    # - EXTENDED_STARTUPINFO_PRESENT flag

    from activecontext.terminal.subprocess_executor import (
        SubprocessTerminalExecutor,
    )

    executor = SubprocessTerminalExecutor(default_cwd=cwd or ".")
    return await executor.execute(
        command=command,
        args=args,
        cwd=cwd,
        env=env,
        timeout=timeout,
    )
