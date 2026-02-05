"""PTY backend protocol and platform implementations.

Provides a long-lived, interactive pseudo-terminal that the agent can
drive by sending input and reading output incrementally.

Platform support:
- Unix/Linux/macOS: ``pty.openpty()`` + async subprocess
- Windows 10+: ``pywinpty`` (ConPTY wrapper used by Jupyter/VS Code)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PtyBackend(Protocol):
    """Async interface to a single interactive PTY session."""

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Start the child process inside the PTY.

        Must be called exactly once.  After ``spawn`` returns the backend
        is ready for ``read`` / ``write`` calls.
        """
        ...

    async def read(self, size: int = 4096) -> bytes:
        """Read up to *size* bytes of output.

        Returns ``b""`` on EOF (child exited and pipe drained).
        """
        ...

    def write(self, data: bytes) -> None:
        """Write *data* to the child's stdin.  Non-blocking."""
        ...

    @property
    def is_alive(self) -> bool:
        """True while the child process is still running."""
        ...

    async def wait(self) -> int:
        """Block until the child exits and return its exit code."""
        ...

    def terminate(self) -> None:
        """Send a graceful termination signal (SIGTERM / Ctrl-C)."""
        ...

    def kill(self) -> None:
        """Force-kill the child (SIGKILL / TerminateProcess)."""
        ...

    def resize(self, columns: int, rows: int) -> None:
        """Update the PTY window size (best-effort)."""
        ...

    def close(self) -> None:
        """Release file descriptors / handles.  Idempotent."""
        ...


# ---------------------------------------------------------------------------
# Unix implementation
# ---------------------------------------------------------------------------


class UnixPtyBackend:
    """PTY backend using ``pty.openpty()`` (Linux / macOS)."""

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self._columns = columns
        self._rows = rows
        self._master_fd: int | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._closed = False
        self._reader: asyncio.StreamReader | None = None
        self._read_transport: Any = None

    # -- spawn ---------------------------------------------------------------

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        import pty

        master_fd, slave_fd = pty.openpty()  # type: ignore[attr-defined]
        self._master_fd = master_fd

        # Set terminal size
        self._set_winsize(master_fd, self._rows, self._columns)

        # Build env
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        process_env["COLUMNS"] = str(self._columns)
        process_env["LINES"] = str(self._rows)
        process_env["TERM"] = process_env.get("TERM", "xterm-256color")

        cmd_list = [command] + (args or [])
        self._process = await asyncio.create_subprocess_exec(
            *cmd_list,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=cwd,
            env=process_env,
        )

        # Close slave in parent — child inherited it.
        os.close(slave_fd)

        # Wrap master_fd in an asyncio StreamReader for non-blocking reads.
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        self._read_transport, _ = await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(master_fd, "rb", 0)
        )
        self._reader = reader

    # -- read / write --------------------------------------------------------

    async def read(self, size: int = 4096) -> bytes:
        if self._reader is None:
            return b""
        try:
            data = await self._reader.read(size)
            return data
        except (OSError, asyncio.IncompleteReadError):
            return b""

    def write(self, data: bytes) -> None:
        if self._master_fd is None or self._closed:
            return
        with contextlib.suppress(OSError):
            os.write(self._master_fd, data)

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        if self._process is None:
            return False
        return self._process.returncode is None

    async def wait(self) -> int:
        if self._process is None:
            return -1
        return await self._process.wait()

    def terminate(self) -> None:
        if self._process and self.is_alive:
            self._process.terminate()

    def kill(self) -> None:
        if self._process and self.is_alive:
            self._process.kill()

    def resize(self, columns: int, rows: int) -> None:
        self._columns = columns
        self._rows = rows
        if self._master_fd is not None and not self._closed:
            self._set_winsize(self._master_fd, rows, columns)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._read_transport is not None:
            self._read_transport.close()
            self._read_transport = None
        # master_fd is owned by the transport now; closing the transport
        # closes the underlying fd.  Set to None to prevent double-close.
        self._master_fd = None
        if self._process and self.is_alive:
            self._process.kill()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int) -> None:
        try:
            import fcntl
            import struct
            import termios

            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)  # type: ignore[attr-defined]
        except Exception:
            pass




# ---------------------------------------------------------------------------
# Windows pywinpty implementation (preferred)
# ---------------------------------------------------------------------------


def _has_pywinpty() -> bool:
    """Check if pywinpty is importable."""
    try:
        from winpty import PtyProcess  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


class WinPtyBackend:
    """PTY backend using ``pywinpty`` (the battle-tested ConPTY wrapper).

    ``pywinpty`` is used by Jupyter, VS Code, and other major tools on
    Windows.  It handles all the ConPTY setup correctly and is significantly
    more reliable than raw ctypes bindings.

    Requires: ``pip install pywinpty``
    """

    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self._columns = columns
        self._rows = rows
        self._closed = False
        self._proc: Any = None  # winpty.PtyProcess
        self._exit_code: int | None = None

    async def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        from winpty import PtyProcess

        cmd_list = [command] + (args or [])

        # Build environment — pywinpty needs the full env dict or None.
        process_env: dict[str, str] | None = None
        if env:
            process_env = os.environ.copy()
            process_env.update(env)

        # Build full command line for spawn
        # pywinpty's PtyProcess.spawn() expects argv as a list or a
        # command string.  It handles ConPTY setup internally.
        self._proc = PtyProcess.spawn(
            cmd_list,
            cwd=cwd,
            env=process_env,
            dimensions=(self._rows, self._columns),
        )

    async def read(self, size: int = 4096) -> bytes:
        if self._closed or self._proc is None:
            return b""

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, self._blocking_read, size)
            return text.encode("utf-8", errors="replace") if text else b""
        except EOFError:
            return b""
        except OSError:
            return b""

    def _blocking_read(self, size: int) -> str:
        """Blocking read — runs in executor thread.

        pywinpty's ``read()`` blocks until data is available or the PTY
        is closed (EOFError).  We call it directly — no polling needed.
        """
        if self._proc is None:
            return ""
        try:
            return self._proc.read(size)  # type: ignore[no-any-return]
        except EOFError:
            raise

    def write(self, data: bytes) -> None:
        if self._closed or self._proc is None:
            return
        with contextlib.suppress(OSError, EOFError):
            self._proc.write(data.decode("utf-8", errors="replace"))

    @property
    def is_alive(self) -> bool:
        if self._proc is None or self._closed:
            return False
        return self._proc.isalive()  # type: ignore[no-any-return]

    async def wait(self) -> int:
        if self._proc is None:
            return self._exit_code if self._exit_code is not None else -1

        loop = asyncio.get_running_loop()
        # pywinpty's wait() blocks until process exits
        exit_ok = await loop.run_in_executor(None, self._proc.wait)
        self._exit_code = self._proc.exitstatus
        if self._exit_code is None:
            self._exit_code = 0 if exit_ok else 1
        return self._exit_code

    def terminate(self) -> None:
        """Graceful termination: try Ctrl-C, then force-kill."""
        if self._proc is None or self._closed:
            return
        with contextlib.suppress(OSError, EOFError):
            self._proc.terminate(force=False)

    def kill(self) -> None:
        if self._proc is None or self._closed:
            return
        with contextlib.suppress(OSError, EOFError):
            self._proc.terminate(force=True)

    def resize(self, columns: int, rows: int) -> None:
        self._columns = columns
        self._rows = rows
        if self._proc is None or self._closed or not self._proc.isalive():
            return
        with contextlib.suppress(Exception):
            self._proc.setwinsize(rows, columns)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                if self._proc.isalive():
                    self._proc.terminate(force=True)
                self._proc.close()
            except Exception:
                pass
            self._proc = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_pty_backend(columns: int = 80, rows: int = 24) -> PtyBackend:
    """Return a platform-appropriate PTY backend instance.

    On Windows, requires ``pywinpty`` (``pip install pywinpty``).

    Raises ``NotImplementedError`` on unsupported platforms.
    """
    if sys.platform in ("linux", "darwin"):
        return UnixPtyBackend(columns, rows)
    if sys.platform == "win32":
        if not _has_pywinpty():
            raise NotImplementedError(
                "PTY on Windows requires pywinpty: pip install pywinpty"
            )
        return WinPtyBackend(columns, rows)
    raise NotImplementedError(f"No PTY backend for platform: {sys.platform}")
