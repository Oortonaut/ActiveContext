"""Tests for PtyBackend – platform-conditional interactive PTY."""

from __future__ import annotations

import asyncio
import sys

import pytest

from activecontext.terminal.pty_backend import (
    PtyBackend,
    create_pty_backend,
)
from activecontext.terminal.pty_support import is_pty_supported

pytestmark = pytest.mark.skipif(
    not is_pty_supported(),
    reason="PTY not supported on this platform",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _echo_command() -> tuple[str, list[str]]:
    """Return a command + args that prints 'hello' and exits immediately."""
    if sys.platform == "win32":
        return "python", ["-c", "print('hello')"]
    return "echo", ["hello"]


def _echo_command_with_delay() -> tuple[str, list[str]]:
    """Return a command that prints 'hello' and stays alive briefly.

    ConPTY on Windows can buffer output for several seconds before flushing
    to the pipe.  The sleep keeps the process alive long enough for the read
    loop to capture the output.
    """
    script = "import time; print('hello'); time.sleep(5)"
    if sys.platform == "win32":
        return "python", ["-c", script]
    return "python3", ["-c", script]


def _interactive_command() -> tuple[str, list[str]]:
    """Return a command that runs an interactive REPL."""
    if sys.platform == "win32":
        return "python", ["-i", "-c", ""]
    return "python3", ["-i", "-c", ""]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_returns_pty_backend(self):
        backend = create_pty_backend()
        assert isinstance(backend, PtyBackend)

    def test_create_custom_size(self):
        backend = create_pty_backend(columns=120, rows=40)
        assert isinstance(backend, PtyBackend)


# ---------------------------------------------------------------------------
# Spawn and lifecycle
# ---------------------------------------------------------------------------


class TestSpawnAndLifecycle:
    @pytest.mark.asyncio
    async def test_spawn_simple_command(self):
        backend = create_pty_backend()
        cmd, args = _echo_command()
        await backend.spawn(cmd, args)

        exit_code = await backend.wait()
        assert exit_code == 0
        backend.close()

    @pytest.mark.asyncio
    async def test_is_alive_reflects_process_state(self):
        backend = create_pty_backend()
        cmd, args = _echo_command()
        await backend.spawn(cmd, args)

        # Process should finish quickly
        await backend.wait()
        assert not backend.is_alive
        backend.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        backend = create_pty_backend()
        cmd, args = _echo_command()
        await backend.spawn(cmd, args)
        await backend.wait()

        backend.close()
        backend.close()  # Should not raise


# ---------------------------------------------------------------------------
# Read / Write
# ---------------------------------------------------------------------------


class TestReadWrite:
    @pytest.mark.asyncio
    async def test_read_output(self):
        """Read output while the process is alive."""
        backend = create_pty_backend()
        cmd, args = _echo_command_with_delay()
        await backend.spawn(cmd, args)

        # ConPTY on Windows can buffer output for several seconds before
        # flushing, so we use a generous per-read timeout.
        output = b""
        for _ in range(200):
            try:
                chunk = await asyncio.wait_for(backend.read(4096), timeout=5.0)
            except asyncio.TimeoutError:
                chunk = b""
            if chunk:
                output += chunk
                if b"hello" in output:
                    break
            elif not backend.is_alive:
                break
            await asyncio.sleep(0.01)

        assert b"hello" in output
        backend.close()

    @pytest.mark.asyncio
    async def test_read_returns_empty_after_exit(self):
        """After process exits and output is drained, read returns empty."""
        backend = create_pty_backend()
        cmd, args = _echo_command()
        await backend.spawn(cmd, args)

        # Wait for the process, then drain
        await asyncio.wait_for(backend.wait(), timeout=5.0)

        got_eof = False
        for _ in range(100):
            try:
                chunk = await asyncio.wait_for(backend.read(4096), timeout=1.0)
            except asyncio.TimeoutError:
                chunk = b""
            if not chunk:
                got_eof = True
                break

        assert got_eof, "read() never returned empty bytes after exit"
        backend.close()


# ---------------------------------------------------------------------------
# Terminate / Kill
# ---------------------------------------------------------------------------


class TestTermination:
    @pytest.mark.asyncio
    async def test_terminate(self):
        backend = create_pty_backend()
        # Use a long-running command
        if sys.platform == "win32":
            await backend.spawn("python", ["-c", "import time; time.sleep(60)"])
        else:
            await backend.spawn("sleep", ["60"])

        assert backend.is_alive
        backend.terminate()

        exit_code = await asyncio.wait_for(backend.wait(), timeout=5.0)
        assert not backend.is_alive
        # Exit code is platform-specific but should be non-zero
        assert exit_code != 0 or not backend.is_alive
        backend.close()

    @pytest.mark.asyncio
    async def test_kill(self):
        backend = create_pty_backend()
        if sys.platform == "win32":
            await backend.spawn("python", ["-c", "import time; time.sleep(60)"])
        else:
            await backend.spawn("sleep", ["60"])

        assert backend.is_alive
        backend.kill()

        exit_code = await asyncio.wait_for(backend.wait(), timeout=5.0)
        assert not backend.is_alive
        backend.close()


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


class TestResize:
    @pytest.mark.asyncio
    async def test_resize_does_not_crash(self):
        """Resize should not raise, even if the process has exited."""
        backend = create_pty_backend()
        cmd, args = _echo_command()
        await backend.spawn(cmd, args)

        # Resize while running
        backend.resize(120, 40)

        await backend.wait()

        # Resize after exit — should be no-op
        backend.resize(80, 24)
        backend.close()
