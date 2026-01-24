"""Integration tests for acp_debug module.

These tests spawn actual subprocesses to verify shutdown behavior.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap

import pytest

from acp_debug.config import Config, ShutdownConfig
from acp_debug.modes.proxy import graceful_shutdown

# Path to Python interpreter
PYTHON = sys.executable


class TestGracefulShutdownIntegration:
    """Integration tests for graceful_shutdown with real subprocesses."""

    @pytest.mark.asyncio
    async def test_shutdown_clean_exit(self) -> None:
        """Process that exits cleanly on interrupt."""
        # Script that handles SIGINT/CTRL_BREAK and exits
        script = textwrap.dedent("""
            import signal
            import sys
            import time

            def handler(signum, frame):
                sys.exit(0)

            # Handle both SIGINT and SIGBREAK (Windows)
            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, handler)

            # Keep running until signaled
            while True:
                time.sleep(0.1)
        """)

        process = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=0x00000200 if sys.platform == "win32" else 0,
        )

        # Give process time to start and register handlers
        await asyncio.sleep(0.2)

        # Should exit cleanly on interrupt
        await graceful_shutdown(process, interrupt_timeout=2.0, terminate_timeout=2.0)

        assert process.returncode == 0

    @pytest.mark.asyncio
    async def test_shutdown_ignores_interrupt(self) -> None:
        """Process that ignores interrupt but exits on terminate."""
        # Script that ignores SIGINT but exits on SIGTERM
        script = textwrap.dedent("""
            import signal
            import sys
            import time

            def ignore(signum, frame):
                pass  # Ignore interrupt

            def terminate(signum, frame):
                sys.exit(0)

            signal.signal(signal.SIGINT, ignore)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, ignore)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, terminate)

            while True:
                time.sleep(0.1)
        """)

        process = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=0x00000200 if sys.platform == "win32" else 0,
        )

        await asyncio.sleep(0.2)

        # Should escalate to terminate after interrupt times out
        await graceful_shutdown(process, interrupt_timeout=0.3, terminate_timeout=2.0)

        # On Windows, terminate() kills immediately; on Unix, SIGTERM triggers handler
        assert process.returncode is not None

    @pytest.mark.asyncio
    async def test_shutdown_requires_kill(self) -> None:
        """Process that ignores all signals requires kill."""
        # Script that ignores all signals
        script = textwrap.dedent("""
            import signal
            import sys
            import time

            def ignore(signum, frame):
                pass

            signal.signal(signal.SIGINT, ignore)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, ignore)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, ignore)

            while True:
                time.sleep(0.1)
        """)

        process = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=0x00000200 if sys.platform == "win32" else 0,
        )

        await asyncio.sleep(0.2)

        # Should escalate all the way to kill
        await graceful_shutdown(process, interrupt_timeout=0.2, terminate_timeout=0.2)

        # Process should be dead (killed)
        assert process.returncode is not None

    @pytest.mark.asyncio
    async def test_shutdown_already_exited(self) -> None:
        """Process that exits before shutdown is called."""
        script = "import sys; sys.exit(42)"

        process = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for it to exit naturally
        await process.wait()

        # graceful_shutdown should handle already-exited process
        await graceful_shutdown(process, interrupt_timeout=1.0, terminate_timeout=1.0)

        assert process.returncode == 42

    @pytest.mark.asyncio
    async def test_configurable_timeouts_respected(self) -> None:
        """Verify configurable timeouts are actually respected."""
        # Script that takes 0.5s to respond to interrupt
        script = textwrap.dedent("""
            import signal
            import sys
            import time

            def delayed_exit(signum, frame):
                time.sleep(0.5)
                sys.exit(0)

            signal.signal(signal.SIGINT, delayed_exit)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, delayed_exit)

            while True:
                time.sleep(0.1)
        """)

        process = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=0x00000200 if sys.platform == "win32" else 0,
        )

        await asyncio.sleep(0.2)

        # With 1s timeout, should wait for clean exit
        start = asyncio.get_event_loop().time()
        await graceful_shutdown(process, interrupt_timeout=2.0, terminate_timeout=2.0)
        elapsed = asyncio.get_event_loop().time() - start

        # Should have waited ~0.5s for clean exit, not timed out
        assert process.returncode == 0
        assert 0.4 < elapsed < 1.5  # Allow some slack


class TestShutdownWithConfig:
    """Tests using Config object for timeouts."""

    @pytest.mark.asyncio
    async def test_config_timeouts_used(self) -> None:
        """Verify Config shutdown timeouts are passed correctly."""
        config = Config(
            shutdown=ShutdownConfig(
                interrupt_timeout=0.5,
                terminate_timeout=0.5,
            )
        )

        # Quick-exit script
        script = textwrap.dedent("""
            import signal
            import sys

            def handler(signum, frame):
                sys.exit(0)

            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, handler)

            import time
            while True:
                time.sleep(0.1)
        """)

        process = await asyncio.create_subprocess_exec(
            PYTHON, "-c", script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=0x00000200 if sys.platform == "win32" else 0,
        )

        await asyncio.sleep(0.2)

        # Use config values
        await graceful_shutdown(
            process,
            interrupt_timeout=config.shutdown.interrupt_timeout,
            terminate_timeout=config.shutdown.terminate_timeout,
        )

        assert process.returncode == 0
