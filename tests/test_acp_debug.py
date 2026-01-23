"""Tests for acp_debug module."""

from __future__ import annotations

import asyncio
import platform
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from acp_debug.config import Config, ShutdownConfig, load_config


class TestShutdownConfig:
    """Tests for ShutdownConfig dataclass."""

    def test_default_values(self) -> None:
        """ShutdownConfig has sensible defaults."""
        config = ShutdownConfig()
        assert config.interrupt_timeout == 2.0
        assert config.terminate_timeout == 3.0

    def test_custom_values(self) -> None:
        """ShutdownConfig accepts custom values."""
        config = ShutdownConfig(interrupt_timeout=5.0, terminate_timeout=10.0)
        assert config.interrupt_timeout == 5.0
        assert config.terminate_timeout == 10.0


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_shutdown_config(self) -> None:
        """Config includes default ShutdownConfig."""
        config = Config()
        assert isinstance(config.shutdown, ShutdownConfig)
        assert config.shutdown.interrupt_timeout == 2.0
        assert config.shutdown.terminate_timeout == 3.0

    def test_custom_shutdown_config(self) -> None:
        """Config accepts custom ShutdownConfig."""
        shutdown = ShutdownConfig(interrupt_timeout=1.0, terminate_timeout=2.0)
        config = Config(shutdown=shutdown)
        assert config.shutdown.interrupt_timeout == 1.0
        assert config.shutdown.terminate_timeout == 2.0


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self, tmp_path: Path) -> None:
        """load_config returns default Config when no file exists."""
        with patch("acp_debug.config.Path.exists", return_value=False):
            config = load_config()
        assert isinstance(config, Config)
        assert config.shutdown.interrupt_timeout == 2.0

    def test_load_yaml_with_shutdown(self, tmp_path: Path) -> None:
        """load_config parses shutdown section from YAML."""
        config_file = tmp_path / "acp-debug.yaml"
        config_file.write_text("""
shutdown:
  interrupt_timeout: 5.0
  terminate_timeout: 10.0
verbose: 2
""")
        config = load_config(config_path=config_file)
        assert config.shutdown.interrupt_timeout == 5.0
        assert config.shutdown.terminate_timeout == 10.0
        assert config.verbose == 2

    def test_load_yaml_partial_shutdown(self, tmp_path: Path) -> None:
        """load_config uses defaults for missing shutdown fields."""
        config_file = tmp_path / "acp-debug.yaml"
        config_file.write_text("""
shutdown:
  interrupt_timeout: 1.0
""")
        config = load_config(config_path=config_file)
        assert config.shutdown.interrupt_timeout == 1.0
        assert config.shutdown.terminate_timeout == 3.0  # default

    def test_load_yaml_no_shutdown(self, tmp_path: Path) -> None:
        """load_config uses default shutdown when section missing."""
        config_file = tmp_path / "acp-debug.yaml"
        config_file.write_text("""
verbose: 1
quiet: true
""")
        config = load_config(config_path=config_file)
        assert config.shutdown.interrupt_timeout == 2.0
        assert config.shutdown.terminate_timeout == 3.0


class TestSignalHandling:
    """Tests for signal handling functions."""

    def test_send_interrupt_windows(self) -> None:
        """_send_interrupt sends CTRL_BREAK_EVENT on Windows."""
        from acp_debug.modes.proxy import _send_interrupt

        process = MagicMock()
        process.pid = 12345

        with patch("acp_debug.modes.proxy._WINDOWS", True):
            with patch("os.kill") as mock_kill:
                import signal
                _send_interrupt(process)
                mock_kill.assert_called_once_with(12345, signal.CTRL_BREAK_EVENT)

    def test_send_interrupt_unix(self) -> None:
        """_send_interrupt sends SIGINT on Unix."""
        from acp_debug.modes.proxy import _send_interrupt

        process = MagicMock()
        process.pid = 12345

        with patch("acp_debug.modes.proxy._WINDOWS", False):
            with patch("os.kill") as mock_kill:
                import signal
                _send_interrupt(process)
                mock_kill.assert_called_once_with(12345, signal.SIGINT)

    def test_send_interrupt_fallback_on_error(self) -> None:
        """_send_interrupt falls back to terminate on OSError."""
        from acp_debug.modes.proxy import _send_interrupt

        process = MagicMock()
        process.pid = 12345

        with patch("acp_debug.modes.proxy._WINDOWS", False):
            with patch("os.kill", side_effect=OSError("No such process")):
                _send_interrupt(process)
                process.terminate.assert_called_once()

    def test_send_terminate_windows(self) -> None:
        """_send_terminate calls process.terminate() on Windows."""
        from acp_debug.modes.proxy import _send_terminate

        process = MagicMock()

        with patch("acp_debug.modes.proxy._WINDOWS", True):
            _send_terminate(process)
            process.terminate.assert_called_once()

    def test_send_terminate_unix(self) -> None:
        """_send_terminate sends SIGTERM on Unix."""
        from acp_debug.modes.proxy import _send_terminate

        process = MagicMock()
        process.pid = 12345

        with patch("acp_debug.modes.proxy._WINDOWS", False):
            with patch("os.kill") as mock_kill:
                import signal
                _send_terminate(process)
                mock_kill.assert_called_once_with(12345, signal.SIGTERM)


class TestGracefulShutdown:
    """Tests for graceful_shutdown function."""

    @pytest.mark.asyncio
    async def test_already_exited(self) -> None:
        """graceful_shutdown does nothing if process already exited."""
        from acp_debug.modes.proxy import graceful_shutdown

        process = MagicMock()
        process.returncode = 0  # Already exited

        await graceful_shutdown(process)
        # No signals should be sent
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_exits_on_interrupt(self) -> None:
        """graceful_shutdown exits after interrupt signal."""
        from acp_debug.modes.proxy import graceful_shutdown

        process = MagicMock()
        process.returncode = None
        process.pid = 12345

        # Process exits after interrupt
        async def wait_and_exit() -> int:
            process.returncode = 0
            return 0

        process.wait = wait_and_exit

        with patch("acp_debug.modes.proxy._send_interrupt") as mock_interrupt:
            await graceful_shutdown(process, interrupt_timeout=1.0)
            mock_interrupt.assert_called_once_with(process)

    @pytest.mark.asyncio
    async def test_escalates_to_terminate(self) -> None:
        """graceful_shutdown escalates to terminate if interrupt times out."""
        from acp_debug.modes.proxy import graceful_shutdown

        process = MagicMock()
        process.returncode = None
        process.pid = 12345

        call_count = 0

        async def wait_slow() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First wait (interrupt) times out
                await asyncio.sleep(10)
            # Second wait (terminate) succeeds
            process.returncode = 0
            return 0

        process.wait = wait_slow

        with patch("acp_debug.modes.proxy._send_interrupt") as mock_interrupt:
            with patch("acp_debug.modes.proxy._send_terminate") as mock_terminate:
                await graceful_shutdown(
                    process,
                    interrupt_timeout=0.01,
                    terminate_timeout=1.0,
                )
                mock_interrupt.assert_called_once()
                mock_terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalates_to_kill(self) -> None:
        """graceful_shutdown escalates to kill if terminate times out."""
        from acp_debug.modes.proxy import graceful_shutdown

        process = MagicMock()
        process.returncode = None
        process.pid = 12345

        async def wait_forever() -> int:
            await asyncio.sleep(100)
            return 0

        async def wait_after_kill() -> int:
            process.returncode = -9
            return -9

        # First two waits timeout, third succeeds after kill
        wait_count = 0

        async def wait_impl() -> int:
            nonlocal wait_count
            wait_count += 1
            if wait_count <= 2:
                await asyncio.sleep(100)
            process.returncode = -9
            return -9

        process.wait = wait_impl

        with patch("acp_debug.modes.proxy._send_interrupt"):
            with patch("acp_debug.modes.proxy._send_terminate"):
                await graceful_shutdown(
                    process,
                    interrupt_timeout=0.01,
                    terminate_timeout=0.01,
                )
                process.kill.assert_called_once()


class TestWindowsFlags:
    """Tests for Windows subprocess creation flags."""

    def test_create_new_process_group_value(self) -> None:
        """CREATE_NEW_PROCESS_GROUP has correct value on Windows."""
        from acp_debug.modes.proxy import _CREATE_NEW_PROCESS_GROUP, _WINDOWS

        if _WINDOWS:
            assert _CREATE_NEW_PROCESS_GROUP == 0x00000200
        else:
            assert _CREATE_NEW_PROCESS_GROUP == 0
