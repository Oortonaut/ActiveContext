"""Tests for infrastructure components (config watcher, logging, entry point).

Tests coverage for:
- src/activecontext/config/watcher.py
- src/activecontext/logging.py
- src/activecontext/__main__.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from activecontext.config.watcher import ConfigWatcher, start_watching, stop_watching
from activecontext.logging import get_logger, setup_logging


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file path."""
    return str(tmp_path / "test.log")


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for config files."""
    config_dir = tmp_path / ".ac"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def temp_config_file(temp_config_dir):
    """Create a temporary config file."""
    config_file = temp_config_dir / "config.yaml"
    config_file.write_text("llm:\n  model: test-model\n")
    return config_file


# =============================================================================
# Config Watcher Tests
# =============================================================================


class TestConfigWatcher:
    """Tests for config file watching and reloading."""

    @pytest.mark.asyncio
    async def test_watcher_initialization(self):
        """Test ConfigWatcher initialization."""
        watcher = ConfigWatcher(session_root="/test/project", poll_interval=1.0)

        assert watcher._session_root == "/test/project"
        assert watcher._poll_interval == 1.0
        assert watcher._running is False
        assert watcher._task is None

    @pytest.mark.asyncio
    async def test_watcher_start_stop(self):
        """Test starting and stopping watcher."""
        watcher = ConfigWatcher(session_root=None, poll_interval=0.1)

        watcher.start()
        assert watcher._running is True
        assert watcher._task is not None

        # Let it run briefly
        await asyncio.sleep(0.05)

        watcher.stop()
        assert watcher._running is False

    @pytest.mark.asyncio
    async def test_watcher_detect_file_modification(
        self, temp_config_file, temp_config_dir
    ):
        """Test detection of file modification."""
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ):
            watcher = ConfigWatcher(poll_interval=0.05)

            # Start watcher
            watcher.start()
            await asyncio.sleep(0.02)

            # Record initial mtimes
            initial_mtimes = watcher._mtimes.copy()

            # Modify the file
            await asyncio.sleep(0.02)
            temp_config_file.write_text("llm:\n  model: modified-model\n")

            # Wait for poll cycle
            await asyncio.sleep(0.1)

            # Watcher should detect change (mtimes updated)
            current_mtimes = watcher._mtimes
            assert current_mtimes != initial_mtimes

            watcher.stop()

    @pytest.mark.asyncio
    async def test_watcher_detect_file_creation(self, temp_config_dir):
        """Test detection of new config file."""
        new_file = temp_config_dir / "new_config.yaml"

        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[new_file],
        ):
            watcher = ConfigWatcher(poll_interval=0.05)
            watcher.start()

            await asyncio.sleep(0.02)

            # Create new file
            new_file.write_text("test: value\n")

            # Wait for detection
            await asyncio.sleep(0.1)

            # File should be tracked
            assert new_file in watcher._mtimes

            watcher.stop()

    @pytest.mark.asyncio
    async def test_watcher_detect_file_deletion(
        self, temp_config_file, temp_config_dir
    ):
        """Test detection of config file deletion."""
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ):
            watcher = ConfigWatcher(poll_interval=0.05)
            watcher.start()

            await asyncio.sleep(0.02)

            # Delete the file
            temp_config_file.unlink()

            # Wait for detection
            await asyncio.sleep(0.1)

            # File should no longer be tracked
            assert temp_config_file not in watcher._mtimes

            watcher.stop()

    @pytest.mark.asyncio
    async def test_watcher_calls_reload_on_change(self, temp_config_file):
        """Test that reload_config is called when changes detected."""
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ), patch("activecontext.config.watcher.reload_config") as mock_reload:
            watcher = ConfigWatcher(poll_interval=0.05)
            watcher.start()

            await asyncio.sleep(0.02)

            # Modify file
            temp_config_file.write_text("llm:\n  model: updated\n")

            # Wait for reload
            await asyncio.sleep(0.15)

            # reload_config should have been called
            assert mock_reload.call_count > 0

            watcher.stop()

    @pytest.mark.asyncio
    async def test_watcher_handles_reload_errors(self, temp_config_file):
        """Test that watcher continues on reload errors."""
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ), patch(
            "activecontext.config.watcher.reload_config",
            side_effect=Exception("Reload error"),
        ):
            watcher = ConfigWatcher(poll_interval=0.05)
            watcher.start()

            await asyncio.sleep(0.02)

            # Modify file (will trigger error)
            temp_config_file.write_text("invalid: yaml: content:\n")

            # Wait - watcher should continue despite error
            await asyncio.sleep(0.15)

            # Watcher should still be running
            assert watcher._running is True

            watcher.stop()

    @pytest.mark.asyncio
    async def test_watcher_context_manager(self, temp_config_file):
        """Test ConfigWatcher as async context manager."""
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ):
            async with ConfigWatcher(poll_interval=0.1) as watcher:
                assert watcher._running is True

            # Should be stopped after exit
            assert watcher._running is False

    @pytest.mark.asyncio
    async def test_global_watcher_start(self, temp_config_file):
        """Test global watcher start function."""
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ):
            watcher = start_watching(session_root="/test", poll_interval=0.1)

            assert watcher is not None
            assert watcher._running is True

            # Cleanup
            stop_watching()

    @pytest.mark.asyncio
    async def test_global_watcher_stop(self):
        """Test global watcher stop function."""
        with patch("activecontext.config.watcher.get_config_paths", return_value=[]):
            watcher = start_watching()
            assert watcher._running is True

            stop_watching()

            # Should be stopped
            assert watcher._running is False

    @pytest.mark.asyncio
    async def test_watcher_idempotent_start(self):
        """Test that calling start twice is idempotent."""
        watcher = ConfigWatcher(poll_interval=0.1)

        watcher.start()
        first_task = watcher._task

        watcher.start()  # Should be no-op
        second_task = watcher._task

        assert first_task is second_task

        watcher.stop()


# =============================================================================
# Logging Tests
# =============================================================================


class TestLogging:
    """Tests for logging configuration."""

    def test_setup_logging_creates_file_handler(self, temp_log_file):
        """Test setup_logging with file configuration."""
        from activecontext.config.schema import LoggingConfig

        # Reset initialization flag for testing
        import activecontext.logging as logging_module

        logging_module._initialized = False

        config = LoggingConfig(level="INFO", file=temp_log_file)
        setup_logging(config)

        # Verify file was created
        assert Path(temp_log_file).exists()

        # Verify logger configured
        logger = get_logger()
        assert logger.level == logging.INFO

    def test_setup_logging_stderr_fallback(self):
        """Test setup_logging falls back to stderr."""
        from activecontext.config.schema import LoggingConfig

        # Reset initialization flag
        import activecontext.logging as logging_module

        logging_module._initialized = False

        # Config with invalid file path
        config = LoggingConfig(level="DEBUG", file="/nonexistent/dir/log.txt")

        # Should not raise, falls back to stderr
        setup_logging(config)

    def test_setup_logging_idempotent(self):
        """Test that calling setup_logging twice is no-op."""
        import activecontext.logging as logging_module

        # First call
        logging_module._initialized = False
        setup_logging()
        assert logging_module._initialized is True

        # Second call should be no-op
        setup_logging()
        assert logging_module._initialized is True

    def test_setup_logging_level_mapping(self):
        """Test log level string to constant mapping."""
        from activecontext.config.schema import LoggingConfig
        import activecontext.logging as logging_module

        test_levels = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for level_str, expected_const in test_levels:
            logging_module._initialized = False
            config = LoggingConfig(level=level_str, file=None)
            setup_logging(config)

            logger = get_logger()
            assert logger.level == expected_const

    def test_get_logger_child(self):
        """Test getting a child logger."""
        parent = get_logger()
        child = get_logger("test_child")

        assert child.name == f"{parent.name}.test_child"

    def test_get_logger_root(self):
        """Test getting root activecontext logger."""
        logger = get_logger()

        assert logger.name == "activecontext"

    def test_logging_format(self, temp_log_file):
        """Test log message format."""
        from activecontext.config.schema import LoggingConfig
        import activecontext.logging as logging_module

        logging_module._initialized = False

        config = LoggingConfig(level="INFO", file=temp_log_file)
        setup_logging(config)

        logger = get_logger()
        logger.info("Test message")

        # Read log file
        log_content = Path(temp_log_file).read_text()

        # Should contain timestamp and message
        assert "Test message" in log_content
        assert "[" in log_content  # Timestamp brackets


# =============================================================================
# Entry Point Tests
# =============================================================================


class TestMainEntryPoint:
    """Tests for __main__ entry point."""

    def test_expand_env_vars(self, monkeypatch):
        """Test _expand_env_vars expands ${VAR_NAME} patterns."""
        from activecontext.__main__ import _expand_env_vars

        monkeypatch.setenv("TEST_VAR", "test_value")
        monkeypatch.setenv("REFERENCE_VAR", "${TEST_VAR}")

        _expand_env_vars()

        # Should expand reference
        assert os.environ["REFERENCE_VAR"] == "test_value"

    def test_expand_env_vars_multiple_refs(self, monkeypatch):
        """Test expansion with multiple variable references."""
        from activecontext.__main__ import _expand_env_vars

        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        monkeypatch.setenv("COMBINED", "${VAR1}:${VAR2}")

        _expand_env_vars()

        assert os.environ["COMBINED"] == "value1:value2"

    def test_expand_env_vars_missing_ref(self, monkeypatch):
        """Test that missing variable references are left as-is."""
        from activecontext.__main__ import _expand_env_vars

        monkeypatch.setenv("MISSING_REF", "${NONEXISTENT_VAR}")

        _expand_env_vars()

        # Should leave unexpanded
        assert os.environ["MISSING_REF"] == "${NONEXISTENT_VAR}"

    def test_setup_parent_death_monitor_no_op(self):
        """Test _setup_parent_death_monitor is no-op."""
        from activecontext.__main__ import _setup_parent_death_monitor

        # Should not raise
        _setup_parent_death_monitor()

    @patch("asyncio.run")
    @patch("activecontext.config.load_config")
    @patch("activecontext.__main__.setup_logging")
    @patch("activecontext.__main__._expand_env_vars")
    @patch("activecontext.core.llm.discovery.get_default_model")
    @patch("os._exit")
    def test_main_startup_sequence(
        self,
        mock_exit,
        mock_get_default_model,
        mock_expand,
        mock_setup_log,
        mock_load_config,
        mock_asyncio_run,
    ):
        """Test main() startup sequence."""
        from activecontext.__main__ import main

        from activecontext.config.schema import LoggingConfig

        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.llm.role = "coding"
        mock_config.llm.provider = "anthropic"
        mock_config.llm.role_providers = []
        mock_config.projection = Mock()
        mock_config.projection.total_budget = 8000
        mock_config.logging = LoggingConfig()  # Use real LoggingConfig with defaults
        mock_load_config.return_value = mock_config
        mock_get_default_model.return_value = "test-model"

        main()

        # Verify call order
        mock_expand.assert_called_once()
        mock_load_config.assert_called_once()
        mock_setup_log.assert_called_once_with(mock_config.logging)
        mock_asyncio_run.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @pytest.mark.asyncio
    @patch("acp.stdio.stdio_streams")
    @patch("activecontext.transport.acp.agent.create_agent")
    async def test_main_async_entry(self, mock_create_agent, mock_stdio):
        """Test _main async entry point."""
        from activecontext.__main__ import _main

        # Mock agent
        mock_agent = Mock()
        mock_agent._current_model_id = "test-model"
        mock_create_agent.return_value = mock_agent

        # Mock stdio streams
        mock_output = AsyncMock()
        mock_input = AsyncMock()
        mock_stdio.return_value = (mock_output, mock_input)

        # Mock connection
        with patch(
            "acp.agent.connection.AgentSideConnection"
        ) as mock_conn_class:
            mock_conn = AsyncMock()
            mock_conn._conn = Mock()
            mock_conn._conn.add_observer = Mock()
            mock_conn.listen = AsyncMock()
            mock_conn.close = AsyncMock()
            mock_conn_class.return_value = mock_conn

            await _main()

            # Verify setup
            mock_create_agent.assert_called_once()
            mock_stdio.assert_called_once()
            mock_conn_class.assert_called_once()

            # Verify connection lifecycle
            mock_conn.listen.assert_called_once()
            mock_conn.close.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestInfrastructureIntegration:
    """Integration tests for infrastructure components."""

    @pytest.mark.asyncio
    async def test_watcher_and_logging_together(self, temp_log_file, temp_config_file):
        """Test config watcher and logging working together."""
        from activecontext.config.schema import LoggingConfig
        import activecontext.logging as logging_module

        # Setup logging
        logging_module._initialized = False
        config = LoggingConfig(level="INFO", file=temp_log_file)
        setup_logging(config)

        # Start watcher
        with patch(
            "activecontext.config.watcher.get_config_paths",
            return_value=[temp_config_file],
        ):
            watcher = ConfigWatcher(poll_interval=0.05)
            watcher.start()

            await asyncio.sleep(0.02)

            # Modify config
            temp_config_file.write_text("llm:\n  model: new-model\n")

            # Wait for change detection
            await asyncio.sleep(0.1)

            watcher.stop()

            # Log file should exist and contain activity
            assert Path(temp_log_file).exists()
