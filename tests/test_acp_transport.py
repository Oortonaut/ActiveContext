"""Tests for ACP transport layer (agent and terminal executor).

Tests coverage for:
- src/activecontext/transport/acp/agent.py
- src/activecontext/terminal/acp_executor.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from activecontext.terminal.acp_executor import ACPTerminalExecutor
from activecontext.terminal.result import ShellResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_acp_client():
    """Create a mock ACP Client for terminal tests."""
    client = Mock()
    client.create_terminal = AsyncMock()
    client.wait_for_terminal_exit = AsyncMock()
    client.terminal_output = AsyncMock()
    client.kill_terminal = AsyncMock()
    client.release_terminal = AsyncMock()
    client.session_update = AsyncMock()
    return client


@pytest.fixture
def terminal_executor(mock_acp_client):
    """Create ACPTerminalExecutor with mock client."""
    return ACPTerminalExecutor(
        client=mock_acp_client,
        session_id="test-session",
        default_cwd="/test/dir",
    )


# =============================================================================
# Agent Initialization Tests
# =============================================================================


class TestAgentInit:
    """Tests for ActiveContextAgent initialization."""

    def test_agent_imports(self):
        """Test that ActiveContextAgent can be imported."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        assert ActiveContextAgent is not None

    def test_default_session_modes_exist(self):
        """Test that default session modes are defined."""
        from activecontext.transport.acp.agent import DEFAULT_SESSION_MODES

        assert len(DEFAULT_SESSION_MODES) > 0
        assert all(hasattr(mode, "id") for mode in DEFAULT_SESSION_MODES)
        assert all(hasattr(mode, "name") for mode in DEFAULT_SESSION_MODES)

    @patch("activecontext.transport.acp.agent.get_default_model")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_agent_initialization(self, mock_session_manager, mock_get_default):
        """Test agent initializes with LLM provider."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        mock_get_default.return_value = "claude-sonnet-4-20250514"

        agent = ActiveContextAgent()

        assert agent._current_model_id == "claude-sonnet-4-20250514"
        assert agent._batch_enabled is True
        assert agent._flush_interval == 0.05
        assert agent._flush_threshold == 100

    @patch("activecontext.transport.acp.agent.get_default_model")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_agent_init_no_llm(self, mock_session_manager, mock_get_default):
        """Test agent initialization when no LLM provider available."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        mock_get_default.return_value = None

        agent = ActiveContextAgent()

        assert agent._current_model_id is None


# =============================================================================
# Nagle Batching Tests
# =============================================================================


class TestNagleBatching:
    """Tests for Nagle-style response chunk batching."""

    @pytest.mark.asyncio
    @patch("activecontext.transport.acp.agent.get_default_model", return_value=None)
    @patch("activecontext.transport.acp.agent.SessionManager")
    async def test_buffer_chunk_accumulates(self, mock_sm, mock_model):
        """Test that _buffer_chunk accumulates text."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        agent._conn = Mock()

        # Buffer some small chunks
        await agent._buffer_chunk("session1", "Hello")
        await agent._buffer_chunk("session1", " ")
        await agent._buffer_chunk("session1", "world")

        # Should be accumulated
        assert agent._chunk_buffers["session1"] == "Hello world"

    @pytest.mark.asyncio
    @patch("activecontext.transport.acp.agent.get_default_model", return_value=None)
    @patch("activecontext.transport.acp.agent.SessionManager")
    async def test_buffer_chunk_flushes_at_threshold(self, mock_sm, mock_model):
        """Test that chunks flush when threshold is reached."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        agent._conn = Mock()
        agent._conn.session_update = AsyncMock()
        agent._flush_threshold = 10  # Small threshold for testing

        # Send text that exceeds threshold
        await agent._buffer_chunk("session1", "x" * 15)

        # Should have flushed immediately
        assert "session1" not in agent._chunk_buffers
        agent._conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    @patch("activecontext.transport.acp.agent.get_default_model", return_value=None)
    @patch("activecontext.transport.acp.agent.SessionManager")
    async def test_delayed_flush_after_interval(self, mock_sm, mock_model):
        """Test that buffered chunks flush after time interval."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        agent._conn = Mock()
        agent._conn.session_update = AsyncMock()
        agent._flush_interval = 0.01  # 10ms for testing

        # Buffer small chunk
        await agent._buffer_chunk("session1", "Short")

        # Should be buffered, not flushed yet
        assert "session1" in agent._chunk_buffers

        # Wait for flush interval
        await asyncio.sleep(0.02)

        # Should have flushed
        assert "session1" not in agent._chunk_buffers

    @pytest.mark.asyncio
    @patch("activecontext.transport.acp.agent.get_default_model", return_value=None)
    @patch("activecontext.transport.acp.agent.SessionManager")
    async def test_flush_chunks_sends_update(self, mock_sm, mock_model):
        """Test that _flush_chunks sends session update."""
        from activecontext.transport.acp.agent import ActiveContextAgent
        import acp

        agent = ActiveContextAgent()
        agent._conn = Mock()
        agent._conn.session_update = AsyncMock()
        agent._chunk_buffers["session1"] = "Test text"

        await agent._flush_chunks("session1")

        # Should have cleared buffer
        assert "session1" not in agent._chunk_buffers

        # Should have sent update
        agent._conn.session_update.assert_called_once()
        call_args = agent._conn.session_update.call_args
        assert call_args[0][0] == "session1"

    @pytest.mark.asyncio
    @patch("activecontext.transport.acp.agent.get_default_model", return_value=None)
    @patch("activecontext.transport.acp.agent.SessionManager")
    async def test_batching_can_be_disabled(self, mock_sm, mock_model):
        """Test that batching can be disabled via config."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        agent._batch_enabled = False

        # Batching disabled means each chunk goes through immediately
        # (This behavior would be in _emit_update, not tested here directly)
        assert agent._batch_enabled is False


# =============================================================================
# Terminal Executor Tests
# =============================================================================


class TestACPTerminalExecutor:
    """Tests for ACP terminal command execution."""

    @pytest.mark.asyncio
    async def test_execute_success(self, terminal_executor, mock_acp_client):
        """Test successful command execution."""
        # Mock successful execution
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(
            output="Command output"
        )

        result = await terminal_executor.execute("echo", args=["hello"])

        assert isinstance(result, ShellResult)
        assert result.command == "echo hello"
        assert result.exit_code == 0
        assert result.output == "Command output"
        assert result.status == "ok"
        assert result.truncated is False

        # Verify cleanup
        mock_acp_client.release_terminal.assert_called_once_with("term-1")

    @pytest.mark.asyncio
    async def test_execute_with_cwd(self, terminal_executor, mock_acp_client):
        """Test command execution with custom cwd."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(output="")

        await terminal_executor.execute("ls", cwd="/custom/dir")

        # Verify cwd was passed
        call_kwargs = mock_acp_client.create_terminal.call_args.kwargs
        assert call_kwargs["cwd"] == "/custom/dir"

    @pytest.mark.asyncio
    async def test_execute_uses_default_cwd(self, terminal_executor, mock_acp_client):
        """Test that default cwd is used when not specified."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(output="")

        await terminal_executor.execute("pwd")

        call_kwargs = mock_acp_client.create_terminal.call_args.kwargs
        assert call_kwargs["cwd"] == "/test/dir"

    @pytest.mark.asyncio
    async def test_execute_with_env_vars(self, terminal_executor, mock_acp_client):
        """Test command execution with environment variables."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(output="")

        env = {"TEST_VAR": "value"}
        await terminal_executor.execute("env", env=env)

        call_kwargs = mock_acp_client.create_terminal.call_args.kwargs
        assert call_kwargs["env"] == env

    @pytest.mark.asyncio
    async def test_execute_non_zero_exit_code(self, terminal_executor, mock_acp_client):
        """Test handling of non-zero exit codes."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=1, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(
            output="Error occurred"
        )

        result = await terminal_executor.execute("false")

        assert result.exit_code == 1
        assert result.status == "error"
        assert result.output == "Error occurred"

    @pytest.mark.asyncio
    async def test_execute_with_signal(self, terminal_executor, mock_acp_client):
        """Test handling of signaled termination."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=143, signal="SIGTERM"
        )
        mock_acp_client.terminal_output.return_value = Mock(
            output="Terminated"
        )

        result = await terminal_executor.execute("sleep", args=["100"])

        assert result.status == "killed"
        assert result.signal == "SIGTERM"

    @pytest.mark.asyncio
    async def test_execute_timeout(self, terminal_executor, mock_acp_client):
        """Test command timeout handling."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        # Simulate timeout by raising asyncio.TimeoutError
        mock_acp_client.wait_for_terminal_exit.side_effect = asyncio.TimeoutError()

        result = await terminal_executor.execute("sleep", args=["1000"], timeout=0.1)

        assert result.status == "timeout"
        assert result.signal == "SIGKILL"
        assert "timed out" in result.output
        assert result.exit_code is None

        # Should have attempted to kill terminal
        mock_acp_client.kill_terminal.assert_called_once_with("term-1")

    @pytest.mark.asyncio
    async def test_execute_output_truncation(self, terminal_executor, mock_acp_client):
        """Test that long output is truncated."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        # Create output longer than limit
        long_output = "x" * 60000
        mock_acp_client.terminal_output.return_value = Mock(output=long_output)

        result = await terminal_executor.execute("cat", output_limit=50000)

        assert result.truncated is True
        assert len(result.output) <= 50100  # Limit + truncation message
        assert "(output truncated)" in result.output

    @pytest.mark.asyncio
    async def test_execute_error_handling_not_found(
        self, terminal_executor, mock_acp_client
    ):
        """Test error handling for command not found."""
        mock_acp_client.create_terminal.side_effect = Exception("Command not found")

        result = await terminal_executor.execute("nonexistent-command")

        assert result.status == "error"
        assert result.exit_code == 127  # Command not found
        assert "Terminal error" in result.output

    @pytest.mark.asyncio
    async def test_execute_error_handling_permission(
        self, terminal_executor, mock_acp_client
    ):
        """Test error handling for permission denied."""
        mock_acp_client.create_terminal.side_effect = Exception(
            "Permission denied"
        )

        result = await terminal_executor.execute("restricted-command")

        assert result.status == "error"
        assert result.exit_code == 126  # Permission denied
        assert "Terminal error" in result.output

    @pytest.mark.asyncio
    async def test_execute_generic_error(self, terminal_executor, mock_acp_client):
        """Test handling of generic execution errors."""
        mock_acp_client.create_terminal.side_effect = Exception("Generic error")

        result = await terminal_executor.execute("command")

        assert result.status == "error"
        assert result.exit_code == 1
        assert "Generic error" in result.output

    @pytest.mark.asyncio
    async def test_execute_cleanup_on_success(self, terminal_executor, mock_acp_client):
        """Test that terminal is released after successful execution."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(output="")

        await terminal_executor.execute("echo", args=["test"])

        mock_acp_client.release_terminal.assert_called_once_with("term-1")

    @pytest.mark.asyncio
    async def test_execute_cleanup_on_error(self, terminal_executor, mock_acp_client):
        """Test that terminal cleanup happens even on error."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.side_effect = Exception("Test error")

        await terminal_executor.execute("command")

        # Should still attempt cleanup
        mock_acp_client.release_terminal.assert_called_once_with("term-1")

    @pytest.mark.asyncio
    async def test_execute_duration_tracking(self, terminal_executor, mock_acp_client):
        """Test that execution duration is tracked."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(output="")

        # Add small delay to ensure measurable duration
        async def delayed_exit(tid):
            await asyncio.sleep(0.01)
            return Mock(exit_code=0, signal=None)

        mock_acp_client.wait_for_terminal_exit = delayed_exit

        result = await terminal_executor.execute("command")

        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_no_timeout(self, terminal_executor, mock_acp_client):
        """Test command execution without timeout."""
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-1")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(output="")

        # No timeout should use wait_for_terminal_exit without asyncio.wait_for
        result = await terminal_executor.execute("command", timeout=None)

        assert result.status == "ok"


# =============================================================================
# Session Management Tests
# =============================================================================


class TestACPSessions:
    """Tests for ACP session management."""

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="gpt-4")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_session_tracking(self, mock_sm, mock_model):
        """Test that agent tracks session metadata."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        # Verify dictionaries initialized
        assert isinstance(agent._sessions_cwd, dict)
        assert isinstance(agent._sessions_model, dict)
        assert isinstance(agent._sessions_mode, dict)


# =============================================================================
# Update Types Tests
# =============================================================================


class TestACPUpdates:
    """Tests for ACP update message handling."""

    def test_update_imports(self):
        """Test that ACP update functions can be imported."""
        import acp

        # These functions should exist for creating updates
        assert hasattr(acp, "update_agent_message_text")
        # Other update types would be tested here


# =============================================================================
# Integration Tests
# =============================================================================


class TestACPIntegration:
    """Integration tests for ACP transport."""

    @pytest.mark.asyncio
    async def test_terminal_executor_full_lifecycle(
        self, terminal_executor, mock_acp_client
    ):
        """Test complete terminal execution lifecycle."""
        # Setup mocks
        mock_acp_client.create_terminal.return_value = Mock(terminal_id="term-123")
        mock_acp_client.wait_for_terminal_exit.return_value = Mock(
            exit_code=0, signal=None
        )
        mock_acp_client.terminal_output.return_value = Mock(
            output="Hello, World!\n"
        )

        # Execute command
        result = await terminal_executor.execute(
            "echo",
            args=["Hello, World!"],
            cwd="/workspace",
            env={"LANG": "en_US.UTF-8"},
            timeout=10.0,
        )

        # Verify result
        assert result.command == "echo Hello, World!"
        assert result.exit_code == 0
        assert result.output == "Hello, World!\n"
        assert result.status == "ok"
        assert result.duration_ms > 0

        # Verify ACP calls
        mock_acp_client.create_terminal.assert_called_once()
        create_kwargs = mock_acp_client.create_terminal.call_args.kwargs
        assert create_kwargs["command"] == "echo Hello, World!"
        assert create_kwargs["cwd"] == "/workspace"
        assert create_kwargs["env"] == {"LANG": "en_US.UTF-8"}

        mock_acp_client.wait_for_terminal_exit.assert_called_once_with("term-123")
        mock_acp_client.terminal_output.assert_called_once_with("term-123")
        mock_acp_client.release_terminal.assert_called_once_with("term-123")


# =============================================================================
# Agent Methods Tests
# =============================================================================


class TestAgentMethods:
    """Tests for ActiveContextAgent methods."""

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_get_available_commands(self, mock_sm, mock_model):
        """Test _get_available_commands returns slash commands."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        commands = agent._get_available_commands()

        assert isinstance(commands, list)
        assert len(commands) > 0

        # Check expected commands exist
        command_names = [c.name for c in commands]
        assert "help" in command_names
        assert "clear" in command_names
        assert "context" in command_names
        assert "title" in command_names
        assert "dashboard" in command_names

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_get_client_info_before_init(self, mock_sm, mock_model):
        """Test get_client_info returns None before initialize."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        client_info, protocol_version = agent.get_client_info()

        assert client_info is None
        assert protocol_version is None

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    @pytest.mark.asyncio
    async def test_initialize_stores_client_info(self, mock_sm, mock_model):
        """Test that initialize stores client info."""
        from activecontext.transport.acp.agent import ActiveContextAgent
        from acp.schema import Implementation

        agent = ActiveContextAgent()

        client_info = Implementation(name="test-client", version="1.0.0")
        response = await agent.initialize(
            protocol_version=1,
            client_info=client_info,
        )

        stored_info, stored_version = agent.get_client_info()

        assert stored_version == 1
        assert stored_info is not None
        assert stored_info["name"] == "test-client"
        assert stored_info["version"] == "1.0.0"

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    @pytest.mark.asyncio
    async def test_initialize_response_format(self, mock_sm, mock_model):
        """Test that initialize returns proper response."""
        from activecontext.transport.acp.agent import ActiveContextAgent
        import acp

        agent = ActiveContextAgent()

        response = await agent.initialize(protocol_version=1)

        assert isinstance(response, acp.InitializeResponse)
        assert response.agent_info.name == "activecontext"
        assert response.agent_info.version == "0.1.0"
        assert response.agent_capabilities is not None

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    @pytest.mark.asyncio
    async def test_initialize_with_capabilities(self, mock_sm, mock_model):
        """Test that initialize handles client capabilities."""
        from activecontext.transport.acp.agent import ActiveContextAgent
        from acp.schema import ClientCapabilities, FileSystemCapability

        agent = ActiveContextAgent()

        client_caps = ClientCapabilities(
            terminal=True,
            fs=FileSystemCapability(read_text_file=True, write_text_file=False),
        )

        await agent.initialize(
            protocol_version=1,
            client_capabilities=client_caps,
        )

        stored_info, _ = agent.get_client_info()
        assert stored_info["capabilities"] is not None
        caps_list = stored_info["capabilities"]

        # Find terminal capability
        terminal_cap = next((c for c in caps_list if c["name"] == "terminal"), None)
        assert terminal_cap is not None
        assert terminal_cap["enabled"] is True

        # Find fs.read_text_file capability
        read_cap = next((c for c in caps_list if c["name"] == "fs.read_text_file"), None)
        assert read_cap is not None
        assert read_cap["enabled"] is True

        # Find fs.write_text_file capability
        write_cap = next((c for c in caps_list if c["name"] == "fs.write_text_file"), None)
        assert write_cap is not None
        assert write_cap["enabled"] is False

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_session_modes_loaded(self, mock_sm, mock_model):
        """Test that session modes are loaded."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert len(agent._session_modes) > 0
        assert agent._default_mode_id is not None


class TestAgentConfigLoading:
    """Tests for agent configuration loading."""

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_batch_config_defaults(self, mock_sm, mock_model):
        """Test batch config defaults are set."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert agent._batch_enabled is True
        assert agent._flush_interval == 0.05
        assert agent._flush_threshold == 100

    @patch("activecontext.transport.acp.agent.get_default_model", return_value=None)
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_agent_without_llm(self, mock_sm, mock_model):
        """Test agent initializes without LLM (no API keys)."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert agent._current_model_id is None

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="gpt-4")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_agent_tracks_model_id(self, mock_sm, mock_model):
        """Test agent tracks current model ID."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert agent._current_model_id == "gpt-4"


class TestAgentSessionModes:
    """Tests for session mode functionality."""

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_default_session_modes_content(self, mock_sm, mock_model):
        """Test default session modes have expected structure."""
        from activecontext.transport.acp.agent import DEFAULT_SESSION_MODES

        assert len(DEFAULT_SESSION_MODES) >= 1

        for mode in DEFAULT_SESSION_MODES:
            assert hasattr(mode, "id")
            assert hasattr(mode, "name")
            assert mode.id is not None
            assert mode.name is not None

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_default_mode_id_exists(self, mock_sm, mock_model):
        """Test default mode ID is valid."""
        from activecontext.transport.acp.agent import (
            ActiveContextAgent,
            DEFAULT_SESSION_MODES,
        )

        agent = ActiveContextAgent()

        mode_ids = [m.id for m in DEFAULT_SESSION_MODES]
        assert agent._default_mode_id in mode_ids


class TestAgentChunkBuffering:
    """Tests for Nagle-style chunk buffering."""

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_chunk_buffer_initialized(self, mock_sm, mock_model):
        """Test chunk buffer is initialized as empty dict."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert isinstance(agent._chunk_buffers, dict)
        assert len(agent._chunk_buffers) == 0

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_flush_tasks_initialized(self, mock_sm, mock_model):
        """Test flush tasks dict is initialized."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert isinstance(agent._flush_tasks, dict)
        assert len(agent._flush_tasks) == 0

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_closed_sessions_tracking(self, mock_sm, mock_model):
        """Test closed sessions set is initialized."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert isinstance(agent._closed_sessions, set)
        assert len(agent._closed_sessions) == 0

    @patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514")
    @patch("activecontext.transport.acp.agent.SessionManager")
    def test_active_prompts_tracking(self, mock_sm, mock_model):
        """Test active prompts dict is initialized."""
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()

        assert isinstance(agent._active_prompts, dict)
        assert len(agent._active_prompts) == 0


class TestCreateAgentFunction:
    """Tests for the create_agent factory function."""

    def test_create_agent_returns_agent(self):
        """Test create_agent returns an ActiveContextAgent."""
        from activecontext.transport.acp.agent import create_agent, ActiveContextAgent

        with patch("activecontext.transport.acp.agent.get_default_model", return_value="claude-sonnet-4-20250514"):
            with patch("activecontext.transport.acp.agent.SessionManager"):
                agent = create_agent()

        assert isinstance(agent, ActiveContextAgent)
