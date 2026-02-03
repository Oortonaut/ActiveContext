"""Tests for ACP agent protocol handler.

Comprehensive tests for src/activecontext/transport/acp/agent.py covering:
- Initialize handshake
- Session lifecycle (new, load, list, delete)
- Prompt handling and streaming
- Slash command dispatch
- Cancel semantics
- Error handling and edge cases
- Nagle-style batching internals
- Permission request flows
- Conversation delegation
- Session mode/model switching
- Agent loop lifecycle
- Update emission and queueing
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import acp
import pytest
from acp.schema import (
    ClientCapabilities,
    FileSystemCapability,
    Implementation,
    TextContentBlock,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODULE = "activecontext.transport.acp.agent"


def _make_agent(**overrides):
    """Create an ActiveContextAgent with standard mocks.

    Patches get_default_model and SessionManager so __init__ does not
    touch real LLM providers or disk.
    """
    with (
        patch(f"{MODULE}.get_default_model", return_value=overrides.get("model")),
        patch(f"{MODULE}.SessionManager") as sm_cls,
    ):
        from activecontext.transport.acp.agent import ActiveContextAgent

        agent = ActiveContextAgent()
        agent._manager = sm_cls.return_value
        agent._manager.get_session = AsyncMock(return_value=None)
        agent._manager.create_session = AsyncMock()
        agent._manager._default_llm = None
        agent._manager._sessions = {}
        return agent


def _make_mock_session(session_id="test-session-1"):
    """Create a minimal mock Session for protocol-level tests."""
    session = AsyncMock()
    session.session_id = session_id
    session.title = "Test Session"
    session.cwd = "/test/cwd"
    session._current_task = None
    session._cancelled = False
    session.save = MagicMock()
    session.set_title = MagicMock()
    session.set_mode = MagicMock()
    session.set_llm = MagicMock()
    session.clear_message_history = MagicMock()
    session.get_context_objects = MagicMock(return_value={})
    session.queue_user_message = MagicMock()
    session.cancel = AsyncMock()
    session._emit_update_callback = None
    session._register_transport_callback = None
    session._unregister_transport_callback = None
    session.timeline = MagicMock()
    session.timeline._terminal_executor = None
    session.timeline._mcp_integration = MagicMock()
    # Make startup an async iterator that yields nothing
    session.startup = AsyncMock(return_value=_empty_async_iter())
    # Make run_agent_loop an async iterator that yields nothing
    session.run_agent_loop = AsyncMock(return_value=_empty_async_iter())
    return session


async def _empty_async_iter():
    """Async generator that yields nothing."""
    return
    yield  # noqa: unreachable – makes this an async generator


def _make_mock_conn():
    """Create a mock ACP Client connection."""
    conn = AsyncMock()
    conn.session_update = AsyncMock()
    conn.request_permission = AsyncMock()
    return conn


# ============================================================================
# TestACPInitialize
# ============================================================================


class TestACPInitialize:
    """Test the initialize handshake."""

    @pytest.mark.asyncio
    async def test_initialize_returns_response(self):
        """initialize() returns a well-formed InitializeResponse."""
        agent = _make_agent()
        resp = await agent.initialize(protocol_version=1)

        assert isinstance(resp, acp.InitializeResponse)
        assert resp.agent_info.name == "activecontext"
        assert resp.agent_info.version == "0.1.0"
        assert resp.protocol_version == acp.PROTOCOL_VERSION

    @pytest.mark.asyncio
    async def test_initialize_stores_protocol_version(self):
        """initialize() records the negotiated protocol version."""
        agent = _make_agent()
        await agent.initialize(protocol_version=42)

        assert agent._protocol_version == 42

    @pytest.mark.asyncio
    async def test_initialize_stores_client_info(self):
        """initialize() records client name and version."""
        agent = _make_agent()
        await agent.initialize(
            protocol_version=1,
            client_info=Implementation(name="rider", version="2025.3"),
        )

        info, _ = agent.get_client_info()
        assert info["name"] == "rider"
        assert info["version"] == "2025.3"

    @pytest.mark.asyncio
    async def test_initialize_with_no_client_info(self):
        """initialize() defaults client name to 'unknown'."""
        agent = _make_agent()
        await agent.initialize(protocol_version=1)

        info, _ = agent.get_client_info()
        assert info["name"] == "unknown"

    @pytest.mark.asyncio
    async def test_initialize_capabilities_terminal(self):
        """Client terminal capability is parsed correctly."""
        agent = _make_agent()
        caps = ClientCapabilities(terminal=True)
        await agent.initialize(protocol_version=1, client_capabilities=caps)

        info, _ = agent.get_client_info()
        terminal = next(c for c in info["capabilities"] if c["name"] == "terminal")
        assert terminal["enabled"] is True

    @pytest.mark.asyncio
    async def test_initialize_capabilities_fs(self):
        """Client filesystem capabilities are parsed correctly."""
        agent = _make_agent()
        caps = ClientCapabilities(
            fs=FileSystemCapability(read_text_file=True, write_text_file=False),
        )
        await agent.initialize(protocol_version=1, client_capabilities=caps)

        info, _ = agent.get_client_info()
        read_cap = next(c for c in info["capabilities"] if c["name"] == "fs.read_text_file")
        assert read_cap["enabled"] is True
        write_cap = next(c for c in info["capabilities"] if c["name"] == "fs.write_text_file")
        assert write_cap["enabled"] is False

    @pytest.mark.asyncio
    async def test_initialize_agent_capabilities(self):
        """Response advertises expected agent capabilities."""
        agent = _make_agent()
        resp = await agent.initialize(protocol_version=1)

        assert resp.agent_capabilities is not None
        assert resp.agent_capabilities.load_session is True

    @pytest.mark.asyncio
    async def test_initialize_prompt_capabilities(self):
        """Response declares prompt capabilities correctly."""
        agent = _make_agent()
        resp = await agent.initialize(protocol_version=1)

        assert resp.agent_capabilities is not None
        prompt_caps = resp.agent_capabilities.prompt_capabilities
        assert prompt_caps is not None
        # Currently supported capabilities
        # Text is implicit (always supported)
        # Future features:
        assert prompt_caps.image is False  # Future: multimodal LLM integration
        assert prompt_caps.audio is False  # Future: voice input
        assert prompt_caps.embedded_context is False  # Future: resource inclusion

    @pytest.mark.asyncio
    async def test_get_client_info_before_initialize(self):
        """get_client_info returns (None, None) before initialize."""
        agent = _make_agent()
        info, ver = agent.get_client_info()
        assert info is None
        assert ver is None


# ============================================================================
# TestACPNewSession
# ============================================================================


class TestACPNewSession:
    """Test session/new handling."""

    @pytest.mark.asyncio
    async def test_new_session_returns_session_id(self):
        """new_session creates a session and returns its ID."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                resp = await agent.new_session(cwd="/project")

        assert isinstance(resp, acp.NewSessionResponse)
        assert resp.session_id == "test-session-1"

    @pytest.mark.asyncio
    async def test_new_session_tracks_cwd(self):
        """new_session records the cwd for later use."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                await agent.new_session(cwd="/my/project")

        assert agent._sessions_cwd["test-session-1"] == "/my/project"
        assert agent._current_cwd == "/my/project"

    @pytest.mark.asyncio
    async def test_new_session_sets_default_mode(self):
        """new_session assigns the default mode to the new session."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                resp = await agent.new_session(cwd="/project")

        assert agent._sessions_mode["test-session-1"] == agent._default_mode_id
        assert resp.modes is not None

    @pytest.mark.asyncio
    async def test_new_session_with_models(self):
        """new_session populates model state when models are available."""
        agent = _make_agent(model="claude-sonnet-4-20250514")
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        model_info = MagicMock()
        model_info.model_id = "claude-sonnet-4-20250514"
        model_info.name = "Claude Sonnet"
        model_info.description = "Sonnet model"

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[model_info]):
                resp = await agent.new_session(cwd="/project")

        assert resp.models is not None
        assert resp.models.current_model_id == "claude-sonnet-4-20250514"
        assert len(resp.models.available_models) == 1

    @pytest.mark.asyncio
    async def test_new_session_sets_up_terminal_executor(self):
        """new_session wires up ACPTerminalExecutor on the session timeline."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                with patch(f"{MODULE}.ACPTerminalExecutor") as mock_executor_cls:
                    await agent.new_session(cwd="/project")

        mock_executor_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_session_saves_to_disk(self):
        """new_session persists the session immediately."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                await agent.new_session(cwd="/project")

        mock_session.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_session_error_raises_request_error(self):
        """new_session raises acp.RequestError on failure."""
        agent = _make_agent()
        agent._manager.create_session = AsyncMock(side_effect=RuntimeError("boom"))
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with pytest.raises(acp.RequestError) as exc_info:
                await agent.new_session(cwd="/project")

        assert exc_info.value.code == -32603
        assert "boom" in str(exc_info.value.data)

    @pytest.mark.asyncio
    async def test_new_session_marks_initialized(self):
        """new_session marks the session for post-setup."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}._find_jetbrains_chat_uuid", return_value=None):
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                await agent.new_session(cwd="/project")

        assert "test-session-1" in agent._sessions_initialized


# ============================================================================
# TestACPLoadSession
# ============================================================================


class TestACPLoadSession:
    """Test session/load handling."""

    @pytest.mark.asyncio
    async def test_load_already_in_memory(self):
        """load_session returns immediately if session is in memory."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        resp = await agent.load_session(cwd="/project", session_id="test-session-1")

        assert isinstance(resp, acp.LoadSessionResponse)

    @pytest.mark.asyncio
    async def test_load_from_disk(self):
        """load_session loads from disk when not in memory."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=None)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}.Session") as session_cls:
            session_cls.from_file = MagicMock(return_value=mock_session)
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                resp = await agent.load_session(cwd="/project", session_id="test-session-1")

        assert isinstance(resp, acp.LoadSessionResponse)

    @pytest.mark.asyncio
    async def test_load_session_not_found(self):
        """load_session returns None if session file does not exist."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        with patch(f"{MODULE}.Session") as session_cls:
            session_cls.from_file = MagicMock(return_value=None)
            resp = await agent.load_session(cwd="/project", session_id="nonexistent")

        assert resp is None

    @pytest.mark.asyncio
    async def test_load_session_tracks_metadata(self):
        """load_session records cwd and mode."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=None)
        agent._conn = _make_mock_conn()

        with patch(f"{MODULE}.Session") as session_cls:
            session_cls.from_file = MagicMock(return_value=mock_session)
            with patch(f"{MODULE}.get_available_models", return_value=[]):
                await agent.load_session(cwd="/loaded/project", session_id="test-session-1")

        assert agent._sessions_cwd["test-session-1"] == "/loaded/project"
        assert "test-session-1" in agent._sessions_mode

    @pytest.mark.asyncio
    async def test_load_session_exception_returns_none(self):
        """load_session returns None on exception."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(side_effect=RuntimeError("disk error"))

        resp = await agent.load_session(cwd="/project", session_id="bad")

        assert resp is None


# ============================================================================
# TestACPListSessions
# ============================================================================


class TestACPListSessions:
    """Test session/list handling."""

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        """list_sessions returns empty when no sessions exist."""
        agent = _make_agent()
        agent._current_cwd = "/project"

        with patch(f"{MODULE}.list_sessions_from_disk", return_value=[]):
            resp = await agent.list_sessions()

        assert resp.sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_returns_persisted(self):
        """list_sessions includes disk-persisted sessions."""
        agent = _make_agent()
        agent._current_cwd = "/project"

        meta = MagicMock()
        meta.session_id = "sid-123"
        meta.cwd = "/project"

        with patch(f"{MODULE}.list_sessions_from_disk", return_value=[meta]):
            resp = await agent.list_sessions()

        assert len(resp.sessions) == 1
        assert resp.sessions[0].session_id == "sid-123"

    @pytest.mark.asyncio
    async def test_list_sessions_no_cwd(self):
        """list_sessions returns empty when no cwd is available."""
        agent = _make_agent()
        agent._current_cwd = None

        resp = await agent.list_sessions()

        assert resp.sessions == []

    @pytest.mark.asyncio
    async def test_list_sessions_uses_provided_cwd(self):
        """list_sessions uses the explicit cwd parameter over stored one."""
        agent = _make_agent()
        agent._current_cwd = "/old"

        with patch(f"{MODULE}.list_sessions_from_disk", return_value=[]) as mock_list:
            await agent.list_sessions(cwd="/new")

        mock_list.assert_called_once_with("/new")


# ============================================================================
# TestACPSetSessionMode
# ============================================================================


class TestACPSetSessionMode:
    """Test session mode switching."""

    @pytest.mark.asyncio
    async def test_set_valid_mode(self):
        """set_session_mode succeeds for a valid mode."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        resp = await agent.set_session_mode(mode_id="normal", session_id="s1")

        assert resp is not None
        assert agent._sessions_mode["s1"] == "normal"
        mock_session.set_mode.assert_called_once_with("normal")

    @pytest.mark.asyncio
    async def test_set_invalid_mode_raises(self):
        """set_session_mode raises RequestError for an invalid mode."""
        agent = _make_agent()

        with pytest.raises(acp.RequestError) as exc_info:
            await agent.set_session_mode(mode_id="nonexistent", session_id="s1")

        assert exc_info.value.code == -32602

    @pytest.mark.asyncio
    async def test_set_mode_session_not_found(self):
        """set_session_mode works even if session not in manager (tracks metadata)."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        resp = await agent.set_session_mode(mode_id="plan", session_id="s1")

        assert resp is not None
        assert agent._sessions_mode["s1"] == "plan"


# ============================================================================
# TestACPSetSessionModel
# ============================================================================


class TestACPSetSessionModel:
    """Test session model switching."""

    @pytest.mark.asyncio
    async def test_set_model_success(self):
        """set_session_model creates new LLM provider and updates session."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        with patch(f"{MODULE}.LiteLLMProvider") as mock_provider:
            resp = await agent.set_session_model(model_id="gpt-4", session_id="s1")

        assert resp is not None
        mock_provider.assert_called_once_with("gpt-4")
        assert agent._sessions_model["s1"] == "gpt-4"
        assert agent._current_model_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_set_model_session_not_found(self):
        """set_session_model raises RequestError when session is missing."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        with pytest.raises(acp.RequestError) as exc_info:
            await agent.set_session_model(model_id="gpt-4", session_id="missing")

        assert exc_info.value.code == -32600


# ============================================================================
# TestACPPrompt
# ============================================================================


class TestACPPrompt:
    """Test prompt handling."""

    @pytest.mark.asyncio
    async def test_prompt_queues_user_message(self):
        """prompt() queues the message and returns end_turn."""
        agent = _make_agent()
        agent._out_of_band_update = True  # async mode - return immediately
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        prompt_blocks = [TextContentBlock(type="text", text="Hello")]
        resp = await agent.prompt(prompt=prompt_blocks, session_id="s1")

        assert resp.stop_reason == "end_turn"
        mock_session.queue_user_message.assert_called_once()
        call_args = mock_session.queue_user_message.call_args
        assert call_args[0][0] == "Hello"

    @pytest.mark.asyncio
    async def test_prompt_cancelled_session(self):
        """prompt() returns cancelled for a session in _closed_sessions."""
        agent = _make_agent()
        agent._closed_sessions.add("s1")

        resp = await agent.prompt(prompt=[], session_id="s1")

        assert resp.stop_reason == "cancelled"

    @pytest.mark.asyncio
    async def test_prompt_session_not_found(self):
        """prompt() raises RequestError when session is not found."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        with pytest.raises(acp.RequestError) as exc_info:
            await agent.prompt(
                prompt=[TextContentBlock(type="text", text="Hi")],
                session_id="missing",
            )

        assert exc_info.value.code == -32600

    @pytest.mark.asyncio
    async def test_prompt_extracts_text_from_blocks(self):
        """prompt() concatenates text from multiple TextContentBlock items."""
        agent = _make_agent()
        agent._out_of_band_update = True
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        blocks = [
            TextContentBlock(type="text", text="Hello "),
            TextContentBlock(type="text", text="world"),
        ]
        await agent.prompt(prompt=blocks, session_id="s1")

        call_args = mock_session.queue_user_message.call_args
        assert call_args[0][0] == "Hello world"

    @pytest.mark.asyncio
    async def test_prompt_routes_to_conversation_transport(self):
        """prompt() routes to active conversation transport if present."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        mock_transport = MagicMock()
        mock_transport.handle_input_response = MagicMock()
        agent._active_transports["s1"] = mock_transport

        blocks = [TextContentBlock(type="text", text="user-input")]
        resp = await agent.prompt(prompt=blocks, session_id="s1")

        assert resp.stop_reason == "end_turn"
        mock_transport.handle_input_response.assert_called_once_with("user-input")
        mock_session.queue_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_prompt_sync_mode_waits_for_completion(self):
        """In sync mode, prompt() waits for message_complete_event."""
        agent = _make_agent()
        agent._out_of_band_update = False
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        # We'll manually trigger the completion event via a background task
        async def trigger_completion():
            await asyncio.sleep(0.01)
            # Find the message ID from the completion events
            for _msg_id, event in agent._message_complete_events.items():
                event.set()
                break

        blocks = [TextContentBlock(type="text", text="Hello")]

        # Start trigger in background
        trigger_task = asyncio.create_task(trigger_completion())
        resp = await agent.prompt(prompt=blocks, session_id="s1")
        await trigger_task

        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_prompt_sync_mode_cancelled_during_wait(self):
        """In sync mode, prompt() returns cancelled when session closes during wait."""
        agent = _make_agent()
        agent._out_of_band_update = False
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        async def close_session():
            await asyncio.sleep(0.01)
            agent._closed_sessions.add("s1")
            # Set completion event to unblock
            for _msg_id, event in agent._message_complete_events.items():
                event.set()
                break

        blocks = [TextContentBlock(type="text", text="Hello")]
        close_task = asyncio.create_task(close_session())
        resp = await agent.prompt(prompt=blocks, session_id="s1")
        await close_task

        assert resp.stop_reason == "cancelled"


# ============================================================================
# TestACPSlashCommands
# ============================================================================


class TestACPSlashCommands:
    """Test slash command dispatch."""

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Test /help returns command listing."""
        agent = _make_agent()
        handled, response = await agent._handle_slash_command("/help", "s1")

        assert handled is True
        assert "Available commands" in response
        assert "/help" in response
        assert "/clear" in response
        assert "/context" in response

    @pytest.mark.asyncio
    async def test_clear_command(self):
        """Test /clear clears message history."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        handled, response = await agent._handle_slash_command("/clear", "s1")

        assert handled is True
        assert "cleared" in response.lower()
        mock_session.clear_message_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_command_with_objects(self):
        """Test /context lists context objects."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        mock_obj = MagicMock()
        mock_obj.GetDigest.return_value = {"type": "TextNode", "path": "main.py"}
        mock_session.get_context_objects.return_value = {"v1": mock_obj}
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        handled, response = await agent._handle_slash_command("/context", "s1")

        assert handled is True
        assert "v1" in response
        assert "TextNode" in response

    @pytest.mark.asyncio
    async def test_context_command_empty(self):
        """Test /context when no context objects exist."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        mock_session.get_context_objects.return_value = {}
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        handled, response = await agent._handle_slash_command("/context", "s1")

        assert handled is True
        assert "No context objects" in response

    @pytest.mark.asyncio
    async def test_title_command_get(self):
        """Test /title without args shows current title."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        mock_session.title = "My Title"
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        handled, response = await agent._handle_slash_command("/title", "s1")

        assert handled is True
        assert "My Title" in response

    @pytest.mark.asyncio
    async def test_title_command_set(self):
        """Test /title with args sets new title."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        handled, response = await agent._handle_slash_command("/title New Title", "s1")

        assert handled is True
        mock_session.set_title.assert_called_once_with("New Title")
        assert "New Title" in response

    @pytest.mark.asyncio
    async def test_unknown_command_passes_through(self):
        """Unknown slash commands are not handled (pass to LLM)."""
        agent = _make_agent()

        handled, response = await agent._handle_slash_command("/foobar", "s1")

        assert handled is False
        assert response == ""

    @pytest.mark.asyncio
    async def test_slash_command_in_prompt(self):
        """Slash commands get handled synchronously in prompt()."""
        agent = _make_agent()
        agent._out_of_band_update = True
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)
        agent._conn = _make_mock_conn()

        blocks = [TextContentBlock(type="text", text="/help")]
        resp = await agent.prompt(prompt=blocks, session_id="s1")

        assert resp.stop_reason == "end_turn"
        # /help should NOT queue a user message
        mock_session.queue_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_command_session_not_found(self):
        """Test /context when session is not found."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        handled, response = await agent._handle_slash_command("/context", "s1")

        assert handled is True
        assert "Session not found" in response

    @pytest.mark.asyncio
    async def test_clear_command_session_not_found(self):
        """Test /clear when session is not found -- still returns handled."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        handled, response = await agent._handle_slash_command("/clear", "s1")

        assert handled is True

    @pytest.mark.asyncio
    async def test_title_set_saves_session(self):
        """Setting title via /title saves the session to disk."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        await agent._handle_slash_command("/title Saved Title", "s1")

        mock_session.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_skill_command_list_skills(self):
        """Test /skill without args lists available skills."""
        agent = _make_agent()

        # Mock discover_skills to return test skills
        with patch("activecontext.skills.discover_skills") as mock_discover:
            mock_skill1 = Mock()
            mock_skill1.name = "skill-a"
            mock_skill1.description = "First skill"
            mock_skill2 = Mock()
            mock_skill2.name = "skill-b"
            mock_skill2.description = "Second skill"
            mock_discover.return_value = [mock_skill1, mock_skill2]

            handled, response = await agent._handle_slash_command("/skill", "s1")

            assert handled is True
            assert "Available skills" in response
            assert "skill-a" in response
            assert "skill-b" in response
            mock_discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_skill_command_no_skills(self):
        """Test /skill when no skills are available."""
        agent = _make_agent()

        with patch("activecontext.skills.discover_skills", return_value=[]):
            handled, response = await agent._handle_slash_command("/skill", "s1")

            assert handled is True
            assert "No skills available" in response

    @pytest.mark.asyncio
    async def test_skill_command_load_skill(self):
        """Test /skill <name> loads and injects skill content."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        mock_session.add_node = MagicMock(return_value="skill-node-1")
        agent._manager.get_session = AsyncMock(return_value=mock_session)
        agent._send_session_update = AsyncMock()

        # Mock load_skill to return a test skill
        with patch("activecontext.skills.load_skill") as mock_load:
            mock_manifest = Mock()
            mock_manifest.name = "test-skill"
            mock_manifest.description = "A test skill"
            mock_manifest.content = "# Test Skill\n\nInstructions here."
            mock_load.return_value = mock_manifest

            handled, response = await agent._handle_slash_command("/skill test-skill", "s1")

            assert handled is True
            mock_load.assert_called_once_with("test-skill")
            mock_session.add_node.assert_called_once()

            # Verify MessageNode was created with correct properties
            call_args = mock_session.add_node.call_args
            node = call_args[0][0]
            from activecontext.context.nodes import MessageRole

            assert node.role == MessageRole.USER
            assert "test-skill" in node.content
            assert node.originator == "skill:test-skill"

            # Verify acknowledgement was sent
            agent._send_session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_skill_command_skill_not_found(self):
        """Test /skill with unknown skill name."""
        agent = _make_agent()

        with patch("activecontext.skills.load_skill", return_value=None):
            handled, response = await agent._handle_slash_command("/skill unknown", "s1")

            assert handled is True
            assert "not found" in response
            assert "unknown" in response

    @pytest.mark.asyncio
    async def test_skill_command_with_args(self):
        """Test /skill <name> <args> passes arguments to skill."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        mock_session.add_node = MagicMock(return_value="skill-node-1")
        agent._manager.get_session = AsyncMock(return_value=mock_session)
        agent._send_session_update = AsyncMock()

        with patch("activecontext.skills.load_skill") as mock_load:
            mock_manifest = Mock()
            mock_manifest.name = "test-skill"
            mock_manifest.description = "A test skill"
            mock_manifest.content = "# Test Skill\n\nInstructions here."
            mock_load.return_value = mock_manifest

            handled, response = await agent._handle_slash_command(
                "/skill test-skill arg1 arg2", "s1"
            )

            assert handled is True
            # Verify arguments are included in skill content
            call_args = mock_session.add_node.call_args
            node = call_args[0][0]
            assert "arg1 arg2" in node.content


# ============================================================================
# TestACPDashboardCommand
# ============================================================================


class TestACPDashboardCommand:
    """Test /dashboard subcommands."""

    @pytest.mark.asyncio
    async def test_dashboard_unknown_subcommand(self):
        """Test /dashboard with unknown subcommand shows usage."""
        agent = _make_agent()

        handled, response = await agent._handle_slash_command("/dashboard badcmd", "s1")

        assert handled is True
        assert "Usage" in response

    @pytest.mark.asyncio
    async def test_dashboard_status_not_running(self):
        """Test /dashboard status when dashboard is not running."""
        agent = _make_agent()

        with patch("activecontext.dashboard.server.is_dashboard_running", return_value=False):
            handled, response = await agent._handle_slash_command("/dashboard status", "s1")

        assert handled is True
        assert "not running" in response.lower()

    @pytest.mark.asyncio
    async def test_dashboard_stop_not_running(self):
        """Test /dashboard stop when dashboard is not running."""
        agent = _make_agent()

        with patch("activecontext.dashboard.server.is_dashboard_running", return_value=False):
            handled, response = await agent._handle_slash_command("/dashboard stop", "s1")

        assert handled is True
        assert "not running" in response.lower()

    @pytest.mark.asyncio
    async def test_dashboard_start_invalid_port(self):
        """Test /dashboard start with invalid port number."""
        agent = _make_agent()

        with patch("activecontext.dashboard.server.is_dashboard_running", return_value=False):
            handled, response = await agent._handle_slash_command("/dashboard start abc", "s1")

        assert handled is True
        assert "Invalid port" in response


# ============================================================================
# TestDashboardAutoStart
# ============================================================================


class TestDashboardAutoStart:
    """Test dashboard auto-start via config."""

    @pytest.mark.asyncio
    async def test_auto_start_when_config_enabled(self):
        """Should auto-start dashboard when config.dashboard.auto_start=True."""
        from activecontext.config.schema import DashboardConfig

        agent = _make_agent()

        mock_config = MagicMock()
        mock_config.dashboard = DashboardConfig(auto_start=True, port=31993)

        with (
            patch("activecontext.config.get_config", return_value=mock_config),
            patch("activecontext.dashboard.is_dashboard_running", return_value=False),
            patch("activecontext.dashboard.start_dashboard", new_callable=AsyncMock) as mock_start,
        ):
            await agent._auto_start_dashboard_if_needed()

            mock_start.assert_called_once()
            assert mock_start.call_args.kwargs["port"] == 31993
            assert mock_start.call_args.kwargs["transport_type"] == "acp"

    @pytest.mark.asyncio
    async def test_no_auto_start_when_config_disabled(self):
        """Should not auto-start when config.dashboard.auto_start=False."""
        from activecontext.config.schema import DashboardConfig

        agent = _make_agent()

        mock_config = MagicMock()
        mock_config.dashboard = DashboardConfig(auto_start=False)

        with (
            patch("activecontext.config.get_config", return_value=mock_config),
            patch("activecontext.dashboard.start_dashboard", new_callable=AsyncMock) as mock_start,
        ):
            await agent._auto_start_dashboard_if_needed()

            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_auto_start_when_already_running(self):
        """Should skip auto-start if dashboard already running."""
        from activecontext.config.schema import DashboardConfig

        agent = _make_agent()

        mock_config = MagicMock()
        mock_config.dashboard = DashboardConfig(auto_start=True, port=31993)

        with (
            patch("activecontext.config.get_config", return_value=mock_config),
            patch("activecontext.dashboard.is_dashboard_running", return_value=True),
            patch("activecontext.dashboard.start_dashboard", new_callable=AsyncMock) as mock_start,
        ):
            await agent._auto_start_dashboard_if_needed()

            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_start_handles_port_conflict(self):
        """Should not crash when port is in use."""
        from activecontext.config.schema import DashboardConfig

        agent = _make_agent()

        mock_config = MagicMock()
        mock_config.dashboard = DashboardConfig(auto_start=True, port=31993)

        with (
            patch("activecontext.config.get_config", return_value=mock_config),
            patch("activecontext.dashboard.is_dashboard_running", return_value=False),
            patch(
                "activecontext.dashboard.start_dashboard",
                new_callable=AsyncMock,
                side_effect=OSError("Address already in use"),
            ),
        ):
            # Should not raise
            await agent._auto_start_dashboard_if_needed()

    @pytest.mark.asyncio
    async def test_auto_start_custom_port(self):
        """Should use configured port."""
        from activecontext.config.schema import DashboardConfig

        agent = _make_agent()

        mock_config = MagicMock()
        mock_config.dashboard = DashboardConfig(auto_start=True, port=9999)

        with (
            patch("activecontext.config.get_config", return_value=mock_config),
            patch("activecontext.dashboard.is_dashboard_running", return_value=False),
            patch("activecontext.dashboard.start_dashboard", new_callable=AsyncMock) as mock_start,
        ):
            await agent._auto_start_dashboard_if_needed()

            assert mock_start.call_args.kwargs["port"] == 9999


# ============================================================================
# TestACPCancel
# ============================================================================


class TestACPCancel:
    """Test session/cancel handling."""

    @pytest.mark.asyncio
    async def test_cancel_marks_session_closed(self):
        """cancel() adds session to _closed_sessions."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        await agent.cancel(session_id="s1")

        # _cleanup_closed_session removes it, so it should be cleaned up
        # but the session cancel was called
        mock_session.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_cancels_active_prompt(self):
        """cancel() cancels any active prompt task."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        # Create a mock active prompt task (use MagicMock, not AsyncMock, for Task-like)
        mock_task = MagicMock()
        mock_task.done = MagicMock(return_value=False)
        mock_task.cancel = MagicMock()
        # Make it awaitable for asyncio.shield/wait_for
        future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        mock_task.__await__ = future.__await__
        agent._active_prompts["s1"] = mock_task

        await agent.cancel(session_id="s1")

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_cleans_up_metadata(self):
        """cancel() cleans up session tracking metadata."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        agent._sessions_cwd["s1"] = "/project"
        agent._sessions_model["s1"] = "gpt-4"
        agent._sessions_mode["s1"] = "normal"

        await agent.cancel(session_id="s1")

        assert "s1" not in agent._sessions_cwd
        assert "s1" not in agent._sessions_model
        assert "s1" not in agent._sessions_mode

    @pytest.mark.asyncio
    async def test_cancel_session_not_found(self):
        """cancel() does not crash when session is not in manager."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        # Should not raise
        await agent.cancel(session_id="nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_cleans_up_chunk_buffers(self):
        """cancel() clears any pending chunk buffers for the session."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)
        agent._nagle._buffers["s1"] = "pending text"

        await agent.cancel(session_id="s1")

        assert not agent._nagle.has_buffered("s1")

    @pytest.mark.asyncio
    async def test_cancel_keeps_session_in_closed_set(self):
        """Cancel keeps session in _closed_sessions so prompt handler can detect cancellation.

        The session is removed from _closed_sessions by the prompt handler after
        it returns stop_reason="cancelled", not during cancel cleanup.
        """
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        await agent.cancel(session_id="s1")

        # Session stays in _closed_sessions until prompt handler consumes it
        assert "s1" in agent._closed_sessions


# ============================================================================
# TestACPForkAndResume
# ============================================================================


class TestACPForkAndResume:
    """Test fork_session and resume_session."""

    @pytest.mark.asyncio
    async def test_fork_session_not_implemented(self):
        """fork_session raises RequestError (not implemented)."""
        agent = _make_agent()

        with pytest.raises(acp.RequestError) as exc_info:
            await agent.fork_session(cwd="/project", session_id="s1")

        assert exc_info.value.code == -32601

    @pytest.mark.asyncio
    async def test_resume_session_success(self):
        """resume_session succeeds when session exists."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)

        resp = await agent.resume_session(cwd="/project", session_id="s1")

        assert resp is not None

    @pytest.mark.asyncio
    async def test_resume_session_not_found(self):
        """resume_session raises RequestError when session is missing."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)

        with pytest.raises(acp.RequestError) as exc_info:
            await agent.resume_session(cwd="/project", session_id="missing")

        assert exc_info.value.code == -32600


# ============================================================================
# TestACPAuthenticate
# ============================================================================


class TestACPAuthenticate:
    """Test authenticate method."""

    @pytest.mark.asyncio
    async def test_authenticate_returns_none(self):
        """authenticate() is not implemented and returns None."""
        agent = _make_agent()

        resp = await agent.authenticate(method_id="basic")

        assert resp is None


# ============================================================================
# TestACPExtMethods
# ============================================================================


class TestACPExtMethods:
    """Test extension method/notification handlers."""

    @pytest.mark.asyncio
    async def test_ext_method_returns_empty(self):
        """ext_method returns an empty dict."""
        agent = _make_agent()

        result = await agent.ext_method(method="custom/foo", params={"bar": 1})

        assert result == {}

    @pytest.mark.asyncio
    async def test_ext_notification_does_not_raise(self):
        """ext_notification completes without error."""
        agent = _make_agent()

        # Should not raise
        await agent.ext_notification(method="custom/event", params={})


# ============================================================================
# TestACPPermissionRequests
# ============================================================================


class TestACPPermissionWiring:
    """Test that permission requesters are wired into sessions."""

    @pytest.mark.asyncio
    async def test_create_session_passes_permission_requesters(self):
        """create_session should receive agent's permission callbacks."""
        agent = _make_agent()
        agent._conn = MagicMock()
        mock_session = _make_mock_session()
        agent._manager.create_session = AsyncMock(return_value=mock_session)
        agent._manager.get_session = AsyncMock(return_value=None)

        # Patch post-setup to avoid side effects
        with (
            patch.object(agent, "_post_session_setup", new_callable=AsyncMock),
            patch.object(agent, "_start_agent_loop", new_callable=AsyncMock),
            patch.object(agent, "_setup_conversation_callbacks"),
        ):
            await agent.new_session(cwd=".")

        call_kwargs = agent._manager.create_session.call_args.kwargs
        assert call_kwargs["permission_requester"] is not None
        assert call_kwargs["shell_permission_requester"] is not None
        assert call_kwargs["website_permission_requester"] is not None
        assert call_kwargs["import_permission_requester"] is not None
        # Verify they're the right methods (bound methods compare equal)
        assert call_kwargs["permission_requester"] == agent._request_file_permission
        assert call_kwargs["shell_permission_requester"] == agent._request_shell_permission

    def test_wire_permission_requesters_sets_timeline_callbacks(self):
        """_wire_permission_requesters should set all 4 timeline callbacks."""
        agent = _make_agent()
        mock_session = MagicMock()
        mock_session.timeline = MagicMock()
        mock_session.timeline._permission_requester = None
        mock_session.timeline._shell_permission_requester = None
        mock_session.timeline._website_permission_requester = None
        mock_session.timeline._import_permission_requester = None

        agent._wire_permission_requesters(mock_session)

        assert mock_session.timeline._permission_requester is not None
        assert mock_session.timeline._shell_permission_requester is not None
        assert mock_session.timeline._website_permission_requester is not None
        assert mock_session.timeline._import_permission_requester is not None


class TestACPPermissionRequests:
    """Test file/shell/website/import permission request flows."""

    @pytest.mark.asyncio
    async def test_file_permission_no_conn(self):
        """_request_file_permission returns denied when no connection."""
        agent = _make_agent()
        agent._conn = None

        granted, persist = await agent._request_file_permission("s1", "/file.txt", "read")

        assert granted is False
        assert persist is False

    @pytest.mark.asyncio
    async def test_file_permission_allow_once(self):
        """_request_file_permission returns (True, False) for allow_once."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "allow_once"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist = await agent._request_file_permission("s1", "/file.txt", "read")

        assert granted is True
        assert persist is False

    @pytest.mark.asyncio
    async def test_file_permission_allow_always(self):
        """_request_file_permission returns (True, True) for allow_always."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "allow_always"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist = await agent._request_file_permission("s1", "/file.txt", "write")

        assert granted is True
        assert persist is True

    @pytest.mark.asyncio
    async def test_file_permission_denied(self):
        """_request_file_permission returns denied when user rejects."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "deny"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist = await agent._request_file_permission("s1", "/file.txt", "read")

        assert granted is False
        assert persist is False

    @pytest.mark.asyncio
    async def test_file_permission_exception(self):
        """_request_file_permission returns denied on exception."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        conn.request_permission.side_effect = RuntimeError("network error")

        granted, persist = await agent._request_file_permission("s1", "/file.txt", "read")

        assert granted is False
        assert persist is False

    @pytest.mark.asyncio
    async def test_shell_permission_no_conn(self):
        """_request_shell_permission returns denied when no connection."""
        agent = _make_agent()
        agent._conn = None

        granted, persist = await agent._request_shell_permission("s1", "rm", ["-rf", "/"])

        assert granted is False
        assert persist is False

    @pytest.mark.asyncio
    async def test_shell_permission_allow_once(self):
        """_request_shell_permission grants once correctly."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "allow_once"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist = await agent._request_shell_permission("s1", "git", ["status"])

        assert granted is True
        assert persist is False

    @pytest.mark.asyncio
    async def test_shell_permission_exception(self):
        """_request_shell_permission returns denied on exception."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        conn.request_permission.side_effect = RuntimeError("timeout")

        granted, persist = await agent._request_shell_permission("s1", "cmd", None)

        assert granted is False

    @pytest.mark.asyncio
    async def test_website_permission_no_conn(self):
        """_request_website_permission returns denied when no connection."""
        agent = _make_agent()
        agent._conn = None

        granted, persist = await agent._request_website_permission(
            "s1", "https://example.com", "GET"
        )

        assert granted is False

    @pytest.mark.asyncio
    async def test_website_permission_allow_always(self):
        """_request_website_permission grants always correctly."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "allow_always"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist = await agent._request_website_permission(
            "s1", "https://api.example.com/data", "POST"
        )

        assert granted is True
        assert persist is True

    @pytest.mark.asyncio
    async def test_website_permission_exception(self):
        """_request_website_permission returns denied on exception."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        conn.request_permission.side_effect = RuntimeError("err")

        granted, persist = await agent._request_website_permission(
            "s1", "https://example.com", "GET"
        )

        assert granted is False

    @pytest.mark.asyncio
    async def test_import_permission_no_conn(self):
        """_request_import_permission returns denied when no connection."""
        agent = _make_agent()
        agent._conn = None

        granted, persist, submod = await agent._request_import_permission("s1", "numpy")

        assert granted is False
        assert persist is False
        assert submod is False

    @pytest.mark.asyncio
    async def test_import_permission_allow_once(self):
        """_request_import_permission grants once correctly."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "allow_once"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist, submod = await agent._request_import_permission("s1", "os.path")

        assert granted is True
        assert persist is False
        assert submod is False

    @pytest.mark.asyncio
    async def test_import_permission_allow_always_with_submodules(self):
        """_request_import_permission handles allow_always_submodules."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "allow_always_submodules"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist, submod = await agent._request_import_permission("s1", "pandas.core")

        assert granted is True
        assert persist is True
        assert submod is True

    @pytest.mark.asyncio
    async def test_import_permission_denied(self):
        """_request_import_permission returns denied when user rejects."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        outcome = MagicMock()
        outcome.outcome = "selected"
        outcome.option_id = "deny"
        conn.request_permission.return_value = MagicMock(outcome=outcome)

        granted, persist, submod = await agent._request_import_permission("s1", "subprocess")

        assert granted is False
        assert persist is False
        assert submod is False

    @pytest.mark.asyncio
    async def test_import_permission_exception(self):
        """_request_import_permission returns denied on exception."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        conn.request_permission.side_effect = RuntimeError("err")

        granted, persist, submod = await agent._request_import_permission("s1", "os")

        assert granted is False


# ============================================================================
# TestACPConversationDelegation
# ============================================================================


class TestACPConversationDelegation:
    """Test conversation transport registration and routing."""

    def test_register_transport(self):
        """register_conversation_transport stores the transport."""
        agent = _make_agent()
        mock_transport = MagicMock()

        agent.register_conversation_transport("s1", mock_transport)

        assert agent._active_transports["s1"] is mock_transport

    def test_unregister_transport(self):
        """unregister_conversation_transport removes the transport."""
        agent = _make_agent()
        agent._active_transports["s1"] = MagicMock()

        agent.unregister_conversation_transport("s1")

        assert "s1" not in agent._active_transports

    def test_unregister_transport_idempotent(self):
        """unregister_conversation_transport is safe when transport not registered."""
        agent = _make_agent()

        # Should not raise
        agent.unregister_conversation_transport("nonexistent")

    @pytest.mark.asyncio
    async def test_handle_input_response_routes_to_transport(self):
        """handle_conversation_input_response routes to registered transport."""
        agent = _make_agent()
        mock_transport = MagicMock()
        mock_transport.handle_input_response = MagicMock()
        agent._active_transports["s1"] = mock_transport

        await agent.handle_conversation_input_response("s1", "user typed this")

        mock_transport.handle_input_response.assert_called_once_with("user typed this")

    @pytest.mark.asyncio
    async def test_handle_input_response_no_transport(self):
        """handle_conversation_input_response is safe when no transport registered."""
        agent = _make_agent()

        # Should not raise, just log warning
        await agent.handle_conversation_input_response("s1", "dropped")

    def test_setup_conversation_callbacks(self):
        """_setup_conversation_callbacks wires callbacks on session."""
        agent = _make_agent()
        mock_session = _make_mock_session()

        agent._setup_conversation_callbacks(mock_session)

        assert mock_session._emit_update_callback is not None
        assert mock_session._register_transport_callback is not None
        assert mock_session._unregister_transport_callback is not None


# ============================================================================
# TestACPOnConnect
# ============================================================================


class TestACPOnConnect:
    """Test on_connect handler."""

    def test_on_connect_stores_client(self):
        """on_connect() stores the client reference."""
        agent = _make_agent()
        mock_conn = MagicMock()

        agent.on_connect(mock_conn)

        assert agent._conn is mock_conn


# ============================================================================
# TestACPSendSessionUpdate
# ============================================================================


class TestACPSendSessionUpdate:
    """Test _send_session_update gating."""

    @pytest.mark.asyncio
    async def test_send_update_no_conn(self):
        """_send_session_update does nothing when no connection."""
        agent = _make_agent()
        agent._conn = None

        # Should not raise
        await agent._send_session_update("s1", MagicMock())

    @pytest.mark.asyncio
    async def test_send_update_closed_session(self):
        """_send_session_update drops updates for closed sessions."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        agent._closed_sessions.add("s1")

        await agent._send_session_update("s1", MagicMock())

        conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_update_success(self):
        """_send_session_update forwards to conn.session_update."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        update = MagicMock()

        await agent._send_session_update("s1", update)

        conn.session_update.assert_called_once_with("s1", update)


# ============================================================================
# TestACPNagleBatching
# ============================================================================


class TestACPNagleBatching:
    """Test Nagle-style chunk buffering via NagleBuffer delegation."""

    @pytest.mark.asyncio
    async def test_buffer_discards_for_closed_session(self):
        """_buffer_chunk discards text when session is closed."""
        agent = _make_agent()
        await agent._nagle.close("s1")

        await agent._buffer_chunk("s1", "should be dropped")

        assert not agent._nagle.has_buffered("s1")

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self):
        """_flush_chunks clears the buffer and sends update."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        agent._nagle._buffers["s1"] = "accumulated"

        await agent._flush_chunks("s1")

        assert not agent._nagle.has_buffered("s1")
        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_skips_closed_session(self):
        """_flush_chunks skips send for closed sessions."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn
        agent._nagle._buffers["s1"] = "text"
        await agent._nagle.close("s1")

        await agent._flush_chunks("s1")

        conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self):
        """_flush_chunks does nothing for empty buffer."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        await agent._flush_chunks("s1")

        conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_session_buffers(self):
        """_cleanup_session_buffers closes the key in NagleBuffer."""
        agent = _make_agent()
        agent._nagle._buffers["s1"] = "leftover"

        await agent._cleanup_session_buffers("s1")

        assert not agent._nagle.has_buffered("s1")

    @pytest.mark.asyncio
    async def test_delayed_flush_sleeps_then_flushes(self):
        """NagleBuffer timer flush works end-to-end."""
        agent = _make_agent()
        agent._nagle.flush_interval = 0.01
        conn = _make_mock_conn()
        agent._conn = conn
        agent._nagle._buffers["s1"] = "delayed text"

        await agent._nagle._delayed_flush("s1")

        assert not agent._nagle.has_buffered("s1")

    @pytest.mark.asyncio
    async def test_buffer_threshold_triggers_immediate_flush(self):
        """_buffer_chunk flushes immediately when threshold is exceeded."""
        agent = _make_agent()
        agent._nagle.flush_threshold = 5
        conn = _make_mock_conn()
        agent._conn = conn

        await agent._buffer_chunk("s1", "abcdef")  # 6 chars > threshold of 5

        assert not agent._nagle.has_buffered("s1")
        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffer_schedules_delayed_flush(self):
        """_buffer_chunk schedules a delayed flush for small chunks."""
        agent = _make_agent()
        agent._nagle.flush_threshold = 100
        agent._nagle.flush_interval = 10  # Very long so it won't fire during test
        conn = _make_mock_conn()
        agent._conn = conn

        await agent._buffer_chunk("s1", "short")

        assert agent._nagle.has_buffered("s1")
        assert agent._nagle.has_pending_flush("s1")

        # Clean up
        await agent._nagle.close("s1")


# ============================================================================
# TestACPUpdateEmission
# ============================================================================


class TestACPUpdateEmission:
    """Test _emit_update and update queueing."""

    @pytest.mark.asyncio
    async def test_emit_queues_when_not_in_prompt(self):
        """When out_of_band_update=False and not in-prompt, updates are queued."""
        agent = _make_agent()
        agent._out_of_band_update = False
        agent._conn = _make_mock_conn()

        update = MagicMock()
        update.kind = MagicMock()
        update.kind.value = "test"
        update.payload = {}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update("s1", update)

        assert "s1" in agent._queued_updates
        assert len(agent._queued_updates["s1"]) == 1

    @pytest.mark.asyncio
    async def test_emit_sends_when_in_prompt(self):
        """When in-prompt, updates are sent immediately regardless of mode."""
        agent = _make_agent()
        agent._out_of_band_update = False
        agent._in_prompt.add("s1")
        conn = _make_mock_conn()
        agent._conn = conn

        from activecontext.session.protocols import UpdateKind

        update = MagicMock()
        update.kind = UpdateKind.RESPONSE_CHUNK
        update.payload = {"text": "hi"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update("s1", update)

        # Should have been sent (either via buffer or direct)
        # Not queued
        assert "s1" not in agent._queued_updates

    @pytest.mark.asyncio
    async def test_flush_queued_updates(self):
        """_flush_queued_updates sends all queued updates."""
        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update1 = MagicMock()
        update1.kind = MagicMock()
        update1.kind.value = "test"
        update1.payload = {}
        update1.timestamp = 0
        update2 = MagicMock()
        update2.kind = MagicMock()
        update2.kind.value = "test2"
        update2.payload = {}
        update2.timestamp = 0
        agent._queued_updates["s1"] = [update1, update2]

        with patch.object(agent, "_emit_update_internal", new_callable=AsyncMock) as mock_emit:
            await agent._flush_queued_updates("s1")

        assert mock_emit.call_count == 2
        assert "s1" not in agent._queued_updates

    @pytest.mark.asyncio
    async def test_flush_queued_updates_empty(self):
        """_flush_queued_updates is a no-op when nothing is queued."""
        agent = _make_agent()

        # Should not raise
        await agent._flush_queued_updates("s1")

    def test_queue_update(self):
        """_queue_update appends to the queue for a session."""
        agent = _make_agent()
        update = MagicMock()
        update.kind = MagicMock()

        agent._queue_update("s1", update)
        agent._queue_update("s1", update)

        assert len(agent._queued_updates["s1"]) == 2


# ============================================================================
# TestACPAgentLoop
# ============================================================================


class TestACPAgentLoop:
    """Test agent loop start/stop."""

    @pytest.mark.asyncio
    async def test_start_agent_loop_creates_task(self):
        """_start_agent_loop creates a background task."""
        agent = _make_agent()
        mock_session = _make_mock_session()

        await agent._start_agent_loop(mock_session)

        assert "test-session-1" in agent._agent_loop_tasks
        # Clean up
        agent._agent_loop_tasks["test-session-1"].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await agent._agent_loop_tasks["test-session-1"]

    @pytest.mark.asyncio
    async def test_start_agent_loop_idempotent(self):
        """_start_agent_loop does not start a second loop if one is running."""
        agent = _make_agent()
        mock_session = _make_mock_session()

        await agent._start_agent_loop(mock_session)
        first_task = agent._agent_loop_tasks["test-session-1"]
        await agent._start_agent_loop(mock_session)
        second_task = agent._agent_loop_tasks["test-session-1"]

        assert first_task is second_task

        # Clean up
        first_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first_task

    def test_stop_agent_loop_cancels_task(self):
        """_stop_agent_loop cancels the background task."""
        agent = _make_agent()
        mock_task = MagicMock()
        mock_task.done.return_value = False
        mock_task.cancel = MagicMock()
        agent._agent_loop_tasks["s1"] = mock_task

        agent._stop_agent_loop("s1")

        mock_task.cancel.assert_called_once()

    def test_stop_agent_loop_no_task(self):
        """_stop_agent_loop is a no-op when no task exists."""
        agent = _make_agent()

        # Should not raise
        agent._stop_agent_loop("nonexistent")


# ============================================================================
# TestACPAvailableCommands
# ============================================================================


class TestACPAvailableCommands:
    """Test _get_available_commands output."""

    def test_commands_have_names(self):
        """All available commands have non-empty names."""
        agent = _make_agent()
        commands = agent._get_available_commands()

        for cmd in commands:
            assert cmd.name
            assert isinstance(cmd.name, str)
            assert len(cmd.name) > 0

    def test_commands_have_descriptions(self):
        """All available commands have descriptions."""
        agent = _make_agent()
        commands = agent._get_available_commands()

        for cmd in commands:
            assert cmd.description
            assert isinstance(cmd.description, str)

    def test_title_command_has_input(self):
        """The 'title' command declares unstructured input."""
        agent = _make_agent()
        commands = agent._get_available_commands()

        title_cmd = next(c for c in commands if c.name == "title")
        assert title_cmd.input is not None

    def test_dashboard_command_has_input(self):
        """The 'dashboard' command declares unstructured input."""
        agent = _make_agent()
        commands = agent._get_available_commands()

        dash_cmd = next(c for c in commands if c.name == "dashboard")
        assert dash_cmd.input is not None


# ============================================================================
# TestACPCleanupClosedSession
# ============================================================================


class TestACPCleanupClosedSession:
    """Test _cleanup_closed_session."""

    def test_cleanup_removes_metadata(self):
        """_cleanup_closed_session removes metadata but keeps _closed_sessions entry.

        The _closed_sessions entry is kept so the prompt handler can detect
        cancellation and return stop_reason="cancelled". The prompt handler
        is responsible for discarding from _closed_sessions after consuming it.
        """
        agent = _make_agent()
        agent._sessions_cwd["s1"] = "/a"
        agent._sessions_model["s1"] = "m"
        agent._sessions_mode["s1"] = "normal"
        agent._closed_sessions.add("s1")

        agent._cleanup_closed_session("s1")

        assert "s1" not in agent._sessions_cwd
        assert "s1" not in agent._sessions_model
        assert "s1" not in agent._sessions_mode
        # _closed_sessions entry is intentionally kept for prompt handler
        assert "s1" in agent._closed_sessions

    def test_cleanup_idempotent(self):
        """_cleanup_closed_session is safe to call twice."""
        agent = _make_agent()
        agent._cleanup_closed_session("s1")
        agent._cleanup_closed_session("s1")  # Should not raise


# ============================================================================
# TestACPJetBrainsChatUUID
# ============================================================================


class TestACPJetBrainsChatUUID:
    """Test _find_jetbrains_chat_uuid helper."""

    def test_disabled_without_env_var(self):
        """Returns None when AC_CLIENT_JETBRAINS is not set."""
        from activecontext.transport.acp.agent import _find_jetbrains_chat_uuid

        with patch.dict("os.environ", {}, clear=True):
            result = _find_jetbrains_chat_uuid()

        assert result is None

    def test_no_localappdata(self):
        """Returns None when LOCALAPPDATA is not set."""
        from activecontext.transport.acp.agent import _find_jetbrains_chat_uuid

        with patch.dict("os.environ", {"AC_CLIENT_JETBRAINS": "1"}, clear=True):
            result = _find_jetbrains_chat_uuid()

        assert result is None

    def test_no_jetbrains_dir(self):
        """Returns None when JetBrains directory does not exist."""
        from activecontext.transport.acp.agent import _find_jetbrains_chat_uuid

        with patch.dict(
            "os.environ",
            {"AC_CLIENT_JETBRAINS": "1", "LOCALAPPDATA": "/nonexistent"},
            clear=True,
        ):
            result = _find_jetbrains_chat_uuid()

        assert result is None


# ============================================================================
# TestACPCreateAgent
# ============================================================================


class TestACPCreateAgent:
    """Test create_agent factory function."""

    def test_create_agent_returns_instance(self):
        """create_agent() returns an ActiveContextAgent."""
        from activecontext.transport.acp.agent import ActiveContextAgent, create_agent

        with (
            patch(f"{MODULE}.get_default_model", return_value=None),
            patch(f"{MODULE}.SessionManager"),
        ):
            agent = create_agent()

        assert isinstance(agent, ActiveContextAgent)


# ============================================================================
# TestACPEmitUpdateInternal
# ============================================================================


class TestACPEmitUpdateInternal:
    """Test _emit_update_internal for various UpdateKind values."""

    @pytest.mark.asyncio
    async def test_statement_executing_sends_thought(self):
        """STATEMENT_EXECUTING sends an agent thought update."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.STATEMENT_EXECUTING
        update.payload = {"source": "v = text('main.py')"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_statement_executed_ok_with_stdout(self):
        """STATEMENT_EXECUTED with stdout sends thought."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.STATEMENT_EXECUTED
        update.payload = {"status": "ok", "stdout": "result text"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_statement_executed_error(self):
        """STATEMENT_EXECUTED with error sends error thought."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.STATEMENT_EXECUTED
        update.payload = {"status": "error", "exception": {"message": "NameError"}}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_response_chunk_buffered(self):
        """RESPONSE_CHUNK with batching enabled goes through buffer."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        agent._batch_enabled = True
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.RESPONSE_CHUNK
        update.payload = {"text": "hello"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        # Should be buffered, not sent directly
        assert agent._nagle.has_buffered("s1") or conn.session_update.called
        # Clean up any scheduled flush
        await agent._nagle.close("s1")

    @pytest.mark.asyncio
    async def test_response_chunk_unbuffered(self):
        """RESPONSE_CHUNK with batching disabled sends directly."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        agent._batch_enabled = False
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.RESPONSE_CHUNK
        update.payload = {"text": "hello"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_projection_ready_with_handles(self):
        """PROJECTION_READY is logged but does not emit a session_update."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.PROJECTION_READY
        update.payload = {"handles": {"v1": "node1", "v2": "node2"}}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_conversation_progress_with_status(self):
        """CONVERSATION_PROGRESS sends status thought."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.CONVERSATION_PROGRESS
        update.payload = {"current": 5, "total": 10, "status": "Installing"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversation_progress_percentage(self):
        """CONVERSATION_PROGRESS sends percentage thought when no status."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.CONVERSATION_PROGRESS
        update.payload = {"current": 3, "total": 10, "status": ""}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_chunk_flushes_pending_chunks(self):
        """Non-RESPONSE_CHUNK update flushes any pending chunk buffer first."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        agent._batch_enabled = True
        conn = _make_mock_conn()
        agent._conn = conn
        agent._nagle._buffers["s1"] = "pending text"

        update = MagicMock()
        update.kind = UpdateKind.STATEMENT_EXECUTING
        update.payload = {"source": "code()"}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        # Buffer should have been flushed (2 calls: flush + statement)
        assert conn.session_update.call_count == 2
        assert not agent._nagle.has_buffered("s1")

    @pytest.mark.asyncio
    async def test_statement_executed_ok_no_stdout(self):
        """STATEMENT_EXECUTED ok with empty stdout sends no update."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        update = MagicMock()
        update.kind = UpdateKind.STATEMENT_EXECUTED
        update.payload = {"status": "ok", "stdout": ""}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        # No thought sent for ok + empty stdout
        conn.session_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_statement_executing_truncates_long_source(self):
        """STATEMENT_EXECUTING truncates source longer than 100 chars."""
        from activecontext.session.protocols import UpdateKind

        agent = _make_agent()
        conn = _make_mock_conn()
        agent._conn = conn

        long_source = "x" * 200
        update = MagicMock()
        update.kind = UpdateKind.STATEMENT_EXECUTING
        update.payload = {"source": long_source}
        update.timestamp = 0

        with patch("activecontext.dashboard.is_dashboard_running", return_value=False):
            await agent._emit_update_internal("s1", update)

        conn.session_update.assert_called_once()


# ============================================================================
# TestACPPostSessionSetup
# ============================================================================


class TestACPPostSessionSetup:
    """Test _post_session_setup hook."""

    @pytest.mark.asyncio
    async def test_post_setup_session_not_found(self):
        """_post_session_setup does nothing when session is not found."""
        agent = _make_agent()
        agent._manager.get_session = AsyncMock(return_value=None)
        agent._conn = _make_mock_conn()

        # Should not raise
        await agent._post_session_setup("nonexistent")

    @pytest.mark.asyncio
    async def test_post_setup_sends_available_commands(self):
        """_post_session_setup advertises slash commands."""
        agent = _make_agent()
        mock_session = _make_mock_session()
        agent._manager.get_session = AsyncMock(return_value=mock_session)
        conn = _make_mock_conn()
        agent._conn = conn

        await agent._post_session_setup("test-session-1")

        # At minimum, available_commands_update should be sent
        assert conn.session_update.called

    @pytest.mark.asyncio
    async def test_post_setup_handles_startup_exception(self):
        """_post_session_setup handles exceptions from session.startup()."""
        agent = _make_agent()
        mock_session = _make_mock_session()

        async def failing_startup():
            raise RuntimeError("startup failed")
            yield  # noqa: unreachable

        mock_session.startup = MagicMock(return_value=failing_startup())
        agent._manager.get_session = AsyncMock(return_value=mock_session)
        conn = _make_mock_conn()
        agent._conn = conn

        # Should not raise
        await agent._post_session_setup("test-session-1")


# ============================================================================
# TestACPDefaultSessionModes
# ============================================================================


class TestACPDefaultSessionModes:
    """Test DEFAULT_SESSION_MODES constant."""

    def test_mode_ids_unique(self):
        """All default mode IDs are unique."""
        from activecontext.transport.acp.agent import DEFAULT_SESSION_MODES

        ids = [m.id for m in DEFAULT_SESSION_MODES]
        assert len(ids) == len(set(ids))

    def test_expected_modes_present(self):
        """Expected mode IDs are present in defaults."""
        from activecontext.transport.acp.agent import DEFAULT_SESSION_MODES

        ids = {m.id for m in DEFAULT_SESSION_MODES}
        assert "normal" in ids
        assert "plan" in ids
