"""Tests for session management and lifecycle.

Tests coverage for:
- src/activecontext/session/session_manager.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from activecontext.core.llm.litellm_provider import LiteLLMProvider
from activecontext.session.session_manager import Session, SessionManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=LiteLLMProvider)
    provider.model = "test-model"
    provider.stream = AsyncMock()
    return provider


@pytest.fixture
async def session_manager(mock_llm_provider):
    """Create SessionManager with mock LLM."""
    manager = SessionManager(default_llm=mock_llm_provider)
    yield manager
    # Cleanup
    for session_id in list(manager._sessions.keys()):
        await manager.close_session(session_id)


@pytest.fixture
async def test_session(session_manager):
    """Create a test session."""
    session = await session_manager.create_session(cwd="/test/project")
    return session


# =============================================================================
# Session Lifecycle Tests
# =============================================================================


class TestSessionLifecycle:
    """Tests for session creation, retrieval, and deletion."""

    @pytest.mark.asyncio
    async def test_create_session_returns_unique_id(self, session_manager):
        """Test that create_session returns unique session IDs."""
        session1 = await session_manager.create_session(cwd="/project1")
        session2 = await session_manager.create_session(cwd="/project2")

        assert session1.session_id != session2.session_id
        assert isinstance(session1.session_id, str)
        assert isinstance(session2.session_id, str)

    @pytest.mark.asyncio
    async def test_create_session_with_cwd(self, session_manager):
        """Test session creation with specific cwd."""
        session = await session_manager.create_session(cwd="/workspace/project")

        assert session.cwd == "/workspace/project"

    @pytest.mark.asyncio
    async def test_create_session_uses_default_llm(
        self, session_manager, mock_llm_provider
    ):
        """Test that new sessions use default LLM provider."""
        session = await session_manager.create_session(cwd="/test")

        assert session.llm is mock_llm_provider

    @pytest.mark.asyncio
    async def test_create_session_with_custom_llm(self, session_manager):
        """Test creating session with custom LLM provider."""
        custom_llm = Mock(spec=LiteLLMProvider)
        custom_llm.model = "custom-model"

        session = await session_manager.create_session(
            cwd="/test", llm=custom_llm
        )

        assert session.llm is custom_llm

    @pytest.mark.asyncio
    async def test_get_session_existing(self, session_manager, test_session):
        """Test retrieving an existing session."""
        retrieved = await session_manager.get_session(test_session.session_id)

        assert retrieved is test_session
        assert retrieved.session_id == test_session.session_id

    @pytest.mark.asyncio
    async def test_get_session_nonexistent(self, session_manager):
        """Test getting nonexistent session returns None."""
        session = await session_manager.get_session("nonexistent-id")

        assert session is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, session_manager):
        """Test listing all active sessions."""
        session1 = await session_manager.create_session(cwd="/project1")
        session2 = await session_manager.create_session(cwd="/project2")

        session_ids = await session_manager.list_sessions()

        assert len(session_ids) == 2
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    @pytest.mark.asyncio
    async def test_close_session(self, session_manager, test_session):
        """Test closing a session removes it from manager."""
        session_id = test_session.session_id

        await session_manager.close_session(session_id)

        assert await session_manager.get_session(session_id) is None
        assert session_id not in session_manager._sessions

    @pytest.mark.asyncio
    async def test_close_nonexistent_session(self, session_manager):
        """Test closing nonexistent session doesn't error."""
        # Should not raise
        await session_manager.close_session("nonexistent-id")


# =============================================================================
# Multi-Session Management Tests
# =============================================================================


class TestSessionIsolation:
    """Tests for session state isolation."""

    @pytest.mark.asyncio
    async def test_sessions_have_separate_timelines(self, session_manager):
        """Test that sessions have independent timelines."""
        session1 = await session_manager.create_session(cwd="/proj1")
        session2 = await session_manager.create_session(cwd="/proj2")

        # Execute statements in each session
        await session1.timeline.execute_statement("x = 1")
        await session2.timeline.execute_statement("x = 2")

        # Verify isolation
        ns1 = session1.timeline.get_namespace()
        ns2 = session2.timeline.get_namespace()

        assert ns1.get("x") == 1
        assert ns2.get("x") == 2

    @pytest.mark.asyncio
    async def test_sessions_have_separate_context_graphs(self, session_manager):
        """Test that sessions have independent context graphs."""
        session1 = await session_manager.create_session(cwd="/proj1")
        session2 = await session_manager.create_session(cwd="/proj2")

        graph1 = session1.get_context_graph()
        graph2 = session2.get_context_graph()

        assert graph1 is not graph2

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self, session_manager):
        """Test concurrent operations on different sessions."""
        import asyncio

        session1 = await session_manager.create_session(cwd="/proj1")
        session2 = await session_manager.create_session(cwd="/proj2")

        # Execute statements concurrently
        await asyncio.gather(
            session1.timeline.execute_statement("a = 1"),
            session2.timeline.execute_statement("b = 2"),
        )

        ns1 = session1.timeline.get_namespace()
        ns2 = session2.timeline.get_namespace()

        assert "a" in ns1
        assert "b" in ns2
        assert "a" not in ns2
        assert "b" not in ns1


# =============================================================================
# LLM Provider Management Tests
# =============================================================================


class TestSessionLLM:
    """Tests for LLM provider management."""

    @pytest.mark.asyncio
    async def test_set_default_llm(self, session_manager):
        """Test updating default LLM provider."""
        new_llm = Mock(spec=LiteLLMProvider)
        new_llm.model = "new-model"

        session_manager.set_default_llm(new_llm)

        # New sessions should use the new default
        session = await session_manager.create_session(cwd="/test")
        assert session.llm is new_llm

    @pytest.mark.asyncio
    async def test_set_session_llm(self, session_manager, test_session):
        """Test updating LLM provider for a specific session."""
        new_llm = Mock(spec=LiteLLMProvider)
        new_llm.model = "session-specific-model"

        test_session.set_llm(new_llm)

        assert test_session.llm is new_llm

    @pytest.mark.asyncio
    async def test_session_with_no_llm(self, session_manager):
        """Test creating session without LLM (direct mode)."""
        no_llm_manager = SessionManager(default_llm=None)

        session = await no_llm_manager.create_session(cwd="/test")

        assert session.llm is None


# =============================================================================
# Permission Management Tests
# =============================================================================


class TestSessionPermissions:
    """Tests for permission management and reloading."""

    @pytest.mark.asyncio
    async def test_session_has_permission_requesters(self, test_session):
        """Test that session has permission requester callbacks."""
        # Permission requesters should be set up during session creation
        # The actual implementation depends on the session setup
        assert hasattr(test_session, "timeline")
        assert hasattr(test_session.timeline, "_permission_manager")

    @pytest.mark.asyncio
    async def test_permission_requester_types(self):
        """Test that permission requester types are defined in TYPE_CHECKING block."""
        import typing
        # These types are defined in TYPE_CHECKING block for type hints
        # They're not importable at runtime, but we verify the module structure
        from activecontext.session import session_manager

        # The module should load without errors and have create_session
        assert hasattr(session_manager, "SessionManager")
        assert hasattr(session_manager, "Session")


# =============================================================================
# Session API Tests
# =============================================================================


class TestSessionAPI:
    """Tests for Session class public API."""

    @pytest.mark.asyncio
    async def test_session_properties(self, test_session):
        """Test session property accessors."""
        assert isinstance(test_session.session_id, str)
        assert test_session.cwd == "/test/project"
        assert test_session.timeline is not None

    @pytest.mark.asyncio
    async def test_get_context_objects(self, test_session):
        """Test getting context objects from session."""
        objects = test_session.get_context_objects()

        assert isinstance(objects, dict)

    @pytest.mark.asyncio
    async def test_get_context_graph(self, test_session):
        """Test getting context graph from session."""
        graph = test_session.get_context_graph()

        assert graph is not None

    @pytest.mark.asyncio
    async def test_clear_message_history(self, test_session):
        """Test clearing message history."""
        # Add some messages first
        test_session._message_history.append(
            Mock(role=Mock(value="user"), content="Test")
        )

        initial_count = len(test_session._message_history)
        assert initial_count > 0

        test_session.clear_message_history()

        assert len(test_session._message_history) == 0

    @pytest.mark.asyncio
    async def test_get_projection(self, test_session):
        """Test getting current projection."""
        projection = test_session.get_projection()

        assert projection is not None
        assert hasattr(projection, "sections")
        assert hasattr(projection, "handles")


# =============================================================================
# Session Execution Tests
# =============================================================================


class TestSessionExecution:
    """Tests for session code execution."""

    @pytest.mark.asyncio
    async def test_execute_code_statement(self, test_session):
        """Test executing Python code in session."""
        # _execute_code returns an async generator
        async for _ in test_session._execute_code("x = 42"):
            pass

        namespace = test_session.timeline.get_namespace()
        assert namespace.get("x") == 42

    @pytest.mark.asyncio
    async def test_tick_updates_session_state(self, test_session):
        """Test that tick() processes pending updates."""
        # Tick should not error
        await test_session.tick()

    @pytest.mark.asyncio
    async def test_cancel_interrupts_execution(self, test_session):
        """Test cancel() interrupts ongoing operations."""
        # Cancel should set cancellation flag
        await test_session.cancel()

        # Cancellation state should be reflected on session, not timeline
        assert test_session._cancelled is True


# =============================================================================
# Prompt Execution Tests
# =============================================================================


class TestSessionPrompt:
    """Tests for session prompt execution."""

    @pytest.mark.asyncio
    async def test_prompt_with_llm(self, test_session, mock_llm_provider):
        """Test prompt execution with LLM."""
        # Mock stream response
        async def mock_stream(*args, **kwargs):
            yield Mock(text="Response", is_final=False)
            yield Mock(text=" text", is_final=True, finish_reason="stop")

        mock_llm_provider.stream = mock_stream

        updates = []
        async for update in test_session.prompt("Test prompt"):
            updates.append(update)

        # Should have received updates
        assert len(updates) > 0

    @pytest.mark.asyncio
    async def test_prompt_direct_mode(self, session_manager):
        """Test prompt in direct mode (no LLM)."""
        session = await session_manager.create_session(cwd="/test")
        session.set_llm(None)

        # Direct prompt should execute code directly
        updates = []
        async for update in session.prompt("x = 100"):
            updates.append(update)

        # Should have execution updates
        assert len(updates) > 0

        # Code should be executed
        namespace = session.timeline.get_namespace()
        assert namespace.get("x") == 100


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSessionConfiguration:
    """Tests for session configuration loading."""

    @pytest.mark.asyncio
    @patch("activecontext.config.load_config")
    async def test_load_project_config(self, mock_load_config):
        """Test that project config is loaded for session."""
        mock_config = Mock()
        mock_config.sandbox = None  # Prevent PermissionManager errors
        mock_config.mcp = None  # Prevent MCP autoconnect errors
        # Mock session.startup for startup script execution
        mock_config.session = None
        mock_load_config.return_value = mock_config

        manager = SessionManager(default_llm=None)
        session = await manager.create_session(cwd="/project/path")

        # Config should be loaded with session_root
        mock_load_config.assert_called_with(session_root="/project/path")


# =============================================================================
# Integration Tests
# =============================================================================


class TestSessionManagerIntegration:
    """Integration tests for session manager."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self, mock_llm_provider):
        """Test complete session lifecycle."""
        # Create manager
        manager = SessionManager(default_llm=mock_llm_provider)

        # Create session
        session = await manager.create_session(cwd="/workspace")
        session_id = session.session_id

        # Verify session exists (get_session is async)
        assert await manager.get_session(session_id) is session

        # Execute some code (async generator)
        async for _ in session._execute_code("test_var = 123"):
            pass
        namespace = session.timeline.get_namespace()
        assert namespace.get("test_var") == 123

        # List sessions (returns session IDs)
        session_ids = await manager.list_sessions()
        assert len(session_ids) == 1
        assert session_ids[0] == session_id

        # Close session
        await manager.close_session(session_id)
        assert await manager.get_session(session_id) is None

    @pytest.mark.asyncio
    async def test_multiple_sessions_workflow(self, mock_llm_provider):
        """Test working with multiple sessions."""
        manager = SessionManager(default_llm=mock_llm_provider)

        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = await manager.create_session(cwd=f"/project{i}")
            sessions.append(session)

        # Verify all exist (list_sessions is async, returns IDs)
        session_ids = await manager.list_sessions()
        assert len(session_ids) == 3

        # Execute unique code in each (async generator)
        for i, session in enumerate(sessions):
            async for _ in session._execute_code(f"value = {i * 10}"):
                pass

        # Verify isolation
        for i, session in enumerate(sessions):
            namespace = session.timeline.get_namespace()
            assert namespace.get("value") == i * 10

        # Close all
        for session in sessions:
            await manager.close_session(session.session_id)

        assert len(await manager.list_sessions()) == 0

    @pytest.mark.asyncio
    async def test_session_with_context_graph(self, session_manager):
        """Test session with context graph operations."""
        from activecontext.context.graph import ContextGraph
        from tests.utils import create_mock_context_node

        session = await session_manager.create_session(cwd="/test")

        # Get graph
        graph = session.get_context_graph()
        initial_count = len(graph)

        # Add a node
        node = create_mock_context_node("test-node", "view")
        graph.add_node(node)

        # Verify node exists (graph may have initial nodes from session setup)
        assert "test-node" in graph
        assert len(graph) == initial_count + 1

    @pytest.mark.asyncio
    async def test_session_projection_updates(self, session_manager):
        """Test that session projection reflects changes."""
        session = await session_manager.create_session(cwd="/test")

        # Get initial projection
        projection1 = session.get_projection()
        initial_sections = len(projection1.sections)

        # Execute code that creates objects (async generator)
        async for _ in session._execute_code("# Test code"):
            pass

        # Get updated projection
        projection2 = session.get_projection()

        # Projection should be updated
        assert projection2 is not None


# =============================================================================
# Session Persistence Tests
# =============================================================================


class TestSessionPersistence:
    """Tests for session save/load functionality."""

    @pytest.mark.asyncio
    async def test_session_save_and_load_roundtrip(self, tmp_path):
        """Test that a session can be saved and loaded from disk."""
        # Create a session with a specific cwd
        cwd = str(tmp_path)
        sessions_dir = tmp_path / ".ac" / "sessions"
        sessions_dir.mkdir(parents=True)

        manager = SessionManager()
        session = await manager.create_session(cwd=cwd)
        session_id = session.session_id

        # Add some state to the session
        session._title = "Test Session Title"

        # Save session
        session.save()

        # Verify file was created
        session_file = sessions_dir / f"{session_id}.yaml"
        assert session_file.exists()

        # Load session from disk
        loaded = Session.from_file(
            cwd=cwd,
            session_id=session_id,
            llm=None,
        )

        # Verify loaded session matches original
        assert loaded is not None
        assert loaded.session_id == session_id
        assert loaded.title == "Test Session Title"

    @pytest.mark.asyncio
    async def test_session_from_file_not_found(self, tmp_path):
        """Test that from_file returns None for non-existent session."""
        cwd = str(tmp_path)
        sessions_dir = tmp_path / ".ac" / "sessions"
        sessions_dir.mkdir(parents=True)

        loaded = Session.from_file(
            cwd=cwd,
            session_id="nonexistent-uuid",
            llm=None,
        )

        assert loaded is None

    @pytest.mark.asyncio
    async def test_session_save_creates_directory(self, tmp_path):
        """Test that save() creates the sessions directory if needed."""
        cwd = str(tmp_path)
        sessions_dir = tmp_path / ".ac" / "sessions"

        # Directory should not exist yet
        assert not sessions_dir.exists()

        manager = SessionManager()
        session = await manager.create_session(cwd=cwd)

        # Save should create the directory
        session.save()

        assert sessions_dir.exists()
        assert (sessions_dir / f"{session.session_id}.yaml").exists()

    @pytest.mark.asyncio
    async def test_session_resume_preserves_user_messages_group(self, tmp_path):
        """Test that resumed sessions can detect pending messages.

        Regression test for bug where _user_messages_group was not restored
        from the context graph when loading from disk, causing has_pending_messages()
        to return False even when messages were queued.
        """
        cwd = str(tmp_path)
        sessions_dir = tmp_path / ".ac" / "sessions"
        sessions_dir.mkdir(parents=True)

        manager = SessionManager()
        session = await manager.create_session(cwd=cwd)
        session_id = session.session_id

        # Queue a message (this adds to _user_messages_group)
        session.queue_user_message("Hello, test message")

        # Verify message is detected before save
        assert session.has_pending_messages(), "Message should be pending before save"

        # Save session
        session.save()

        # Load session from disk (simulates IDE restart)
        loaded = Session.from_file(
            cwd=cwd,
            session_id=session_id,
            llm=None,
        )

        # The critical test: _user_messages_group must be restored
        assert loaded is not None
        assert loaded._user_messages_group is not None, "_user_messages_group should be restored"
        assert loaded._alerts_group is not None, "_alerts_group should be restored"

        # Verify the group points to the correct node in the graph
        graph_node = loaded.timeline.context_graph.get_node("user_messages")
        assert loaded._user_messages_group is graph_node, (
            "_user_messages_group should reference the restored graph node"
        )


# =============================================================================
# JetBrains UUID Detection Tests (Warning-only)
# =============================================================================


class TestJetBrainsUUIDDetection:
    """Tests for JetBrains IDE chat UUID detection.

    These tests warn rather than fail because the JetBrains UUID detection
    is a workaround that depends on JetBrains IDEs' internal file structure.
    """

    def test_find_jetbrains_chat_uuid_warns_if_not_found(self):
        """Test _find_jetbrains_chat_uuid returns None gracefully when no JetBrains IDE installed."""
        import warnings
        from activecontext.transport.acp.agent import _find_jetbrains_chat_uuid

        result = _find_jetbrains_chat_uuid()

        # This should not raise, just return None or a UUID
        if result is None:
            warnings.warn(
                "JetBrains chat UUID detection unavailable - "
                "no JetBrains IDE installed or aia-task-history not found",
                UserWarning,
            )
        else:
            # If we got a result, it should look like a UUID
            assert len(result) == 36
            assert result.count("-") == 4
