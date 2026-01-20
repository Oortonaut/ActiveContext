"""Tests for agent handle and manager modules."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from datetime import datetime, timezone

from activecontext.agents.handle import AgentHandle
from activecontext.agents.manager import AgentManager
from activecontext.agents.schema import AgentEntry, AgentMessage, AgentState
from activecontext.agents.registry import AgentTypeRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_scratchpad_manager():
    """Create a mock ScratchpadManager."""
    manager = Mock()
    manager._cwd = "/test/project"
    manager.register_agent = Mock(return_value=AgentEntry(
        id="test1234",
        session_id="sess-1234",
        agent_type="explorer",
        task="Test task",
        parent_id=None,
        state=AgentState.SPAWNED,
    ))
    manager.get_agent = Mock(return_value=None)
    manager.get_all_agents = Mock(return_value=[])
    manager.update_agent = Mock(return_value=None)
    manager.send_message = Mock(return_value=AgentMessage(
        id="msg-1234",
        sender="agent_a",
        recipient="agent_b",
        content="Hello",
    ))
    manager.get_messages = Mock(return_value=[])
    manager.mark_message_status = Mock()
    manager.unregister_agent = Mock()
    return manager


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    manager = Mock()

    # Create a mock session
    mock_session = Mock()
    mock_session.session_id = "sess-1234"
    mock_session.timeline = Mock()
    mock_session.timeline._agent_manager = None
    mock_session.timeline._agent_id = None
    mock_session.timeline._setup_agent_namespace = Mock()
    mock_session.cancel = AsyncMock()

    manager.create_session = AsyncMock(return_value=mock_session)
    manager.close_session = AsyncMock()
    return manager


@pytest.fixture
def type_registry():
    """Create a type registry with default types."""
    return AgentTypeRegistry()


@pytest.fixture
def agent_manager(mock_session_manager, mock_scratchpad_manager, type_registry):
    """Create an AgentManager with mocked dependencies."""
    return AgentManager(
        session_manager=mock_session_manager,
        scratchpad_manager=mock_scratchpad_manager,
        type_registry=type_registry,
    )


@pytest.fixture
def agent_entry():
    """Create a sample AgentEntry."""
    return AgentEntry(
        id="agent123",
        session_id="sess-456",
        agent_type="explorer",
        task="Find auth code",
        parent_id="parent1",
        state=AgentState.RUNNING,
    )


# =============================================================================
# AgentHandle Tests
# =============================================================================


class TestAgentHandleInit:
    """Tests for AgentHandle initialization."""

    def test_init_stores_agent_id(self, agent_manager):
        """Test that init stores the agent ID."""
        handle = AgentHandle("test-id", agent_manager)

        assert handle._agent_id == "test-id"
        assert handle._manager is agent_manager

    def test_agent_id_property(self, agent_manager):
        """Test agent_id property returns correct ID."""
        handle = AgentHandle("my-agent", agent_manager)

        assert handle.agent_id == "my-agent"


class TestAgentHandleProperties:
    """Tests for AgentHandle property accessors."""

    def test_state_returns_agent_state(self, agent_manager, agent_entry):
        """Test state property returns the agent's state."""
        agent_manager._agents["agent123"] = agent_entry
        handle = AgentHandle("agent123", agent_manager)

        assert handle.state == AgentState.RUNNING

    def test_state_returns_terminated_when_not_found(self, agent_manager):
        """Test state returns TERMINATED when agent not found."""
        handle = AgentHandle("nonexistent", agent_manager)

        assert handle.state == AgentState.TERMINATED

    def test_task_returns_agent_task(self, agent_manager, agent_entry):
        """Test task property returns the agent's task."""
        agent_manager._agents["agent123"] = agent_entry
        handle = AgentHandle("agent123", agent_manager)

        assert handle.task == "Find auth code"

    def test_task_returns_empty_when_not_found(self, agent_manager):
        """Test task returns empty string when agent not found."""
        handle = AgentHandle("nonexistent", agent_manager)

        assert handle.task == ""

    def test_agent_type_returns_type(self, agent_manager, agent_entry):
        """Test agent_type property returns the agent's type."""
        agent_manager._agents["agent123"] = agent_entry
        handle = AgentHandle("agent123", agent_manager)

        assert handle.agent_type == "explorer"

    def test_agent_type_returns_empty_when_not_found(self, agent_manager):
        """Test agent_type returns empty string when agent not found."""
        handle = AgentHandle("nonexistent", agent_manager)

        assert handle.agent_type == ""

    def test_parent_id_returns_parent(self, agent_manager, agent_entry):
        """Test parent_id property returns the parent ID."""
        agent_manager._agents["agent123"] = agent_entry
        handle = AgentHandle("agent123", agent_manager)

        assert handle.parent_id == "parent1"

    def test_parent_id_returns_none_when_not_found(self, agent_manager):
        """Test parent_id returns None when agent not found."""
        handle = AgentHandle("nonexistent", agent_manager)

        assert handle.parent_id is None


class TestAgentHandleToDict:
    """Tests for AgentHandle to_dict method."""

    def test_to_dict_with_existing_agent(self, agent_manager, agent_entry):
        """Test to_dict returns agent data."""
        agent_manager._agents["agent123"] = agent_entry
        handle = AgentHandle("agent123", agent_manager)

        result = handle.to_dict()

        assert result["id"] == "agent123"
        assert result["agent_type"] == "explorer"
        assert result["task"] == "Find auth code"
        assert result["state"] == "running"

    def test_to_dict_with_missing_agent(self, agent_manager):
        """Test to_dict returns minimal dict for missing agent."""
        handle = AgentHandle("nonexistent", agent_manager)

        result = handle.to_dict()

        assert result["agent_id"] == "nonexistent"
        assert result["state"] == "terminated"


class TestAgentHandleRepr:
    """Tests for AgentHandle __repr__."""

    def test_repr_format(self, agent_manager, agent_entry):
        """Test __repr__ returns expected format."""
        agent_manager._agents["agent123"] = agent_entry
        handle = AgentHandle("agent123", agent_manager)

        result = repr(handle)

        assert "AgentHandle" in result
        assert "agent123" in result
        assert "running" in result

    def test_repr_with_terminated_agent(self, agent_manager):
        """Test __repr__ shows terminated state for missing agent."""
        handle = AgentHandle("missing", agent_manager)

        result = repr(handle)

        assert "terminated" in result


# =============================================================================
# AgentManager Tests
# =============================================================================


class TestAgentManagerInit:
    """Tests for AgentManager initialization."""

    def test_init_with_provided_registry(
        self, mock_session_manager, mock_scratchpad_manager, type_registry
    ):
        """Test init with provided type registry."""
        manager = AgentManager(
            mock_session_manager, mock_scratchpad_manager, type_registry
        )

        assert manager._type_registry is type_registry
        assert manager._session_manager is mock_session_manager
        assert manager._scratchpad_manager is mock_scratchpad_manager

    def test_init_creates_default_registry(
        self, mock_session_manager, mock_scratchpad_manager
    ):
        """Test init creates default registry when not provided."""
        manager = AgentManager(mock_session_manager, mock_scratchpad_manager)

        assert manager._type_registry is not None
        assert isinstance(manager._type_registry, AgentTypeRegistry)

    def test_init_creates_empty_caches(
        self, mock_session_manager, mock_scratchpad_manager
    ):
        """Test init creates empty agent caches."""
        manager = AgentManager(mock_session_manager, mock_scratchpad_manager)

        assert manager._agents == {}
        assert manager._agent_sessions == {}
        assert manager._shared_nodes == {}


class TestAgentManagerTypeRegistry:
    """Tests for type_registry property."""

    def test_type_registry_property(self, agent_manager, type_registry):
        """Test type_registry property returns the registry."""
        assert agent_manager.type_registry is type_registry


class TestAgentManagerSpawn:
    """Tests for spawn_agent method."""

    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, agent_manager, mock_session_manager):
        """Test successful agent spawn."""
        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Find files",
        )

        assert isinstance(handle, AgentHandle)
        mock_session_manager.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_agent_with_cwd(
        self, agent_manager, mock_session_manager
    ):
        """Test spawn_agent with custom cwd."""
        await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Find files",
            cwd="/custom/path",
        )

        call_kwargs = mock_session_manager.create_session.call_args.kwargs
        assert call_kwargs["cwd"] == "/custom/path"

    @pytest.mark.asyncio
    async def test_spawn_agent_unknown_type_raises(self, agent_manager):
        """Test spawn_agent raises for unknown type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            await agent_manager.spawn_agent(
                agent_type="nonexistent",
                task="Task",
            )

    @pytest.mark.asyncio
    async def test_spawn_agent_caches_entry(self, agent_manager):
        """Test spawn_agent caches the agent entry."""
        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Find files",
        )

        # Should be in local cache
        assert handle.agent_id in agent_manager._agents

    @pytest.mark.asyncio
    async def test_spawn_agent_caches_session(self, agent_manager):
        """Test spawn_agent caches the session."""
        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Find files",
        )

        assert handle.agent_id in agent_manager._agent_sessions


class TestAgentManagerGetAgent:
    """Tests for get_agent method."""

    def test_get_agent_from_cache(self, agent_manager, agent_entry):
        """Test get_agent returns from local cache."""
        agent_manager._agents["agent123"] = agent_entry

        result = agent_manager.get_agent("agent123")

        assert result is agent_entry

    def test_get_agent_from_scratchpad(
        self, agent_manager, mock_scratchpad_manager, agent_entry
    ):
        """Test get_agent falls back to scratchpad."""
        mock_scratchpad_manager.get_agent.return_value = agent_entry

        result = agent_manager.get_agent("agent123")

        assert result is agent_entry
        mock_scratchpad_manager.get_agent.assert_called_with("agent123")

    def test_get_agent_not_found(self, agent_manager):
        """Test get_agent returns None when not found."""
        result = agent_manager.get_agent("nonexistent")

        assert result is None


class TestAgentManagerGetSession:
    """Tests for get_session method."""

    def test_get_session_found(self, agent_manager):
        """Test get_session returns cached session."""
        mock_session = Mock()
        agent_manager._agent_sessions["agent123"] = mock_session

        result = agent_manager.get_session("agent123")

        assert result is mock_session

    def test_get_session_not_found(self, agent_manager):
        """Test get_session returns None when not found."""
        result = agent_manager.get_session("nonexistent")

        assert result is None


class TestAgentManagerListAgents:
    """Tests for list_agents method."""

    def test_list_agents_delegates_to_scratchpad(
        self, agent_manager, mock_scratchpad_manager, agent_entry
    ):
        """Test list_agents delegates to scratchpad."""
        mock_scratchpad_manager.get_all_agents.return_value = [agent_entry]

        result = agent_manager.list_agents()

        assert result == [agent_entry]
        mock_scratchpad_manager.get_all_agents.assert_called_once()


class TestAgentManagerUpdateState:
    """Tests for update_agent_state method."""

    @pytest.mark.asyncio
    async def test_update_state_success(
        self, agent_manager, mock_scratchpad_manager, agent_entry
    ):
        """Test update_agent_state updates state."""
        mock_scratchpad_manager.update_agent.return_value = agent_entry
        agent_manager._agents["agent123"] = agent_entry

        result = await agent_manager.update_agent_state(
            "agent123", AgentState.PAUSED
        )

        assert result is agent_entry
        mock_scratchpad_manager.update_agent.assert_called_with(
            "agent123", state=AgentState.PAUSED, task=None
        )

    @pytest.mark.asyncio
    async def test_update_state_with_task(
        self, agent_manager, mock_scratchpad_manager, agent_entry
    ):
        """Test update_agent_state with new task."""
        mock_scratchpad_manager.update_agent.return_value = agent_entry

        await agent_manager.update_agent_state(
            "agent123", AgentState.RUNNING, task="New task"
        )

        mock_scratchpad_manager.update_agent.assert_called_with(
            "agent123", state=AgentState.RUNNING, task="New task"
        )

    @pytest.mark.asyncio
    async def test_update_state_updates_cache(
        self, agent_manager, mock_scratchpad_manager, agent_entry
    ):
        """Test update_agent_state updates local cache."""
        updated_entry = AgentEntry(
            id="agent123",
            session_id="sess-456",
            agent_type="explorer",
            task="Find auth code",
            state=AgentState.PAUSED,
        )
        mock_scratchpad_manager.update_agent.return_value = updated_entry
        agent_manager._agents["agent123"] = agent_entry

        await agent_manager.update_agent_state("agent123", AgentState.PAUSED)

        assert agent_manager._agents["agent123"] is updated_entry


class TestAgentManagerMessaging:
    """Tests for agent messaging methods."""

    @pytest.mark.asyncio
    async def test_send_message(self, agent_manager, mock_scratchpad_manager):
        """Test send_message sends via scratchpad."""
        result = await agent_manager.send_message(
            sender="agent_a",
            recipient="agent_b",
            content="Hello",
        )

        assert result == "msg-1234"
        mock_scratchpad_manager.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_with_refs(
        self, agent_manager, mock_scratchpad_manager
    ):
        """Test send_message with node refs."""
        await agent_manager.send_message(
            sender="agent_a",
            recipient="agent_b",
            content="Check this",
            node_refs=["node1", "node2"],
        )

        call_kwargs = mock_scratchpad_manager.send_message.call_args.kwargs
        assert call_kwargs["node_refs"] == ["node1", "node2"]

    def test_get_messages(self, agent_manager, mock_scratchpad_manager):
        """Test get_messages retrieves from scratchpad."""
        agent_manager.get_messages("agent123", status="pending")

        mock_scratchpad_manager.get_messages.assert_called_with(
            "agent123", "pending"
        )

    def test_mark_message_delivered(self, agent_manager, mock_scratchpad_manager):
        """Test mark_message_delivered updates status."""
        agent_manager.mark_message_delivered("msg-123")

        mock_scratchpad_manager.mark_message_status.assert_called_with(
            "msg-123", "delivered"
        )

    def test_mark_message_read(self, agent_manager, mock_scratchpad_manager):
        """Test mark_message_read updates status."""
        agent_manager.mark_message_read("msg-123")

        mock_scratchpad_manager.mark_message_status.assert_called_with(
            "msg-123", "read"
        )


class TestAgentManagerSharedNodes:
    """Tests for shared node registry."""

    def test_share_node(self, agent_manager):
        """Test share_node registers node."""
        node = Mock()
        node.node_id = "node-123"

        result = agent_manager.share_node(node)

        assert result == "node-123"
        assert agent_manager._shared_nodes["node-123"] is node

    def test_get_shared_node_found(self, agent_manager):
        """Test get_shared_node returns shared node."""
        node = Mock()
        node.node_id = "node-123"
        agent_manager._shared_nodes["node-123"] = node

        result = agent_manager.get_shared_node("node-123")

        assert result is node

    def test_get_shared_node_not_found(self, agent_manager):
        """Test get_shared_node returns None when not found."""
        result = agent_manager.get_shared_node("nonexistent")

        assert result is None

    def test_unshare_node_success(self, agent_manager):
        """Test unshare_node removes node."""
        node = Mock()
        node.node_id = "node-123"
        agent_manager._shared_nodes["node-123"] = node

        result = agent_manager.unshare_node("node-123")

        assert result is True
        assert "node-123" not in agent_manager._shared_nodes

    def test_unshare_node_not_found(self, agent_manager):
        """Test unshare_node returns False when not found."""
        result = agent_manager.unshare_node("nonexistent")

        assert result is False


class TestAgentManagerLifecycle:
    """Tests for agent lifecycle methods."""

    @pytest.mark.asyncio
    async def test_pause_agent(self, agent_manager, mock_scratchpad_manager):
        """Test pause_agent updates state to PAUSED."""
        await agent_manager.pause_agent("agent123")

        mock_scratchpad_manager.update_agent.assert_called_with(
            "agent123", state=AgentState.PAUSED, task=None
        )

    @pytest.mark.asyncio
    async def test_resume_agent(self, agent_manager, mock_scratchpad_manager):
        """Test resume_agent updates state to RUNNING."""
        await agent_manager.resume_agent("agent123")

        mock_scratchpad_manager.update_agent.assert_called_with(
            "agent123", state=AgentState.RUNNING, task=None
        )

    @pytest.mark.asyncio
    async def test_terminate_agent(
        self, agent_manager, mock_scratchpad_manager, mock_session_manager
    ):
        """Test terminate_agent cleans up agent."""
        # Setup cached session
        mock_session = Mock()
        mock_session.cancel = AsyncMock()
        mock_session.session_id = "sess-123"
        agent_manager._agent_sessions["agent123"] = mock_session
        agent_manager._agents["agent123"] = Mock()

        await agent_manager.terminate_agent("agent123")

        # Should update state
        mock_scratchpad_manager.update_agent.assert_called()
        # Should cancel session
        mock_session.cancel.assert_called_once()
        # Should close session
        mock_session_manager.close_session.assert_called_with("sess-123")
        # Should remove from caches
        assert "agent123" not in agent_manager._agents
        assert "agent123" not in agent_manager._agent_sessions
        # Should unregister from scratchpad
        mock_scratchpad_manager.unregister_agent.assert_called_with("agent123")

    @pytest.mark.asyncio
    async def test_terminate_agent_without_session(
        self, agent_manager, mock_scratchpad_manager
    ):
        """Test terminate_agent works without cached session."""
        await agent_manager.terminate_agent("agent123")

        # Should still update state and unregister
        mock_scratchpad_manager.update_agent.assert_called()
        mock_scratchpad_manager.unregister_agent.assert_called_with("agent123")


class TestAgentManagerPendingMessages:
    """Tests for has_pending_messages method."""

    def test_has_pending_messages_true(
        self, agent_manager, mock_scratchpad_manager
    ):
        """Test has_pending_messages returns True when messages exist."""
        mock_scratchpad_manager.get_messages.return_value = [Mock()]

        result = agent_manager.has_pending_messages("agent123")

        assert result is True

    def test_has_pending_messages_false(
        self, agent_manager, mock_scratchpad_manager
    ):
        """Test has_pending_messages returns False when no messages."""
        mock_scratchpad_manager.get_messages.return_value = []

        result = agent_manager.has_pending_messages("agent123")

        assert result is False
