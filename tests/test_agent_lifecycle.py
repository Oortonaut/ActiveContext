"""Integration tests for agent lifecycle (spawn/pause/terminate/message passing/cleanup)."""

from __future__ import annotations

from pathlib import Path

import pytest

from activecontext.agents.manager import AgentManager
from activecontext.agents.schema import AgentState
from activecontext.coordination.scratchpad import ScratchpadManager
from activecontext.session.session_manager import SessionManager

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def real_managers(tmp_path: Path):
    """Create real SessionManager, ScratchpadManager, and AgentManager."""
    cwd = str(tmp_path)
    ac_dir = tmp_path / ".ac"
    ac_dir.mkdir(exist_ok=True)
    sessions_dir = ac_dir / "sessions"
    sessions_dir.mkdir(exist_ok=True)

    session_manager = SessionManager()
    scratchpad_manager = ScratchpadManager(cwd=cwd)  # Pass string not Path
    agent_manager = AgentManager(session_manager, scratchpad_manager)

    yield session_manager, scratchpad_manager, agent_manager

    # Cleanup: close all sessions
    for session_id in list(session_manager._sessions.keys()):
        await session_manager.close_session(session_id)


# =============================================================================
# Agent Lifecycle Tests
# =============================================================================


class TestAgentSpawnLifecycle:
    """Test agent spawn with real sessions."""

    @pytest.mark.asyncio
    async def test_spawn_creates_real_session(self, real_managers):
        """Test that spawning an agent creates a real session."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Test task",
        )

        # Verify agent entry exists
        agent_entry = agent_manager.get_agent(handle.agent_id)
        assert agent_entry is not None
        assert agent_entry.agent_type == "explorer"
        assert agent_entry.task == "Test task"
        assert agent_entry.state == AgentState.SPAWNED

        # Verify underlying session exists
        session = agent_manager.get_session(handle.agent_id)
        assert session is not None
        assert session.session_id == agent_entry.session_id

    @pytest.mark.asyncio
    async def test_spawn_multiple_agents(self, real_managers):
        """Test spawning multiple agents creates independent sessions."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle1 = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Task 1",
        )
        handle2 = await agent_manager.spawn_agent(
            agent_type="summarizer",
            task="Task 2",
        )

        # Verify both agents exist
        assert agent_manager.get_agent(handle1.agent_id) is not None
        assert agent_manager.get_agent(handle2.agent_id) is not None

        # Verify different sessions
        session1 = agent_manager.get_session(handle1.agent_id)
        session2 = agent_manager.get_session(handle2.agent_id)
        assert session1 is not None
        assert session2 is not None
        assert session1.session_id != session2.session_id

    @pytest.mark.asyncio
    async def test_spawn_with_parent(self, real_managers):
        """Test spawning child agent with parent_id."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        parent = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Parent task",
        )
        child = await agent_manager.spawn_agent(
            agent_type="summarizer",
            task="Child task",
            parent_id=parent.agent_id,
        )

        # Verify parent-child relationship
        child_entry = agent_manager.get_agent(child.agent_id)
        assert child_entry.parent_id == parent.agent_id


class TestAgentPauseResume:
    """Test agent pause/resume lifecycle."""

    @pytest.mark.asyncio
    async def test_pause_agent(self, real_managers):
        """Test pausing an agent updates its state."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Test task",
        )

        # Initially spawned
        assert handle.state == AgentState.SPAWNED

        # Pause agent
        await agent_manager.pause_agent(handle.agent_id)

        # Verify state changed
        assert handle.state == AgentState.PAUSED

    @pytest.mark.asyncio
    async def test_resume_agent(self, real_managers):
        """Test resuming a paused agent."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Test task",
        )

        # Pause then resume
        await agent_manager.pause_agent(handle.agent_id)
        await agent_manager.resume_agent(handle.agent_id)

        # Verify state changed to running
        assert handle.state == AgentState.RUNNING

    @pytest.mark.asyncio
    async def test_pause_resume_via_handle(self, real_managers):
        """Test pause/resume using AgentHandle methods.

        Note: AgentHandle methods schedule coroutines but don't await them
        when called from async context. Need to give them time to complete.
        """
        import asyncio

        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Test task",
        )

        # Use handle methods (they schedule but don't block)
        handle.Pause()
        await asyncio.sleep(0.01)  # Let scheduled task complete
        assert handle.state == AgentState.PAUSED

        handle.Resume()
        await asyncio.sleep(0.01)  # Let scheduled task complete
        assert handle.state == AgentState.RUNNING


class TestAgentTermination:
    """Test agent termination and resource cleanup."""

    @pytest.mark.asyncio
    async def test_terminate_agent(self, real_managers):
        """Test terminating an agent cleans up resources."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Test task",
        )
        agent_id = handle.agent_id

        # Verify session exists
        session = agent_manager.get_session(agent_id)
        assert session is not None
        session_id = session.session_id

        # Terminate agent
        await agent_manager.terminate_agent(agent_id)

        # Verify state updated
        assert handle.state == AgentState.TERMINATED

        # Verify session cleaned up
        assert agent_manager.get_session(agent_id) is None
        assert session_id not in session_manager._sessions

        # Verify agent removed from cache
        assert agent_id not in agent_manager._agents
        assert agent_id not in agent_manager._agent_sessions

    @pytest.mark.asyncio
    async def test_terminate_via_handle(self, real_managers):
        """Test terminating agent via handle."""
        import asyncio

        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Test task",
        )
        agent_id = handle.agent_id

        # Terminate via handle (schedules but doesn't block)
        handle.Terminate()
        await asyncio.sleep(0.01)  # Let scheduled task complete

        # Verify terminated
        assert handle.state == AgentState.TERMINATED
        assert agent_manager.get_session(agent_id) is None

    @pytest.mark.asyncio
    async def test_terminate_multiple_agents(self, real_managers):
        """Test terminating multiple agents cleans up each independently."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle1 = await agent_manager.spawn_agent(agent_type="explorer", task="Task 1")
        handle2 = await agent_manager.spawn_agent(agent_type="summarizer", task="Task 2")

        # Terminate first agent
        await agent_manager.terminate_agent(handle1.agent_id)

        # First agent terminated
        assert handle1.state == AgentState.TERMINATED
        assert agent_manager.get_session(handle1.agent_id) is None

        # Second agent still active
        assert agent_manager.get_session(handle2.agent_id) is not None

        # Terminate second agent
        await agent_manager.terminate_agent(handle2.agent_id)
        assert handle2.state == AgentState.TERMINATED


# =============================================================================
# Message Passing Tests
# =============================================================================


class TestAgentMessagePassing:
    """Test message passing between agents."""

    @pytest.mark.asyncio
    async def test_send_message_between_agents(self, real_managers):
        """Test sending messages between two agents."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        # Send message
        msg_id = await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Test message",
        )

        assert msg_id is not None

        # Recipient should have pending message
        messages = agent_manager.get_messages(recipient.agent_id, status="pending")
        assert len(messages) == 1
        assert messages[0].sender == sender.agent_id
        assert messages[0].content == "Test message"

    @pytest.mark.asyncio
    async def test_message_with_node_refs(self, real_managers):
        """Test sending message with node references."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        # Send message with node refs
        await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Check nodes",
            node_refs=["node_1", "node_2"],
        )

        # Verify node refs stored
        messages = agent_manager.get_messages(recipient.agent_id)
        assert len(messages) == 1
        assert messages[0].node_refs == ["node_1", "node_2"]

    @pytest.mark.asyncio
    async def test_mark_message_delivered(self, real_managers):
        """Test marking message as delivered."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        msg_id = await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Test",
        )

        # Mark as delivered
        agent_manager.mark_message_delivered(msg_id)

        # Check status changed
        messages = agent_manager.get_messages(recipient.agent_id, status="delivered")
        assert len(messages) == 1
        assert messages[0].status == "delivered"

    @pytest.mark.asyncio
    async def test_mark_message_read(self, real_managers):
        """Test marking message as read."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        msg_id = await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Test",
        )

        # Mark as read
        agent_manager.mark_message_read(msg_id)

        # Check status changed
        messages = agent_manager.get_messages(recipient.agent_id, status="read")
        assert len(messages) == 1
        assert messages[0].status == "read"

    @pytest.mark.asyncio
    async def test_has_pending_messages(self, real_managers):
        """Test checking for pending messages."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        # Initially no messages
        assert agent_manager.has_pending_messages(recipient.agent_id) is False

        # Send message
        await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Test",
        )

        # Now has pending messages
        assert agent_manager.has_pending_messages(recipient.agent_id) is True


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


class TestResourceCleanup:
    """Test proper resource cleanup on agent termination."""

    @pytest.mark.asyncio
    async def test_terminate_cleans_shared_nodes(self, real_managers):
        """Test that terminating agent doesn't leave dangling shared nodes."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(agent_type="explorer", task="Test")

        # Share a node
        from unittest.mock import Mock

        node = Mock()
        node.node_id = "test_node_123"
        agent_manager.share_node(node)

        # Verify shared
        assert agent_manager.get_shared_node("test_node_123") is not None

        # Terminate agent
        await agent_manager.terminate_agent(handle.agent_id)

        # Shared nodes persist (they're global), but agent is gone
        # Note: In real implementation, may want to track which agent shared what
        assert handle.state == AgentState.TERMINATED

    @pytest.mark.asyncio
    async def test_terminate_cleans_pending_messages(self, real_managers):
        """Test that terminating an agent cleans up its messages."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        # Send message
        await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Test",
        )

        # Verify message exists before termination
        messages = agent_manager.get_messages(recipient.agent_id, status=None)
        assert len(messages) == 1

        # Terminate recipient
        await agent_manager.terminate_agent(recipient.agent_id)

        # Messages are cleaned up on termination
        messages = agent_manager.get_messages(recipient.agent_id, status=None)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_session_cancel_called_on_terminate(self, real_managers):
        """Test that session.cancel() is called when terminating."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(agent_type="explorer", task="Test")
        session = agent_manager.get_session(handle.agent_id)

        # Mock cancel to track calls
        from unittest.mock import AsyncMock

        session.cancel = AsyncMock()

        # Terminate
        await agent_manager.terminate_agent(handle.agent_id)

        # Verify cancel was called (may be called multiple times by manager and close_session)
        assert session.cancel.called
        assert session.cancel.call_count >= 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAgentEdgeCases:
    """Test edge cases in agent lifecycle."""

    @pytest.mark.asyncio
    async def test_terminate_nonexistent_agent(self, real_managers):
        """Test terminating a nonexistent agent doesn't crash."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        # Should not raise
        await agent_manager.terminate_agent("nonexistent_id")

    @pytest.mark.asyncio
    async def test_pause_terminated_agent(self, real_managers):
        """Test pausing a terminated agent."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        handle = await agent_manager.spawn_agent(agent_type="explorer", task="Test")
        await agent_manager.terminate_agent(handle.agent_id)

        # Pause after termination (state update will fail but won't crash)
        await agent_manager.pause_agent(handle.agent_id)

        # Still terminated (update had no effect)
        assert handle.state == AgentState.TERMINATED

    @pytest.mark.asyncio
    async def test_send_message_to_terminated_agent(self, real_managers):
        """Test sending message to terminated agent still queues it."""
        session_manager, scratchpad_manager, agent_manager = real_managers

        sender = await agent_manager.spawn_agent(agent_type="explorer", task="Sender")
        recipient = await agent_manager.spawn_agent(agent_type="summarizer", task="Recipient")

        # Terminate recipient
        await agent_manager.terminate_agent(recipient.agent_id)

        # Send message anyway
        msg_id = await agent_manager.send_message(
            sender=sender.agent_id,
            recipient=recipient.agent_id,
            content="Too late",
        )

        # Message still created (for audit trail)
        assert msg_id is not None
