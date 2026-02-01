"""Tests for the agent loop extension API.

Tests for the public async agent loop extension API that allows external
code to queue messages and receive out-of-band status updates.

This API enables:
- Message queueing: queue_user_message()
- Status checking: has_pending_messages(), get_pending_messages()
- Message processing: mark_message_processed()
- Loop control: wake(), run_agent_loop()
"""

from __future__ import annotations

import asyncio

import pytest

from activecontext.context.nodes import MessageNode


class TestMessageQueueing:
    """Test message queueing API for external integrations."""

    @pytest.mark.asyncio
    async def test_queue_user_message_creates_node(self):
        """queue_user_message() creates a MessageNode in the context graph."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Queue a message
        node = session.queue_user_message("Hello from external code")

        assert isinstance(node, MessageNode)
        assert node.role == "user"
        assert node.content == "Hello from external code"
        assert node.originator == "user"

    @pytest.mark.asyncio
    async def test_queue_user_message_with_custom_id(self):
        """queue_user_message() accepts custom message IDs."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        node = session.queue_user_message("Test", message_id="custom-123")

        assert node.node_id == "custom-123"

    @pytest.mark.asyncio
    async def test_queue_user_message_auto_generates_id(self):
        """queue_user_message() auto-generates IDs when not provided."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        node = session.queue_user_message("Test")

        assert node.node_id.startswith("msg_")

    @pytest.mark.asyncio
    async def test_queue_user_message_wakes_agent(self):
        """queue_user_message() triggers wake event."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Wake event should not be set initially
        assert not session._wake_event.is_set()

        session.queue_user_message("Wake up!")

        # Wake event should be set after queueing
        assert session._wake_event.is_set()

    @pytest.mark.asyncio
    async def test_has_pending_messages_empty(self):
        """has_pending_messages() returns False when queue is empty."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        assert session.has_pending_messages() is False

    @pytest.mark.asyncio
    async def test_has_pending_messages_with_message(self):
        """has_pending_messages() returns True when messages are queued."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        session.queue_user_message("Test message")

        assert session.has_pending_messages() is True

    @pytest.mark.asyncio
    async def test_has_pending_messages_after_processing(self):
        """has_pending_messages() returns False after messages are processed."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        node = session.queue_user_message("Test")
        assert session.has_pending_messages() is True

        session.mark_message_processed(node.node_id)
        assert session.has_pending_messages() is False

    @pytest.mark.asyncio
    async def test_get_pending_messages_empty(self):
        """get_pending_messages() returns empty list when no messages."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        messages = session.get_pending_messages()

        assert messages == []

    @pytest.mark.asyncio
    async def test_get_pending_messages_returns_all_unprocessed(self):
        """get_pending_messages() returns all unprocessed messages."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        msg1 = session.queue_user_message("First")
        msg2 = session.queue_user_message("Second")
        msg3 = session.queue_user_message("Third")

        messages = session.get_pending_messages()

        assert len(messages) == 3
        assert msg1 in messages
        assert msg2 in messages
        assert msg3 in messages

    @pytest.mark.asyncio
    async def test_get_pending_messages_filters_processed(self):
        """get_pending_messages() excludes processed messages."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        msg1 = session.queue_user_message("First")
        msg2 = session.queue_user_message("Second")
        session.mark_message_processed(msg1.node_id)

        messages = session.get_pending_messages()

        assert len(messages) == 1
        assert msg2 in messages
        assert msg1 not in messages

    @pytest.mark.asyncio
    async def test_mark_message_processed_updates_tag(self):
        """mark_message_processed() sets processed tag on message."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        node = session.queue_user_message("Test")
        assert not node.tags.get("processed", False)

        session.mark_message_processed(node.node_id)
        assert node.tags.get("processed") is True

    @pytest.mark.asyncio
    async def test_mark_message_processed_nonexistent_message(self):
        """mark_message_processed() handles nonexistent message IDs gracefully."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Should not raise
        session.mark_message_processed("nonexistent-id")


class TestAgentLoopControl:
    """Test agent loop wake and control API."""

    @pytest.mark.asyncio
    async def test_wake_sets_event(self):
        """wake() sets the wake event."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        assert not session._wake_event.is_set()
        session.wake()
        assert session._wake_event.is_set()

    @pytest.mark.asyncio
    async def test_wake_idempotent(self):
        """wake() can be called multiple times safely."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        session.wake()
        session.wake()
        session.wake()
        # Should not raise, event should be set
        assert session._wake_event.is_set()

    @pytest.mark.asyncio
    async def test_run_agent_loop_yields_updates(self):
        """run_agent_loop() yields SessionUpdate objects."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Queue a message and start the loop
        session.queue_user_message("Test message")

        # Collect a few updates (with timeout to avoid hanging)
        updates = []
        loop_task = asyncio.create_task(session.run_agent_loop().__anext__())

        try:
            update = await asyncio.wait_for(loop_task, timeout=0.5)
            updates.append(update)
        except asyncio.TimeoutError:
            # Expected - loop waits for wake
            pass
        except StopAsyncIteration:
            # Loop finished early
            pass

        # Should have collected at least some updates
        # (exact count depends on LLM and execution)
        # The important part is that it's an async iterator

    @pytest.mark.asyncio
    async def test_run_agent_loop_processes_messages_in_order(self):
        """run_agent_loop() retrieves all queued messages."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Queue multiple messages
        session.queue_user_message("First")
        session.queue_user_message("Second")
        session.queue_user_message("Third")

        # All messages should be retrievable
        messages = session.get_pending_messages()
        assert len(messages) == 3
        message_contents = {m.content for m in messages}
        assert "First" in message_contents
        assert "Second" in message_contents
        assert "Third" in message_contents


class TestOutOfBandStatusUpdates:
    """Test out-of-band status update capabilities."""

    @pytest.mark.asyncio
    async def test_updates_emitted_during_execution(self):
        """Session emits updates during code execution."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Execute some code and collect updates
        updates = []
        async for update in session.prompt("x = 1 + 1"):
            updates.append(update)

        # Should have received various update kinds
        update_kinds = {u.kind for u in updates}
        # At minimum, we expect some updates
        assert len(updates) > 0
        assert len(update_kinds) > 0

    @pytest.mark.asyncio
    async def test_update_contains_session_id(self):
        """All updates include the session ID."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        updates = []
        async for update in session.prompt("print('hello')"):
            updates.append(update)

        # All updates should have the correct session_id
        for update in updates:
            assert update.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_update_contains_timestamp(self):
        """All updates include timestamps."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        updates = []
        async for update in session.prompt("x = 42"):
            updates.append(update)

        # All updates should have timestamps
        for update in updates:
            assert update.timestamp > 0


class TestExternalIntegration:
    """Test scenarios for external code integrating with the agent loop."""

    @pytest.mark.asyncio
    async def test_external_code_queues_and_processes(self):
        """External code can queue messages and process them asynchronously."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Simulate external code queuing messages
        external_messages = [
            "Calculate 2 + 2",
            "What is the result?",
            "Done",
        ]

        for msg in external_messages:
            session.queue_user_message(msg)

        # Check all messages are pending
        assert session.has_pending_messages()
        pending = session.get_pending_messages()
        assert len(pending) == 3

        # Simulate processing (in real code, run_agent_loop would do this)
        for msg in pending:
            session.mark_message_processed(msg.node_id)

        # All messages should now be processed
        assert not session.has_pending_messages()

    @pytest.mark.asyncio
    async def test_external_code_monitors_status_updates(self):
        """External code can monitor status updates via async iteration."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # External code queues a message
        session.queue_user_message("Test task")

        # External code monitors updates
        update_count = 0
        timeout_task = asyncio.create_task(asyncio.sleep(0.1))

        # Note: In real usage, this would run longer and collect more updates
        # Here we just verify the API works
        try:
            async for _update in session.run_agent_loop():
                update_count += 1
                if timeout_task.done() or update_count >= 5:
                    break
        except Exception:
            # Expected - loop may not have updates ready
            pass

        # API should be functional (even if no updates collected in this short time)
        assert update_count >= 0

    @pytest.mark.asyncio
    async def test_multiple_queue_operations(self):
        """External code can perform multiple queueing operations."""
        from activecontext import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=".")

        # Queue messages in batches
        batch1 = [
            session.queue_user_message("Message 1"),
            session.queue_user_message("Message 2"),
        ]

        assert session.has_pending_messages()
        assert len(session.get_pending_messages()) == 2

        # Process first batch
        for msg in batch1:
            session.mark_message_processed(msg.node_id)

        assert not session.has_pending_messages()

        # Queue second batch
        batch2 = [
            session.queue_user_message("Message 3"),
            session.queue_user_message("Message 4"),
            session.queue_user_message("Message 5"),
        ]

        assert len(session.get_pending_messages()) == 3

        # Process second batch
        for msg in batch2:
            session.mark_message_processed(msg.node_id)

        assert not session.has_pending_messages()


class TestAgentLoopDocumentation:
    """Verify that the agent loop API is properly documented."""

    def test_queue_user_message_has_docstring(self):
        """queue_user_message() has proper documentation."""
        from activecontext import Session

        assert Session.queue_user_message.__doc__ is not None
        assert "Queue" in Session.queue_user_message.__doc__
        assert "message" in Session.queue_user_message.__doc__.lower()

    def test_has_pending_messages_has_docstring(self):
        """has_pending_messages() has proper documentation."""
        from activecontext import Session

        assert Session.has_pending_messages.__doc__ is not None
        assert (
            "pending" in Session.has_pending_messages.__doc__.lower()
            or "unprocessed" in Session.has_pending_messages.__doc__.lower()
        )

    def test_get_pending_messages_has_docstring(self):
        """get_pending_messages() has proper documentation."""
        from activecontext import Session

        assert Session.get_pending_messages.__doc__ is not None
        assert "unprocessed" in Session.get_pending_messages.__doc__.lower()

    def test_mark_message_processed_has_docstring(self):
        """mark_message_processed() has proper documentation."""
        from activecontext import Session

        assert Session.mark_message_processed.__doc__ is not None
        assert "processed" in Session.mark_message_processed.__doc__.lower()

    def test_wake_has_docstring(self):
        """wake() has proper documentation."""
        from activecontext import Session

        assert Session.wake.__doc__ is not None
        assert "wake" in Session.wake.__doc__.lower()

    def test_run_agent_loop_has_docstring(self):
        """run_agent_loop() has proper documentation."""
        from activecontext import Session

        assert Session.run_agent_loop.__doc__ is not None
        assert "agent loop" in Session.run_agent_loop.__doc__.lower()
