"""Tests for BlockingConversationTask."""

from __future__ import annotations

import pytest

from activecontext.session.protocols import IOMode, TaskStatus
from activecontext.session.tasks import BlockingConversationTask


class TestBlockingConversationTaskBasics:
    """Basic BlockingConversationTask tests."""

    def test_create_default(self) -> None:
        """Test creating a task with defaults."""
        task = BlockingConversationTask()
        assert task.task_id.startswith("conv-")
        assert task.task_type == "blocking_conversation"
        assert task.io_mode == IOMode.SYNC
        assert task.status == TaskStatus.PENDING

    def test_create_with_id_and_name(self) -> None:
        """Test creating a task with custom id and name."""
        task = BlockingConversationTask(task_id="my-task", name="My Conversation")
        assert task.task_id == "my-task"
        assert task._name == "My Conversation"

    @pytest.mark.asyncio
    async def test_lifecycle(self) -> None:
        """Test task lifecycle (start, pause, resume, stop)."""
        task = BlockingConversationTask()

        # Initial state
        assert task.status == TaskStatus.PENDING

        # Start
        await task.start()
        assert task.status == TaskStatus.RUNNING
        assert task._started_at is not None

        # Pause
        await task.pause()
        assert task.status == TaskStatus.PAUSED

        # Resume
        await task.resume()
        assert task.status == TaskStatus.RUNNING

        # Stop
        await task.stop()
        assert task.status == TaskStatus.DONE
        assert task._completed_at is not None


class TestBlockingConversationTaskSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self) -> None:
        """Test serializing a task."""
        task = BlockingConversationTask(task_id="test-1", name="Test Task")
        data = task.to_dict()

        assert data["task_id"] == "test-1"
        assert data["task_type"] == "blocking_conversation"
        assert data["name"] == "Test Task"
        assert data["status"] == "pending"
        assert data["io_mode"] == "sync"
        assert "created_at" in data

    def test_from_dict(self) -> None:
        """Test deserializing a task."""
        data = {
            "task_id": "restored-1",
            "name": "Restored Task",
            "status": "running",
            "created_at": 1000.0,
            "started_at": 1001.0,
        }
        task = BlockingConversationTask.from_dict(data)

        assert task.task_id == "restored-1"
        assert task._name == "Restored Task"
        assert task.status == TaskStatus.RUNNING
        assert task._created_at == 1000.0
        assert task._started_at == 1001.0


class TestBlockingConversationTaskInteraction:
    """Test conversation interaction methods."""

    @pytest.mark.asyncio
    async def test_ask_with_callback(self) -> None:
        """Test asking a question with a response callback."""
        responses = ["Option A"]

        async def callback(question: dict) -> str:
            return responses.pop(0)

        task = BlockingConversationTask(response_callback=callback)
        await task.start()

        result = await task.ask("Pick one:", ["Option A", "Option B", "Option C"])
        assert result == "Option A"
        assert len(task.questions) == 1
        assert len(task.responses) == 1
        assert task.questions[0]["type"] == "select"
        assert task.questions[0]["prompt"] == "Pick one:"

    @pytest.mark.asyncio
    async def test_confirm_yes(self) -> None:
        """Test confirm returns True for 'yes'."""
        async def callback(question: dict) -> str:
            return "Yes"

        task = BlockingConversationTask(response_callback=callback)
        await task.start()

        result = await task.confirm("Are you sure?")
        assert result is True

    @pytest.mark.asyncio
    async def test_confirm_no(self) -> None:
        """Test confirm returns False for 'No'."""
        async def callback(question: dict) -> str:
            return "No"

        task = BlockingConversationTask(response_callback=callback)
        await task.start()

        result = await task.confirm("Are you sure?")
        assert result is False

    @pytest.mark.asyncio
    async def test_text_input(self) -> None:
        """Test text input with response."""
        async def callback(question: dict) -> str:
            return "user input"

        task = BlockingConversationTask(response_callback=callback)
        await task.start()

        result = await task.text_input("Enter your name:", default="Anonymous")
        assert result == "user input"
        assert task.questions[0]["type"] == "text"
        assert task.questions[0]["default"] == "Anonymous"

    @pytest.mark.asyncio
    async def test_text_input_empty_uses_default(self) -> None:
        """Test that empty text input uses default."""
        async def callback(question: dict) -> str:
            return ""

        task = BlockingConversationTask(response_callback=callback)
        await task.start()

        result = await task.text_input("Enter your name:", default="Anonymous")
        assert result == "Anonymous"

    @pytest.mark.asyncio
    async def test_ask_requires_running(self) -> None:
        """Test that ask raises if task not running."""
        async def callback(question: dict) -> str:
            return "test"

        task = BlockingConversationTask(response_callback=callback)
        # Don't start the task

        with pytest.raises(RuntimeError, match="task status is pending"):
            await task.ask("Question?", ["A", "B"])

    @pytest.mark.asyncio
    async def test_ask_requires_callback(self) -> None:
        """Test that ask raises if no callback."""
        task = BlockingConversationTask()  # No callback
        await task.start()

        with pytest.raises(RuntimeError, match="No response callback"):
            await task.ask("Question?", ["A", "B"])


class TestBlockingConversationTaskHistory:
    """Test conversation history tracking."""

    @pytest.mark.asyncio
    async def test_conversation_history(self) -> None:
        """Test tracking full conversation history."""
        responses = ["A", "B", "my text"]

        async def callback(question: dict) -> str:
            return responses.pop(0)

        task = BlockingConversationTask(response_callback=callback)
        await task.start()

        await task.ask("Question 1?", ["A", "B", "C"])
        await task.ask("Question 2?", ["A", "B"])
        await task.text_input("Enter text:")

        history = task.conversation_history
        assert len(history) == 3
        assert history[0]["response"] == "A"
        assert history[1]["response"] == "B"
        assert history[2]["response"] == "my text"


class TestBlockingConversationTaskInjection:
    """Test external response injection for UI integration."""

    @pytest.mark.asyncio
    async def test_inject_response(self) -> None:
        """Test injecting a response externally."""
        task = BlockingConversationTask()
        await task.start()

        task.inject_response("injected value")
        result = await task.wait_for_injected_response()
        assert result == "injected value"

    @pytest.mark.asyncio
    async def test_inject_response_empty(self) -> None:
        """Test waiting with no injection returns empty string."""
        import asyncio

        task = BlockingConversationTask()
        await task.start()

        # Inject empty response
        task.inject_response("")

        result = await task.wait_for_injected_response()
        assert result == ""
