"""BlockingConversationTask: Sync request/response task.

A BlockingConversationTask is a synchronous conversation flow where
each request blocks until a response is received. Examples include:
- MCP server selection menu
- Permission dialogs
- User confirmations

Unlike Agent tasks, BlockingConversationTask:
- Does not have a Timeline
- Does not access the ContextGraph
- Has private state for the conversation
- Uses IOMode.SYNC for request/response patterns
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any

from activecontext.logging import get_logger
from activecontext.session.protocols import IOMode, TaskStatus

log = get_logger("blocking_conversation")

if TYPE_CHECKING:
    pass


class BlockingConversationTask:
    """Sync request/response task for user interactions.

    This task type handles blocking conversations where the system
    presents a question or menu and waits for user response.

    Attributes:
        task_id: Unique identifier for this task
        task_type: Always "blocking_conversation"
        io_mode: Always IOMode.SYNC
        status: Current TaskStatus
    """

    def __init__(
        self,
        task_id: str | None = None,
        name: str = "",
        response_callback: Any = None,
    ) -> None:
        """Initialize a BlockingConversationTask.

        Args:
            task_id: Unique identifier (generated if None)
            name: Human-readable name for the task
            response_callback: Callback to deliver questions to user and get responses.
                               Signature: async def callback(question: dict) -> str
        """
        self._task_id = task_id or f"conv-{uuid.uuid4().hex[:8]}"
        self._name = name
        self._status = TaskStatus.PENDING
        self._response_callback = response_callback

        # Conversation state
        self._questions: list[dict[str, Any]] = []
        self._responses: list[str] = []
        self._created_at = time.time()
        self._started_at: float | None = None
        self._completed_at: float | None = None

        # For blocking on responses
        self._response_event = asyncio.Event()
        self._pending_response: str | None = None

    # -------------------------------------------------------------------------
    # TaskProtocol implementation
    # -------------------------------------------------------------------------

    @property
    def task_id(self) -> str:
        """Unique identifier for this task."""
        return self._task_id

    @property
    def task_type(self) -> str:
        """Type of task."""
        return "blocking_conversation"

    @property
    def io_mode(self) -> IOMode:
        """I/O mode for this task."""
        return IOMode.SYNC

    @property
    def status(self) -> TaskStatus:
        """Current status of the task."""
        return self._status

    async def start(self) -> None:
        """Start the task."""
        self._status = TaskStatus.RUNNING
        self._started_at = time.time()
        log.debug("Started blocking conversation task %s", self._task_id)

    async def pause(self) -> None:
        """Pause the task (not typically used for blocking conversations)."""
        self._status = TaskStatus.PAUSED
        log.debug("Paused blocking conversation task %s", self._task_id)

    async def resume(self) -> None:
        """Resume the task."""
        self._status = TaskStatus.RUNNING
        log.debug("Resumed blocking conversation task %s", self._task_id)

    async def stop(self) -> None:
        """Stop and clean up the task."""
        self._status = TaskStatus.DONE
        self._completed_at = time.time()
        # Wake up any waiting calls
        self._response_event.set()
        log.debug("Stopped blocking conversation task %s", self._task_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize task state for persistence."""
        return {
            "task_id": self._task_id,
            "task_type": self.task_type,
            "name": self._name,
            "status": self._status.value,
            "io_mode": self.io_mode.value,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "question_count": len(self._questions),
            "response_count": len(self._responses),
        }

    # -------------------------------------------------------------------------
    # Conversation methods
    # -------------------------------------------------------------------------

    async def ask(self, prompt: str, options: list[str]) -> str:
        """Ask the user a question with predefined options.

        Args:
            prompt: The question to ask
            options: List of valid options

        Returns:
            The selected option (one of the options list)

        Raises:
            RuntimeError: If task is not running or no response callback
        """
        if self._status != TaskStatus.RUNNING:
            raise RuntimeError(f"Cannot ask: task status is {self._status.value}")

        if not self._response_callback:
            raise RuntimeError("No response callback configured")

        question = {
            "type": "select",
            "prompt": prompt,
            "options": options,
            "timestamp": time.time(),
        }
        self._questions.append(question)

        # Deliver question and wait for response
        response: str = await self._response_callback(question)
        self._responses.append(response)

        return response

    async def confirm(self, message: str) -> bool:
        """Ask the user for a yes/no confirmation.

        Args:
            message: The confirmation message

        Returns:
            True if confirmed, False otherwise
        """
        response = await self.ask(message, ["Yes", "No"])
        return response.lower() in ("yes", "y", "true", "1")

    async def text_input(self, prompt: str, default: str = "") -> str:
        """Ask the user for text input.

        Args:
            prompt: The input prompt
            default: Default value if user enters nothing

        Returns:
            The user's input text

        Raises:
            RuntimeError: If task is not running or no response callback
        """
        if self._status != TaskStatus.RUNNING:
            raise RuntimeError(f"Cannot ask: task status is {self._status.value}")

        if not self._response_callback:
            raise RuntimeError("No response callback configured")

        question = {
            "type": "text",
            "prompt": prompt,
            "default": default,
            "timestamp": time.time(),
        }
        self._questions.append(question)

        response: str = await self._response_callback(question)
        self._responses.append(response)

        return response if response else default

    # -------------------------------------------------------------------------
    # External response injection (for testing or UI integration)
    # -------------------------------------------------------------------------

    def inject_response(self, response: str) -> None:
        """Inject a response externally (for testing or UI integration).

        This allows external code to provide responses when not using
        the response_callback pattern.

        Args:
            response: The response to inject
        """
        self._pending_response = response
        self._response_event.set()

    async def wait_for_injected_response(self) -> str:
        """Wait for an injected response.

        Returns:
            The injected response
        """
        await self._response_event.wait()
        self._response_event.clear()
        response: str = self._pending_response if self._pending_response else ""
        self._pending_response = None
        return response

    # -------------------------------------------------------------------------
    # Conversation history
    # -------------------------------------------------------------------------

    @property
    def questions(self) -> list[dict[str, Any]]:
        """Get all questions asked."""
        return list(self._questions)

    @property
    def responses(self) -> list[str]:
        """Get all responses received."""
        return list(self._responses)

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        """Get the full conversation history as Q&A pairs."""
        history = []
        for i, question in enumerate(self._questions):
            response = self._responses[i] if i < len(self._responses) else None
            history.append({
                "question": question,
                "response": response,
            })
        return history

    # -------------------------------------------------------------------------
    # Factory methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockingConversationTask":
        """Create a task from serialized data.

        Args:
            data: Serialized task data from to_dict()

        Returns:
            Reconstructed BlockingConversationTask
        """
        task = cls(
            task_id=data.get("task_id"),
            name=data.get("name", ""),
        )
        task._status = TaskStatus(data.get("status", "pending"))
        task._created_at = data.get("created_at", time.time())
        task._started_at = data.get("started_at")
        task._completed_at = data.get("completed_at")
        return task
