"""AgentLoopManager: Event-driven agent loop lifecycle management.

Extracted from Session to separate concerns:
- Session owns orchestration (LLM, projections, message history)
- AgentLoopManager owns loop lifecycle (wake, run, stop, message queue)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import GroupNode, MessageNode
    from activecontext.session.timeline import Timeline

log = logging.getLogger(__name__)


@dataclass
class SessionUpdate:
    """Update from session to transport layer.

    Imported here to avoid circular dependency - matches the one in session_manager.
    """
    kind: str
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentLoopManager:
    """Manages event-driven agent loop lifecycle.

    Owns:
    - Wake event for async signaling
    - Running state
    - User message queue in context graph

    Delegates:
    - Prompt execution to Session via callback
    - Tick processing to Session via callback

    This separation allows:
    - Testing loop mechanics independently
    - Cleaner Session focused on orchestration
    - Potential for different loop strategies
    """

    def __init__(
        self,
        *,
        session_id: str,
        context_graph: "ContextGraph",
        timeline: "Timeline",
        user_messages_group: "GroupNode | None",
        prompt_callback: Callable[[str], AsyncIterator[Any]],
        tick_callback: Callable[[], Awaitable[list[Any]]],
    ) -> None:
        """Initialize the agent loop manager.

        Args:
            session_id: Session identifier for logging.
            context_graph: The context graph for node operations.
            timeline: Timeline for checking pending work.
            user_messages_group: Group node for message queue.
            prompt_callback: Async iterator callback for prompt processing.
            tick_callback: Async callback for tick processing.
        """
        self._session_id = session_id
        self._context_graph = context_graph
        self._timeline = timeline
        self._user_messages_group = user_messages_group
        self._prompt_callback = prompt_callback
        self._tick_callback = tick_callback

        # Event-driven loop state
        self._wake_event = asyncio.Event()
        self._running = False

        # Track cancellation state
        self._cancelled = False

    # -------------------------------------------------------------------------
    # Message Queue Operations
    # -------------------------------------------------------------------------

    def queue_user_message(
        self,
        content: str,
        message_id: str | None = None,
    ) -> "MessageNode":
        """Queue a user message for async processing.

        Creates a MessageNode and adds it to the user_messages group.
        The message will be processed by the agent loop when it wakes.

        Args:
            content: Message content.
            message_id: Optional explicit message ID.

        Returns:
            The created MessageNode.
        """
        from activecontext.context.nodes import MessageNode
        from activecontext.context.nodes.enums import MessageRole
        from activecontext.context.state import Expansion

        node_id = message_id or f"msg_{int(time.time() * 1000)}"

        msg = MessageNode(
            node_id=node_id,
            role=MessageRole.USER,
            content=content,
            originator="user",
            default_expansion=Expansion.CONTENT,
        )
        self._context_graph.add_node(msg)

        # Link to user_messages group
        if self._user_messages_group:
            self._context_graph.link(node_id, "user_messages")

        # Wake the loop
        self._wake_event.set()

        log.debug("Queued message %s: %s", node_id, content[:50] if content else "")
        return msg

    def has_pending_messages(self) -> bool:
        """Check if there are unprocessed messages in the queue."""
        return len(self.get_pending_messages()) > 0

    def get_pending_messages(self) -> list["MessageNode"]:
        """Get all unprocessed messages from the queue.

        Returns:
            List of MessageNode that haven't been processed yet.
        """
        from activecontext.context.nodes import MessageNode

        children = self._context_graph.get_children("user_messages")
        pending = []
        for child_id in children:
            node = self._context_graph.get_node(child_id)
            if isinstance(node, MessageNode) and not node.processed:
                pending.append(node)
        return pending

    def mark_message_processed(self, message_id: str) -> None:
        """Mark a message as processed.

        Args:
            message_id: ID of the message to mark.
        """
        from activecontext.context.nodes import MessageNode

        node = self._context_graph.get_node(message_id)
        if isinstance(node, MessageNode):
            node.processed = True

    # -------------------------------------------------------------------------
    # Loop Control
    # -------------------------------------------------------------------------

    def wake(self) -> None:
        """Wake the agent loop to process pending work."""
        self._wake_event.set()

    def stop(self) -> None:
        """Stop the agent loop gracefully."""
        self._running = False
        self._wake_event.set()  # Wake it so it can exit

    @property
    def is_running(self) -> bool:
        """Check if the agent loop is currently running."""
        return self._running

    def has_pending_work(self) -> bool:
        """Check if there's work to do.

        Checks:
        - Pending user messages
        - Pending wake prompts from Timeline
        - Queued events in Timeline
        """
        return (
            self.has_pending_messages()
            or self._timeline.has_pending_wake_prompt()
            or len(self._timeline.get_queued_events()) > 0
        )

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    async def run(self) -> AsyncIterator[Any]:
        """Event-driven agent loop. Idle until wake, process until queue empty.

        This is the main processing loop for the async prompt model. It:
        1. Waits for a wake signal (message queued, file changed, etc.)
        2. Processes all pending messages via prompt_callback
        3. Runs tick phase via tick_callback
        4. Yields updates for each significant event

        Yields:
            SessionUpdate objects for streaming to the transport.
        """
        self._running = True
        self._cancelled = False
        log.info("Agent loop started for session %s", self._session_id)

        try:
            while self._running:
                # Wait for wake signal
                try:
                    await self._wake_event.wait()
                except asyncio.CancelledError:
                    log.info("Agent loop cancelled for session %s", self._session_id)
                    self._cancelled = True
                    break

                self._wake_event.clear()
                log.debug("Agent loop woke for session %s", self._session_id)

                # Process all pending work
                while self.has_pending_work() and self._running:
                    # Process next message
                    async for update in self._process_next_message():
                        yield update

                    # Run tick phase
                    tick_updates = await self._tick_callback()
                    for update in tick_updates:
                        yield update

        finally:
            self._running = False
            log.info("Agent loop stopped for session %s", self._session_id)

    async def _process_next_message(self) -> AsyncIterator[Any]:
        """Process the next pending user message.

        Gets the oldest unprocessed message from the user_messages inbox,
        marks it as processed, and processes it through the prompt callback.

        Yields:
            Updates from prompt processing.
        """
        messages = self.get_pending_messages()
        if not messages:
            return

        # Process oldest message first (FIFO)
        msg = messages[0]
        content = msg.content
        message_id = msg.node_id
        log.debug("Processing message %s: %s", message_id, content[:50] if content else "")

        # Mark as processed so it won't be picked up again
        self.mark_message_processed(message_id)

        try:
            # Delegate to Session for actual prompt processing
            async for update in self._prompt_callback(content):
                yield update

            # Send completion notification (skip if cancelled)
            if not self._cancelled:
                yield SessionUpdate(
                    kind="projection_ready",
                    session_id=self._session_id,
                    payload={
                        "message_id": message_id,
                        "completed": True,
                    },
                    timestamp=time.time(),
                )

        except asyncio.CancelledError:
            self._cancelled = True
            raise
        except Exception as e:
            log.error("Error processing message %s: %s", message_id, e)
            yield SessionUpdate(
                kind="error",
                session_id=self._session_id,
                payload={
                    "message_id": message_id,
                    "error": str(e),
                },
                timestamp=time.time(),
            )
