"""Conversation coordination for delegated handlers.

This module provides the infrastructure for delegating conversations to
external handlers (shells, menus, debuggers, etc.) while maintaining
the unified MessageNode flow through the context graph.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import MessageNode
from activecontext.context.state import Expansion
from activecontext.protocols.conversation import InputType
from activecontext.session.protocols import SessionUpdate, UpdateKind

if TYPE_CHECKING:
    from activecontext.protocols.conversation import ConversationHandler
    from activecontext.session.session_manager import Session


class SessionConversationTransport:
    """Implementation of ConversationTransport using Session's MessageNode infrastructure.

    All communication flows through Session._add_message() → MessageNode → ContextGraph.
    This ensures delegated conversations are visible in projections and maintain
    the unified message flow.

    Example:
        >>> transport = SessionConversationTransport(session, originator="shell:bash")
        >>> await transport.send_output("$ ls -la")
        >>> response = await transport.request_input("")  # Empty prompt - shell provides its own
        >>> print(f"User entered: {response}")
    """

    def __init__(
        self,
        session: Session,
        originator: str,
        forward_permissions: bool = True,
        update_callback: Any | None = None,  # Callable[[SessionUpdate], Awaitable[None]]
    ):
        """Initialize conversation transport.

        Args:
            session: Session that owns this conversation
            originator: Identifier for MessageNodes (e.g., "shell:bash", "mcp:menu")
            forward_permissions: If True, handler inherits session permissions
            update_callback: Optional async callback for emitting SessionUpdates
                (for ACP integration)
        """
        self._session = session
        self._originator = originator
        self._forward_permissions = forward_permissions
        self._response_queue: asyncio.Queue[str] = asyncio.Queue()
        self._cancelled = False
        self._update_callback = update_callback

    async def send_output(
        self,
        text: str,
        *,
        append: bool = True,
        originator: str | None = None,
    ) -> None:
        """Send output to user (creates MessageNode with handler's originator).

        Args:
            text: Output text to display
            append: If True, append to existing output. If False, replace.
            originator: Override originator for this message (default: handler's originator)
        """
        # Create MessageNode in context graph
        node = MessageNode(
            role="user",  # All non-agent messages use role="user"
            originator=originator or self._originator,
            content=text,
            expansion=Expansion.DETAILS,
        )
        self._session.timeline.context_graph.add_node(node)

        # Emit SessionUpdate for real-time display in IDE
        await self._emit_update(
            SessionUpdate(
                kind=UpdateKind.RESPONSE_CHUNK,
                session_id=self._session.session_id,
                payload={"text": text},
                timestamp=time.time(),
            )
        )

    async def send_progress(
        self,
        current: int,
        total: int,
        *,
        status: str = "",
        show_percentage: bool = True,
    ) -> None:
        """Send progress update (non-blocking).

        Args:
            current: Current progress value
            total: Total progress value
            status: Optional status message
            show_percentage: Show percentage calculation
        """
        payload: dict[str, Any] = {
            "current": current,
            "total": total,
            "originator": self._originator,
        }
        if status:
            payload["status"] = status
        if show_percentage and total > 0:
            payload["percentage"] = (current / total) * 100

        await self._emit_update(
            SessionUpdate(
                kind=UpdateKind.CONVERSATION_PROGRESS,
                session_id=self._session.session_id,
                payload=payload,
                timestamp=time.time(),
            )
        )

    async def request_input(
        self,
        prompt: str,
        *,
        input_type: InputType = InputType.TEXT,
        choices: list[str] | None = None,
        default: str | None = None,
    ) -> str:
        """Request input from user (blocking until response).

        Flow:
        1. Creates MessageNode(role="user", originator=handler_originator, content=prompt)
        2. Emits CONVERSATION_INPUT_REQUEST update (triggers IDE prompt)
        3. Waits for response in queue
        4. Creates MessageNode(role="user", originator="user", content=response)
        5. Returns response string

        Args:
            prompt: Prompt text to display
            input_type: Type of input to request
            choices: Available choices (for CHOICE type)
            default: Default value

        Returns:
            User's input response

        Raises:
            asyncio.CancelledError: If conversation is cancelled
        """
        # Add prompt as MessageNode if non-empty
        if prompt:
            prompt_node = MessageNode(
                role="user",
                originator=self._originator,
                content=prompt,
                expansion=Expansion.DETAILS,
            )
            self._session.timeline.context_graph.add_node(prompt_node)

        # Emit prompt as RESPONSE_CHUNK (not custom INPUT_REQUEST)
        await self._emit_update(
            SessionUpdate(
                kind=UpdateKind.RESPONSE_CHUNK,
                session_id=self._session.session_id,
                payload={"text": prompt},
                timestamp=time.time(),
            )
        )

        # Wait for response (raises CancelledError if cancelled)
        response = await self._response_queue.get()

        # Add response as MessageNode
        response_node = MessageNode(
            role="user",
            originator="user",  # Response always from user
            content=response,
            expansion=Expansion.DETAILS,
        )
        self._session.timeline.context_graph.add_node(response_node)

        return response

    def handle_input_response(self, response: str) -> None:
        """Called when user responds (from ACP transport or coordinator).

        Args:
            response: User's input response
        """
        self._response_queue.put_nowait(response)

    def check_cancelled(self) -> bool:
        """Check if user cancelled (regular code path, not exception).

        Returns:
            True if conversation was cancelled
        """
        return self._cancelled

    def check_input(self) -> bool:
        """Check if input is available without blocking (for sync polling).

        Returns:
            True if input response is queued and available
        """
        return not self._response_queue.empty()

    def get_session_id(self) -> str:
        """Get session ID for this conversation.

        Returns:
            Session identifier
        """
        return self._session.session_id

    def get_originator(self) -> str:
        """Get the originator identifier for this conversation.

        Returns:
            Originator string (e.g., "shell:bash", "mcp:menu")
        """
        return self._originator

    async def _emit_update(self, update: SessionUpdate) -> None:
        """Emit a SessionUpdate via the session's update mechanism.

        Args:
            update: Update to emit
        """
        # Phase 2 (ACP Integration): Emit updates via callback if provided
        if self._update_callback:
            await self._update_callback(update)


class ConversationHandle:
    """Handle for manually operating a delegated conversation.

    Allows coordinator agents to:
    - Set up connections without blocking
    - Check if handler is waiting for input
    - Send responses without detaching
    - Attach input messages to context nodes for inspection

    Example:
        >>> handle = ConversationHandle(handler, transport)
        >>> await handle.start()
        >>> while not handle.is_done():
        ...     if handle.is_waiting():
        ...         prompt = handle.get_last_prompt_node()
        ...         await handle.send_input("response")
        >>> result = await handle.wait()
    """

    def __init__(
        self,
        handler: ConversationHandler,
        transport: SessionConversationTransport,
    ):
        """Initialize conversation handle.

        Args:
            handler: Handler to run
            transport: Transport for communication
        """
        self._handler = handler
        self._transport = transport
        self._task: asyncio.Task[Any] | None = None
        self._result: Any = None

    async def start(self) -> None:
        """Start the handler in background (non-blocking)."""
        self._task = asyncio.create_task(self._handler.handle(self._transport))

    def is_waiting(self) -> bool:
        """Check if handler is waiting for input.

        Returns:
            True if handler is blocked on input request
        """
        return not self._transport.check_input() and not self.is_done()

    def is_done(self) -> bool:
        """Check if handler has completed.

        Returns:
            True if handler finished execution
        """
        return self._task is not None and self._task.done()

    async def send_input(self, response: str) -> None:
        """Send input response to handler.

        Args:
            response: Input response text
        """
        self._transport.handle_input_response(response)

    def get_last_prompt_node(self) -> MessageNode | None:
        """Get the most recent input request as a MessageNode.

        Allows coordinator to inspect what the handler is asking for.

        Returns:
            Most recent MessageNode with handler's originator, or None
        """
        graph = self._transport._session.timeline.context_graph
        # ContextGraph is iterable, so convert to list
        nodes = list(graph)

        candidates = [
            n
            for n in nodes
            if isinstance(n, MessageNode)
            and n.originator == self._transport.get_originator()
            and n.role == "user"
        ]

        if not candidates:
            return None

        # Return most recent (highest created_at)
        return max(candidates, key=lambda n: n.created_at)

    async def wait(self) -> Any:
        """Wait for handler to complete and return result.

        Returns:
            Handler's return value

        Raises:
            asyncio.CancelledError: If handler was cancelled
        """
        if self._task:
            self._result = await self._task
        return self._result

    async def cancel(self) -> None:
        """Cancel the handler."""
        if self._task and not self._task.done():
            self._transport._cancelled = True
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
