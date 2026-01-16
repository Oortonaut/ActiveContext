"""Session and SessionManager implementations.

The SessionManager is the top-level entry point for both ACP and Direct transports.
Each Session wraps a Timeline and provides the high-level interface for prompts,
ticks, and projections.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from activecontext.session.protocols import (
    Projection,
    SessionUpdate,
    UpdateKind,
)
from activecontext.session.timeline import Timeline

if TYPE_CHECKING:
    from activecontext.core.llm.provider import LLMProvider, Message


class Session:
    """A session wrapping a timeline with prompt and tick handling.

    Implements SessionProtocol.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str,
        timeline: Timeline,
        llm: LLMProvider | None = None,
    ) -> None:
        self._session_id = session_id
        self._cwd = cwd
        self._timeline = timeline
        self._llm = llm
        self._cancelled = False
        self._current_task: asyncio.Task[Any] | None = None
        self._conversation: list[Message] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def llm(self) -> LLMProvider | None:
        return self._llm

    def set_llm(self, llm: LLMProvider | None) -> None:
        """Set or update the LLM provider."""
        self._llm = llm

    async def prompt(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process a user prompt.

        If an LLM provider is configured:
        1. Build projection from current context
        2. Send prompt + projection to LLM
        3. Stream response, parsing and executing code blocks
        4. Run tick phase

        If no LLM:
        - Execute content directly if it looks like Python code
        - Otherwise echo the prompt
        """
        self._cancelled = False

        if self._cancelled:
            return

        if self._llm:
            # LLM-powered mode
            async for update in self._prompt_with_llm(content):
                yield update
        else:
            # Direct execution mode (fallback)
            async for update in self._prompt_direct(content):
                yield update

        # Final projection
        projection = self.get_projection()
        yield SessionUpdate(
            kind=UpdateKind.PROJECTION_READY,
            session_id=self._session_id,
            payload={
                "handles": projection.handles,
                "summaries": projection.summaries,
                "deltas": projection.deltas,
            },
            timestamp=time.time(),
        )

    async def _prompt_with_llm(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process prompt using the LLM provider."""
        from activecontext.core.llm.provider import Message, Role
        from activecontext.core.prompts import (
            SYSTEM_PROMPT,
            build_user_message,
            parse_response,
        )

        # Build the user message with context
        projection = self.get_projection()
        user_content = build_user_message(content, projection)

        # Build messages for LLM
        messages = [Message(role=Role.SYSTEM, content=SYSTEM_PROMPT)]
        messages.extend(self._conversation)
        messages.append(Message(role=Role.USER, content=user_content))

        # Stream response from LLM
        full_response = ""
        async for chunk in self._llm.stream(messages):  # type: ignore[union-attr]
            if chunk.text:
                full_response += chunk.text
                yield SessionUpdate(
                    kind=UpdateKind.RESPONSE_CHUNK,
                    session_id=self._session_id,
                    payload={"text": chunk.text},
                    timestamp=time.time(),
                )

        # Update conversation history
        self._conversation.append(Message(role=Role.USER, content=user_content))
        self._conversation.append(Message(role=Role.ASSISTANT, content=full_response))

        # Parse response and execute code blocks
        parsed = parse_response(full_response)
        for segment_type, segment_content in parsed.segments:
            if segment_type == "code" and segment_content:
                # Execute the code block
                async for update in self._execute_code(segment_content):
                    yield update

        # Run tick phase
        tick_updates = await self.tick()
        for update in tick_updates:
            yield update

    async def _prompt_direct(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process prompt in direct execution mode (no LLM)."""
        # Check if content looks like Python code
        is_code = any(
            content.strip().startswith(prefix)
            for prefix in ("import ", "from ", "def ", "class ", "=", "view(", "group(")
        ) or "=" in content

        if is_code:
            async for update in self._execute_code(content):
                yield update
            tick_updates = await self.tick()
            for update in tick_updates:
                yield update
        else:
            yield SessionUpdate(
                kind=UpdateKind.RESPONSE_CHUNK,
                session_id=self._session_id,
                payload={"text": f"[No LLM configured] Received: {content}"},
                timestamp=time.time(),
            )

    async def _execute_code(self, source: str) -> AsyncIterator[SessionUpdate]:
        """Execute a code block and yield updates."""
        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_PARSED,
            session_id=self._session_id,
            payload={"source": source},
            timestamp=time.time(),
        )

        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTING,
            session_id=self._session_id,
            payload={"source": source},
            timestamp=time.time(),
        )

        result = await self._timeline.execute_statement(source)

        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTED,
            session_id=self._session_id,
            payload={
                "execution_id": result.execution_id,
                "statement_id": result.statement_id,
                "status": result.status.value,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exception": result.exception,
                "state_diff": {
                    "added": result.state_diff.added,
                    "changed": result.state_diff.changed,
                    "deleted": result.state_diff.deleted,
                },
                "duration_ms": result.duration_ms,
            },
            timestamp=time.time(),
        )

    async def tick(self) -> list[SessionUpdate]:
        """Run the tick phase for all context objects.

        Tick ordering:
        1. Apply ready async payloads (TODO)
        2. Sync ticks for running nodes
        3. Periodic ticks (TODO)
        4. Group recompute
        """
        updates: list[SessionUpdate] = []
        timestamp = time.time()

        context_objects = self._timeline.get_context_objects()

        for obj_id, obj in context_objects.items():
            if hasattr(obj, "mode") and obj.mode == "running":
                updates.append(
                    SessionUpdate(
                        kind=UpdateKind.TICK_APPLIED,
                        session_id=self._session_id,
                        payload={
                            "object_id": obj_id,
                            "tick_kind": "sync",
                            "digest": obj.GetDigest() if hasattr(obj, "GetDigest") else {},
                        },
                        timestamp=timestamp,
                    )
                )

        return updates

    async def cancel(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    def get_projection(self) -> Projection:
        """Build the LLM projection from current session state."""
        context_objects = self._timeline.get_context_objects()

        handles = {}
        for obj_id, obj in context_objects.items():
            if hasattr(obj, "GetDigest"):
                handles[obj_id] = obj.GetDigest()

        return Projection(
            handles=handles,
            summaries=[],
            deltas=[],
            token_budget=8000,
        )

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()


class SessionManager:
    """Manages multiple sessions with 1:1 session-timeline mapping.

    Implements SessionManagerProtocol.
    """

    def __init__(self, default_llm: LLMProvider | None = None) -> None:
        """Initialize the session manager.

        Args:
            default_llm: Default LLM provider for new sessions
        """
        self._sessions: dict[str, Session] = {}
        self._default_llm = default_llm

    def set_default_llm(self, llm: LLMProvider | None) -> None:
        """Set the default LLM provider for new sessions."""
        self._default_llm = llm

    async def create_session(
        self,
        cwd: str,
        session_id: str | None = None,
        llm: LLMProvider | None = None,
    ) -> Session:
        """Create a new session with its own timeline.

        Args:
            cwd: Working directory for the session
            session_id: Optional specific ID; generated if not provided
            llm: Optional LLM provider; uses default if not provided
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        timeline = Timeline(session_id)
        session = Session(
            session_id=session_id,
            cwd=cwd,
            timeline=timeline,
            llm=llm or self._default_llm,
        )
        self._sessions[session_id] = session

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    async def load_session(
        self,
        session_id: str,
        cwd: str,
    ) -> Session | None:
        """Load a previously persisted session.

        TODO: Implement persistence loading. For now, returns None.
        """
        return None

    async def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    async def close_session(self, session_id: str) -> None:
        """Close and clean up a session."""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            await session.cancel()
