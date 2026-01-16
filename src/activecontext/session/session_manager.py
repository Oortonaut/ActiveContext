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

from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine
from activecontext.logging import get_logger
from activecontext.session.protocols import (
    Projection,
    SessionUpdate,
    UpdateKind,
)
from activecontext.session.timeline import Timeline

log = get_logger("session")

if TYPE_CHECKING:
    from activecontext.config.schema import Config
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
        config: Config | None = None,
    ) -> None:
        self._session_id = session_id
        self._cwd = cwd
        self._timeline = timeline
        self._llm = llm
        self._config = config
        self._cancelled = False
        self._current_task: asyncio.Task[Any] | None = None
        self._conversation: list[Message] = []
        self._projection_engine = self._build_projection_engine(config)

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

    def _build_projection_engine(self, config: Config | None) -> ProjectionEngine:
        """Build ProjectionEngine from config or defaults."""
        if config and config.projection:
            proj = config.projection
            projection_config = ProjectionConfig(
                total_budget=proj.total_budget if proj.total_budget is not None else 8000,
                conversation_ratio=(
                    proj.conversation_ratio if proj.conversation_ratio is not None else 0.3
                ),
                views_ratio=proj.views_ratio if proj.views_ratio is not None else 0.5,
                groups_ratio=proj.groups_ratio if proj.groups_ratio is not None else 0.2,
            )
            return ProjectionEngine(projection_config)
        return ProjectionEngine()

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
        """Process prompt using the LLM provider.

        Runs an agent loop: LLM responds, code is executed, results feed back
        to LLM until it calls done() or produces no code blocks.
        """
        import os

        from activecontext.core.llm.provider import Message, Role
        from activecontext.core.prompts import SYSTEM_PROMPT, parse_response

        # Reset done signal at start of each prompt
        self._timeline.reset_done()

        max_iterations = 10  # Safety limit
        iteration = 0
        current_request = content

        while iteration < max_iterations:
            iteration += 1

            # Build projection (contains conversation history and view contents)
            projection = self.get_projection()

            # Build messages: system prompt + projection + current request
            projection_content = projection.render()
            if projection_content:
                user_message = f"{projection_content}\n\n## Current Request\n\n{current_request}"
            else:
                user_message = current_request

            # Debug logging
            if os.environ.get("ACTIVECONTEXT_DEBUG"):
                tokens_est = len(projection_content) // 4 if projection_content else 0
                log.debug("=== ITERATION %d ===", iteration)
                log.debug("=== PROJECTION (%d tokens) ===", tokens_est)
                log.debug("%s", projection_content or "(empty)")
                log.debug("=== END PROJECTION ===")
                log.debug(
                    "Request: %s%s",
                    current_request[:200],
                    "..." if len(current_request) > 200 else "",
                )

            messages = [
                Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                Message(role=Role.USER, content=user_message),
            ]

            # Stream response from LLM
            full_response = ""
            async for chunk in self._llm.stream(messages):  # type: ignore[union-attr]
                if self._cancelled:
                    return
                if chunk.text:
                    full_response += chunk.text
                    yield SessionUpdate(
                        kind=UpdateKind.RESPONSE_CHUNK,
                        session_id=self._session_id,
                        payload={"text": chunk.text},
                        timestamp=time.time(),
                    )

            # Update conversation history
            self._conversation.append(
                Message(role=Role.USER, content=current_request, actor="user")
            )
            self._conversation.append(
                Message(role=Role.ASSISTANT, content=full_response, actor="agent")
            )

            # Parse response and execute code blocks
            parsed = parse_response(full_response)
            code_blocks = parsed.code_blocks
            execution_results: list[str] = []

            for code in code_blocks:
                if self._cancelled:
                    return
                async for update in self._execute_code(code):
                    yield update
                    # Collect execution output for feedback
                    if update.kind == UpdateKind.STATEMENT_EXECUTED:
                        stdout = update.payload.get("stdout", "")
                        stderr = update.payload.get("stderr", "")
                        exception = update.payload.get("exception")
                        if stdout:
                            execution_results.append(f"Output:\n{stdout}")
                        if stderr:
                            execution_results.append(f"Stderr:\n{stderr}")
                        if exception:
                            execution_results.append(
                                f"Error: {exception.get('type')}: {exception.get('message')}"
                            )

            # Run tick phase
            tick_updates = await self.tick()
            for update in tick_updates:
                yield update

            # Check if agent called done()
            if self._timeline.is_done():
                log.debug("Agent called done(), stopping loop")
                break

            # If no code was executed, the agent is done (legacy behavior)
            if not code_blocks:
                log.debug("No code blocks, stopping loop")
                break

            # Otherwise, loop back with execution results as the next request
            if execution_results:
                current_request = "Execution results:\n" + "\n".join(execution_results)
            else:
                current_request = "Code executed successfully. Continue or respond."

    async def _prompt_direct(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process prompt in direct execution mode (no LLM)."""
        # Check if content looks like Python code
        is_code = (
            any(
                content.strip().startswith(prefix)
                for prefix in ("import ", "from ", "def ", "class ", "=", "view(", "group(")
            )
            or "=" in content
        )

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
        return self._projection_engine.build(
            context_objects=self._timeline.get_context_objects(),
            conversation=self._conversation,
            cwd=self._cwd,
        )

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all tracked context objects (views, groups)."""
        return self._timeline.get_context_objects()

    async def _setup_initial_context(self) -> None:
        """Set up initial context views for a new session.

        Creates an example view of CONTEXT_GUIDE.md if it exists,
        demonstrating the view() DSL.
        """
        from pathlib import Path

        # Check for context guide in cwd or package location
        guide_paths = [
            Path(self._cwd) / "CONTEXT_GUIDE.md",
            Path(__file__).parent.parent.parent.parent / "CONTEXT_GUIDE.md",
        ]

        for guide_path in guide_paths:
            if guide_path.exists():
                # Execute the view creation as if the LLM did it
                # This shows the user the DSL syntax as an example
                rel_path = guide_path.name
                try:
                    # Make path relative to cwd if possible
                    rel_path = str(guide_path.relative_to(self._cwd))
                except ValueError:
                    rel_path = str(guide_path)

                # Use forward slashes for cross-platform compatibility
                safe_path = rel_path.replace("\\", "/")
                source = f'guide = view("{safe_path}", tokens=1500)'
                await self._timeline.execute_statement(source)
                break


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

        # Load project-level config (merges with system/user config)
        project_config = self._load_project_config(cwd)

        timeline = Timeline(session_id, cwd=cwd)
        session = Session(
            session_id=session_id,
            cwd=cwd,
            timeline=timeline,
            llm=llm or self._default_llm,
            config=project_config,
        )
        self._sessions[session_id] = session

        # Initialize with example context view if guide exists
        await session._setup_initial_context()

        return session

    def _load_project_config(self, cwd: str) -> Config | None:
        """Load project-level config for a session.

        Args:
            cwd: Working directory (project root) for the session.

        Returns:
            Merged config including project-level settings, or None if unavailable.
        """
        try:
            from activecontext.config import load_config

            return load_config(session_root=cwd)
        except ImportError:
            log.debug("Config module not available")
            return None
        except Exception as e:
            log.warning("Failed to load project config: %s", e)
            return None

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
