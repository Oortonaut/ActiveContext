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

from activecontext.context.graph import ContextGraph
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine
from activecontext.logging import get_logger
from activecontext.session.permissions import PermissionManager, ShellPermissionManager
from activecontext.session.protocols import (
    Projection,
    SessionUpdate,
    UpdateKind,
)
from activecontext.session.timeline import Timeline

log = get_logger("session")

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from activecontext.config.schema import Config
    from activecontext.core.llm.provider import LLMProvider, Message
    from activecontext.terminal.protocol import TerminalExecutor

    # Type for file permission requester callback:
    # async (session_id, path, mode) -> (granted, persist)
    PermissionRequester = Callable[
        [str, str, str], Coroutine[Any, Any, tuple[bool, bool]]
    ]

    # Type for shell permission requester callback:
    # async (session_id, command, args) -> (granted, persist)
    ShellPermissionRequester = Callable[
        [str, str, list[str] | None], Coroutine[Any, Any, tuple[bool, bool]]
    ]


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
        """Run the tick phase for context graph nodes.

        Tick ordering:
        1. Apply ready async payloads (TODO)
        2. Sync ticks for running nodes (calls Recompute → notify_parents cascade)
        3. Periodic ticks based on tick_freq
        4. Group summaries auto-invalidated via on_child_changed() during cascade
        """
        updates: list[SessionUpdate] = []
        timestamp = time.time()

        # Get running nodes from the context graph
        context_graph = self._timeline.context_graph
        running_nodes = context_graph.get_running_nodes()

        for node in running_nodes:
            tick_kind: str | None = None

            # Check tick frequency
            if node.tick_freq == "Sync":
                # Sync tick: recompute on every tick
                tick_kind = "sync"
            elif node.tick_freq and node.tick_freq.startswith("Periodic:"):
                # Periodic tick: check interval
                try:
                    interval_str = node.tick_freq.split(":")[1]
                    # Parse interval (e.g., "5s" -> 5.0 seconds)
                    if interval_str.endswith("s"):
                        interval = float(interval_str[:-1])
                    elif interval_str.endswith("m"):
                        interval = float(interval_str[:-1]) * 60
                    else:
                        interval = float(interval_str)

                    if timestamp - node.updated_at >= interval:
                        tick_kind = "periodic"
                except (ValueError, IndexError):
                    log.warning("Invalid tick frequency: %s", node.tick_freq)

            if tick_kind:
                # Call Recompute which triggers notify_parents() cascade
                node.Recompute()
                node.updated_at = timestamp

                updates.append(
                    SessionUpdate(
                        kind=UpdateKind.TICK_APPLIED,
                        session_id=self._session_id,
                        payload={
                            "node_id": node.node_id,
                            "tick_kind": tick_kind,
                            "digest": node.GetDigest(),
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
            context_graph=self._timeline.context_graph,
            context_objects=self._timeline.get_context_objects(),
            conversation=self._conversation,
            cwd=self._cwd,
        )

    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all tracked context objects (legacy compatibility)."""
        return self._timeline.get_context_objects()

    def get_context_graph(self) -> ContextGraph:
        """Get the context graph (DAG of context nodes)."""
        return self._timeline.context_graph

    @property
    def context_graph(self) -> ContextGraph:
        """The context graph (DAG of context nodes)."""
        return self._timeline.context_graph

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
        # Track config reload unregister callbacks per session
        self._reload_unregisters: dict[str, Callable[[], None]] = {}

    def set_default_llm(self, llm: LLMProvider | None) -> None:
        """Set the default LLM provider for new sessions."""
        self._default_llm = llm

    async def create_session(
        self,
        cwd: str,
        session_id: str | None = None,
        llm: LLMProvider | None = None,
        terminal_executor: TerminalExecutor | None = None,
        permission_requester: PermissionRequester | None = None,
        shell_permission_requester: ShellPermissionRequester | None = None,
    ) -> Session:
        """Create a new session with its own timeline.

        Args:
            cwd: Working directory for the session
            session_id: Optional specific ID; generated if not provided
            llm: Optional LLM provider; uses default if not provided
            terminal_executor: Optional terminal executor for shell commands;
                defaults to SubprocessTerminalExecutor if not provided
            permission_requester: Optional callback for ACP file permission prompts;
                async (session_id, path, mode) -> (granted, persist)
            shell_permission_requester: Optional callback for ACP shell permission prompts;
                async (session_id, command, args) -> (granted, persist)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        # Load project-level config (merges with system/user config)
        project_config = self._load_project_config(cwd)

        # Create permission manager from config
        sandbox_config = project_config.sandbox if project_config else None
        permission_manager = PermissionManager.from_config(cwd, sandbox_config)

        # Create shell permission manager from config
        shell_permission_manager = ShellPermissionManager.from_config(sandbox_config)

        timeline = Timeline(
            session_id,
            cwd=cwd,
            permission_manager=permission_manager,
            terminal_executor=terminal_executor,
            permission_requester=permission_requester,  # type: ignore[arg-type]
            shell_permission_manager=shell_permission_manager,
            shell_permission_requester=shell_permission_requester,  # type: ignore[arg-type]
        )
        session = Session(
            session_id=session_id,
            cwd=cwd,
            timeline=timeline,
            llm=llm or self._default_llm,
            config=project_config,
        )
        self._sessions[session_id] = session

        # Register for config reload to update permissions
        unregister = self._setup_permission_reload(
            session_id, cwd, permission_manager
        )
        self._reload_unregisters[session_id] = unregister

        # Initialize with example context view if guide exists
        await session._setup_initial_context()

        return session

    def _setup_permission_reload(
        self,
        session_id: str,
        cwd: str,
        permission_manager: PermissionManager,
    ) -> Callable[[], None]:
        """Set up config reload callback for a session's permission manager.

        Args:
            session_id: Session ID for tracking.
            cwd: Working directory for config loading.
            permission_manager: PermissionManager to update on reload.

        Returns:
            Unregister function to remove the callback.
        """
        try:
            from activecontext.config.loader import on_config_reload

            def on_reload(new_config: Config) -> None:
                # Only update if session still exists
                if session_id in self._sessions:
                    sandbox_config = new_config.sandbox if new_config else None
                    permission_manager.reload(sandbox_config)
                    log.debug(
                        "Permissions reloaded for session %s: %d rules",
                        session_id,
                        len(permission_manager.rules),
                    )

            return on_config_reload(on_reload)
        except ImportError:
            log.debug("Config reload not available")
            return lambda: None

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

            # Unregister config reload callback
            if session_id in self._reload_unregisters:
                unregister = self._reload_unregisters.pop(session_id)
                unregister()
