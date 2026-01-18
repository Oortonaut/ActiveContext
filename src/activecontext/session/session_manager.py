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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import GroupNode, MessageNode, SessionNode
from activecontext.context.state import TickFrequency
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine
from activecontext.logging import get_logger
from activecontext.session.permissions import ImportGuard, PermissionManager, ShellPermissionManager
from activecontext.session.protocols import (
    Projection,
    SessionUpdate,
    UpdateKind,
)
from activecontext.session.storage import (
    generate_default_title,
    load_session_data,
    save_session,
    get_session_path,
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

    # Type for website permission requester callback:
    # async (session_id, url, method) -> (granted, persist)
    WebsitePermissionRequester = Callable[
        [str, str, str], Coroutine[Any, Any, tuple[bool, bool]]
    ]

    # Type for import permission requester callback:
    # async (session_id, module) -> (granted, persist, include_submodules)
    ImportPermissionRequester = Callable[
        [str, str], Coroutine[Any, Any, tuple[bool, bool, bool]]
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
        title: str | None = None,
        created_at: datetime | None = None,
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
        self._show_message_actors = True  # Show actor info by default

        # Session persistence metadata
        self._title = title or generate_default_title()
        self._created_at = created_at or datetime.now()

        # Create SessionNode for agent situational awareness
        self._session_node = SessionNode(
            node_id="session",  # Fixed ID for easy lookup
            tokens=500,
            mode="running",
            tick_frequency=TickFrequency.turn(),
            session_start_time=self._created_at.timestamp(),
        )
        self._timeline.context_graph.add_node(self._session_node)

        # Track timing for turn statistics
        self._turn_start_time: float | None = None

        # Context stack for structural containment
        # When non-empty, new nodes are added as children of stack.top
        self._context_stack: list[str] = []

        # Register set_title callback with timeline
        self._timeline.set_title_callback(self.set_title)

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

    @property
    def title(self) -> str:
        """Session title for display in IDE menus."""
        return self._title

    def set_title(self, title: str) -> None:
        """Set the session title.

        Args:
            title: New title for the session.
        """
        self._title = title

    @property
    def created_at(self) -> datetime:
        """When the session was created."""
        return self._created_at

    # -------------------------------------------------------------------------
    # Context Stack Operations
    # -------------------------------------------------------------------------

    @property
    def current_group(self) -> str | None:
        """Get the current container group ID, or None for root level.

        When a group is on the stack, new nodes added via add_node()
        will automatically become children of that group.
        """
        return self._context_stack[-1] if self._context_stack else None

    def push_group(self, group_id: str) -> None:
        """Push a group onto the context stack.

        New nodes added via add_node() will become children of this group
        until pop_group() is called. Also syncs with Timeline so DSL-created
        nodes (view(), group(), etc.) are automatically linked.

        Args:
            group_id: The ID of the group to push.
        """
        self._context_stack.append(group_id)
        # Sync with Timeline for DSL node creation
        self._timeline.set_current_group(group_id)

    def pop_group(self) -> str | None:
        """Pop the current group from the context stack.

        Returns:
            The popped group ID, or None if stack was empty.
        """
        if not self._context_stack:
            return None

        popped = self._context_stack.pop()

        # Sync with Timeline: set to new top or clear
        new_top = self._context_stack[-1] if self._context_stack else None
        self._timeline.set_current_group(new_top)

        return popped

    def add_node(self, node: Any) -> str:
        """Add a node to the context graph, linking to current group if any.

        This is the primary method for adding context nodes. When a group
        is on the context stack, the node becomes a child of that group.

        Args:
            node: The ContextNode to add.

        Returns:
            The node's ID.
        """
        from activecontext.context.nodes import ContextNode

        if not isinstance(node, ContextNode):
            raise TypeError(f"Expected ContextNode, got {type(node).__name__}")

        self._timeline.context_graph.add_node(node)

        if self.current_group:
            self._timeline.context_graph.link(node.node_id, self.current_group)

        return node.node_id

    def begin_tool_use(self, tool_name: str, args: dict[str, Any] | None = None) -> str:
        """Begin a tool use scope by creating a group and pushing it to the stack.

        Creates a GroupNode for the tool call, adds a tool_call MessageNode
        as its first child, and pushes the group onto the context stack.
        Subsequent nodes added via add_node() will be children of this group.

        Args:
            tool_name: Name of the tool being invoked.
            args: Optional arguments passed to the tool.

        Returns:
            The group ID (can be used with end_tool_use).
        """
        # Create the tool group at current level
        group = GroupNode(summary_prompt=f"Tool: {tool_name}")
        self.add_node(group)  # Links to current_group if nested

        # Create tool_call message as child of the group
        tool_call = MessageNode(
            role="tool_call",
            content="",
            actor=f"tool:{tool_name}",
            tool_name=tool_name,
            tool_args=args or {},
        )
        self._timeline.context_graph.add_node(tool_call)
        self._timeline.context_graph.link(tool_call.node_id, group.node_id)

        # Push group to stack - subsequent add_node calls will be children
        self.push_group(group.node_id)

        return group.node_id

    def end_tool_use(self, summary: str | None = None) -> str | None:
        """End the current tool use scope.

        Pops the group from the context stack and optionally sets its summary.
        The summary becomes the collapsed representation of the tool use.

        Args:
            summary: Optional summary text for the group.

        Returns:
            The popped group ID, or None if no group was active.
        """
        group_id = self.pop_group()

        if group_id and summary:
            group = self._timeline.context_graph.get_node(group_id)
            if isinstance(group, GroupNode):
                group.cached_summary = summary

        return group_id

    def _add_message(self, message: Message) -> MessageNode:
        """Add a message to conversation and sync to context graph.

        Creates a MessageNode and adds it to the context graph via add_node(),
        which links to the current group if any. This enables:
        - ID-based referencing in the rendered context
        - Proper role alternation for LLM compatibility
        - Structural containment when inside a tool use scope

        Args:
            message: The Message to add

        Returns:
            The created MessageNode
        """
        # Add to conversation list
        self._conversation.append(message)

        # Create MessageNode
        msg_node = MessageNode(
            role=message.role.value,  # Convert Role enum to string
            content=message.content,
            actor=message.actor,
            tokens=500,  # Default token budget for messages
        )

        # Add via add_node() which handles current_group linking
        self.add_node(msg_node)

        return msg_node

    def save(self) -> Path:
        """Save the session to disk.

        Saves to $cwd/.ac/sessions/<session_id>.yaml

        Returns:
            Path to the saved session file.
        """
        # Get timeline statement sources
        timeline_sources = [stmt.source for stmt in self._timeline.get_statements()]

        return save_session(
            cwd=self._cwd,
            session_id=self._session_id,
            title=self._title,
            created_at=self._created_at,
            conversation=self._conversation,
            timeline_sources=timeline_sources,
            context_graph=self._timeline.context_graph,
        )

    @classmethod
    def from_file(
        cls,
        cwd: str,
        session_id: str,
        llm: LLMProvider | None = None,
        config: Config | None = None,
    ) -> Session | None:
        """Load a session from disk.

        Args:
            cwd: Project working directory.
            session_id: Session identifier.
            llm: Optional LLM provider.
            config: Optional config.

        Returns:
            Loaded Session, or None if file doesn't exist.
        """
        from activecontext.core.llm.provider import Message, Role

        session_path = get_session_path(cwd, session_id)
        data = load_session_data(session_path)
        if not data:
            return None

        # Reconstruct the context graph
        context_graph = ContextGraph.from_dict(data.context_graph)

        # Create timeline with the restored graph
        timeline = Timeline(
            session_id=session_id,
            cwd=cwd,
            context_graph=context_graph,
        )

        # Restore timeline statement history (for replay tracking)
        # Note: We don't re-execute, just record the sources
        for source in data.timeline:
            from activecontext.session.protocols import Statement
            stmt = Statement(
                statement_id=str(uuid.uuid4()),
                index=len(timeline._statements),
                source=source,
                timestamp=data.created_at.timestamp(),
            )
            timeline._statements.append(stmt)

        # Create session
        session = cls(
            session_id=session_id,
            cwd=cwd,
            timeline=timeline,
            llm=llm,
            config=config,
            title=data.title,
            created_at=data.created_at,
        )

        # Restore conversation
        for msg_data in data.conversation:
            role_str = msg_data.get("role", "user")
            role = Role.USER if role_str == "user" else Role.ASSISTANT
            msg = Message(
                role=role,
                content=msg_data.get("content", ""),
                actor=msg_data.get("actor"),
            )
            session._conversation.append(msg)

        # Populate timeline._context_objects from graph (for legacy compatibility)
        for node in context_graph:
            timeline._context_objects[node.node_id] = node

        # Link session's SessionNode to the restored graph's session node (if exists)
        restored_session_node = context_graph.get_node("session")
        if isinstance(restored_session_node, SessionNode):
            session._session_node = restored_session_node
        # If no session node was saved, one was already created in __init__

        return session

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
            turn_start = time.time()

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

            # Update conversation history (syncs to context graph)
            self._add_message(
                Message(role=Role.USER, content=current_request, actor="user")
            )
            self._add_message(
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

            # Record turn statistics in SessionNode
            turn_duration_ms = (time.time() - turn_start) * 1000
            tokens_used = len(full_response) // 4  # Rough estimate
            action_desc = None
            if code_blocks:
                action_desc = f"Executed {len(code_blocks)} code block(s)"
            self._session_node.record_turn(
                tokens_used=tokens_used,
                duration_ms=turn_duration_ms,
                action_description=action_desc,
            )

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
        1. Process pending shell results (async completions from background tasks)
        2. Turn ticks for running nodes (calls Recompute → notify_parents cascade)
        3. Periodic ticks based on tick_frequency
        4. Group summaries auto-invalidated via on_child_changed() during cascade
        5. Check wait conditions and prepare wake prompts
        """
        updates: list[SessionUpdate] = []
        timestamp = time.time()

        # 1. Process pending shell results from background tasks
        # This applies async results and triggers node change notifications
        updated_shell_nodes = self._timeline.process_pending_shell_results()
        for node_id in updated_shell_nodes:
            node = self._timeline.context_graph.get_node(node_id)
            if node:
                updates.append(
                    SessionUpdate(
                        kind=UpdateKind.NODE_CHANGED,
                        session_id=self._session_id,
                        payload={
                            "node_id": node_id,
                            "node_type": node.node_type,
                            "change": "shell_completed",
                            "digest": node.GetDigest(),
                        },
                        timestamp=timestamp,
                    )
                )

        # 2-4. Process running nodes with tick frequencies
        context_graph = self._timeline.context_graph
        running_nodes = context_graph.get_running_nodes()

        for node in running_nodes:
            tick_kind: str | None = None

            # Check tick frequency
            if node.tick_frequency is None:
                # No tick frequency set, skip
                continue

            if node.tick_frequency.mode == "turn":
                # Turn tick: recompute on every tick (replaces "Sync")
                tick_kind = "turn"
            elif node.tick_frequency.mode == "periodic":
                # Periodic tick: check interval
                if node.tick_frequency.interval is None:
                    log.warning("Periodic tick frequency without interval for node %s", node.node_id)
                    continue

                if timestamp - node.updated_at >= node.tick_frequency.interval:
                    tick_kind = "periodic"
            elif node.tick_frequency.mode == "async":
                # Async mode - not yet implemented
                pass
            elif node.tick_frequency.mode == "never":
                # Never tick
                continue

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

    def check_wait_condition(self) -> tuple[bool, str | None]:
        """Check if an active wait condition is satisfied.

        Returns:
            Tuple of (is_satisfied, prompt_to_inject).
        """
        return self._timeline.check_wait_condition()

    def is_waiting(self) -> bool:
        """Check if session is waiting for a condition."""
        return self._timeline.is_waiting()

    def clear_wait_condition(self) -> None:
        """Clear the active wait condition."""
        self._timeline.clear_wait_condition()

    async def cancel(self) -> None:
        """Cancel the current operation and all running shell commands."""
        self._cancelled = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        # Also cancel all running shell tasks
        self._timeline.cancel_all_shells()

    def get_projection(self) -> Projection:
        """Build the LLM projection from current session state."""
        return self._projection_engine.build(
            context_graph=self._timeline.context_graph,
            context_objects=self._timeline.get_context_objects(),
            conversation=self._conversation,
            cwd=self._cwd,
            show_message_actors=self._show_message_actors,
        )

    def show_message_ids(self, show: bool) -> None:
        """Control whether message actors are shown in conversation rendering.

        Args:
            show: If True, show actor info like "(user)", "(agent)". If False, hide actors.
        """
        self._show_message_actors = show

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

        Creates an example view of context_guide.md if it exists,
        demonstrating the view() DSL.
        """
        from pathlib import Path

        # Check for context guide in cwd first, then prompts directory
        guide_paths = [
            Path(self._cwd) / "CONTEXT_GUIDE.md",
            Path(__file__).parent.parent / "prompts" / "context_guide.md",
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

        # Auto-connect to configured MCP servers
        await self._setup_mcp_autoconnect()

    async def _setup_mcp_autoconnect(self) -> None:
        """Connect to MCP servers marked as auto_connect in config."""
        if not self._config or not self._config.mcp:
            return

        import logging

        _log = logging.getLogger("activecontext.session")

        for server_config in self._config.mcp.servers:
            if server_config.auto_connect:
                try:
                    source = f'mcp_connect("{server_config.name}")'
                    await self._timeline.execute_statement(source)
                    _log.info(f"Auto-connected to MCP server '{server_config.name}'")
                except Exception as e:
                    _log.warning(
                        f"Failed to auto-connect to MCP server '{server_config.name}': {e}"
                    )


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
        website_permission_requester: WebsitePermissionRequester | None = None,
        import_permission_requester: ImportPermissionRequester | None = None,
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
            website_permission_requester: Optional callback for ACP website permission prompts;
                async (session_id, url, method) -> (granted, persist)
            import_permission_requester: Optional callback for ACP import permission prompts;
                async (session_id, module) -> (granted, persist, include_submodules)
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

        # Create website permission manager from config
        from activecontext.session.permissions import WebsitePermissionManager

        website_permission_manager = WebsitePermissionManager.from_config(
            sandbox_config, Path(cwd)
        )

        # Create import guard from config
        import_config = sandbox_config.imports if sandbox_config else None
        import_guard = ImportGuard.from_config(import_config)

        # Create scratchpad manager for work coordination
        from activecontext.coordination import ScratchpadManager

        scratchpad_manager = ScratchpadManager(cwd)

        # Get MCP config if available
        mcp_config = project_config.mcp if project_config else None

        timeline = Timeline(
            session_id,
            cwd=cwd,
            permission_manager=permission_manager,
            terminal_executor=terminal_executor,
            permission_requester=permission_requester,  # type: ignore[arg-type]
            import_guard=import_guard,
            import_permission_requester=import_permission_requester,  # type: ignore[arg-type]
            shell_permission_manager=shell_permission_manager,
            shell_permission_requester=shell_permission_requester,  # type: ignore[arg-type]
            website_permission_manager=website_permission_manager,
            website_permission_requester=website_permission_requester,  # type: ignore[arg-type]
            scratchpad_manager=scratchpad_manager,
            mcp_config=mcp_config,
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
            session_id, cwd, permission_manager, shell_permission_manager, website_permission_manager, import_guard
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
        shell_permission_manager: ShellPermissionManager | None = None,
        website_permission_manager: Any | None = None,
        import_guard: ImportGuard | None = None,
    ) -> Callable[[], None]:
        """Set up config reload callback for a session's permission managers.

        Args:
            session_id: Session ID for tracking.
            cwd: Working directory for config loading.
            permission_manager: PermissionManager to update on reload.
            shell_permission_manager: Optional ShellPermissionManager to update on reload.
            website_permission_manager: Optional WebsitePermissionManager to update on reload.
            import_guard: Optional ImportGuard to update on reload.

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
                    if shell_permission_manager:
                        shell_permission_manager.reload(sandbox_config)
                        log.debug(
                            "Shell permissions reloaded for session %s: %d rules",
                            session_id,
                            len(shell_permission_manager.rules),
                        )
                    if website_permission_manager:
                        website_permission_manager.reload(sandbox_config)
                        log.debug(
                            "Website permissions reloaded for session %s: %d rules",
                            session_id,
                            len(website_permission_manager.rules),
                        )
                    if import_guard:
                        import_config = sandbox_config.imports if sandbox_config else None
                        import_guard.reload(import_config)
                        log.debug(
                            "Import whitelist reloaded for session %s: %d modules",
                            session_id,
                            len(import_guard.allowed_modules),
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

            # Unregister from work coordination scratchpad
            if session._timeline._scratchpad_manager:
                session._timeline._scratchpad_manager.unregister()

            await session.cancel()

            # Unregister config reload callback
            if session_id in self._reload_unregisters:
                unregister = self._reload_unregisters.pop(session_id)
                unregister()
