"""Agent: Script subclass with LLM integration.

An Agent extends Script to add LLM-powered conversation capabilities.
It manages message history, processes prompts through an LLM, and
executes code blocks from LLM responses.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import MessageNode, MessageRole
from activecontext.logging import get_logger
from activecontext.session.protocols import (
    SessionUpdate,
    TaskStatus,
    UpdateKind,
)
from activecontext.session.script import Script

log = get_logger("agent")

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from activecontext.config.schema import Config, MCPConfig
    from activecontext.context.dump import ContextDumpWriter
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import ContextNode, SessionNode
    from activecontext.coordination import ScratchpadManager
    from activecontext.core.llm.provider import LLMProvider, Message
    from activecontext.session.permissions import (
        ImportGuard,
        PermissionManager,
        ShellPermissionManager,
        WebsitePermissionManager,
    )
    from activecontext.session.protocols import Projection
    from activecontext.session.timeline import Timeline
    from activecontext.terminal.protocol import TerminalExecutor

    # Permission requester callback types
    PermissionRequester = Any
    ShellPermissionRequester = Any
    WebsitePermissionRequester = Any
    ImportPermissionRequester = Any


class Agent(Script):
    """Script with LLM integration and conversation.

    Agent extends Script to add:
    - LLM provider for generating responses
    - Message history for context
    - Prompt processing with code execution
    - Agent loop with wake conditions

    This is the primary task type for LLM-powered sessions.
    """

    def __init__(
        self,
        agent_id: str | None,
        context_graph: ContextGraph,
        cwd: str,
        llm: LLMProvider | None = None,
        timeline: Timeline | None = None,
        config: Config | None = None,
        permission_manager: PermissionManager | None = None,
        terminal_executor: TerminalExecutor | None = None,
        permission_requester: PermissionRequester | None = None,
        import_guard: ImportGuard | None = None,
        import_permission_requester: ImportPermissionRequester | None = None,
        shell_permission_manager: ShellPermissionManager | None = None,
        shell_permission_requester: ShellPermissionRequester | None = None,
        website_permission_manager: WebsitePermissionManager | None = None,
        website_permission_requester: WebsitePermissionRequester | None = None,
        scratchpad_manager: ScratchpadManager | None = None,
        mcp_config: MCPConfig | None = None,
    ) -> None:
        """Initialize the Agent.

        Args:
            agent_id: Unique identifier (generated if None)
            context_graph: The Session's shared context graph
            cwd: Working directory
            llm: LLM provider for generating responses
            timeline: Optional existing Timeline (if None, creates one)
            config: Configuration
            permission_manager: File permission manager
            terminal_executor: Shell command executor
            permission_requester: Callback for file permission prompts
            import_guard: Module import whitelist guard
            import_permission_requester: Callback for import permission prompts
            shell_permission_manager: Shell command permission manager
            shell_permission_requester: Callback for shell permission prompts
            website_permission_manager: Website access permission manager
            website_permission_requester: Callback for website permission prompts
            scratchpad_manager: Work coordination scratchpad
            mcp_config: MCP server configuration
        """
        super().__init__(
            script_id=agent_id,
            context_graph=context_graph,
            cwd=cwd,
            timeline=timeline,
            config=config,
            permission_manager=permission_manager,
            terminal_executor=terminal_executor,
            permission_requester=permission_requester,
            import_guard=import_guard,
            import_permission_requester=import_permission_requester,
            shell_permission_manager=shell_permission_manager,
            shell_permission_requester=shell_permission_requester,
            website_permission_manager=website_permission_manager,
            website_permission_requester=website_permission_requester,
            scratchpad_manager=scratchpad_manager,
            mcp_config=mcp_config,
            llm_provider=llm,
        )

        # LLM provider for generating responses
        self._llm = llm

        # Message history for context
        self._message_history: list[Message] = []

        # Cancellation flag
        self._cancelled = False

        # Wake event for agent loop
        self._wake_event = asyncio.Event()

        # Agent loop task
        self._running = False
        self._agent_task: asyncio.Task[Any] | None = None

        # Session node reference (set by Session)
        self._session_node: SessionNode | None = None

        # Tick callback (set by Session)
        self._tick_callback: Callable[[], Awaitable[list[SessionUpdate]]] | None = None

        # Projection callback (set by Session)
        self._get_projection_callback: Callable[[], Projection] | None = None

        # Add node callback (set by Session)
        self._add_node_callback: Callable[[ContextNode], str] | None = None

        # Context dump writer (set by Session)
        self._context_dump: ContextDumpWriter | None = None

    # -------------------------------------------------------------------------
    # TaskProtocol overrides
    # -------------------------------------------------------------------------

    @property
    def task_type(self) -> str:
        """Type of task."""
        return "agent"

    async def stop(self) -> None:
        """Stop the agent and clean up."""
        self._running = False
        self._cancelled = True
        self._wake_event.set()  # Wake up if waiting
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._agent_task
        await super().stop()

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent state for persistence."""
        base = super().to_dict()
        base["task_type"] = "agent"
        base["message_count"] = len(self._message_history)
        return base

    # -------------------------------------------------------------------------
    # LLM access
    # -------------------------------------------------------------------------

    @property
    def llm(self) -> LLMProvider | None:
        """Get the LLM provider."""
        return self._llm

    def set_llm(self, llm: LLMProvider | None) -> None:
        """Set the LLM provider."""
        self._llm = llm

    # -------------------------------------------------------------------------
    # Message history
    # -------------------------------------------------------------------------

    @property
    def message_history(self) -> list[Message]:
        """Get the message history."""
        return self._message_history

    def clear_message_history(self) -> None:
        """Clear the message history."""
        self._message_history.clear()

    def add_message(self, message: Message) -> MessageNode:
        """Add a message to conversation and context graph.

        Creates a MessageNode and adds it to the context graph.

        Args:
            message: The Message to add

        Returns:
            The created MessageNode
        """
        # Add to conversation list
        self._message_history.append(message)

        # Create MessageNode
        msg_node = MessageNode(
            role=MessageRole(message.role.value),  # Convert LLM Role enum to MessageRole
            content=message.content,
            originator=message.originator,
        )

        # Add via callback if available (Session handles group linking)
        if self._add_node_callback:
            self._add_node_callback(msg_node)
        else:
            # Direct add to graph
            self._context_graph.add_node(msg_node)

        return msg_node

    # -------------------------------------------------------------------------
    # Prompt processing
    # -------------------------------------------------------------------------

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
        if self._get_projection_callback:
            projection = self._get_projection_callback()
            yield SessionUpdate(
                kind=UpdateKind.PROJECTION_READY,
                session_id=self._script_id,
                payload={
                    "handles": projection.handles,
                },
                timestamp=time.time(),
            )

    async def _prompt_with_llm(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process prompt using the LLM provider.

        Runs an agent loop: LLM responds, code is executed, results feed back
        to LLM until it calls done() or produces no code blocks.
        """
        from activecontext.core.llm.provider import Message, Role
        from activecontext.core.prompts import parse_response

        # Reset done signal at start of each prompt
        self._timeline.reset_done()

        max_iterations = 10  # Safety limit
        iteration = 0

        # Add initial user message to context graph
        self.add_message(Message(role=Role.USER, content=content, originator="user"))

        while iteration < max_iterations:
            iteration += 1
            turn_start = time.time()

            # Build projection (renders visible nodes from context graph)
            if not self._get_projection_callback:
                log.warning("No projection callback set for agent")
                break

            projection = self._get_projection_callback()
            projection_content = projection.render()

            # Log projection size
            n_chars = len(projection_content or "")
            tokens_est = n_chars // 4
            log.debug(
                "Iteration %d, projection %d chars (~%d tokens)",
                iteration,
                n_chars,
                tokens_est,
            )

            # Write context dump file if configured
            if self._context_dump:
                self._context_dump.write(projection)

            # Send only the projection to the LLM (no system prompt)
            messages = [
                Message(role=Role.USER, content=projection_content or ""),
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
                        session_id=self._script_id,
                        payload={"text": chunk.text},
                        timestamp=time.time(),
                    )

            # Record response as metadata-only MessageNode
            msg_node = self.add_message(
                Message(role=Role.ASSISTANT, content=full_response, originator="agent")
            )

            # Parse and ingest segments via Timeline
            parsed = parse_response(full_response)
            segment_results = await self._timeline.ingest_segments(
                parsed.segments,
                source_message_id=msg_node.node_id,
            )

            had_executable = any(r.is_executable for r in segment_results)
            execution_results: list[str] = []

            # Yield execution updates and collect output for feedback
            for r in segment_results:
                if self._cancelled:
                    return
                if r.is_executable and r.execution_result is not None:
                    result = r.execution_result
                    yield SessionUpdate(
                        kind=UpdateKind.STATEMENT_EXECUTED,
                        session_id=self._script_id,
                        payload={
                            "execution_id": result.execution_id,
                            "statement_id": result.statement_id,
                            "status": result.status.value,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "exception": result.exception,
                            "state_trace": {
                                "added": result.state_trace.added,
                                "changed": result.state_trace.changed,
                                "deleted": result.state_trace.deleted,
                            },
                            "duration_ms": result.duration_ms,
                        },
                        timestamp=time.time(),
                    )
                    # Collect execution output for feedback
                    if result.stdout:
                        execution_results.append(f"Output:\n{result.stdout}")
                    if result.stderr:
                        execution_results.append(f"Stderr:\n{result.stderr}")
                    if result.exception:
                        execution_results.append(
                            f"Error: {result.exception.get('type')}: "
                            f"{result.exception.get('message')}"
                        )

            # Run tick phase
            if self._tick_callback:
                tick_updates = await self._tick_callback()
                for update in tick_updates:
                    yield update

            # Record turn statistics in SessionNode
            turn_duration_ms = (time.time() - turn_start) * 1000
            tokens_used = len(full_response) // 4  # Rough estimate
            action_desc = None
            if had_executable:
                exec_count = sum(1 for r in segment_results if r.is_executable)
                action_desc = f"Executed {exec_count} code block(s)"
            if self._session_node:
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
            if not had_executable:
                log.debug("No code blocks, stopping loop")
                break

            # Add execution results as a message (will appear in next projection)
            if execution_results:
                result_content = "Execution results:\n" + "\n".join(execution_results)
            else:
                result_content = "Code executed successfully."
            self.add_message(Message(role=Role.USER, content=result_content, originator="system"))

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
            if self._tick_callback:
                tick_updates = await self._tick_callback()
                for update in tick_updates:
                    yield update
        else:
            yield SessionUpdate(
                kind=UpdateKind.RESPONSE_CHUNK,
                session_id=self._script_id,
                payload={"text": f"[No LLM configured] Received: {content}"},
                timestamp=time.time(),
            )

    async def _execute_code(self, source: str) -> AsyncIterator[SessionUpdate]:
        """Execute a code block and yield updates."""
        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_PARSED,
            session_id=self._script_id,
            payload={"source": source},
            timestamp=time.time(),
        )

        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTING,
            session_id=self._script_id,
            payload={"source": source},
            timestamp=time.time(),
        )

        result = await self._timeline.execute_statement(source)

        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTED,
            session_id=self._script_id,
            payload={
                "execution_id": result.execution_id,
                "statement_id": result.statement_id,
                "status": result.status.value,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exception": result.exception,
                "state_trace": {
                    "added": result.state_trace.added,
                    "changed": result.state_trace.changed,
                    "deleted": result.state_trace.deleted,
                },
                "duration_ms": result.duration_ms,
            },
            timestamp=time.time(),
        )

    # -------------------------------------------------------------------------
    # Agent loop
    # -------------------------------------------------------------------------

    async def run_agent_loop(
        self,
        has_pending_work: Callable[[], bool] | None = None,
        process_next_message: Callable[[], AsyncIterator[SessionUpdate]] | None = None,
    ) -> AsyncIterator[SessionUpdate]:
        """Event-driven agent loop. Idle until wake, process until queue empty.

        This is the main processing loop for the async prompt model. It:
        1. Waits for a wake signal (message queued, file changed, etc.)
        2. Processes all pending messages
        3. Runs tick phase to handle async completions
        4. Yields SessionUpdate for each significant event

        Args:
            has_pending_work: Callback to check for pending work
            process_next_message: Callback to process next message

        Yields:
            SessionUpdate objects for streaming to the transport
        """
        self._running = True
        self._status = TaskStatus.RUNNING
        log.info("Agent loop started for %s", self._script_id)

        while self._running:
            # Wait for wake signal
            try:
                await self._wake_event.wait()
            except asyncio.CancelledError:
                log.info("Agent loop cancelled for %s", self._script_id)
                break

            self._wake_event.clear()
            log.debug("Agent loop woke for %s", self._script_id)

            # Process all pending work
            while has_pending_work and has_pending_work() and self._running:
                # Process next message
                if process_next_message:
                    async for update in process_next_message():
                        yield update

                # Run tick phase
                if self._tick_callback:
                    tick_updates = await self._tick_callback()
                    for update in tick_updates:
                        yield update

        log.info("Agent loop stopped for %s", self._script_id)

    def wake(self) -> None:
        """Wake the agent loop."""
        self._wake_event.set()

    def stop_loop(self) -> None:
        """Signal the agent loop to stop."""
        self._running = False
        self._wake_event.set()  # Wake up if waiting

    async def cancel(self) -> None:
        """Cancel the current operation."""
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Configuration (set by Session)
    # -------------------------------------------------------------------------

    def set_session_node(self, node: SessionNode) -> None:
        """Set the session node reference for recording statistics."""
        self._session_node = node

    def set_tick_callback(
        self, callback: Callable[[], Awaitable[list[SessionUpdate]]] | None
    ) -> None:
        """Set the tick callback."""
        self._tick_callback = callback

    def set_projection_callback(self, callback: Callable[[], Projection] | None) -> None:
        """Set the projection callback."""
        self._get_projection_callback = callback

    def set_add_node_callback(self, callback: Callable[[ContextNode], str] | None) -> None:
        """Set the add node callback."""
        self._add_node_callback = callback

    def set_context_dump(self, dump_writer: ContextDumpWriter | None) -> None:
        """Set the context dump writer."""
        self._context_dump = dump_writer
