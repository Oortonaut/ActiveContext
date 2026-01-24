"""Script: Base class for REPL and event loop execution.

A Script owns a Timeline and provides the execution environment for
Python statements. It is the base class for Agent (which adds LLM integration).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from activecontext.logging import get_logger
from activecontext.session.protocols import (
    ExecutionResult,
    IOMode,
    QueuedEvent,
    TaskProtocol,
    TaskStatus,
)
from activecontext.session.timeline import Timeline

log = get_logger("script")

if TYPE_CHECKING:
    from activecontext.config.schema import Config, MCPConfig
    from activecontext.context.graph import ContextGraph
    from activecontext.coordination import ScratchpadManager
    from activecontext.session.permissions import (
        ImportGuard,
        PermissionManager,
        ShellPermissionManager,
        WebsitePermissionManager,
    )
    from activecontext.terminal.protocol import TerminalExecutor

    # Permission requester callback types
    PermissionRequester = Any
    ShellPermissionRequester = Any
    WebsitePermissionRequester = Any
    ImportPermissionRequester = Any


class Script(TaskProtocol):
    """Base class providing REPL and event loop. Owns Timeline.

    Script is the execution unit within a Session. Each Script:
    - Owns a Timeline for Python statement execution
    - Shares the Session's ContextGraph
    - Can process events and run an event loop
    - Provides the namespace for DSL functions

    Use Agent (a Script subclass) for LLM-powered interactions.
    """

    def __init__(
        self,
        script_id: str | None,
        context_graph: ContextGraph,
        cwd: str,
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
        """Initialize the Script.

        Args:
            script_id: Unique identifier (generated if None)
            context_graph: The Session's shared context graph
            cwd: Working directory
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
        self._script_id = script_id or str(uuid.uuid4())
        self._context_graph = context_graph
        self._cwd = cwd
        self._config = config
        self._status = TaskStatus.PENDING

        # Event queue for async event processing
        self._event_queue: asyncio.Queue[QueuedEvent] = asyncio.Queue()

        # Create or use provided Timeline - Script owns this
        if timeline is not None:
            self._timeline = timeline
        else:
            self._timeline = Timeline(
                session_id=self._script_id,  # Timeline uses script_id as session_id
                context_graph=context_graph,
                cwd=cwd,
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
            )

    # -------------------------------------------------------------------------
    # TaskProtocol implementation
    # -------------------------------------------------------------------------

    @property
    def task_id(self) -> str:
        """Unique identifier for this script."""
        return self._script_id

    @property
    def task_type(self) -> str:
        """Type of task."""
        return "script"

    @property
    def io_mode(self) -> IOMode:
        """I/O mode - async for Script."""
        return IOMode.ASYNC

    @property
    def status(self) -> TaskStatus:
        """Current status."""
        return self._status

    async def start(self) -> None:
        """Start the script."""
        self._status = TaskStatus.RUNNING
        log.info("Script %s started", self._script_id)

    async def pause(self) -> None:
        """Pause the script."""
        self._status = TaskStatus.PAUSED
        log.info("Script %s paused", self._script_id)

    async def resume(self) -> None:
        """Resume the script."""
        if self._status == TaskStatus.PAUSED:
            self._status = TaskStatus.RUNNING
            log.info("Script %s resumed", self._script_id)

    async def stop(self) -> None:
        """Stop the script and clean up."""
        self._status = TaskStatus.DONE
        await self._timeline.close()
        log.info("Script %s stopped", self._script_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize script state for persistence."""
        return {
            "script_id": self._script_id,
            "task_type": self.task_type,
            "status": self._status.value,
            "cwd": self._cwd,
            "statements": [s.__dict__ for s in self._timeline.get_statements()],
        }

    # -------------------------------------------------------------------------
    # Timeline access (delegation)
    # -------------------------------------------------------------------------

    @property
    def timeline(self) -> Timeline:
        """Get the underlying Timeline."""
        return self._timeline

    @property
    def context_graph(self) -> ContextGraph:
        """Get the context graph (shared with Session)."""
        return self._context_graph

    @property
    def cwd(self) -> str:
        """Working directory."""
        return self._cwd

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(self, statement: str) -> ExecutionResult:
        """Execute a Python statement.

        Args:
            statement: Python source code to execute

        Returns:
            ExecutionResult with status and output
        """
        return await self._timeline.execute_statement(statement)

    def get_namespace(self) -> dict[str, Any]:
        """Get the current Python namespace."""
        return self._timeline.get_namespace()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all tracked context objects (ViewHandle, GroupHandle, etc.)."""
        return self._timeline.get_context_objects()

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    async def send_event(self, event: QueuedEvent) -> None:
        """Queue an event for processing.

        Args:
            event: Event to queue
        """
        await self._event_queue.put(event)

    async def receive_event(self, timeout: float | None = None) -> QueuedEvent | None:
        """Receive the next queued event.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            Next event, or None if timeout
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._event_queue.get(), timeout)
            return await self._event_queue.get()
        except asyncio.TimeoutError:
            return None

    def has_pending_events(self) -> bool:
        """Check if there are pending events."""
        return not self._event_queue.empty()

    # -------------------------------------------------------------------------
    # Event loop (override in subclasses)
    # -------------------------------------------------------------------------

    async def run_loop(self) -> None:
        """Run the event loop.

        Override in subclasses to implement custom loop behavior.
        The default implementation processes events from the queue.
        """
        self._status = TaskStatus.RUNNING

        while self._status == TaskStatus.RUNNING:
            event = await self.receive_event(timeout=1.0)
            if event is not None:
                await self._handle_event(event)

    async def _handle_event(self, event: QueuedEvent) -> None:
        """Handle a single event.

        Override in subclasses for custom event handling.

        Args:
            event: Event to handle
        """
        log.debug("Script %s handling event: %s", self._script_id, event.event_name)
        # Default: fire the event to Timeline's event system
        self._timeline.fire_event(event.event_name, event.data)

    # -------------------------------------------------------------------------
    # Configuration callbacks (set by Session)
    # -------------------------------------------------------------------------

    def set_title_callback(self, callback: Any) -> None:
        """Set the callback for title changes."""
        self._timeline.set_title_callback(callback)

    def set_path_resolver(self, resolver: Any) -> None:
        """Set the path resolver callback for @prompts/ etc."""
        self._timeline._path_resolver = resolver

    def set_delegate_conversation(self, callback: Any) -> None:
        """Set the conversation delegation callback."""
        self._timeline._delegate_conversation = callback

    def set_create_conversation_handle(self, callback: Any) -> None:
        """Set the conversation handle creation callback."""
        self._timeline._create_conversation_handle = callback

    def configure_file_watcher(self, file_watch_config: Any) -> None:
        """Configure the file watcher from config."""
        self._timeline.configure_file_watcher(file_watch_config)

    # -------------------------------------------------------------------------
    # Shell/MCP manager access
    # -------------------------------------------------------------------------

    def process_pending_shell_results(self) -> list[str]:
        """Process pending shell results. Returns list of updated node IDs."""
        return self._timeline.process_pending_shell_results()

    def process_file_changes(self) -> list[str]:
        """Process pending file changes. Returns wake prompts."""
        return self._timeline.process_file_changes()

    # -------------------------------------------------------------------------
    # Wait conditions
    # -------------------------------------------------------------------------

    def is_waiting(self) -> bool:
        """Check if there's an active wait condition."""
        return self._timeline.is_waiting()

    def get_wait_condition(self) -> Any:
        """Get the current wait condition."""
        return self._timeline.get_wait_condition()

    def clear_wait_condition(self) -> None:
        """Clear the current wait condition."""
        self._timeline.clear_wait_condition()

    def check_wait_condition(self) -> tuple[bool, str | None]:
        """Check if wait condition is satisfied.

        Returns:
            (satisfied, wake_prompt) tuple
        """
        return self._timeline.check_wait_condition()

    # -------------------------------------------------------------------------
    # Done signal
    # -------------------------------------------------------------------------

    def is_done(self) -> bool:
        """Check if done() was called."""
        return self._timeline.is_done()

    def get_done_message(self) -> str | None:
        """Get the done message if done() was called."""
        return self._timeline.get_done_message()

    def reset_done(self) -> None:
        """Reset the done flag."""
        self._timeline.reset_done()
