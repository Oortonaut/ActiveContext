"""Timeline: wrapper around StatementLog and PythonExec.

The Timeline is the canonical history of executed Python statements
for a session. It manages:
- Statement recording and indexing
- Python namespace execution
- Replay/re-execution from any point
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    ArtifactNode,
    ContextNode,
    GroupNode,
    LockNode,
    LockStatus,
    MCPServerNode,
    ShellNode,
    ShellStatus,
    TextNode,
    TopicNode,
    WorkNode,
)
from activecontext.context.state import NodeState, NotificationLevel, TickFrequency
from activecontext.mcp import (
    MCPClientManager,
)
from activecontext.session.permissions import (
    ImportDenied,
    ImportGuard,
    PermissionDenied,
    PermissionManager,
    ShellPermissionManager,
    WebsitePermissionDenied,
    WebsitePermissionManager,
    make_safe_fetch,
    make_safe_import,
    make_safe_open,
    write_import_to_config,
    write_permission_to_config,
    write_shell_permission_to_config,
    write_website_permission_to_config,
)
from activecontext.session.protocols import (
    EventHandler,
    EventResponse,
    ExecutionResult,
    ExecutionStatus,
    NamespaceTrace,
    QueuedEvent,
    Statement,
    WaitCondition,
    WaitMode,
)
from activecontext.session.xml_parser import is_xml_command, parse_xml_to_python
from activecontext.terminal.result import ShellResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from activecontext.agents.manager import AgentManager
    from activecontext.config.schema import FileWatchConfig, MCPConfig, MCPServerConfig
    from activecontext.coordination.scratchpad import ScratchpadManager
    from activecontext.terminal.protocol import TerminalExecutor
    from activecontext.watching import FileChangeEvent

    # Type for file permission requester callback:
    # async (session_id, path, mode) -> (granted, persist)
    PermissionRequester = Callable[[str, str, str], "asyncio.Future[tuple[bool, bool]]"]

    # Type for shell permission requester callback:
    # async (session_id, command, args) -> (granted, persist)
    ShellPermissionRequester = Callable[
        [str, str, list[str] | None], "asyncio.Future[tuple[bool, bool]]"
    ]

    # Type for website permission requester callback:
    # async (session_id, url, method) -> (granted, persist)
    WebsitePermissionRequester = Callable[
        [str, str, str], "asyncio.Future[tuple[bool, bool]]"
    ]

    # Type for import permission requester callback:
    # async (session_id, module) -> (granted, persist, include_submodules)
    ImportPermissionRequester = Callable[
        [str, str], "asyncio.Future[tuple[bool, bool, bool]]"
    ]


@dataclass
class _ExecutionRecord:
    """Internal record of a statement execution."""

    execution_id: str
    statement_id: str
    started_at: float
    ended_at: float
    status: ExecutionStatus
    stdout: str
    stderr: str
    exception: dict[str, Any] | None
    state_trace: NamespaceTrace



class LazyNodeNamespace(dict):
    """Dict subclass that falls back to graph lookup for node display IDs.

    Allows direct access to nodes by display_id (e.g., text_1, group_2)
    in the DSL namespace without explicit assignment.

    User-defined variables take precedence over node lookups.
    """

    def __init__(
        self,
        graph_getter: Callable[[], ContextGraph | None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._graph_getter = graph_getter

    def __getitem__(self, key: str) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            graph = self._graph_getter()
            if graph:
                node = graph.get_node_by_display_id(key)
                if node:
                    return node
            raise


class Timeline:
    """Statement timeline with controlled Python execution.

    Each session has one Timeline that tracks all executed statements
    and maintains the Python namespace.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str = ".",
        context_graph: ContextGraph | None = None,
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
        self._session_id = session_id
        self._cwd = cwd
        self._statements: list[Statement] = []
        self._executions: dict[str, list[_ExecutionRecord]] = {}  # statement_id -> executions

        # Context graph (DAG of context nodes)
        self._context_graph = context_graph or ContextGraph()

        # Permission manager for file access control
        self._permission_manager = permission_manager

        # Import guard for module whitelist control
        self._import_guard = import_guard

        # Import permission requester callback for ACP permission prompts
        # Called when ImportDenied is raised: async (sid, module) -> (granted, persist, include_submodules)
        self._import_permission_requester = import_permission_requester

        # Permission requester callback for ACP permission prompts
        # Called when PermissionDenied is raised: async (sid, path, mode) -> (granted, persist)
        self._permission_requester = permission_requester

        # Shell permission manager for command access control
        self._shell_permission_manager = shell_permission_manager

        # Shell permission requester callback for ACP permission prompts
        # Called when shell command is denied: async (sid, cmd, args) -> (granted, persist)
        self._shell_permission_requester = shell_permission_requester

        # Website permission manager for HTTP/HTTPS access control
        self._website_permission_manager = website_permission_manager

        # Website permission requester callback for ACP permission prompts
        # Called when website access is denied: async (sid, url, method) -> (granted, persist)
        self._website_permission_requester = website_permission_requester

        # Terminal executor for shell commands (default to subprocess)
        if terminal_executor is None:
            from activecontext.terminal.subprocess_executor import (
                SubprocessTerminalExecutor,
            )

            terminal_executor = SubprocessTerminalExecutor(default_cwd=cwd)
        self._terminal_executor = terminal_executor

        # Work coordination scratchpad manager (must be set before _setup_namespace)
        self._scratchpad_manager = scratchpad_manager

        # WorkNode for displaying coordination status (created on first work_on)
        self._work_node: WorkNode | None = None

        # MCP client manager for external tool servers (must be set before _setup_namespace)
        self._mcp_client_manager = MCPClientManager(config=mcp_config)

        # MCPServerNodes keyed by server name (for tracking in graph)
        self._mcp_server_nodes: dict[str, MCPServerNode] = {}

        # Controlled Python namespace
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()



        # Max output capture per statement
        self._max_stdout = 50000
        self._max_stderr = 10000

        # Done signal from agent
        self._done_called = False
        self._done_message: str | None = None

        # Callback for setting session title (set by Session after creation)
        self._set_title_callback: Callable[[str], None] | None = None

        # Pending async shell results (node_id, result) - populated by background tasks
        # Processed at tick time to trigger node updates and notifications
        self._pending_shell_results: list[tuple[str, ShellResult]] = []

        # Background tasks for shell execution
        self._shell_tasks: dict[str, asyncio.Task[None]] = {}

        # Pending async lock results (node_id, status, error_msg) - populated by background tasks
        self._pending_lock_results: list[tuple[str, LockStatus, str | None]] = []

        # Background tasks for lock acquisition
        self._lock_tasks: dict[str, asyncio.Task[None]] = {}

        # Active file locks (node_id -> file handle for releasing)
        self._active_locks: dict[str, Any] = {}

        # Active wait condition (blocks turn until satisfied)
        self._wait_condition: WaitCondition | None = None

        # Current group for automatic node linking (set by Session for tool scoping)
        self._current_group_id: str | None = None

        # Agent manager for multi-agent support (set by AgentManager after spawn)
        self._agent_manager: AgentManager | None = None
        self._agent_id: str | None = None

        # Event handling system
        self._event_handlers: dict[str, EventHandler] = {}
        self._queued_events: list[QueuedEvent] = []
        self._setup_default_event_handlers()

        # Text buffer storage - Session replaces this with its own dict
        # For standalone Timeline usage, this provides a default
        self._text_buffers: dict[str, TextBuffer] = {}

        # File watcher for detecting external file changes
        from activecontext.watching import FileWatcher
        self._file_watcher = FileWatcher(
            cwd=Path(cwd),
            poll_interval=1.0,
        )
        # Callback set by Session when agent loop starts
        self._on_file_changed: Callable[[FileChangeEvent], None] | None = None

    def configure_file_watcher(self, config: FileWatchConfig | None) -> None:
        """Configure the file watcher from config.

        Args:
            config: FileWatchConfig from session config, or None to use defaults
        """
        if config is None:
            return

        if not config.enabled:
            # Disable file watching - replace with a no-op watcher
            from activecontext.watching import FileWatcher
            self._file_watcher = FileWatcher(cwd=Path(self._cwd), poll_interval=float('inf'))
            return

        self._file_watcher.poll_interval = config.poll_interval

    def set_title_callback(self, callback: Callable[[str], None] | None) -> None:
        """Set the callback for set_title() DSL function.

        Args:
            callback: Function to call with new title, or None to disable.
        """
        self._set_title_callback = callback
        # Re-setup namespace to include/exclude set_title
        self._setup_namespace()

    @property
    def cwd(self) -> str:
        return self._cwd

    def _setup_namespace(self) -> None:
        """Initialize the Python namespace with injected functions."""
        # Import builtins we want to expose
        import builtins

        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if not name.startswith("_")
        }

        # Wrap open() with permission checks if permission_manager is provided
        if self._permission_manager:
            safe_builtins["open"] = make_safe_open(self._permission_manager)

        # Wrap __import__ with whitelist checks if import_guard is provided
        if self._import_guard:
            safe_builtins["__import__"] = make_safe_import(self._import_guard)
        else:
            # Expose default __import__ for imports to work
            safe_builtins["__import__"] = builtins.__import__

        self._namespace = LazyNodeNamespace(lambda: self._context_graph, {
            "__builtins__": safe_builtins,
            "__name__": "__activecontext__",
            "__session_id__": self._session_id,
            # Type enums for LLM use
            "NodeState": NodeState,
            "TickFrequency": TickFrequency,
            "NotificationLevel": NotificationLevel,
            # Context node constructors
            "text": self._make_text_node,
            "group": self._make_group_node,
            "topic": self._make_topic_node,
            "artifact": self._make_artifact_node,
            "markdown": self._make_markdown_node,
            "view": self._make_view,
            # DAG manipulation
            "link": self._link,
            "unlink": self._unlink,
            # Checkpointing
            "checkpoint": self._checkpoint,
            "restore": self._restore,
            "checkpoints": self._list_checkpoints,
            "branch": self._branch,
            # Utility functions
            "ls": self._ls_handles,
            "show": self._show_handle,
            "ls_permissions": self._ls_permissions,
            "ls_imports": self._ls_imports,
            "ls_shell_permissions": self._ls_shell_permissions,
            "ls_website_permissions": self._ls_website_permissions,
            # Shell execution
            "shell": self._shell,
            # HTTP/HTTPS requests
            "fetch": self._fetch,
            # Agent control
            "done": self._done,
            # Session title
            "set_title": self._set_title,
            # Notification control
            "notify": self._set_notify,
            # Async wait control
            "wait": self._wait,
            "wait_all": self._wait_all,
            "wait_any": self._wait_any,
            # File locking
            "lock_file": self._lock_file,
            "lock_release": self._lock_release,
        })

        # Add work coordination functions if scratchpad manager is available
        if self._scratchpad_manager:
            self._namespace.update({
                "work_on": self._work_on,
                "work_check": self._work_check,
                "work_update": self._work_update,
                "work_done": self._work_done,
                "work_list": self._work_list,
            })

        # Add MCP functions
        self._namespace.update({
            "mcp_connect": self._mcp_connect,
            "mcp_disconnect": self._mcp_disconnect,
            "mcp_list": self._mcp_list,
            "mcp_tools": self._mcp_tools,
        })

        # Inject connected MCP server proxies into namespace
        self._namespace.update(self._mcp_client_manager.generate_namespace_bindings())

    def _setup_agent_namespace(self) -> None:
        """Add agent functions to namespace when agent manager is available.

        Called by AgentManager after setting _agent_manager.
        """
        if self._agent_manager is None:
            return

        self._namespace.update({
            "spawn": self._spawn_agent,
            "send": self._send_message,
            "send_update": self._send_update,
            "recv": self._recv_messages,
            "wait_message": self._wait_message,
            "agents": self._list_agents,
            "agent_status": self._get_agent_status,
            "pause_agent": self._pause_agent,
            "resume_agent": self._resume_agent,
            "terminate_agent": self._terminate_agent,
            "get_shared_node": self._get_shared_node,
            # Event system
            "event_response": self._event_response,
            "wait": self._wait_event,
            "EventResponse": EventResponse,
            # File watching
            "wait_file_change": self._wait_file_change,
            "on_file_change": self._on_file_change,
        })

    def _setup_default_event_handlers(self) -> None:
        """Set up default event handlers for built-in events."""
        # Default: queue messages (don't wake unless explicitly waiting)
        self._event_handlers["message"] = EventHandler(
            event_name="message",
            response=EventResponse.QUEUE,
            prompt_template="Message from {sender}: {content}",
        )
        # Default: queue agent_done events
        self._event_handlers["agent_done"] = EventHandler(
            event_name="agent_done",
            response=EventResponse.QUEUE,
            prompt_template="Agent {agent_id} completed with state: {state}",
        )
        # Default: queue tick events (rarely used as wake trigger)
        self._event_handlers["tick"] = EventHandler(
            event_name="tick",
            response=EventResponse.QUEUE,
            prompt_template="Tick occurred",
        )
        # Default: queue file change events
        self._event_handlers["file_changed"] = EventHandler(
            event_name="file_changed",
            response=EventResponse.QUEUE,
            prompt_template="File changed: {path}",
        )
        # Default: wake on MCP async results (agent likely waiting)
        self._event_handlers["mcp_result"] = EventHandler(
            event_name="mcp_result",
            response=EventResponse.WAKE,
            prompt_template="MCP {tool_name} completed",
        )

    def _event_response(
        self,
        event_name: str,
        response: EventResponse,
        prompt: str = "",
    ) -> None:
        """Set the response type for an event.

        DSL function: event_response(name, response, prompt)

        Args:
            event_name: Event name ("message", "agent_done", "tick", or custom)
            response: EventResponse.WAKE or EventResponse.QUEUE
            prompt: Template for wake prompt (can use {placeholders})

        Example:
            event_response("message", EventResponse.WAKE, "Message: {content}")
        """
        self._event_handlers[event_name] = EventHandler(
            event_name=event_name,
            response=response,
            prompt_template=prompt or f"Event: {event_name}",
        )

    def _wait_file_change(
        self,
        paths: list[str] | str,
        wake_prompt: str = "File(s) changed: {paths}",
        timeout: float | None = None,
    ) -> None:
        """Wait for specific file changes.

        DSL function: wait_file_change(paths, wake_prompt, timeout)

        Sets up a one-time WAKE handler for file changes on the specified paths.
        The agent will be woken when any of the files change.

        Args:
            paths: Path(s) to watch (relative to session cwd)
            wake_prompt: Prompt template for wake (supports {paths} placeholder)
            timeout: Optional timeout in seconds

        Example:
            wait_file_change("src/main.py", wake_prompt="main.py was modified!")
            wait_file_change(["src/*.py"], timeout=60.0)
        """
        if isinstance(paths, str):
            paths = [paths]

        # Register a one-time WAKE handler for file_changed events
        # The handler will match any of the specified paths
        for path in paths:
            handler_key = f"file_changed:{path}"
            self._event_handlers[handler_key] = EventHandler(
                event_name="file_changed",
                response=EventResponse.WAKE,
                prompt_template=wake_prompt,
                once=True,
            )

        # Signal that we're done for this turn (waiting)
        self._done_called = True

    def _on_file_change(
        self,
        response: str = "queue",
        prompt: str = "File changed: {path}",
    ) -> None:
        """Configure global file change response.

        DSL function: on_file_change(response, prompt)

        Sets the default response for file_changed events.

        Args:
            response: "wake" to interrupt agent, "queue" to batch
            prompt: Wake prompt template (supports {path}, {change_type})

        Example:
            on_file_change(response="wake", prompt="Code changed: {path}")
            on_file_change(response="queue")  # Default behavior
        """
        self._event_handlers["file_changed"] = EventHandler(
            event_name="file_changed",
            response=EventResponse.WAKE if response == "wake" else EventResponse.QUEUE,
            prompt_template=prompt,
        )

    def _wait_event(
        self,
        target: Any,
        event_name: str,
        prompt: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Wait for a specific event from a target.

        DSL function: wait(target, event_name, prompt=None, timeout=None)

        Sets a one-time WAKE handler for the specified event.

        Args:
            target: AgentHandle, agent_id string, or None for global events
            event_name: Event to wait for ("agent_done", "message", etc.)
            prompt: Optional wake prompt override
            timeout: Optional timeout in seconds

        Example:
            wait(child_agent, "agent_done")  # Wake when child completes
        """
        from activecontext.agents.handle import AgentHandle

        # Resolve target_id
        target_id: str | None = None
        if isinstance(target, AgentHandle):
            target_id = target.agent_id
        elif isinstance(target, str):
            target_id = target

        # Create one-time wake handler
        handler_key = f"{event_name}:{target_id}" if target_id else event_name
        default_prompt = self._event_handlers.get(event_name, EventHandler(
            event_name=event_name,
            response=EventResponse.QUEUE,
            prompt_template=f"Event {event_name} occurred",
        )).prompt_template

        self._event_handlers[handler_key] = EventHandler(
            event_name=event_name,
            response=EventResponse.WAKE,
            prompt_template=prompt or default_prompt,
            once=True,
            target_id=target_id,
        )

        # Set up wait condition
        self._wait_condition = WaitCondition(
            node_ids=[],
            mode=WaitMode.AGENT if event_name == "agent_done" else WaitMode.MESSAGE,
            wake_prompt=prompt or default_prompt,
            timeout=timeout,
            timeout_prompt=f"Timed out waiting for {event_name}" if timeout else None,
            agent_id=target_id,
        )
        self._done_called = True  # End turn to wait

    def fire_event(self, event_name: str, data: dict[str, Any]) -> str | None:
        """Fire an event and return wake prompt if handler says WAKE.

        Called internally when events occur. Returns prompt if agent should wake.

        Args:
            event_name: Event name
            data: Event data (used for prompt template formatting)

        Returns:
            Wake prompt if WAKE response, None if QUEUE
        """
        # Check for targeted handler first (e.g., "agent_done:abc123")
        target_id = data.get("agent_id") or data.get("sender")
        handler_key = f"{event_name}:{target_id}" if target_id else None

        handler = None
        if handler_key and handler_key in self._event_handlers:
            handler = self._event_handlers[handler_key]
        elif event_name in self._event_handlers:
            handler = self._event_handlers[event_name]

        if not handler:
            # No handler, queue by default
            self._queued_events.append(QueuedEvent(event_name=event_name, data=data))
            return None

        if handler.response == EventResponse.WAKE:
            # Format wake prompt
            try:
                prompt = handler.prompt_template.format(**data)
            except KeyError:
                prompt = handler.prompt_template

            # Remove if one-time handler
            if handler.once:
                if handler_key and handler_key in self._event_handlers:
                    del self._event_handlers[handler_key]
                elif event_name in self._event_handlers and self._event_handlers[event_name].once:
                    del self._event_handlers[event_name]

            return prompt
        else:
            # Queue the event
            self._queued_events.append(QueuedEvent(event_name=event_name, data=data))
            return None

    def process_file_changes(self) -> list[str]:
        """Process pending file changes from the file watcher.

        Checks for file changes, fires events for each, and returns
        any wake prompts that should be processed.

        Returns:
            List of wake prompts (empty if no WAKE handlers triggered)
        """
        wake_prompts: list[str] = []

        for event in self._file_watcher.check_changes():
            # Fire the file_changed event
            wake_prompt = self.fire_event("file_changed", event.to_dict())
            if wake_prompt:
                wake_prompts.append(wake_prompt)

            # Update affected TextNodes
            for node_id in event.node_ids:
                node = self._context_graph.get_node(node_id)
                if isinstance(node, TextNode):
                    # Mark node as needing re-render
                    node._mark_changed(
                        description=f"File '{event.path.name}' {event.change_type}",
                    )

            # Call the session's file change callback if set
            if self._on_file_changed:
                self._on_file_changed(event)

        return wake_prompts

    def get_queued_events(self, event_name: str | None = None) -> list[QueuedEvent]:
        """Get queued events, optionally filtered by name.

        Args:
            event_name: Filter by event name, or None for all

        Returns:
            List of queued events
        """
        if event_name is None:
            return list(self._queued_events)
        return [e for e in self._queued_events if e.event_name == event_name]

    def has_pending_wake_prompt(self) -> bool:
        """Check if there's a pending wake prompt from the wait system.

        Returns:
            True if check_wait_condition would return a wake prompt
        """
        if self._wait_condition is None:
            return False

        # Check if wait condition is satisfied (without consuming it)
        satisfied, _ = self.check_wait_condition()
        return satisfied

    def clear_queued_events(self, event_name: str | None = None) -> int:
        """Clear queued events, optionally filtered by name.

        Args:
            event_name: Clear only this event type, or None for all

        Returns:
            Number of events cleared
        """
        if event_name is None:
            count = len(self._queued_events)
            self._queued_events.clear()
            return count

        original_count = len(self._queued_events)
        self._queued_events = [e for e in self._queued_events if e.event_name != event_name]
        return original_count - len(self._queued_events)

    def _ls_permissions(self) -> list[dict[str, Any]]:
        """List current file permissions (read-only inspection).

        Returns:
            List of permission rules with pattern, mode, and source.
        """
        if self._permission_manager:
            return self._permission_manager.list_permissions()
        return []

    def _ls_imports(self) -> dict[str, Any]:
        """List import whitelist configuration (read-only inspection).

        Returns:
            Dict with allowed_modules list, allow_submodules, and allow_all flags.
        """
        if self._import_guard:
            return {
                "allowed_modules": self._import_guard.list_allowed(),
                "allow_submodules": self._import_guard.allow_submodules,
                "allow_all": self._import_guard.allow_all,
            }
        return {
            "allowed_modules": [],
            "allow_submodules": True,
            "allow_all": True,  # No guard means unrestricted
        }

    def _ls_shell_permissions(self) -> dict[str, Any]:
        """List shell permission configuration (read-only inspection).

        Returns:
            Dict with shell permission rules and deny_by_default flag.
        """
        if self._shell_permission_manager:
            return {
                "rules": self._shell_permission_manager.list_permissions(),
                "deny_by_default": self._shell_permission_manager.deny_by_default,
            }
        return {
            "rules": [],
            "deny_by_default": True,  # No manager means default deny
        }

    def _ls_website_permissions(self) -> dict[str, Any]:
        """List website permission configuration (read-only inspection).

        Returns:
            Dict with website permission rules, deny_by_default flag, and allow_localhost.
        """
        if self._website_permission_manager:
            return {
                "rules": self._website_permission_manager.list_permissions(),
                "deny_by_default": self._website_permission_manager.deny_by_default,
                "allow_localhost": self._website_permission_manager.allow_localhost,
            }
        return {
            "rules": [],
            "deny_by_default": True,  # No manager means default deny
            "allow_localhost": False,
        }

    def _make_text_node(
        self,
        path: str,
        *,
        pos: str = "1:0",
        tokens: int = 2000,
        state: NodeState = NodeState.ALL,
        mode: str = "paused",
        parent: ContextNode | str | None = None,
    ) -> TextNode:
        """Create a TextNode and add to the context graph.

        Args:
            path: File path relative to session cwd
            pos: Start position as "line:col" (1-indexed)
            tokens: Token budget for rendering
            state: Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL)
            mode: "paused" or "running"
            parent: Optional parent node or node ID (defaults to current_group if set)

        Returns:
            The created TextNode
        """
        node = TextNode(
            path=path,
            pos=pos,
            tokens=tokens,
            state=state,
            mode=mode,
        )

        # Add to graph
        self._context_graph.add_node(node)

        # Determine parent: explicit > current_group > none
        effective_parent = parent
        if effective_parent is None and self._current_group_id:
            effective_parent = self._current_group_id

        # Link to parent if set
        if effective_parent:
            parent_id = effective_parent.node_id if isinstance(effective_parent, ContextNode) else effective_parent
            self._context_graph.link(node.node_id, parent_id)

        # Register with file watcher for external change detection
        self._file_watcher.register_path(path, node.node_id)

        return node

    def _make_group_node(
        self,
        *members: ContextNode | str,
        tokens: int = 500,
        state: NodeState = NodeState.SUMMARY,
        mode: str = "paused",
        summary: str | None = None,
        parent: ContextNode | str | None = None,
    ) -> GroupNode:
        """Create a GroupNode that summarizes its members.

        Args:
            *members: Child nodes or node IDs to include in the group
            tokens: Token budget for summary
            state: Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL)
            mode: "paused" or "running"
            summary: Optional pre-computed summary text
            parent: Optional parent node or node ID (defaults to current_group if set)

        Returns:
            The created GroupNode
        """
        node = GroupNode(
            tokens=tokens,
            state=state,
            mode=mode,
            cached_summary=summary,
            summary_stale=summary is None,  # Not stale if summary provided
        )

        # Add to graph
        self._context_graph.add_node(node)

        # Determine parent: explicit > current_group > none
        effective_parent = parent
        if effective_parent is None and self._current_group_id:
            effective_parent = self._current_group_id

        # Link to parent if set
        if effective_parent:
            parent_id = effective_parent.node_id if isinstance(effective_parent, ContextNode) else effective_parent
            self._context_graph.link(node.node_id, parent_id)

        # Link members as children of this group
        for member in members:
            if isinstance(member, ContextNode):
                member_id = member.node_id
            else:
                # member is already a node ID string
                member_id = member
            
            self._context_graph.link(member_id, node.node_id)

        return node

    def _make_topic_node(
        self,
        title: str,
        *,
        tokens: int = 1000,
        status: str = "active",
        parent: ContextNode | str | None = None,
    ) -> TopicNode:
        """Create a TopicNode for conversation segmentation.

        Args:
            title: Short title for the topic
            tokens: Token budget for rendering
            status: "active", "resolved", or "deferred"
            parent: Optional parent node or node ID (defaults to current_group if set)

        Returns:
            The created TopicNode
        """
        node = TopicNode(
            title=title,
            tokens=tokens,
            status=status,
        )

        # Add to graph
        self._context_graph.add_node(node)

        # Determine parent: explicit > current_group > none
        effective_parent = parent
        if effective_parent is None and self._current_group_id:
            effective_parent = self._current_group_id

        # Link to parent if set
        if effective_parent:
            parent_id = effective_parent.node_id if isinstance(effective_parent, ContextNode) else effective_parent
            self._context_graph.link(node.node_id, parent_id)

        return node

    def _make_artifact_node(
        self,
        artifact_type: str = "code",
        *,
        content: str = "",
        language: str | None = None,
        tokens: int = 500,
        parent: ContextNode | str | None = None,
    ) -> ArtifactNode:
        """Create an ArtifactNode for code/output.

        Args:
            artifact_type: "code", "output", "error", or "file"
            content: The artifact content
            language: Programming language (for code)
            tokens: Token budget
            parent: Optional parent node or node ID (defaults to current_group if set)

        Returns:
            The created ArtifactNode
        """
        node = ArtifactNode(
            artifact_type=artifact_type,
            content=content,
            language=language,
            tokens=tokens,
        )

        # Add to graph
        self._context_graph.add_node(node)

        # Determine parent: explicit > current_group > none
        effective_parent = parent
        if effective_parent is None and self._current_group_id:
            effective_parent = self._current_group_id

        # Link to parent if set
        if effective_parent:
            parent_id = effective_parent.node_id if isinstance(effective_parent, ContextNode) else effective_parent
            self._context_graph.link(node.node_id, parent_id)

        return node

    def _make_markdown_node(
        self,
        path: str,
        *,
        content: str | None = None,
        tokens: int = 2000,
        state: NodeState = NodeState.DETAILS,
        parent: ContextNode | str | None = None,
    ) -> TextNode:
        """Create a tree of TextNodes from a markdown file.

        Parses the markdown heading hierarchy into a tree of TextNodes,
        where each heading section is a separate node with line range references.

        Args:
            path: File path relative to session cwd
            content: Markdown content (if None, reads from path)
            tokens: Token budget for root node
            state: Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL)
            parent: Optional parent node or node ID (defaults to current_group if set)

        Returns:
            The root TextNode (children are accessible via children_ids)
        """
        import os

        from activecontext.context.buffer import TextBuffer
        from activecontext.context.markdown_parser import parse_markdown
        from activecontext.context.nodes import MediaType

        # Get or create text buffer for the file
        if content is not None:
            # Create buffer from provided content
            buffer = TextBuffer(
                path=path,
                lines=content.split("\n"),
            )
            self._text_buffers[buffer.buffer_id] = buffer
        else:
            # Check if we already have a buffer for this path
            full_path = os.path.join(self._cwd, path)
            buffer = None
            for existing in self._text_buffers.values():
                if existing.path == full_path or existing.path == path:
                    buffer = existing
                    break

            if buffer is None:
                # Create new buffer from file
                buffer = TextBuffer.from_file(path, cwd=self._cwd)
                self._text_buffers[buffer.buffer_id] = buffer

        # Parse markdown to get heading sections
        buffer_content = "\n".join(buffer.lines)
        result = parse_markdown(buffer_content)

        if not result.sections:
            # No headings - create single TextNode for entire file
            node = TextNode(
                path=path,
                tokens=tokens,
                state=state,
                media_type=MediaType.MARKDOWN,
                buffer_id=buffer.buffer_id,
                start_line=1,
                end_line=len(buffer.lines),
            )
            self._context_graph.add_node(node)
            return node

        # Create TextNode for each heading section
        all_nodes: list[TextNode] = []
        section_nodes: dict[int, TextNode] = {}  # section index -> node

        for i, section in enumerate(result.sections):
            node = TextNode(
                path=path,
                tokens=tokens // max(len(result.sections), 1),
                state=state,
                media_type=MediaType.MARKDOWN,
                buffer_id=buffer.buffer_id,
                start_line=section.start_line,
                end_line=section.end_line,
            )
            # Store heading info in tags for rendering
            node.tags["heading"] = section.title
            node.tags["level"] = section.level
            all_nodes.append(node)
            section_nodes[i] = node

        # Add all nodes to graph first
        for node in all_nodes:
            self._context_graph.add_node(node)

        # Build hierarchy based on heading levels
        # Stack: [(node, level)]
        root = all_nodes[0]
        stack: list[tuple[TextNode, int]] = [(root, result.sections[0].level)]

        for i in range(1, len(all_nodes)):
            node = all_nodes[i]
            level = result.sections[i].level

            # Find parent: pop until we find a node with lower level
            while len(stack) > 1 and stack[-1][1] >= level:
                stack.pop()

            parent_node = stack[-1][0]
            # Link child to parent (nodes are in graph, so this uses graph.link)
            parent_node.add_child(node)

            stack.append((node, level))

        # Determine parent: explicit > current_group > none
        effective_parent = parent
        if effective_parent is None and self._current_group_id:
            effective_parent = self._current_group_id

        # Link root to parent if set
        if effective_parent:
            parent_id = effective_parent.node_id if isinstance(effective_parent, ContextNode) else effective_parent
            self._context_graph.link(root.node_id, parent_id)

        return root

    def _make_view(
        self,
        media_type: str,
        path: str,
        *,
        tokens: int = 2000,
        state: NodeState = NodeState.ALL,
        **kwargs: Any,
    ) -> TextNode:
        """Dispatcher for creating text views based on media type.

        Routes to text() or markdown() based on the media_type parameter.

        Args:
            media_type: "text" or "markdown"
            path: File path relative to session cwd
            tokens: Token budget for rendering
            state: Rendering state
            **kwargs: Additional arguments passed to underlying function

        Returns:
            TextNode (or root TextNode for markdown)

        Raises:
            ValueError: If media_type is not recognized
        """
        if media_type == "markdown":
            return self._make_markdown_node(path, tokens=tokens, state=state, **kwargs)
        elif media_type == "text":
            return self._make_text_node(path, tokens=tokens, state=state, **kwargs)
        else:
            raise ValueError(f"Unknown media_type: {media_type}. Use 'text' or 'markdown'.")

    def _link(
        self,
        child: ContextNode | str,
        parent: ContextNode | str,
    ) -> bool:
        """Link a child node to a parent node.

        A node can have multiple parents (DAG structure).

        Args:
            child: Child node or node ID
            parent: Parent node or node ID

        Returns:
            True if link was created, False if failed
        """
        child_id = child.node_id if isinstance(child, ContextNode) else child
        parent_id = parent.node_id if isinstance(parent, ContextNode) else parent
        return self._context_graph.link(child_id, parent_id)

    def _unlink(
        self,
        child: ContextNode | str,
        parent: ContextNode | str,
    ) -> bool:
        """Remove link between child and parent.

        Args:
            child: Child node or node ID
            parent: Parent node or node ID

        Returns:
            True if link was removed, False if failed
        """
        child_id = child.node_id if isinstance(child, ContextNode) else child
        parent_id = parent.node_id if isinstance(parent, ContextNode) else parent
        return self._context_graph.unlink(child_id, parent_id)

    def _checkpoint(self, name: str) -> Any:
        """Create a checkpoint of the current DAG structure.

        Captures the organizational structure (edges) and group state,
        allowing later restoration via restore().

        Args:
            name: Human-readable name for the checkpoint

        Returns:
            The created Checkpoint object
        """
        return self._context_graph.checkpoint(name)

    def _restore(self, name_or_checkpoint: str | Any) -> None:
        """Restore DAG structure from a checkpoint.

        Replaces current parent/child links with those from the checkpoint.
        Content nodes are preserved; only organizational structure changes.

        Args:
            name_or_checkpoint: Checkpoint name (str) or Checkpoint object
        """
        self._context_graph.restore(name_or_checkpoint)

    def _list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with their metadata.

        Returns:
            List of checkpoint digests (name, created_at, edge_count, etc.)
        """
        return [cp.get_digest() for cp in self._context_graph.get_checkpoints()]

    def _branch(self, name: str) -> Any:
        """Save current structure as a checkpoint and continue.

        Convenience function that creates a checkpoint of the current state,
        allowing you to continue modifying the DAG while preserving the
        checkpoint for later restoration.

        Args:
            name: Name for the checkpoint

        Returns:
            The created Checkpoint object
        """
        return self._context_graph.checkpoint(name)

    def _ls_handles(self) -> list[dict[str, Any]]:
        """List all context object handles with brief digests."""
        return [node.GetDigest() for node in self._context_graph]

    def _show_handle(self, obj: Any, *, lod: int | None = None, tokens: int | None = None) -> str:
        """Force render a handle (placeholder)."""
        digest = obj.GetDigest() if hasattr(obj, "GetDigest") else str(obj)
        return f"[{digest}]"

    def _shell(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = 30.0,
        *,
        tokens: int = 2000,
        state: NodeState = NodeState.DETAILS,
    ) -> ShellNode:
        """Execute a shell command asynchronously, returning a ShellNode.

        Creates a ShellNode in the context graph and starts the subprocess
        in the background. The node's status updates when the command completes,
        and change notifications propagate up the DAG at tick time.

        Args:
            command: The command to execute (e.g., "pytest", "git").
            args: Optional list of arguments (e.g., ["tests/", "-v"]).
            cwd: Working directory. If None, uses session cwd.
            env: Additional environment variables.
            timeout: Timeout in seconds (default: 30). None for no timeout.
            tokens: Token budget for rendering output (default: 2000).
            state: Initial rendering state (default: DETAILS).

        Returns:
            ShellNode that tracks the command execution.
        """
        # Create the ShellNode
        node = ShellNode(
            command=command,
            args=args or [],
            tokens=tokens,
            state=state,
        )

        # Add to context graph
        self._context_graph.add_node(node)

        # Start background execution
        task = asyncio.create_task(
            self._shell_background_task(
                node_id=node.node_id,
                command=command,
                args=args,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        )
        self._shell_tasks[node.node_id] = task

        return node

    async def _shell_background_task(
        self,
        node_id: str,
        command: str,
        args: list[str] | None,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: float | None,
    ) -> None:
        """Background task that executes a shell command and stores the result.

        This runs in the background while the agent continues. When complete,
        it stores the result in _pending_shell_results for processing at tick.
        """
        # Get the node (stays PENDING until permission is granted)
        node = self._context_graph.get_node(node_id)

        result: ShellResult

        # Check permission if manager is configured
        if self._shell_permission_manager:
            if not self._shell_permission_manager.check_access(command, args):
                # Permission denied - try to request
                if self._shell_permission_requester:
                    try:
                        granted, persist = await self._shell_permission_requester(
                            self._session_id, command, args
                        )
                    except asyncio.CancelledError:
                        # Cancelled during permission request
                        full_cmd = f"{command} {' '.join(args or [])}"
                        result = ShellResult(
                            command=full_cmd,
                            exit_code=-1,
                            output="Cancelled",
                            truncated=False,
                            status="cancelled",
                            signal="SIGTERM",
                            duration_ms=0,
                        )
                        self._pending_shell_results.append((node_id, result))
                        return

                    if granted:
                        if persist:
                            # "Allow always" - write to config file
                            write_shell_permission_to_config(
                                Path(self._cwd), command, args
                            )
                            # Reload config to pick up new rule
                            from activecontext.config import load_config

                            config = load_config(session_root=self._cwd)
                            self._shell_permission_manager.reload(config.sandbox)
                        else:
                            # "Allow once" - grant temporary access
                            self._shell_permission_manager.grant_temporary(command, args)
                    else:
                        # Denied - store error result
                        full_cmd = f"{command} {' '.join(args or [])}"
                        result = ShellResult(
                            command=full_cmd,
                            exit_code=126,  # Permission denied exit code
                            output=f"Shell command denied by sandbox policy: {full_cmd}",
                            truncated=False,
                            status="error",
                            signal=None,
                            duration_ms=0,
                        )
                        self._pending_shell_results.append((node_id, result))
                        return
                else:
                    # No requester available - store error result
                    full_cmd = f"{command} {' '.join(args or [])}"
                    result = ShellResult(
                        command=full_cmd,
                        exit_code=126,
                        output=f"Shell command denied by sandbox policy: {full_cmd}",
                        truncated=False,
                        status="error",
                        signal=None,
                        duration_ms=0,
                    )
                    self._pending_shell_results.append((node_id, result))
                    return

        # Permission granted (or no manager) - mark as running and execute
        if isinstance(node, ShellNode):
            node.shell_status = ShellStatus.RUNNING
            node.started_at_exec = time.time()

        try:
            result = await self._terminal_executor.execute(
                command=command,
                args=args,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        except asyncio.CancelledError:
            full_cmd = f"{command} {' '.join(args or [])}"
            result = ShellResult(
                command=full_cmd,
                exit_code=-1,
                output="Cancelled",
                truncated=False,
                status="cancelled",
                signal="SIGTERM",
                duration_ms=0,
            )
        except Exception as e:
            full_cmd = f"{command} {' '.join(args or [])}"
            result = ShellResult(
                command=full_cmd,
                exit_code=-1,
                output=f"Error: {e}",
                truncated=False,
                status="error",
                signal=None,
                duration_ms=0,
            )

        # Store result for processing at tick time
        self._pending_shell_results.append((node_id, result))

    def process_pending_shell_results(self) -> list[str]:
        """Process pending shell results and update nodes.

        Called during tick to apply async shell results. This triggers
        node change notifications that propagate up the DAG.

        Returns:
            List of node IDs that were updated.
        """
        updated_nodes: list[str] = []

        while self._pending_shell_results:
            node_id, result = self._pending_shell_results.pop(0)

            node = self._context_graph.get_node(node_id)
            if not isinstance(node, ShellNode):
                continue

            # Apply the result - this triggers _mark_changed() and notifications
            if result.status == "timeout":
                node.set_timeout(result.output, result.duration_ms)
            elif result.signal:
                node.set_completed(
                    exit_code=result.exit_code or -1,
                    output=result.output,
                    duration_ms=result.duration_ms,
                    truncated=result.truncated,
                    signal=result.signal,
                )
            else:
                node.set_completed(
                    exit_code=result.exit_code or 0,
                    output=result.output,
                    duration_ms=result.duration_ms,
                    truncated=result.truncated,
                )

            updated_nodes.append(node_id)

            # Clean up task reference
            self._shell_tasks.pop(node_id, None)

        return updated_nodes

    def cancel_shell(self, node_id: str) -> bool:
        """Cancel a running shell command.

        Args:
            node_id: The ShellNode ID to cancel.

        Returns:
            True if a task was cancelled, False if not found/already done.
        """
        task = self._shell_tasks.get(node_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def cancel_all_shells(self) -> int:
        """Cancel all running shell commands.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        for node_id, task in list(self._shell_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    # =========================================================================
    # File locking DSL functions
    # =========================================================================

    def _lock_file(
        self,
        lockfile: str,
        timeout: float = 30.0,
        *,
        tokens: int = 200,
        state: NodeState = NodeState.COLLAPSED,
    ) -> LockNode:
        """Acquire an exclusive file lock asynchronously, returning a LockNode.

        Creates a LockNode in the context graph and starts lock acquisition
        in the background. The node's status updates when the lock is acquired
        or times out, and change notifications propagate up the DAG at tick time.

        The lock uses OS-level file locking (fcntl on Unix, msvcrt on Windows).
        The lockfile is created if it doesn't exist.

        Args:
            lockfile: Path to the lock file (will be created if needed).
            timeout: Timeout in seconds for acquisition (default: 30).
            tokens: Token budget for rendering (default: 200).
            state: Initial rendering state (default: COLLAPSED).

        Returns:
            LockNode that tracks the lock status.

        Example:
            lock = lock_file(".mylock", timeout=10)
            wait(lock, wake_prompt="Lock acquired, proceeding...")
        """
        import os

        # Resolve path relative to cwd
        if not os.path.isabs(lockfile):
            lockfile = os.path.join(self._cwd, lockfile)

        # Create the LockNode
        node = LockNode(
            lockfile=lockfile,
            timeout=timeout,
            tokens=tokens,
            state=state,
        )

        # Add to context graph
        self._context_graph.add_node(node)

        # Start background lock acquisition
        task = asyncio.create_task(
            self._lock_background_task(
                node_id=node.node_id,
                lockfile=lockfile,
                timeout=timeout,
            )
        )
        self._lock_tasks[node.node_id] = task

        return node

    async def _lock_background_task(
        self,
        node_id: str,
        lockfile: str,
        timeout: float,
    ) -> None:
        """Background task that acquires a file lock with timeout.

        This runs in the background while the agent continues. When complete,
        it stores the result in _pending_lock_results for processing at tick.
        """
        import sys

        start_time = time.time()
        poll_interval = 0.1  # 100ms between lock attempts

        try:
            # Create the lock file if it doesn't exist
            Path(lockfile).parent.mkdir(parents=True, exist_ok=True)

            # Open file for locking (create if needed)
            lock_fd = open(lockfile, "w")

            # Try to acquire lock with timeout
            while True:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    # Timeout - close file and report
                    lock_fd.close()
                    self._pending_lock_results.append((node_id, LockStatus.TIMEOUT, None))
                    return

                try:
                    # Platform-specific locking
                    if sys.platform == "win32":
                        import msvcrt
                        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Lock acquired - store file handle for later release
                    self._active_locks[node_id] = lock_fd
                    self._pending_lock_results.append((node_id, LockStatus.ACQUIRED, None))
                    return

                except OSError:
                    # Lock held by another process - wait and retry
                    await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            self._pending_lock_results.append((node_id, LockStatus.ERROR, "Cancelled"))
        except Exception as e:
            self._pending_lock_results.append((node_id, LockStatus.ERROR, str(e)))

    def _lock_release(self, lock: LockNode | str) -> bool:
        """Release a file lock and remove the lock file.

        Args:
            lock: The LockNode or node_id to release.

        Returns:
            True if lock was released, False if not found or not held.

        Example:
            lock_release(lock)  # or lock_release(lock.node_id)
        """
        import os
        import sys

        node_id = lock.node_id if isinstance(lock, LockNode) else lock

        # Get the lock node
        node = self._context_graph.get_node(node_id)
        if not isinstance(node, LockNode):
            return False

        # Check if we hold the lock
        lock_fd = self._active_locks.get(node_id)
        if lock_fd is None:
            return False

        try:
            # Release the lock
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # May already be released
            else:
                import fcntl
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass  # May already be released

            # Close the file handle
            lock_fd.close()

            # Remove the lock file
            try:
                os.unlink(node.lockfile)
            except OSError:
                pass  # File may already be removed

            # Update node state
            node.set_released()

            # Clean up
            del self._active_locks[node_id]

            # Cancel any pending acquisition task
            task = self._lock_tasks.pop(node_id, None)
            if task and not task.done():
                task.cancel()

            return True

        except Exception:
            return False

    def process_pending_lock_results(self) -> list[str]:
        """Process pending lock results and update nodes.

        Called during tick to apply async lock results. This triggers
        node change notifications that propagate up the DAG.

        Returns:
            List of node IDs that were updated.
        """
        import os

        updated_nodes: list[str] = []

        while self._pending_lock_results:
            node_id, status, error_msg = self._pending_lock_results.pop(0)

            node = self._context_graph.get_node(node_id)
            if not isinstance(node, LockNode):
                continue

            # Apply the result - this triggers _mark_changed() and notifications
            if status == LockStatus.ACQUIRED:
                node.set_acquired(os.getpid())
            elif status == LockStatus.TIMEOUT:
                node.set_timeout()
            elif status == LockStatus.ERROR:
                node.set_error(error_msg or "Unknown error")

            updated_nodes.append(node_id)

            # Clean up task reference
            self._lock_tasks.pop(node_id, None)

        return updated_nodes

    def cancel_lock(self, node_id: str) -> bool:
        """Cancel a pending lock acquisition.

        Args:
            node_id: The LockNode ID to cancel.

        Returns:
            True if a task was cancelled, False if not found/already done.
        """
        task = self._lock_tasks.get(node_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def cancel_all_locks(self) -> int:
        """Cancel all pending lock acquisitions.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        for node_id, task in list(self._lock_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def release_all_locks(self) -> int:
        """Release all held locks.

        Returns:
            Number of locks released.
        """
        released = 0
        for node_id in list(self._active_locks.keys()):
            if self._lock_release(node_id):
                released += 1
        return released

    async def close(self) -> None:
        """Clean up all background tasks and resources.

        Should be called when done with the Timeline to ensure proper cleanup
        of asyncio tasks. Can be used with `async with` or called directly.

        Example:
            timeline = Timeline("session-id", cwd="/path")
            try:
                await timeline.execute_statement(...)
            finally:
                await timeline.close()
        """
        # Cancel all pending shell tasks
        self.cancel_all_shells()

        # Cancel all pending lock tasks and release held locks
        self.cancel_all_locks()
        self.release_all_locks()

        # Give cancelled tasks a chance to complete
        if self._shell_tasks or self._lock_tasks:
            await asyncio.sleep(0)

    async def __aenter__(self) -> Timeline:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit - ensures cleanup."""
        await self.close()

    def _fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: Any = None,
        json: Any = None,
        timeout: float = 30.0,
    ) -> Any:
        """Perform HTTP/HTTPS request with permission checking.

        Returns a coroutine that must be awaited by execute_statement.
        If a website_permission_manager is configured, the URL will be
        checked against the permission rules before execution.

        Args:
            url: The URL to fetch.
            method: HTTP method (GET, POST, PUT, DELETE, etc.).
            headers: Optional request headers.
            data: Optional request body data.
            json: Optional JSON request body.
            timeout: Timeout in seconds (default: 30).

        Returns:
            Coroutine that resolves to httpx.Response.
        """
        return self._fetch_with_permission(
            url=url,
            method=method,
            headers=headers,
            data=data,
            json=json,
            timeout=timeout,
        )

    async def _fetch_with_permission(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: Any = None,
        json: Any = None,
        timeout: float = 30.0,
    ) -> Any:
        """Execute HTTP request with permission check and request flow.

        Args:
            url: The URL to fetch.
            method: HTTP method.
            headers: Optional request headers.
            data: Optional request body data.
            json: Optional JSON request body.
            timeout: Timeout in seconds.

        Returns:
            httpx.Response from execution, or raises WebsitePermissionDenied if denied.
        """
        # Check permission if manager is configured
        if self._website_permission_manager:
            if not self._website_permission_manager.check_access(url, method):
                # Permission denied - try to request
                if self._website_permission_requester:
                    granted, persist = await self._website_permission_requester(
                        self._session_id, url, method
                    )

                    if granted:
                        if persist:
                            # "Allow always" - write to config file
                            write_website_permission_to_config(
                                Path(self._cwd), url, method
                            )
                            # Reload config to pick up new rule
                            from activecontext.config import load_config

                            config = load_config(session_root=self._cwd)
                            self._website_permission_manager.reload(config.sandbox)
                        else:
                            # "Allow once" - grant temporary access
                            self._website_permission_manager.grant_temporary(url, method)
                    else:
                        # Denied - raise exception
                        raise WebsitePermissionDenied(url=url, method=method)
                else:
                    # No requester available - raise exception
                    raise WebsitePermissionDenied(url=url, method=method)

        # Permission granted (or no manager) - execute request
        if self._website_permission_manager:
            safe_fetch = make_safe_fetch(self._website_permission_manager)
            return await safe_fetch(
                url=url,
                method=method,
                headers=headers,
                data=data,
                json=json,
                timeout=timeout,
            )
        else:
            # No permission manager - execute directly
            import httpx

            async with httpx.AsyncClient(timeout=timeout) as client:
                return await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=data,
                    json=json,
                )

    def _done(self, message: str = "") -> None:
        """Signal that the agent has completed its task.

        Args:
            message: Final message to send to the user.
        """
        self._done_called = True
        self._done_message = message
        if message:
            print(message)

    def _set_notify(
        self,
        node: ContextNode | str,
        level: NotificationLevel | str = NotificationLevel.WAKE,
    ) -> ContextNode:
        """Set notification level for a node.

        Controls how changes to this node are communicated to the agent:
        - IGNORE: Changes propagate but no notification generated (default)
        - HOLD: Notification queued, delivered at tick boundary
        - WAKE: Notification queued AND agent woken immediately

        Args:
            node: Node object or node_id string
            level: NotificationLevel or string ("ignore", "hold", "wake")

        Returns:
            The node (for chaining)

        Examples:
            notify(v, NotificationLevel.WAKE)  # Wake on changes
            notify(v, "hold")  # Queue notifications
            v = text("file.py").SetNotify(NotificationLevel.WAKE)  # Fluent API
        """
        if isinstance(node, str):
            resolved = self._context_graph.get_node(node)
            if not resolved:
                raise ValueError(f"Node not found: {node}")
            node = resolved

        if isinstance(level, str):
            level = NotificationLevel(level)

        node.notification_level = level
        return node

    def _set_title(self, title: str) -> None:
        """Set the session title.

        This updates the session title that appears in IDE menus
        and conversation history.

        Args:
            title: New title for the session.
        """
        if self._set_title_callback:
            self._set_title_callback(title)
            print(f"Session title set to: {title}")
        else:
            print("Warning: set_title not available (no session callback registered)")

    def is_done(self) -> bool:
        """Check if done() was called."""
        return self._done_called

    def get_done_message(self) -> str | None:
        """Get the message passed to done(), if any."""
        return self._done_message

    def reset_done(self) -> None:
        """Reset the done signal (call at start of each prompt)."""
        self._done_called = False
        self._done_message = None

    # =========================================================================
    # Wait condition DSL functions
    # =========================================================================

    def _wait(
        self,
        node: ContextNode | str,
        *,
        wake_prompt: str = "Node completed.",
        timeout: float | None = None,
        timeout_prompt: str | None = None,
        failure_prompt: str | None = None,
    ) -> None:
        """Wait for a single node to complete.

        Ends the current turn and waits for the specified node (typically a
        ShellNode) to complete. Ticks continue while waiting. When the node
        completes, the wake_prompt is injected and the agent turn resumes.

        Args:
            node: The node to wait for (ContextNode or node_id string).
            wake_prompt: Prompt injected when node completes. Can use {node}
                for the completed node.
            timeout: Optional timeout in seconds.
            timeout_prompt: Prompt injected if timeout expires.
            failure_prompt: Prompt injected if node fails.
        """
        node_id = node.node_id if isinstance(node, ContextNode) else node
        self._wait_condition = WaitCondition(
            node_ids=[node_id],
            mode=WaitMode.SINGLE,
            wake_prompt=wake_prompt,
            timeout=timeout,
            timeout_prompt=timeout_prompt,
            failure_prompt=failure_prompt,
        )
        # Mark as done-like to end the turn
        self._done_called = True

    def _wait_all(
        self,
        *nodes: ContextNode | str,
        wake_prompt: str = "All nodes completed.",
        timeout: float | None = None,
        timeout_prompt: str | None = None,
        failure_prompt: str | None = None,
    ) -> None:
        """Wait for all specified nodes to complete.

        Ends the current turn and waits for all specified nodes to complete.
        Ticks continue while waiting. When all nodes complete, the wake_prompt
        is injected and the agent turn resumes.

        Args:
            *nodes: Nodes to wait for (ContextNode or node_id strings).
            wake_prompt: Prompt injected when all complete.
            timeout: Optional timeout in seconds.
            timeout_prompt: Prompt injected if timeout expires.
            failure_prompt: Prompt injected if any node fails.
        """
        node_ids = [
            n.node_id if isinstance(n, ContextNode) else n
            for n in nodes
        ]
        self._wait_condition = WaitCondition(
            node_ids=node_ids,
            mode=WaitMode.ALL,
            wake_prompt=wake_prompt,
            timeout=timeout,
            timeout_prompt=timeout_prompt,
            failure_prompt=failure_prompt,
        )
        self._done_called = True

    def _wait_any(
        self,
        *nodes: ContextNode | str,
        wake_prompt: str = "A node completed: {node}",
        timeout: float | None = None,
        timeout_prompt: str | None = None,
        failure_prompt: str | None = None,
        cancel_others: bool = False,
    ) -> None:
        """Wait for any of the specified nodes to complete.

        Ends the current turn and waits for the first node to complete.
        Ticks continue while waiting. When any node completes, the wake_prompt
        is injected and the agent turn resumes.

        Args:
            *nodes: Nodes to wait for (ContextNode or node_id strings).
            wake_prompt: Prompt injected when first completes. Use {node}
                for the completed node.
            timeout: Optional timeout in seconds.
            timeout_prompt: Prompt injected if timeout expires.
            failure_prompt: Prompt injected if any node fails.
            cancel_others: If True, cancel remaining nodes when first completes.
        """
        node_ids = [
            n.node_id if isinstance(n, ContextNode) else n
            for n in nodes
        ]
        self._wait_condition = WaitCondition(
            node_ids=node_ids,
            mode=WaitMode.ANY,
            wake_prompt=wake_prompt,
            timeout=timeout,
            timeout_prompt=timeout_prompt,
            failure_prompt=failure_prompt,
            cancel_others=cancel_others,
        )
        self._done_called = True

    # === Work Coordination DSL Functions ===

    def _work_on(
        self,
        intent: str,
        *files: str,
        mode: str = "write",
        dependencies: list[str] | None = None,
    ) -> WorkNode:
        """Register intent and files being worked on.

        Creates a WorkNode in the context graph to display coordination status.
        Also registers with the project-wide scratchpad for cross-agent visibility.

        Args:
            intent: Human-readable description of work
            *files: File paths being accessed
            mode: Access mode for files ("read" or "write")
            dependencies: Additional files needed (read-only)

        Returns:
            WorkNode for displaying coordination status

        Example:
            work_on("Implementing OAuth2", "src/auth/oauth.py", "src/auth/config.py")
        """
        from activecontext.coordination.schema import FileAccess

        if not self._scratchpad_manager:
            raise RuntimeError("Scratchpad manager not configured")

        # Build file access list
        file_accesses = [FileAccess(path=f, mode=mode) for f in files]

        # Register with scratchpad
        entry = self._scratchpad_manager.register(
            session_id=self._session_id,
            intent=intent,
            files=file_accesses,
            dependencies=dependencies,
        )

        # Check for conflicts
        all_paths = list(files) + (dependencies or [])
        conflicts = self._scratchpad_manager.get_conflicts(all_paths, mode)

        # Create or update WorkNode
        if self._work_node is None:
            node_id = f"work_{uuid.uuid4().hex[:8]}"
            self._work_node = WorkNode(
                node_id=node_id,
                tokens=200,
                state=NodeState.DETAILS,
                intent=entry.intent,
                work_status=entry.status,
                files=[f.to_dict() for f in file_accesses],
                dependencies=dependencies or [],
                conflicts=[c.to_dict() for c in conflicts],
                agent_id=entry.id,
            )
            self._context_graph.add_node(self._work_node)
        else:
            self._work_node.intent = entry.intent
            self._work_node.work_status = entry.status
            self._work_node.files = [f.to_dict() for f in file_accesses]
            self._work_node.dependencies = dependencies or []
            self._work_node.set_conflicts([c.to_dict() for c in conflicts])
            self._work_node.agent_id = entry.id

        return self._work_node

    def _work_check(self, *files: str, mode: str = "write") -> list[dict[str, str]]:
        """Check for conflicts on files before modifying.

        Args:
            *files: File paths to check
            mode: Access mode we want ("read" or "write")

        Returns:
            List of conflicts: [{agent_id, file, their_mode, their_intent}, ...]

        Example:
            conflicts = work_check("src/auth/utils.py")
            if conflicts:
                print(f"Warning: {conflicts[0]['agent_id']} is working on this")
        """
        if not self._scratchpad_manager:
            return []

        conflicts = self._scratchpad_manager.get_conflicts(list(files), mode)
        return [c.to_dict() for c in conflicts]

    def _work_update(
        self,
        intent: str | None = None,
        files: list[str] | None = None,
        mode: str = "write",
        dependencies: list[str] | None = None,
        status: str | None = None,
    ) -> WorkNode | None:
        """Update current work registration.

        Args:
            intent: New intent description
            files: New file list (replaces existing)
            mode: Access mode for new files
            dependencies: New dependencies
            status: New status (active/paused/done)

        Returns:
            Updated WorkNode, or None if not registered
        """
        from activecontext.coordination.schema import FileAccess

        if not self._scratchpad_manager:
            return None

        # Build file access list if provided
        file_accesses: list[FileAccess] | None = None
        if files is not None:
            file_accesses = [FileAccess(path=f, mode=mode) for f in files]

        # Update scratchpad entry
        entry = self._scratchpad_manager.update(
            intent=intent,
            files=file_accesses,
            dependencies=dependencies,
            status=status,
        )

        if entry is None:
            return None

        # Update WorkNode
        if self._work_node:
            if intent is not None:
                self._work_node.intent = intent
            if file_accesses is not None:
                self._work_node.files = [f.to_dict() for f in file_accesses]
            if dependencies is not None:
                self._work_node.dependencies = dependencies
            if status is not None:
                self._work_node.work_status = status

            # Refresh conflicts
            all_paths = [f["path"] for f in self._work_node.files]
            conflicts = self._scratchpad_manager.get_conflicts(all_paths, mode)
            self._work_node.set_conflicts([c.to_dict() for c in conflicts])

        return self._work_node

    def _work_done(self) -> None:
        """Mark work as complete and unregister.

        Removes this agent's entry from the scratchpad and hides the WorkNode.
        """
        if not self._scratchpad_manager:
            return

        self._scratchpad_manager.unregister()

        if self._work_node:
            self._work_node.work_status = "done"
            self._work_node.state = NodeState.HIDDEN

    def _work_list(self) -> list[dict[str, Any]]:
        """List all active work entries from all agents.

        Returns:
            List of work entries with files, intent, status, etc.
        """
        if not self._scratchpad_manager:
            return []

        entries = self._scratchpad_manager.get_all_entries()
        return [
            {
                "agent_id": e.id,
                "session_id": e.session_id,
                "intent": e.intent,
                "status": e.status,
                "files": [f.to_dict() for f in e.files],
                "dependencies": e.dependencies,
                "started_at": e.started_at.isoformat(),
                "updated_at": e.updated_at.isoformat(),
            }
            for e in entries
        ]

    # =========================================================================
    # Multi-Agent DSL Functions
    # =========================================================================

    def _spawn_agent(
        self,
        agent_type: str,
        task: str,
        **kwargs: Any,
    ) -> AgentHandle:
        """Spawn a new child agent.

        DSL function: spawn(agent_type, task, **kwargs)

        Args:
            agent_type: Type ID (explorer, summarizer, etc.)
            task: Task description
            **kwargs: Additional arguments for session creation

        Returns:
            AgentHandle for interacting with the spawned agent

        Example:
            agent = spawn("explorer", task="find auth code")
            agent.Send("look for OAuth")
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")


        # Spawn synchronously using existing loop
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # Create a coroutine and schedule it
            coro = self._agent_manager.spawn_agent(
                agent_type=agent_type,
                task=task,
                parent_id=self._agent_id,
                cwd=self._cwd,
                **kwargs,
            )
            # Use run_coroutine_threadsafe if we're not in the main thread
            # or use ensure_future if we're in an async context
            future = asyncio.ensure_future(coro)
            # This won't block properly in sync context, so we need a different approach
            # For now, we'll raise if called from non-async context
            return future  # type: ignore  # Will be awaited by exec
        except RuntimeError:
            # No running loop - shouldn't happen in normal DSL execution
            raise RuntimeError("spawn() must be called in async context")

    async def _spawn_agent_async(
        self,
        agent_type: str,
        task: str,
        **kwargs: Any,
    ) -> AgentHandle:
        """Async version of spawn_agent for internal use."""
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        return await self._agent_manager.spawn_agent(
            agent_type=agent_type,
            task=task,
            parent_id=self._agent_id,
            cwd=self._cwd,
            **kwargs,
        )

    def _send_message(
        self,
        target: Any,  # AgentHandle or str (agent_id)
        content: str,
        *node_refs: ContextNode,
    ) -> str:
        """Send a message to another agent.

        DSL function: send(target, content, *node_refs)

        Args:
            target: AgentHandle or agent ID string
            content: Message content
            *node_refs: Nodes to share with the recipient

        Returns:
            Message ID

        Example:
            send(agent, "summarize this group", my_group)
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        # Resolve target to agent_id
        if isinstance(target, AgentHandle):
            recipient = target.agent_id
        elif isinstance(target, str):
            recipient = target
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(target)}")

        # Share nodes and collect IDs
        ref_ids: list[str] = []
        for node in node_refs:
            self._agent_manager.share_node(node)
            ref_ids.append(node.node_id)

        # Get sender ID
        sender = self._agent_id or "unknown"

        # Send message (sync operation via scratchpad)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            coro = self._agent_manager.send_message(
                sender=sender,
                recipient=recipient,
                content=content,
                node_refs=ref_ids,
            )
            return asyncio.ensure_future(coro)  # type: ignore
        except RuntimeError:
            raise RuntimeError("send() must be called in async context")

    def _send_update(
        self,
        content: str,
        *node_refs: ContextNode,
    ) -> str:
        """Send an update message to parent agent and continue running.

        DSL function: send_update(content, *node_refs)

        Unlike send(), this sends to the parent agent specifically and
        does NOT end the turn - the agent continues executing.

        Args:
            content: Message content
            *node_refs: Nodes to share with the parent

        Returns:
            Message ID

        Example:
            send_update("Found 3 auth files so far", partial_results)
            # Agent continues working...
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        # Get parent ID
        entry = self._agent_manager.get_agent(self._agent_id) if self._agent_id else None
        if not entry or not entry.parent_id:
            raise RuntimeError("No parent agent to send update to")

        parent_id = entry.parent_id

        # Share nodes and collect IDs
        ref_ids: list[str] = []
        for node in node_refs:
            self._agent_manager.share_node(node)
            ref_ids.append(node.node_id)

        # Get sender ID
        sender = self._agent_id or "unknown"

        # Send message (sync operation via scratchpad)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            coro = self._agent_manager.send_message(
                sender=sender,
                recipient=parent_id,
                content=content,
                node_refs=ref_ids,
            )
            future = asyncio.ensure_future(coro)
            # Don't set _done_called - agent continues running
            return future  # type: ignore
        except RuntimeError:
            raise RuntimeError("send_update() must be called in async context")

    def _recv_messages(self) -> list[dict[str, Any]]:
        """Receive pending messages for this agent.

        DSL function: recv()

        Returns:
            List of message dictionaries with sender, content, node_refs, etc.

        Example:
            messages = recv()
            for msg in messages:
                print(f"From {msg['sender']}: {msg['content']}")
        """
        if not self._agent_manager:
            return []

        if not self._agent_id:
            return []

        messages = self._agent_manager.get_messages(self._agent_id, status="pending")

        # Mark as read and convert to dicts
        result: list[dict[str, Any]] = []
        for msg in messages:
            self._agent_manager.mark_message_read(msg.id)
            result.append({
                "id": msg.id,
                "sender": msg.sender,
                "content": msg.content,
                "node_refs": msg.node_refs,
                "created_at": msg.created_at.isoformat(),
                "reply_to": msg.reply_to,
                "metadata": msg.metadata,
            })

        return result

    def _wait_message(
        self,
        timeout: float | None = None,
        wake_prompt: str = "You have received a message: {content}",
        timeout_prompt: str = "Timed out waiting for message",
    ) -> None:
        """Wait for an incoming message.

        DSL function: wait_message(timeout=None, wake_prompt="...")

        Sets up a wait condition that blocks the turn until a message arrives.

        Args:
            timeout: Optional timeout in seconds
            wake_prompt: Prompt template injected when message arrives
                        Can use {sender}, {content}, {node_refs}
            timeout_prompt: Prompt injected on timeout
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        if not self._agent_id:
            raise RuntimeError("No agent ID set")

        self._wait_condition = WaitCondition(
            node_ids=[],
            mode=WaitMode.MESSAGE,
            wake_prompt=wake_prompt,
            timeout=timeout,
            timeout_prompt=timeout_prompt,
            agent_id=self._agent_id,
        )
        self._done_called = True  # End turn to wait

    def _list_agents(self) -> list[dict[str, Any]]:
        """List all active agents.

        DSL function: agents()

        Returns:
            List of agent info dictionaries
        """
        if not self._agent_manager:
            return []

        entries = self._agent_manager.list_agents()
        return [
            {
                "id": e.id,
                "type": e.agent_type,
                "task": e.task,
                "state": e.state.value,
                "parent_id": e.parent_id,
                "session_id": e.session_id,
            }
            for e in entries
        ]

    def _get_agent_status(self, agent: Any) -> dict[str, Any]:
        """Get status of a specific agent.

        DSL function: agent_status(agent)

        Args:
            agent: AgentHandle or agent ID string

        Returns:
            Agent status dictionary
        """
        if not self._agent_manager:
            return {"error": "Agent manager not available"}

        from activecontext.agents.handle import AgentHandle

        # Resolve agent_id
        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            return {"error": f"Expected AgentHandle or str, got {type(agent)}"}

        entry = self._agent_manager.get_agent(agent_id)
        if not entry:
            return {"error": f"Agent {agent_id} not found"}

        return {
            "id": entry.id,
            "type": entry.agent_type,
            "task": entry.task,
            "state": entry.state.value,
            "parent_id": entry.parent_id,
            "session_id": entry.session_id,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

    def _pause_agent(self, agent: Any) -> None:
        """Pause an agent.

        DSL function: pause_agent(agent)

        Args:
            agent: AgentHandle or agent ID string
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(agent)}")

        import asyncio
        asyncio.ensure_future(self._agent_manager.pause_agent(agent_id))

    def _resume_agent(self, agent: Any) -> None:
        """Resume a paused agent.

        DSL function: resume_agent(agent)

        Args:
            agent: AgentHandle or agent ID string
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(agent)}")

        import asyncio
        asyncio.ensure_future(self._agent_manager.resume_agent(agent_id))

    def _terminate_agent(self, agent: Any) -> None:
        """Terminate an agent.

        DSL function: terminate_agent(agent)

        Args:
            agent: AgentHandle or agent ID string
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(agent)}")

        import asyncio
        asyncio.ensure_future(self._agent_manager.terminate_agent(agent_id))

    def _get_shared_node(self, node_id: str) -> ContextNode | None:
        """Get a shared node by ID.

        DSL function: get_shared_node(node_id)

        Args:
            node_id: Node ID from another agent's message

        Returns:
            The shared node, or None if not found
        """
        if not self._agent_manager:
            return None

        return self._agent_manager.get_shared_node(node_id)

    # =========================================================================
    # MCP (Model Context Protocol) DSL Functions
    # =========================================================================

    async def _mcp_connect(
        self,
        name: str | None = None,
        *,
        command: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        tokens: int = 1000,
        state: NodeState = NodeState.DETAILS,
    ) -> MCPServerNode:
        """Connect to an MCP server.

        Args:
            name: Server name from config, or custom name for dynamic connection
            command: For stdio transport: command and args to spawn server
            url: For streamable-http transport: server URL
            env: Environment variables for the server process
            tokens: Token budget for tool documentation
            state: Initial rendering state

        Returns:
            MCPServerNode representing the connection

        Examples:
            # Connect to configured server
            fs = mcp_connect("filesystem")

            # Dynamic stdio connection
            gh = mcp_connect("github", command=["npx", "-y", "@mcp/server-github"])

            # Dynamic HTTP connection
            tools = mcp_connect("remote", url="http://localhost:8000/mcp")
        """
        from activecontext.config.schema import MCPConnectMode, MCPServerConfig

        # Build config if dynamic
        config: MCPServerConfig | None = None
        if command or url:
            config = MCPServerConfig(
                name=name or "dynamic",
                command=command,
                url=url,
                env=env or {},
                transport="stdio" if command else "streamable-http",
            )
            name = config.name
        elif name is None:
            raise ValueError("Must provide 'name' or 'command'/'url'")

        # Check if server is disabled (NEVER mode) when using config from file
        if config is None and self._mcp_client_manager.config:
            for server_config in self._mcp_client_manager.config.servers:
                if server_config.name == name:
                    if server_config.connect == MCPConnectMode.NEVER:
                        raise ValueError(
                            f"MCP server '{name}' is disabled (connect=never)"
                        )
                    break

        # Connect via MCPClientManager
        connection = await self._mcp_client_manager.connect(name=name, config=config)

        # Create or update MCPServerNode
        if name in self._mcp_server_nodes:
            node = self._mcp_server_nodes[name]
        else:
            node = MCPServerNode(
                server_name=name,
                tokens=tokens,
                state=state,
            )
            self._mcp_server_nodes[name] = node
            self._context_graph.add_node(node)

            # Wire up MCP result callback to fire events
            node.set_on_result_callback(self.fire_event)

        # Update node from connection
        node.update_from_connection(connection)

        # Register with MCPManagerNode
        from activecontext.context.nodes import MCPManagerNode

        mcp_manager = self._context_graph.get_node("mcp_manager")
        if mcp_manager and isinstance(mcp_manager, MCPManagerNode):
            # Link in graph (server as child of manager)
            self._context_graph.link(node.node_id, mcp_manager.node_id)
            mcp_manager.register_server(node)

        # Update namespace with the server proxy
        bindings = self._mcp_client_manager.generate_namespace_bindings()
        if name in bindings:
            proxy = bindings[name]
            # Augment proxy with tool() method for accessing MCPToolNode children
            proxy._mcp_node = node
            proxy.tool = node.tool
            proxy.tool_nodes = node.tool_nodes
            self._namespace[name] = proxy

        # Add tool nodes to namespace with {server}_{tool} naming
        for tool_name, tool_node_id in node._tool_nodes.items():
            tool_node = self._context_graph.get_node(tool_node_id)
            if tool_node:
                # e.g., filesystem_read_file
                namespace_key = f"{name}_{tool_name}"
                self._namespace[namespace_key] = tool_node

        return node

    async def _mcp_disconnect(self, name: str) -> None:
        """Disconnect from an MCP server.

        Args:
            name: Name of the server to disconnect
        """
        await self._mcp_client_manager.disconnect(name)

        # Update node status and clean up tool children
        if name in self._mcp_server_nodes:
            node = self._mcp_server_nodes[name]

            # Remove tool nodes from namespace and graph
            for tool_name, tool_node_id in list(node._tool_nodes.items()):
                # Remove from namespace
                namespace_key = f"{name}_{tool_name}"
                if namespace_key in self._namespace:
                    del self._namespace[namespace_key]
                # Remove from graph
                tool_node = self._context_graph.get_node(tool_node_id)
                if tool_node:
                    tool_node._mark_changed(
                        description=f"Tool '{tool_name}' removed (server disconnected)"
                    )
                    self._context_graph.remove_node(tool_node_id)
            node._tool_nodes.clear()

            node.status = "disconnected"
            node.tools = []
            node.resources = []
            node.prompts = []
            node._mark_changed(description=f"MCP {name}: disconnected")

        # Unregister from MCPManagerNode
        from activecontext.context.nodes import MCPManagerNode

        mcp_manager = self._context_graph.get_node("mcp_manager")
        if mcp_manager and isinstance(mcp_manager, MCPManagerNode):
            mcp_manager.unregister_server(name)

        # Remove from namespace
        if name in self._namespace:
            del self._namespace[name]

    def _mcp_list(self) -> list[dict[str, Any]]:
        """List all MCP server connections and their status.

        Returns:
            List of connection info dicts with name, status, tool/resource counts.
        """
        return [
            {
                "name": conn.name,
                "status": conn.status.value,
                "tools": len(conn.tools),
                "resources": len(conn.resources),
                "prompts": len(conn.prompts),
            }
            for conn in self._mcp_client_manager.list_connections()
        ]

    def _mcp_tools(self, server: str | None = None) -> list[dict[str, Any]]:
        """List available MCP tools, optionally filtered by server.

        Args:
            server: Optional server name to filter by

        Returns:
            List of tool info dicts with server, name, description.
        """
        tools = self._mcp_client_manager.get_all_tools()
        if server:
            tools = [t for t in tools if t.server_name == server]
        return [
            {
                "server": t.server_name,
                "name": t.name,
                "description": t.description,
            }
            for t in tools
        ]

    async def _mcp_cleanup(self) -> None:
        """Disconnect from all MCP servers. Called during session cleanup."""
        await self._mcp_client_manager.disconnect_all()

    def is_waiting(self) -> bool:
        """Check if a wait condition is active."""
        return self._wait_condition is not None

    def get_wait_condition(self) -> WaitCondition | None:
        """Get the active wait condition, if any."""
        return self._wait_condition

    def clear_wait_condition(self) -> None:
        """Clear the active wait condition (call when condition is satisfied)."""
        self._wait_condition = None

    def check_wait_condition(self) -> tuple[bool, str | None]:
        """Check if the active wait condition is satisfied.

        Returns:
            Tuple of (is_satisfied, prompt_to_inject).
            If satisfied, the prompt is the appropriate wake/timeout/failure prompt.
            If not satisfied, returns (False, None).
        """
        if not self._wait_condition:
            return False, None

        condition = self._wait_condition

        # Check timeout first
        if condition.is_timed_out():
            prompt = condition.timeout_prompt or f"Wait timed out after {condition.timeout}s"
            return True, prompt

        # Get nodes - support both ShellNode and LockNode
        nodes: list[ShellNode | LockNode] = []
        for node_id in condition.node_ids:
            node = self._context_graph.get_node(node_id)
            if isinstance(node, (ShellNode, LockNode)):
                nodes.append(node)

        if not nodes:
            # No valid nodes found - treat as satisfied with error
            return True, "Wait condition has no valid nodes."

        # Check for failures (ShellNode or LockNode)
        failed_nodes: list[ShellNode | LockNode] = []
        for n in nodes:
            if isinstance(n, ShellNode) and n.shell_status == ShellStatus.FAILED or isinstance(n, LockNode) and n.lock_status in (LockStatus.ERROR, LockStatus.TIMEOUT):
                failed_nodes.append(n)

        if failed_nodes and condition.failure_prompt:
            failed = failed_nodes[0]
            # Build prompt based on node type
            if isinstance(failed, ShellNode):
                prompt = condition.failure_prompt.format(
                    node=failed,
                    node_id=failed.node_id,
                    command=failed.full_command,
                    exit_code=failed.exit_code,
                    output=failed.output[:500] if failed.output else "",
                )
            else:  # LockNode
                prompt = condition.failure_prompt.format(
                    node=failed,
                    node_id=failed.node_id,
                    lockfile=failed.lockfile,
                    status=failed.lock_status.value,
                    error=failed.error_message or "",
                )
            return True, prompt

        # Check completion based on mode
        completed_nodes = [n for n in nodes if n.is_complete]

        if condition.mode == WaitMode.SINGLE or condition.mode == WaitMode.ALL:
            # Need all nodes to complete
            if len(completed_nodes) == len(nodes):
                if len(nodes) == 1:
                    node = completed_nodes[0]
                    prompt = self._format_wake_prompt(condition.wake_prompt, node)
                else:
                    prompt = condition.wake_prompt
                return True, prompt

        elif condition.mode == WaitMode.ANY:
            # Need any node to complete
            if completed_nodes:
                first_completed = completed_nodes[0]
                prompt = self._format_wake_prompt(condition.wake_prompt, first_completed)

                # Cancel others if requested
                if condition.cancel_others:
                    for node_id in condition.node_ids:
                        if node_id != first_completed.node_id:
                            # Cancel shells or locks
                            self.cancel_shell(node_id)
                            self.cancel_lock(node_id)

                return True, prompt

        elif condition.mode == WaitMode.MESSAGE:
            # Wait for incoming message
            if self._agent_manager and condition.agent_id:
                messages = self._agent_manager.get_messages(
                    condition.agent_id, status="pending"
                )
                if messages:
                    msg = messages[0]
                    # Mark as delivered
                    self._agent_manager.mark_message_delivered(msg.id)
                    # Format wake prompt with message info
                    prompt = condition.wake_prompt.format(
                        sender=msg.sender,
                        content=msg.content,
                        node_refs=msg.node_refs,
                        message_id=msg.id,
                    )
                    return True, prompt

        elif condition.mode == WaitMode.AGENT:
            # Wait for another agent to complete
            if self._agent_manager and condition.agent_id:
                from activecontext.agents.schema import AgentState

                entry = self._agent_manager.get_agent(condition.agent_id)
                if entry and entry.state in (AgentState.DONE, AgentState.TERMINATED):
                    prompt = condition.wake_prompt.format(
                        agent_id=entry.id,
                        state=entry.state.value,
                        task=entry.task,
                    )
                    return True, prompt

        return False, None

    def _format_wake_prompt(self, template: str, node: ShellNode | LockNode) -> str:
        """Format a wake prompt template with node-specific attributes."""
        if isinstance(node, ShellNode):
            return template.format(
                node=node,
                node_id=node.node_id,
                command=node.full_command,
                exit_code=node.exit_code,
                output=node.output[:500] if node.output else "",
            )
        else:  # LockNode
            return template.format(
                node=node,
                node_id=node.node_id,
                lockfile=node.lockfile,
                status=node.lock_status.value,
                error=node.error_message or "",
            )

    @property
    def session_id(self) -> str:
        return self._session_id

    def _capture_namespace_trace(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> NamespaceTrace:
        """Compute the trace between two namespace snapshots."""
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added = {k: type(after[k]).__name__ for k in after_keys - before_keys}
        deleted = list(before_keys - after_keys)

        changed = {}
        for k in before_keys & after_keys:
            if before[k] is not after[k]:
                changed[k] = f"{type(before[k]).__name__} -> {type(after[k]).__name__}"

        return NamespaceTrace(added=added, changed=changed, deleted=deleted)

    def _snapshot_namespace(self) -> dict[str, Any]:
        """Create a shallow snapshot of user-defined namespace entries."""
        # Exclude injected DSL functions and types
        excluded = {
            "NodeState", "TickFrequency", "NotificationLevel",
            "text", "group", "topic", "artifact", "markdown", "view",
            "link", "unlink",
            "checkpoint", "restore", "checkpoints", "branch",
            "ls", "show", "ls_permissions", "ls_imports", "ls_shell_permissions",
            "ls_website_permissions",
            "shell", "fetch", "done", "set_title", "notify",
            "wait", "wait_all", "wait_any",
            "lock_file", "lock_release",
            "work_on", "work_check", "work_update", "work_done", "work_list",
            "mcp_connect", "mcp_disconnect", "mcp_list", "mcp_tools",
        }
        return {
            k: v
            for k, v in self._namespace.items()
            if not k.startswith("__") and k not in excluded
        }

    async def _await_namespace_coroutines(self) -> None:
        """Await any coroutines stored in the namespace and replace with results.

        This handles cases like `result = shell("echo", ["hello"])` where
        exec() stores a coroutine in the namespace that needs to be awaited.
        """
        for key, value in list(self._namespace.items()):
            if asyncio.iscoroutine(value):
                self._namespace[key] = await value

    async def execute_statement(self, source: str) -> ExecutionResult:
        """Execute a Python statement and record it.

        Supports both Python syntax and XML-style tags:
            Python: v = view("main.py", tokens=2000)
            XML:    <view name="v" path="main.py" tokens="2000"/>

        If a PermissionDenied exception is raised and a permission_requester
        callback is configured, the user will be prompted for permission.
        On grant, the statement is retried.
        """
        statement_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        timestamp = time.time()

        # Convert XML to Python if needed
        original_source = source
        if is_xml_command(source):
            try:
                source = parse_xml_to_python(source)
            except ValueError as e:
                # Return error result for malformed XML
                return ExecutionResult(
                    execution_id=execution_id,
                    statement_id=statement_id,
                    status=ExecutionStatus.ERROR,
                    stdout="",
                    stderr="",
                    exception={
                        "type": "XMLParseError",
                        "message": str(e),
                        "traceback": f"Failed to parse XML: {original_source}",
                    },
                    state_trace=NamespaceTrace(added={}, changed={}, deleted=[]),
                    duration_ms=0.0,
                )

        # Record the statement (with original source for history)
        stmt = Statement(
            statement_id=statement_id,
            index=len(self._statements),
            source=source,
            timestamp=timestamp,
        )
        self._statements.append(stmt)

        # Execute with permission request retry loop
        max_permission_retries = 3  # Prevent infinite permission loops
        for _attempt in range(max_permission_retries):
            result = await self._execute_statement_inner(
                source, statement_id, execution_id
            )

            # Check if we got a PermissionDenied error
            if (
                result.status == ExecutionStatus.ERROR
                and result.exception
                and result.exception.get("type") == "PermissionDenied"
            ):
                perm_info = result.exception.get("_permission_info")
                if perm_info:
                    perm_path, perm_mode, perm_original = perm_info

                    # Try to request permission if requester is available
                    if self._permission_requester and self._permission_manager:
                        # Request permission from user via ACP
                        granted, persist = await self._permission_requester(
                            self._session_id, perm_path, perm_mode
                        )

                        if granted:
                            # User granted permission
                            if persist:
                                # "Allow always" - write to config file
                                write_permission_to_config(
                                    Path(self._cwd), perm_original, perm_mode
                                )
                                # Reload config to pick up new rule
                                from activecontext.config import load_config

                                config = load_config(session_root=self._cwd)
                                self._permission_manager.reload(config.sandbox)
                            else:
                                # "Allow once" - grant temporary access
                                self._permission_manager.grant_temporary(perm_path, perm_mode)

                            # Retry the statement with new permission
                            execution_id = str(uuid.uuid4())  # New execution ID for retry
                            continue

                    # No requester or user denied - convert to PermissionError for LLM
                    # The LLM shouldn't see the internal PermissionDenied type
                    result = ExecutionResult(
                        execution_id=result.execution_id,
                        statement_id=result.statement_id,
                        status=result.status,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        exception={
                            "type": "PermissionError",
                            "message": f"Access denied: {perm_mode} access to '{perm_path}'",
                            "traceback": result.exception.get("traceback", ""),
                        },
                        state_trace=result.state_trace,
                        duration_ms=result.duration_ms,
                    )

            # Check if we got an ImportDenied error
            if (
                result.status == ExecutionStatus.ERROR
                and result.exception
                and result.exception.get("type") == "ImportDenied"
            ):
                import_info = result.exception.get("_import_info")
                if import_info:
                    module, top_level = import_info

                    # Try to request permission if requester is available
                    if self._import_permission_requester and self._import_guard:
                        # Request permission from user via ACP
                        granted, persist, include_submodules = await self._import_permission_requester(
                            self._session_id, module
                        )

                        if granted:
                            # User granted permission
                            if persist:
                                # "Allow always" - write to config file
                                write_import_to_config(
                                    Path(self._cwd), top_level, include_submodules
                                )
                                # Reload config to pick up new rule
                                from activecontext.config import load_config

                                config = load_config(session_root=self._cwd)
                                self._import_guard.reload(config.sandbox.imports)
                            else:
                                # "Allow once" - grant temporary access
                                self._import_guard.grant_temporary(top_level, include_submodules)

                            # Retry the statement with new permission
                            execution_id = str(uuid.uuid4())  # New execution ID for retry
                            continue

                    # No requester or user denied - convert to ImportError for LLM
                    # The LLM shouldn't see the internal ImportDenied type
                    result = ExecutionResult(
                        execution_id=result.execution_id,
                        statement_id=result.statement_id,
                        status=result.status,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        exception={
                            "type": "ImportError",
                            "message": f"Import denied: '{module}' is not in the allowed modules whitelist",
                            "traceback": result.exception.get("traceback", ""),
                        },
                        state_trace=result.state_trace,
                        duration_ms=result.duration_ms,
                    )

            # No permission retry needed or granted, return result
            return result

        # Max retries exceeded - should not normally happen
        return result

    async def _execute_statement_inner(
        self, source: str, statement_id: str, execution_id: str
    ) -> ExecutionResult:
        """Inner execution logic for a single statement attempt."""
        # Capture namespace before
        ns_before = self._snapshot_namespace()

        # Execute with output capture
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        started_at = time.time()
        status = ExecutionStatus.OK
        exception_info: dict[str, Any] | None = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Use exec for statements, eval for expressions
                try:
                    # Try as expression first
                    result = eval(source, self._namespace)
                    # Handle coroutines (e.g., from shell())
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # Fall back to exec for statements
                    exec(source, self._namespace)
                    # After exec, check for coroutines in new namespace entries
                    await self._await_namespace_coroutines()
        except PermissionDenied as e:
            # Special handling for permission denied - include metadata for retry
            status = ExecutionStatus.ERROR
            exception_info = {
                "type": "PermissionDenied",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "_permission_info": (e.path, e.mode, e.original_path),
            }
        except ImportDenied as e:
            # Special handling for import denied - include metadata for retry
            status = ExecutionStatus.ERROR
            exception_info = {
                "type": "ImportDenied",
                "message": str(e),
                "traceback": traceback.format_exc(),
                "_import_info": (e.module, e.top_level),
            }
        except Exception as e:
            status = ExecutionStatus.ERROR
            exception_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

        ended_at = time.time()

        # Capture outputs (truncated)
        stdout_val = stdout_capture.getvalue()[: self._max_stdout]
        stderr_val = stderr_capture.getvalue()[: self._max_stderr]

        # Compute namespace diff
        ns_after = self._snapshot_namespace()
        state_trace = self._capture_namespace_trace(ns_before, ns_after)

        # Record execution
        record = _ExecutionRecord(
            execution_id=execution_id,
            statement_id=statement_id,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            stdout=stdout_val,
            stderr=stderr_val,
            exception=exception_info,
            state_trace=state_trace,
        )
        self._executions.setdefault(statement_id, []).append(record)

        return ExecutionResult(
            execution_id=execution_id,
            statement_id=statement_id,
            status=status,
            stdout=stdout_val,
            stderr=stderr_val,
            exception=exception_info,
            state_trace=state_trace,
            duration_ms=(ended_at - started_at) * 1000,
        )

    async def replay_from(self, statement_index: int) -> AsyncIterator[ExecutionResult]:
        """Re-execute statements from a given index."""
        if statement_index < 0 or statement_index >= len(self._statements):
            return

        # Reset namespace and context
        self._namespace.clear()
        self._context_graph.clear()
        self._setup_namespace()

        # Replay statements from start to get to clean state, then from index
        for stmt in self._statements[:statement_index]:
            # Execute silently to rebuild state
            await self.execute_statement(stmt.source)

        # Now replay from index, yielding results
        for stmt in self._statements[statement_index:]:
            result = await self.execute_statement(stmt.source)
            yield result

    def get_statements(self) -> list[Statement]:
        """Get all statements in the timeline."""
        return list(self._statements)

    def get_namespace(self) -> dict[str, Any]:
        """Get current Python namespace snapshot."""
        return self._snapshot_namespace()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all context objects from the graph."""
        return {node.node_id: node for node in self._context_graph}

    def get_context_graph(self) -> ContextGraph:
        """Get the context graph."""
        return self._context_graph

    @property
    def context_graph(self) -> ContextGraph:
        """The context graph (DAG of context nodes)."""
        return self._context_graph

    @property
    def current_group_id(self) -> str | None:
        """Current group ID for automatic node linking.

        When set, nodes created by DSL functions (view(), group(), etc.)
        will automatically be linked as children of this group.
        """
        return self._current_group_id

    def set_current_group(self, group_id: str | None) -> None:
        """Set the current group for automatic node linking.

        Args:
            group_id: Group ID to use as parent for new nodes, or None to clear.
        """
        self._current_group_id = group_id

    def clear_current_group(self) -> None:
        """Clear the current group (nodes will be added as roots)."""
        self._current_group_id = None

    @property
    def permission_manager(self) -> PermissionManager | None:
        """The permission manager for file access control."""
        return self._permission_manager

    def set_permission_manager(self, permission_manager: PermissionManager | None) -> None:
        """Set or update the permission manager.

        Updates the namespace to use the new permission manager's safe_open.

        Args:
            permission_manager: New permission manager (or None to disable).
        """
        self._permission_manager = permission_manager
        # Rebuild namespace to update the open() wrapper
        self._setup_namespace()

    @property
    def import_guard(self) -> ImportGuard | None:
        """The import guard for module whitelist control."""
        return self._import_guard

    def set_import_guard(self, import_guard: ImportGuard | None) -> None:
        """Set or update the import guard.

        Updates the namespace to use the new import guard's safe_import.

        Args:
            import_guard: New import guard (or None to allow all imports).
        """
        self._import_guard = import_guard
        # Rebuild namespace to update the __import__ wrapper
        self._setup_namespace()

    @property
    def shell_permission_manager(self) -> ShellPermissionManager | None:
        """The shell permission manager for command access control."""
        return self._shell_permission_manager

    def set_shell_permission_manager(
        self, shell_permission_manager: ShellPermissionManager | None
    ) -> None:
        """Set or update the shell permission manager.

        Args:
            shell_permission_manager: New shell permission manager (or None to disable).
        """
        self._shell_permission_manager = shell_permission_manager
