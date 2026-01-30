"""Timeline: wrapper around StatementLog and PythonExec.

The Timeline is the canonical history of executed Python statements
for a session. It manages:
- Statement recording and indexing
- Python namespace execution
- Replay/re-execution from any point
"""

from __future__ import annotations

import ast
import asyncio
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from types import FunctionType

# Module-level lock for stdout/stderr redirection.
# redirect_stdout is NOT async-safe: when multiple async tasks use it concurrently,
# they corrupt each other's contexts because sys.stdout is a global.
# This lock ensures only one task at a time can capture output.
_stdout_redirect_lock = asyncio.Lock()
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
    ShellNode,
    ShellStatus,
    TextNode,
    TopicNode,
)
from activecontext.context.state import Expansion, NotificationLevel, TickFrequency
from activecontext.context.view import NodeView
from activecontext.session.agent_spawner import AgentSpawner
from activecontext.session.lock_manager import LockManager
from activecontext.session.mcp_integration import MCPIntegration
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
from activecontext.session.shell_manager import ShellManager
from activecontext.session.work_coordinator import WorkCoordinator
from activecontext.session.xml_parser import is_xml_command, parse_xml_to_python

if TYPE_CHECKING:
    from collections.abc import Callable

    from activecontext.agents.manager import AgentManager
    from activecontext.config.schema import FileWatchConfig, MCPConfig
    from activecontext.context.buffer import TextBuffer
    from activecontext.context.view import ChoiceView, LoopView, SequenceView, StateView
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
    WebsitePermissionRequester = Callable[[str, str, str], "asyncio.Future[tuple[bool, bool]]"]

    # Type for import permission requester callback:
    # async (session_id, module) -> (granted, persist, include_submodules)
    ImportPermissionRequester = Callable[[str, str], "asyncio.Future[tuple[bool, bool, bool]]"]


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


class ScriptNamespace(dict[str, Any]):
    """Dict subclass that falls back to graph/view lookup for node IDs.

    Allows direct access to nodes in the DSL namespace without explicit assignment.
    Node IDs have the format {node_type}_{seq} (e.g., 'text_1', 'group_2').

    Returns NodeView wrappers for nodes to enable view-based state management.
    User-defined variables take precedence over node lookups.

    Lookup order: namespace → views dict → graph (by node_id) → MCP nodes (by server_name) → KeyError
    """

    def __init__(
        self,
        graph_getter: Callable[[], ContextGraph | None],
        views_getter: Callable[[], dict[str, NodeView]],
        mcp_nodes_getter: Callable[[], dict[str, Any]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._graph_getter = graph_getter
        self._views_getter = views_getter
        self._mcp_nodes_getter = mcp_nodes_getter

    def __getitem__(self, key: str) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            # First check views dict by key directly
            views = self._views_getter()
            if key in views:
                return views[key]

            # Fall back to graph lookup by node_id
            graph = self._graph_getter()
            if graph:
                node = graph.get_node(key)
                if node:
                    # Check if view already exists for this node's actual node_id
                    if node.node_id in views:
                        return views[node.node_id]
                    # Create new view and store by node_id
                    view = NodeView(node)
                    views[node.node_id] = view
                    return view

            # Fall back to MCP server node lookup by server_name
            if self._mcp_nodes_getter is not None:
                mcp_nodes = self._mcp_nodes_getter()
                if key in mcp_nodes:
                    node = mcp_nodes[key]
                    if node.node_id in views:
                        return views[node.node_id]
                    view = NodeView(node)
                    views[node.node_id] = view
                    return view

            raise


class NodeLookup:
    """Dict-like accessor for node lookup by name with fuzzy matching.

    Supports multiple lookup patterns:
    - Exact variable name in namespace
    - Exact node_id (e.g., 'text_1', 'group_2')
    - Partial match on path (for TextNodes)
    - Partial match on title
    - Fuzzy matching on any of the above

    Returns NodeView wrappers for nodes.

    Usage:
        nodes["my_view"]      # Exact or fuzzy lookup
        nodes.my_view         # Attribute access
        get("partial")        # DSL function
    """

    __slots__ = ("_namespace_getter", "_graph_getter", "_views_getter")

    def __init__(
        self,
        namespace_getter: Callable[[], dict[str, Any]],
        graph_getter: Callable[[], ContextGraph | None],
        views_getter: Callable[[], dict[str, NodeView]],
    ) -> None:
        self._namespace_getter = namespace_getter
        self._graph_getter = graph_getter
        self._views_getter = views_getter

    def _get_or_create_view(self, node: ContextNode) -> NodeView:
        """Get existing view or create new one for a node."""
        views = self._views_getter()
        if node.node_id in views:
            return views[node.node_id]
        view = NodeView(node)
        views[node.node_id] = view
        return view

    def _score_match(self, pattern: str, candidate: str) -> float:
        """Score how well a pattern matches a candidate string.

        Returns:
            Score from 0.0 (no match) to 1.0 (exact match).
            Partial matches return values between 0.0 and 1.0.
        """
        if not pattern or not candidate:
            return 0.0

        pattern_lower = pattern.lower()
        candidate_lower = candidate.lower()

        # Exact match
        if pattern_lower == candidate_lower:
            return 1.0

        # Candidate contains pattern as substring
        if pattern_lower in candidate_lower:
            # Higher score for matches at start
            if candidate_lower.startswith(pattern_lower):
                return 0.9
            # Score based on pattern length relative to candidate
            return 0.5 + (len(pattern) / len(candidate)) * 0.3

        # Pattern is substring of candidate (partial match)
        if candidate_lower in pattern_lower:
            return 0.3

        return 0.0

    def _find_best_match(self, name: str) -> NodeView | None:
        """Find the best matching node for the given name.

        Lookup order:
        1. Exact match in namespace (variable name)
        2. Exact match by node_id
        3. Best fuzzy match considering:
           - Variable names
           - Node IDs
           - File paths (for TextNodes)
           - Titles

        Returns:
            NodeView for best match, or None if no match found.
        """
        namespace = self._namespace_getter()
        graph = self._graph_getter()
        views = self._views_getter()

        if not graph:
            return None

        # 1. Exact match in namespace
        if name in namespace:
            value = namespace[name]
            if isinstance(value, NodeView):
                return value
            if isinstance(value, ContextNode):
                return self._get_or_create_view(value)

        # 2. Exact match by node_id
        node = graph.get_node(name)
        if node:
            return self._get_or_create_view(node)

        # 3. Exact match in views
        if name in views:
            return views[name]

        # 4. Fuzzy matching - collect all candidates with scores
        candidates: list[tuple[float, str, NodeView | ContextNode]] = []

        # Check namespace variables that are nodes
        for var_name, value in namespace.items():
            if var_name.startswith("_"):
                continue
            if isinstance(value, NodeView):
                score = self._score_match(name, var_name)
                if score > 0:
                    candidates.append((score, var_name, value))
            elif isinstance(value, ContextNode):
                score = self._score_match(name, var_name)
                if score > 0:
                    candidates.append((score, var_name, value))

        # Check all nodes in graph
        for node in graph:
            # Score by node_id
            score = self._score_match(name, node.node_id)
            if score > 0:
                candidates.append((score, node.node_id, node))

            # Score by title
            if node.title:
                title_score = self._score_match(name, node.title)
                if title_score > 0:
                    candidates.append((title_score, node.title, node))

            # Score by path (for TextNodes)
            if isinstance(node, TextNode) and node.path:
                path = node.path
                # Match against full path
                path_score = self._score_match(name, path)
                if path_score > 0:
                    candidates.append((path_score, path, node))

                # Also match against filename only
                from pathlib import Path as PathLib

                filename = PathLib(path).name
                filename_score = self._score_match(name, filename)
                if filename_score > 0:
                    candidates.append((filename_score, filename, node))

        # Return best match if any
        if candidates:
            # Sort by score descending, then by match text length (prefer shorter)
            candidates.sort(key=lambda x: (-x[0], len(x[1])))
            best = candidates[0][2]
            if isinstance(best, NodeView):
                return best
            return self._get_or_create_view(best)

        return None

    def get(self, name: str) -> NodeView | None:
        """Look up a node by name with fuzzy matching.

        Args:
            name: Name to search for (variable name, node_id, path, or title)

        Returns:
            NodeView for matching node, or None if no match found.

        Raises:
            TypeError: If name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError(f"get() requires a string argument, got {type(name).__name__}")
        return self._find_best_match(name)

    def __getitem__(self, name: str) -> NodeView:
        """Look up a node by name, raising KeyError if not found.

        Args:
            name: Name to search for

        Returns:
            NodeView for matching node

        Raises:
            KeyError: If no matching node is found
        """
        result = self._find_best_match(name)
        if result is None:
            raise KeyError(f"No node found matching '{name}'")
        return result

    def __getattr__(self, name: str) -> NodeView:
        """Attribute access for node lookup.

        Args:
            name: Attribute name (node name to search for)

        Returns:
            NodeView for matching node

        Raises:
            AttributeError: If no matching node is found
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        result = self._find_best_match(name)
        if result is None:
            raise AttributeError(f"No node found matching '{name}'")
        return result

    def __contains__(self, name: str) -> bool:
        """Check if a node matching the name exists."""
        return self._find_best_match(name) is not None

    def keys(self) -> list[str]:
        """Return all available node identifiers."""
        identifiers: list[str] = []
        namespace = self._namespace_getter()
        graph = self._graph_getter()

        # Add namespace variable names that are nodes
        for var_name, value in namespace.items():
            if var_name.startswith("_"):
                continue
            if isinstance(value, (NodeView, ContextNode)):
                identifiers.append(var_name)

        # Add node_ids
        if graph:
            for node in graph:
                if node.node_id not in identifiers:
                    identifiers.append(node.node_id)

        return identifiers

    def values(self) -> list[NodeView]:
        """Return all available nodes as NodeViews."""
        graph = self._graph_getter()
        if not graph:
            return []
        return [self._get_or_create_view(node) for node in graph]

    def items(self) -> list[tuple[str, NodeView]]:
        """Return (node_id, NodeView) pairs for all nodes."""
        graph = self._graph_getter()
        if not graph:
            return []
        return [(node.node_id, self._get_or_create_view(node)) for node in graph]

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        graph = self._graph_getter()
        return len(graph) if graph else 0

    def __iter__(self) -> Any:
        """Iterate over node identifiers."""
        return iter(self.keys())

    def __repr__(self) -> str:
        return f"NodeLookup({len(self)} nodes)"


class Timeline:
    """Statement timeline with controlled Python execution.

    Each session has one Timeline that tracks all executed statements
    and maintains the Python namespace.
    """

    def __init__(
        self,
        session_id: str,
        context_graph: ContextGraph,
        cwd: str = ".",
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

        # Context graph (DAG of context nodes) - injected by Session
        self._context_graph = context_graph

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

        # Controlled Python namespace (created before MCP setup)
        self._namespace: dict[str, Any] = {}

        # View graph: node_id -> NodeView for rendering
        self._views: dict[str, NodeView] = {}

        # MCP integration manager (must be set before _setup_namespace)
        self._mcp_integration = MCPIntegration(
            mcp_config=mcp_config,
            context_graph=self._context_graph,
            namespace=self._namespace,
            fire_event=self.fire_event,
        )

        # Shell execution manager (must be set before _setup_namespace)
        self._shell_manager = ShellManager(
            context_graph=self._context_graph,
            terminal_executor=self._terminal_executor,
            shell_permission_manager=self._shell_permission_manager,
            shell_permission_requester=self._shell_permission_requester,
            session_id=session_id,
            cwd=cwd,
        )

        # File lock manager (must be set before _setup_namespace)
        self._lock_manager = LockManager(
            context_graph=self._context_graph,
            cwd=cwd,
        )

        # Agent spawner for multi-agent coordination (will be configured later)
        self._agent_spawner = AgentSpawner(
            agent_manager=None,  # Set later by AgentManager
            agent_id=None,  # Set later when spawned
            context_graph=self._context_graph,
            cwd=cwd,
        )

        # Work coordinator for multi-agent file coordination
        self._work_coordinator = WorkCoordinator(
            session_id=session_id,
            context_graph=self._context_graph,
            scratchpad_manager=scratchpad_manager,
        )

        # Set up namespace with DSL functions
        self._setup_namespace()

        # Max output capture per statement
        self._max_stdout = 50000
        self._max_stderr = 10000

        # Done signal from agent
        self._done_called = False
        self._done_message: str | None = None

        # Callback for setting session title (set by Session after creation)
        self._set_title_callback: Callable[[str], None] | None = None

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

        # Path resolver callback for @prompts/ and other prefixes (set by Session)
        # Returns (resolved_path, content_or_none)
        self._path_resolver: Callable[[str], tuple[str, str | None]] | None = None

        # Conversation delegation callback (set by Session for interact/connect DSL)
        # Signature: (handler, originator, pause_agent, forward_permissions) -> result
        self._delegate_conversation: Callable[..., Awaitable[Any]] | None = None

        # Conversation handle creation callback (set by Session for connect() DSL)
        # Signature: (handler, originator, forward_permissions) -> ConversationHandle
        self._create_conversation_handle: Callable[..., Any] | None = None

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

            self._file_watcher = FileWatcher(cwd=Path(self._cwd), poll_interval=float("inf"))
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

    @property
    def views(self) -> dict[str, NodeView]:
        """Get the view graph (node_id -> NodeView) for rendering."""
        return self._views

    def _setup_namespace(self) -> None:
        """Initialize the Python namespace with injected functions."""
        # Import builtins we want to expose
        import builtins

        safe_builtins = {
            name: getattr(builtins, name) for name in dir(builtins) if not name.startswith("_")
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

        self._namespace = ScriptNamespace(
            lambda: self._context_graph,
            lambda: self._views,
            lambda: self._mcp_integration._mcp_server_nodes,
            {
                "__builtins__": safe_builtins,
                "__name__": "__activecontext__",
                "__session_id__": self._session_id,
                # Type enums for LLM use
                "Expansion": Expansion,
                "TickFrequency": TickFrequency,
                "NotificationLevel": NotificationLevel,
                # Context node constructors
                "text": self._make_text_node,
                "group": self._make_group_node,
                "topic": self._make_topic_node,
                "artifact": self._make_artifact_node,
                "markdown": self._make_markdown_node,
                "view": self._make_view,
                "choice": self._make_choice_view,
                # Progression views
                "sequence": self._make_sequence_view,
                "loop_view": self._make_loop_view,
                "state_machine": self._make_state_machine,
                # DAG manipulation
                "link": self._link,
                "unlink": self._unlink,
                # Traversal control
                "hide": self._hide,
                "unhide": self._unhide,
                # Checkpointing
                "checkpoint": self._checkpoint,
                "restore": self._restore,
                "checkpoints": self._list_checkpoints,
                "branch": self._branch,
                # Utility functions
                "ls": self._ls_handles,
                "show": self._show_handle,
                # Node lookup
                "get": self._get_node,
                "nodes": self._create_node_lookup(),
                "ls_permissions": self._ls_permissions,
                "ls_imports": self._ls_imports,
                "ls_shell_permissions": self._ls_shell_permissions,
                "ls_website_permissions": self._ls_website_permissions,
                # Shell execution
                "shell": self._shell_manager.execute,
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
                # Conversation delegation
                "interact": self._interact,
                "connect": self._connect,
                # File locking
                "lock_file": self._lock_manager.acquire,
                "lock_release": self._lock_manager.release,
            },
        )

        # Add work coordination functions if scratchpad manager is available
        if self._scratchpad_manager:
            self._namespace.update(
                {
                    "work_on": self._work_coordinator.work_on,
                    "work_check": self._work_coordinator.work_check,
                    "work_update": self._work_coordinator.work_update,
                    "work_done": self._work_coordinator.work_done,
                    "work_list": self._work_coordinator.work_list,
                }
            )

        # Add MCP functions
        self._namespace.update(
            {
                "mcp_connect": self._mcp_integration.connect,
                "mcp_disconnect": self._mcp_integration.disconnect,
                "mcp_list": self._mcp_integration.list_connections,
                "mcp_tools": self._mcp_integration.list_tools,
            }
        )

        # Inject connected MCP server proxies into namespace
        self._namespace.update(self._mcp_integration.generate_namespace_bindings())

        # Update MCPIntegration's namespace reference (it was initialized with the
        # old dict before _setup_namespace replaced self._namespace)
        self._mcp_integration._namespace = self._namespace

    def _setup_agent_namespace(self) -> None:
        """Add agent functions to namespace when agent manager is available.

        Called by AgentManager after setting _agent_manager and _agent_id.
        """
        if self._agent_manager is None:
            return

        # Propagate agent manager and agent ID to AgentSpawner
        self._agent_spawner.set_agent_manager(self._agent_manager)
        if self._agent_id:
            self._agent_spawner.set_agent_id(self._agent_id)

        # Delegate agent DSL bindings to AgentSpawner
        self._agent_spawner.setup_namespace(self._namespace)

        # Add wait_message (manages Timeline wait state, so stays in Timeline)
        self._namespace["wait_message"] = self._wait_message

        # Add non-agent event system functions
        self._namespace.update(
            {
                # Event system
                "event_response": self._event_response,
                "wait": self._wait_event,
                "EventResponse": EventResponse,
                # File watching
                "wait_file_change": self._wait_file_change,
                "on_file_change": self._on_file_change,
            }
        )

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
        default_prompt = self._event_handlers.get(
            event_name,
            EventHandler(
                event_name=event_name,
                response=EventResponse.QUEUE,
                prompt_template=f"Event {event_name} occurred",
            ),
        ).prompt_template

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
        expansion: Expansion = Expansion.ALL,
        mode: str = "paused",
        parent: ContextNode | str | None = None,
    ) -> NodeView:
        """Create a TextNode and add to the context graph.

        Args:
            path: File path relative to session cwd
            pos: Start position as "line:col" (1-indexed)
            expansion: Rendering expansion (HEADER, CONTENT, INDEX, ALL)
            mode: "paused" or "running"
            parent: Optional parent node or node ID (defaults to current_group if set)

        Returns:
            NodeView wrapping the created TextNode
        """
        node = TextNode(
            path=path,
            pos=pos,
            expansion=expansion,
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
            parent_id = (
                effective_parent.node_id
                if isinstance(effective_parent, ContextNode)
                else effective_parent
            )
            self._context_graph.link(node.node_id, parent_id)

        # Register with file watcher for external change detection
        self._file_watcher.register_path(path, node.node_id)

        # Create and store NodeView
        view = NodeView(node, expand=expansion)
        self._views[node.node_id] = view
        return view

    def _make_group_node(
        self,
        *members: ContextNode | NodeView | str,
        expansion: Expansion = Expansion.CONTENT,
        mode: str = "paused",
        summary: str | None = None,
        parent: ContextNode | NodeView | str | None = None,
    ) -> NodeView:
        """Create a GroupNode that summarizes its members.

        Args:
            *members: Child nodes, views, or node IDs to include in the group
            expansion: Rendering expansion (HEADER, CONTENT, INDEX, ALL)
            mode: "paused" or "running"
            summary: Optional pre-computed summary text
            parent: Optional parent node, view, or node ID (defaults to current_group if set)

        Returns:
            NodeView wrapping the created GroupNode
        """
        node = GroupNode(
            expansion=expansion,
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
            if isinstance(effective_parent, (NodeView, ContextNode)):
                parent_id = effective_parent.node_id
            else:
                parent_id = effective_parent
            self._context_graph.link(node.node_id, parent_id)

        # Link members as children of this group
        for member in members:
            if isinstance(member, (NodeView, ContextNode)):
                member_id = member.node_id
            else:
                # member is already a node ID string
                member_id = member

            self._context_graph.link(member_id, node.node_id)

        # Create and store NodeView
        view = NodeView(node, expand=expansion)
        self._views[node.node_id] = view
        return view


    def _make_choice_view(
        self,
        *children: ContextNode | NodeView | str,
        selected: str | None = None,
        expansion: Expansion = Expansion.ALL,
        parent: ContextNode | NodeView | str | None = None,
    ) -> ChoiceView:
        """Create a ChoiceView wrapping a GroupNode for dropdown-like selection.

        Only the selected child's content is visible at a time, while other
        children remain hidden. Use ChoiceView.select() to switch selection.

        Args:
            *children: Child nodes, views, or node IDs to include as options
            selected: Node ID of the initially selected child (default: first child)
            expansion: Rendering expansion for the group
            parent: Optional parent node (defaults to current_group if set)

        Returns:
            ChoiceView wrapping the created GroupNode
        """
        from activecontext.context.view import ChoiceView

        # Create the underlying group
        group_view = self._make_group_node(
            *children, expansion=expansion, parent=parent
        )

        # Default to first child if no selection specified
        if selected is None and children:
            first = children[0]
            if isinstance(first, NodeView) or isinstance(first, ContextNode):
                selected = first.node_id
            else:
                selected = first  # Already a node ID string

        # Create ChoiceView wrapping the group
        choice_view = ChoiceView(group_view.node(), selected_id=selected, expand=expansion)
        self._views[group_view.node_id] = choice_view
        return choice_view

    def _make_sequence_view(
        self,
        *children: ContextNode | NodeView | str,
        expansion: Expansion = Expansion.ALL,
        parent: ContextNode | NodeView | str | None = None,
    ) -> SequenceView:
        """Create a SequenceView for ordered sequential progression.

        Agent works through steps in order. Current step is visible, others hidden.
        Tracks completion state per step.

        Args:
            *children: Child nodes, views, or node IDs representing steps
            expansion: Rendering expansion for the group
            parent: Optional parent node (defaults to current_group if set)

        Returns:
            SequenceView wrapping the created GroupNode

        Example:
            seq = sequence(step1, step2, step3)
            seq.advance()       # Mark current complete, move to next
            seq.back()          # Go back one step
            seq.mark_complete() # Mark current complete without advancing
            print(seq.progress) # "2/3"
        """
        from activecontext.context.view import SequenceView

        # Create the underlying group
        group_view = self._make_group_node(
            *children, expansion=expansion, parent=parent
        )

        # Create SequenceView wrapping the group (starts at first child)
        seq_view = SequenceView(group_view.node(), expand=expansion)
        self._views[group_view.node_id] = seq_view
        return seq_view

    def _make_loop_view(
        self,
        child: ContextNode | NodeView | str,
        max_iterations: int | None = None,
        expansion: Expansion = Expansion.ALL,
        parent: ContextNode | NodeView | str | None = None,
    ) -> LoopView:
        """Create a LoopView for iterative refinement.

        Agent iterates on a single prompt, accumulating state across iterations.
        Optional max_iterations limit.

        Args:
            child: The node, view, or node ID to iterate on
            max_iterations: Maximum iterations allowed (None = unlimited)
            expansion: Rendering expansion
            parent: Optional parent node

        Returns:
            LoopView wrapping the child node

        Example:
            loop = loop_view(review_prompt, max_iterations=5)
            loop.iterate(feedback="Add error handling")
            loop.iterate(feedback="Looks good!")
            loop.done()  # Exit early
            print(loop.iteration)  # Current iteration number
            print(loop.state)      # Accumulated state dict
        """
        from activecontext.context.view import LoopView

        # Resolve child to node
        if isinstance(child, str):
            node = self._context_graph.get_node(child)
            if node is None:
                raise ValueError(f"Unknown node ID: {child}")
        elif isinstance(child, NodeView):
            node = child.node()
        else:
            node = child

        # Link to parent if specified
        if parent is not None:
            parent_id = self._resolve_node_id(parent)
            self._context_graph.link(node.node_id, parent_id)
        elif self._current_group_id:
            self._context_graph.link(node.node_id, self._current_group_id)

        # Create LoopView wrapping the node
        loop_view = LoopView(node, max_iterations=max_iterations, expand=expansion)
        self._views[node.node_id] = loop_view
        return loop_view

    def _make_state_machine(
        self,
        *children: ContextNode | NodeView | str,
        states: dict[str, str] | None = None,
        transitions: dict[str, list[str]] | None = None,
        initial: str | None = None,
        expansion: Expansion = Expansion.ALL,
        parent: ContextNode | NodeView | str | None = None,
    ) -> StateView:
        """Create a StateView for state machine navigation.

        Agent navigates between named states following transition rules.
        Only current state's content is visible.

        Args:
            *children: Child nodes as states (optional, use with auto-generated states)
            states: Mapping of state names to child node IDs
            transitions: Mapping of state names to list of allowed next states
            initial: Initial state name (default: first state)
            expansion: Rendering expansion
            parent: Optional parent node

        Returns:
            StateView wrapping the created GroupNode

        Example:
            fsm = state_machine(
                states={"idle": idle_node.node_id, "working": work_node.node_id},
                transitions={
                    "idle": ["working"],
                    "working": ["idle"]
                },
                initial="idle"
            )
            fsm.transition("working")
            print(fsm.can_transition("idle"))  # True
        """
        from activecontext.context.view import StateView

        # If children provided without states dict, generate states from children
        if children and not states:
            states = {}
            for i, child in enumerate(children):
                if isinstance(child, str):
                    node = self._context_graph.get_node(child)
                    if node is None:
                        raise ValueError(f"Unknown node ID: {child}")
                    child_id = child
                elif isinstance(child, NodeView):
                    node = child.node()
                    child_id = node.node_id
                else:
                    node = child
                    child_id = node.node_id

                # Use title or node_id as state name
                state_name = getattr(node, "title", None) or f"state_{i}"
                states[state_name] = child_id

        # Create the underlying group with all state nodes as children
        if states:
            node_ids = list(states.values())
            group_view = self._make_group_node(
                *node_ids, expansion=expansion, parent=parent
            )
        else:
            # Empty state machine
            group_view = self._make_group_node(
                expansion=expansion, parent=parent
            )

        # Default transitions: allow any state to any state
        if transitions is None and states:
            all_states = list(states.keys())
            transitions = {s: [t for t in all_states if t != s] for s in all_states}

        # Create StateView wrapping the group
        state_view = StateView(
            group_view.node(),
            states=states or {},
            transitions=transitions or {},
            initial=initial,
            expand=expansion,
        )
        self._views[group_view.node_id] = state_view
        return state_view

    def _resolve_node_id(self, node_or_id: ContextNode | NodeView | str) -> str:
        """Resolve a node, view, or ID string to a node ID.

        Args:
            node_or_id: Node, view, or node ID string

        Returns:
            Node ID string
        """
        if isinstance(node_or_id, str):
            return node_or_id
        elif isinstance(node_or_id, NodeView):
            return str(node_or_id.node_id)
        else:
            return str(node_or_id.node_id)

    def _make_topic_node(
        self,
        title: str,
        *,
        status: str = "active",
        parent: ContextNode | NodeView | str | None = None,
    ) -> NodeView:
        """Create a TopicNode for conversation segmentation.

        Args:
            title: Short title for the topic
            status: "active", "resolved", or "deferred"
            parent: Optional parent node, view, or node ID (defaults to current_group if set)

        Returns:
            NodeView wrapping the created TopicNode
        """
        node = TopicNode(
            title=title,
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
            if isinstance(effective_parent, (NodeView, ContextNode)):
                parent_id = effective_parent.node_id
            else:
                parent_id = effective_parent
            self._context_graph.link(node.node_id, parent_id)

        # Create and store NodeView
        view = NodeView(node)
        self._views[node.node_id] = view
        return view

    def _make_artifact_node(
        self,
        artifact_type: str = "code",
        *,
        content: str = "",
        language: str | None = None,
        parent: ContextNode | NodeView | str | None = None,
    ) -> NodeView:
        """Create an ArtifactNode for code/output.

        Args:
            artifact_type: "code", "output", "error", or "file"
            content: The artifact content
            language: Programming language (for code)
            parent: Optional parent node, view, or node ID (defaults to current_group if set)

        Returns:
            NodeView wrapping the created ArtifactNode
        """
        node = ArtifactNode(
            artifact_type=artifact_type,
            content=content,
            language=language,
        )

        # Add to graph
        self._context_graph.add_node(node)

        # Determine parent: explicit > current_group > none
        effective_parent = parent
        if effective_parent is None and self._current_group_id:
            effective_parent = self._current_group_id

        # Link to parent if set
        if effective_parent:
            if isinstance(effective_parent, (NodeView, ContextNode)):
                parent_id = effective_parent.node_id
            else:
                parent_id = effective_parent
            self._context_graph.link(node.node_id, parent_id)

        # Create and store NodeView
        view = NodeView(node)
        self._views[node.node_id] = view
        return view

    def _make_markdown_node(
        self,
        path: str,
        *,
        content: str | None = None,
        expansion: Expansion = Expansion.ALL,
        parent: ContextNode | NodeView | str | None = None,
    ) -> NodeView:
        """Create a tree of TextNodes from a markdown file.

        Parses the markdown heading hierarchy into a tree of TextNodes,
        where each heading section is a separate node with line range references.

        Args:
            path: File path relative to session cwd
            content: Markdown content (if None, reads from path)
            expansion: Rendering expansion (HEADER, CONTENT, INDEX, ALL)
            parent: Optional parent node, view, or node ID (defaults to current_group if set)

        Returns:
            NodeView wrapping the root TextNode (children are accessible via children_ids)
        """
        import os

        from activecontext.context.buffer import TextBuffer
        from activecontext.context.markdown_parser import parse_markdown
        from activecontext.core.tokens import MediaType

        # Resolve path prefixes (e.g., @prompts/) to content via callback
        if content is None and self._path_resolver is not None:
            resolved_path, resolved_content = self._path_resolver(path)
            if resolved_content is not None:
                path = resolved_path
                content = resolved_content

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

        # At this point buffer is guaranteed to be non-None
        assert buffer is not None

        # Parse markdown to get heading sections
        buffer_content = "\n".join(buffer.lines)
        result = parse_markdown(buffer_content)

        if not result.sections:
            # No headings - create single TextNode for entire file
            node = TextNode(
                path=path,
                expansion=expansion,
                media_type=MediaType.MARKDOWN,
                buffer_id=buffer.buffer_id,
                start_line=1,
                end_line=len(buffer.lines),
            )
            self._context_graph.add_node(node)
            # Create and store NodeView
            view = NodeView(node, expand=expansion)
            self._views[node.node_id] = view
            return view

        # Create TextNode for each heading section
        all_nodes: list[TextNode] = []
        section_nodes: dict[int, TextNode] = {}  # section index -> node

        for i, section in enumerate(result.sections):
            node = TextNode(
                path=path,
                expansion=expansion,
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

        # Add all nodes to graph first and create views
        for node in all_nodes:
            self._context_graph.add_node(node)
            view = NodeView(node, expand=expansion)
            self._views[node.node_id] = view

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
            if isinstance(effective_parent, (NodeView, ContextNode)):
                parent_id = effective_parent.node_id
            else:
                parent_id = effective_parent
            self._context_graph.link(root.node_id, parent_id)

        # Return the root's NodeView
        return self._views[root.node_id]

    def _make_view(
        self,
        media_type: str,
        path: str,
        *,
        expansion: Expansion = Expansion.ALL,
        **kwargs: Any,
    ) -> NodeView:
        """Dispatcher for creating text views based on media type.

        Routes to text() or markdown() based on the media_type parameter.

        Args:
            media_type: "text" or "markdown"
            path: File path relative to session cwd
            expansion: Rendering expansion
            **kwargs: Additional arguments passed to underlying function

        Returns:
            NodeView wrapping the created node

        Raises:
            ValueError: If media_type is not recognized
        """
        if media_type == "markdown":
            return self._make_markdown_node(path, expansion=expansion, **kwargs)
        elif media_type == "text":
            return self._make_text_node(path, expansion=expansion, **kwargs)
        else:
            raise ValueError(f"Unknown media_type: {media_type}. Use 'text' or 'markdown'.")

    def _link(
        self,
        child: ContextNode | NodeView | str,
        parent: ContextNode | NodeView | str,
    ) -> bool:
        """Link a child node to a parent node.

        A node can have multiple parents (DAG structure).

        Args:
            child: Child node, view, or node ID
            parent: Parent node, view, or node ID

        Returns:
            True if link was created, False if failed
        """
        if isinstance(child, (NodeView, ContextNode)):
            child_id = child.node_id
        else:
            child_id = child

        if isinstance(parent, (NodeView, ContextNode)):
            parent_id = parent.node_id
        else:
            parent_id = parent

        return self._context_graph.link(child_id, parent_id)

    def _unlink(
        self,
        child: ContextNode | NodeView | str,
        parent: ContextNode | NodeView | str,
    ) -> bool:
        """Remove link between child and parent.

        Args:
            child: Child node, view, or node ID
            parent: Parent node, view, or node ID

        Returns:
            True if link was removed, False if failed
        """
        if isinstance(child, (NodeView, ContextNode)):
            child_id = child.node_id
        else:
            child_id = child

        if isinstance(parent, (NodeView, ContextNode)):
            parent_id = parent.node_id
        else:
            parent_id = parent

        return self._context_graph.unlink(child_id, parent_id)

    def _hide(self, *nodes: NodeView | ContextNode | str) -> int:
        """Hide nodes from projection traversal.

        Sets view.hide = True, excluding it from rendering while
        retaining all state for potential restoration via unhide().

        The previous expand state is stored in view.tags['_hidden_expand']
        so it can be restored later.

        Args:
            *nodes: One or more NodeViews, nodes, or node IDs to hide

        Returns:
            Number of nodes successfully hidden

        Example:
            hide(text_1)              # Hide single node
            hide(text_1, text_2)      # Hide multiple nodes
            hide("text_1", group_2)   # Mix of IDs and objects
        """
        count = 0
        for item in nodes:
            # Handle NodeView
            if isinstance(item, NodeView):
                view = item
                if view.hide:
                    continue  # Already hidden
                # Store previous expand for restoration
                view.tags["_hidden_expand"] = view.expand.value
                view.hide = True
                count += 1
            # Handle ContextNode
            elif isinstance(item, ContextNode):
                node = item
                # Create a view wrapper if needed - for now, find in namespace
                found_view = self._find_view_for_node(node)
                if found_view is not None:
                    if found_view.hide:
                        continue
                    found_view.tags["_hidden_expand"] = found_view.expand.value
                    found_view.hide = True
                    count += 1
            # Handle string (node ID)
            elif isinstance(item, str):
                resolved = self._context_graph.get_node(item)
                if resolved is None:
                    continue
                found_view = self._find_view_for_node(resolved)
                if found_view is not None:
                    if found_view.hide:
                        continue
                    found_view.tags["_hidden_expand"] = found_view.expand.value
                    found_view.hide = True
                    count += 1

        return count

    def _find_view_for_node(self, node: ContextNode) -> NodeView | None:
        """Find NodeView wrapping a ContextNode.

        Args:
            node: The ContextNode to find a view for

        Returns:
            NodeView wrapping the node, or None if not found
        """
        return self._views.get(node.node_id)

    def _unhide(
        self,
        *nodes: NodeView | ContextNode | str,
        expand: Expansion | None = None,
    ) -> int:
        """Restore hidden nodes to projection traversal.

        Reverses the effect of hide() by setting view.hide = False and
        restoring the previous expand state (or a specified one).

        Args:
            *nodes: One or more NodeViews, nodes, or node IDs to restore
            expand: Optional expand state to set. If None, restores to the
                   state before hide() was called, or DETAILS if unknown.

        Returns:
            Number of nodes successfully restored

        Example:
            unhide(text_1)                        # Restore to previous state
            unhide(text_1, text_2)                # Restore multiple
            unhide(text_1, expand=Expansion.CONTENT)  # Force specific expand
        """
        count = 0
        for item in nodes:
            view: NodeView | None = None

            # Handle NodeView
            if isinstance(item, NodeView):
                view = item
            # Handle ContextNode
            elif isinstance(item, ContextNode):
                view = self._find_view_for_node(item)
            # Handle string (node ID)
            elif isinstance(item, str):
                resolved = self._context_graph.get_node(item)
                if resolved is not None:
                    view = self._find_view_for_node(resolved)

            if view is None:
                continue

            # Skip if not hidden
            if not view.hide:
                continue

            # Determine target expand state
            if expand is not None:
                target = expand
            elif "_hidden_expand" in view.tags:
                # Restore to previous state
                target = Expansion(view.tags["_hidden_expand"])
                del view.tags["_hidden_expand"]
            else:
                # Default to ALL if no stored state
                target = Expansion.ALL

            # Unhide and set expand
            view.hide = False
            view.expand = target
            count += 1

        return count

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

    def _create_node_lookup(self) -> NodeLookup:
        """Create a NodeLookup accessor for the DSL namespace."""
        return NodeLookup(
            namespace_getter=lambda: self._namespace,
            graph_getter=lambda: self._context_graph,
            views_getter=lambda: self._views,
        )

    def _get_node(self, name: str) -> NodeView | None:
        """Look up a node by name with fuzzy matching.

        This is the DSL get() function for node lookup.

        Args:
            name: Name to search for. Can be:
                - Variable name in namespace (e.g., "my_view")
                - Node ID (e.g., "text_1", "group_2")
                - File path or partial path (for TextNodes)
                - Node title

        Returns:
            NodeView for matching node, or None if no match found.

        Examples:
            get("my_view")     # Lookup by variable name
            get("text_1")      # Lookup by node ID
            get("main.py")     # Lookup by file path
            get("main")        # Fuzzy match on path
        """
        lookup = self._create_node_lookup()
        return lookup.get(name)

    def _show_handle(self, obj: Any, *, lod: int | None = None) -> str:
        """Force render a handle (placeholder)."""
        digest = obj.GetDigest() if hasattr(obj, "GetDigest") else str(obj)
        return f"[{digest}]"

    def process_pending_shell_results(self) -> list[str]:
        """Process pending shell results and update nodes.

        Delegates to ShellManager for processing async shell results.

        Returns:
            List of node IDs that were updated.
        """
        return self._shell_manager.process_pending_results()

    # =========================================================================
    # File locking DSL functions
    # =========================================================================

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
        self._shell_manager.cancel_all()

        # Cancel all pending lock tasks and release held locks
        self._lock_manager.cancel_all()
        self._lock_manager.release_all()

        # Disconnect from all MCP servers
        await self._mcp_integration.cleanup()

        # Give cancelled tasks a chance to complete
        if self._shell_manager.has_pending_tasks() or self._lock_manager.has_pending_tasks():
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
                            write_website_permission_to_config(Path(self._cwd), url, method)
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
        node_ids = [n.node_id if isinstance(n, ContextNode) else n for n in nodes]
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
        node_ids = [n.node_id if isinstance(n, ContextNode) else n for n in nodes]
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

    # === Conversation Delegation DSL Functions ===

    def _interact(
        self,
        command: str | Any,  # str or ConversationHandler
        *args: str,
        originator: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an interactive command, delegating conversation control to a handler.

        This DSL function allows agents to attach the user directly to an
        interactive process (shell, debugger, menu, etc.) via conversation
        delegation. All input/output flows through MessageNodes in the context graph.

        This is a blocking operation - control returns when the handler completes.
        For non-blocking operation, use connect() instead.

        Args:
            command: Command to run (e.g., "/bin/bash") or ConversationHandler instance
            *args: Command arguments (if command is string)
            originator: Override originator identifier (default: derived from command)
            cwd: Working directory for shell commands (default: session cwd)
            **kwargs: Handler options

        Returns:
            Result from the interactive session

        Example:
            >>> result = interact("/bin/bash")
            >>> print(f"Shell exited with code {result['exit_code']}")

            >>> result = interact("gdb", "./myapp")

            >>> result = interact(my_custom_handler, originator="custom:handler")
        """
        if self._delegate_conversation is None:
            raise RuntimeError(
                "interact() requires Session integration. "
                "Ensure the session is properly initialized."
            )

        from activecontext.handlers import InteractiveShellHandler

        # Determine handler and originator
        if isinstance(command, str):
            # Shell command
            handler = InteractiveShellHandler(
                shell=command if command else None,
                args=args,
                cwd=cwd or self._cwd,
            )
            originator = originator or f"shell:{command or 'default'}"
        else:
            # Assume it's a ConversationHandler
            handler = command
            originator = originator or "conversation"

        # Run synchronously using asyncio
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context - create a task and wait
            # This happens when called from within an async function
            future = asyncio.ensure_future(
                self._delegate_conversation(
                    handler,
                    originator=originator,
                    pause_agent=kwargs.get("pause_agent", True),
                    forward_permissions=kwargs.get("forward_permissions", True),
                )
            )
            # Return the future/awaitable for the caller to await
            return future
        else:
            # Not in async context - use run_until_complete
            return loop.run_until_complete(
                self._delegate_conversation(
                    handler,
                    originator=originator,
                    pause_agent=kwargs.get("pause_agent", True),
                    forward_permissions=kwargs.get("forward_permissions", True),
                )
            )

    def _connect(
        self,
        command: str | Any,  # str or ConversationHandler
        *args: str,
        originator: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> Any:  # Returns ConversationHandle
        """Create a non-blocking conversation connection.

        Similar to interact(), but returns a ConversationHandle immediately
        for manual operation. The handler runs in the background, and the
        caller can check status, send input, and wait for completion.

        Args:
            command: Command to run (e.g., "/bin/bash") or ConversationHandler instance
            *args: Command arguments (if command is string)
            originator: Override originator identifier (default: derived from command)
            cwd: Working directory for shell commands (default: session cwd)
            **kwargs: Handler options

        Returns:
            ConversationHandle for manual operation

        Example:
            >>> handle = connect("/bin/bash")
            >>> await handle.start()
            >>> while not handle.is_done():
            ...     if handle.is_waiting():
            ...         prompt = handle.get_last_prompt_node()
            ...         await handle.send_input("ls -la")
            >>> result = await handle.wait()

            >>> # Run multiple shells concurrently
            >>> sh1 = connect("bash")
            >>> sh2 = connect("python")
            >>> await sh1.start()
            >>> await sh2.start()
            >>> # Coordinator can manage both...
        """
        if self._create_conversation_handle is None:
            raise RuntimeError(
                "connect() requires Session integration. "
                "Ensure the session is properly initialized."
            )

        from activecontext.handlers import InteractiveShellHandler

        # Determine handler and originator
        if isinstance(command, str):
            # Shell command
            handler = InteractiveShellHandler(
                shell=command if command else None,
                args=args,
                cwd=cwd or self._cwd,
            )
            originator = originator or f"shell:{command or 'default'}"
        else:
            # Assume it's a ConversationHandler
            handler = command
            originator = originator or "conversation"

        # Create handle via Session
        handle = self._create_conversation_handle(
            handler,
            originator=originator,
            forward_permissions=kwargs.get("forward_permissions", True),
        )

        # Start the handler in the background
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create task to start the handler
            asyncio.create_task(handle.start())
        else:
            # Not in async context - caller must await handle.start()
            pass

        return handle

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

    # =========================================================================
    # Multi-Agent DSL Functions
    # =========================================================================

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
            if (
                isinstance(n, ShellNode)
                and n.shell_status == ShellStatus.FAILED
                or isinstance(n, LockNode)
                and n.lock_status in (LockStatus.ERROR, LockStatus.TIMEOUT)
            ):
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
                            self._shell_manager.cancel(node_id)
                            self._lock_manager.cancel(node_id)

                return True, prompt

        elif condition.mode == WaitMode.MESSAGE:
            # Wait for incoming message
            if self._agent_manager and condition.agent_id:
                messages = self._agent_manager.get_messages(condition.agent_id, status="pending")
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

        elif condition.mode == WaitMode.PROGRESSION:
            # Wait for progression view (SequenceView/LoopView) to complete
            for node_id in condition.node_ids:
                view = self._views.get(node_id)
                if view is None:
                    continue

                # Check SequenceView completion
                if hasattr(view, "is_complete") and view.is_complete:
                    prompt = condition.wake_prompt.format(
                        node_id=node_id,
                        progress=getattr(view, "progress", ""),
                    )
                    return True, prompt

                # Check LoopView completion
                if hasattr(view, "is_done") and view.is_done:
                    prompt = condition.wake_prompt.format(
                        node_id=node_id,
                        iteration=getattr(view, "iteration", 0),
                        state=getattr(view, "state", {}),
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
            "Expansion",
            "TickFrequency",
            "NotificationLevel",
            "text",
            "group",
            "topic",
            "artifact",
            "markdown",
            "view",
            "choice",
            "sequence",
            "loop_view",
            "state_machine",
            "link",
            "unlink",
            "hide",
            "unhide",
            "checkpoint",
            "restore",
            "checkpoints",
            "branch",
            "ls",
            "show",
            "get",
            "nodes",
            "ls_permissions",
            "ls_imports",
            "ls_shell_permissions",
            "ls_website_permissions",
            "shell",
            "fetch",
            "done",
            "set_title",
            "notify",
            "wait",
            "wait_all",
            "wait_any",
            "lock_file",
            "lock_release",
            "work_on",
            "work_check",
            "work_update",
            "work_done",
            "work_list",
            "mcp_connect",
            "mcp_disconnect",
            "mcp_list",
            "mcp_tools",
            "connect",
            "interact",  # Conversation delegation DSL functions
        }
        return {
            k: v for k, v in self._namespace.items() if not k.startswith("__") and k not in excluded
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
            Python: v = view("main.py")
            XML:    <view name="v" path="main.py"/>

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
        result: ExecutionResult | None = None
        for _attempt in range(max_permission_retries):
            result = await self._execute_statement_inner(source, statement_id, execution_id)

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
                        (
                            granted,
                            persist,
                            include_submodules,
                        ) = await self._import_permission_requester(self._session_id, module)

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
        # result is guaranteed to be set since max_permission_retries > 0
        assert result is not None, "Permission retry loop must execute at least once"
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
        result = None
        was_expression = False

        # CO_COROUTINE flag indicates code contains top-level await
        CO_COROUTINE = 0x80

        try:
            # Phase 1: Compile and execute with stdout capture (short lock)
            # The lock ensures redirect_stdout is async-safe across concurrent tasks.
            # Use PyCF_ALLOW_TOP_LEVEL_AWAIT to support 'await' in DSL code.
            coro_to_await = None

            async with _stdout_redirect_lock:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Parse to check structure - detect multi-line blocks ending with expression
                    try:
                        tree = ast.parse(source)
                    except SyntaxError:
                        # If parsing fails, let the compile below handle the error
                        tree = None

                    # Check if we have multiple statements with final expression
                    # (like Python REPL behavior: print result of last expression)
                    final_expr_source = None
                    statements_source = None
                    if (
                        tree is not None
                        and len(tree.body) > 1
                        and isinstance(tree.body[-1], ast.Expr)
                    ):
                        # Extract final expression and preceding statements
                        final_expr = tree.body[-1]
                        statements = tree.body[:-1]

                        # Compile statements (all but last)
                        statements_tree = ast.Module(body=statements, type_ignores=[])
                        ast.fix_missing_locations(statements_tree)
                        statements_source = compile(
                            statements_tree,
                            "<dsl>",
                            "exec",
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                        )

                        # Compile final expression
                        expr_tree = ast.Expression(body=final_expr.value)
                        ast.fix_missing_locations(expr_tree)
                        final_expr_source = compile(
                            expr_tree,
                            "<dsl>",
                            "eval",
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                        )

                    if statements_source is not None and final_expr_source is not None:
                        # Check if statements contain await - if so, fall back to
                        # original exec() behavior since we can't easily await between
                        # executing statements and evaluating the final expression
                        if statements_source.co_flags & CO_COROUTINE:
                            # Async statements - use original exec() path
                            # This means the final expression result won't be captured,
                            # but at least the code will execute correctly
                            compiled = compile(
                                source,
                                "<dsl>",
                                "exec",
                                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                            )
                            func = FunctionType(compiled, self._namespace)
                            coro_to_await = func()
                            was_expression = False
                        else:
                            # Sync statements - execute them first
                            exec(statements_source, self._namespace)

                            # Evaluate final expression to capture result
                            if final_expr_source.co_flags & CO_COROUTINE:
                                result = eval(final_expr_source, self._namespace)
                                coro_to_await = result
                            else:
                                result = eval(final_expr_source, self._namespace)
                            was_expression = True
                    else:
                        # Original logic: try as single expression first
                        try:
                            compiled = compile(
                                source,
                                "<dsl>",
                                "eval",
                                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                            )
                            if compiled.co_flags & CO_COROUTINE:
                                # Expression contains await - eval returns coroutine
                                result = eval(compiled, self._namespace)
                                coro_to_await = result
                            else:
                                result = eval(compiled, self._namespace)
                            was_expression = True
                        except SyntaxError:
                            # Fall back to exec for statements
                            compiled = compile(
                                source,
                                "<dsl>",
                                "exec",
                                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                            )
                            if compiled.co_flags & CO_COROUTINE:
                                # Statement contains await - create function and call it
                                func = FunctionType(compiled, self._namespace)
                                coro_to_await = func()
                            else:
                                exec(compiled, self._namespace)
                            was_expression = False

            # Phase 2: Await any coroutine OUTSIDE the lock
            # Long-running operations (like mcp_connect) must not hold the lock.
            if coro_to_await is not None:
                result = await coro_to_await
            elif asyncio.iscoroutine(result):
                result = await result

            # Phase 3: After exec, handle namespace coroutines OUTSIDE the lock
            if not was_expression:
                await self._await_namespace_coroutines()

            # Phase 4: Print result with brief lock
            if result is not None:
                async with _stdout_redirect_lock:
                    with redirect_stdout(stdout_capture):
                        print(repr(result))

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
