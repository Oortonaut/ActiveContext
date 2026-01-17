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
    TopicNode,
    ViewNode,
)
from activecontext.context.state import NodeState, TickFrequency
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
    ExecutionResult,
    ExecutionStatus,
    NamespaceDiff,
    Statement,
)
from activecontext.session.xml_parser import is_xml_command, parse_xml_to_python

if TYPE_CHECKING:
    from collections.abc import Callable

    from activecontext.terminal.protocol import TerminalExecutor

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
    state_diff: NamespaceDiff


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

        # Controlled Python namespace
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

        # Legacy: tracked context objects for backward compatibility
        self._context_objects: dict[str, Any] = {}

        # Max output capture per statement
        self._max_stdout = 50000
        self._max_stderr = 10000

        # Done signal from agent
        self._done_called = False
        self._done_message: str | None = None

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

        self._namespace = {
            "__builtins__": safe_builtins,
            "__name__": "__activecontext__",
            "__session_id__": self._session_id,
            # Type enums for LLM use
            "NodeState": NodeState,
            "TickFrequency": TickFrequency,
            # Context node constructors
            "view": self._make_view_node,
            "group": self._make_group_node,
            "topic": self._make_topic_node,
            "artifact": self._make_artifact_node,
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
        }

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

    def _make_view_node(
        self,
        path: str,
        *,
        pos: str = "1:0",
        tokens: int = 2000,
        state: NodeState = NodeState.ALL,
        mode: str = "paused",
        parent: ContextNode | str | None = None,
    ) -> ViewNode:
        """Create a ViewNode and add to the context graph.

        Args:
            path: File path relative to session cwd
            pos: Start position as "line:col" (1-indexed)
            tokens: Token budget for rendering
            state: Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL)
            mode: "paused" or "running"
            parent: Optional parent node or node ID

        Returns:
            The created ViewNode
        """
        node = ViewNode(
            path=path,
            pos=pos,
            tokens=tokens,
            state=state,
            mode=mode,
        )

        # Add to graph
        self._context_graph.add_node(node)

        # Link to parent if provided
        if parent:
            parent_id = parent.node_id if isinstance(parent, ContextNode) else parent
            self._context_graph.link(node.node_id, parent_id)

        # Legacy compatibility
        self._context_objects[node.node_id] = node
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
            parent: Optional parent node or node ID

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

        # Link to parent if provided
        if parent:
            parent_id = parent.node_id if isinstance(parent, ContextNode) else parent
            self._context_graph.link(node.node_id, parent_id)

        # Link members as children of this group
        for member in members:
            if isinstance(member, ContextNode):
                member_id = member.node_id
            else:
                # member is already a node ID string
                member_id = member
            
            self._context_graph.link(member_id, node.node_id)

        # Legacy compatibility
        self._context_objects[node.node_id] = node
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
            parent: Optional parent node or node ID

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

        # Link to parent if provided
        if parent:
            parent_id = parent.node_id if isinstance(parent, ContextNode) else parent
            self._context_graph.link(node.node_id, parent_id)

        # Legacy compatibility
        self._context_objects[node.node_id] = node
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
            parent: Optional parent node or node ID

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

        # Link to parent if provided
        if parent:
            parent_id = parent.node_id if isinstance(parent, ContextNode) else parent
            self._context_graph.link(node.node_id, parent_id)

        # Legacy compatibility
        self._context_objects[node.node_id] = node
        return node

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
        return [obj.GetDigest() for obj in self._context_objects.values()]

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
    ) -> Any:
        """Execute a shell command with permission checking.

        Returns a coroutine that must be awaited by execute_statement.
        If a shell_permission_manager is configured, the command will be
        checked against the permission rules before execution.

        Args:
            command: The command to execute (e.g., "pytest", "git").
            args: Optional list of arguments (e.g., ["tests/", "-v"]).
            cwd: Working directory. If None, uses session cwd.
            env: Additional environment variables.
            timeout: Timeout in seconds (default: 30). None for no timeout.

        Returns:
            Coroutine that resolves to ShellResult.
        """
        return self._shell_with_permission(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

    async def _shell_with_permission(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = 30.0,
    ) -> Any:
        """Execute shell command with permission check and request flow.

        Args:
            command: The command to execute.
            args: Optional list of arguments.
            cwd: Working directory.
            env: Additional environment variables.
            timeout: Timeout in seconds.

        Returns:
            ShellResult from execution, or error result if denied.
        """
        from activecontext.terminal.result import ShellResult

        # Check permission if manager is configured
        if self._shell_permission_manager:
            if not self._shell_permission_manager.check_access(command, args):
                # Permission denied - try to request
                if self._shell_permission_requester:
                    granted, persist = await self._shell_permission_requester(
                        self._session_id, command, args
                    )

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
                        # Denied - return error result
                        full_cmd = f"{command} {' '.join(args or [])}"
                        return ShellResult(
                            command=full_cmd,
                            exit_code=126,  # Permission denied exit code
                            output=f"Shell command denied by sandbox policy: {full_cmd}",
                            truncated=False,
                            status="error",
                            signal=None,
                            duration_ms=0,
                        )
                else:
                    # No requester available - return error result
                    full_cmd = f"{command} {' '.join(args or [])}"
                    return ShellResult(
                        command=full_cmd,
                        exit_code=126,
                        output=f"Shell command denied by sandbox policy: {full_cmd}",
                        truncated=False,
                        status="error",
                        signal=None,
                        duration_ms=0,
                    )

        # Permission granted (or no manager) - execute
        return await self._terminal_executor.execute(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )

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

    @property
    def session_id(self) -> str:
        return self._session_id

    def _capture_namespace_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> NamespaceDiff:
        """Compute the diff between two namespace snapshots."""
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added = {k: type(after[k]).__name__ for k in after_keys - before_keys}
        deleted = list(before_keys - after_keys)

        changed = {}
        for k in before_keys & after_keys:
            if before[k] is not after[k]:
                changed[k] = f"{type(before[k]).__name__} -> {type(after[k]).__name__}"

        return NamespaceDiff(added=added, changed=changed, deleted=deleted)

    def _snapshot_namespace(self) -> dict[str, Any]:
        """Create a shallow snapshot of user-defined namespace entries."""
        # Exclude injected DSL functions and types
        excluded = {
            "NodeState", "TickFrequency",
            "view", "group", "topic", "artifact",
            "link", "unlink",
            "checkpoint", "restore", "checkpoints", "branch",
            "ls", "show", "ls_permissions", "ls_imports", "ls_shell_permissions",
            "ls_website_permissions",
            "shell", "fetch", "done",
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
                    state_diff=NamespaceDiff(added={}, changed={}, deleted=[]),
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
                        state_diff=result.state_diff,
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
                        state_diff=result.state_diff,
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
        state_diff = self._capture_namespace_diff(ns_before, ns_after)

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
            state_diff=state_diff,
        )
        self._executions.setdefault(statement_id, []).append(record)

        return ExecutionResult(
            execution_id=execution_id,
            statement_id=statement_id,
            status=status,
            stdout=stdout_val,
            stderr=stderr_val,
            exception=exception_info,
            state_diff=state_diff,
            duration_ms=(ended_at - started_at) * 1000,
        )

    async def replay_from(self, statement_index: int) -> AsyncIterator[ExecutionResult]:
        """Re-execute statements from a given index."""
        if statement_index < 0 or statement_index >= len(self._statements):
            return

        # Reset namespace and context
        self._namespace.clear()
        self._context_objects.clear()
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
        """Get all tracked context objects (legacy compatibility)."""
        return dict(self._context_objects)

    def get_context_graph(self) -> ContextGraph:
        """Get the context graph."""
        return self._context_graph

    @property
    def context_graph(self) -> ContextGraph:
        """The context graph (DAG of context nodes)."""
        return self._context_graph

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
