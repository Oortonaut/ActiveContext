"""Timeline: wrapper around StatementLog and PythonExec.

The Timeline is the canonical history of executed Python statements
for a session. It manages:
- Statement recording and indexing
- Python namespace execution
- Replay/re-execution from any point
"""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    ArtifactNode,
    ContextNode,
    GroupNode,
    TopicNode,
    ViewNode,
)
from activecontext.session.protocols import (
    ExecutionResult,
    ExecutionStatus,
    NamespaceDiff,
    Statement,
)

if TYPE_CHECKING:
    pass


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
    ) -> None:
        self._session_id = session_id
        self._cwd = cwd
        self._statements: list[Statement] = []
        self._executions: dict[str, list[_ExecutionRecord]] = {}  # statement_id -> executions

        # Context graph (DAG of context nodes)
        self._context_graph = context_graph or ContextGraph()

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

        self._namespace = {
            "__builtins__": safe_builtins,
            "__name__": "__activecontext__",
            "__session_id__": self._session_id,
            # Context node constructors
            "view": self._make_view_node,
            "group": self._make_group_node,
            "topic": self._make_topic_node,
            "artifact": self._make_artifact_node,
            # DAG manipulation
            "link": self._link,
            "unlink": self._unlink,
            # Utility functions
            "ls": self._ls_handles,
            "show": self._show_handle,
            # Agent control
            "done": self._done,
        }

    def _make_view_node(
        self,
        path: str,
        *,
        pos: str = "1:0",
        tokens: int = 2000,
        lod: int = 0,
        mode: str = "paused",
        parent: ContextNode | str | None = None,
    ) -> ViewNode:
        """Create a ViewNode and add to the context graph.

        Args:
            path: File path relative to session cwd
            pos: Start position as "line:col" (1-indexed)
            tokens: Token budget for rendering
            lod: Level of detail (0=raw, 1=structured, 2=summary, 3=diff)
            mode: "paused" or "running"
            parent: Optional parent node or node ID

        Returns:
            The created ViewNode
        """
        node = ViewNode(
            path=path,
            pos=pos,
            tokens=tokens,
            lod=lod,
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
        *members: ContextNode,
        tokens: int = 500,
        lod: int = 1,
        mode: str = "paused",
        parent: ContextNode | str | None = None,
    ) -> GroupNode:
        """Create a GroupNode that summarizes its members.

        Args:
            *members: Child nodes to include in the group
            tokens: Token budget for summary
            lod: Level of detail
            mode: "paused" or "running"
            parent: Optional parent node or node ID

        Returns:
            The created GroupNode
        """
        node = GroupNode(
            tokens=tokens,
            lod=lod,
            mode=mode,
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
                self._context_graph.link(member.node_id, node.node_id)

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

    def _ls_handles(self) -> list[dict[str, Any]]:
        """List all context object handles with brief digests."""
        return [obj.GetDigest() for obj in self._context_objects.values()]

    def _show_handle(self, obj: Any, *, lod: int | None = None, tokens: int | None = None) -> str:
        """Force render a handle (placeholder)."""
        digest = obj.GetDigest() if hasattr(obj, "GetDigest") else str(obj)
        return f"[{digest}]"

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
        # Exclude injected DSL functions
        excluded = {"view", "group", "topic", "artifact", "link", "unlink", "ls", "show", "done"}
        return {
            k: v
            for k, v in self._namespace.items()
            if not k.startswith("__") and k not in excluded
        }

    async def execute_statement(self, source: str) -> ExecutionResult:
        """Execute a Python statement and record it."""
        statement_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        timestamp = time.time()

        # Record the statement
        stmt = Statement(
            statement_id=statement_id,
            index=len(self._statements),
            source=source,
            timestamp=timestamp,
        )
        self._statements.append(stmt)

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
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # Fall back to exec for statements
                    exec(source, self._namespace)
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
