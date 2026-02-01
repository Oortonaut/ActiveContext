"""TaskGraph Bridge -- Python API for task-graph MCP coordination.

Wraps task-graph MCP tool calls in a clean Python interface.
Gracefully no-ops when task-graph is not connected, so callers
don't need conditional logic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

_log = logging.getLogger("activecontext.coordination.task_bridge")

# Well-known server name for task-graph MCP
TASK_GRAPH_SERVER = "task-graph"


class MCPCaller(Protocol):
    """Protocol for making MCP tool calls.

    This abstracts over the actual MCP client implementation so the bridge
    can be tested with simple mocks and doesn't couple to MCPClientManager.
    """

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> Any: ...

    def is_connected(self, server_name: str) -> bool: ...


@dataclass
class TaskInfo:
    """Summary of a task from the task graph."""

    id: str
    title: str
    status: str
    priority: int = 5
    points: int = 0
    tags: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)


@dataclass
class WorkerInfo:
    """Summary of a connected worker."""

    worker_id: str
    tags: list[str] = field(default_factory=list)
    claimed_tasks: list[str] = field(default_factory=list)


class TaskGraphBridge:
    """Bridge to task-graph MCP server for work coordination.

    All methods gracefully no-op when the task-graph MCP server
    is not connected, returning empty results or False.

    Usage:
        bridge = TaskGraphBridge(mcp_caller, worker_id="my-agent")
        await bridge.connect()

        tasks = await bridge.list_ready_tasks()
        if tasks:
            await bridge.claim(tasks[0].id)
            # ... do work ...
            await bridge.complete(tasks[0].id)

        await bridge.disconnect()

    Legacy usage (MCPClientManager wrapper):
        bridge = TaskGraphBridge.from_mcp_manager(mcp_manager, agent_id="w-1")
    """

    SERVER_NAME = TASK_GRAPH_SERVER

    def __init__(
        self,
        mcp_caller: MCPCaller | None = None,
        worker_id: str = "",
        tags: list[str] | None = None,
    ) -> None:
        self._mcp = mcp_caller
        self._worker_id = worker_id
        self._tags = tags or []
        self._connected = False
        # Legacy state for WorkCoordinator integration
        self._current_task_id: str | None = None
        self._started_at: float | None = None

    # --- Class methods for backward compatibility ---

    @classmethod
    def from_mcp_manager(
        cls,
        mcp_client_manager: Any,
        agent_id: str | None = None,
    ) -> TaskGraphBridge:
        """Create a bridge from an MCPClientManager (legacy API).

        Args:
            mcp_client_manager: MCPClientManager instance.
            agent_id: Agent identifier for task-graph operations.

        Returns:
            Configured TaskGraphBridge.
        """
        adapter = _MCPManagerAdapter(mcp_client_manager)
        return cls(
            mcp_caller=adapter,
            worker_id=agent_id or "",
        )

    # --- Properties ---

    @property
    def available(self) -> bool:
        """Whether the task-graph MCP server is reachable."""
        if self._mcp is None:
            return False
        return self._mcp.is_connected(self.SERVER_NAME)

    @property
    def is_active(self) -> bool:
        """Whether the task-graph MCP server is reachable (legacy alias)."""
        return self.available

    @property
    def connected(self) -> bool:
        """Whether we've connected as a worker."""
        return self._connected and self.available

    @property
    def agent_id(self) -> str | None:
        """The agent ID used for task-graph operations."""
        return self._worker_id or None

    @agent_id.setter
    def agent_id(self, value: str | None) -> None:
        self._worker_id = value or ""

    @property
    def current_task_id(self) -> str | None:
        """The task ID currently being worked on."""
        return self._current_task_id

    # --- Internal call helper ---

    async def _call(self, tool: str, **kwargs: Any) -> Any:
        """Make a task-graph MCP call. Returns None if unavailable."""
        if not self.available or self._mcp is None:
            return None
        try:
            return await self._mcp.call_tool(self.SERVER_NAME, tool, kwargs)
        except Exception as e:
            _log.debug("TaskGraph call %s failed: %s", tool, e)
            return None

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call a task-graph tool, returning parsed result or None on failure.

        This is the legacy call path used by WorkCoordinator integration.
        It handles MCPToolResult parsing for the MCPManagerAdapter.
        """
        if not self.available or self._mcp is None:
            return None

        try:
            result = await self._mcp.call_tool(self.SERVER_NAME, tool_name, arguments)

            # Handle MCPToolResult from the adapter
            if hasattr(result, "is_error"):
                if result.is_error:
                    _log.warning(
                        "task-graph.%s failed: %s",
                        tool_name,
                        result.error_message or "unknown error",
                    )
                    return None

                # Extract text content from MCPToolResult
                for block in result.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        try:
                            return dict(json.loads(block["text"]))
                        except (json.JSONDecodeError, KeyError):
                            return {"text": block.get("text", "")}
                return {}

            # Handle dict result from direct MCPCaller
            if isinstance(result, dict):
                return result

            return {} if result is not None else None

        except Exception:
            _log.debug("task-graph.%s call failed", tool_name, exc_info=True)
            return None

    # --- Worker Lifecycle ---

    async def connect(self, force: bool = False) -> bool:
        """Connect as a worker. Returns True if successful."""
        result = await self._call(
            "connect",
            worker_id=self._worker_id,
            tags=self._tags,
            force=force,
        )
        if result is not None:
            self._connected = True
            if isinstance(result, dict):
                self._worker_id = result.get("worker_id", self._worker_id)
            return True
        return False

    async def disconnect(self) -> None:
        """Disconnect worker, releasing claims."""
        if self._connected:
            await self._call("disconnect", worker_id=self._worker_id)
            self._connected = False

    # --- Task Operations ---

    async def list_ready_tasks(self) -> list[TaskInfo]:
        """List tasks that are ready to be claimed."""
        result = await self._call("list_tasks", ready=True)
        if result is None or not isinstance(result, dict):
            return []
        return [
            TaskInfo(
                id=t.get("id", ""),
                title=t.get("title", ""),
                status=t.get("status", "pending"),
                priority=t.get("priority", 5),
                points=t.get("points", 0),
                tags=t.get("tags", []),
                blocked_by=t.get("blocked_by", []),
            )
            for t in result.get("tasks", [])
        ]

    async def claim(self, task_id: str) -> bool:
        """Claim a task. Returns True if successful."""
        result = await self._call("claim", worker_id=self._worker_id, task=task_id)
        if result is not None:
            self._current_task_id = task_id
            self._started_at = time.monotonic()
            return True
        return False

    async def update(
        self,
        task_id: str,
        status: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Update a task."""
        args: dict[str, Any] = {"task": task_id}
        if status is not None:
            args["status"] = status
        args.update(kwargs)
        result = await self._call("update", **args)
        return result is not None

    async def complete(self, task_id: str) -> bool:
        """Mark a task as completed."""
        success = await self.update(task_id, status="completed")
        if success and task_id == self._current_task_id:
            self._current_task_id = None
            self._started_at = None
        return success

    async def fail(self, task_id: str) -> bool:
        """Mark a task as failed."""
        success = await self.update(task_id, status="failed")
        if success and task_id == self._current_task_id:
            self._current_task_id = None
            self._started_at = None
        return success

    async def create_task(
        self,
        title: str,
        description: str = "",
        parent: str | None = None,
        priority: int = 5,
        tags: list[str] | None = None,
        *,
        # Legacy parameters from WorkCoordinator integration
        points: int | None = None,
        claim: bool = False,
    ) -> str | None:
        """Create a new task. Returns task ID or None."""
        args: dict[str, Any] = {
            "title": title,
            "priority": priority,
        }
        if description:
            args["description"] = description
        if parent:
            args["parent"] = parent
        if tags:
            args["tags"] = tags
        if points is not None:
            args["points"] = points

        # Use _call_tool for legacy adapter compatibility
        result = await self._call_tool("create", args)
        if result is None:
            return None

        task_id = result.get("id")

        if task_id and claim:
            await self.claim_task(task_id)

        return task_id

    async def attach_note(self, task_id: str, content: str, note_type: str = "note") -> bool:
        """Attach a note to a task."""
        result = await self._call(
            "attach",
            task=task_id,
            type=note_type,
            content=content,
        )
        return result is not None

    async def log_metrics(
        self,
        task_id: str | None = None,
        *,
        wall_ms: int | None = None,
    ) -> bool:
        """Log metrics for a task."""
        task_id = task_id or self._current_task_id
        if not task_id or not self._worker_id:
            return False

        args: dict[str, Any] = {
            "agent": self._worker_id,
            "task": task_id,
        }
        if wall_ms is not None:
            args["wall_ms"] = wall_ms

        result = await self._call_tool("log_metrics", args)
        return result is not None

    # --- Query ---

    async def get_task(self, task_id: str) -> TaskInfo | None:
        """Get a single task."""
        result = await self._call("get", task=task_id)
        if result is None or not isinstance(result, dict):
            return None
        return TaskInfo(
            id=result.get("id", task_id),
            title=result.get("title", ""),
            status=result.get("status", ""),
            priority=result.get("priority", 5),
            points=result.get("points", 0),
            tags=result.get("tags", []),
            blocked_by=result.get("blocked_by", []),
        )

    async def list_workers(self) -> list[WorkerInfo]:
        """List connected workers."""
        result = await self._call("list_agents")
        if result is None or not isinstance(result, dict):
            return []
        return [
            WorkerInfo(
                worker_id=w.get("worker_id", ""),
                tags=w.get("tags", []),
                claimed_tasks=w.get("claimed_tasks", []),
            )
            for w in result.get("agents", [])
        ]

    # --- Legacy methods for WorkCoordinator integration ---

    async def claim_task(self, task_id: str) -> bool:
        """Claim a task for this agent (legacy API).

        Args:
            task_id: Task ID to claim.

        Returns:
            True if claimed successfully.
        """
        if not self._worker_id:
            _log.warning("Cannot claim task: no agent_id set")
            return False

        result = await self._call_tool(
            "claim",
            {"task": task_id, "worker_id": self._worker_id},
        )
        if result is not None and result.get("success", False):
            self._current_task_id = task_id
            self._started_at = time.monotonic()
            return True
        return False

    async def update_task(
        self,
        task_id: str | None = None,
        *,
        status: str | None = None,
        description: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """Update a task in task-graph (legacy API).

        Args:
            task_id: Task ID (defaults to current_task_id).
            status: New status.
            description: New description.
            reason: Reason for update.

        Returns:
            True if updated successfully.
        """
        task_id = task_id or self._current_task_id
        if not task_id or not self._worker_id:
            return False

        args: dict[str, Any] = {
            "task": task_id,
            "worker_id": self._worker_id,
        }
        if status:
            args["status"] = status
        if description:
            args["description"] = description
        if reason:
            args["reason"] = reason

        result = await self._call_tool("update", args)
        return result is not None

    async def complete_task(
        self,
        task_id: str | None = None,
        *,
        reason: str | None = None,
    ) -> bool:
        """Mark a task as completed and log wall time (legacy API).

        Args:
            task_id: Task ID (defaults to current_task_id).
            reason: Completion reason.

        Returns:
            True if completed successfully.
        """
        task_id = task_id or self._current_task_id
        if not task_id:
            return False

        # Log wall time if we tracked the start
        if self._started_at is not None:
            wall_ms = int((time.monotonic() - self._started_at) * 1000)
            await self.log_metrics(task_id, wall_ms=wall_ms)

        success = await self.update_task(task_id, status="completed", reason=reason)

        if success and task_id == self._current_task_id:
            self._current_task_id = None
            self._started_at = None

        return success

    async def thinking(self, thought: str) -> bool:
        """Send a thinking/heartbeat update.

        Args:
            thought: Current status message.

        Returns:
            True if sent successfully.
        """
        if not self._worker_id:
            return False

        args: dict[str, Any] = {
            "agent": self._worker_id,
            "thought": thought,
        }
        if self._current_task_id:
            args["tasks"] = [self._current_task_id]

        result = await self._call_tool("thinking", args)
        return result is not None


class _MCPManagerAdapter:
    """Adapts MCPClientManager to the MCPCaller protocol.

    This adapter allows the bridge to work with both the new MCPCaller
    protocol (for clean testing) and the existing MCPClientManager
    (for backward compatibility with WorkCoordinator/Timeline).
    """

    def __init__(self, mcp_client_manager: Any) -> None:
        self._mgr = mcp_client_manager

    def is_connected(self, server_name: str) -> bool:
        from activecontext.mcp.types import MCPConnectionStatus

        conn = self._mgr.get_connection(server_name)
        return conn is not None and conn.status == MCPConnectionStatus.CONNECTED

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        conn = self._mgr.get_connection(server_name)
        if conn is None:
            return None
        return await conn.call_tool(tool_name, arguments)
