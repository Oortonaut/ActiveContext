"""Bridge between ActiveContext work coordination and task-graph MCP server.

Routes work lifecycle events (work_on, work_update, work_done) through the
task-graph MCP server for time tracking and task creation. All operations
are graceful no-ops when task-graph is not connected.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from activecontext.mcp.client import MCPClientManager

_log = logging.getLogger("activecontext.coordination.task_bridge")

# Well-known server name for task-graph MCP
TASK_GRAPH_SERVER = "task-graph"


class TaskGraphBridge:
    """Routes work lifecycle events to task-graph MCP when connected.

    Wraps task-graph MCP tool calls (create, claim, update, log_metrics)
    with automatic no-op behavior when the server is not available.

    Usage:
        bridge = TaskGraphBridge(mcp_client_manager, agent_id="worker-1")
        if bridge.is_active:
            task_id = await bridge.create_task("Fix auth bug")
            await bridge.claim_task(task_id)
            # ... do work ...
            await bridge.complete_task(task_id, reason="Fixed")
    """

    def __init__(
        self,
        mcp_client_manager: MCPClientManager,
        agent_id: str | None = None,
    ) -> None:
        """Initialize the bridge.

        Args:
            mcp_client_manager: MCP client manager to find task-graph server.
            agent_id: Agent identifier for task-graph operations.
        """
        self._mcp = mcp_client_manager
        self._agent_id = agent_id
        self._current_task_id: str | None = None
        self._started_at: float | None = None

    @property
    def is_active(self) -> bool:
        """Whether task-graph MCP server is connected."""
        from activecontext.mcp.types import MCPConnectionStatus

        conn = self._mcp.get_connection(TASK_GRAPH_SERVER)
        return conn is not None and conn.status == MCPConnectionStatus.CONNECTED

    @property
    def agent_id(self) -> str | None:
        """The agent ID used for task-graph operations."""
        return self._agent_id

    @agent_id.setter
    def agent_id(self, value: str | None) -> None:
        self._agent_id = value

    @property
    def current_task_id(self) -> str | None:
        """The task ID currently being worked on."""
        return self._current_task_id

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Call a task-graph tool, returning parsed result or None on failure."""
        if not self.is_active:
            return None

        conn = self._mcp.get_connection(TASK_GRAPH_SERVER)
        if conn is None:
            return None

        try:
            result = await conn.call_tool(tool_name, arguments)
            if result.is_error:
                _log.warning(
                    "task-graph.%s failed: %s",
                    tool_name,
                    result.error_message or "unknown error",
                )
                return None

            # Extract text content from result
            for block in result.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    import json

                    try:
                        return cast(dict[str, Any], json.loads(block["text"]))
                    except (json.JSONDecodeError, KeyError):
                        return {"text": block.get("text", "")}
            return {}

        except Exception:
            _log.debug("task-graph.%s call failed", tool_name, exc_info=True)
            return None

    async def create_task(
        self,
        title: str,
        *,
        description: str | None = None,
        parent: str | None = None,
        tags: list[str] | None = None,
        points: int | None = None,
        priority: int | None = None,
        claim: bool = False,
    ) -> str | None:
        """Create a task in task-graph.

        Args:
            title: Task title.
            description: Detailed description.
            parent: Parent task ID.
            tags: Categorization tags.
            points: Effort estimate.
            priority: Priority 0-10.
            claim: If True, immediately claim the task.

        Returns:
            Task ID string, or None if not active.
        """
        args: dict[str, Any] = {"title": title}
        if description:
            args["description"] = description
        if parent:
            args["parent"] = parent
        if tags:
            args["tags"] = tags
        if points is not None:
            args["points"] = points
        if priority is not None:
            args["priority"] = priority

        result = await self._call_tool("create", args)
        if result is None:
            return None

        task_id = result.get("id")
        if task_id and claim:
            await self.claim_task(task_id)

        return task_id

    async def claim_task(self, task_id: str) -> bool:
        """Claim a task for this agent.

        Args:
            task_id: Task ID to claim.

        Returns:
            True if claimed successfully.
        """
        if not self._agent_id:
            _log.warning("Cannot claim task: no agent_id set")
            return False

        result = await self._call_tool(
            "claim",
            {"task": task_id, "worker_id": self._agent_id},
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
        """Update a task in task-graph.

        Args:
            task_id: Task ID (defaults to current_task_id).
            status: New status.
            description: New description.
            reason: Reason for update.

        Returns:
            True if updated successfully.
        """
        task_id = task_id or self._current_task_id
        if not task_id or not self._agent_id:
            return False

        args: dict[str, Any] = {
            "task": task_id,
            "worker_id": self._agent_id,
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
        """Mark a task as completed and log wall time.

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

        success = await self.update_task(
            task_id, status="completed", reason=reason
        )

        if success and task_id == self._current_task_id:
            self._current_task_id = None
            self._started_at = None

        return success

    async def log_metrics(
        self,
        task_id: str | None = None,
        *,
        wall_ms: int | None = None,
    ) -> bool:
        """Log metrics for a task.

        Args:
            task_id: Task ID (defaults to current_task_id).
            wall_ms: Wall clock time in milliseconds.

        Returns:
            True if logged successfully.
        """
        task_id = task_id or self._current_task_id
        if not task_id or not self._agent_id:
            return False

        args: dict[str, Any] = {
            "agent": self._agent_id,
            "task": task_id,
        }
        if wall_ms is not None:
            args["wall_ms"] = wall_ms

        result = await self._call_tool("log_metrics", args)
        return result is not None

    async def thinking(self, thought: str) -> bool:
        """Send a thinking/heartbeat update.

        Args:
            thought: Current status message.

        Returns:
            True if sent successfully.
        """
        if not self._agent_id:
            return False

        args: dict[str, Any] = {
            "agent": self._agent_id,
            "thought": thought,
        }
        if self._current_task_id:
            args["tasks"] = [self._current_task_id]

        result = await self._call_tool("thinking", args)
        return result is not None
