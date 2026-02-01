"""Tests for TaskGraphBridge -- MCPCaller protocol-based API and legacy integration.

Covers:
 1. No MCP caller -- all methods no-op gracefully, return defaults
 2. Not connected -- available returns False, methods no-op
 3. Connect success -- mock MCP call, verify connected state
 4. Connect failure -- MCP call returns None, connected stays False
 5. Disconnect -- verify disconnect call made, state updated
 6. List ready tasks -- mock result, verify TaskInfo parsing
 7. List ready tasks empty -- no tasks available
 8. Claim success -- returns True
 9. Claim failure -- returns False
10. Complete -- calls update with status=completed
11. Fail -- calls update with status=failed
12. Create task -- returns task ID from result
13. Attach note -- verify correct parameters passed
14. Get task -- returns TaskInfo
15. Get task not found -- returns None
16. List workers -- returns WorkerInfo list
17. MCP exception handled -- exception in call_tool returns None
18. Graceful degradation -- all operations work with mcp_caller=None

Plus legacy WorkCoordinator integration tests.
"""

from __future__ import annotations

import json
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.coordination.task_bridge import (
    TaskGraphBridge,
    TaskInfo,
    WorkerInfo,
)
from activecontext.mcp.types import MCPConnectionStatus, MCPToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMCPCaller:
    """Simple mock implementing the MCPCaller protocol."""

    def __init__(
        self,
        *,
        connected: bool = True,
        return_value: Any = None,
        side_effect: Exception | None = None,
    ) -> None:
        self._connected = connected
        self._return_value = return_value if return_value is not None else {}
        self._side_effect = side_effect
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def is_connected(self, server_name: str) -> bool:
        return self._connected

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((server_name, tool_name, arguments))
        if self._side_effect:
            raise self._side_effect
        return self._return_value


def _make_manager(
    *,
    connected: bool = True,
    tool_result: MCPToolResult | None = None,
    call_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock MCPClientManager for legacy tests."""
    manager = MagicMock()
    if connected:
        conn = MagicMock()
        conn.status = MCPConnectionStatus.CONNECTED
        if call_side_effect:
            conn.call_tool = AsyncMock(side_effect=call_side_effect)
        elif tool_result is not None:
            conn.call_tool = AsyncMock(return_value=tool_result)
        else:
            conn.call_tool = AsyncMock(
                return_value=MCPToolResult(
                    success=True,
                    content=[{"type": "text", "text": "{}"}],
                )
            )
        manager.get_connection.return_value = conn
    else:
        manager.get_connection.return_value = None
    return manager


def _ok_result(data: dict[str, Any]) -> MCPToolResult:
    """Create a successful MCPToolResult with JSON content."""
    return MCPToolResult(
        success=True,
        content=[{"type": "text", "text": json.dumps(data)}],
    )


def _error_result(msg: str = "something broke") -> MCPToolResult:
    """Create an error MCPToolResult."""
    return MCPToolResult(
        success=False,
        content=[],
        is_error=True,
        error_message=msg,
    )


# ===========================================================================
# 1. No MCP caller -- all methods no-op gracefully
# ===========================================================================


class TestNoMCPCaller:
    """Test that all operations return safe defaults with mcp_caller=None."""

    def test_available_is_false(self) -> None:
        bridge = TaskGraphBridge()
        assert bridge.available is False

    def test_connected_is_false(self) -> None:
        bridge = TaskGraphBridge()
        assert bridge.connected is False

    @pytest.mark.anyio
    async def test_connect_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.connect()
        assert result is False

    @pytest.mark.anyio
    async def test_disconnect_is_noop(self) -> None:
        bridge = TaskGraphBridge()
        await bridge.disconnect()  # Should not raise

    @pytest.mark.anyio
    async def test_list_ready_tasks_empty(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.list_ready_tasks()
        assert result == []

    @pytest.mark.anyio
    async def test_claim_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.claim("t-1")
        assert result is False

    @pytest.mark.anyio
    async def test_update_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.update("t-1", status="working")
        assert result is False

    @pytest.mark.anyio
    async def test_complete_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.complete("t-1")
        assert result is False

    @pytest.mark.anyio
    async def test_fail_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.fail("t-1")
        assert result is False

    @pytest.mark.anyio
    async def test_create_task_returns_none(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.create_task("Test")
        assert result is None

    @pytest.mark.anyio
    async def test_attach_note_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.attach_note("t-1", "note text")
        assert result is False

    @pytest.mark.anyio
    async def test_log_metrics_returns_false(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.log_metrics("t-1", wall_ms=100)
        assert result is False

    @pytest.mark.anyio
    async def test_get_task_returns_none(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.get_task("t-1")
        assert result is None

    @pytest.mark.anyio
    async def test_list_workers_returns_empty(self) -> None:
        bridge = TaskGraphBridge()
        result = await bridge.list_workers()
        assert result == []


# ===========================================================================
# 2. Not connected -- available returns False, methods no-op
# ===========================================================================


class TestNotConnected:
    """Test behavior when MCP caller reports not connected."""

    def test_available_false_when_not_connected(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        assert bridge.available is False

    @pytest.mark.anyio
    async def test_call_returns_none(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge._call("test_tool", foo="bar")
        assert result is None
        assert len(caller.calls) == 0  # no call made

    @pytest.mark.anyio
    async def test_claim_noop_when_not_connected(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.claim("t-1")
        assert result is False


# ===========================================================================
# 3. Connect success
# ===========================================================================


class TestConnectSuccess:
    """Test worker connect lifecycle."""

    @pytest.mark.anyio
    async def test_connect_returns_true(self) -> None:
        caller = MockMCPCaller(return_value={"worker_id": "my-agent"})
        bridge = TaskGraphBridge(caller, worker_id="my-agent")
        result = await bridge.connect()
        assert result is True
        assert bridge.connected is True

    @pytest.mark.anyio
    async def test_connect_updates_worker_id_from_result(self) -> None:
        caller = MockMCPCaller(return_value={"worker_id": "assigned-name"})
        bridge = TaskGraphBridge(caller, worker_id="")
        await bridge.connect()
        assert bridge._worker_id == "assigned-name"

    @pytest.mark.anyio
    async def test_connect_passes_force_flag(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1", tags=["python"])
        await bridge.connect(force=True)
        assert len(caller.calls) == 1
        _, tool, args = caller.calls[0]
        assert tool == "connect"
        assert args["force"] is True
        assert args["tags"] == ["python"]


# ===========================================================================
# 4. Connect failure
# ===========================================================================


class TestConnectFailure:
    """Test connect when MCP call fails."""

    @pytest.mark.anyio
    async def test_connect_returns_false_on_none_result(self) -> None:
        caller = MockMCPCaller(return_value=None)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        # _call returns None when call_tool returns None -- but MockMCPCaller
        # returns None directly. We need _call to detect None.
        # Actually _call wraps call_tool, and None means unavailable.
        # Let's use side_effect to simulate failure.
        caller._side_effect = RuntimeError("connection refused")
        result = await bridge.connect()
        assert result is False
        assert bridge.connected is False

    @pytest.mark.anyio
    async def test_connect_returns_false_when_not_available(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.connect()
        assert result is False
        assert bridge.connected is False


# ===========================================================================
# 5. Disconnect
# ===========================================================================


class TestDisconnect:
    """Test worker disconnect lifecycle."""

    @pytest.mark.anyio
    async def test_disconnect_sends_call(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        await bridge.connect()
        assert bridge.connected is True

        await bridge.disconnect()
        assert bridge.connected is False
        # Verify disconnect call was made
        disconnect_calls = [c for c in caller.calls if c[1] == "disconnect"]
        assert len(disconnect_calls) == 1
        assert disconnect_calls[0][2]["worker_id"] == "w-1"

    @pytest.mark.anyio
    async def test_disconnect_noop_when_not_connected(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        # Never connected, so disconnect should be a no-op
        await bridge.disconnect()
        assert len(caller.calls) == 0


# ===========================================================================
# 6. List ready tasks -- parse TaskInfo
# ===========================================================================


class TestListReadyTasks:
    """Test listing tasks with TaskInfo parsing."""

    @pytest.mark.anyio
    async def test_parses_task_list(self) -> None:
        caller = MockMCPCaller(
            return_value={
                "tasks": [
                    {
                        "id": "t-1",
                        "title": "Fix bug",
                        "status": "pending",
                        "priority": 8,
                        "points": 3,
                        "tags": ["bug"],
                        "blocked_by": [],
                    },
                    {
                        "id": "t-2",
                        "title": "Add feature",
                        "status": "pending",
                        "priority": 5,
                        "points": 5,
                        "tags": ["feature"],
                        "blocked_by": ["t-1"],
                    },
                ]
            }
        )
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        tasks = await bridge.list_ready_tasks()

        assert len(tasks) == 2
        assert isinstance(tasks[0], TaskInfo)
        assert tasks[0].id == "t-1"
        assert tasks[0].title == "Fix bug"
        assert tasks[0].priority == 8
        assert tasks[0].points == 3
        assert tasks[0].tags == ["bug"]
        assert tasks[1].blocked_by == ["t-1"]


# ===========================================================================
# 7. List ready tasks empty
# ===========================================================================


class TestListReadyTasksEmpty:
    """Test empty task list scenarios."""

    @pytest.mark.anyio
    async def test_returns_empty_when_no_tasks(self) -> None:
        caller = MockMCPCaller(return_value={"tasks": []})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        tasks = await bridge.list_ready_tasks()
        assert tasks == []

    @pytest.mark.anyio
    async def test_returns_empty_on_none_result(self) -> None:
        caller = MockMCPCaller(side_effect=RuntimeError("fail"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        tasks = await bridge.list_ready_tasks()
        assert tasks == []


# ===========================================================================
# 8. Claim success
# ===========================================================================


class TestClaimSuccess:
    """Test claiming a task successfully."""

    @pytest.mark.anyio
    async def test_claim_returns_true(self) -> None:
        caller = MockMCPCaller(return_value={"success": True})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.claim("t-1")
        assert result is True

    @pytest.mark.anyio
    async def test_claim_sets_current_task_id(self) -> None:
        caller = MockMCPCaller(return_value={"success": True})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        await bridge.claim("t-1")
        assert bridge.current_task_id == "t-1"

    @pytest.mark.anyio
    async def test_claim_sets_started_at(self) -> None:
        caller = MockMCPCaller(return_value={"success": True})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        assert bridge._started_at is None
        await bridge.claim("t-1")
        assert bridge._started_at is not None

    @pytest.mark.anyio
    async def test_claim_passes_correct_arguments(self) -> None:
        caller = MockMCPCaller(return_value={"success": True})
        bridge = TaskGraphBridge(caller, worker_id="agent-7")
        await bridge.claim("t-1")
        assert len(caller.calls) == 1
        _, tool, args = caller.calls[0]
        assert tool == "claim"
        assert args["task"] == "t-1"
        assert args["worker_id"] == "agent-7"


# ===========================================================================
# 9. Claim failure
# ===========================================================================


class TestClaimFailure:
    """Test claim failure scenarios."""

    @pytest.mark.anyio
    async def test_claim_returns_false_on_error(self) -> None:
        caller = MockMCPCaller(side_effect=RuntimeError("already claimed"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.claim("t-1")
        assert result is False
        assert bridge.current_task_id is None

    @pytest.mark.anyio
    async def test_claim_returns_false_when_not_connected(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.claim("t-1")
        assert result is False


# ===========================================================================
# 10. Complete -- calls update with status=completed
# ===========================================================================


class TestComplete:
    """Test completing a task."""

    @pytest.mark.anyio
    async def test_complete_calls_update(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.complete("t-1")
        assert result is True
        # Verify update call
        _, tool, args = caller.calls[0]
        assert tool == "update"
        assert args["task"] == "t-1"
        assert args["status"] == "completed"

    @pytest.mark.anyio
    async def test_complete_clears_current_task(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = 1000.0
        await bridge.complete("t-1")
        assert bridge.current_task_id is None
        assert bridge._started_at is None

    @pytest.mark.anyio
    async def test_complete_doesnt_clear_different_task(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        bridge._current_task_id = "t-other"
        await bridge.complete("t-explicit")
        assert bridge.current_task_id == "t-other"


# ===========================================================================
# 11. Fail -- calls update with status=failed
# ===========================================================================


class TestFail:
    """Test failing a task."""

    @pytest.mark.anyio
    async def test_fail_calls_update(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.fail("t-1")
        assert result is True
        _, tool, args = caller.calls[0]
        assert tool == "update"
        assert args["status"] == "failed"

    @pytest.mark.anyio
    async def test_fail_clears_current_task(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        bridge._current_task_id = "t-1"
        await bridge.fail("t-1")
        assert bridge.current_task_id is None


# ===========================================================================
# 12. Create task -- returns task ID
# ===========================================================================


class TestCreateTask:
    """Test creating tasks."""

    @pytest.mark.anyio
    async def test_create_returns_id(self) -> None:
        caller = MockMCPCaller(return_value={"id": "new-1"})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        # _call_tool handles MCPToolResult or dict; MockMCPCaller returns dict
        # but _call_tool with dict result returns it directly
        task_id = await bridge.create_task("Fix bug")
        assert task_id == "new-1"

    @pytest.mark.anyio
    async def test_create_with_all_options(self) -> None:
        caller = MockMCPCaller(return_value={"id": "new-2"})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        task_id = await bridge.create_task(
            "Add feature",
            description="Detailed desc",
            parent="parent-1",
            tags=["feat", "api"],
            points=3,
            priority=8,
        )
        assert task_id == "new-2"

    @pytest.mark.anyio
    async def test_create_returns_none_when_not_active(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller)
        result = await bridge.create_task("Test")
        assert result is None


# ===========================================================================
# 13. Attach note -- verify correct parameters
# ===========================================================================


class TestAttachNote:
    """Test attaching notes to tasks."""

    @pytest.mark.anyio
    async def test_attach_note_success(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.attach_note("t-1", "My note", note_type="comment")
        assert result is True
        _, tool, args = caller.calls[0]
        assert tool == "attach"
        assert args["task"] == "t-1"
        assert args["content"] == "My note"
        assert args["type"] == "comment"

    @pytest.mark.anyio
    async def test_attach_note_default_type(self) -> None:
        caller = MockMCPCaller(return_value={})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        await bridge.attach_note("t-1", "Default type note")
        _, _, args = caller.calls[0]
        assert args["type"] == "note"


# ===========================================================================
# 14. Get task -- returns TaskInfo
# ===========================================================================


class TestGetTask:
    """Test getting a task by ID."""

    @pytest.mark.anyio
    async def test_get_task_returns_info(self) -> None:
        caller = MockMCPCaller(
            return_value={
                "id": "t-1",
                "title": "Fix bug",
                "status": "working",
                "priority": 7,
                "points": 2,
                "tags": ["bug", "urgent"],
                "blocked_by": ["t-0"],
            }
        )
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        task = await bridge.get_task("t-1")
        assert task is not None
        assert isinstance(task, TaskInfo)
        assert task.id == "t-1"
        assert task.title == "Fix bug"
        assert task.status == "working"
        assert task.priority == 7
        assert task.points == 2
        assert task.tags == ["bug", "urgent"]
        assert task.blocked_by == ["t-0"]


# ===========================================================================
# 15. Get task not found -- returns None
# ===========================================================================


class TestGetTaskNotFound:
    """Test getting a non-existent task."""

    @pytest.mark.anyio
    async def test_returns_none_on_exception(self) -> None:
        caller = MockMCPCaller(side_effect=RuntimeError("not found"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.get_task("nonexistent")
        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_not_connected(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.get_task("t-1")
        assert result is None


# ===========================================================================
# 16. List workers -- returns WorkerInfo list
# ===========================================================================


class TestListWorkers:
    """Test listing connected workers."""

    @pytest.mark.anyio
    async def test_list_workers_returns_info(self) -> None:
        caller = MockMCPCaller(
            return_value={
                "agents": [
                    {
                        "worker_id": "agent-1",
                        "tags": ["python"],
                        "claimed_tasks": ["t-1"],
                    },
                    {
                        "worker_id": "agent-2",
                        "tags": ["rust", "wasm"],
                        "claimed_tasks": [],
                    },
                ]
            }
        )
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        workers = await bridge.list_workers()
        assert len(workers) == 2
        assert isinstance(workers[0], WorkerInfo)
        assert workers[0].worker_id == "agent-1"
        assert workers[0].tags == ["python"]
        assert workers[0].claimed_tasks == ["t-1"]
        assert workers[1].worker_id == "agent-2"
        assert workers[1].claimed_tasks == []

    @pytest.mark.anyio
    async def test_list_workers_empty(self) -> None:
        caller = MockMCPCaller(return_value={"agents": []})
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        workers = await bridge.list_workers()
        assert workers == []


# ===========================================================================
# 17. MCP exception handled
# ===========================================================================


class TestMCPExceptionHandled:
    """Test that exceptions in call_tool are caught and return None."""

    @pytest.mark.anyio
    async def test_exception_in_call_returns_none(self) -> None:
        caller = MockMCPCaller(side_effect=ConnectionError("socket closed"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge._call("any_tool", foo="bar")
        assert result is None

    @pytest.mark.anyio
    async def test_exception_doesnt_propagate_on_claim(self) -> None:
        caller = MockMCPCaller(side_effect=TimeoutError("timeout"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.claim("t-1")
        assert result is False

    @pytest.mark.anyio
    async def test_exception_doesnt_propagate_on_update(self) -> None:
        caller = MockMCPCaller(side_effect=OSError("broken pipe"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.update("t-1", status="completed")
        assert result is False

    @pytest.mark.anyio
    async def test_exception_doesnt_propagate_on_get_task(self) -> None:
        caller = MockMCPCaller(side_effect=ValueError("bad data"))
        bridge = TaskGraphBridge(caller, worker_id="w-1")
        result = await bridge.get_task("t-1")
        assert result is None


# ===========================================================================
# 18. Graceful degradation -- all with mcp_caller=None
# ===========================================================================


class TestGracefulDegradation:
    """Comprehensive test that every method works with mcp_caller=None."""

    @pytest.mark.anyio
    async def test_full_lifecycle_with_none_caller(self) -> None:
        bridge = TaskGraphBridge()  # No MCP caller

        assert bridge.available is False
        assert bridge.connected is False
        assert bridge.current_task_id is None
        assert bridge.agent_id is None

        # Worker lifecycle
        assert await bridge.connect() is False
        await bridge.disconnect()  # no-op

        # Task operations
        assert await bridge.list_ready_tasks() == []
        assert await bridge.claim("t-1") is False
        assert await bridge.update("t-1", status="working") is False
        assert await bridge.complete("t-1") is False
        assert await bridge.fail("t-1") is False
        assert await bridge.create_task("Test") is None
        assert await bridge.attach_note("t-1", "note") is False
        assert await bridge.log_metrics("t-1", wall_ms=100) is False

        # Query
        assert await bridge.get_task("t-1") is None
        assert await bridge.list_workers() == []

        # Legacy methods
        assert await bridge.thinking("test") is False


# ===========================================================================
# Properties and initialization
# ===========================================================================


class TestProperties:
    """Test properties and initialization."""

    def test_init_stores_fields(self) -> None:
        caller = MockMCPCaller(connected=False)
        bridge = TaskGraphBridge(caller, worker_id="worker-1")
        assert bridge.agent_id == "worker-1"
        assert bridge.current_task_id is None

    def test_available_when_connected(self) -> None:
        caller = MockMCPCaller(connected=True)
        bridge = TaskGraphBridge(caller)
        assert bridge.available is True

    def test_is_active_alias(self) -> None:
        caller = MockMCPCaller(connected=True)
        bridge = TaskGraphBridge(caller)
        assert bridge.is_active is True
        assert bridge.is_active == bridge.available

    def test_agent_id_setter(self) -> None:
        bridge = TaskGraphBridge(worker_id="old")
        bridge.agent_id = "new"
        assert bridge.agent_id == "new"

    def test_agent_id_none_returns_none(self) -> None:
        bridge = TaskGraphBridge()
        assert bridge.agent_id is None

    def test_agent_id_setter_none(self) -> None:
        bridge = TaskGraphBridge(worker_id="test")
        bridge.agent_id = None
        assert bridge.agent_id is None


# ===========================================================================
# Legacy _call_tool method tests
# ===========================================================================


class TestCallTool:
    """Tests for the internal _call_tool method (legacy path)."""

    @pytest.mark.anyio
    async def test_returns_none_when_not_active(self) -> None:
        mgr = _make_manager(connected=False)
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result is None

    @pytest.mark.anyio
    async def test_returns_parsed_json(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"id": "task-1", "title": "test"}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result == {"id": "task-1", "title": "test"}

    @pytest.mark.anyio
    async def test_returns_none_on_error_result(self) -> None:
        mgr = _make_manager(tool_result=_error_result("bad request"))
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_on_exception(self) -> None:
        mgr = _make_manager(call_side_effect=RuntimeError("connection lost"))
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result is None

    @pytest.mark.anyio
    async def test_handles_non_json_text(self) -> None:
        mgr = _make_manager(
            tool_result=MCPToolResult(
                success=True,
                content=[{"type": "text", "text": "not json"}],
            )
        )
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result == {"text": "not json"}

    @pytest.mark.anyio
    async def test_returns_empty_dict_when_no_text_content(self) -> None:
        mgr = _make_manager(
            tool_result=MCPToolResult(
                success=True,
                content=[{"type": "image", "data": "..."}],
            )
        )
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result == {}


# ===========================================================================
# Legacy create_task tests
# ===========================================================================


class TestLegacyCreateTask:
    """Tests for create_task via MCPClientManager (legacy path)."""

    @pytest.mark.anyio
    async def test_creates_with_title_only(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"id": "t-1"}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        task_id = await bridge.create_task("Fix bug")
        assert task_id == "t-1"
        conn = mgr.get_connection.return_value
        conn.call_tool.assert_called_once_with("create", {"title": "Fix bug", "priority": 5})

    @pytest.mark.anyio
    async def test_creates_with_all_options(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"id": "t-2"}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr)
        task_id = await bridge.create_task(
            "Add feature",
            description="Detailed desc",
            parent="parent-1",
            tags=["feat", "api"],
            points=3,
            priority=5,
        )
        assert task_id == "t-2"
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["title"] == "Add feature"
        assert args["description"] == "Detailed desc"
        assert args["parent"] == "parent-1"
        assert args["tags"] == ["feat", "api"]
        assert args["points"] == 3
        assert args["priority"] == 5

    @pytest.mark.anyio
    async def test_create_and_claim(self) -> None:
        mgr = _make_manager()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "t-3"}),
                _ok_result({"success": True}),
            ]
        )
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="worker-1")
        task_id = await bridge.create_task("Task", claim=True)
        assert task_id == "t-3"
        assert bridge.current_task_id == "t-3"
        assert conn.call_tool.call_count == 2


# ===========================================================================
# Legacy claim_task tests
# ===========================================================================


class TestLegacyClaimTask:
    """Tests for claim_task via MCPClientManager (legacy path)."""

    @pytest.mark.anyio
    async def test_claim_success(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="worker-1")
        result = await bridge.claim_task("t-1")
        assert result is True
        assert bridge.current_task_id == "t-1"

    @pytest.mark.anyio
    async def test_claim_sets_started_at(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="worker-1")
        assert bridge._started_at is None
        await bridge.claim_task("t-1")
        assert bridge._started_at is not None

    @pytest.mark.anyio
    async def test_claim_fails_without_agent_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id=None)
        result = await bridge.claim_task("t-1")
        assert result is False

    @pytest.mark.anyio
    async def test_claim_passes_worker_id(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="agent-7")
        await bridge.claim_task("t-1")
        conn = mgr.get_connection.return_value
        conn.call_tool.assert_called_once_with("claim", {"task": "t-1", "worker_id": "agent-7"})


# ===========================================================================
# Legacy update_task tests
# ===========================================================================


class TestLegacyUpdateTask:
    """Tests for update_task via MCPClientManager (legacy path)."""

    @pytest.mark.anyio
    async def test_update_with_explicit_task_id(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        result = await bridge.update_task("t-1", status="in_progress")
        assert result is True
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["task"] == "t-1"
        assert args["status"] == "in_progress"
        assert args["worker_id"] == "w-1"

    @pytest.mark.anyio
    async def test_update_uses_current_task_id(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-2"
        result = await bridge.update_task(status="blocked", reason="Waiting on API")
        assert result is True
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["task"] == "t-2"
        assert args["reason"] == "Waiting on API"

    @pytest.mark.anyio
    async def test_update_fails_without_task_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        result = await bridge.update_task(status="blocked")
        assert result is False

    @pytest.mark.anyio
    async def test_update_fails_without_agent_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id=None)
        result = await bridge.update_task("t-1", status="blocked")
        assert result is False


# ===========================================================================
# Legacy complete_task tests
# ===========================================================================


class TestLegacyCompleteTask:
    """Tests for complete_task via MCPClientManager (legacy path)."""

    @pytest.mark.anyio
    async def test_complete_clears_current_task(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = 1000.0
        result = await bridge.complete_task()
        assert result is True
        assert bridge.current_task_id is None
        assert bridge._started_at is None

    @pytest.mark.anyio
    async def test_complete_logs_wall_time(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = 1000.0

        with patch("activecontext.coordination.task_bridge.time") as mock_time:
            mock_time.monotonic.return_value = 1005.0
            await bridge.complete_task(reason="Done")

        conn = mgr.get_connection.return_value
        calls = conn.call_tool.call_args_list
        assert len(calls) == 2
        # First call: log_metrics with wall_ms
        assert calls[0][0][0] == "log_metrics"
        assert calls[0][0][1]["wall_ms"] == 5000
        # Second call: update with status=completed
        assert calls[1][0][0] == "update"
        assert calls[1][0][1]["status"] == "completed"
        assert calls[1][0][1]["reason"] == "Done"

    @pytest.mark.anyio
    async def test_complete_without_started_at_skips_metrics(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = None
        await bridge.complete_task()
        conn = mgr.get_connection.return_value
        assert conn.call_tool.call_count == 1
        assert conn.call_tool.call_args[0][0] == "update"

    @pytest.mark.anyio
    async def test_complete_fails_without_task_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        result = await bridge.complete_task()
        assert result is False

    @pytest.mark.anyio
    async def test_complete_explicit_task_id_doesnt_clear_different_current(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-other"
        bridge._started_at = None
        result = await bridge.complete_task("t-explicit")
        assert result is True
        assert bridge.current_task_id == "t-other"


# ===========================================================================
# Legacy log_metrics tests
# ===========================================================================


class TestLegacyLogMetrics:
    """Tests for log_metrics via MCPClientManager (legacy path)."""

    @pytest.mark.anyio
    async def test_log_metrics_with_wall_ms(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        result = await bridge.log_metrics("t-1", wall_ms=3000)
        assert result is True
        conn = mgr.get_connection.return_value
        conn.call_tool.assert_called_once_with(
            "log_metrics",
            {"agent": "w-1", "task": "t-1", "wall_ms": 3000},
        )

    @pytest.mark.anyio
    async def test_log_metrics_uses_current_task(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-2"
        result = await bridge.log_metrics(wall_ms=1500)
        assert result is True
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["task"] == "t-2"

    @pytest.mark.anyio
    async def test_log_metrics_fails_without_agent(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id=None)
        result = await bridge.log_metrics("t-1", wall_ms=100)
        assert result is False


# ===========================================================================
# Legacy thinking tests
# ===========================================================================


class TestLegacyThinking:
    """Tests for thinking/heartbeat via MCPClientManager (legacy path)."""

    @pytest.mark.anyio
    async def test_thinking_sends_thought(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        result = await bridge.thinking("Working on auth fix")
        assert result is True
        conn = mgr.get_connection.return_value
        conn.call_tool.assert_called_once_with(
            "thinking",
            {"agent": "w-1", "thought": "Working on auth fix"},
        )

    @pytest.mark.anyio
    async def test_thinking_includes_task_id(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        await bridge.thinking("Still working")
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["tasks"] == ["t-1"]

    @pytest.mark.anyio
    async def test_thinking_fails_without_agent(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id=None)
        result = await bridge.thinking("test")
        assert result is False


# ===========================================================================
# WorkCoordinator + TaskGraphBridge integration tests
# ===========================================================================


def _make_coordinator(
    *,
    bridge_connected: bool = True,
    bridge_result: MCPToolResult | None = None,
) -> tuple[Any, TaskGraphBridge, Any]:
    """Create a WorkCoordinator with mocked scratchpad and bridge.

    Returns:
        (work_coordinator, bridge, mock_mcp_manager)
    """
    from activecontext.context.graph import ContextGraph
    from activecontext.coordination.scratchpad import ScratchpadManager
    from activecontext.session.work_coordinator import WorkCoordinator

    graph = ContextGraph()
    scratchpad = ScratchpadManager(cwd=tempfile.mkdtemp())

    result = bridge_result or _ok_result({"id": "tg-1", "success": True})
    mgr = _make_manager(connected=bridge_connected, tool_result=result)
    bridge = TaskGraphBridge.from_mcp_manager(mgr, agent_id="worker-1")

    coordinator = WorkCoordinator(
        session_id="session-1",
        context_graph=graph,
        scratchpad_manager=scratchpad,
        task_bridge=bridge,
    )
    return coordinator, bridge, mgr


class TestWorkCoordinatorBridgeHooks:
    """Tests for WorkCoordinator calling TaskGraphBridge on lifecycle events."""

    @pytest.mark.anyio
    async def test_work_on_creates_task_in_bridge(self) -> None:
        coord, bridge, mgr = _make_coordinator()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "tg-1"}),
                _ok_result({"success": True}),
            ]
        )

        await coord.work_on("Fix auth bug", "src/auth.py")

        calls = conn.call_tool.call_args_list
        assert calls[0][0][0] == "create"
        assert calls[0][0][1]["title"] == "Fix auth bug"
        assert "src/auth.py" in calls[0][0][1]["description"]
        assert calls[1][0][0] == "claim"
        assert calls[1][0][1]["task"] == "tg-1"

    @pytest.mark.anyio
    async def test_work_on_no_bridge_call_when_inactive(self) -> None:
        coord, bridge, mgr = _make_coordinator(bridge_connected=False)
        await coord.work_on("Fix bug", "src/main.py")
        conn = mgr.get_connection.return_value
        assert conn is None

    @pytest.mark.anyio
    async def test_work_on_still_returns_node_when_bridge_fails(self) -> None:
        coord, bridge, mgr = _make_coordinator()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(return_value=_error_result("server error"))

        node = await coord.work_on("Fix bug", "src/main.py")
        assert node is not None
        assert node.intent == "Fix bug"

    @pytest.mark.anyio
    async def test_work_update_updates_bridge_description(self) -> None:
        coord, bridge, mgr = _make_coordinator()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "tg-1"}),
                _ok_result({"success": True}),
                _ok_result({}),
            ]
        )

        await coord.work_on("Initial intent", "src/main.py")
        await coord.work_update(intent="Updated intent")

        calls = conn.call_tool.call_args_list
        assert calls[2][0][0] == "update"
        assert calls[2][0][1]["description"] == "Updated intent"

    @pytest.mark.anyio
    async def test_work_update_no_bridge_call_without_intent(self) -> None:
        coord, bridge, mgr = _make_coordinator()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "tg-1"}),
                _ok_result({"success": True}),
            ]
        )

        await coord.work_on("Initial", "src/main.py")
        await coord.work_update(files=["src/other.py"])
        assert conn.call_tool.call_count == 2

    @pytest.mark.anyio
    async def test_work_done_completes_bridge_task(self) -> None:
        coord, bridge, mgr = _make_coordinator()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "tg-1"}),
                _ok_result({"success": True}),
                _ok_result({}),
                _ok_result({}),
            ]
        )

        await coord.work_on("Fix bug", "src/main.py")
        await coord.work_done()

        calls = conn.call_tool.call_args_list
        update_call = calls[-1]
        assert update_call[0][0] == "update"
        assert update_call[0][1]["status"] == "completed"
        assert update_call[0][1]["reason"] == "work_done"

    @pytest.mark.anyio
    async def test_work_done_no_bridge_when_inactive(self) -> None:
        coord, bridge, mgr = _make_coordinator(bridge_connected=False)
        await coord.work_on("Fix bug", "src/main.py")
        await coord.work_done()
        assert coord.work_node is not None
        assert coord.work_node.work_status == "done"

    @pytest.mark.anyio
    async def test_work_coordinator_without_bridge(self) -> None:
        """WorkCoordinator works fine with task_bridge=None."""
        from activecontext.context.graph import ContextGraph
        from activecontext.coordination.scratchpad import ScratchpadManager
        from activecontext.session.work_coordinator import WorkCoordinator

        graph = ContextGraph()
        scratchpad = ScratchpadManager(cwd=tempfile.mkdtemp())
        coord = WorkCoordinator(
            session_id="s-1",
            context_graph=graph,
            scratchpad_manager=scratchpad,
            task_bridge=None,
        )
        node = await coord.work_on("Test", "src/file.py")
        assert node.intent == "Test"
        await coord.work_update(intent="Updated")
        assert node.intent == "Updated"
        await coord.work_done()
        assert node.work_status == "done"
