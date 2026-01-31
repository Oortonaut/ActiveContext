"""Tests for TaskGraphBridge and WorkCoordinator integration with task-graph MCP."""

from __future__ import annotations

import json
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.coordination.task_bridge import TASK_GRAPH_SERVER, TaskGraphBridge
from activecontext.mcp.types import MCPConnectionStatus, MCPToolResult


def _make_manager(
    *,
    connected: bool = True,
    tool_result: MCPToolResult | None = None,
    call_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock MCPClientManager with optional connection."""
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


class TestTaskGraphBridgeProperties:
    """Tests for bridge properties and initialization."""

    def test_init_stores_fields(self) -> None:
        mgr = _make_manager(connected=False)
        bridge = TaskGraphBridge(mgr, agent_id="worker-1")
        assert bridge.agent_id == "worker-1"
        assert bridge.current_task_id is None

    def test_is_active_when_connected(self) -> None:
        mgr = _make_manager(connected=True)
        bridge = TaskGraphBridge(mgr)
        assert bridge.is_active is True
        mgr.get_connection.assert_called_with(TASK_GRAPH_SERVER)

    def test_is_active_when_not_connected(self) -> None:
        mgr = _make_manager(connected=False)
        bridge = TaskGraphBridge(mgr)
        assert bridge.is_active is False

    def test_is_active_when_connection_error(self) -> None:
        mgr = MagicMock()
        conn = MagicMock()
        conn.status = MCPConnectionStatus.ERROR
        mgr.get_connection.return_value = conn
        bridge = TaskGraphBridge(mgr)
        assert bridge.is_active is False

    def test_agent_id_setter(self) -> None:
        mgr = _make_manager(connected=False)
        bridge = TaskGraphBridge(mgr, agent_id="old")
        bridge.agent_id = "new"
        assert bridge.agent_id == "new"


class TestCallTool:
    """Tests for the internal _call_tool method."""

    @pytest.mark.anyio
    async def test_returns_none_when_not_active(self) -> None:
        mgr = _make_manager(connected=False)
        bridge = TaskGraphBridge(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result is None

    @pytest.mark.anyio
    async def test_returns_parsed_json(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"id": "task-1", "title": "test"}))
        bridge = TaskGraphBridge(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result == {"id": "task-1", "title": "test"}

    @pytest.mark.anyio
    async def test_returns_none_on_error_result(self) -> None:
        mgr = _make_manager(tool_result=_error_result("bad request"))
        bridge = TaskGraphBridge(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_on_exception(self) -> None:
        mgr = _make_manager(call_side_effect=RuntimeError("connection lost"))
        bridge = TaskGraphBridge(mgr)
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
        bridge = TaskGraphBridge(mgr)
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
        bridge = TaskGraphBridge(mgr)
        result = await bridge._call_tool("create", {"title": "test"})
        assert result == {}


class TestCreateTask:
    """Tests for create_task method."""

    @pytest.mark.anyio
    async def test_creates_with_title_only(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"id": "t-1"}))
        bridge = TaskGraphBridge(mgr)
        task_id = await bridge.create_task("Fix bug")
        assert task_id == "t-1"
        conn = mgr.get_connection.return_value
        conn.call_tool.assert_called_once_with("create", {"title": "Fix bug"})

    @pytest.mark.anyio
    async def test_creates_with_all_options(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"id": "t-2"}))
        bridge = TaskGraphBridge(mgr)
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
        # First call returns task ID, second (claim) returns success
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "t-3"}),
                _ok_result({"success": True}),
            ]
        )
        bridge = TaskGraphBridge(mgr, agent_id="worker-1")
        task_id = await bridge.create_task("Task", claim=True)
        assert task_id == "t-3"
        assert bridge.current_task_id == "t-3"
        assert conn.call_tool.call_count == 2

    @pytest.mark.anyio
    async def test_returns_none_when_not_active(self) -> None:
        mgr = _make_manager(connected=False)
        bridge = TaskGraphBridge(mgr)
        result = await bridge.create_task("Test")
        assert result is None


class TestClaimTask:
    """Tests for claim_task method."""

    @pytest.mark.anyio
    async def test_claim_success(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge(mgr, agent_id="worker-1")
        result = await bridge.claim_task("t-1")
        assert result is True
        assert bridge.current_task_id == "t-1"

    @pytest.mark.anyio
    async def test_claim_sets_started_at(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge(mgr, agent_id="worker-1")
        assert bridge._started_at is None
        await bridge.claim_task("t-1")
        assert bridge._started_at is not None

    @pytest.mark.anyio
    async def test_claim_fails_without_agent_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge(mgr, agent_id=None)
        result = await bridge.claim_task("t-1")
        assert result is False

    @pytest.mark.anyio
    async def test_claim_passes_worker_id(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({"success": True}))
        bridge = TaskGraphBridge(mgr, agent_id="agent-7")
        await bridge.claim_task("t-1")
        conn = mgr.get_connection.return_value
        conn.call_tool.assert_called_once_with(
            "claim", {"task": "t-1", "worker_id": "agent-7"}
        )


class TestUpdateTask:
    """Tests for update_task method."""

    @pytest.mark.anyio
    async def test_update_with_explicit_task_id(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
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
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
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
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        result = await bridge.update_task(status="blocked")
        assert result is False

    @pytest.mark.anyio
    async def test_update_fails_without_agent_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge(mgr, agent_id=None)
        result = await bridge.update_task("t-1", status="blocked")
        assert result is False


class TestCompleteTask:
    """Tests for complete_task method."""

    @pytest.mark.anyio
    async def test_complete_clears_current_task(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = 1000.0
        result = await bridge.complete_task()
        assert result is True
        assert bridge.current_task_id is None
        assert bridge._started_at is None

    @pytest.mark.anyio
    async def test_complete_logs_wall_time(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = 1000.0

        with patch("activecontext.coordination.task_bridge.time") as mock_time:
            mock_time.monotonic.return_value = 1005.0  # 5 seconds later
            await bridge.complete_task(reason="Done")

        conn = mgr.get_connection.return_value
        # Should have called log_metrics then update
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
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        bridge._started_at = None
        await bridge.complete_task()
        conn = mgr.get_connection.return_value
        # Only update, no log_metrics
        assert conn.call_tool.call_count == 1
        assert conn.call_tool.call_args[0][0] == "update"

    @pytest.mark.anyio
    async def test_complete_fails_without_task_id(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        result = await bridge.complete_task()
        assert result is False

    @pytest.mark.anyio
    async def test_complete_explicit_task_id_doesnt_clear_different_current(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        bridge._current_task_id = "t-other"
        bridge._started_at = None
        result = await bridge.complete_task("t-explicit")
        assert result is True
        # current_task_id should NOT be cleared (different task)
        assert bridge.current_task_id == "t-other"


class TestLogMetrics:
    """Tests for log_metrics method."""

    @pytest.mark.anyio
    async def test_log_metrics_with_wall_ms(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
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
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        bridge._current_task_id = "t-2"
        result = await bridge.log_metrics(wall_ms=1500)
        assert result is True
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["task"] == "t-2"

    @pytest.mark.anyio
    async def test_log_metrics_fails_without_agent(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge(mgr, agent_id=None)
        result = await bridge.log_metrics("t-1", wall_ms=100)
        assert result is False


class TestThinking:
    """Tests for thinking/heartbeat method."""

    @pytest.mark.anyio
    async def test_thinking_sends_thought(self) -> None:
        mgr = _make_manager(tool_result=_ok_result({}))
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
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
        bridge = TaskGraphBridge(mgr, agent_id="w-1")
        bridge._current_task_id = "t-1"
        await bridge.thinking("Still working")
        conn = mgr.get_connection.return_value
        args = conn.call_tool.call_args[0][1]
        assert args["tasks"] == ["t-1"]

    @pytest.mark.anyio
    async def test_thinking_fails_without_agent(self) -> None:
        mgr = _make_manager()
        bridge = TaskGraphBridge(mgr, agent_id=None)
        result = await bridge.thinking("test")
        assert result is False


# ---------------------------------------------------------------------------
# WorkCoordinator + TaskGraphBridge integration tests
# ---------------------------------------------------------------------------


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
    bridge = TaskGraphBridge(mgr, agent_id="worker-1")

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
        # create returns id, claim returns success
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "tg-1"}),
                _ok_result({"success": True}),
            ]
        )

        await coord.work_on("Fix auth bug", "src/auth.py")

        calls = conn.call_tool.call_args_list
        # First call: create task
        assert calls[0][0][0] == "create"
        assert calls[0][0][1]["title"] == "Fix auth bug"
        assert "src/auth.py" in calls[0][0][1]["description"]
        # Second call: claim task
        assert calls[1][0][0] == "claim"
        assert calls[1][0][1]["task"] == "tg-1"

    @pytest.mark.anyio
    async def test_work_on_no_bridge_call_when_inactive(self) -> None:
        coord, bridge, mgr = _make_coordinator(bridge_connected=False)
        await coord.work_on("Fix bug", "src/main.py")
        # get_connection returns None → no call_tool calls
        conn = mgr.get_connection.return_value
        assert conn is None  # not connected

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
                _ok_result({}),  # update call
            ]
        )

        await coord.work_on("Initial intent", "src/main.py")
        await coord.work_update(intent="Updated intent")

        calls = conn.call_tool.call_args_list
        # Third call should be bridge update
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
        # Update files only (no intent) — should NOT call bridge
        await coord.work_update(files=["src/other.py"])
        assert conn.call_tool.call_count == 2  # only create + claim

    @pytest.mark.anyio
    async def test_work_done_completes_bridge_task(self) -> None:
        coord, bridge, mgr = _make_coordinator()
        conn = mgr.get_connection.return_value
        conn.call_tool = AsyncMock(
            side_effect=[
                _ok_result({"id": "tg-1"}),       # create
                _ok_result({"success": True}),     # claim
                _ok_result({}),                    # log_metrics
                _ok_result({}),                    # update (complete)
            ]
        )

        await coord.work_on("Fix bug", "src/main.py")
        await coord.work_done()

        calls = conn.call_tool.call_args_list
        # Last call should be update with status=completed
        update_call = calls[-1]
        assert update_call[0][0] == "update"
        assert update_call[0][1]["status"] == "completed"
        assert update_call[0][1]["reason"] == "work_done"

    @pytest.mark.anyio
    async def test_work_done_no_bridge_when_inactive(self) -> None:
        coord, bridge, mgr = _make_coordinator(bridge_connected=False)
        # work_on without bridge
        await coord.work_on("Fix bug", "src/main.py")
        # work_done should still work (just scratchpad + node cleanup)
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
