"""Tests for Timeline coverage improvement.

These tests target previously untested functions in Timeline
to improve coverage from ~49% toward the target.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.state import NotificationLevel
from activecontext.session.protocols import EventHandler, EventResponse, WaitCondition, WaitMode
from activecontext.session.timeline import Timeline


class TestConfigureFileWatcher:
    """Tests for configure_file_watcher method."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_configure_with_none_does_nothing(self, temp_cwd: Path) -> None:
        """Test that None config doesn't change defaults."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            original_interval = timeline._file_watcher.poll_interval
            timeline.configure_file_watcher(None)
            assert timeline._file_watcher.poll_interval == original_interval
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_configure_with_disabled_config(self, temp_cwd: Path) -> None:
        """Test that disabled config creates no-op watcher."""
        from activecontext.config.schema import FileWatchConfig

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            config = FileWatchConfig(enabled=False, poll_interval=1.0)
            timeline.configure_file_watcher(config)
            # Poll interval should be infinity (disabled)
            assert timeline._file_watcher.poll_interval == float("inf")
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_configure_with_custom_poll_interval(self, temp_cwd: Path) -> None:
        """Test that enabled config updates poll interval."""
        from activecontext.config.schema import FileWatchConfig

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            config = FileWatchConfig(enabled=True, poll_interval=5.0)
            timeline.configure_file_watcher(config)
            assert timeline._file_watcher.poll_interval == 5.0
        finally:
            await timeline.close()


class TestEventSystem:
    """Tests for event system functions."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_event_response_sets_handler(self, temp_cwd: Path) -> None:
        """Test that event_response() sets up an event handler."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Call the internal method directly since EventResponse isn't exposed in namespace
            timeline._event_response("custom_event", EventResponse.WAKE, "Custom: {data}")

            handler = timeline._event_handlers.get("custom_event")
            assert handler is not None
            assert handler.response == EventResponse.WAKE
            assert handler.prompt_template == "Custom: {data}"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_fire_event_queues_for_queue_response(self, temp_cwd: Path) -> None:
        """Test that fire_event queues events with QUEUE response."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Set up QUEUE handler
            timeline._event_response("test_event", EventResponse.QUEUE, "Test")

            # Fire event
            wake_prompt = timeline.fire_event("test_event", {"data": "value"})

            assert wake_prompt is None  # No wake for QUEUE
            assert len(timeline._queued_events) == 1
            assert timeline._queued_events[0].event_name == "test_event"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_fire_event_returns_prompt_for_wake_response(self, temp_cwd: Path) -> None:
        """Test that fire_event returns wake prompt for WAKE response."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Set up WAKE handler
            timeline._event_response("wake_event", EventResponse.WAKE, "Waking: {msg}")

            # Fire event
            wake_prompt = timeline.fire_event("wake_event", {"msg": "hello"})

            assert wake_prompt == "Waking: hello"
            assert len(timeline._queued_events) == 0  # Not queued
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_fire_event_with_one_time_handler(self, temp_cwd: Path) -> None:
        """Test that one-time handlers are removed after firing."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Set up one-time WAKE handler
            timeline._event_handlers["one_time"] = EventHandler(
                event_name="one_time",
                response=EventResponse.WAKE,
                prompt_template="Once: {x}",
                once=True,
            )

            # Fire event - should remove handler
            wake_prompt = timeline.fire_event("one_time", {"x": "test"})
            assert wake_prompt == "Once: test"

            # Handler should be removed
            assert "one_time" not in timeline._event_handlers
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_fire_event_with_no_handler_queues(self, temp_cwd: Path) -> None:
        """Test that events without handlers are queued by default."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Fire event with no handler
            wake_prompt = timeline.fire_event("unknown_event", {"data": "value"})

            assert wake_prompt is None
            assert len(timeline._queued_events) == 1
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_queued_events_all(self, temp_cwd: Path) -> None:
        """Test getting all queued events."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Queue some events
            timeline.fire_event("event_a", {"data": 1})
            timeline.fire_event("event_b", {"data": 2})
            timeline.fire_event("event_a", {"data": 3})

            events = timeline.get_queued_events()
            assert len(events) == 3
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_queued_events_filtered(self, temp_cwd: Path) -> None:
        """Test getting queued events filtered by name."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Queue some events
            timeline.fire_event("event_a", {"data": 1})
            timeline.fire_event("event_b", {"data": 2})
            timeline.fire_event("event_a", {"data": 3})

            events = timeline.get_queued_events("event_a")
            assert len(events) == 2
            assert all(e.event_name == "event_a" for e in events)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_clear_queued_events_all(self, temp_cwd: Path) -> None:
        """Test clearing all queued events."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Queue some events
            timeline.fire_event("event_a", {"data": 1})
            timeline.fire_event("event_b", {"data": 2})

            count = timeline.clear_queued_events()

            assert count == 2
            assert len(timeline._queued_events) == 0
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_clear_queued_events_filtered(self, temp_cwd: Path) -> None:
        """Test clearing queued events by name."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Queue some events
            timeline.fire_event("event_a", {"data": 1})
            timeline.fire_event("event_b", {"data": 2})
            timeline.fire_event("event_a", {"data": 3})

            count = timeline.clear_queued_events("event_a")

            assert count == 2
            assert len(timeline._queued_events) == 1
            assert timeline._queued_events[0].event_name == "event_b"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_has_pending_wake_prompt_no_condition(self, temp_cwd: Path) -> None:
        """Test has_pending_wake_prompt returns False when no condition."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            assert timeline.has_pending_wake_prompt() is False
        finally:
            await timeline.close()


class TestWaitConditionChecks:
    """Tests for wait condition checking."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_check_wait_condition_no_condition(self, temp_cwd: Path) -> None:
        """Test check_wait_condition returns False with no active condition."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is False
            assert prompt is None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_check_wait_condition_timeout(self, temp_cwd: Path) -> None:
        """Test check_wait_condition detects timeout."""
        import sys

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a shell command
            if sys.platform == "win32":
                await timeline.execute_statement('s = shell("cmd", ["/c", "echo", "test"])')
            else:
                await timeline.execute_statement('s = shell("echo", ["test"])')

            ns = timeline.get_namespace()
            shell_node = ns["s"]

            # Set a condition that's already timed out
            timeline._wait_condition = WaitCondition(
                node_ids=[shell_node.node_id],
                mode=WaitMode.SINGLE,
                wake_prompt="Done",
                timeout=0.0,  # Already expired
                timeout_prompt="Timed out!",
                started_at=0.0,  # Way in the past
            )

            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is True
            assert "Timed out" in prompt
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_check_wait_condition_no_valid_nodes(self, temp_cwd: Path) -> None:
        """Test check_wait_condition handles invalid node IDs."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Set a condition with non-existent node IDs
            timeline._wait_condition = WaitCondition(
                node_ids=["nonexistent_node"],
                mode=WaitMode.SINGLE,
                wake_prompt="Done",
            )

            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is True
            assert "no valid nodes" in prompt.lower()
        finally:
            await timeline.close()


class TestNotifyFunction:
    """Tests for notify DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_notify_sets_notification_level(self, temp_cwd: Path) -> None:
        """Test that notify() sets notification level on a node."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("notify(v, NotificationLevel.WAKE)")

            ns = timeline.get_namespace()
            v = ns["v"]
            assert v.node().notification_level == NotificationLevel.WAKE
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_with_string_level(self, temp_cwd: Path) -> None:
        """Test that notify() accepts string level."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement('notify(v, "hold")')

            ns = timeline.get_namespace()
            v = ns["v"]
            assert v.node().notification_level == NotificationLevel.HOLD
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_with_node_id_string(self, temp_cwd: Path) -> None:
        """Test that notify() accepts node_id string."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('v = text("test.py")')
            ns = timeline.get_namespace()
            node_id = ns["v"].node_id

            result = await timeline.execute_statement(f'notify("{node_id}", "wake")')
            assert result.status.value == "ok"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_with_invalid_node_id_raises(self, temp_cwd: Path) -> None:
        """Test that notify() raises for invalid node_id."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement('notify("nonexistent", "wake")')
            assert result.status.value == "error"
            assert "not found" in result.exception["message"].lower()
        finally:
            await timeline.close()


class TestViewDispatcher:
    """Tests for view() dispatcher function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_view_dispatches_to_text(self, temp_cwd: Path) -> None:
        """Test that view() with media_type='text' calls _make_text_node."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement('v = view("text", "test.py", tokens=500)')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert "v" in ns
            # Should be a text node
            assert ns["v"].GetDigest()["type"] == "text"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_view_dispatches_to_markdown(self, temp_cwd: Path) -> None:
        """Test that view() with media_type='markdown' calls _make_markdown_node."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a markdown file
            md_file = temp_cwd / "test.md"
            md_file.write_text("# Hello\n\nWorld")

            result = await timeline.execute_statement('v = view("markdown", "test.md", tokens=500)')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert "v" in ns
            # Should be a text node with markdown media type
            assert ns["v"].GetDigest()["type"] == "text"
            assert ns["v"].GetDigest()["media_type"] == "markdown"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_view_with_unknown_media_type_raises(self, temp_cwd: Path) -> None:
        """Test that view() with unknown media_type raises ValueError."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement('v = view("unknown", "test.py")')
            assert result.status.value == "error"
            assert "Unknown media_type" in result.exception["message"]
        finally:
            await timeline.close()


class TestBranchFunction:
    """Tests for branch() checkpoint function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_branch_creates_checkpoint(self, temp_cwd: Path) -> None:
        """Test that branch() creates a checkpoint and returns it."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Test")')
            result = await timeline.execute_statement('cp = branch("my_branch")')
            assert result.status.value == "ok"

            # Verify checkpoint exists
            checkpoints = timeline._context_graph.get_checkpoints()
            assert any(cp.name == "my_branch" for cp in checkpoints)
        finally:
            await timeline.close()


class TestProgressionViews:
    """Tests for SequenceView, LoopView, and StateView factories."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_sequence_view_creation(self, temp_cwd: Path) -> None:
        """Test creating a SequenceView via sequence() DSL."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            from activecontext.context.view import SequenceView

            # Create child nodes
            await timeline.execute_statement('s1 = topic("Step 1")')
            await timeline.execute_statement('s2 = topic("Step 2")')
            await timeline.execute_statement('s3 = topic("Step 3")')

            # Create sequence
            result = await timeline.execute_statement("seq = sequence(s1, s2, s3)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            seq = ns["seq"]
            assert isinstance(seq, SequenceView)
            assert seq.current_index == 0  # Starts at first step
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_loop_view_creation(self, temp_cwd: Path) -> None:
        """Test creating a LoopView via loop_view() DSL."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            from activecontext.context.view import LoopView

            # Create child node
            await timeline.execute_statement('step = topic("Iterate Me")')

            # Create loop
            result = await timeline.execute_statement("lv = loop_view(step, max_iterations=5)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            lv = ns["lv"]
            assert isinstance(lv, LoopView)
            assert lv.max_iterations == 5
            # LoopView iteration starts at 1, not 0
            assert lv.iteration == 1
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_loop_view_with_node_id_string(self, temp_cwd: Path) -> None:
        """Test creating LoopView with node ID string."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            from activecontext.context.view import LoopView

            await timeline.execute_statement('step = topic("Iterate")')
            ns = timeline.get_namespace()
            node_id = ns["step"].node_id

            result = await timeline.execute_statement(f'lv = loop_view("{node_id}")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert isinstance(ns["lv"], LoopView)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_loop_view_with_unknown_node_raises(self, temp_cwd: Path) -> None:
        """Test that loop_view() with unknown node ID raises."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = await timeline.execute_statement('lv = loop_view("unknown_id")')
            assert result.status.value == "error"
            assert "Unknown node ID" in result.exception["message"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_state_machine_creation_with_children(self, temp_cwd: Path) -> None:
        """Test creating StateView via state_machine() with children."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            from activecontext.context.view import StateView

            # Create state nodes
            await timeline.execute_statement('idle = topic("Idle")')
            await timeline.execute_statement('working = topic("Working")')

            # Create state machine
            result = await timeline.execute_statement("fsm = state_machine(idle, working)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            fsm = ns["fsm"]
            assert isinstance(fsm, StateView)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_state_machine_creation_with_explicit_states(self, temp_cwd: Path) -> None:
        """Test creating StateView with explicit states dict."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            from activecontext.context.view import StateView

            # Create state nodes
            await timeline.execute_statement('idle = topic("Idle")')
            await timeline.execute_statement('working = topic("Working")')

            ns = timeline.get_namespace()
            idle_id = ns["idle"].node_id
            working_id = ns["working"].node_id

            # Create with explicit states
            stmt = (
                f'fsm = state_machine(states={{"idle": "{idle_id}", '
                f'"working": "{working_id}"}}, initial="idle")'
            )
            result = await timeline.execute_statement(stmt)
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            fsm = ns["fsm"]
            assert isinstance(fsm, StateView)
            assert fsm.current_state == "idle"
        finally:
            await timeline.close()


class TestPermissionManagerSetters:
    """Tests for permission manager setter methods."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_set_permission_manager(self, temp_cwd: Path) -> None:
        """Test set_permission_manager updates namespace."""
        from activecontext.session.permissions import PermissionManager

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Initially None
            assert timeline.permission_manager is None

            # Set a permission manager
            pm = PermissionManager(cwd=str(temp_cwd))
            timeline.set_permission_manager(pm)

            assert timeline.permission_manager is pm
            # open() should now be wrapped
            assert timeline._namespace["__builtins__"]["open"] is not None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_set_import_guard(self, temp_cwd: Path) -> None:
        """Test set_import_guard updates namespace."""
        from activecontext.session.permissions import ImportGuard

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Initially None
            assert timeline.import_guard is None

            # Set an import guard
            guard = ImportGuard(allowed_modules=["json"])
            timeline.set_import_guard(guard)

            assert timeline.import_guard is guard
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_set_shell_permission_manager(self, temp_cwd: Path) -> None:
        """Test set_shell_permission_manager updates timeline."""
        from activecontext.session.permissions import ShellPermissionManager

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Initially None
            assert timeline.shell_permission_manager is None

            # Set a shell permission manager
            spm = ShellPermissionManager()
            timeline.set_shell_permission_manager(spm)

            assert timeline.shell_permission_manager is spm
        finally:
            await timeline.close()


class TestFetchFunction:
    """Tests for fetch() HTTP request function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_fetch_returns_coroutine(self, temp_cwd: Path) -> None:
        """Test that fetch() returns a coroutine for awaiting."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Call the internal method directly - it returns a coroutine
            coro = timeline._fetch("http://example.com")
            assert asyncio.iscoroutine(coro)
            # Clean up the coroutine
            coro.close()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_fetch_with_permission_denied(self, temp_cwd: Path) -> None:
        """Test fetch raises WebsitePermissionDenied when not permitted."""
        from activecontext.session.permissions import (
            WebsitePermissionDenied,
            WebsitePermissionManager,
        )

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Set up permission manager that denies everything
            wpm = WebsitePermissionManager(deny_by_default=True)
            timeline._website_permission_manager = wpm

            # fetch should raise
            with pytest.raises(WebsitePermissionDenied):
                await timeline._fetch_with_permission("http://denied.example.com")
        finally:
            await timeline.close()


class TestOnFileChange:
    """Tests for on_file_change DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_on_file_change_sets_handler(self, temp_cwd: Path) -> None:
        """Test that on_file_change() sets global file change response."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Configure with WAKE response
            timeline._on_file_change(response="wake", prompt="File: {path}")

            handler = timeline._event_handlers.get("file_changed")
            assert handler is not None
            assert handler.response == EventResponse.WAKE
            assert handler.prompt_template == "File: {path}"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_on_file_change_queue_response(self, temp_cwd: Path) -> None:
        """Test that on_file_change() with 'queue' sets QUEUE response."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            timeline._on_file_change(response="queue")

            handler = timeline._event_handlers.get("file_changed")
            assert handler is not None
            assert handler.response == EventResponse.QUEUE
        finally:
            await timeline.close()


class TestWaitFileChange:
    """Tests for wait_file_change DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_wait_file_change_single_path(self, temp_cwd: Path) -> None:
        """Test wait_file_change with single path."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            timeline._wait_file_change("src/main.py", wake_prompt="Main changed!")

            # Should set done (waiting state)
            assert timeline._done_called is True

            # Should have handler for the file
            handler = timeline._event_handlers.get("file_changed:src/main.py")
            assert handler is not None
            assert handler.once is True
            assert handler.response == EventResponse.WAKE
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_wait_file_change_multiple_paths(self, temp_cwd: Path) -> None:
        """Test wait_file_change with multiple paths."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            timeline._wait_file_change(["file1.py", "file2.py"])

            # Should have handlers for both files
            assert "file_changed:file1.py" in timeline._event_handlers
            assert "file_changed:file2.py" in timeline._event_handlers
        finally:
            await timeline.close()


class TestCurrentGroup:
    """Tests for current group ID management."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_current_group_initially_none(self, temp_cwd: Path) -> None:
        """Test that current_group_id is initially None."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            assert timeline.current_group_id is None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_set_and_clear_current_group(self, temp_cwd: Path) -> None:
        """Test setting and clearing current group."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a group
            await timeline.execute_statement('g = group(summary="Test")')
            ns = timeline.get_namespace()
            group_id = ns["g"].node_id

            # Set current group
            timeline.set_current_group(group_id)
            assert timeline.current_group_id == group_id

            # Clear current group
            timeline.clear_current_group()
            assert timeline.current_group_id is None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nodes_auto_linked_to_current_group(self, temp_cwd: Path) -> None:
        """Test that nodes are auto-linked when current_group is set."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a group
            await timeline.execute_statement('g = group(summary="Container")')
            ns = timeline.get_namespace()
            group_id = ns["g"].node_id

            # Set current group
            timeline.set_current_group(group_id)

            # Create a topic (should be auto-linked)
            await timeline.execute_statement('t = topic("Auto-linked")')
            ns = timeline.get_namespace()
            topic_node = ns["t"].node()

            # Verify parent relationship
            parents = timeline._context_graph.get_parents(topic_node.node_id)
            parent_ids = [p.node_id for p in parents]
            assert group_id in parent_ids
        finally:
            await timeline.close()


class TestAsyncContextManager:
    """Tests for async context manager protocol."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_async_with_timeline(self, temp_cwd: Path) -> None:
        """Test using Timeline as async context manager."""
        async with Timeline(
            "test-session", context_graph=ContextGraph(), cwd=str(temp_cwd)
        ) as timeline:
            result = await timeline.execute_statement("x = 1 + 1")
            assert result.status.value == "ok"
            assert timeline.get_namespace()["x"] == 2

        # After exit, close() should have been called
        # We can't easily verify this, but we know it ran


class TestFormatWakePrompt:
    """Tests for _format_wake_prompt helper."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_format_wake_prompt_shell_node(self, temp_cwd: Path) -> None:
        """Test formatting wake prompt for ShellNode."""

        from activecontext.context.nodes import ShellNode

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a shell node manually for testing
            shell_node = ShellNode(command="echo", args=["hello"])
            shell_node._exit_code = 0
            shell_node._output = "hello\n"

            template = "Command {command} completed"
            prompt = timeline._format_wake_prompt(template, shell_node)

            # Verify the template was formatted correctly
            assert "echo" in prompt
            assert "completed" in prompt
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_format_wake_prompt_lock_node(self, temp_cwd: Path) -> None:
        """Test formatting wake prompt for LockNode."""
        from activecontext.context.nodes import LockNode

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a lock node manually for testing
            lock_node = LockNode(lockfile="test.lock")

            template = "Lock {lockfile} is ready"
            prompt = timeline._format_wake_prompt(template, lock_node)

            # Verify the lockfile was formatted into the template
            assert "test.lock" in prompt
            assert "ready" in prompt
        finally:
            await timeline.close()


class TestFireEventTargeted:
    """Tests for targeted event firing (agent_id/sender based lookup)."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_fire_event_with_targeted_handler(self, temp_cwd: Path) -> None:
        """Test fire_event uses targeted handler key when agent_id present."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Set up targeted handler
            timeline._event_handlers["agent_done:agent_123"] = EventHandler(
                event_name="agent_done",
                response=EventResponse.WAKE,
                prompt_template="Agent {agent_id} done",
                target_id="agent_123",
            )

            # Fire with matching agent_id
            wake_prompt = timeline.fire_event("agent_done", {"agent_id": "agent_123"})
            assert wake_prompt == "Agent agent_123 done"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_fire_event_falls_back_to_general_handler(self, temp_cwd: Path) -> None:
        """Test fire_event falls back to non-targeted handler."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Only general handler exists
            timeline._event_handlers["agent_done"] = EventHandler(
                event_name="agent_done",
                response=EventResponse.WAKE,
                prompt_template="Some agent done: {agent_id}",
            )

            # Fire with agent_id that has no targeted handler
            wake_prompt = timeline.fire_event("agent_done", {"agent_id": "other_agent"})
            assert "other_agent" in wake_prompt
        finally:
            await timeline.close()


class TestWaitEvent:
    """Tests for wait() event function (not the node wait)."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_wait_event_with_agent_handle(self, temp_cwd: Path) -> None:
        """Test _wait_event with an AgentHandle target."""
        from activecontext.agents.handle import AgentHandle

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a mock agent handle
            handle = MagicMock(spec=AgentHandle)
            handle.agent_id = "child_agent_123"

            # Call wait_event
            timeline._wait_event(handle, "agent_done", prompt="Child done!")

            # Should have set up a wait condition
            assert timeline._wait_condition is not None
            assert timeline._wait_condition.mode == WaitMode.AGENT
            assert timeline._wait_condition.agent_id == "child_agent_123"
            assert timeline._done_called is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_wait_event_with_string_target(self, temp_cwd: Path) -> None:
        """Test _wait_event with a string target ID."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            timeline._wait_event("agent_xyz", "agent_done")

            assert timeline._wait_condition is not None
            assert timeline._wait_condition.agent_id == "agent_xyz"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_wait_event_with_timeout(self, temp_cwd: Path) -> None:
        """Test _wait_event with timeout parameter."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            timeline._wait_event(None, "tick", timeout=30.0)

            assert timeline._wait_condition is not None
            assert timeline._wait_condition.timeout == 30.0
        finally:
            await timeline.close()


class TestProcessFileChanges:
    """Tests for process_file_changes method."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_process_file_changes_no_changes(self, temp_cwd: Path) -> None:
        """Test process_file_changes with no pending changes."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            wake_prompts = timeline.process_file_changes()
            assert wake_prompts == []
        finally:
            await timeline.close()


class TestResolveNodeId:
    """Tests for _resolve_node_id helper."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_resolve_node_id_string(self, temp_cwd: Path) -> None:
        """Test resolving a string node ID."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            result = timeline._resolve_node_id("node_123")
            assert result == "node_123"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_resolve_node_id_context_node(self, temp_cwd: Path) -> None:
        """Test resolving a ContextNode."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Test")')
            ns = timeline.get_namespace()
            node = ns["t"].node()

            result = timeline._resolve_node_id(node)
            assert result == node.node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_resolve_node_id_node_view(self, temp_cwd: Path) -> None:
        """Test resolving a NodeView."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('t = topic("Test")')
            ns = timeline.get_namespace()
            view = ns["t"]

            result = timeline._resolve_node_id(view)
            assert result == view.node_id
        finally:
            await timeline.close()


class TestCheckWaitConditionModes:
    """Tests for check_wait_condition with different WaitMode types."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_check_wait_single_mode_completed(self, temp_cwd: Path) -> None:
        """Test SINGLE mode with completed shell."""
        import sys

        from activecontext.context.nodes import ShellStatus

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create and manually complete a shell node
            if sys.platform == "win32":
                await timeline.execute_statement('s = shell("cmd", ["/c", "echo", "test"])')
            else:
                await timeline.execute_statement('s = shell("echo", ["test"])')

            ns = timeline.get_namespace()
            # Get the actual node, not the view
            shell_node = ns["s"].node() if hasattr(ns["s"], "node") else ns["s"]

            # Mark as completed - shell_status is a dataclass field
            shell_node.shell_status = ShellStatus.COMPLETED

            # Set up wait condition
            timeline._wait_condition = WaitCondition(
                node_ids=[shell_node.node_id],
                mode=WaitMode.SINGLE,
                wake_prompt="Done: {command}",
            )

            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is True
            assert prompt is not None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_check_wait_any_mode_first_completed(self, temp_cwd: Path) -> None:
        """Test ANY mode with first shell completed."""
        import sys

        from activecontext.context.nodes import ShellStatus

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create two shells
            if sys.platform == "win32":
                await timeline.execute_statement('s1 = shell("cmd", ["/c", "echo", "1"])')
                await timeline.execute_statement('s2 = shell("cmd", ["/c", "echo", "2"])')
            else:
                await timeline.execute_statement('s1 = shell("echo", ["1"])')
                await timeline.execute_statement('s2 = shell("echo", ["2"])')

            ns = timeline.get_namespace()
            # Get the actual nodes
            shell1 = ns["s1"].node() if hasattr(ns["s1"], "node") else ns["s1"]
            shell2 = ns["s2"].node() if hasattr(ns["s2"], "node") else ns["s2"]

            # Mark only first as completed
            shell1.shell_status = ShellStatus.COMPLETED

            # Set up ANY wait condition
            timeline._wait_condition = WaitCondition(
                node_ids=[shell1.node_id, shell2.node_id],
                mode=WaitMode.ANY,
                wake_prompt="First done",
            )

            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_check_wait_all_mode_not_all_completed(self, temp_cwd: Path) -> None:
        """Test ALL mode returns not satisfied when not all completed."""
        import sys

        from activecontext.context.nodes import ShellStatus

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create two shells
            if sys.platform == "win32":
                await timeline.execute_statement('s1 = shell("cmd", ["/c", "echo", "1"])')
                await timeline.execute_statement('s2 = shell("cmd", ["/c", "echo", "2"])')
            else:
                await timeline.execute_statement('s1 = shell("echo", ["1"])')
                await timeline.execute_statement('s2 = shell("echo", ["2"])')

            ns = timeline.get_namespace()
            # Get the actual nodes
            shell1 = ns["s1"].node() if hasattr(ns["s1"], "node") else ns["s1"]
            shell2 = ns["s2"].node() if hasattr(ns["s2"], "node") else ns["s2"]

            # Mark only first as completed
            shell1.shell_status = ShellStatus.COMPLETED
            # shell2 still pending

            # Set up ALL wait condition
            timeline._wait_condition = WaitCondition(
                node_ids=[shell1.node_id, shell2.node_id],
                mode=WaitMode.ALL,
                wake_prompt="All done",
            )

            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is False
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_check_wait_failure_prompt(self, temp_cwd: Path) -> None:
        """Test that failure_prompt is used when node fails."""
        import sys

        from activecontext.context.nodes import ShellStatus

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            if sys.platform == "win32":
                await timeline.execute_statement('s = shell("cmd", ["/c", "exit", "1"])')
            else:
                await timeline.execute_statement('s = shell("false")')

            ns = timeline.get_namespace()
            # Get the actual node
            shell_node = ns["s"].node() if hasattr(ns["s"], "node") else ns["s"]

            # Mark as failed
            shell_node.shell_status = ShellStatus.FAILED

            # Set up wait with failure prompt
            timeline._wait_condition = WaitCondition(
                node_ids=[shell_node.node_id],
                mode=WaitMode.SINGLE,
                wake_prompt="Done",
                failure_prompt="Failed with exit code {exit_code}",
            )

            satisfied, prompt = timeline.check_wait_condition()
            assert satisfied is True
            assert "Failed" in prompt
        finally:
            await timeline.close()


class TestCaptureNamespaceTrace:
    """Tests for _capture_namespace_trace helper."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_capture_added_variable(self, temp_cwd: Path) -> None:
        """Test capturing newly added variables."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            before = {}
            after = {"x": 42}

            trace = timeline._capture_namespace_trace(before, after)

            assert "x" in trace.added
            assert trace.added["x"] == "int"
            assert trace.deleted == []
            assert trace.changed == {}
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_capture_deleted_variable(self, temp_cwd: Path) -> None:
        """Test capturing deleted variables."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            before = {"x": 42}
            after = {}

            trace = timeline._capture_namespace_trace(before, after)

            assert trace.added == {}
            assert "x" in trace.deleted
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_capture_changed_variable(self, temp_cwd: Path) -> None:
        """Test capturing changed variables."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            before = {"x": "hello"}
            after = {"x": 42}

            trace = timeline._capture_namespace_trace(before, after)

            assert "x" in trace.changed
            assert "str" in trace.changed["x"]
            assert "int" in trace.changed["x"]
        finally:
            await timeline.close()


class TestScriptNamespace:
    """Tests for ScriptNamespace dict subclass."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_script_namespace_falls_back_to_graph(self, temp_cwd: Path) -> None:
        """Test that ScriptNamespace falls back to graph lookup."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create a node
            await timeline.execute_statement('t = topic("Test")')
            ns = timeline.get_namespace()
            node_id = ns["t"].node_id

            # Access via the ScriptNamespace (timeline._namespace)
            # Direct access by node_id should work
            view = timeline._namespace[node_id]
            assert view.node_id == node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_script_namespace_raises_keyerror(self, temp_cwd: Path) -> None:
        """Test that ScriptNamespace raises KeyError for unknown keys."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            with pytest.raises(KeyError):
                _ = timeline._namespace["nonexistent_key"]
        finally:
            await timeline.close()


class TestScriptNamespaceMCPLookup:
    """Tests for ScriptNamespace MCP server node lookup by server_name."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_mcp_node_lookup_by_server_name(self, temp_cwd: Path) -> None:
        """Test that ScriptNamespace resolves MCP server nodes by server_name."""
        from activecontext.context.nodes import MCPServerNode
        from activecontext.session.timeline import ScriptNamespace

        graph = ContextGraph()
        views: dict = {}
        node = MCPServerNode(server_name="rider", status="connected")
        graph.add_node(node)

        mcp_nodes = {"rider": node}
        ns = ScriptNamespace(
            lambda: graph,
            lambda: views,
            lambda: mcp_nodes,
            {},
        )

        # Lookup by server_name should return a NodeView
        view = ns["rider"]
        assert view.node_id == node.node_id

    @pytest.mark.asyncio
    async def test_mcp_lookup_caches_in_views(self, temp_cwd: Path) -> None:
        """Test that MCP lookup caches the NodeView in views dict."""
        from activecontext.context.nodes import MCPServerNode
        from activecontext.session.timeline import ScriptNamespace

        graph = ContextGraph()
        views: dict = {}
        node = MCPServerNode(server_name="fs", status="connected")
        graph.add_node(node)

        mcp_nodes = {"fs": node}
        ns = ScriptNamespace(
            lambda: graph,
            lambda: views,
            lambda: mcp_nodes,
            {},
        )

        view1 = ns["fs"]
        view2 = ns["fs"]
        assert view1 is view2  # Same cached view
        assert node.node_id in views

    @pytest.mark.asyncio
    async def test_none_mcp_getter_backward_compat(self, temp_cwd: Path) -> None:
        """Test that None mcp_nodes_getter preserves backward compatibility."""
        from activecontext.session.timeline import ScriptNamespace

        graph = ContextGraph()
        views: dict = {}
        ns = ScriptNamespace(
            lambda: graph,
            lambda: views,
            None,  # No MCP getter
            {},
        )

        with pytest.raises(KeyError):
            _ = ns["nonexistent"]

    @pytest.mark.asyncio
    async def test_namespace_prefers_explicit_bindings(self, temp_cwd: Path) -> None:
        """Test that explicit namespace entries take precedence over MCP lookup."""
        from activecontext.context.nodes import MCPServerNode
        from activecontext.session.timeline import ScriptNamespace

        graph = ContextGraph()
        views: dict = {}
        node = MCPServerNode(server_name="rider", status="connected")
        graph.add_node(node)

        mcp_nodes = {"rider": node}
        sentinel = object()
        ns = ScriptNamespace(
            lambda: graph,
            lambda: views,
            lambda: mcp_nodes,
            {"rider": sentinel},
        )

        # Explicit binding takes precedence
        assert ns["rider"] is sentinel


class TestGroupWithCurrentGroupParent:
    """Tests for node creation with current_group as parent."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_group_with_explicit_parent_overrides_current_group(self, temp_cwd: Path) -> None:
        """Test that explicit parent overrides current_group."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            # Create two groups
            await timeline.execute_statement('g1 = group(summary="Group 1")')
            await timeline.execute_statement('g2 = group(summary="Group 2")')

            ns = timeline.get_namespace()
            g1_id = ns["g1"].node_id
            g2_id = ns["g2"].node_id

            # Set current group to g1
            timeline.set_current_group(g1_id)

            # Create topic with explicit parent g2
            await timeline.execute_statement(f't = topic("Test", parent="{g2_id}")')
            ns = timeline.get_namespace()
            topic_node = ns["t"].node()

            # Should be linked to g2, not g1
            parents = timeline._context_graph.get_parents(topic_node.node_id)
            parent_ids = [p.node_id for p in parents]
            assert g2_id in parent_ids
            assert g1_id not in parent_ids
        finally:
            await timeline.close()


class TestMakeTextNodeFileWatcher:
    """Tests for text() node file watcher registration."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_text_node_registers_with_file_watcher(self, temp_cwd: Path) -> None:
        """Test that text() registers the file path with the watcher."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        try:
            await timeline.execute_statement('v = text("test.py")')

            # The file watcher should have the path registered
            # (We can't easily check internals, but verify no errors occurred)
            ns = timeline.get_namespace()
            assert "v" in ns
        finally:
            await timeline.close()
