"""Tests for the agent notification system.

Tests cover:
- NotificationLevel enum and Notification dataclass
- ContextNode notification emission and header formatting
- ContextGraph notification collection and deduplication
- Session Alerts group and tick() processing
- notify() DSL function
"""

import asyncio
import time
from pathlib import Path

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    ArtifactNode,
    ContextNode,
    GroupNode,
    TextNode,
    Trace,
)
from activecontext.context.state import Notification, NotificationLevel


# =============================================================================
# NotificationLevel and Notification Tests
# =============================================================================


class TestNotificationLevel:
    """Tests for NotificationLevel enum."""

    def test_enum_values(self) -> None:
        """Test that all expected enum values exist."""
        assert NotificationLevel.IGNORE.value == "ignore"
        assert NotificationLevel.HOLD.value == "hold"
        assert NotificationLevel.WAKE.value == "wake"

    def test_str_conversion(self) -> None:
        """Test string conversion."""
        assert str(NotificationLevel.IGNORE) == "ignore"
        assert str(NotificationLevel.HOLD) == "hold"
        assert str(NotificationLevel.WAKE) == "wake"

    def test_from_string(self) -> None:
        """Test creating enum from string value."""
        assert NotificationLevel("ignore") == NotificationLevel.IGNORE
        assert NotificationLevel("hold") == NotificationLevel.HOLD
        assert NotificationLevel("wake") == NotificationLevel.WAKE


class TestNotification:
    """Tests for Notification dataclass."""

    def test_creation(self) -> None:
        """Test basic notification creation."""
        notif = Notification(
            node_id="test_node",
            trace_id="test_node:1",
            header="text#1: file changed",
            level="wake",
        )
        assert notif.node_id == "test_node"
        assert notif.trace_id == "test_node:1"
        assert notif.header == "text#1: file changed"
        assert notif.level == "wake"

    def test_timestamp_auto_set(self) -> None:
        """Test that timestamp is automatically set."""
        before = time.time()
        notif = Notification(
            node_id="test",
            trace_id="test:1",
            header="test",
            level="hold",
        )
        after = time.time()
        assert before <= notif.timestamp <= after


# =============================================================================
# ContextNode Notification Tests
# =============================================================================


class TestContextNodeNotification:
    """Tests for ContextNode notification behavior."""

    def test_default_notification_level(self) -> None:
        """Test that default notification level is IGNORE."""
        node = TextNode(path="test.py")
        assert node.notification_level == NotificationLevel.IGNORE

    def test_set_notification_level(self) -> None:
        """Test setting notification level directly."""
        node = TextNode(path="test.py")
        node.notification_level = NotificationLevel.WAKE
        assert node.notification_level == NotificationLevel.WAKE

    def test_set_notify_fluent(self) -> None:
        """Test SetNotify fluent API."""
        node = TextNode(path="test.py")
        result = node.SetNotify(NotificationLevel.HOLD)
        assert result is node  # Returns self for chaining
        assert node.notification_level == NotificationLevel.HOLD

    def test_is_subscription_point_default(self) -> None:
        """Test default is_subscription_point is False."""
        node = TextNode(path="test.py")
        assert node.is_subscription_point is False

    def test_emit_notification_when_ignore(self) -> None:
        """Test that IGNORE level does not emit notifications."""
        graph = ContextGraph()
        node = TextNode(path="test.py", notification_level=NotificationLevel.IGNORE)
        graph.add_node(node)

        trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="test")
        node._mark_changed(trace)

        # No notifications should be emitted
        assert len(graph._pending_notifications) == 0

    def test_emit_notification_when_hold(self) -> None:
        """Test that HOLD level emits notifications."""
        graph = ContextGraph()
        node = TextNode(path="test.py", notification_level=NotificationLevel.HOLD)
        graph.add_node(node)

        trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="test change")
        node._mark_changed(trace)

        assert len(graph._pending_notifications) == 1
        notif = graph._pending_notifications[0]
        assert notif.node_id == node.node_id
        assert notif.level == "hold"

    def test_emit_notification_when_wake(self) -> None:
        """Test that WAKE level emits notifications and sets wake flag."""
        graph = ContextGraph()
        node = TextNode(path="test.py", notification_level=NotificationLevel.WAKE)
        graph.add_node(node)

        trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="test change")
        node._mark_changed(trace)

        assert len(graph._pending_notifications) == 1
        assert graph.has_wake_notification() is True
        notif = graph._pending_notifications[0]
        assert notif.level == "wake"

    def test_format_notification_header_default(self) -> None:
        """Test default header formatting."""
        node = ArtifactNode(content="test", artifact_type="code")
        node.display_sequence = 5
        trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="content updated")

        header = node._format_notification_header(trace)
        assert header == "artifact#5: content updated"

    def test_format_notification_header_textnode(self) -> None:
        """Test TextNode header formatting with line changes."""
        node = TextNode(path="test.py", pos="10:0")
        node.display_sequence = 3

        # Trace with diff-like content
        trace = Trace(
            node_id=node.node_id,
            old_version=0,
            new_version=1,
            description="file modified",
            content="+line1\n+line2\n-removed\n+added",
        )

        header = node._format_notification_header(trace)
        # Should show line changes: 3 added, 1 removed
        assert "text#3" in header
        assert "+3" in header or "3" in header  # Added lines
        assert "-1" in header or "1" in header  # Removed lines

    def test_format_notification_header_textnode_no_diff(self) -> None:
        """Test TextNode header without diff content falls back to description."""
        node = TextNode(path="test.py", pos="1:0")
        node.display_sequence = 1
        trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="reloaded")

        header = node._format_notification_header(trace)
        assert header == "text#1: reloaded"


# =============================================================================
# ContextGraph Notification Tests
# =============================================================================


class TestContextGraphNotification:
    """Tests for ContextGraph notification collection."""

    def test_emit_notification_adds_to_queue(self) -> None:
        """Test that emit_notification adds to pending list."""
        graph = ContextGraph()
        graph.emit_notification(
            node_id="test",
            trace_id="test:1",
            header="test header",
            level=NotificationLevel.HOLD,
        )

        assert len(graph._pending_notifications) == 1
        notif = graph._pending_notifications[0]
        assert notif.node_id == "test"
        assert notif.trace_id == "test:1"
        assert notif.header == "test header"
        assert notif.level == "hold"

    def test_deduplication_by_trace_id(self) -> None:
        """Test that same trace_id is not added twice."""
        graph = ContextGraph()

        # Emit same trace twice
        graph.emit_notification("test", "test:1", "header1", NotificationLevel.HOLD)
        graph.emit_notification("test", "test:1", "header2", NotificationLevel.HOLD)

        # Should only have one notification
        assert len(graph._pending_notifications) == 1
        assert graph._pending_notifications[0].header == "header1"

    def test_different_trace_ids_both_added(self) -> None:
        """Test that different trace_ids are both added."""
        graph = ContextGraph()

        graph.emit_notification("test", "test:1", "header1", NotificationLevel.HOLD)
        graph.emit_notification("test", "test:2", "header2", NotificationLevel.HOLD)

        assert len(graph._pending_notifications) == 2

    def test_has_wake_notification_false_initially(self) -> None:
        """Test that has_wake_notification is False initially."""
        graph = ContextGraph()
        assert graph.has_wake_notification() is False

    def test_has_wake_notification_true_after_wake(self) -> None:
        """Test that has_wake_notification is True after WAKE notification."""
        graph = ContextGraph()
        graph.emit_notification("test", "test:1", "header", NotificationLevel.WAKE)
        assert graph.has_wake_notification() is True

    def test_has_wake_notification_false_after_hold(self) -> None:
        """Test that has_wake_notification stays False for HOLD."""
        graph = ContextGraph()
        graph.emit_notification("test", "test:1", "header", NotificationLevel.HOLD)
        assert graph.has_wake_notification() is False

    def test_flush_notifications_returns_all(self) -> None:
        """Test that flush_notifications returns all pending."""
        graph = ContextGraph()
        graph.emit_notification("n1", "n1:1", "h1", NotificationLevel.HOLD)
        graph.emit_notification("n2", "n2:1", "h2", NotificationLevel.WAKE)

        notifications = graph.flush_notifications()

        assert len(notifications) == 2
        assert notifications[0].node_id == "n1"
        assert notifications[1].node_id == "n2"

    def test_flush_notifications_clears_state(self) -> None:
        """Test that flush_notifications clears all state."""
        graph = ContextGraph()
        graph.emit_notification("test", "test:1", "header", NotificationLevel.WAKE)

        assert len(graph._pending_notifications) == 1
        assert graph.has_wake_notification() is True
        assert len(graph._seen_traces) == 1

        graph.flush_notifications()

        assert len(graph._pending_notifications) == 0
        assert graph.has_wake_notification() is False
        assert len(graph._seen_traces) == 0

    def test_flush_allows_same_trace_again(self) -> None:
        """Test that after flush, same trace_id can be added again."""
        graph = ContextGraph()

        graph.emit_notification("test", "test:1", "header", NotificationLevel.HOLD)
        graph.flush_notifications()

        # Same trace_id should work again after flush
        graph.emit_notification("test", "test:1", "header", NotificationLevel.HOLD)
        assert len(graph._pending_notifications) == 1


# =============================================================================
# Session Integration Tests
# =============================================================================


class TestSessionNotificationIntegration:
    """Tests for Session notification integration."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_alerts_group_created(self, temp_cwd: Path) -> None:
        """Test that Alerts group is created in session."""
        from activecontext.session.session_manager import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=str(temp_cwd))

        try:
            # Check alerts group exists
            assert session._alerts_group is not None
            assert session._alerts_group.node_id == "alerts"

            # Check it's linked to root
            graph = session._timeline.context_graph
            alerts = graph.get_node("alerts")
            assert alerts is not None
            assert "context" in alerts.parent_ids
        finally:
            await manager.close_session(session.session_id)

    @pytest.mark.asyncio
    async def test_tick_processes_notifications(self, temp_cwd: Path) -> None:
        """Test that tick() processes notifications into Alerts group."""
        from activecontext.session.session_manager import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=str(temp_cwd))

        try:
            # Create a node with HOLD notification level
            graph = session._timeline.context_graph
            node = TextNode(path="test.py", notification_level=NotificationLevel.HOLD)
            graph.add_node(node)

            # Trigger a change
            trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="changed")
            node._mark_changed(trace)

            # Run tick
            await session.tick()

            # Check alerts group has content
            alerts_children = graph.get_children("alerts")
            assert len(alerts_children) == 1
            assert alerts_children[0].content == node._format_notification_header(trace)
        finally:
            await manager.close_session(session.session_id)

    @pytest.mark.asyncio
    async def test_wake_notification_sets_event(self, temp_cwd: Path) -> None:
        """Test that WAKE notification sets the wake event."""
        from activecontext.session.session_manager import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=str(temp_cwd))

        try:
            # Clear wake event
            session._wake_event.clear()
            assert not session._wake_event.is_set()

            # Create node with WAKE level
            graph = session._timeline.context_graph
            node = TextNode(path="test.py", notification_level=NotificationLevel.WAKE)
            graph.add_node(node)

            # Trigger change
            trace = Trace(node_id=node.node_id, old_version=0, new_version=1, description="changed")
            node._mark_changed(trace)

            # Run tick
            await session.tick()

            # Wake event should be set
            assert session._wake_event.is_set()
        finally:
            await manager.close_session(session.session_id)


# =============================================================================
# DSL Function Tests
# =============================================================================


class TestNotifyDSLFunction:
    """Tests for notify() DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_notify_sets_level(self, temp_cwd: Path) -> None:
        """Test that notify() sets notification level."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Create a node
            await timeline.execute_statement('v = text("test.py")')
            ns = timeline._namespace

            # Check default level
            assert ns["v"].notification_level == NotificationLevel.IGNORE

            # Use notify()
            await timeline.execute_statement("notify(v, NotificationLevel.WAKE)")
            assert ns["v"].notification_level == NotificationLevel.WAKE
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_with_string_level(self, temp_cwd: Path) -> None:
        """Test that notify() accepts string level."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement('notify(v, "hold")')

            assert timeline._namespace["v"].notification_level == NotificationLevel.HOLD
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_with_node_id_string(self, temp_cwd: Path) -> None:
        """Test that notify() accepts node_id string."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            node_id = timeline._namespace["v"].node_id

            await timeline.execute_statement(f'notify("{node_id}", NotificationLevel.WAKE)')

            assert timeline._namespace["v"].notification_level == NotificationLevel.WAKE
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_returns_node(self, temp_cwd: Path) -> None:
        """Test that notify() returns the node for chaining."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("result = notify(v, NotificationLevel.HOLD)")

            # Result should be the same node
            assert timeline._namespace["result"] is timeline._namespace["v"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notify_invalid_node_raises(self, temp_cwd: Path) -> None:
        """Test that notify() with invalid node_id raises ValueError."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('notify("nonexistent", NotificationLevel.WAKE)')
            assert result.status.value == "error"
            assert result.exception is not None
            # Check exception message contains "not found"
            exc_msg = str(result.exception.get("message", "")).lower()
            assert "not found" in exc_msg or "nonexistent" in exc_msg
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_notification_level_in_namespace(self, temp_cwd: Path) -> None:
        """Test that NotificationLevel is available in namespace."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        try:
            # Should be able to use NotificationLevel directly
            await timeline.execute_statement("level = NotificationLevel.WAKE")
            assert timeline._namespace["level"] == NotificationLevel.WAKE
        finally:
            await timeline.close()
