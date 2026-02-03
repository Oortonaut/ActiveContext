"""Tests for the agent notification system.

Tests cover:
- NotificationLevel enum and Notification dataclass
- ContextNode notification flags and header formatting
- Push-based notification flag delivery to ancestors
- Session tick processing of notification flags
- notify() DSL function
"""

import time
from pathlib import Path

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    ArtifactNode,
    GroupNode,
    TextNode,
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
            header="text_1: file changed",
            level=NotificationLevel.WAKE,
        )
        assert notif.node_id == "test_node"
        assert notif.trace_id == "test_node:1"
        assert notif.header == "text_1: file changed"
        assert notif.level == NotificationLevel.WAKE

    def test_timestamp_auto_set(self) -> None:
        """Test that timestamp is automatically set."""
        before = time.time()
        notif = Notification(
            node_id="test",
            trace_id="test:1",
            header="test",
            level=NotificationLevel.HOLD,
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

    def test_notification_level_assignment(self) -> None:
        """Test direct notification_level assignment."""
        node = TextNode(path="test.py")
        node.notification_level = NotificationLevel.HOLD
        assert node.notification_level == NotificationLevel.HOLD

    def test_is_subscription_point_default(self) -> None:
        """Test default is_subscription_point is False."""
        node = TextNode(path="test.py")
        assert node.is_subscription_point is False

    def test_ignore_level_no_ancestor_flags(self) -> None:
        """Test that IGNORE level does not set flags on ancestors."""
        graph = ContextGraph()
        parent = GroupNode(node_id="parent")
        child = TextNode(
            node_id="child", path="test.py", notification_level=NotificationLevel.IGNORE
        )
        graph.add_node(parent)
        graph.add_node(child)
        graph.link("child", "parent")

        child.mark_changed("test")

        assert parent._notified is False
        assert parent._wake_notified is False

    def test_hold_level_sets_notified_flag(self) -> None:
        """Test that HOLD level sets _notified on ancestor nodes."""
        graph = ContextGraph()
        parent = GroupNode(node_id="parent")
        child = TextNode(node_id="child", path="test.py", notification_level=NotificationLevel.HOLD)
        graph.add_node(parent)
        graph.add_node(child)
        graph.link("child", "parent")

        child.mark_changed("test change")

        assert parent._notified is True
        assert parent._wake_notified is False

    def test_wake_level_sets_both_flags(self) -> None:
        """Test that WAKE level sets both _notified and _wake_notified."""
        graph = ContextGraph()
        parent = GroupNode(node_id="parent")
        child = TextNode(node_id="child", path="test.py", notification_level=NotificationLevel.WAKE)
        graph.add_node(parent)
        graph.add_node(child)
        graph.link("child", "parent")

        child.mark_changed("test change")

        assert parent._notified is True
        assert parent._wake_notified is True

    def test_flags_reach_all_ancestors(self) -> None:
        """Test that notification flags are set on all ancestors in the DAG."""
        graph = ContextGraph()
        grandparent = GroupNode(node_id="grandparent")
        parent = GroupNode(node_id="parent")
        child = TextNode(node_id="child", path="test.py", notification_level=NotificationLevel.HOLD)
        graph.add_node(grandparent)
        graph.add_node(parent)
        graph.add_node(child)
        graph.link("parent", "grandparent")
        graph.link("child", "parent")

        child.mark_changed("test change")

        assert parent._notified is True
        assert grandparent._notified is True

    def test_format_notification_header_default(self) -> None:
        """Test default header formatting."""
        node = ArtifactNode(node_id="artifact_5", content="test", artifact_type="code")

        header = node._format_notification_header("content updated")
        assert header == "artifact_5: content updated"

    def test_format_notification_header_textnode(self) -> None:
        """Test TextNode header formatting with position info."""
        node = TextNode(node_id="text_3", path="test.py", pos="10:0")

        header = node._format_notification_header("file modified")
        # Should show position info
        assert "text_3" in header
        assert "10:0" in header

    def test_format_notification_header_textnode_no_diff(self) -> None:
        """Test TextNode header with description."""
        node = TextNode(node_id="text_1", path="test.py", pos="1:0")

        header = node._format_notification_header("reloaded")
        assert "text_1" in header
        assert "reloaded" in header


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
    async def test_tick_clears_notification_flags(self, temp_cwd: Path) -> None:
        """Test that tick() clears notification flags from nodes."""
        from activecontext.session.session_manager import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=str(temp_cwd))

        try:
            graph = session._timeline.context_graph

            # Create parent and child with HOLD notification
            parent = GroupNode(node_id="test_parent")
            child = TextNode(
                node_id="test_child",
                path="test.py",
                notification_level=NotificationLevel.HOLD,
            )
            graph.add_node(parent)
            graph.add_node(child)
            graph.link("test_child", "test_parent")

            # Trigger a change — sets _notified on parent
            child.mark_changed("changed")
            assert parent._notified is True

            # Run tick — should clear flags
            await session.tick()
            assert parent._notified is False
            assert parent._wake_notified is False
        finally:
            await manager.close_session(session.session_id)

    @pytest.mark.asyncio
    async def test_wake_notification_sets_event(self, temp_cwd: Path) -> None:
        """Test that WAKE notification sets the wake event during tick."""
        from activecontext.session.session_manager import SessionManager

        manager = SessionManager()
        session = await manager.create_session(cwd=str(temp_cwd))

        try:
            # Clear wake event
            session._wake_event.clear()
            assert not session._wake_event.is_set()

            graph = session._timeline.context_graph

            # Create parent and child with WAKE level
            parent = GroupNode(node_id="test_parent")
            child = TextNode(
                node_id="test_child",
                path="test.py",
                notification_level=NotificationLevel.WAKE,
            )
            graph.add_node(parent)
            graph.add_node(child)
            graph.link("test_child", "test_parent")

            # Trigger change
            child.mark_changed("changed")

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

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

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

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

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

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

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

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

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

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement(
                'notify("nonexistent", NotificationLevel.WAKE)'
            )
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

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Should be able to use NotificationLevel directly
            await timeline.execute_statement("level = NotificationLevel.WAKE")
            assert timeline._namespace["level"] == NotificationLevel.WAKE
        finally:
            await timeline.close()
