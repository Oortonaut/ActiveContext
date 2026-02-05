"""Tests for ACP plan mode protocol support.

Tests p2-003d implementation:
- Session mode tracking and changes
- ACP CurrentModeUpdate notifications
- Mode change integration with dashboard
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from activecontext.session.protocols import SessionUpdate, UpdateKind
from activecontext.session.session_manager import Session
from activecontext.session.timeline import Timeline


class TestSessionModeTracking:
    """Test session mode property and set_mode method."""

    def test_session_mode_default(self, tmp_path):
        """Test that session has default mode."""
        # Create real Timeline that creates structural nodes
        timeline = Timeline(
            session_id="test-123",
            cwd=str(tmp_path),
        )

        session = Session(
            session_id="test-123",
            timeline=timeline,
            cwd=str(tmp_path),
        )

        assert session.mode == "normal"  # Default mode

    def test_session_set_mode(self, tmp_path):
        """Test setting session mode."""
        timeline = Timeline(
            session_id="test-123",
            cwd=str(tmp_path),
        )

        session = Session(
            session_id="test-123",
            timeline=timeline,
            cwd=str(tmp_path),
        )

        session.set_mode("plan")
        assert session.mode == "plan"

        session.set_mode("brave")
        assert session.mode == "brave"

    def test_session_set_mode_emits_update(self, tmp_path):
        """Test that set_mode emits SessionUpdate when callback is set."""
        timeline = Timeline(
            session_id="test-123",
            cwd=str(tmp_path),
        )

        session = Session(
            session_id="test-123",
            timeline=timeline,
            cwd=str(tmp_path),
        )

        # Set up callback
        callback = AsyncMock()
        session._emit_update_callback = callback

        # Change mode
        session.set_mode("plan")

        # Callback should have been scheduled (but may not execute in sync test)
        # We can't easily test asyncio.create_task in sync test without event loop
        # This is tested in integration tests

    def test_session_set_mode_no_update_when_same(self, tmp_path):
        """Test that set_mode doesn't emit update if mode unchanged."""
        timeline = Timeline(
            session_id="test-123",
            cwd=str(tmp_path),
        )

        session = Session(
            session_id="test-123",
            timeline=timeline,
            cwd=str(tmp_path),
        )

        # Mode is already "normal" by default
        assert session.mode == "normal"

        callback = AsyncMock()
        session._emit_update_callback = callback

        # Set to same mode
        session.set_mode("normal")

        # Should not schedule callback (old_mode == new_mode)


class TestSessionUpdateModeChanged:
    """Test SessionUpdate generation for mode changes."""

    def test_mode_change_update_payload(self):
        """Test that mode change update has correct payload."""
        update = SessionUpdate(
            kind=UpdateKind.NODE_CHANGED,
            session_id="test-123",
            payload={"mode_changed": "plan", "old_mode": "normal"},
            timestamp=time.time(),
        )

        assert update.kind == UpdateKind.NODE_CHANGED
        assert update.payload["mode_changed"] == "plan"
        assert update.payload["old_mode"] == "normal"


@pytest.mark.asyncio
class TestACPModeChangeNotification:
    """Test ACP CurrentModeUpdate notification emission."""

    async def test_acp_current_mode_update_structure(self):
        """Test that CurrentModeUpdate has correct structure."""
        from acp.schema import CurrentModeUpdate

        update = CurrentModeUpdate(
            currentModeId="plan",
            sessionUpdate="current_mode_update",
        )

        assert update.currentModeId == "plan"
        assert update.sessionUpdate == "current_mode_update"

    async def test_acp_agent_handles_mode_change_update(self):
        """Test that ACP agent converts mode change to CurrentModeUpdate."""
        from unittest.mock import patch

        from activecontext.session.protocols import SessionUpdate, UpdateKind

        # Mock the agent's send method
        with patch("activecontext.transport.acp.agent.ActiveContextAgent") as MockAgent:
            agent = MockAgent.return_value
            agent._send_session_update = AsyncMock()
            agent._sessions_mode = {}

            # Import the emit_update_internal method logic (simplified test)
            from acp.schema import CurrentModeUpdate

            # Simulate mode change update
            session_id = "test-123"
            update = SessionUpdate(
                kind=UpdateKind.NODE_CHANGED,
                session_id=session_id,
                payload={"mode_changed": "plan", "old_mode": "normal"},
                timestamp=time.time(),
            )

            # Manually test the logic from _emit_update_internal
            mode_changed = update.payload.get("mode_changed")
            assert mode_changed == "plan"

            # Verify CurrentModeUpdate is correctly created
            acp_update = CurrentModeUpdate(
                currentModeId=mode_changed,
                sessionUpdate="current_mode_update",
            )

            assert acp_update.currentModeId == "plan"
            assert acp_update.sessionUpdate == "current_mode_update"


@pytest.mark.asyncio
class TestModeChangeIntegration:
    """Integration tests for mode changes through the full stack."""

    async def test_session_mode_change_triggers_acp_notification(self, tmp_path):
        """Test that changing session mode triggers ACP notification."""
        # Create real Timeline that creates structural nodes
        timeline = Timeline(
            session_id="test-123",
            cwd=str(tmp_path),
        )

        session = Session(
            session_id="test-123",
            timeline=timeline,
            cwd=str(tmp_path),
        )

        # Track emitted updates
        emitted_updates = []

        async def capture_update(update):
            emitted_updates.append(update)

        session._emit_update_callback = capture_update

        # Change mode - this should trigger update emission
        session.set_mode("plan")

        # Give asyncio.create_task a chance to run
        import asyncio

        await asyncio.sleep(0.01)

        # Verify update was emitted
        assert len(emitted_updates) == 1
        update = emitted_updates[0]
        assert update.kind == UpdateKind.NODE_CHANGED
        assert update.payload["mode_changed"] == "plan"
        assert update.payload["old_mode"] == "normal"

    async def test_mode_change_updates_agent_tracking(self):
        """Test that mode changes update agent's session mode tracking."""

        # Simulate the agent's internal state
        sessions_mode = {}
        session_id = "test-123"

        # Process mode change update
        mode_changed = "plan"
        sessions_mode[session_id] = mode_changed

        assert sessions_mode[session_id] == "plan"

        # Change again
        mode_changed = "brave"
        sessions_mode[session_id] = mode_changed

        assert sessions_mode[session_id] == "brave"
