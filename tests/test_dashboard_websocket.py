"""Comprehensive tests for the dashboard websocket ConnectionManager."""

import asyncio

import pytest

from activecontext.dashboard.websocket import ConnectionManager


class MockWebSocket:
    """Mock WebSocket for testing."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.accepted = False
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.sent_messages: list = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: dict) -> None:
        if self.should_fail:
            raise Exception("WebSocket connection failed")
        self.sent_messages.append(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason


@pytest.fixture
def manager() -> ConnectionManager:
    """Create a fresh ConnectionManager for each test."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket() -> MockWebSocket:
    """Create a mock websocket."""
    return MockWebSocket()


@pytest.fixture
def failing_websocket() -> MockWebSocket:
    """Create a mock websocket that fails on send."""
    return MockWebSocket(should_fail=True)


class TestConnectionManagerInit:
    """Tests for ConnectionManager initialization."""

    def test_init_creates_empty_connections(self, manager: ConnectionManager) -> None:
        """ConnectionManager should initialize with empty connections."""
        assert manager._connections == {}

    def test_init_creates_lock(self, manager: ConnectionManager) -> None:
        """ConnectionManager should initialize with an asyncio lock."""
        assert isinstance(manager._lock, asyncio.Lock)


class TestConnect:
    """Tests for the connect method."""

    @pytest.mark.asyncio
    async def test_connect_accepts_websocket(
        self, manager: ConnectionManager, mock_websocket: MockWebSocket
    ) -> None:
        """Connect should accept the websocket."""
        await manager.connect(mock_websocket, "session1")
        assert mock_websocket.accepted is True

    @pytest.mark.asyncio
    async def test_connect_adds_to_session(
        self, manager: ConnectionManager, mock_websocket: MockWebSocket
    ) -> None:
        """Connect should add websocket to the session's connection set."""
        await manager.connect(mock_websocket, "session1")
        assert mock_websocket in manager._connections["session1"]

    @pytest.mark.asyncio
    async def test_connect_multiple_to_same_session(
        self, manager: ConnectionManager
    ) -> None:
        """Multiple websockets can connect to the same session."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session1")
        await manager.connect(ws3, "session1")

        assert len(manager._connections["session1"]) == 3
        assert ws1 in manager._connections["session1"]
        assert ws2 in manager._connections["session1"]
        assert ws3 in manager._connections["session1"]

    @pytest.mark.asyncio
    async def test_connect_multiple_sessions(self, manager: ConnectionManager) -> None:
        """Websockets can connect to different sessions."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session2")

        assert ws1 in manager._connections["session1"]
        assert ws2 in manager._connections["session2"]
        assert len(manager._connections) == 2


class TestDisconnect:
    """Tests for the disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_removes_websocket(
        self, manager: ConnectionManager, mock_websocket: MockWebSocket
    ) -> None:
        """Disconnect should remove websocket from session."""
        await manager.connect(mock_websocket, "session1")
        await manager.disconnect(mock_websocket, "session1")

        assert mock_websocket not in manager._connections.get("session1", set())

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_empty_session(
        self, manager: ConnectionManager, mock_websocket: MockWebSocket
    ) -> None:
        """Disconnect should remove empty session entries."""
        await manager.connect(mock_websocket, "session1")
        await manager.disconnect(mock_websocket, "session1")

        assert "session1" not in manager._connections

    @pytest.mark.asyncio
    async def test_disconnect_preserves_other_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Disconnect should preserve other connections in same session."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session1")
        await manager.disconnect(ws1, "session1")

        assert ws1 not in manager._connections["session1"]
        assert ws2 in manager._connections["session1"]
        assert "session1" in manager._connections

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_session(
        self, manager: ConnectionManager, mock_websocket: MockWebSocket
    ) -> None:
        """Disconnect should handle nonexistent session gracefully."""
        # Should not raise
        await manager.disconnect(mock_websocket, "nonexistent")

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_websocket(
        self, manager: ConnectionManager
    ) -> None:
        """Disconnect should handle websocket not in session gracefully."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, "session1")
        # ws2 was never connected
        await manager.disconnect(ws2, "session1")

        # ws1 should still be there
        assert ws1 in manager._connections["session1"]


class TestBroadcast:
    """Tests for the broadcast method."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_session_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast should send message to all connections in session."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session1")
        await manager.connect(ws3, "session1")

        message = {"type": "update", "data": "test"}
        await manager.broadcast("session1", message)

        assert message in ws1.sent_messages
        assert message in ws2.sent_messages
        assert message in ws3.sent_messages

    @pytest.mark.asyncio
    async def test_broadcast_only_to_target_session(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast should only send to target session."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session2")

        message = {"type": "update"}
        await manager.broadcast("session1", message)

        assert message in ws1.sent_messages
        assert message not in ws2.sent_messages

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast should remove connections that fail to send."""
        ws_good = MockWebSocket()
        ws_dead = MockWebSocket(should_fail=True)

        await manager.connect(ws_good, "session1")
        await manager.connect(ws_dead, "session1")

        message = {"type": "update"}
        await manager.broadcast("session1", message)

        # Good websocket should receive message
        assert message in ws_good.sent_messages
        # Dead websocket should be removed
        assert ws_dead not in manager._connections["session1"]
        assert ws_good in manager._connections["session1"]

    @pytest.mark.asyncio
    async def test_broadcast_removes_all_dead_connections_in_session(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast should remove all dead connections from a session."""
        ws_dead1 = MockWebSocket(should_fail=True)
        ws_dead2 = MockWebSocket(should_fail=True)

        await manager.connect(ws_dead1, "session1")
        await manager.connect(ws_dead2, "session1")

        message = {"type": "update"}
        await manager.broadcast("session1", message)

        # Both dead connections should be removed (session key may remain with empty set)
        assert ws_dead1 not in manager._connections.get("session1", set())
        assert ws_dead2 not in manager._connections.get("session1", set())
        assert manager.get_connection_count("session1") == 0

    @pytest.mark.asyncio
    async def test_broadcast_nonexistent_session(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast to nonexistent session should do nothing."""
        # Should not raise
        await manager.broadcast("nonexistent", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_empty_session(self, manager: ConnectionManager) -> None:
        """Broadcast to session with no connections should do nothing."""
        manager._connections["session1"] = set()
        # Should not raise
        await manager.broadcast("session1", {"type": "test"})


class TestBroadcastAll:
    """Tests for the broadcast_all method."""

    @pytest.mark.asyncio
    async def test_broadcast_all_sends_to_all_sessions(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast_all should send to all connections in all sessions."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session2")
        await manager.connect(ws3, "session3")

        message = {"type": "global_update"}
        await manager.broadcast_all(message)

        assert message in ws1.sent_messages
        assert message in ws2.sent_messages
        assert message in ws3.sent_messages

    @pytest.mark.asyncio
    async def test_broadcast_all_with_no_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast_all with no connections should do nothing."""
        # Should not raise
        await manager.broadcast_all({"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_all_removes_dead_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast_all should remove dead connections from all sessions."""
        ws_good = MockWebSocket()
        ws_dead = MockWebSocket(should_fail=True)

        await manager.connect(ws_good, "session1")
        await manager.connect(ws_dead, "session2")

        message = {"type": "global"}
        await manager.broadcast_all(message)

        # Good connection should remain
        assert ws_good in manager._connections["session1"]
        # Dead connection should be removed (session key may remain with empty set)
        assert ws_dead not in manager._connections.get("session2", set())
        assert manager.get_connection_count("session2") == 0


class TestGetConnectionCount:
    """Tests for the get_connection_count method."""

    @pytest.mark.asyncio
    async def test_get_connection_count_empty(
        self, manager: ConnectionManager
    ) -> None:
        """Get connection count with no connections should return 0."""
        assert manager.get_connection_count() == 0
        assert manager.get_connection_count("session1") == 0

    @pytest.mark.asyncio
    async def test_get_connection_count_specific_session(
        self, manager: ConnectionManager
    ) -> None:
        """Get connection count for specific session."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session1")
        await manager.connect(ws3, "session2")

        assert manager.get_connection_count("session1") == 2
        assert manager.get_connection_count("session2") == 1

    @pytest.mark.asyncio
    async def test_get_connection_count_total(
        self, manager: ConnectionManager
    ) -> None:
        """Get total connection count across all sessions."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session1")
        await manager.connect(ws3, "session2")

        assert manager.get_connection_count() == 3

    @pytest.mark.asyncio
    async def test_get_connection_count_nonexistent_session(
        self, manager: ConnectionManager
    ) -> None:
        """Get connection count for nonexistent session should return 0."""
        ws = MockWebSocket()
        await manager.connect(ws, "session1")

        assert manager.get_connection_count("nonexistent") == 0


class TestCloseAll:
    """Tests for the close_all method."""

    @pytest.mark.asyncio
    async def test_close_all_closes_all_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Close_all should close all websocket connections."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session1")
        await manager.connect(ws3, "session2")

        await manager.close_all("Server shutdown")

        assert ws1.closed is True
        assert ws2.closed is True
        assert ws3.closed is True

    @pytest.mark.asyncio
    async def test_close_all_passes_reason(
        self, manager: ConnectionManager
    ) -> None:
        """Close_all should pass the reason to websocket close."""
        ws = MockWebSocket()
        await manager.connect(ws, "session1")

        await manager.close_all("Test shutdown reason")

        assert ws.close_reason == "Test shutdown reason"

    @pytest.mark.asyncio
    async def test_close_all_clears_connections(
        self, manager: ConnectionManager
    ) -> None:
        """Close_all should clear all connections."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()

        await manager.connect(ws1, "session1")
        await manager.connect(ws2, "session2")

        await manager.close_all("shutdown")

        assert manager._connections == {}

    @pytest.mark.asyncio
    async def test_close_all_empty(self, manager: ConnectionManager) -> None:
        """Close_all with no connections should do nothing."""
        # Should not raise
        await manager.close_all("shutdown")
        assert manager._connections == {}

    @pytest.mark.asyncio
    async def test_close_all_handles_close_exceptions(
        self, manager: ConnectionManager
    ) -> None:
        """Close_all should handle exceptions during close gracefully."""

        class FailingCloseWebSocket(MockWebSocket):
            async def close(self, code: int = 1000, reason: str = "") -> None:
                raise Exception("Close failed")

        ws_good = MockWebSocket()
        ws_failing = FailingCloseWebSocket()

        await manager.connect(ws_good, "session1")
        await manager.connect(ws_failing, "session2")

        # Should not raise, should still close good websocket
        await manager.close_all("shutdown")

        assert ws_good.closed is True
        assert manager._connections == {}


class TestConcurrency:
    """Tests for concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_connects(self, manager: ConnectionManager) -> None:
        """Multiple concurrent connects should be thread-safe."""
        websockets = [MockWebSocket() for _ in range(10)]

        async def connect_ws(ws: MockWebSocket, session_id: str) -> None:
            await manager.connect(ws, session_id)

        # Connect all websockets concurrently
        await asyncio.gather(
            *[connect_ws(ws, f"session{i % 3}") for i, ws in enumerate(websockets)]
        )

        # Verify all connections were made
        total = manager.get_connection_count()
        assert total == 10

    @pytest.mark.asyncio
    async def test_concurrent_disconnects(self, manager: ConnectionManager) -> None:
        """Multiple concurrent disconnects should be thread-safe."""
        websockets = [MockWebSocket() for _ in range(10)]

        # First, connect all
        for i, ws in enumerate(websockets):
            await manager.connect(ws, f"session{i % 3}")

        async def disconnect_ws(ws: MockWebSocket, session_id: str) -> None:
            await manager.disconnect(ws, session_id)

        # Disconnect all concurrently
        await asyncio.gather(
            *[disconnect_ws(ws, f"session{i % 3}") for i, ws in enumerate(websockets)]
        )

        # Verify all disconnected
        assert manager.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(self, manager: ConnectionManager) -> None:
        """Multiple concurrent broadcasts should be thread-safe."""
        websockets = [MockWebSocket() for _ in range(5)]

        for ws in websockets:
            await manager.connect(ws, "session1")

        async def do_broadcast(msg_id: int) -> None:
            await manager.broadcast("session1", {"id": msg_id})

        # Broadcast concurrently
        await asyncio.gather(*[do_broadcast(i) for i in range(10)])

        # Each websocket should have received all 10 messages
        for ws in websockets:
            assert len(ws.sent_messages) == 10

    @pytest.mark.asyncio
    async def test_concurrent_connect_and_broadcast(
        self, manager: ConnectionManager
    ) -> None:
        """Concurrent connects and broadcasts should be thread-safe."""
        websockets = [MockWebSocket() for _ in range(5)]

        async def connect_and_broadcast(ws: MockWebSocket, idx: int) -> None:
            await manager.connect(ws, "session1")
            await manager.broadcast("session1", {"from": idx})

        await asyncio.gather(
            *[connect_and_broadcast(ws, i) for i, ws in enumerate(websockets)]
        )

        # All should be connected
        assert manager.get_connection_count("session1") == 5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_same_websocket_multiple_sessions(
        self, manager: ConnectionManager
    ) -> None:
        """Same websocket can be in multiple sessions."""
        ws = MockWebSocket()

        await manager.connect(ws, "session1")
        await manager.connect(ws, "session2")

        assert ws in manager._connections["session1"]
        assert ws in manager._connections["session2"]
        assert manager.get_connection_count() == 2

    @pytest.mark.asyncio
    async def test_reconnect_same_websocket(
        self, manager: ConnectionManager
    ) -> None:
        """Reconnecting same websocket to same session should work."""
        ws = MockWebSocket()

        await manager.connect(ws, "session1")
        await manager.connect(ws, "session1")

        # Set should deduplicate
        assert manager.get_connection_count("session1") == 1

    @pytest.mark.asyncio
    async def test_broadcast_with_mixed_healthy_dead(
        self, manager: ConnectionManager
    ) -> None:
        """Broadcast with mix of healthy and dead connections."""
        ws_good1 = MockWebSocket()
        ws_dead1 = MockWebSocket(should_fail=True)
        ws_good2 = MockWebSocket()
        ws_dead2 = MockWebSocket(should_fail=True)

        await manager.connect(ws_good1, "session1")
        await manager.connect(ws_dead1, "session1")
        await manager.connect(ws_good2, "session1")
        await manager.connect(ws_dead2, "session1")

        message = {"type": "test"}
        await manager.broadcast("session1", message)

        # Good ones should receive message and stay
        assert message in ws_good1.sent_messages
        assert message in ws_good2.sent_messages
        assert ws_good1 in manager._connections["session1"]
        assert ws_good2 in manager._connections["session1"]

        # Dead ones should be removed
        assert ws_dead1 not in manager._connections["session1"]
        assert ws_dead2 not in manager._connections["session1"]

        assert manager.get_connection_count("session1") == 2

    @pytest.mark.asyncio
    async def test_empty_message_broadcast(self, manager: ConnectionManager) -> None:
        """Broadcasting empty message should work."""
        ws = MockWebSocket()
        await manager.connect(ws, "session1")

        await manager.broadcast("session1", {})

        assert {} in ws.sent_messages

    @pytest.mark.asyncio
    async def test_large_message_broadcast(self, manager: ConnectionManager) -> None:
        """Broadcasting large message should work."""
        ws = MockWebSocket()
        await manager.connect(ws, "session1")

        large_message = {"data": "x" * 100000}
        await manager.broadcast("session1", large_message)

        assert large_message in ws.sent_messages

    @pytest.mark.asyncio
    async def test_special_session_ids(self, manager: ConnectionManager) -> None:
        """Session IDs with special characters should work."""
        ws = MockWebSocket()

        special_ids = [
            "",
            "session with spaces",
            "session/with/slashes",
            "session:with:colons",
            "session-with-dashes_and_underscores",
            "日本語セッション",
        ]

        for session_id in special_ids:
            await manager.connect(ws, session_id)
            assert ws in manager._connections[session_id]
            await manager.disconnect(ws, session_id)

    @pytest.mark.asyncio
    async def test_get_connection_count_after_operations(
        self, manager: ConnectionManager
    ) -> None:
        """Connection count should be accurate after various operations."""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket(should_fail=True)

        # Initial state
        assert manager.get_connection_count() == 0

        # After connects
        await manager.connect(ws1, "session1")
        assert manager.get_connection_count() == 1

        await manager.connect(ws2, "session1")
        assert manager.get_connection_count() == 2

        await manager.connect(ws3, "session2")
        assert manager.get_connection_count() == 3

        # After disconnect
        await manager.disconnect(ws1, "session1")
        assert manager.get_connection_count() == 2

        # After broadcast removes dead connection
        await manager.broadcast("session2", {"test": True})
        assert manager.get_connection_count() == 1

        # After close_all
        await manager.close_all("done")
        assert manager.get_connection_count() == 0
