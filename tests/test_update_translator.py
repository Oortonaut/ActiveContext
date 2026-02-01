"""Tests for LSP update translator."""

from __future__ import annotations

from typing import Any

import pytest

from activecontext.session.protocols import SessionUpdate, UpdateKind
from activecontext.transport.lsp.update_translator import UpdateTranslator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLSPServer:
    """Mock LSP server that captures sent notifications."""

    def __init__(self) -> None:
        self.notifications: list[tuple[str, Any]] = []

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Record notification for later inspection."""
        self.notifications.append((method, params))

    def get_notification(self, index: int = 0) -> tuple[str, Any]:
        """Get notification by index."""
        return self.notifications[index]

    def get_notifications_by_method(self, method: str) -> list[Any]:
        """Get all params for notifications with the given method."""
        return [params for m, params in self.notifications if m == method]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResponseChunk:
    """Test RESPONSE_CHUNK update translation."""

    @pytest.mark.asyncio
    async def test_response_chunk_lsp_aware(self) -> None:
        """LSP-aware mode sends $/activeContext/output."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=True)

        update = SessionUpdate(
            kind=UpdateKind.RESPONSE_CHUNK,
            session_id="sess-1",
            payload={"chunk": "Hello world"},
            timestamp=1234.5,
        )

        await translator.translate(update)

        assert len(server.notifications) == 1
        method, params = server.get_notification(0)
        assert method == "$/activeContext/output"
        assert params["sessionId"] == "sess-1"
        assert params["content"] == "Hello world"
        assert params["timestamp"] == 1234.5

    @pytest.mark.asyncio
    async def test_response_chunk_standard_mode(self) -> None:
        """Standard mode logs but doesn't send notification."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        update = SessionUpdate(
            kind=UpdateKind.RESPONSE_CHUNK,
            session_id="sess-1",
            payload={"chunk": "Hello world"},
        )

        await translator.translate(update)

        # No notifications sent in standard mode
        assert len(server.notifications) == 0


class TestStatementExecution:
    """Test STATEMENT_EXECUTING and STATEMENT_EXECUTED translation."""

    @pytest.mark.asyncio
    async def test_statement_executing(self) -> None:
        """STATEMENT_EXECUTING sends $/progress begin."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        update = SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTING,
            session_id="sess-1",
            payload={
                "statement_id": "stmt-42",
                "source": "v = text('main.py', tokens=2000)",
            },
        )

        await translator.translate(update)

        assert len(server.notifications) == 1
        method, params = server.get_notification(0)
        assert method == "$/progress"
        assert params["token"] == "statement-stmt-42"
        assert params["value"]["kind"] == "begin"
        assert params["value"]["title"] == "Executing statement"
        assert "text('main.py'" in params["value"]["message"]

    @pytest.mark.asyncio
    async def test_statement_executed_success(self) -> None:
        """STATEMENT_EXECUTED with ok status sends $/progress end."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        # First, executing
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTING,
                session_id="sess-1",
                payload={"statement_id": "stmt-42", "source": "print('hi')"},
            )
        )

        # Then executed
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTED,
                session_id="sess-1",
                payload={"statement_id": "stmt-42", "status": "ok"},
            )
        )

        assert len(server.notifications) == 2
        method, params = server.get_notification(1)
        assert method == "$/progress"
        assert params["token"] == "statement-stmt-42"
        assert params["value"]["kind"] == "end"
        assert params["value"]["message"] == "Completed"

    @pytest.mark.asyncio
    async def test_statement_executed_error(self) -> None:
        """STATEMENT_EXECUTED with error status includes error in message."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        # Executing
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTING,
                session_id="sess-1",
                payload={"statement_id": "stmt-99", "source": "bad_function()"},
            )
        )

        # Executed with error
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTED,
                session_id="sess-1",
                payload={
                    "statement_id": "stmt-99",
                    "status": "error",
                    "error": "NameError: name 'bad_function' is not defined",
                },
            )
        )

        assert len(server.notifications) == 2
        method, params = server.get_notification(1)
        assert method == "$/progress"
        assert params["value"]["kind"] == "end"
        assert "NameError" in params["value"]["message"]

    @pytest.mark.asyncio
    async def test_statement_long_source_truncated(self) -> None:
        """Long statement source is truncated in progress message."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        long_source = "x = " + "a" * 200

        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTING,
                session_id="sess-1",
                payload={"statement_id": "stmt-1", "source": long_source},
            )
        )

        method, params = server.get_notification(0)
        # Should be truncated to 100 chars
        assert len(params["value"]["message"]) == 100


class TestProjectionReady:
    """Test PROJECTION_READY translation."""

    @pytest.mark.asyncio
    async def test_projection_ready_lsp_aware(self) -> None:
        """LSP-aware mode sends $/activeContext/projectionReady."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=True)

        update = SessionUpdate(
            kind=UpdateKind.PROJECTION_READY,
            session_id="sess-1",
            payload={"projection": {"content": "...", "tokens": 500}},
            timestamp=9999.0,
        )

        await translator.translate(update)

        assert len(server.notifications) == 1
        method, params = server.get_notification(0)
        assert method == "$/activeContext/projectionReady"
        assert params["sessionId"] == "sess-1"
        assert params["timestamp"] == 9999.0
        assert params["projection"]["tokens"] == 500

    @pytest.mark.asyncio
    async def test_projection_ready_standard_mode(self) -> None:
        """Standard mode ignores PROJECTION_READY."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        update = SessionUpdate(
            kind=UpdateKind.PROJECTION_READY,
            session_id="sess-1",
            payload={"projection": {}},
        )

        await translator.translate(update)

        assert len(server.notifications) == 0


class TestNodeChanged:
    """Test NODE_CHANGED translation."""

    @pytest.mark.asyncio
    async def test_node_changed_lsp_aware(self) -> None:
        """LSP-aware mode sends $/activeContext/nodeChanged."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=True)

        update = SessionUpdate(
            kind=UpdateKind.NODE_CHANGED,
            session_id="sess-1",
            payload={"node_id": "view-123", "change": "state_updated"},
            timestamp=5555.5,
        )

        await translator.translate(update)

        assert len(server.notifications) == 1
        method, params = server.get_notification(0)
        assert method == "$/activeContext/nodeChanged"
        assert params["sessionId"] == "sess-1"
        assert params["nodeId"] == "view-123"
        assert params["change"] == "state_updated"
        assert params["timestamp"] == 5555.5

    @pytest.mark.asyncio
    async def test_node_changed_standard_mode(self) -> None:
        """Standard mode ignores NODE_CHANGED."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        update = SessionUpdate(
            kind=UpdateKind.NODE_CHANGED,
            session_id="sess-1",
            payload={"node_id": "view-123", "change": "state_updated"},
        )

        await translator.translate(update)

        assert len(server.notifications) == 0


class TestError:
    """Test ERROR update translation."""

    @pytest.mark.asyncio
    async def test_error_sends_window_show_message(self) -> None:
        """ERROR sends window/showMessage notification."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        update = SessionUpdate(
            kind=UpdateKind.ERROR,
            session_id="sess-1",
            payload={"error": "File not found", "details": "/path/to/missing.py"},
        )

        await translator.translate(update)

        assert len(server.notifications) == 1
        method, params = server.get_notification(0)
        assert method == "window/showMessage"
        assert params["type"] == 1  # Error = 1
        assert params["message"] == "File not found: /path/to/missing.py"

    @pytest.mark.asyncio
    async def test_error_without_details(self) -> None:
        """ERROR without details sends just the error message."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        update = SessionUpdate(
            kind=UpdateKind.ERROR,
            session_id="sess-1",
            payload={"error": "Something went wrong"},
        )

        await translator.translate(update)

        method, params = server.get_notification(0)
        assert params["message"] == "Something went wrong"


class TestIgnoredUpdates:
    """Test that non-translated update kinds are ignored."""

    @pytest.mark.asyncio
    async def test_tick_applied_ignored(self) -> None:
        """TICK_APPLIED updates are not translated."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=True)

        update = SessionUpdate(
            kind=UpdateKind.TICK_APPLIED,
            session_id="sess-1",
            payload={},
        )

        await translator.translate(update)

        assert len(server.notifications) == 0

    @pytest.mark.asyncio
    async def test_conversation_progress_ignored(self) -> None:
        """CONVERSATION_PROGRESS updates are not translated."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=True)

        update = SessionUpdate(
            kind=UpdateKind.CONVERSATION_PROGRESS,
            session_id="sess-1",
            payload={},
        )

        await translator.translate(update)

        assert len(server.notifications) == 0


class TestMultipleUpdates:
    """Test multiple updates in sequence."""

    @pytest.mark.asyncio
    async def test_full_execution_cycle(self) -> None:
        """Complete execution cycle sends proper sequence of notifications."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=False)

        # Statement executing
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTING,
                session_id="sess-1",
                payload={"statement_id": "stmt-1", "source": "x = 42"},
            )
        )

        # Statement executed
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.STATEMENT_EXECUTED,
                session_id="sess-1",
                payload={"statement_id": "stmt-1", "status": "ok"},
            )
        )

        assert len(server.notifications) == 2

        # Check begin notification
        begin_method, begin_params = server.get_notification(0)
        assert begin_method == "$/progress"
        assert begin_params["value"]["kind"] == "begin"

        # Check end notification
        end_method, end_params = server.get_notification(1)
        assert end_method == "$/progress"
        assert end_params["value"]["kind"] == "end"

    @pytest.mark.asyncio
    async def test_lsp_aware_mixed_updates(self) -> None:
        """LSP-aware mode handles mixed standard and custom notifications."""
        server = MockLSPServer()
        translator = UpdateTranslator(server, lsp_aware=True)

        # Response chunk (custom)
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.RESPONSE_CHUNK,
                session_id="sess-1",
                payload={"chunk": "Hello"},
            )
        )

        # Error (standard)
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.ERROR,
                session_id="sess-1",
                payload={"error": "Oops"},
            )
        )

        # Node changed (custom)
        await translator.translate(
            SessionUpdate(
                kind=UpdateKind.NODE_CHANGED,
                session_id="sess-1",
                payload={"node_id": "n1", "change": "updated"},
            )
        )

        assert len(server.notifications) == 3
        methods = [m for m, _ in server.notifications]
        assert methods == [
            "$/activeContext/output",
            "window/showMessage",
            "$/activeContext/nodeChanged",
        ]
