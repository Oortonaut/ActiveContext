"""Tests for RemoteNode -- proxy for remote CAP plugin nodes.

Tests the full RemoteNode interface: construction, cached renders,
fallback renders, method queuing, dirty flag, attribute proxy,
serialization round-trip, and async tick with mocked connections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from activecontext.context.state import Expansion
from activecontext.plugins.protocol import (
    RenderSnapshot,
    TokenEstimate,
)
from activecontext.plugins.remote_node import RemoteNode, _estimate_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(**kwargs: Any) -> RemoteNode:
    """Create a RemoteNode with sensible defaults for testing."""
    defaults: dict[str, Any] = {
        "node_id": "test1234",
        "_actual_node_type": "lint",
        "_remote_id": "remote_1",
        "_server_name": "lint-server",
    }
    defaults.update(kwargs)
    return RemoteNode(**defaults)


def _make_sync_result(
    state: dict[str, Any] | None = None,
    renders: dict[str, str] | None = None,
    tokens: dict[str, int] | None = None,
    digest: dict[str, Any] | None = None,
    notifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a mock node/sync result dict."""
    return {
        "state": state or {"errors": 3, "display_name": "Lint Results"},
        "renders": renders
        or {
            "header": "### lint.1 Lint Results {#test1234} all",
            "content": "Found 3 errors\n",
            "detail": "Error details here\n",
        },
        "tokens": tokens or {"collapsed": 10, "summary": 25, "detail": 40},
        "digest": digest
        or {
            "id": "remote_1",
            "type": "lint",
            "errors": 3,
        },
        "notifications": notifications or [],
    }


def _mock_connection(sync_result: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock PluginConnection with send_request as AsyncMock."""
    conn = MagicMock()
    conn.send_request = AsyncMock(return_value=sync_result or _make_sync_result())
    return conn


# ---------------------------------------------------------------------------
# 1. Basic construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Test RemoteNode creation and default state."""

    def test_basic_creation(self) -> None:
        """Create a RemoteNode with minimal args."""
        node = _make_node()
        assert node.node_id == "test1234"
        assert node.node_type == "lint"
        assert node._actual_node_type == "lint"
        assert node._remote_id == "remote_1"
        assert node._server_name == "lint-server"
        assert node.is_remote is True

    def test_default_remote_type(self) -> None:
        """Default _actual_node_type is 'remote'."""
        node = RemoteNode(node_id="x")
        assert node.node_type == "remote"

    def test_remote_id_defaults_to_node_id(self) -> None:
        """If _remote_id is empty, __post_init__ sets it to node_id."""
        node = RemoteNode(node_id="abc123", _actual_node_type="test")
        assert node._remote_id == "abc123"

    def test_initial_state_clean(self) -> None:
        """New node is not dirty and has no pending calls."""
        node = _make_node()
        assert node._dirty is False
        assert node._pending_calls == []
        assert node._stale is False
        assert node._connection is None

    def test_repr(self) -> None:
        """repr includes type, server, and connection status."""
        node = _make_node()
        r = repr(node)
        assert "lint" in r
        assert "lint-server" in r
        assert "disconnected" in r


# ---------------------------------------------------------------------------
# 2. Cached renders
# ---------------------------------------------------------------------------


class TestCachedRenders:
    """Test that render methods return cached values."""

    def test_render_header_from_cache(self) -> None:
        """render_header returns cached header when available."""
        node = _make_node(
            _cached_renders=RenderSnapshot(
                header="### cached header",
                content="body text\n",
                detail="detail text\n",
            )
        )
        assert node.render_header() == "### cached header"

    def test_render_content_from_cache(self) -> None:
        """render_content returns cached content."""
        node = _make_node(_cached_renders=RenderSnapshot(content="body text\n"))
        assert node.render_content() == "body text\n"

    def test_render_content_merges_content_and_detail(self) -> None:
        """render_content returns merged content + detail from cache."""
        node = _make_node(
            _cached_renders=RenderSnapshot(
                content="content\n",
                detail="detail\n",
            )
        )
        result = node.render_content()
        assert result == "content\ndetail\n"

    def test_render_content_only_content(self) -> None:
        """render_content returns content when no detail in cache."""
        node = _make_node(
            _cached_renders=RenderSnapshot(
                content="content\n",
            )
        )
        result = node.render_content()
        assert result == "content\n"


# ---------------------------------------------------------------------------
# 3. Fallback renders
# ---------------------------------------------------------------------------


class TestFallbackRenders:
    """Test render fallbacks when no cached data."""

    def test_render_header_fallback(self) -> None:
        """render_header falls back to ContextNode.render_header."""
        node = _make_node()
        header = node.render_header()
        # The fallback uses the uniform header format; verify it's non-empty
        assert header
        # It should contain the node_id reference
        assert node.node_id in header

    def test_render_content_empty_fallback(self) -> None:
        """render_content returns empty string when no cache."""
        node = _make_node()
        assert node.render_content() == ""

    def test_render_content_empty_no_detail(self) -> None:
        """render_content returns empty string when no cache at all."""
        node = _make_node()
        assert node.render_content() == ""

    def test_render_digest_fallback(self) -> None:
        """render_digest falls back to type + (remote)."""
        node = _make_node()
        name = node.render_digest()
        assert "lint" in name
        assert "remote" in name

    def test_render_digest_from_title(self) -> None:
        """render_digest uses title if set."""
        node = _make_node(title="My Lint Node")
        assert node.render_digest() == "My Lint Node"

    def test_render_digest_from_cached_digest(self) -> None:
        """render_digest reads summary from cached digest."""
        node = _make_node(_cached_digest={"summary": "Lint Results"})
        assert node.render_digest() == "Lint Results"


# ---------------------------------------------------------------------------
# 4. Queue method call
# ---------------------------------------------------------------------------


class TestQueueMethodCall:
    """Test method call queuing."""

    def test_queue_single_call(self) -> None:
        """Queue one method call."""
        node = _make_node()
        node.queue_method_call("set_content", args=["hello"])
        assert len(node._pending_calls) == 1
        call = node._pending_calls[0]
        assert call.method == "set_content"
        assert call.args == ("hello",)
        assert call.kwargs == {}

    def test_queue_with_kwargs(self) -> None:
        """Queue a call with keyword arguments."""
        node = _make_node()
        node.queue_method_call("configure", kwargs={"verbose": True})
        call = node._pending_calls[0]
        assert call.kwargs == {"verbose": True}

    def test_queue_marks_dirty(self) -> None:
        """Queuing a call sets the dirty flag."""
        node = _make_node()
        assert node._dirty is False
        node.queue_method_call("cancel")
        assert node._dirty is True

    def test_queue_multiple_calls(self) -> None:
        """Multiple calls accumulate in order."""
        node = _make_node()
        node.queue_method_call("first")
        node.queue_method_call("second")
        node.queue_method_call("third")
        assert len(node._pending_calls) == 3
        assert [c.method for c in node._pending_calls] == [
            "first",
            "second",
            "third",
        ]


# ---------------------------------------------------------------------------
# 5. Mark dirty
# ---------------------------------------------------------------------------


class TestMarkDirty:
    """Test dirty flag management."""

    def test_mark_dirty_sets_flag(self) -> None:
        """mark_dirty() sets _dirty to True."""
        node = _make_node()
        assert node._dirty is False
        node.mark_dirty()
        assert node._dirty is True

    def test_mark_dirty_idempotent(self) -> None:
        """Calling mark_dirty() twice is safe."""
        node = _make_node()
        node.mark_dirty()
        node.mark_dirty()
        assert node._dirty is True


# ---------------------------------------------------------------------------
# 6. __getattr__ from cache
# ---------------------------------------------------------------------------


class TestGetattr:
    """Test attribute proxy to cached state."""

    def test_read_cached_property(self) -> None:
        """Access cached state as attribute."""
        node = _make_node(_cached_state={"errors": 5, "warnings": 2})
        assert node.errors == 5
        assert node.warnings == 2

    def test_missing_attribute_raises(self) -> None:
        """Accessing non-existent attribute raises AttributeError."""
        node = _make_node()
        with pytest.raises(AttributeError, match="no_such_attr"):
            _ = node.no_such_attr

    def test_private_attrs_not_proxied(self) -> None:
        """Private attributes (starting with _) are not looked up in cache."""
        node = _make_node(_cached_state={"_secret": 42})
        with pytest.raises(AttributeError):
            _ = node._nonexistent

    def test_real_attrs_take_precedence(self) -> None:
        """Actual dataclass fields are not overridden by cached state."""
        node = _make_node(_cached_state={"node_id": "fake"})
        # node_id is a real field and should NOT be proxied
        assert node.node_id == "test1234"


# ---------------------------------------------------------------------------
# 7. to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    """Test serialization round-trip."""

    def test_to_dict_includes_remote_fields(self) -> None:
        """to_dict includes all remote-specific fields."""
        node = _make_node(
            _cached_state={"errors": 3},
            _cached_renders=RenderSnapshot(header="h", content="c", detail="d"),
            _cached_tokens=TokenEstimate(collapsed=10, summary=20, detail=30),
            _cached_digest={"id": "test"},
        )
        d = node.to_dict()
        assert d["node_type"] == "remote"
        assert d["_actual_node_type"] == "lint"
        assert d["_remote_id"] == "remote_1"
        assert d["_server_name"] == "lint-server"
        assert d["_cached_state"] == {"errors": 3}
        assert d["_cached_renders"]["header"] == "h"
        assert d["_cached_tokens"]["collapsed"] == 10

    def test_to_dict_excludes_connection(self) -> None:
        """to_dict does not include _connection."""
        node = _make_node()
        node._connection = MagicMock()
        d = node.to_dict()
        assert "_connection" not in d

    def test_round_trip(self) -> None:
        """from_dict(to_dict()) produces equivalent node."""
        original = _make_node(
            _cached_state={"errors": 3},
            _cached_renders=RenderSnapshot(header="h", content="c", detail="d"),
            _cached_tokens=TokenEstimate(collapsed=10, summary=20, detail=30),
            _cached_digest={"id": "test"},
        )
        original.default_expansion = Expansion.CONTENT
        original.mode = "running"
        original.version = 5
        original.title = "Lint Check"

        data = original.to_dict()
        restored = RemoteNode.from_dict(data)

        assert restored.node_id == original.node_id
        assert restored._actual_node_type == "lint"
        assert restored._remote_id == "remote_1"
        assert restored._server_name == "lint-server"
        assert restored._cached_state == {"errors": 3}
        assert restored._cached_renders.header == "h"
        assert restored._cached_renders.content == "c"
        assert restored._cached_tokens.collapsed == 10
        assert restored.default_expansion == Expansion.CONTENT
        assert restored.mode == "running"
        assert restored.version == 5
        assert restored.title == "Lint Check"

    def test_from_dict_no_connection(self) -> None:
        """from_dict produces a node with no connection (transient)."""
        data = _make_node().to_dict()
        restored = RemoteNode.from_dict(data)
        assert restored._connection is None
        assert restored.has_connection is False


# ---------------------------------------------------------------------------
# 8. Tick with no changes
# ---------------------------------------------------------------------------


class TestTickNoChanges:
    """Test tick when nothing needs syncing."""

    @pytest.mark.asyncio
    async def test_tick_clean_node_no_rpc(self) -> None:
        """Tick on a clean node sends no RPC."""
        conn = _mock_connection()
        node = _make_node(_connection=conn)
        # Not dirty, no pending calls
        await node.tick_async()
        conn.send_request.assert_not_called()

    @pytest.mark.asyncio
    async def test_recompute_is_noop(self) -> None:
        """Recompute() does nothing for RemoteNode."""
        node = _make_node()
        # Should not raise
        node.Recompute()


# ---------------------------------------------------------------------------
# 9. Tick with pending calls
# ---------------------------------------------------------------------------


class TestTickWithPendingCalls:
    """Test tick sends sync RPC with queued calls."""

    @pytest.mark.asyncio
    async def test_tick_sends_sync_with_calls(self) -> None:
        """Tick sends node/sync with pending calls."""
        conn = _mock_connection()
        node = _make_node(_connection=conn)
        node.queue_method_call("set_content", args=["hello"])
        node.queue_method_call("validate")

        await node.tick_async(cwd="/project")

        conn.send_request.assert_called_once()
        call_args = conn.send_request.call_args
        method = call_args[0][0]
        params = call_args[0][1]

        assert method == "node/sync"
        assert params.node_id == "remote_1"
        assert params.cwd == "/project"
        assert len(params.calls) == 2
        assert params.calls[0].method == "set_content"
        assert params.calls[0].args == ["hello"]
        assert params.calls[1].method == "validate"

    @pytest.mark.asyncio
    async def test_tick_clears_pending_on_success(self) -> None:
        """Successful tick clears pending calls and dirty flag."""
        conn = _mock_connection()
        node = _make_node(_connection=conn)
        node.queue_method_call("do_work")

        await node.tick_async()

        assert node._pending_calls == []
        assert node._dirty is False
        assert node._stale is False


# ---------------------------------------------------------------------------
# 10. Tick with dirty flag
# ---------------------------------------------------------------------------


class TestTickWithDirtyFlag:
    """Test tick sends sync when dirty (no pending calls)."""

    @pytest.mark.asyncio
    async def test_dirty_flag_triggers_sync(self) -> None:
        """A dirty node syncs even without pending calls."""
        conn = _mock_connection()
        node = _make_node(_connection=conn)
        node.mark_dirty()

        await node.tick_async()

        conn.send_request.assert_called_once()
        params = conn.send_request.call_args[0][1]
        assert params.calls == []  # No pending calls, just a sync

    @pytest.mark.asyncio
    async def test_dirty_flag_cleared_after_sync(self) -> None:
        """Dirty flag is cleared after successful sync."""
        conn = _mock_connection()
        node = _make_node(_connection=conn)
        node.mark_dirty()

        await node.tick_async()

        assert node._dirty is False


# ---------------------------------------------------------------------------
# 11. Tick updates caches
# ---------------------------------------------------------------------------


class TestTickUpdatesCaches:
    """Test that sync response updates all cached data."""

    @pytest.mark.asyncio
    async def test_caches_updated_from_sync(self) -> None:
        """All caches updated from sync result."""
        result = _make_sync_result(
            state={"errors": 7, "display_name": "Updated Lint"},
            renders={
                "header": "### new header",
                "content": "new content\n",
                "detail": "new detail\n",
            },
            tokens={"collapsed": 15, "summary": 30, "detail": 50},
            digest={"id": "remote_1", "errors": 7},
        )
        conn = _mock_connection(sync_result=result)
        node = _make_node(_connection=conn)
        node.mark_dirty()

        await node.tick_async()

        assert node._cached_state == {
            "errors": 7,
            "display_name": "Updated Lint",
        }
        assert node._cached_renders.header == "### new header"
        assert node._cached_renders.content == "new content\n"
        assert node._cached_renders.detail == "new detail\n"
        assert node._cached_tokens.collapsed == 15
        assert node._cached_tokens.summary == 30
        assert node._cached_tokens.detail == 50
        assert node._cached_digest == {"id": "remote_1", "errors": 7}

    @pytest.mark.asyncio
    async def test_version_incremented_after_sync(self) -> None:
        """Version is bumped after successful sync."""
        conn = _mock_connection()
        node = _make_node(_connection=conn)
        initial_version = node.version
        node.mark_dirty()

        await node.tick_async()

        assert node.version > initial_version

    @pytest.mark.asyncio
    async def test_sync_with_notifications(self) -> None:
        """Notifications in sync result are processed."""
        result = _make_sync_result(
            notifications=[
                {"description": "Lint completed", "level": "hold"},
            ]
        )
        conn = _mock_connection(sync_result=result)
        node = _make_node(_connection=conn)
        node.mark_dirty()

        # Should not raise
        await node.tick_async()

        # Version should be incremented (notifications call _mark_changed)
        assert node.version > 0


# ---------------------------------------------------------------------------
# 12. Tick error handling
# ---------------------------------------------------------------------------


class TestTickErrorHandling:
    """Test graceful degradation on sync failure."""

    @pytest.mark.asyncio
    async def test_rpc_failure_marks_stale(self) -> None:
        """Failed sync marks node as stale."""
        conn = MagicMock()
        conn.send_request = AsyncMock(side_effect=Exception("Connection lost"))
        node = _make_node(_connection=conn)
        node.mark_dirty()

        await node.tick_async()

        assert node._stale is True
        assert node._last_sync_error is not None
        assert "Connection lost" in node._last_sync_error

    @pytest.mark.asyncio
    async def test_rpc_failure_preserves_pending_calls(self) -> None:
        """Failed sync keeps pending calls for retry."""
        conn = MagicMock()
        conn.send_request = AsyncMock(side_effect=Exception("timeout"))
        node = _make_node(_connection=conn)
        node.queue_method_call("important_method")

        await node.tick_async()

        # Pending calls should NOT be cleared on failure
        assert len(node._pending_calls) == 1
        assert node._pending_calls[0].method == "important_method"
        assert node._dirty is True  # Still dirty

    @pytest.mark.asyncio
    async def test_no_connection_marks_stale(self) -> None:
        """Tick without connection marks node as stale."""
        node = _make_node()  # No connection set
        node.mark_dirty()

        await node.tick_async()

        assert node._stale is True

    @pytest.mark.asyncio
    async def test_stale_cleared_on_success(self) -> None:
        """Successful sync after failure clears stale flag."""
        conn = _mock_connection()
        node = _make_node(_connection=conn, _stale=True)
        node.mark_dirty()

        await node.tick_async()

        assert node._stale is False
        assert node._last_sync_error is None


# ---------------------------------------------------------------------------
# Additional: Token estimation and connection management
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    """Test get_token_breakdown with and without cached tokens."""

    def test_cached_tokens_returned(self) -> None:
        """get_token_breakdown returns cached values."""
        node = _make_node(_cached_tokens=TokenEstimate(collapsed=10, summary=20, detail=30))
        info = node.get_token_breakdown()
        assert info.collapsed == 10
        assert info.summary == 20
        assert info.detail == 30

    def test_fallback_estimation_from_renders(self) -> None:
        """get_token_breakdown estimates from renders when no cache."""
        node = _make_node(
            _cached_renders=RenderSnapshot(
                header="A" * 40,  # ~10 tokens
                content="B" * 80,  # ~20 tokens
                detail="C" * 120,  # ~30 tokens
            )
        )
        info = node.get_token_breakdown()
        assert info.collapsed == _estimate_tokens("A" * 40)
        assert info.summary == _estimate_tokens("B" * 80)
        assert info.detail == _estimate_tokens("C" * 120)

    def test_empty_renders_zero_tokens(self) -> None:
        """Empty renders produce zero token estimates."""
        node = _make_node()
        info = node.get_token_breakdown()
        assert info.collapsed == 0
        assert info.summary == 0
        assert info.detail == 0


class TestConnectionManagement:
    """Test connection set/clear helpers."""

    def test_set_connection(self) -> None:
        """set_connection stores connection and clears stale."""
        node = _make_node(_stale=True)
        conn = MagicMock()
        node.set_connection(conn)
        assert node._connection is conn
        assert node._stale is False
        assert node.has_connection is True

    def test_clear_connection(self) -> None:
        """clear_connection removes connection and marks stale."""
        conn = MagicMock()
        node = _make_node(_connection=conn)
        node.clear_connection()
        assert node._connection is None
        assert node._stale is True
        assert node.has_connection is False

    def test_is_stale_property(self) -> None:
        """is_stale reflects _stale flag."""
        node = _make_node()
        assert node.is_stale is False
        node._stale = True
        assert node.is_stale is True


class TestDigest:
    """Test GetDigest / get_digest."""

    def test_cached_digest(self) -> None:
        """GetDigest returns cached digest when available."""
        node = _make_node(_cached_digest={"id": "r1", "type": "lint", "errors": 3})
        d = node.GetDigest()
        assert d == {"id": "r1", "type": "lint", "errors": 3}

    def test_fallback_digest(self) -> None:
        """GetDigest builds fallback when no cache."""
        node = _make_node()
        d = node.GetDigest()
        assert d["id"] == "test1234"
        assert d["type"] == "lint"
        assert d["remote"] is True
        assert d["server"] == "lint-server"

    def test_get_digest_adapter(self) -> None:
        """get_digest() delegates to GetDigest()."""
        node = _make_node(_cached_digest={"x": 1})
        assert node.get_digest() == {"x": 1}
