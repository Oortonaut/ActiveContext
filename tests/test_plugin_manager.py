"""Tests for the CAP Plugin Manager.

Tests cover:
1. Construction and initial state
2. Connection lifecycle (connect, disconnect, disconnect_all)
3. Error handling (duplicate connect, missing command/transport, unknown server)
4. Node creation and destruction
5. Tick synchronization (process_pending_results)
6. Query methods (list_connections, get_connection, get_remote_node)
7. Notification routing (node/dirty marks correct node)
8. Host API handlers (query_roots, resolve_root)
9. Roots management (set_roots)
10. Event firing on connect/disconnect
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.plugins.connection import PluginConnection
from activecontext.plugins.descriptor import NodePluginDescriptor, PluginSource
from activecontext.plugins.manager import (
    PluginConnectionInfo,
    PluginManager,
    PluginServerConfig,
)
from activecontext.plugins.remote_node import RemoteNode
from activecontext.plugins.transport import PluginTransportError
from activecontext.plugins.wire import (
    Methods,
    NodeTypeSchema,
    PluginConnectionStatus,
    RootInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_registry() -> MagicMock:
    """Create a mock NodeTypeRegistry."""
    registry = MagicMock()
    registry.register_plugin = MagicMock()
    registry.unregister_plugin = MagicMock(return_value=True)
    return registry


def _make_mock_connection(
    name: str = "test-server",
    server_name: str = "TestServer",
    server_version: str = "1.0.0",
    node_types: list[NodeTypeSchema] | None = None,
    descriptors: list[NodePluginDescriptor] | None = None,
) -> MagicMock:
    """Create a mock PluginConnection with sensible defaults."""
    conn = MagicMock(spec=PluginConnection)
    conn.name = name
    conn.status = PluginConnectionStatus.CONNECTED
    conn.server_name = server_name
    conn.server_version = server_version

    if node_types is None:
        node_types = [NodeTypeSchema(node_type="lint", description="Lint check")]
    conn.node_types = node_types

    if descriptors is None:
        descriptors = [
            NodePluginDescriptor(
                node_type="lint",
                source=PluginSource.REMOTE,
                schema=NodeTypeSchema(node_type="lint"),
                server_name=name,
            )
        ]
    conn.descriptors = descriptors

    conn.connect = AsyncMock(
        return_value={
            "server_name": server_name,
            "server_version": server_version,
            "node_types": [],
        }
    )
    conn.disconnect = AsyncMock()
    conn.send_request = AsyncMock()
    conn.send_notification = AsyncMock()
    return conn


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestPluginManagerConstruction:
    """Verify initial state after creation."""

    def test_initial_state(self) -> None:
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry, session_id="sess_1", cwd="/project")

        assert mgr._registry is registry
        assert mgr._session_id == "sess_1"
        assert mgr._cwd == "/project"
        assert mgr._connections == {}
        assert mgr._remote_nodes == {}
        assert mgr._node_to_server == {}
        assert mgr._roots == []
        assert mgr._fire_event is None

    def test_initial_state_with_fire_event(self) -> None:
        registry = _make_mock_registry()
        events: list[tuple[str, dict[str, Any]]] = []

        def fire(event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

        mgr = PluginManager(registry=registry, fire_event=fire)
        assert mgr._fire_event is fire

    def test_default_parameters(self) -> None:
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)
        assert mgr._session_id == ""
        assert mgr._cwd == ""


# ---------------------------------------------------------------------------
# Connection lifecycle tests
# ---------------------------------------------------------------------------


class TestPluginManagerConnect:
    """Connection lifecycle tests."""

    @pytest.mark.asyncio
    async def test_connect_with_command(self) -> None:
        """Connect with command spawns connection and registers node types."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry, session_id="sess_1", cwd="/project")

        mock_conn = _make_mock_connection()

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            conn = await mgr.connect("test-server", command=["python", "-m", "plugin"])

        assert conn is mock_conn
        assert "test-server" in mgr._connections
        mock_conn.connect.assert_awaited_once_with(
            session_id="sess_1",
            cwd="/project",
            roots=[],
        )
        registry.register_plugin.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_transport(self) -> None:
        """Connect with pre-configured transport."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_transport = MagicMock()
        mock_conn = _make_mock_connection()

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            conn = await mgr.connect("test-server", transport=mock_transport)

        assert conn is mock_conn
        assert "test-server" in mgr._connections

    @pytest.mark.asyncio
    async def test_connect_already_connected_raises(self) -> None:
        """Connecting to an already-connected server raises ValueError."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        with pytest.raises(ValueError, match="Already connected to 'test-server'"):
            await mgr.connect("test-server", command=["echo"])

    @pytest.mark.asyncio
    async def test_connect_without_command_or_transport_raises(self) -> None:
        """Must provide either command or transport."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        with pytest.raises(ValueError, match="Must provide either command .* or transport"):
            await mgr.connect("test-server")

    @pytest.mark.asyncio
    async def test_connect_registers_multiple_node_types(self) -> None:
        """Multiple node types from server are all registered."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        desc1 = NodePluginDescriptor(
            node_type="lint",
            source=PluginSource.REMOTE,
            schema=NodeTypeSchema(node_type="lint"),
            server_name="test-server",
        )
        desc2 = NodePluginDescriptor(
            node_type="format",
            source=PluginSource.REMOTE,
            schema=NodeTypeSchema(node_type="format"),
            server_name="test-server",
        )

        mock_conn = _make_mock_connection(descriptors=[desc1, desc2])

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        assert registry.register_plugin.call_count == 2

    @pytest.mark.asyncio
    async def test_connect_handles_registration_failure(self) -> None:
        """If a node type fails to register, others still succeed."""
        registry = _make_mock_registry()
        # First call raises, second succeeds
        registry.register_plugin.side_effect = [
            ValueError("Cannot override builtin"),
            None,
        ]
        mgr = PluginManager(registry=registry)

        desc1 = NodePluginDescriptor(
            node_type="shell",
            source=PluginSource.REMOTE,
            schema=NodeTypeSchema(node_type="shell"),
            server_name="test-server",
        )
        desc2 = NodePluginDescriptor(
            node_type="custom_lint",
            source=PluginSource.REMOTE,
            schema=NodeTypeSchema(node_type="custom_lint"),
            server_name="test-server",
        )

        mock_conn = _make_mock_connection(descriptors=[desc1, desc2])

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            # Should not raise despite registration failure
            conn = await mgr.connect("test-server", command=["echo"])

        assert conn is mock_conn
        assert registry.register_plugin.call_count == 2

    @pytest.mark.asyncio
    async def test_connect_fires_event(self) -> None:
        """Fire event callback is called on successful connect."""
        registry = _make_mock_registry()
        events: list[tuple[str, dict[str, Any]]] = []

        def fire(event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

        mgr = PluginManager(registry=registry, fire_event=fire)

        mock_conn = _make_mock_connection()
        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        assert len(events) == 1
        assert events[0] == ("plugin_connected", {"name": "test-server"})

    @pytest.mark.asyncio
    async def test_connect_with_roots(self) -> None:
        """Roots are passed to the connection on connect."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry, session_id="s1", cwd="/proj")
        mgr.set_roots(
            [
                RootInfo(uri="file:///proj", name="cwd"),
                RootInfo(uri="file:///home", name="home"),
            ]
        )

        mock_conn = _make_mock_connection()
        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        mock_conn.connect.assert_awaited_once_with(
            session_id="s1",
            cwd="/proj",
            roots=[
                RootInfo(uri="file:///proj", name="cwd"),
                RootInfo(uri="file:///home", name="home"),
            ],
        )


# ---------------------------------------------------------------------------
# Disconnect tests
# ---------------------------------------------------------------------------


class TestPluginManagerDisconnect:
    """Disconnect lifecycle tests."""

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Disconnect unregisters types and marks nodes stale."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        # Add a remote node tracked by this server
        node = RemoteNode(
            _actual_node_type="lint",
            _connection=mock_conn,
            _remote_id="node_1",
            _server_name="test-server",
        )
        mgr._remote_nodes["node_1"] = node
        mgr._node_to_server["node_1"] = "test-server"

        await mgr.disconnect("test-server")

        assert "test-server" not in mgr._connections
        registry.unregister_plugin.assert_called_once_with("lint")
        mock_conn.disconnect.assert_awaited_once()
        # Node should be stale
        assert node._stale is True
        assert node._connection is None

    @pytest.mark.asyncio
    async def test_disconnect_unknown_noop(self) -> None:
        """Disconnecting an unknown server is a no-op."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        # Should not raise
        await mgr.disconnect("nonexistent")

    @pytest.mark.asyncio
    async def test_disconnect_fires_event(self) -> None:
        """Fire event callback is called on disconnect."""
        registry = _make_mock_registry()
        events: list[tuple[str, dict[str, Any]]] = []

        def fire(event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

        mgr = PluginManager(registry=registry, fire_event=fire)

        mock_conn = _make_mock_connection()
        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        events.clear()  # Clear the connect event
        await mgr.disconnect("test-server")

        assert len(events) == 1
        assert events[0] == ("plugin_disconnected", {"name": "test-server"})

    @pytest.mark.asyncio
    async def test_disconnect_all(self) -> None:
        """Disconnect all servers."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn1 = _make_mock_connection(name="server-1")
        mock_conn2 = _make_mock_connection(
            name="server-2",
            descriptors=[
                NodePluginDescriptor(
                    node_type="format",
                    source=PluginSource.REMOTE,
                    schema=NodeTypeSchema(node_type="format"),
                    server_name="server-2",
                )
            ],
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            side_effect=[mock_conn1, mock_conn2],
        ):
            await mgr.connect("server-1", command=["echo"])
            await mgr.connect("server-2", command=["echo"])

        assert len(mgr._connections) == 2
        await mgr.disconnect_all()
        assert len(mgr._connections) == 0

        mock_conn1.disconnect.assert_awaited_once()
        mock_conn2.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_only_marks_own_nodes_stale(self) -> None:
        """Only nodes belonging to the disconnected server are marked stale."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn1 = _make_mock_connection(name="server-1")
        mock_conn2 = _make_mock_connection(
            name="server-2",
            descriptors=[
                NodePluginDescriptor(
                    node_type="format",
                    source=PluginSource.REMOTE,
                    schema=NodeTypeSchema(node_type="format"),
                    server_name="server-2",
                )
            ],
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            side_effect=[mock_conn1, mock_conn2],
        ):
            await mgr.connect("server-1", command=["echo"])
            await mgr.connect("server-2", command=["echo"])

        # Node from server-1
        node1 = RemoteNode(
            _actual_node_type="lint",
            _connection=mock_conn1,
            _remote_id="n1",
            _server_name="server-1",
        )
        mgr._remote_nodes["n1"] = node1
        mgr._node_to_server["n1"] = "server-1"

        # Node from server-2
        node2 = RemoteNode(
            _actual_node_type="format",
            _connection=mock_conn2,
            _remote_id="n2",
            _server_name="server-2",
        )
        mgr._remote_nodes["n2"] = node2
        mgr._node_to_server["n2"] = "server-2"

        # Disconnect only server-1
        await mgr.disconnect("server-1")

        assert node1._stale is True
        assert node1._connection is None
        assert node2._stale is False
        assert node2._connection is mock_conn2


# ---------------------------------------------------------------------------
# Node creation tests
# ---------------------------------------------------------------------------


class TestPluginManagerCreateNode:
    """Node creation and destruction tests."""

    @pytest.mark.asyncio
    async def test_create_node(self) -> None:
        """Create a remote node via the server."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        mock_conn.send_request = AsyncMock(
            return_value={
                "node_id": "lint_abc",
                "initial_state": {"errors": 0, "path": "main.py"},
            }
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        node = await mgr.create_node("test-server", "lint", "main.py", fix=True)

        assert isinstance(node, RemoteNode)
        assert node._actual_node_type == "lint"
        assert node._remote_id == "lint_abc"
        assert node._server_name == "test-server"
        assert node._cached_state == {"errors": 0, "path": "main.py"}
        assert node._connection is mock_conn

        # Verify tracked
        assert mgr._remote_nodes["lint_abc"] is node
        assert mgr._node_to_server["lint_abc"] == "test-server"

        # Verify RPC was sent correctly
        mock_conn.send_request.assert_awaited_with(
            Methods.NODE_CREATE,
            {
                "node_type": "lint",
                "args": ["main.py"],
                "kwargs": {"fix": True},
                "node_id": "",
            },
        )

    @pytest.mark.asyncio
    async def test_create_node_with_explicit_id(self) -> None:
        """Create node with an explicit node_id."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        mock_conn.send_request = AsyncMock(
            return_value={
                "node_id": "my_node",
                "initial_state": {},
            }
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        node = await mgr.create_node("test-server", "lint", node_id="my_node")

        assert node._remote_id == "my_node"
        assert "my_node" in mgr._remote_nodes

    @pytest.mark.asyncio
    async def test_create_node_on_unknown_server_raises(self) -> None:
        """Creating a node on an unconnected server raises ValueError."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        with pytest.raises(ValueError, match="Not connected to 'nonexistent'"):
            await mgr.create_node("nonexistent", "lint")

    @pytest.mark.asyncio
    async def test_destroy_node(self) -> None:
        """Destroy a remote node."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        mock_conn.send_request = AsyncMock(
            return_value={
                "node_id": "lint_abc",
                "initial_state": {},
            }
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        await mgr.create_node("test-server", "lint")

        # Reset mock to verify destroy call
        mock_conn.send_request.reset_mock()
        mock_conn.send_request = AsyncMock(return_value=None)

        await mgr.destroy_node("lint_abc")

        mock_conn.send_request.assert_awaited_once_with(
            Methods.NODE_DESTROY,
            {"node_id": "lint_abc"},
        )
        assert "lint_abc" not in mgr._remote_nodes
        assert "lint_abc" not in mgr._node_to_server

    @pytest.mark.asyncio
    async def test_destroy_unknown_node_noop(self) -> None:
        """Destroying an unknown node is a no-op."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        # Should not raise
        await mgr.destroy_node("nonexistent")

    @pytest.mark.asyncio
    async def test_destroy_node_rpc_failure_still_cleans_up(self) -> None:
        """If destroy RPC fails, node is still removed from tracking."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        mock_conn.send_request = AsyncMock(
            return_value={
                "node_id": "lint_abc",
                "initial_state": {},
            }
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        await mgr.create_node("test-server", "lint")

        # Make destroy fail
        mock_conn.send_request = AsyncMock(side_effect=PluginTransportError("Connection lost"))

        await mgr.destroy_node("lint_abc")

        # Should still be cleaned up
        assert "lint_abc" not in mgr._remote_nodes
        assert "lint_abc" not in mgr._node_to_server


# ---------------------------------------------------------------------------
# Tick synchronization tests
# ---------------------------------------------------------------------------


class TestPluginManagerSync:
    """process_pending_results (tick sync) tests."""

    @pytest.mark.asyncio
    async def test_process_pending_results(self) -> None:
        """Dirty nodes get synced on tick."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry, cwd="/project")

        node = RemoteNode(
            _actual_node_type="lint",
            _remote_id="n1",
            _server_name="test-server",
        )
        node._dirty = True
        node.tick_async = AsyncMock()  # type: ignore[method-assign]

        mgr._remote_nodes["n1"] = node

        synced = await mgr.process_pending_results()

        assert synced == ["n1"]
        node.tick_async.assert_awaited_once_with("/project")

    @pytest.mark.asyncio
    async def test_process_pending_results_with_pending_calls(self) -> None:
        """Nodes with pending calls also get synced."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        from activecontext.plugins.protocol import MethodCall

        node = RemoteNode(
            _actual_node_type="lint",
            _remote_id="n1",
            _server_name="test-server",
        )
        node._pending_calls = [MethodCall(method="recheck")]
        node.tick_async = AsyncMock()  # type: ignore[method-assign]

        mgr._remote_nodes["n1"] = node

        synced = await mgr.process_pending_results()

        assert synced == ["n1"]

    @pytest.mark.asyncio
    async def test_process_pending_results_skips_clean_nodes(self) -> None:
        """Clean nodes are not synced."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        node = RemoteNode(
            _actual_node_type="lint",
            _remote_id="n1",
            _server_name="test-server",
        )
        node._dirty = False
        node._pending_calls = []
        node.tick_async = AsyncMock()  # type: ignore[method-assign]

        mgr._remote_nodes["n1"] = node

        synced = await mgr.process_pending_results()

        assert synced == []
        node.tick_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_pending_results_with_error(self) -> None:
        """One node fails sync, others still synced."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        node_ok = RemoteNode(
            _actual_node_type="lint",
            _remote_id="n1",
            _server_name="test-server",
        )
        node_ok._dirty = True
        node_ok.tick_async = AsyncMock()  # type: ignore[method-assign]

        node_fail = RemoteNode(
            _actual_node_type="format",
            _remote_id="n2",
            _server_name="test-server",
        )
        node_fail._dirty = True
        node_fail.tick_async = AsyncMock(  # type: ignore[method-assign]
            side_effect=PluginTransportError("Connection lost")
        )

        mgr._remote_nodes["n1"] = node_ok
        mgr._remote_nodes["n2"] = node_fail

        synced = await mgr.process_pending_results()

        # n1 should be synced, n2 should not (error)
        assert "n1" in synced
        assert "n2" not in synced

    @pytest.mark.asyncio
    async def test_process_pending_results_empty(self) -> None:
        """No remote nodes returns empty list."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        synced = await mgr.process_pending_results()
        assert synced == []


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


class TestPluginManagerQuery:
    """Query method tests."""

    @pytest.mark.asyncio
    async def test_list_connections(self) -> None:
        """List connections returns info for all connected servers."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection(
            name="test-server",
            server_name="TestServer",
            server_version="1.0.0",
        )

        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        conns = mgr.list_connections()
        assert len(conns) == 1
        info = conns[0]
        assert isinstance(info, PluginConnectionInfo)
        assert info.name == "test-server"
        assert info.status == "connected"
        assert info.server_name == "TestServer"
        assert info.server_version == "1.0.0"
        assert info.node_types == ["lint"]
        assert info.transport == "stdio"

    def test_list_connections_empty(self) -> None:
        """No connections returns empty list."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)
        assert mgr.list_connections() == []

    @pytest.mark.asyncio
    async def test_get_connection(self) -> None:
        """Get connection by name."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mock_conn = _make_mock_connection()
        with patch(
            "activecontext.plugins.manager.PluginConnection",
            return_value=mock_conn,
        ):
            await mgr.connect("test-server", command=["echo"])

        assert mgr.get_connection("test-server") is mock_conn
        assert mgr.get_connection("nonexistent") is None

    def test_get_remote_node(self) -> None:
        """Get remote node by ID."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        node = RemoteNode(
            _actual_node_type="lint",
            _remote_id="n1",
            _server_name="test-server",
        )
        mgr._remote_nodes["n1"] = node

        assert mgr.get_remote_node("n1") is node
        assert mgr.get_remote_node("nonexistent") is None


# ---------------------------------------------------------------------------
# Notification routing tests
# ---------------------------------------------------------------------------


class TestPluginManagerNotifications:
    """Notification routing tests."""

    def test_node_dirty_notification(self) -> None:
        """node/dirty notification marks correct node dirty."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        node = RemoteNode(
            _actual_node_type="lint",
            _remote_id="n1",
            _server_name="test-server",
        )
        node._dirty = False
        mgr._remote_nodes["n1"] = node

        mgr._on_notification(Methods.NODE_DIRTY, {"node_id": "n1"})

        assert node._dirty is True

    def test_node_dirty_notification_unknown_node(self) -> None:
        """node/dirty for unknown node does not raise."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        # Should not raise
        mgr._on_notification(Methods.NODE_DIRTY, {"node_id": "nonexistent"})

    def test_notification_fires_event(self) -> None:
        """Notifications trigger fire_event callback."""
        registry = _make_mock_registry()
        events: list[tuple[str, dict[str, Any]]] = []

        def fire(event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

        mgr = PluginManager(registry=registry, fire_event=fire)

        mgr._on_notification(Methods.NODE_DIRTY, {"node_id": "n1"})

        assert len(events) == 1
        assert events[0][0] == "plugin_notification"
        assert events[0][1]["method"] == Methods.NODE_DIRTY
        assert events[0][1]["params"]["node_id"] == "n1"

    def test_non_dirty_notification_still_fires_event(self) -> None:
        """Other notification types still fire events."""
        registry = _make_mock_registry()
        events: list[tuple[str, dict[str, Any]]] = []

        def fire(event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

        mgr = PluginManager(registry=registry, fire_event=fire)

        mgr._on_notification(
            Methods.NODE_NOTIFICATION,
            {"node_id": "n1", "description": "Lint complete"},
        )

        assert len(events) == 1
        assert events[0][0] == "plugin_notification"


# ---------------------------------------------------------------------------
# Host API tests
# ---------------------------------------------------------------------------


class TestPluginManagerHostAPI:
    """Host API handler tests."""

    def test_query_roots(self) -> None:
        """host/query_roots returns configured roots."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)
        mgr.set_roots(
            [
                RootInfo(uri="file:///project", name="cwd"),
                RootInfo(uri="file:///home/user", name="home"),
            ]
        )

        result = mgr._on_host_api(Methods.HOST_QUERY_ROOTS, {}, 1)

        assert result is not None
        assert len(result["roots"]) == 2
        assert result["roots"][0]["uri"] == "file:///project"
        assert result["roots"][0]["name"] == "cwd"
        assert result["roots"][1]["uri"] == "file:///home/user"
        assert result["roots"][1]["name"] == "home"

    def test_query_roots_empty(self) -> None:
        """host/query_roots with no roots returns empty list."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        result = mgr._on_host_api(Methods.HOST_QUERY_ROOTS, {}, 1)

        assert result == {"roots": []}

    def test_resolve_root_found(self) -> None:
        """host/resolve_root resolves a known root URI."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)
        mgr.set_roots(
            [
                RootInfo(uri="file:///project", name="cwd"),
            ]
        )

        result = mgr._on_host_api(
            Methods.HOST_RESOLVE_ROOT,
            {"uri": "file:///project"},
            2,
        )

        assert result is not None
        assert result["path"] == "/project"

    def test_resolve_root_not_found(self) -> None:
        """host/resolve_root for unknown URI returns error."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        result = mgr._on_host_api(
            Methods.HOST_RESOLVE_ROOT,
            {"uri": "file:///unknown"},
            3,
        )

        assert result is not None
        assert result["path"] == ""
        assert result["error"] == "Root not found"

    def test_unhandled_method(self) -> None:
        """Unknown host API method returns None."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        result = mgr._on_host_api("unknown/method", {}, 4)
        assert result is None


# ---------------------------------------------------------------------------
# Roots management tests
# ---------------------------------------------------------------------------


class TestPluginManagerRoots:
    """Roots management tests."""

    def test_set_roots(self) -> None:
        """set_roots updates internal roots."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        roots = [
            RootInfo(uri="file:///a", name="a"),
            RootInfo(uri="file:///b", name="b"),
        ]
        mgr.set_roots(roots)

        assert mgr._roots == roots

    def test_set_roots_replaces_previous(self) -> None:
        """set_roots replaces, not appends."""
        registry = _make_mock_registry()
        mgr = PluginManager(registry=registry)

        mgr.set_roots([RootInfo(uri="file:///a", name="a")])
        mgr.set_roots([RootInfo(uri="file:///b", name="b")])

        assert len(mgr._roots) == 1
        assert mgr._roots[0].uri == "file:///b"


# ---------------------------------------------------------------------------
# PluginServerConfig tests
# ---------------------------------------------------------------------------


class TestPluginServerConfig:
    """PluginServerConfig dataclass tests."""

    def test_defaults(self) -> None:
        cfg = PluginServerConfig(name="test")
        assert cfg.name == "test"
        assert cfg.command is None
        assert cfg.address is None
        assert cfg.transport == "stdio"
        assert cfg.env is None
        assert cfg.cwd is None
        assert cfg.auto_connect is False

    def test_full_config(self) -> None:
        cfg = PluginServerConfig(
            name="lint",
            command=["python", "-m", "lint_server"],
            address="localhost:50051",
            transport="grpc",
            env={"DEBUG": "1"},
            cwd="/project",
            auto_connect=True,
        )
        assert cfg.name == "lint"
        assert cfg.command == ["python", "-m", "lint_server"]
        assert cfg.address == "localhost:50051"
        assert cfg.transport == "grpc"
        assert cfg.auto_connect is True
