"""Tests for PluginManager integration with Timeline and DSL.

Tests cover:
1. PluginManager is created in Timeline init
2. plugin_connect DSL function works (mock transport)
3. plugin_disconnect removes node types from namespace
4. plugin_list returns connection info
5. Node type constructors are added to namespace on connect
6. Node type constructors are removed on disconnect
7. Tick calls process_pending_results
8. Cleanup disconnects all plugins
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.config.schema import PluginsConfig, PluginServerConfig
from activecontext.context.graph import ContextGraph
from activecontext.plugins.connection import PluginConnection
from activecontext.plugins.descriptor import NodePluginDescriptor, PluginSource
from activecontext.plugins.manager import PluginConnectionInfo, PluginManager
from activecontext.plugins.wire import NodeTypeSchema, PluginConnectionStatus
from activecontext.session.timeline import Timeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node_type_schema(
    node_type: str = "lint", description: str = "Lint check"
) -> NodeTypeSchema:
    """Create a simple NodeTypeSchema for tests."""
    return NodeTypeSchema(node_type=node_type, description=description)


def _make_descriptor(
    node_type: str = "lint", server_name: str = "test-server"
) -> NodePluginDescriptor:
    """Create a NodePluginDescriptor for tests."""
    return NodePluginDescriptor(
        node_type=node_type,
        source=PluginSource.REMOTE,
        schema=NodeTypeSchema(node_type=node_type),
        server_name=server_name,
    )


def _make_mock_connection(
    name: str = "test-server",
    server_name: str = "TestServer",
    server_version: str = "1.0.0",
    node_types: list[NodeTypeSchema] | None = None,
    descriptors: list[NodePluginDescriptor] | None = None,
) -> MagicMock:
    """Create a mock PluginConnection."""
    conn = MagicMock(spec=PluginConnection)
    conn.name = name
    conn.status = PluginConnectionStatus.CONNECTED
    conn.server_name = server_name
    conn.server_version = server_version

    if node_types is None:
        node_types = [_make_node_type_schema()]
    conn.node_types = node_types

    if descriptors is None:
        descriptors = [_make_descriptor(server_name=name)]
    conn.descriptors = descriptors

    conn.connect = AsyncMock()
    conn.disconnect = AsyncMock()
    conn.send_request = AsyncMock()
    return conn


def _create_timeline(tmp_path: Path, plugins_config: PluginsConfig | None = None) -> Timeline:
    """Create a Timeline with default settings for testing."""
    return Timeline(
        "test-session",
        context_graph=ContextGraph(),
        cwd=str(tmp_path),
        plugins_config=plugins_config,
    )


# ---------------------------------------------------------------------------
# 1. PluginManager is created in Timeline init
# ---------------------------------------------------------------------------


class TestPluginManagerCreation:
    """Verify PluginManager is created during Timeline initialization."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_plugin_manager_exists(self, temp_cwd: Path) -> None:
        """Timeline should have a _plugin_manager attribute after init."""
        timeline = _create_timeline(temp_cwd)
        assert hasattr(timeline, "_plugin_manager")
        assert isinstance(timeline._plugin_manager, PluginManager)

    def test_plugin_manager_has_correct_session_id(self, temp_cwd: Path) -> None:
        """PluginManager should be initialized with the session ID."""
        timeline = _create_timeline(temp_cwd)
        assert timeline._plugin_manager._session_id == "test-session"

    def test_plugin_manager_has_correct_cwd(self, temp_cwd: Path) -> None:
        """PluginManager should be initialized with the Timeline cwd."""
        timeline = _create_timeline(temp_cwd)
        assert timeline._plugin_manager._cwd == str(temp_cwd)

    def test_plugins_config_stored(self, temp_cwd: Path) -> None:
        """plugins_config parameter should be stored for later use."""
        config = PluginsConfig(servers=[])
        timeline = _create_timeline(temp_cwd, plugins_config=config)
        assert timeline._plugins_config is config


# ---------------------------------------------------------------------------
# 2. plugin_connect DSL function works (mock transport)
# ---------------------------------------------------------------------------


class TestPluginConnectDSL:
    """Test the plugin_connect DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_plugin_connect_in_namespace(self, temp_cwd: Path) -> None:
        """plugin_connect should be available in the DSL namespace."""
        timeline = _create_timeline(temp_cwd)
        try:
            assert "plugin_connect" in timeline._namespace
            assert callable(timeline._namespace["plugin_connect"])
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_plugin_connect_calls_manager(self, temp_cwd: Path) -> None:
        """plugin_connect should delegate to PluginManager.connect."""
        timeline = _create_timeline(temp_cwd)
        try:
            mock_conn = _make_mock_connection()
            with (
                patch.object(
                    timeline._plugin_manager,
                    "connect",
                    new_callable=AsyncMock,
                    return_value=mock_conn,
                ) as mock_connect,
                patch.object(
                    timeline._plugin_manager,
                    "get_connection",
                    return_value=mock_conn,
                ),
            ):
                result = await timeline._plugin_connect(
                    "test-server", command=["python", "-m", "test_plugin"]
                )
                mock_connect.assert_called_once_with(
                    "test-server", command=["python", "-m", "test_plugin"]
                )
                assert result is mock_conn
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_plugin_connect_resolves_config(self, temp_cwd: Path) -> None:
        """plugin_connect without command should resolve from plugins_config."""
        config = PluginsConfig(
            servers=[
                PluginServerConfig(
                    name="my-plugin",
                    command=["python", "-m", "my_plugin"],
                    env={"KEY": "val"},
                )
            ]
        )
        timeline = _create_timeline(temp_cwd, plugins_config=config)
        try:
            mock_conn = _make_mock_connection(name="my-plugin")
            with (
                patch.object(
                    timeline._plugin_manager,
                    "connect",
                    new_callable=AsyncMock,
                    return_value=mock_conn,
                ) as mock_connect,
                patch.object(
                    timeline._plugin_manager,
                    "get_connection",
                    return_value=mock_conn,
                ),
            ):
                await timeline._plugin_connect("my-plugin")
                mock_connect.assert_called_once_with(
                    "my-plugin",
                    command=["python", "-m", "my_plugin"],
                    env={"KEY": "val"},
                )
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 3. plugin_disconnect removes node types from namespace
# ---------------------------------------------------------------------------


class TestPluginDisconnectDSL:
    """Test the plugin_disconnect DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_plugin_disconnect_in_namespace(self, temp_cwd: Path) -> None:
        """plugin_disconnect should be available in the DSL namespace."""
        timeline = _create_timeline(temp_cwd)
        try:
            assert "plugin_disconnect" in timeline._namespace
            assert callable(timeline._namespace["plugin_disconnect"])
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_plugin_disconnect_removes_constructors(self, temp_cwd: Path) -> None:
        """plugin_disconnect should remove node type constructors from namespace."""
        timeline = _create_timeline(temp_cwd)
        try:
            # Simulate a connected server with a node type in the namespace
            mock_conn = _make_mock_connection(
                name="test-server",
                node_types=[_make_node_type_schema("lint")],
            )
            timeline._plugin_manager._connections["test-server"] = mock_conn

            # Register constructors
            timeline._register_plugin_node_types("test-server")
            assert "lint" in timeline._namespace

            # Now disconnect -- should remove 'lint' from namespace
            with patch.object(
                timeline._plugin_manager,
                "disconnect",
                new_callable=AsyncMock,
            ):
                await timeline._plugin_disconnect("test-server")
                assert "lint" not in timeline._namespace
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 4. plugin_list returns connection info
# ---------------------------------------------------------------------------


class TestPluginListDSL:
    """Test the plugin_list DSL function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_plugin_list_in_namespace(self, temp_cwd: Path) -> None:
        """plugin_list should be available in the DSL namespace."""
        timeline = _create_timeline(temp_cwd)
        try:
            assert "plugin_list" in timeline._namespace
            assert callable(timeline._namespace["plugin_list"])
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_plugin_list_delegates_to_manager(self, temp_cwd: Path) -> None:
        """plugin_list should delegate to PluginManager.list_connections."""
        timeline = _create_timeline(temp_cwd)
        try:
            expected = [
                PluginConnectionInfo(
                    name="test-server",
                    status="connected",
                    server_name="TestServer",
                    server_version="1.0.0",
                    node_types=["lint"],
                    transport="stdio",
                )
            ]
            with patch.object(
                timeline._plugin_manager,
                "list_connections",
                return_value=expected,
            ):
                result = timeline._plugin_list()
                assert result == expected
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_plugin_list_empty_initially(self, temp_cwd: Path) -> None:
        """plugin_list should return empty list when no plugins connected."""
        timeline = _create_timeline(temp_cwd)
        try:
            result = timeline._plugin_list()
            assert result == []
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 5. Node type constructors are added to namespace on connect
# ---------------------------------------------------------------------------


class TestNodeTypeRegistration:
    """Test that node type constructors are registered in the namespace."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_constructors_added_on_connect(self, temp_cwd: Path) -> None:
        """Connecting a plugin should add its node type constructors to namespace."""
        timeline = _create_timeline(temp_cwd)
        try:
            mock_conn = _make_mock_connection(
                name="test-server",
                node_types=[
                    _make_node_type_schema("lint"),
                    _make_node_type_schema("code_graph"),
                ],
            )
            timeline._plugin_manager._connections["test-server"] = mock_conn

            # Verify not in namespace yet
            assert "lint" not in timeline._namespace
            assert "code_graph" not in timeline._namespace

            # Register
            timeline._register_plugin_node_types("test-server")

            # Now should be callable
            assert "lint" in timeline._namespace
            assert "code_graph" in timeline._namespace
            assert callable(timeline._namespace["lint"])
            assert callable(timeline._namespace["code_graph"])
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_builtin_names_not_overridden(self, temp_cwd: Path) -> None:
        """Plugin node types should not override builtin DSL functions."""
        timeline = _create_timeline(temp_cwd)
        try:
            # Save original text function
            original_text = timeline._namespace["text"]

            mock_conn = _make_mock_connection(
                name="test-server",
                node_types=[_make_node_type_schema("text")],  # Conflicts with builtin
            )
            timeline._plugin_manager._connections["test-server"] = mock_conn

            timeline._register_plugin_node_types("test-server")

            # Original should be preserved
            assert timeline._namespace["text"] is original_text
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_factory_function_name(self, temp_cwd: Path) -> None:
        """Factory functions should have meaningful __name__ and __doc__."""
        timeline = _create_timeline(temp_cwd)
        try:
            factory = timeline._make_plugin_node_factory("test-server", "lint")
            assert factory.__name__ == "lint"
            assert "lint" in factory.__doc__
            assert "test-server" in factory.__doc__
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_factory_creates_node_via_manager(self, temp_cwd: Path) -> None:
        """Calling a factory function should delegate to PluginManager.create_node."""
        timeline = _create_timeline(temp_cwd)
        try:
            # Set up graph with 'context' root
            from activecontext.context.nodes import TopicNode

            root = TopicNode(title="root")
            root.node_id = "context"
            timeline._context_graph.add_node(root)

            # Mock create_node to return a mock remote node
            mock_node = MagicMock()
            mock_node.node_id = "lint_1"
            mock_node.node_type = "lint"
            mock_node.parent_ids = set()
            mock_node.children_ids = set()
            mock_node.child_order = None
            mock_node.display_sequence = None
            mock_node.mode = "idle"
            mock_node._graph = None

            with patch.object(
                timeline._plugin_manager,
                "create_node",
                new_callable=AsyncMock,
                return_value=mock_node,
            ) as mock_create:
                factory = timeline._make_plugin_node_factory("test-server", "lint")
                result = await factory("file.py", severity="error")
                mock_create.assert_called_once_with(
                    "test-server", "lint", "file.py", severity="error"
                )
                assert result is mock_node
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 6. Node type constructors are removed on disconnect
# ---------------------------------------------------------------------------


class TestNodeTypeUnregistration:
    """Test that node type constructors are removed from namespace."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_constructors_removed_on_disconnect(self, temp_cwd: Path) -> None:
        """Disconnecting a plugin should remove its node type constructors."""
        timeline = _create_timeline(temp_cwd)
        try:
            mock_conn = _make_mock_connection(
                name="test-server",
                node_types=[_make_node_type_schema("lint")],
            )
            timeline._plugin_manager._connections["test-server"] = mock_conn

            # Register then unregister
            timeline._register_plugin_node_types("test-server")
            assert "lint" in timeline._namespace

            timeline._unregister_plugin_node_types("test-server")
            assert "lint" not in timeline._namespace
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_server_is_noop(self, temp_cwd: Path) -> None:
        """Unregistering constructors for a non-connected server should be safe."""
        timeline = _create_timeline(temp_cwd)
        try:
            # Should not raise
            timeline._unregister_plugin_node_types("nonexistent")
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_register_all_restores_constructors(self, temp_cwd: Path) -> None:
        """_register_all_plugin_node_types should restore all constructors."""
        timeline = _create_timeline(temp_cwd)
        try:
            mock_conn = _make_mock_connection(
                name="srv-a",
                node_types=[_make_node_type_schema("lint_a")],
            )
            mock_conn2 = _make_mock_connection(
                name="srv-b",
                node_types=[_make_node_type_schema("lint_b")],
            )
            timeline._plugin_manager._connections["srv-a"] = mock_conn
            timeline._plugin_manager._connections["srv-b"] = mock_conn2

            timeline._register_all_plugin_node_types()

            assert "lint_a" in timeline._namespace
            assert "lint_b" in timeline._namespace
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 7. Tick calls process_pending_results
# ---------------------------------------------------------------------------


class TestTickIntegration:
    """Test that tick processing includes plugin sync."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_process_pending_plugin_results(self, temp_cwd: Path) -> None:
        """process_pending_plugin_results should delegate to manager."""
        timeline = _create_timeline(temp_cwd)
        try:
            with patch.object(
                timeline._plugin_manager,
                "process_pending_results",
                new_callable=AsyncMock,
                return_value=["node_1", "node_2"],
            ) as mock_process:
                result = await timeline.process_pending_plugin_results()
                mock_process.assert_called_once()
                assert result == ["node_1", "node_2"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_process_pending_plugin_results_empty(self, temp_cwd: Path) -> None:
        """process_pending_plugin_results returns empty when no dirty nodes."""
        timeline = _create_timeline(temp_cwd)
        try:
            result = await timeline.process_pending_plugin_results()
            assert result == []
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 8. Cleanup disconnects all plugins
# ---------------------------------------------------------------------------


class TestCleanupIntegration:
    """Test that cleanup properly disconnects all plugins."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_close_disconnects_all_plugins(self, temp_cwd: Path) -> None:
        """Timeline.close() should call plugin_manager.disconnect_all()."""
        timeline = _create_timeline(temp_cwd)
        with patch.object(
            timeline._plugin_manager,
            "disconnect_all",
            new_callable=AsyncMock,
        ) as mock_disconnect:
            await timeline.close()
            mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_via_context_manager(self, temp_cwd: Path) -> None:
        """async with Timeline should disconnect plugins on exit."""
        timeline = _create_timeline(temp_cwd)
        with patch.object(
            timeline._plugin_manager,
            "disconnect_all",
            new_callable=AsyncMock,
        ) as mock_disconnect:
            async with timeline:
                pass
            mock_disconnect.assert_called_once()


# ---------------------------------------------------------------------------
# 9. Setup plugins auto-connect
# ---------------------------------------------------------------------------


class TestSetupPlugins:
    """Test the _setup_plugins method for auto-connecting servers."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_setup_plugins_auto_connects(self, temp_cwd: Path) -> None:
        """_setup_plugins should connect servers with connect=AUTO."""
        from activecontext.config.schema import PluginConnectMode

        config = PluginsConfig(
            servers=[
                PluginServerConfig(
                    name="auto-server",
                    command=["python", "-m", "auto_plugin"],
                    connect=PluginConnectMode.AUTO,
                ),
                PluginServerConfig(
                    name="manual-server",
                    command=["python", "-m", "manual_plugin"],
                    connect=PluginConnectMode.MANUAL,
                ),
            ]
        )
        timeline = _create_timeline(temp_cwd, plugins_config=config)
        try:
            mock_conn = _make_mock_connection(name="auto-server")
            with (
                patch.object(
                    timeline._plugin_manager,
                    "connect",
                    new_callable=AsyncMock,
                    return_value=mock_conn,
                ) as mock_connect,
                patch.object(
                    timeline._plugin_manager,
                    "get_connection",
                    return_value=mock_conn,
                ),
            ):
                await timeline._setup_plugins()
                # Only auto-server should be connected
                mock_connect.assert_called_once_with(
                    name="auto-server",
                    command=["python", "-m", "auto_plugin"],
                    env=None,
                )
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_setup_plugins_skips_no_command(self, temp_cwd: Path) -> None:
        """_setup_plugins should skip servers without a command."""
        from activecontext.config.schema import PluginConnectMode

        config = PluginsConfig(
            servers=[
                PluginServerConfig(
                    name="no-cmd",
                    command=[],
                    connect=PluginConnectMode.AUTO,
                ),
            ]
        )
        timeline = _create_timeline(temp_cwd, plugins_config=config)
        try:
            with patch.object(
                timeline._plugin_manager,
                "connect",
                new_callable=AsyncMock,
            ) as mock_connect:
                await timeline._setup_plugins()
                mock_connect.assert_not_called()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_setup_plugins_handles_failure(self, temp_cwd: Path) -> None:
        """_setup_plugins should log warning and continue on connection failure."""
        from activecontext.config.schema import PluginConnectMode

        config = PluginsConfig(
            servers=[
                PluginServerConfig(
                    name="fail-server",
                    command=["python", "-m", "bad_plugin"],
                    connect=PluginConnectMode.AUTO,
                ),
            ]
        )
        timeline = _create_timeline(temp_cwd, plugins_config=config)
        try:
            with patch.object(
                timeline._plugin_manager,
                "connect",
                new_callable=AsyncMock,
                side_effect=RuntimeError("connection failed"),
            ):
                # Should not raise
                await timeline._setup_plugins()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_setup_plugins_no_config(self, temp_cwd: Path) -> None:
        """_setup_plugins with no config should be a no-op."""
        timeline = _create_timeline(temp_cwd, plugins_config=None)
        try:
            # Should not raise even with no config
            with patch(
                "activecontext.config.get_config",
                side_effect=RuntimeError("no config"),
            ):
                await timeline._setup_plugins()
        finally:
            await timeline.close()


# ---------------------------------------------------------------------------
# 10. DSL execution via execute_statement
# ---------------------------------------------------------------------------


class TestPluginDSLExecution:
    """Test that plugin DSL functions work via execute_statement."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_plugin_list_via_execute(self, temp_cwd: Path) -> None:
        """plugin_list() should be callable via execute_statement."""
        timeline = _create_timeline(temp_cwd)
        try:
            result = await timeline.execute_statement("conns = plugin_list()")
            assert result.status.value == "ok"
            ns = timeline.get_namespace()
            assert ns["conns"] == []
        finally:
            await timeline.close()
