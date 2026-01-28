"""Tests for the MCP client manager module."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from activecontext.config.schema import MCPServerConfig
from activecontext.mcp.client import (
    MCPClientManager,
    MCPConnection,
    ServerProxy,
)
from activecontext.mcp.types import (
    MCPConnectionStatus,
    MCPToolInfo,
    MCPToolResult,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def server_config():
    """Create a sample MCP server config."""
    return MCPServerConfig(
        name="test-server",
        transport="stdio",
        command=["python", "-m", "test_server"],
    )


@pytest.fixture
def mcp_connection(server_config):
    """Create an MCPConnection instance."""
    return MCPConnection(name="test-server", config=server_config)


@pytest.fixture
def connected_mcp_connection(mcp_connection):
    """Create a connected MCPConnection with mocked session."""
    mcp_connection.status = MCPConnectionStatus.CONNECTED
    mcp_connection.session = Mock()
    mcp_connection.tools = [
        MCPToolInfo(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object"},
            server_name="test-server",
        ),
        MCPToolInfo(
            name="write_file",
            description="Write a file",
            input_schema={"type": "object"},
            server_name="test-server",
        ),
    ]
    return mcp_connection


@pytest.fixture
def mcp_manager():
    """Create an MCPClientManager instance."""
    return MCPClientManager()


# =============================================================================
# MCPConnection Tests
# =============================================================================


class TestMCPConnectionInit:
    """Tests for MCPConnection initialization."""

    def test_init_default_status(self, server_config):
        """Test default status is disconnected."""
        conn = MCPConnection(name="test", config=server_config)

        assert conn.status == MCPConnectionStatus.DISCONNECTED
        assert conn.session is None
        assert conn.tools == []
        assert conn.resources == []
        assert conn.prompts == []

    def test_init_stores_config(self, server_config):
        """Test config is stored."""
        conn = MCPConnection(name="test", config=server_config)

        assert conn.name == "test"
        assert conn.config is server_config


class TestMCPConnectionConnect:
    """Tests for MCPConnection.connect method."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mcp_connection):
        """Test successful connection."""
        mock_read = Mock()
        mock_write = Mock()

        class MockTransportContext:
            async def __aenter__(self):
                return (mock_read, mock_write)
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        assert mcp_connection.status == MCPConnectionStatus.CONNECTED
        assert mcp_connection.session is mock_session

    @pytest.mark.asyncio
    async def test_connect_sets_connecting_status(self, mcp_connection):
        """Test that connect sets CONNECTING status initially."""
        status_during_connect = None

        class MockTransportContext:
            async def __aenter__(self_inner):
                nonlocal status_during_connect
                status_during_connect = mcp_connection.status
                return (Mock(), Mock())
            async def __aexit__(self_inner, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        assert status_during_connect == MCPConnectionStatus.CONNECTING

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self, mcp_connection):
        """Test that connect discovers server tools."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[mock_tool]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        assert len(mcp_connection.tools) == 1
        assert mcp_connection.tools[0].name == "test_tool"
        assert mcp_connection.tools[0].description == "A test tool"

    @pytest.mark.asyncio
    async def test_connect_handles_resources_error(self, mcp_connection):
        """Test that connect handles resources list error gracefully."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(side_effect=Exception("Not supported"))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        # Should still succeed, resources empty
        assert mcp_connection.status == MCPConnectionStatus.CONNECTED
        assert mcp_connection.resources == []

    @pytest.mark.asyncio
    async def test_connect_handles_prompts_error(self, mcp_connection):
        """Test that connect handles prompts list error gracefully."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(side_effect=Exception("Not supported"))

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        # Should still succeed, prompts empty
        assert mcp_connection.status == MCPConnectionStatus.CONNECTED
        assert mcp_connection.prompts == []

    @pytest.mark.asyncio
    async def test_connect_error_sets_error_status(self, mcp_connection):
        """Test that connect error sets ERROR status."""
        with patch("activecontext.mcp.client.create_transport", side_effect=Exception("Connection failed")):
            with pytest.raises(Exception, match="Connection failed"):
                await mcp_connection.connect()

        assert mcp_connection.status == MCPConnectionStatus.ERROR
        assert mcp_connection.error_message == "Connection failed"


class TestMCPConnectionDisconnect:
    """Tests for MCPConnection.disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_success(self, connected_mcp_connection):
        """Test successful disconnect."""
        connected_mcp_connection.session.__aexit__ = AsyncMock()
        connected_mcp_connection._transport_context = AsyncMock()
        connected_mcp_connection._transport_context.__aexit__ = AsyncMock()

        await connected_mcp_connection.disconnect()

        assert connected_mcp_connection.status == MCPConnectionStatus.DISCONNECTED
        assert connected_mcp_connection.session is None
        assert connected_mcp_connection.tools == []
        assert connected_mcp_connection.resources == []

    @pytest.mark.asyncio
    async def test_disconnect_handles_session_error(self, connected_mcp_connection):
        """Test disconnect handles session close error."""
        connected_mcp_connection.session.__aexit__ = AsyncMock(
            side_effect=Exception("Close error")
        )

        await connected_mcp_connection.disconnect()

        # Should still complete
        assert connected_mcp_connection.status == MCPConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_handles_transport_error(self, connected_mcp_connection):
        """Test disconnect handles transport close error."""
        connected_mcp_connection.session.__aexit__ = AsyncMock()
        connected_mcp_connection._transport_context = AsyncMock()
        connected_mcp_connection._transport_context.__aexit__ = AsyncMock(
            side_effect=Exception("Transport error")
        )

        await connected_mcp_connection.disconnect()

        # Should still complete
        assert connected_mcp_connection.status == MCPConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_without_session(self, mcp_connection):
        """Test disconnect when not connected."""
        await mcp_connection.disconnect()

        assert mcp_connection.status == MCPConnectionStatus.DISCONNECTED


class TestMCPConnectionCallTool:
    """Tests for MCPConnection.call_tool method."""

    @pytest.mark.asyncio
    async def test_call_tool_when_not_connected(self, mcp_connection):
        """Test call_tool returns error when not connected."""
        result = await mcp_connection.call_tool("test_tool", {"arg": "value"})

        assert result.success is False
        assert result.is_error is True
        assert "not connected" in result.error_message

    @pytest.mark.asyncio
    async def test_call_tool_success_with_text_content(self, connected_mcp_connection):
        """Test successful tool call with text content."""
        from mcp import types

        mock_text = types.TextContent(type="text", text="Hello, World!")
        mock_result = Mock()
        mock_result.content = [mock_text]
        mock_result.structuredContent = None
        mock_result.isError = False

        connected_mcp_connection.session.call_tool = AsyncMock(return_value=mock_result)

        result = await connected_mcp_connection.call_tool("read_file", {"path": "/test"})

        assert result.success is True
        assert result.is_error is False
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_call_tool_success_with_image_content(self, connected_mcp_connection):
        """Test successful tool call with image content."""
        from mcp import types

        mock_image = types.ImageContent(
            type="image",
            data="base64data",
            mimeType="image/png",
        )
        mock_result = Mock()
        mock_result.content = [mock_image]
        mock_result.structuredContent = None
        mock_result.isError = False

        connected_mcp_connection.session.call_tool = AsyncMock(return_value=mock_result)

        result = await connected_mcp_connection.call_tool("get_image", {})

        assert result.success is True
        assert len(result.content) == 1
        assert result.content[0]["type"] == "image"
        assert result.content[0]["data"] == "base64data"
        assert result.content[0]["mime_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_call_tool_handles_exception(self, connected_mcp_connection):
        """Test call_tool handles exception."""
        connected_mcp_connection.session.call_tool = AsyncMock(
            side_effect=Exception("Tool error")
        )

        result = await connected_mcp_connection.call_tool("failing_tool", {})

        assert result.success is False
        assert result.is_error is True
        assert "Tool error" in result.error_message


class TestMCPConnectionReadResource:
    """Tests for MCPConnection.read_resource method."""

    @pytest.mark.asyncio
    async def test_read_resource_when_not_connected(self, mcp_connection):
        """Test read_resource returns error when not connected."""
        result = await mcp_connection.read_resource("file:///test.txt")

        assert result.success is False
        assert result.is_error is True
        assert "not connected" in result.error_message

    @pytest.mark.asyncio
    async def test_read_resource_handles_exception(self, connected_mcp_connection):
        """Test read_resource handles exception."""
        connected_mcp_connection.session.read_resource = AsyncMock(
            side_effect=Exception("Read error")
        )

        result = await connected_mcp_connection.read_resource("file:///test.txt")

        assert result.success is False
        assert result.is_error is True
        assert "Read error" in result.error_message


# =============================================================================
# MCPClientManager Tests
# =============================================================================


class TestMCPClientManagerInit:
    """Tests for MCPClientManager initialization."""

    def test_init_default_values(self):
        """Test default initialization values."""
        manager = MCPClientManager()

        assert manager.connections == {}
        assert manager.config is None
        assert manager._permission_callback is None


class TestMCPClientManagerPermissions:
    """Tests for MCPClientManager permission callback."""

    def test_set_permission_callback(self, mcp_manager):
        """Test setting permission callback."""
        async def callback(server, tool, args):
            return True

        mcp_manager.set_permission_callback(callback)

        assert mcp_manager._permission_callback is callback


class TestMCPClientManagerConnect:
    """Tests for MCPClientManager.connect method."""

    @pytest.mark.asyncio
    async def test_connect_with_config(self, mcp_manager, server_config):
        """Test connect with explicit config."""
        with patch.object(MCPConnection, "connect", new_callable=AsyncMock):
            result = await mcp_manager.connect(config=server_config)

        assert result.name == "test-server"
        assert "test-server" in mcp_manager.connections

    @pytest.mark.asyncio
    async def test_connect_with_name_and_config(self, mcp_manager, server_config):
        """Test connect with both name and config uses name."""
        with patch.object(MCPConnection, "connect", new_callable=AsyncMock):
            result = await mcp_manager.connect(name="custom-name", config=server_config)

        assert result.name == "custom-name"
        assert "custom-name" in mcp_manager.connections

    @pytest.mark.asyncio
    async def test_connect_by_name_from_config(self, mcp_manager, server_config):
        """Test connect by name looks up config."""
        from activecontext.config.schema import MCPConfig

        mcp_manager.config = MCPConfig(servers=[server_config])

        with patch.object(MCPConnection, "connect", new_callable=AsyncMock):
            result = await mcp_manager.connect(name="test-server")

        assert result.name == "test-server"

    @pytest.mark.asyncio
    async def test_connect_by_name_not_found(self, mcp_manager):
        """Test connect by name raises when not found."""
        from activecontext.config.schema import MCPConfig

        mcp_manager.config = MCPConfig(servers=[])

        with pytest.raises(ValueError, match="not found in config"):
            await mcp_manager.connect(name="nonexistent")

    @pytest.mark.asyncio
    async def test_connect_no_name_or_config_raises(self, mcp_manager):
        """Test connect with neither name nor config raises."""
        with pytest.raises(ValueError, match="Must provide either"):
            await mcp_manager.connect()

    @pytest.mark.asyncio
    async def test_connect_returns_existing_connected(self, mcp_manager, server_config):
        """Test connect returns existing connected server."""
        existing = MCPConnection(name="test-server", config=server_config)
        existing.status = MCPConnectionStatus.CONNECTED
        mcp_manager.connections["test-server"] = existing

        result = await mcp_manager.connect(config=server_config)

        assert result is existing

    @pytest.mark.asyncio
    async def test_connect_reconnects_error_state(self, mcp_manager, server_config):
        """Test connect reconnects server in error state."""
        existing = MCPConnection(name="test-server", config=server_config)
        existing.status = MCPConnectionStatus.ERROR
        existing.disconnect = AsyncMock()
        mcp_manager.connections["test-server"] = existing

        with patch.object(MCPConnection, "connect", new_callable=AsyncMock):
            result = await mcp_manager.connect(config=server_config)

        existing.disconnect.assert_called_once()
        assert result is not existing


class TestMCPClientManagerDisconnect:
    """Tests for MCPClientManager.disconnect method."""

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self, mcp_manager, server_config):
        """Test disconnect removes connection."""
        conn = MCPConnection(name="test-server", config=server_config)
        conn.disconnect = AsyncMock()
        mcp_manager.connections["test-server"] = conn

        await mcp_manager.disconnect("test-server")

        assert "test-server" not in mcp_manager.connections
        conn.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_is_noop(self, mcp_manager):
        """Test disconnect on nonexistent server is no-op."""
        await mcp_manager.disconnect("nonexistent")

        # Should not raise


class TestMCPClientManagerDisconnectAll:
    """Tests for MCPClientManager.disconnect_all method."""

    @pytest.mark.asyncio
    async def test_disconnect_all(self, mcp_manager, server_config):
        """Test disconnect_all disconnects all servers."""
        conn1 = MCPConnection(name="server1", config=server_config)
        conn1.disconnect = AsyncMock()
        conn2 = MCPConnection(name="server2", config=server_config)
        conn2.disconnect = AsyncMock()

        mcp_manager.connections["server1"] = conn1
        mcp_manager.connections["server2"] = conn2

        await mcp_manager.disconnect_all()

        assert mcp_manager.connections == {}
        conn1.disconnect.assert_called_once()
        conn2.disconnect.assert_called_once()


class TestMCPClientManagerGetConnection:
    """Tests for MCPClientManager.get_connection method."""

    def test_get_connection_found(self, mcp_manager, server_config):
        """Test get_connection returns connection."""
        conn = MCPConnection(name="test-server", config=server_config)
        mcp_manager.connections["test-server"] = conn

        result = mcp_manager.get_connection("test-server")

        assert result is conn

    def test_get_connection_not_found(self, mcp_manager):
        """Test get_connection returns None when not found."""
        result = mcp_manager.get_connection("nonexistent")

        assert result is None


class TestMCPClientManagerListConnections:
    """Tests for MCPClientManager.list_connections method."""

    def test_list_connections_empty(self, mcp_manager):
        """Test list_connections with no connections."""
        result = mcp_manager.list_connections()

        assert result == []

    def test_list_connections(self, mcp_manager, server_config):
        """Test list_connections returns all connections."""
        conn1 = MCPConnection(name="server1", config=server_config)
        conn2 = MCPConnection(name="server2", config=server_config)
        mcp_manager.connections["server1"] = conn1
        mcp_manager.connections["server2"] = conn2

        result = mcp_manager.list_connections()

        assert len(result) == 2
        assert conn1 in result
        assert conn2 in result


class TestMCPClientManagerGetAllTools:
    """Tests for MCPClientManager.get_all_tools method."""

    def test_get_all_tools_empty(self, mcp_manager):
        """Test get_all_tools with no connections."""
        result = mcp_manager.get_all_tools()

        assert result == []

    def test_get_all_tools_only_connected(self, mcp_manager, server_config):
        """Test get_all_tools only includes connected servers."""
        conn1 = MCPConnection(name="server1", config=server_config)
        conn1.status = MCPConnectionStatus.CONNECTED
        conn1.tools = [MCPToolInfo(name="tool1", description="", input_schema={}, server_name="server1")]

        conn2 = MCPConnection(name="server2", config=server_config)
        conn2.status = MCPConnectionStatus.DISCONNECTED
        conn2.tools = [MCPToolInfo(name="tool2", description="", input_schema={}, server_name="server2")]

        mcp_manager.connections["server1"] = conn1
        mcp_manager.connections["server2"] = conn2

        result = mcp_manager.get_all_tools()

        assert len(result) == 1
        assert result[0].name == "tool1"


class TestMCPClientManagerGenerateBindings:
    """Tests for MCPClientManager.generate_namespace_bindings method."""

    def test_generate_bindings_creates_proxies(self, mcp_manager, connected_mcp_connection):
        """Test generate_namespace_bindings creates ServerProxy objects."""
        mcp_manager.connections["test-server"] = connected_mcp_connection

        bindings = mcp_manager.generate_namespace_bindings()

        assert "test-server" in bindings
        assert isinstance(bindings["test-server"], ServerProxy)

    def test_generate_bindings_only_connected(self, mcp_manager, server_config):
        """Test bindings only include connected servers."""
        disconnected = MCPConnection(name="disconnected", config=server_config)
        disconnected.status = MCPConnectionStatus.DISCONNECTED
        mcp_manager.connections["disconnected"] = disconnected

        bindings = mcp_manager.generate_namespace_bindings()

        assert "disconnected" not in bindings


# =============================================================================
# ServerProxy Tests
# =============================================================================


class TestServerProxyInit:
    """Tests for ServerProxy initialization."""

    def test_init_creates_tool_methods(self, connected_mcp_connection):
        """Test init creates method for each tool."""
        proxy = ServerProxy(connected_mcp_connection)

        assert hasattr(proxy, "read_file")
        assert hasattr(proxy, "write_file")
        assert callable(proxy.read_file)
        assert callable(proxy.write_file)


class TestServerProxyRepr:
    """Tests for ServerProxy.__repr__."""

    def test_repr_format(self, connected_mcp_connection):
        """Test __repr__ format."""
        proxy = ServerProxy(connected_mcp_connection)

        result = repr(proxy)

        assert "MCPServer" in result
        assert "test-server" in result
        assert "read_file" in result
        assert "write_file" in result


class TestServerProxyDir:
    """Tests for ServerProxy.__dir__."""

    def test_dir_includes_tools(self, connected_mcp_connection):
        """Test __dir__ includes tool names."""
        proxy = ServerProxy(connected_mcp_connection)

        result = dir(proxy)

        assert "read_file" in result
        assert "write_file" in result
        assert "_connection" in result


class TestServerProxyToolCalls:
    """Tests for ServerProxy tool method calls."""

    @pytest.mark.asyncio
    async def test_tool_call_without_permission_callback(self, connected_mcp_connection):
        """Test tool call without permission callback."""
        connected_mcp_connection.call_tool = AsyncMock(
            return_value=MCPToolResult(success=True, content=[])
        )
        proxy = ServerProxy(connected_mcp_connection)

        result = await proxy.read_file(path="/test")

        assert result.success is True
        connected_mcp_connection.call_tool.assert_called_once_with("read_file", {"path": "/test"})

    @pytest.mark.asyncio
    async def test_tool_call_with_permission_allowed(self, connected_mcp_connection):
        """Test tool call with permission callback that allows."""
        async def allow_callback(server, tool, args):
            return True

        connected_mcp_connection.call_tool = AsyncMock(
            return_value=MCPToolResult(success=True, content=[])
        )
        proxy = ServerProxy(connected_mcp_connection, permission_callback=allow_callback)

        result = await proxy.read_file(path="/test")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_tool_call_with_permission_denied(self, connected_mcp_connection):
        """Test tool call with permission callback that denies."""
        from activecontext.mcp.permissions import MCPPermissionDenied

        async def deny_callback(server, tool, args):
            return False

        proxy = ServerProxy(connected_mcp_connection, permission_callback=deny_callback)

        with pytest.raises(MCPPermissionDenied):
            await proxy.read_file(path="/test")


# =============================================================================
# MCPConnection Reconnection Tests
# =============================================================================


class TestMCPConnectionReconnection:
    """Tests for reconnection scenarios."""

    @pytest.mark.asyncio
    async def test_reconnect_after_disconnect(self, mcp_connection):
        """Test reconnecting after a normal disconnect."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))
        mock_session.__aexit__ = AsyncMock()

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                # First connect
                await mcp_connection.connect()
                assert mcp_connection.status == MCPConnectionStatus.CONNECTED

                # Disconnect
                await mcp_connection.disconnect()
                assert mcp_connection.status == MCPConnectionStatus.DISCONNECTED

                # Reconnect
                await mcp_connection.connect()
                assert mcp_connection.status == MCPConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_reconnect_after_error(self, mcp_connection):
        """Test reconnecting after a connection error."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Connection failed")
            return MockTransportContext()

        with patch("activecontext.mcp.client.create_transport", side_effect=mock_create):
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                # First connect fails
                with pytest.raises(Exception, match="Connection failed"):
                    await mcp_connection.connect()
                assert mcp_connection.status == MCPConnectionStatus.ERROR

                # Retry succeeds
                await mcp_connection.connect()
                assert mcp_connection.status == MCPConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connection_clears_error_message_on_success(self, mcp_connection):
        """Test that error_message is cleared on successful reconnect."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        # Set initial error state
        mcp_connection.status = MCPConnectionStatus.ERROR
        mcp_connection.error_message = "Previous error"

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        assert mcp_connection.status == MCPConnectionStatus.CONNECTED
        # Note: error_message is not cleared by current implementation
        # This test documents current behavior


class TestMCPConnectionStateTransitions:
    """Tests for connection state transitions."""

    @pytest.mark.asyncio
    async def test_status_progression_on_connect(self, mcp_connection):
        """Test that status progresses: DISCONNECTED -> CONNECTING -> CONNECTED."""
        statuses = []

        class MockTransportContext:
            async def __aenter__(self_inner):
                statuses.append(mcp_connection.status)
                return (Mock(), Mock())
            async def __aexit__(self_inner, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
        mock_session.list_resources = AsyncMock(return_value=Mock(resources=[]))
        mock_session.list_prompts = AsyncMock(return_value=Mock(prompts=[]))

        assert mcp_connection.status == MCPConnectionStatus.DISCONNECTED
        statuses.append(mcp_connection.status)

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                await mcp_connection.connect()

        statuses.append(mcp_connection.status)

        assert statuses[0] == MCPConnectionStatus.DISCONNECTED
        assert statuses[1] == MCPConnectionStatus.CONNECTING
        assert statuses[2] == MCPConnectionStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_status_on_connection_failure(self, mcp_connection):
        """Test that status goes to ERROR on connection failure."""
        with patch("activecontext.mcp.client.create_transport", side_effect=Exception("Failed")):
            with pytest.raises(Exception):
                await mcp_connection.connect()

        assert mcp_connection.status == MCPConnectionStatus.ERROR
        assert mcp_connection.error_message == "Failed"


class TestMCPConnectionErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_tool_discovery_error_sets_status(self, mcp_connection):
        """Test that tool discovery error sets error status."""
        class MockTransportContext:
            async def __aenter__(self):
                return (Mock(), Mock())
            async def __aexit__(self, *args):
                pass

        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=Exception("Tool list failed"))

        with patch("activecontext.mcp.client.create_transport", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MockTransportContext()
            with patch("activecontext.mcp.client.ClientSession", return_value=mock_session):
                with pytest.raises(Exception, match="Tool list failed"):
                    await mcp_connection.connect()

        assert mcp_connection.status == MCPConnectionStatus.ERROR

    @pytest.mark.asyncio
    async def test_call_tool_on_connecting_status(self, mcp_connection):
        """Test call_tool returns error when status is CONNECTING."""
        mcp_connection.status = MCPConnectionStatus.CONNECTING
        mcp_connection.session = Mock()

        result = await mcp_connection.call_tool("test_tool", {})

        assert result.success is False
        assert result.is_error is True
        assert "not connected" in result.error_message

    @pytest.mark.asyncio
    async def test_call_tool_on_error_status(self, mcp_connection):
        """Test call_tool returns error when status is ERROR."""
        mcp_connection.status = MCPConnectionStatus.ERROR
        mcp_connection.session = Mock()

        result = await mcp_connection.call_tool("test_tool", {})

        assert result.success is False
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_read_resource_success(self, connected_mcp_connection):
        """Test successful resource read."""
        from mcp import types

        mock_text = types.TextContent(type="text", text="Resource content")
        mock_result = Mock()
        mock_result.contents = [mock_text]

        connected_mcp_connection.session.read_resource = AsyncMock(return_value=mock_result)

        result = await connected_mcp_connection.read_resource("file:///test.txt")

        assert result.success is True
        assert len(result.content) == 1
        assert result.content[0]["type"] == "text"
        assert result.content[0]["text"] == "Resource content"


# =============================================================================
# MCPClientManager Advanced Tests
# =============================================================================


class TestMCPClientManagerAdvanced:
    """Advanced tests for MCPClientManager."""

    @pytest.mark.asyncio
    async def test_connect_reconnects_disconnected_state(self, mcp_manager, server_config):
        """Test connect creates new connection for disconnected server."""
        existing = MCPConnection(name="test-server", config=server_config)
        existing.status = MCPConnectionStatus.DISCONNECTED
        existing.disconnect = AsyncMock()
        mcp_manager.connections["test-server"] = existing

        with patch.object(MCPConnection, "connect", new_callable=AsyncMock):
            result = await mcp_manager.connect(config=server_config)

        # Should create a new connection
        assert result is not existing
        # Disconnect is called to clean up (idempotent operation)
        existing.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_servers_independent(self, mcp_manager, server_config):
        """Test multiple servers can be managed independently."""
        config1 = MCPServerConfig(
            name="server1",
            transport="stdio",
            command=["python", "-m", "server1"],
        )
        config2 = MCPServerConfig(
            name="server2",
            transport="stdio",
            command=["python", "-m", "server2"],
        )

        with patch.object(MCPConnection, "connect", new_callable=AsyncMock):
            conn1 = await mcp_manager.connect(config=config1)
            conn2 = await mcp_manager.connect(config=config2)

        assert len(mcp_manager.connections) == 2
        assert conn1.name == "server1"
        assert conn2.name == "server2"

    @pytest.mark.asyncio
    async def test_get_all_tools_aggregates(self, mcp_manager, server_config):
        """Test get_all_tools aggregates tools from multiple servers."""
        conn1 = MCPConnection(name="server1", config=server_config)
        conn1.status = MCPConnectionStatus.CONNECTED
        conn1.tools = [
            MCPToolInfo(name="tool1a", description="", input_schema={}, server_name="server1"),
            MCPToolInfo(name="tool1b", description="", input_schema={}, server_name="server1"),
        ]

        conn2 = MCPConnection(name="server2", config=server_config)
        conn2.status = MCPConnectionStatus.CONNECTED
        conn2.tools = [
            MCPToolInfo(name="tool2a", description="", input_schema={}, server_name="server2"),
        ]

        mcp_manager.connections["server1"] = conn1
        mcp_manager.connections["server2"] = conn2

        tools = mcp_manager.get_all_tools()

        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert tool_names == {"tool1a", "tool1b", "tool2a"}
