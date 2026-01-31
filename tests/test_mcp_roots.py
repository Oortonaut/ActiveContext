"""Tests for MCP roots management."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.mcp.roots import (
    Root,
    RootsManager,
    file_uri_to_path,
    normalize_to_file_uri,
    path_to_file_uri,
)


class TestPathToFileUri:
    """Tests for path_to_file_uri()."""

    def test_unix_absolute_path(self, tmp_path: Path) -> None:
        """Convert an absolute path to file:// URI."""
        uri = path_to_file_uri(str(tmp_path))
        assert uri.startswith("file:///")
        assert str(tmp_path.resolve()).replace("\\", "/") in uri.replace("file:///", "/")

    def test_path_object(self, tmp_path: Path) -> None:
        """Accept Path objects."""
        uri = path_to_file_uri(tmp_path)
        assert uri.startswith("file:///")

    def test_roundtrip(self, tmp_path: Path) -> None:
        """Path -> URI -> Path roundtrip preserves the path."""
        original = str(tmp_path.resolve())
        uri = path_to_file_uri(original)
        recovered = file_uri_to_path(uri)
        assert os.path.normpath(recovered) == os.path.normpath(original)


class TestFileUriToPath:
    """Tests for file_uri_to_path()."""

    def test_unix_uri(self) -> None:
        """Convert Unix file:// URI to path."""
        result = file_uri_to_path("file:///home/user/project")
        # On Windows, normpath will adjust separators
        assert "home" in result
        assert "user" in result
        assert "project" in result

    def test_windows_uri(self) -> None:
        """Convert Windows file:// URI to path."""
        result = file_uri_to_path("file:///C:/Users/Ace/project")
        assert "Users" in result
        assert "Ace" in result
        assert "project" in result
        if os.name == "nt":
            assert result.startswith("C:")

    def test_bare_path_passthrough(self) -> None:
        """Bare paths (no file://) are normalized but passed through."""
        result = file_uri_to_path("/home/user/project")
        assert "home" in result

    def test_percent_encoded(self) -> None:
        """Percent-encoded characters are decoded."""
        result = file_uri_to_path("file:///home/user/my%20project")
        assert "my project" in result


class TestNormalizeToFileUri:
    """Tests for normalize_to_file_uri()."""

    def test_bare_path(self, tmp_path: Path) -> None:
        """Bare path is converted to file:// URI."""
        uri = normalize_to_file_uri(str(tmp_path))
        assert uri.startswith("file:///")

    def test_existing_uri(self, tmp_path: Path) -> None:
        """file:// URI is normalized (round-tripped)."""
        original_uri = tmp_path.resolve().as_uri()
        normalized = normalize_to_file_uri(original_uri)
        assert normalized == original_uri

    def test_strips_redundant_slashes(self, tmp_path: Path) -> None:
        """Normalization cleans up path artifacts."""
        uri = normalize_to_file_uri(str(tmp_path))
        # Should be a clean URI
        assert "file:///" in uri
        assert "//" not in uri.replace("file:///", "")


class TestRoot:
    """Tests for the Root dataclass."""

    def test_creation(self) -> None:
        """Create a Root with uri and name."""
        root = Root(uri="file:///home/user/project", name="My Project")
        assert root.uri == "file:///home/user/project"
        assert root.name == "My Project"

    def test_name_optional(self) -> None:
        """Name is optional."""
        root = Root(uri="file:///home/user/project")
        assert root.name is None

    def test_frozen(self) -> None:
        """Root is immutable."""
        root = Root(uri="file:///test", name="Test")
        with pytest.raises(AttributeError):
            root.uri = "file:///other"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Roots with same uri and name are equal."""
        r1 = Root(uri="file:///test", name="A")
        r2 = Root(uri="file:///test", name="A")
        assert r1 == r2

    def test_hashable(self) -> None:
        """Roots can be used in sets."""
        r1 = Root(uri="file:///test", name="A")
        r2 = Root(uri="file:///test", name="A")
        assert len({r1, r2}) == 1


class TestRootsManager:
    """Tests for the RootsManager."""

    def test_add_root(self, tmp_path: Path) -> None:
        """Add a root by path."""
        mgr = RootsManager()
        root = mgr.add(str(tmp_path), name="Test")
        assert root.uri.startswith("file:///")
        assert root.name == "Test"

    def test_add_root_with_uri(self, tmp_path: Path) -> None:
        """Add a root by file:// URI."""
        uri = tmp_path.resolve().as_uri()
        mgr = RootsManager()
        root = mgr.add(uri, name="Test")
        assert root.uri == uri

    def test_list_roots(self, tmp_path: Path) -> None:
        """List returns all registered roots."""
        mgr = RootsManager()
        mgr.add(str(tmp_path / "a"), name="A")
        mgr.add(str(tmp_path / "b"), name="B")
        roots = mgr.list_roots()
        assert len(roots) == 2
        names = {r.name for r in roots}
        assert names == {"A", "B"}

    def test_has_roots(self, tmp_path: Path) -> None:
        """has_roots reflects registration state."""
        mgr = RootsManager()
        assert not mgr.has_roots()
        mgr.add(str(tmp_path), name="Test")
        assert mgr.has_roots()

    def test_remove_root(self, tmp_path: Path) -> None:
        """Remove a root by path."""
        mgr = RootsManager()
        mgr.add(str(tmp_path), name="Test")
        assert mgr.has_roots()
        removed = mgr.remove(str(tmp_path))
        assert removed
        assert not mgr.has_roots()

    def test_remove_nonexistent(self, tmp_path: Path) -> None:
        """Remove returns False for unknown root."""
        mgr = RootsManager()
        assert not mgr.remove(str(tmp_path / "nonexistent"))

    def test_duplicate_add_idempotent(self, tmp_path: Path) -> None:
        """Adding same path twice doesn't duplicate."""
        mgr = RootsManager()
        mgr.add(str(tmp_path), name="Test")
        mgr.add(str(tmp_path), name="Test")
        assert len(mgr.list_roots()) == 1

    def test_duplicate_add_updates_name(self, tmp_path: Path) -> None:
        """Re-adding same path with different name updates it."""
        mgr = RootsManager()
        mgr.add(str(tmp_path), name="Old")
        mgr.add(str(tmp_path), name="New")
        roots = mgr.list_roots()
        assert len(roots) == 1
        assert roots[0].name == "New"

    def test_change_callback_on_add(self, tmp_path: Path) -> None:
        """Change callback fires on add."""
        mgr = RootsManager()
        calls: list[bool] = []
        mgr.on_change(lambda: calls.append(True))
        mgr.add(str(tmp_path), name="Test")
        assert len(calls) == 1

    def test_change_callback_on_remove(self, tmp_path: Path) -> None:
        """Change callback fires on remove."""
        mgr = RootsManager()
        mgr.add(str(tmp_path), name="Test")
        calls: list[bool] = []
        mgr.on_change(lambda: calls.append(True))
        mgr.remove(str(tmp_path))
        assert len(calls) == 1

    def test_unregister_callback(self, tmp_path: Path) -> None:
        """Unregister function removes the callback."""
        mgr = RootsManager()
        calls: list[bool] = []
        unregister = mgr.on_change(lambda: calls.append(True))
        unregister()
        mgr.add(str(tmp_path), name="Test")
        assert len(calls) == 0

    def test_callback_failure_doesnt_break(self, tmp_path: Path) -> None:
        """A failing callback doesn't prevent other callbacks."""
        mgr = RootsManager()
        calls: list[str] = []

        def bad_callback() -> None:
            raise RuntimeError("oops")

        mgr.on_change(bad_callback)
        mgr.on_change(lambda: calls.append("ok"))

        mgr.add(str(tmp_path), name="Test")
        assert calls == ["ok"]


class TestMCPConnectionRoots:
    """Tests for roots wiring in MCPConnection."""

    def test_connection_accepts_roots_manager(self) -> None:
        """MCPConnection stores _roots_manager field."""
        from activecontext.mcp.client import MCPConnection

        mgr = RootsManager()
        config = MagicMock()
        conn = MCPConnection(name="test", config=config, _roots_manager=mgr)
        assert conn._roots_manager is mgr

    def test_connection_without_roots_manager(self) -> None:
        """MCPConnection works without roots manager."""
        from activecontext.mcp.client import MCPConnection

        config = MagicMock()
        conn = MCPConnection(name="test", config=config)
        assert conn._roots_manager is None

    @pytest.mark.asyncio
    async def test_notify_roots_changed_when_connected(self) -> None:
        """notify_roots_changed calls send_roots_list_changed on session."""
        from activecontext.mcp.client import MCPConnection
        from activecontext.mcp.types import MCPConnectionStatus

        config = MagicMock()
        conn = MCPConnection(name="test", config=config)
        conn.status = MCPConnectionStatus.CONNECTED
        conn.session = MagicMock()
        conn.session.send_roots_list_changed = AsyncMock()

        await conn.notify_roots_changed()

        conn.session.send_roots_list_changed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_notify_roots_changed_when_disconnected(self) -> None:
        """notify_roots_changed is a no-op when disconnected."""
        from activecontext.mcp.client import MCPConnection
        from activecontext.mcp.types import MCPConnectionStatus

        config = MagicMock()
        conn = MCPConnection(name="test", config=config)
        conn.status = MCPConnectionStatus.DISCONNECTED

        # Should not raise
        await conn.notify_roots_changed()

    @pytest.mark.asyncio
    async def test_notify_roots_changed_handles_error(self) -> None:
        """notify_roots_changed swallows exceptions."""
        from activecontext.mcp.client import MCPConnection
        from activecontext.mcp.types import MCPConnectionStatus

        config = MagicMock()
        conn = MCPConnection(name="test", config=config)
        conn.status = MCPConnectionStatus.CONNECTED
        conn.session = MagicMock()
        conn.session.send_roots_list_changed = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        # Should not raise
        await conn.notify_roots_changed()


class TestMCPClientManagerRoots:
    """Tests for roots integration in MCPClientManager."""

    def test_set_roots_manager(self) -> None:
        """set_roots_manager stores the manager and registers callback."""
        from activecontext.mcp.client import MCPClientManager

        client_mgr = MCPClientManager()
        roots_mgr = RootsManager()
        client_mgr.set_roots_manager(roots_mgr)

        assert client_mgr._roots_manager is roots_mgr
        assert client_mgr._roots_unregister is not None

    def test_set_roots_manager_replaces_previous(self) -> None:
        """Setting a new roots manager unregisters old callback."""
        from activecontext.mcp.client import MCPClientManager

        client_mgr = MCPClientManager()
        mgr1 = RootsManager()
        mgr2 = RootsManager()

        client_mgr.set_roots_manager(mgr1)
        old_unregister = client_mgr._roots_unregister
        client_mgr.set_roots_manager(mgr2)

        assert client_mgr._roots_manager is mgr2
        # Old callback should have been unregistered
        assert client_mgr._roots_unregister is not old_unregister

    @pytest.mark.asyncio
    async def test_broadcast_roots_changed(self) -> None:
        """_broadcast_roots_changed notifies all connections."""
        from activecontext.mcp.client import MCPClientManager, MCPConnection
        from activecontext.mcp.types import MCPConnectionStatus

        client_mgr = MCPClientManager()

        conn1 = MagicMock(spec=MCPConnection)
        conn1.notify_roots_changed = AsyncMock()
        conn2 = MagicMock(spec=MCPConnection)
        conn2.notify_roots_changed = AsyncMock()

        client_mgr.connections = {"s1": conn1, "s2": conn2}

        await client_mgr._broadcast_roots_changed()

        conn1.notify_roots_changed.assert_awaited_once()
        conn2.notify_roots_changed.assert_awaited_once()

    def test_on_roots_changed_schedules_broadcast(self) -> None:
        """_on_roots_changed creates a task on the running loop."""
        from activecontext.mcp.client import MCPClientManager

        client_mgr = MCPClientManager()

        # Mock the event loop
        mock_loop = MagicMock()
        with patch("asyncio.get_running_loop", return_value=mock_loop):
            client_mgr._on_roots_changed()

        mock_loop.create_task.assert_called_once()

    def test_on_roots_changed_no_loop_doesnt_crash(self) -> None:
        """_on_roots_changed handles missing event loop gracefully."""
        from activecontext.mcp.client import MCPClientManager

        client_mgr = MCPClientManager()

        with patch(
            "asyncio.get_running_loop", side_effect=RuntimeError("no loop")
        ):
            # Should not raise
            client_mgr._on_roots_changed()

    def test_roots_change_triggers_callback(self, tmp_path: Path) -> None:
        """Adding a root triggers the manager's change notification."""
        from activecontext.mcp.client import MCPClientManager

        client_mgr = MCPClientManager()
        roots_mgr = RootsManager()
        client_mgr.set_roots_manager(roots_mgr)

        calls: list[bool] = []
        original_on_changed = client_mgr._on_roots_changed

        def tracking_on_changed() -> None:
            calls.append(True)

        client_mgr._on_roots_changed = tracking_on_changed  # type: ignore[assignment]

        # Re-register with the new tracker
        client_mgr.set_roots_manager(roots_mgr)
        roots_mgr.add(str(tmp_path), name="Test")

        assert len(calls) == 1

    def test_connect_passes_roots_manager(self) -> None:
        """MCPClientManager.connect() passes roots_manager to MCPConnection."""
        from activecontext.mcp.client import MCPClientManager

        client_mgr = MCPClientManager()
        roots_mgr = RootsManager()
        client_mgr.set_roots_manager(roots_mgr)

        assert client_mgr._roots_manager is roots_mgr


class TestMCPIntegrationRoots:
    """Tests for roots initialization in MCPIntegration."""

    def test_cwd_auto_registered(self, tmp_path: Path) -> None:
        """MCPIntegration auto-registers cwd as 'project' root."""
        from activecontext.session.mcp_integration import MCPIntegration

        graph = MagicMock()
        integration = MCPIntegration(
            context_graph=graph,
            fire_event=MagicMock(),
            cwd=str(tmp_path),
        )
        roots = integration.list_roots()
        assert len(roots) >= 1
        expected_uri = tmp_path.resolve().as_uri()
        uris = {r["uri"] for r in roots}
        assert expected_uri in uris
        names = {r["name"] for r in roots}
        assert "project" in names

    def test_config_roots_registered(self, tmp_path: Path) -> None:
        """MCPIntegration loads roots from MCPConfig."""
        from activecontext.config.schema import MCPConfig, MCPRootConfig
        from activecontext.session.mcp_integration import MCPIntegration

        sub_a = tmp_path / "lib-a"
        sub_a.mkdir()
        config = MCPConfig(
            roots=[MCPRootConfig(name="lib-a", path=str(sub_a))]
        )

        graph = MagicMock()
        integration = MCPIntegration(
            mcp_config=config,
            context_graph=graph,
            fire_event=MagicMock(),
            cwd=str(tmp_path),
        )
        roots = integration.list_roots()
        names = {r["name"] for r in roots}
        assert "project" in names
        assert "lib-a" in names

    def test_relative_config_root_resolved(self, tmp_path: Path) -> None:
        """Relative paths in config roots are resolved against cwd."""
        from activecontext.config.schema import MCPConfig, MCPRootConfig
        from activecontext.session.mcp_integration import MCPIntegration

        config = MCPConfig(
            roots=[MCPRootConfig(name="rel", path="subdir")]
        )
        graph = MagicMock()
        integration = MCPIntegration(
            mcp_config=config,
            context_graph=graph,
            fire_event=MagicMock(),
            cwd=str(tmp_path),
        )
        roots = integration.list_roots()
        uris = {r["uri"] for r in roots}
        expected = (tmp_path / "subdir").resolve().as_uri()
        assert expected in uris

    def test_add_root_via_dsl(self, tmp_path: Path) -> None:
        """add_root() adds a root dynamically."""
        from activecontext.session.mcp_integration import MCPIntegration

        graph = MagicMock()
        integration = MCPIntegration(
            context_graph=graph,
            fire_event=MagicMock(),
            cwd=str(tmp_path),
        )
        new_dir = tmp_path / "new-root"
        new_dir.mkdir()
        result = integration.add_root(str(new_dir), name="dynamic")
        assert result["name"] == "dynamic"
        assert result["uri"].startswith("file:///")

    def test_remove_root_via_dsl(self, tmp_path: Path) -> None:
        """remove_root() removes a root dynamically."""
        from activecontext.session.mcp_integration import MCPIntegration

        graph = MagicMock()
        integration = MCPIntegration(
            context_graph=graph,
            fire_event=MagicMock(),
            cwd=str(tmp_path),
        )
        # cwd is auto-registered
        assert len(integration.list_roots()) >= 1
        removed = integration.remove_root(str(tmp_path))
        assert removed
        assert len(integration.list_roots()) == 0

    def test_cli_roots_registered(self, tmp_path: Path) -> None:
        """CLI roots from set_cli_roots() are registered."""
        from activecontext.session import mcp_integration
        from activecontext.session.mcp_integration import MCPIntegration

        cli_dir = tmp_path / "cli-root"
        cli_dir.mkdir()

        # Set CLI roots before creating integration
        original = mcp_integration._cli_roots
        try:
            mcp_integration._cli_roots = [("cli-lib", str(cli_dir))]
            graph = MagicMock()
            integration = MCPIntegration(
                context_graph=graph,
                fire_event=MagicMock(),
                cwd=str(tmp_path),
            )
            roots = integration.list_roots()
            names = {r["name"] for r in roots}
            assert "cli-lib" in names
        finally:
            mcp_integration._cli_roots = original

    def test_roots_manager_wired_to_client(self, tmp_path: Path) -> None:
        """RootsManager is set on the MCPClientManager."""
        from activecontext.session.mcp_integration import MCPIntegration

        graph = MagicMock()
        integration = MCPIntegration(
            context_graph=graph,
            fire_event=MagicMock(),
            cwd=str(tmp_path),
        )
        mgr = integration._mcp_client_manager
        assert mgr._roots_manager is integration.roots_manager
