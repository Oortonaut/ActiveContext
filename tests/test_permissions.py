"""Tests for the file permission system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from activecontext.config.schema import (
    FilePermissionConfig,
    ImportConfig,
    SandboxConfig,
)
from activecontext.session.permissions import (
    ImportDenied,
    ImportGuard,
    PermissionDenied,
    PermissionManager,
    PermissionRule,
    make_safe_import,
    make_safe_open,
    write_permission_to_config,
)


class TestPermissionManager:
    """Test the PermissionManager class."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        # Create some test files and directories
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "input.txt").write_text("test input")
        (tmp_path / "output").mkdir()
        (tmp_path / "secrets").mkdir()
        (tmp_path / "secrets" / "key.txt").write_text("secret key")
        (tmp_path / "main.py").write_text("# main script")
        (tmp_path / "README.md").write_text("# README")
        return tmp_path

    def test_default_config_read_only_cwd(self, temp_cwd: Path) -> None:
        """Test default config grants read-only access to cwd."""
        manager = PermissionManager.from_config(str(temp_cwd), None)

        # Should have one auto rule for cwd
        assert len(manager.rules) == 1
        assert manager.rules[0].mode == "read"
        assert manager.rules[0].source == "auto"

        # Read access should be granted
        assert manager.check_access(str(temp_cwd / "main.py"), "read")
        assert manager.check_access(str(temp_cwd / "data" / "input.txt"), "read")

        # Write access should be denied
        assert not manager.check_access(str(temp_cwd / "main.py"), "write")

    def test_allow_cwd_write(self, temp_cwd: Path) -> None:
        """Test allow_cwd_write grants write access."""
        config = SandboxConfig(
            allow_cwd=True,
            allow_cwd_write=True,
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Both read and write should be granted
        assert manager.check_access(str(temp_cwd / "main.py"), "read")
        assert manager.check_access(str(temp_cwd / "main.py"), "write")

    def test_explicit_read_permission(self, temp_cwd: Path) -> None:
        """Test explicit read permission via config."""
        config = SandboxConfig(
            allow_cwd=False,
            file_permissions=[
                FilePermissionConfig(pattern="./data/**", mode="read"),
            ],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # data/ should be readable
        assert manager.check_access(str(temp_cwd / "data" / "input.txt"), "read")

        # Other paths should be denied
        assert not manager.check_access(str(temp_cwd / "main.py"), "read")
        assert not manager.check_access(str(temp_cwd / "secrets" / "key.txt"), "read")

    def test_explicit_write_permission(self, temp_cwd: Path) -> None:
        """Test explicit write permission via config."""
        config = SandboxConfig(
            allow_cwd=False,
            file_permissions=[
                FilePermissionConfig(pattern="./output/**", mode="write"),
            ],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # output/ should be writable
        assert manager.check_access(str(temp_cwd / "output" / "result.txt"), "write")

        # But not readable (write doesn't grant read)
        assert not manager.check_access(str(temp_cwd / "output" / "result.txt"), "read")

    def test_all_mode_permission(self, temp_cwd: Path) -> None:
        """Test 'all' mode grants both read and write."""
        config = SandboxConfig(
            allow_cwd=False,
            file_permissions=[
                FilePermissionConfig(pattern="./data/**", mode="all"),
            ],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # data/ should be both readable and writable
        assert manager.check_access(str(temp_cwd / "data" / "input.txt"), "read")
        assert manager.check_access(str(temp_cwd / "data" / "input.txt"), "write")

    def test_glob_pattern_matching(self, temp_cwd: Path) -> None:
        """Test glob pattern matching for file extensions."""
        config = SandboxConfig(
            allow_cwd=False,
            file_permissions=[
                FilePermissionConfig(pattern="*.py", mode="read"),
            ],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # .py files should be readable
        assert manager.check_access(str(temp_cwd / "main.py"), "read")

        # .md files should not be readable
        assert not manager.check_access(str(temp_cwd / "README.md"), "read")

    def test_deny_by_default(self, temp_cwd: Path) -> None:
        """Test deny_by_default behavior."""
        config = SandboxConfig(
            allow_cwd=False,
            deny_by_default=True,
            file_permissions=[],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # All access should be denied
        assert not manager.check_access(str(temp_cwd / "main.py"), "read")
        assert not manager.check_access(str(temp_cwd / "main.py"), "write")

    def test_allow_by_default(self, temp_cwd: Path) -> None:
        """Test deny_by_default=False allows unlisted paths."""
        config = SandboxConfig(
            allow_cwd=False,
            deny_by_default=False,
            file_permissions=[],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # All access should be allowed
        assert manager.check_access(str(temp_cwd / "main.py"), "read")
        assert manager.check_access(str(temp_cwd / "main.py"), "write")

    def test_path_outside_cwd_denied(self, temp_cwd: Path) -> None:
        """Test paths outside cwd are denied when allow_absolute=False."""
        config = SandboxConfig(
            allow_cwd=True,
            allow_absolute=False,
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Path outside cwd should be denied
        assert not manager.check_access("/etc/passwd", "read")
        assert not manager.check_access("/tmp/test.txt", "read")

    def test_allow_absolute_paths(self, temp_cwd: Path) -> None:
        """Test allow_absolute=True permits paths outside cwd."""
        config = SandboxConfig(
            allow_cwd=True,
            allow_absolute=True,
            deny_by_default=False,
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Path outside cwd should be allowed (if deny_by_default is False)
        assert manager.check_access("/tmp/test.txt", "read")

    def test_relative_path_resolution(self, temp_cwd: Path) -> None:
        """Test relative paths are resolved against cwd."""
        config = SandboxConfig(allow_cwd=True)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Relative path should work
        assert manager.check_access("main.py", "read")
        assert manager.check_access("./data/input.txt", "read")

    def test_path_traversal_blocked(self, temp_cwd: Path) -> None:
        """Test ../ traversal is properly resolved."""
        config = SandboxConfig(
            allow_cwd=True,
            allow_absolute=False,
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Traversal that stays in cwd should work
        assert manager.check_access(str(temp_cwd / "data" / ".." / "main.py"), "read")

        # Traversal outside cwd should be blocked
        assert not manager.check_access(str(temp_cwd / ".." / "other"), "read")

    def test_reload_updates_rules(self, temp_cwd: Path) -> None:
        """Test reload() updates permission rules."""
        config1 = SandboxConfig(allow_cwd=True, allow_cwd_write=False)
        manager = PermissionManager.from_config(str(temp_cwd), config1)

        # Initially read-only
        assert manager.check_access(str(temp_cwd / "main.py"), "read")
        assert not manager.check_access(str(temp_cwd / "main.py"), "write")

        # Reload with write access
        config2 = SandboxConfig(allow_cwd=True, allow_cwd_write=True)
        manager.reload(config2)

        # Now should have write access
        assert manager.check_access(str(temp_cwd / "main.py"), "write")

    def test_list_permissions(self, temp_cwd: Path) -> None:
        """Test list_permissions returns rule details."""
        config = SandboxConfig(
            allow_cwd=True,
            file_permissions=[
                FilePermissionConfig(pattern="./data/**", mode="read"),
            ],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)

        perms = manager.list_permissions()
        assert len(perms) == 2  # auto cwd + explicit data rule
        assert perms[0]["source"] == "auto"
        assert perms[1]["source"] == "config"
        assert "data" in perms[1]["pattern"]


class TestSafeOpen:
    """Test the safe_open wrapper."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "allowed.txt").write_text("allowed content")
        (tmp_path / "forbidden").mkdir()
        (tmp_path / "forbidden" / "secret.txt").write_text("secret content")
        return tmp_path

    def test_safe_open_read_allowed(self, temp_cwd: Path) -> None:
        """Test safe_open allows reading permitted files."""
        config = SandboxConfig(allow_cwd=True)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        safe_open = make_safe_open(manager)

        # Should be able to read allowed file
        with safe_open(str(temp_cwd / "allowed.txt"), "r") as f:
            content = f.read()
        assert content == "allowed content"

    def test_safe_open_read_denied(self, temp_cwd: Path) -> None:
        """Test safe_open raises PermissionDenied for denied reads."""
        config = SandboxConfig(
            allow_cwd=False,
            file_permissions=[],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)
        safe_open = make_safe_open(manager)

        # Should raise PermissionDenied with proper metadata
        with pytest.raises(PermissionDenied) as exc_info:
            safe_open(str(temp_cwd / "allowed.txt"), "r")

        assert exc_info.value.mode == "read"
        assert "allowed.txt" in exc_info.value.path

    def test_safe_open_write_denied(self, temp_cwd: Path) -> None:
        """Test safe_open raises PermissionDenied for denied writes."""
        config = SandboxConfig(allow_cwd=True, allow_cwd_write=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        safe_open = make_safe_open(manager)

        # Should raise PermissionDenied for write
        with pytest.raises(PermissionDenied) as exc_info:
            safe_open(str(temp_cwd / "new.txt"), "w")

        assert exc_info.value.mode == "write"
        assert "new.txt" in exc_info.value.path

    def test_safe_open_write_allowed(self, temp_cwd: Path) -> None:
        """Test safe_open allows writing to permitted files."""
        config = SandboxConfig(allow_cwd=True, allow_cwd_write=True)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        safe_open = make_safe_open(manager)

        # Should be able to write
        test_file = temp_cwd / "new.txt"
        with safe_open(str(test_file), "w") as f:
            f.write("new content")
        assert test_file.read_text() == "new content"

    def test_safe_open_detects_write_modes(self, temp_cwd: Path) -> None:
        """Test safe_open correctly identifies write modes."""
        config = SandboxConfig(allow_cwd=True, allow_cwd_write=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        safe_open = make_safe_open(manager)

        # All these should require write permission
        write_modes = ["w", "w+", "a", "a+", "x", "r+"]
        for mode in write_modes:
            with pytest.raises(PermissionDenied) as exc_info:
                safe_open(str(temp_cwd / "test.txt"), mode)
            assert exc_info.value.mode == "write"


class TestConfigParsing:
    """Test sandbox config parsing from YAML."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary config directory."""
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        return config_dir

    def test_sandbox_config_from_yaml(self, temp_config_dir: Path) -> None:
        """Test loading sandbox config from YAML file."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
sandbox:
  allow_cwd: true
  allow_cwd_write: true
  deny_by_default: true
  allow_absolute: false
  file_permissions:
    - pattern: "./data/**"
      mode: read
    - pattern: "./output/**"
      mode: write
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.allow_cwd is True
        assert config.sandbox.allow_cwd_write is True
        assert config.sandbox.deny_by_default is True
        assert config.sandbox.allow_absolute is False
        assert len(config.sandbox.file_permissions) == 2
        assert config.sandbox.file_permissions[0].pattern == "./data/**"
        assert config.sandbox.file_permissions[0].mode == "read"
        assert config.sandbox.file_permissions[1].pattern == "./output/**"
        assert config.sandbox.file_permissions[1].mode == "write"

    def test_sandbox_config_defaults(self, temp_config_dir: Path) -> None:
        """Test sandbox config uses sensible defaults."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
# Empty config should use defaults
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.allow_cwd is True
        assert config.sandbox.allow_cwd_write is False
        assert config.sandbox.deny_by_default is True
        assert config.sandbox.allow_absolute is False
        assert config.sandbox.file_permissions == []


class TestTimelineIntegration:
    """Test Timeline integration with PermissionManager."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "allowed.txt").write_text("allowed content")
        return tmp_path

    @pytest.mark.asyncio
    async def test_timeline_blocks_unauthorized_read(self, temp_cwd: Path) -> None:
        """Test Timeline blocks unauthorized file reads."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        timeline = Timeline("test-session", cwd=str(temp_cwd), permission_manager=manager)

        # Use forward slashes to avoid Windows path escaping issues
        file_path = (temp_cwd / "allowed.txt").as_posix()
        result = await timeline.execute_statement(
            f'open("{file_path}", "r").read()'
        )

        assert result.status.value == "error"
        assert result.exception is not None
        assert "PermissionError" in result.exception["type"]

    @pytest.mark.asyncio
    async def test_timeline_allows_authorized_read(self, temp_cwd: Path) -> None:
        """Test Timeline allows authorized file reads."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=True)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        timeline = Timeline("test-session", cwd=str(temp_cwd), permission_manager=manager)

        # Use forward slashes to avoid Windows path escaping issues
        file_path = (temp_cwd / "allowed.txt").as_posix()
        result = await timeline.execute_statement(
            f'open("{file_path}", "r").read()'
        )

        assert result.status.value == "ok"
        assert "allowed content" in result.stdout

    @pytest.mark.asyncio
    async def test_timeline_ls_permissions(self, temp_cwd: Path) -> None:
        """Test Timeline exposes ls_permissions function."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(
            allow_cwd=True,
            file_permissions=[
                FilePermissionConfig(pattern="./data/**", mode="read"),
            ],
        )
        manager = PermissionManager.from_config(str(temp_cwd), config)
        timeline = Timeline("test-session", cwd=str(temp_cwd), permission_manager=manager)

        result = await timeline.execute_statement("ls_permissions()")

        assert result.status.value == "ok"
        assert "auto" in result.stdout  # Should show auto rule
        assert "config" in result.stdout  # Should show config rule

    @pytest.mark.asyncio
    async def test_timeline_without_permission_manager(self, temp_cwd: Path) -> None:
        """Test Timeline works without permission manager (no restrictions)."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        # Use forward slashes to avoid Windows path escaping issues
        file_path = (temp_cwd / "allowed.txt").as_posix()
        result = await timeline.execute_statement(
            f'open("{file_path}", "r").read()'
        )

        # Should work without restrictions
        assert result.status.value == "ok"
        assert "allowed content" in result.stdout
