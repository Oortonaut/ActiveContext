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


class TestImportGuard:
    """Test the ImportGuard class."""

    def test_empty_whitelist_denies_all(self) -> None:
        """Test empty whitelist denies all imports."""
        guard = ImportGuard()

        assert not guard.is_allowed("os")
        assert not guard.is_allowed("sys")
        assert not guard.is_allowed("json")

    def test_allow_all_bypasses_whitelist(self) -> None:
        """Test allow_all=True permits any import."""
        guard = ImportGuard(allow_all=True)

        assert guard.is_allowed("os")
        assert guard.is_allowed("subprocess")
        assert guard.is_allowed("anything")

    def test_exact_module_match(self) -> None:
        """Test exact module name matching."""
        guard = ImportGuard(allowed_modules={"json", "math"})

        assert guard.is_allowed("json")
        assert guard.is_allowed("math")
        assert not guard.is_allowed("os")
        assert not guard.is_allowed("sys")

    def test_submodule_allowed_by_default(self) -> None:
        """Test submodules are allowed when parent is whitelisted."""
        guard = ImportGuard(
            allowed_modules={"os"},
            allow_submodules=True,
        )

        assert guard.is_allowed("os")
        assert guard.is_allowed("os.path")
        assert guard.is_allowed("os.path.join")
        assert not guard.is_allowed("sys")

    def test_submodule_denied_when_disabled(self) -> None:
        """Test submodules are denied when allow_submodules=False."""
        guard = ImportGuard(
            allowed_modules={"os"},
            allow_submodules=False,
        )

        assert guard.is_allowed("os")
        assert not guard.is_allowed("os.path")

    def test_explicit_submodule_whitelist(self) -> None:
        """Test explicitly whitelisted submodules work regardless of allow_submodules."""
        guard = ImportGuard(
            allowed_modules={"os.path"},
            allow_submodules=False,
        )

        # os itself is not allowed
        assert not guard.is_allowed("os")
        # But os.path is explicitly allowed
        assert guard.is_allowed("os.path")

    def test_add_module(self) -> None:
        """Test adding modules to whitelist."""
        guard = ImportGuard()
        assert not guard.is_allowed("json")

        guard.add_module("json")
        assert guard.is_allowed("json")

    def test_remove_module(self) -> None:
        """Test removing modules from whitelist."""
        guard = ImportGuard(allowed_modules={"json", "math"})
        assert guard.is_allowed("json")

        guard.remove_module("json")
        assert not guard.is_allowed("json")
        assert guard.is_allowed("math")  # math still allowed

    def test_list_allowed(self) -> None:
        """Test listing allowed modules."""
        guard = ImportGuard(allowed_modules={"json", "math", "os"})

        allowed = guard.list_allowed()
        assert allowed == ["json", "math", "os"]  # Should be sorted

    def test_from_config(self) -> None:
        """Test creating ImportGuard from ImportConfig."""
        config = ImportConfig(
            allowed_modules=["json", "math"],
            allow_submodules=False,
            allow_all=False,
        )

        guard = ImportGuard.from_config(config)

        assert guard.is_allowed("json")
        assert guard.is_allowed("math")
        assert not guard.is_allowed("os")
        assert not guard.allow_submodules
        assert not guard.allow_all

    def test_from_none_config(self) -> None:
        """Test creating ImportGuard from None config uses defaults."""
        guard = ImportGuard.from_config(None)

        assert guard.allowed_modules == set()
        assert guard.allow_submodules is True
        assert guard.allow_all is False


class TestSafeImport:
    """Test the make_safe_import wrapper."""

    def test_allowed_import_succeeds(self) -> None:
        """Test importing whitelisted modules succeeds."""
        guard = ImportGuard(allowed_modules={"json"})
        safe_import = make_safe_import(guard)

        # Should be able to import json
        module = safe_import("json")
        assert module.__name__ == "json"

    def test_denied_import_raises(self) -> None:
        """Test importing non-whitelisted modules raises ImportDenied."""
        guard = ImportGuard(allowed_modules={"json"})
        safe_import = make_safe_import(guard)

        with pytest.raises(ImportDenied) as exc_info:
            safe_import("os")

        assert exc_info.value.module == "os"
        assert exc_info.value.top_level == "os"

    def test_submodule_import_with_parent_allowed(self) -> None:
        """Test importing submodules when parent is whitelisted."""
        guard = ImportGuard(
            allowed_modules={"os"},
            allow_submodules=True,
        )
        safe_import = make_safe_import(guard)

        # Should be able to import os.path
        module = safe_import("os.path")
        assert "path" in module.__name__

    def test_from_import_allowed(self) -> None:
        """Test 'from X import Y' style imports."""
        guard = ImportGuard(allowed_modules={"json"})
        safe_import = make_safe_import(guard)

        # Simulate: from json import dumps
        module = safe_import("json", fromlist=("dumps",))
        assert hasattr(module, "dumps")

    def test_from_import_denied_module(self) -> None:
        """Test 'from X import Y' with denied module."""
        guard = ImportGuard(allowed_modules={"json"})
        safe_import = make_safe_import(guard)

        with pytest.raises(ImportDenied):
            safe_import("os", fromlist=("path",))


class TestImportTimelineIntegration:
    """Test Timeline integration with ImportGuard."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_timeline_blocks_unauthorized_import(self, temp_cwd: Path) -> None:
        """Test Timeline blocks unauthorized imports."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json"})
        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        result = await timeline.execute_statement("import os")

        assert result.status.value == "error"
        assert result.exception is not None
        assert "ImportDenied" in result.exception["type"]

    @pytest.mark.asyncio
    async def test_timeline_allows_authorized_import(self, temp_cwd: Path) -> None:
        """Test Timeline allows authorized imports."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json"})
        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        result = await timeline.execute_statement("import json")

        assert result.status.value == "ok"
        assert "json" in timeline.get_namespace()

    @pytest.mark.asyncio
    async def test_timeline_allows_from_import(self, temp_cwd: Path) -> None:
        """Test Timeline allows 'from X import Y' for whitelisted modules."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json"})
        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        result = await timeline.execute_statement("from json import dumps")

        assert result.status.value == "ok"
        assert "dumps" in timeline.get_namespace()

    @pytest.mark.asyncio
    async def test_timeline_ls_imports(self, temp_cwd: Path) -> None:
        """Test Timeline exposes ls_imports function."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json", "math"})
        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        result = await timeline.execute_statement("ls_imports()")

        assert result.status.value == "ok"
        assert "json" in result.stdout
        assert "math" in result.stdout

    @pytest.mark.asyncio
    async def test_timeline_without_import_guard(self, temp_cwd: Path) -> None:
        """Test Timeline works without import guard (no restrictions)."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        result = await timeline.execute_statement("import json")

        # Should work without restrictions
        assert result.status.value == "ok"
        assert "json" in timeline.get_namespace()

    @pytest.mark.asyncio
    async def test_timeline_import_guard_with_allow_all(self, temp_cwd: Path) -> None:
        """Test Timeline with allow_all=True permits any import."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allow_all=True)
        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        # Should allow any import
        result = await timeline.execute_statement("import os")
        assert result.status.value == "ok"

    @pytest.mark.asyncio
    async def test_timeline_set_import_guard(self, temp_cwd: Path) -> None:
        """Test setting import guard after timeline creation."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=str(temp_cwd))

        # Initially no guard, imports work
        result = await timeline.execute_statement("import os")
        assert result.status.value == "ok"

        # Set guard
        guard = ImportGuard(allowed_modules={"json"})
        timeline.set_import_guard(guard)

        # Now os should be blocked
        result = await timeline.execute_statement("import sys")
        assert result.status.value == "error"
        assert "ImportDenied" in result.exception["type"]


class TestImportConfigParsing:
    """Test import config parsing from YAML."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary config directory."""
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        return config_dir

    def test_import_config_from_yaml(self, temp_config_dir: Path) -> None:
        """Test loading import config from YAML file."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
sandbox:
  imports:
    allowed_modules:
      - json
      - math
      - collections
    allow_submodules: false
    allow_all: false
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.imports.allowed_modules == ["json", "math", "collections"]
        assert config.sandbox.imports.allow_submodules is False
        assert config.sandbox.imports.allow_all is False

    def test_import_config_defaults(self, temp_config_dir: Path) -> None:
        """Test import config uses sensible defaults."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
# Empty config should use defaults
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.imports.allowed_modules == []
        assert config.sandbox.imports.allow_submodules is True
        assert config.sandbox.imports.allow_all is False


class TestPermissionDenied:
    """Test the PermissionDenied exception."""

    def test_permission_denied_carries_metadata(self) -> None:
        """Test PermissionDenied exception carries correct metadata."""
        exc = PermissionDenied(
            path="/abs/path/to/file.txt",
            mode="read",
            original_path="./file.txt",
        )

        assert exc.path == "/abs/path/to/file.txt"
        assert exc.mode == "read"
        assert exc.original_path == "./file.txt"
        assert "read" in str(exc)
        assert "/abs/path/to/file.txt" in str(exc)

    def test_permission_denied_write_mode(self) -> None:
        """Test PermissionDenied with write mode."""
        exc = PermissionDenied(
            path="/abs/path/to/output.txt",
            mode="write",
            original_path="output.txt",
        )

        assert exc.mode == "write"
        assert "write" in str(exc)


class TestTemporaryGrants:
    """Test the temporary grant functionality."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "protected.txt").write_text("protected content")
        return tmp_path

    def test_grant_temporary_allows_access(self, temp_cwd: Path) -> None:
        """Test grant_temporary() allows subsequent access."""
        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Initially denied
        file_path = str(temp_cwd / "protected.txt")
        assert not manager.check_access(file_path, "read")

        # Grant temporary access
        manager.grant_temporary(file_path, "read")

        # Now should be allowed
        assert manager.check_access(file_path, "read")

    def test_grant_temporary_is_mode_specific(self, temp_cwd: Path) -> None:
        """Test temporary grants are specific to the mode."""
        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        file_path = str(temp_cwd / "protected.txt")

        # Grant read access
        manager.grant_temporary(file_path, "read")

        # Read should be allowed, write should be denied
        assert manager.check_access(file_path, "read")
        assert not manager.check_access(file_path, "write")

    def test_clear_temporary_grants(self, temp_cwd: Path) -> None:
        """Test clear_temporary_grants() removes all temporary grants."""
        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        file_path = str(temp_cwd / "protected.txt")

        # Grant and verify
        manager.grant_temporary(file_path, "read")
        assert manager.check_access(file_path, "read")

        # Clear and verify
        manager.clear_temporary_grants()
        assert not manager.check_access(file_path, "read")


class TestWritePermissionToConfig:
    """Test write_permission_to_config functionality."""

    def test_creates_config_file_if_not_exists(self, tmp_path: Path) -> None:
        """Test creating a new config file with permission."""
        write_permission_to_config(tmp_path, "./secret.txt", "read")

        config_path = tmp_path / ".ac" / "config.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "sandbox" in data
        assert "file_permissions" in data["sandbox"]
        assert len(data["sandbox"]["file_permissions"]) == 1
        assert data["sandbox"]["file_permissions"][0]["pattern"] == "./secret.txt"
        assert data["sandbox"]["file_permissions"][0]["mode"] == "read"

    def test_appends_to_existing_config(self, tmp_path: Path) -> None:
        """Test appending permission to existing config."""
        # Create initial config
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text(
            """
llm:
  model: test-model
sandbox:
  file_permissions:
    - pattern: ./existing.txt
      mode: read
"""
        )

        # Add new permission
        write_permission_to_config(tmp_path, "./new.txt", "write")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Should have both permissions
        assert len(data["sandbox"]["file_permissions"]) == 2
        assert data["sandbox"]["file_permissions"][0]["pattern"] == "./existing.txt"
        assert data["sandbox"]["file_permissions"][1]["pattern"] == "./new.txt"
        assert data["sandbox"]["file_permissions"][1]["mode"] == "write"

        # Should preserve other config
        assert data["llm"]["model"] == "test-model"

    def test_does_not_duplicate_existing_rule(self, tmp_path: Path) -> None:
        """Test that duplicate rules are not added."""
        # Add first rule
        write_permission_to_config(tmp_path, "./file.txt", "read")

        # Try to add the same rule again
        write_permission_to_config(tmp_path, "./file.txt", "read")

        config_path = tmp_path / ".ac" / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Should only have one rule
        assert len(data["sandbox"]["file_permissions"]) == 1


class TestPermissionRequestFlow:
    """Test the permission request flow in Timeline."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "protected.txt").write_text("protected content")
        return tmp_path

    @pytest.mark.asyncio
    async def test_permission_request_allow_once(self, temp_cwd: Path) -> None:
        """Test allow_once grants temporary access."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Mock permission requester that always grants once
        async def mock_requester(
            session_id: str, path: str, mode: str
        ) -> tuple[bool, bool]:
            return (True, False)  # granted=True, persist=False

        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            permission_manager=manager,
            permission_requester=mock_requester,
        )

        file_path = (temp_cwd / "protected.txt").as_posix()
        result = await timeline.execute_statement(f'open("{file_path}", "r").read()')

        assert result.status.value == "ok"
        assert "protected content" in result.stdout

    @pytest.mark.asyncio
    async def test_permission_request_allow_always(self, temp_cwd: Path) -> None:
        """Test allow_always persists permission to config."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Mock permission requester that grants always
        async def mock_requester(
            session_id: str, path: str, mode: str
        ) -> tuple[bool, bool]:
            return (True, True)  # granted=True, persist=True

        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            permission_manager=manager,
            permission_requester=mock_requester,
        )

        file_path = (temp_cwd / "protected.txt").as_posix()
        result = await timeline.execute_statement(f'open("{file_path}", "r").read()')

        assert result.status.value == "ok"
        assert "protected content" in result.stdout

        # Verify config was updated
        config_path = temp_cwd / ".ac" / "config.yaml"
        assert config_path.exists()

    @pytest.mark.asyncio
    async def test_permission_request_denied(self, temp_cwd: Path) -> None:
        """Test denied permission returns PermissionError."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # Mock permission requester that denies
        async def mock_requester(
            session_id: str, path: str, mode: str
        ) -> tuple[bool, bool]:
            return (False, False)  # denied

        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            permission_manager=manager,
            permission_requester=mock_requester,
        )

        file_path = (temp_cwd / "protected.txt").as_posix()
        result = await timeline.execute_statement(f'open("{file_path}", "r").read()')

        assert result.status.value == "error"
        assert result.exception is not None
        assert result.exception["type"] == "PermissionError"

    @pytest.mark.asyncio
    async def test_no_requester_returns_permission_error(self, temp_cwd: Path) -> None:
        """Test that without a requester, PermissionError is returned."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # No permission requester
        timeline = Timeline(
            "test-session",
            cwd=str(temp_cwd),
            permission_manager=manager,
        )

        file_path = (temp_cwd / "protected.txt").as_posix()
        result = await timeline.execute_statement(f'open("{file_path}", "r").read()')

        assert result.status.value == "error"
        assert result.exception is not None
        # Should be PermissionError, not PermissionDenied
        assert result.exception["type"] == "PermissionError"
