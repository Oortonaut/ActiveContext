"""Tests for the file permission system."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from activecontext.config.schema import (
    FilePermissionConfig,
    ImportConfig,
    SandboxConfig,
    ShellPermissionConfig,
)
from activecontext.context.graph import ContextGraph
from activecontext.session.permissions import (
    GlobSegment,
    ImportDenied,
    ImportGuard,
    LiteralSegment,
    MatchResult,
    PatternMatcher,
    PermissionDenied,
    PermissionManager,
    PermissionRule,
    PlaceholderSegment,
    ShellPermissionDenied,
    ShellPermissionManager,
    ShellPermissionRule,
    TypeValidator,
    is_typed_pattern,
    make_safe_import,
    make_safe_open,
    parse_pattern,
    write_permission_to_config,
    write_shell_permission_to_config,
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
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd), permission_manager=manager)

        try:
            # Use forward slashes to avoid Windows path escaping issues
            file_path = (temp_cwd / "allowed.txt").as_posix()
            result = await timeline.execute_statement(
                f'open("{file_path}", "r").read()'
            )

            assert result.status.value == "error"
            assert result.exception is not None
            assert "PermissionError" in result.exception["type"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_allows_authorized_read(self, temp_cwd: Path) -> None:
        """Test Timeline allows authorized file reads."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=True)
        manager = PermissionManager.from_config(str(temp_cwd), config)
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd), permission_manager=manager)

        try:
            # Use forward slashes to avoid Windows path escaping issues
            # Use pathlib.Path.read_text() to ensure file is properly closed
            file_path = (temp_cwd / "allowed.txt").as_posix()
            result = await timeline.execute_statement(
                f'__import__("pathlib").Path("{file_path}").read_text()'
            )

            assert result.status.value == "ok"
            assert "allowed content" in result.stdout
        finally:
            await timeline.close()

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
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd), permission_manager=manager)

        try:
            result = await timeline.execute_statement("ls_permissions()")

            assert result.status.value == "ok"
            assert "auto" in result.stdout  # Should show auto rule
            assert "config" in result.stdout  # Should show config rule
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_without_permission_manager(self, temp_cwd: Path) -> None:
        """Test Timeline works without permission manager (no restrictions)."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Use forward slashes to avoid Windows path escaping issues
            # Use pathlib.Path.read_text() to ensure file is properly closed
            file_path = (temp_cwd / "allowed.txt").as_posix()
            result = await timeline.execute_statement(
                f'__import__("pathlib").Path("{file_path}").read_text()'
            )

            # Should work without restrictions
            assert result.status.value == "ok"
            assert "allowed content" in result.stdout
        finally:
            await timeline.close()


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
        # Note: __import__("os.path") returns os module, not os.path
        # The import succeeds if no exception is raised
        module = safe_import("os.path")
        assert module.__name__ == "os"  # Parent module is returned
        assert hasattr(module, "path")  # But path submodule is loaded

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
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        try:
            result = await timeline.execute_statement("import os")

            assert result.status.value == "error"
            assert result.exception is not None
            # ImportDenied is converted to ImportError for LLM consumption
            assert result.exception["type"] == "ImportError"
            assert "not in the allowed modules whitelist" in result.exception["message"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_allows_authorized_import(self, temp_cwd: Path) -> None:
        """Test Timeline allows authorized imports."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json"})
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        try:
            result = await timeline.execute_statement("import json")

            assert result.status.value == "ok"
            assert "json" in timeline.get_namespace()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_allows_from_import(self, temp_cwd: Path) -> None:
        """Test Timeline allows 'from X import Y' for whitelisted modules."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json"})
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        try:
            result = await timeline.execute_statement("from json import dumps")

            assert result.status.value == "ok"
            assert "dumps" in timeline.get_namespace()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_ls_imports(self, temp_cwd: Path) -> None:
        """Test Timeline exposes ls_imports function."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allowed_modules={"json", "math"})
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        try:
            result = await timeline.execute_statement("ls_imports()")

            assert result.status.value == "ok"
            assert "json" in result.stdout
            assert "math" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_without_import_guard(self, temp_cwd: Path) -> None:
        """Test Timeline works without import guard (no restrictions)."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("import json")

            # Should work without restrictions
            assert result.status.value == "ok"
            assert "json" in timeline.get_namespace()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_import_guard_with_allow_all(self, temp_cwd: Path) -> None:
        """Test Timeline with allow_all=True permits any import."""
        from activecontext.session.timeline import Timeline

        guard = ImportGuard(allow_all=True)
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            import_guard=guard,
        )

        try:
            # Should allow any import
            result = await timeline.execute_statement("import os")
            assert result.status.value == "ok"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_set_import_guard(self, temp_cwd: Path) -> None:
        """Test setting import guard after timeline creation."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Initially no guard, imports work
            result = await timeline.execute_statement("import os")
            assert result.status.value == "ok"

            # Set guard
            guard = ImportGuard(allowed_modules={"json"})
            timeline.set_import_guard(guard)

            # Now os should be blocked
            result = await timeline.execute_statement("import sys")
            assert result.status.value == "error"
            # ImportDenied is converted to ImportError for LLM consumption
            assert result.exception["type"] == "ImportError"
        finally:
            await timeline.close()


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
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            permission_manager=manager,
            permission_requester=mock_requester,
        )

        try:
            # Use lambda to ensure file is properly closed while still using open()
            # (permission checks only wrap open(), not pathlib)
            file_path = (temp_cwd / "protected.txt").as_posix()
            result = await timeline.execute_statement(
                f'(lambda f: (f.read(), f.close())[0])(open("{file_path}", "r"))'
            )

            assert result.status.value == "ok"
            assert "protected content" in result.stdout
        finally:
            await timeline.close()

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
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            permission_manager=manager,
            permission_requester=mock_requester,
        )

        try:
            # Use lambda to ensure file is properly closed while still using open()
            # (permission checks only wrap open(), not pathlib)
            file_path = (temp_cwd / "protected.txt").as_posix()
            result = await timeline.execute_statement(
                f'(lambda f: (f.read(), f.close())[0])(open("{file_path}", "r"))'
            )

            assert result.status.value == "ok"
            assert "protected content" in result.stdout

            # Verify config was updated
            config_path = temp_cwd / ".ac" / "config.yaml"
            assert config_path.exists()
        finally:
            await timeline.close()

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
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            permission_manager=manager,
            permission_requester=mock_requester,
        )

        try:
            file_path = (temp_cwd / "protected.txt").as_posix()
            result = await timeline.execute_statement(f'open("{file_path}", "r").read()')

            assert result.status.value == "error"
            assert result.exception is not None
            assert result.exception["type"] == "PermissionError"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_no_requester_returns_permission_error(self, temp_cwd: Path) -> None:
        """Test that without a requester, PermissionError is returned."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(allow_cwd=False)
        manager = PermissionManager.from_config(str(temp_cwd), config)

        # No permission requester
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            permission_manager=manager,
        )

        try:
            file_path = (temp_cwd / "protected.txt").as_posix()
            result = await timeline.execute_statement(f'open("{file_path}", "r").read()')

            assert result.status.value == "error"
            assert result.exception is not None
            # Should be PermissionError, not PermissionDenied
            assert result.exception["type"] == "PermissionError"
        finally:
            await timeline.close()


# =============================================================================
# Shell Permission Tests
# =============================================================================


class TestShellPermissionDenied:
    """Test the ShellPermissionDenied exception."""

    def test_shell_permission_denied_carries_metadata(self) -> None:
        """Test ShellPermissionDenied exception carries correct metadata."""
        exc = ShellPermissionDenied(
            command="rm",
            full_command="rm -rf /important",
            command_args=["-rf", "/important"],
        )

        assert exc.command == "rm"
        assert exc.command_args == ["-rf", "/important"]
        assert exc.full_command == "rm -rf /important"
        assert "rm -rf /important" in str(exc)

    def test_shell_permission_denied_no_args(self) -> None:
        """Test ShellPermissionDenied with no arguments."""
        exc = ShellPermissionDenied(
            command="whoami",
            full_command="whoami",
            command_args=None,
        )

        assert exc.command == "whoami"
        assert exc.command_args is None
        assert "whoami" in str(exc)


class TestShellPermissionManager:
    """Test the ShellPermissionManager class."""

    def test_default_config_denies_all(self) -> None:
        """Test default config denies all shell commands."""
        manager = ShellPermissionManager.from_config(None)

        assert manager.deny_by_default is True
        assert len(manager.rules) == 0
        assert not manager.check_access("git", ["status"])
        assert not manager.check_access("npm", ["run", "build"])

    def test_explicit_allow_rule(self) -> None:
        """Test explicit allow rule permits command."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config)

        assert manager.check_access("git", ["status"])
        assert manager.check_access("git", ["commit", "-m", "test"])
        assert not manager.check_access("npm", ["install"])

    def test_explicit_deny_rule(self) -> None:
        """Test explicit deny rule blocks command."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="rm -rf *", allow=False),
                ShellPermissionConfig(pattern="rm *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config)

        # rm -rf should be denied (first rule matches)
        assert not manager.check_access("rm", ["-rf", "/tmp/test"])

        # rm without -rf should be allowed
        assert manager.check_access("rm", ["file.txt"])

    def test_glob_pattern_matching(self) -> None:
        """Test glob patterns work correctly."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="npm run *", allow=True),
                ShellPermissionConfig(pattern="pytest *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config)

        assert manager.check_access("npm", ["run", "build"])
        assert manager.check_access("npm", ["run", "test"])
        assert manager.check_access("pytest", ["tests/"])
        assert not manager.check_access("npm", ["install"])
        assert not manager.check_access("pip", ["install", "requests"])

    def test_deny_by_default_false(self) -> None:
        """Test deny_by_default=False allows unlisted commands."""
        config = SandboxConfig(
            shell_permissions=[],
            shell_deny_by_default=False,
        )
        manager = ShellPermissionManager.from_config(config)

        # All commands should be allowed
        assert manager.check_access("anything")
        assert manager.check_access("rm", ["-rf", "/"])

    def test_first_match_wins(self) -> None:
        """Test first matching rule wins."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git push *", allow=False),
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config)

        # git push should be denied (first rule)
        assert not manager.check_access("git", ["push", "origin", "main"])

        # other git commands should be allowed (second rule)
        assert manager.check_access("git", ["status"])
        assert manager.check_access("git", ["commit", "-m", "test"])

    def test_reload_updates_rules(self) -> None:
        """Test reload() updates permission rules."""
        config1 = SandboxConfig(
            shell_permissions=[],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config1)

        # Initially denied
        assert not manager.check_access("git", ["status"])

        # Reload with allow rule
        config2 = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager.reload(config2)

        # Now should be allowed
        assert manager.check_access("git", ["status"])

    def test_list_permissions(self) -> None:
        """Test list_permissions returns rule details."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git *", allow=True),
                ShellPermissionConfig(pattern="rm -rf *", allow=False),
            ],
        )
        manager = ShellPermissionManager.from_config(config)

        perms = manager.list_permissions()
        assert len(perms) == 2
        assert perms[0]["pattern"] == "git *"
        assert perms[0]["allow"] is True
        assert perms[1]["pattern"] == "rm -rf *"
        assert perms[1]["allow"] is False


class TestShellTemporaryGrants:
    """Test the shell temporary grant functionality."""

    def test_grant_temporary_allows_access(self) -> None:
        """Test grant_temporary() allows subsequent access."""
        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # Initially denied
        assert not manager.check_access("dangerous", ["command"])

        # Grant temporary access
        manager.grant_temporary("dangerous", ["command"])

        # Now should be allowed
        assert manager.check_access("dangerous", ["command"])

    def test_grant_temporary_is_exact_match(self) -> None:
        """Test temporary grants match exactly."""
        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # Grant for specific command + args
        manager.grant_temporary("rm", ["file.txt"])

        # Exact match should be allowed
        assert manager.check_access("rm", ["file.txt"])

        # Different args should be denied
        assert not manager.check_access("rm", ["other.txt"])
        assert not manager.check_access("rm", ["-rf", "file.txt"])

    def test_clear_temporary_grants(self) -> None:
        """Test clear_temporary_grants() removes all temporary grants."""
        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # Grant and verify
        manager.grant_temporary("test", ["command"])
        assert manager.check_access("test", ["command"])

        # Clear and verify
        manager.clear_temporary_grants()
        assert not manager.check_access("test", ["command"])


class TestWriteShellPermissionToConfig:
    """Test write_shell_permission_to_config functionality."""

    def test_creates_config_file_if_not_exists(self, tmp_path: Path) -> None:
        """Test creating a new config file with shell permission."""
        write_shell_permission_to_config(tmp_path, "git", ["status"])

        config_path = tmp_path / ".ac" / "config.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "sandbox" in data
        assert "shell_permissions" in data["sandbox"]
        assert len(data["sandbox"]["shell_permissions"]) == 1
        assert data["sandbox"]["shell_permissions"][0]["pattern"] == "git status"
        assert data["sandbox"]["shell_permissions"][0]["allow"] is True

    def test_appends_to_existing_config(self, tmp_path: Path) -> None:
        """Test appending shell permission to existing config."""
        # Create initial config
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text(
            """
llm:
  model: test-model
sandbox:
  shell_permissions:
    - pattern: git *
      allow: true
"""
        )

        # Add new permission
        write_shell_permission_to_config(tmp_path, "npm", ["run", "test"])

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Should have both permissions
        assert len(data["sandbox"]["shell_permissions"]) == 2
        assert data["sandbox"]["shell_permissions"][0]["pattern"] == "git *"
        assert data["sandbox"]["shell_permissions"][1]["pattern"] == "npm run test"

        # Should preserve other config
        assert data["llm"]["model"] == "test-model"

    def test_does_not_duplicate_existing_rule(self, tmp_path: Path) -> None:
        """Test that duplicate rules are not added."""
        # Add first rule
        write_shell_permission_to_config(tmp_path, "pytest", ["tests/"])

        # Try to add the same rule again
        write_shell_permission_to_config(tmp_path, "pytest", ["tests/"])

        config_path = tmp_path / ".ac" / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Should only have one rule
        assert len(data["sandbox"]["shell_permissions"]) == 1

    def test_command_without_args(self, tmp_path: Path) -> None:
        """Test writing permission for command without args."""
        write_shell_permission_to_config(tmp_path, "whoami", None)

        config_path = tmp_path / ".ac" / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert data["sandbox"]["shell_permissions"][0]["pattern"] == "whoami"


class TestShellPermissionConfigParsing:
    """Test shell permission config parsing from YAML."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary config directory."""
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        return config_dir

    def test_shell_config_from_yaml(self, temp_config_dir: Path) -> None:
        """Test loading shell config from YAML file."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
sandbox:
  shell_deny_by_default: true
  shell_permissions:
    - pattern: "git *"
      allow: true
    - pattern: "npm run *"
      allow: true
    - pattern: "rm -rf *"
      allow: false
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.shell_deny_by_default is True
        assert len(config.sandbox.shell_permissions) == 3
        assert config.sandbox.shell_permissions[0].pattern == "git *"
        assert config.sandbox.shell_permissions[0].allow is True
        assert config.sandbox.shell_permissions[2].pattern == "rm -rf *"
        assert config.sandbox.shell_permissions[2].allow is False

    def test_shell_config_defaults(self, temp_config_dir: Path) -> None:
        """Test shell config uses sensible defaults."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
# Empty config should use defaults
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.shell_deny_by_default is True
        assert config.sandbox.shell_permissions == []


class TestShellTimelineIntegration:
    """Test Timeline integration with ShellPermissionManager."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.mark.asyncio
    async def test_timeline_blocks_unauthorized_shell(self, temp_cwd: Path) -> None:
        """Test Timeline blocks unauthorized shell commands."""
        import asyncio
        from activecontext.session.timeline import Timeline
        from activecontext.context.nodes import ShellNode, ShellStatus

        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            shell_permission_manager=manager,
        )

        try:
            result = await timeline.execute_statement('s = shell("echo", ["hello"])')

            assert result.status.value == "ok"
            # Shell now returns ShellNode immediately with PENDING status
            ns = timeline.get_namespace()
            assert "s" in ns
            shell_node = ns["s"]
            assert isinstance(shell_node, ShellNode)

            # Wait briefly for background task to complete
            await asyncio.sleep(0.1)

            # Process pending results (applies async completions)
            timeline.process_pending_shell_results()

            # The shell should have been denied (exit code 126)
            assert shell_node.shell_status == ShellStatus.FAILED
            assert shell_node.exit_code == 126
            assert "denied" in shell_node.output.lower()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_allows_authorized_shell(self, temp_cwd: Path) -> None:
        """Test Timeline allows authorized shell commands."""
        import asyncio
        import sys
        from activecontext.session.timeline import Timeline
        from activecontext.context.nodes import ShellNode, ShellStatus

        # Windows uses cmd.exe for echo; Unix uses echo directly
        if sys.platform == "win32":
            pattern = "cmd *"
            cmd = "cmd"
            args = ["/c", "echo", "hello"]
        else:
            pattern = "echo *"
            cmd = "echo"
            args = ["hello"]

        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern=pattern, allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config)
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            shell_permission_manager=manager,
        )

        try:
            result = await timeline.execute_statement(f's = shell("{cmd}", {args!r})')

            assert result.status.value == "ok"
            ns = timeline.get_namespace()
            assert "s" in ns
            shell_node = ns["s"]
            assert isinstance(shell_node, ShellNode)

            # Wait for background task to complete
            await asyncio.sleep(0.5)

            # Process pending results
            timeline.process_pending_shell_results()

            # Should have completed successfully with output
            assert shell_node.is_complete
            assert shell_node.exit_code == 0
            assert "hello" in shell_node.output
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_ls_shell_permissions(self, temp_cwd: Path) -> None:
        """Test Timeline exposes ls_shell_permissions function."""
        from activecontext.session.timeline import Timeline

        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config)
        timeline = Timeline(
            "test-session",
            context_graph=ContextGraph(),
            cwd=str(temp_cwd),
            shell_permission_manager=manager,
        )

        try:
            result = await timeline.execute_statement("ls_shell_permissions()")

            assert result.status.value == "ok"
            assert "git *" in result.stdout
            assert "deny_by_default" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_timeline_without_shell_permission_manager(
        self, temp_cwd: Path
    ) -> None:
        """Test Timeline works without shell permission manager (no restrictions)."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('shell("echo", ["hello"])')

            # Should work without restrictions
            assert result.status.value == "ok"
        finally:
            await timeline.close()


class TestShellPermissionRequestFlow:
    """Test the shell permission request flow in Timeline."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory."""
        return tmp_path

    @pytest.fixture
    async def timeline_factory(self, temp_cwd: Path):
        """Factory for creating timelines with proper cleanup."""
        import asyncio
        timelines: list = []

        def create(**kwargs):
            from activecontext.session.timeline import Timeline
            tl = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd), **kwargs)
            timelines.append(tl)
            return tl

        yield create

        # Cleanup: cancel all background tasks
        for tl in timelines:
            for task in tl._shell_tasks.values():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    @pytest.mark.asyncio
    async def test_shell_permission_request_allow_once(self, timeline_factory) -> None:
        """Test allow_once grants temporary shell access."""
        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # Mock permission requester that always grants once
        async def mock_requester(
            session_id: str, command: str, args: list[str] | None
        ) -> tuple[bool, bool]:
            return (True, False)  # granted=True, persist=False

        timeline = timeline_factory(
            shell_permission_manager=manager,
            shell_permission_requester=mock_requester,
        )

        result = await timeline.execute_statement('shell("echo", ["hello"])')

        assert result.status.value == "ok"
        # Should not be error/denied (exit=126)
        assert "exit=126" not in result.stdout

    @pytest.mark.asyncio
    async def test_shell_permission_request_allow_always(self, temp_cwd: Path, timeline_factory) -> None:
        """Test allow_always persists shell permission to config."""
        import asyncio

        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # Mock permission requester that grants always
        async def mock_requester(
            session_id: str, command: str, args: list[str] | None
        ) -> tuple[bool, bool]:
            return (True, True)  # granted=True, persist=True

        timeline = timeline_factory(
            shell_permission_manager=manager,
            shell_permission_requester=mock_requester,
        )

        result = await timeline.execute_statement('s = shell("echo", ["hello"])')
        assert result.status.value == "ok"

        # Wait for background task to complete
        await asyncio.sleep(0.2)

        # Verify config was updated
        config_path = temp_cwd / ".ac" / "config.yaml"
        assert config_path.exists(), f"Config file not found at {config_path}"

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "sandbox" in data
        assert "shell_permissions" in data["sandbox"]
        # Should have added echo hello rule
        patterns = [p["pattern"] for p in data["sandbox"]["shell_permissions"]]
        assert "echo hello" in patterns

    @pytest.mark.asyncio
    async def test_shell_permission_request_denied(self, timeline_factory) -> None:
        """Test denied shell permission returns error result."""
        from activecontext.context.nodes import ShellNode, ShellStatus

        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # Mock permission requester that denies
        async def mock_requester(
            session_id: str, command: str, args: list[str] | None
        ) -> tuple[bool, bool]:
            return (False, False)  # denied

        timeline = timeline_factory(
            shell_permission_manager=manager,
            shell_permission_requester=mock_requester,
        )

        result = await timeline.execute_statement('s = shell("echo", ["hello"])')

        assert result.status.value == "ok"

        # Wait for background task and process results
        import asyncio
        await asyncio.sleep(0.2)
        timeline.process_pending_shell_results()

        # Check the ShellNode has permission denied status
        ns = timeline.get_namespace()
        assert "s" in ns
        shell_node = ns["s"]
        assert isinstance(shell_node, ShellNode)
        # Permission denied should set exit_code=126 and FAILED status
        assert shell_node.shell_status == ShellStatus.FAILED
        assert shell_node.exit_code == 126

    @pytest.mark.asyncio
    async def test_no_shell_requester_returns_denied(self, timeline_factory) -> None:
        """Test that without a requester, shell command is denied."""
        import asyncio
        from activecontext.context.nodes import ShellNode, ShellStatus

        config = SandboxConfig(shell_deny_by_default=True)
        manager = ShellPermissionManager.from_config(config)

        # No permission requester
        timeline = timeline_factory(
            shell_permission_manager=manager,
        )

        result = await timeline.execute_statement('s = shell("echo", ["hello"])')

        assert result.status.value == "ok"

        # Wait for background task and process results
        await asyncio.sleep(0.2)
        timeline.process_pending_shell_results()

        # Check the ShellNode has permission denied status
        ns = timeline.get_namespace()
        assert "s" in ns
        shell_node = ns["s"]
        assert isinstance(shell_node, ShellNode)
        # Permission denied should set exit_code=126 and FAILED status
        assert shell_node.shell_status == ShellStatus.FAILED
        assert shell_node.exit_code == 126


# =============================================================================
# Typed Placeholder Pattern Tests
# =============================================================================


class TestPatternParsing:
    """Test the pattern parsing functionality."""

    def test_parse_literal_pattern(self) -> None:
        """Test parsing a pattern with only literals."""
        segments = parse_pattern("git status")
        assert len(segments) == 2
        assert isinstance(segments[0], LiteralSegment)
        assert segments[0].value == "git"
        assert isinstance(segments[1], LiteralSegment)
        assert segments[1].value == "status"

    def test_parse_named_placeholder(self) -> None:
        """Test parsing a pattern with named placeholder."""
        segments = parse_pattern("rm -rf {target:dir}")
        assert len(segments) == 3
        assert isinstance(segments[0], LiteralSegment)
        assert segments[0].value == "rm"
        assert isinstance(segments[1], LiteralSegment)
        assert segments[1].value == "-rf"
        assert isinstance(segments[2], PlaceholderSegment)
        assert segments[2].name == "target"
        assert segments[2].type == "dir"

    def test_parse_anonymous_placeholder(self) -> None:
        """Test parsing a pattern with anonymous placeholder."""
        segments = parse_pattern("git add {:args}")
        assert len(segments) == 3
        assert isinstance(segments[0], LiteralSegment)
        assert isinstance(segments[1], LiteralSegment)
        assert isinstance(segments[2], PlaceholderSegment)
        assert segments[2].name is None
        assert segments[2].type == "args"

    def test_parse_glob_pattern(self) -> None:
        """Test parsing a pattern with legacy glob."""
        segments = parse_pattern("npm run *")
        assert len(segments) == 3
        assert isinstance(segments[0], LiteralSegment)
        assert isinstance(segments[1], LiteralSegment)
        assert isinstance(segments[2], GlobSegment)

    def test_parse_mixed_pattern(self) -> None:
        """Test parsing a pattern with mixed elements."""
        segments = parse_pattern("git add {:args} {file:r}")
        assert len(segments) == 4
        assert isinstance(segments[0], LiteralSegment)
        assert isinstance(segments[1], LiteralSegment)
        assert isinstance(segments[2], PlaceholderSegment)
        assert segments[2].type == "args"
        assert isinstance(segments[3], PlaceholderSegment)
        assert segments[3].name == "file"
        assert segments[3].type == "r"

    def test_is_typed_pattern(self) -> None:
        """Test is_typed_pattern detection."""
        assert is_typed_pattern("rm -rf {target:dir}")
        assert is_typed_pattern("{:args}")
        assert not is_typed_pattern("git *")
        assert not is_typed_pattern("npm run build")


class TestTypeValidator:
    """Test the TypeValidator class."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "test_dir").mkdir()
        (tmp_path / "test_file.txt").write_text("test content")
        return tmp_path

    def test_validate_dir_type(self, temp_cwd: Path) -> None:
        """Test dir type validation."""
        validator = TypeValidator(cwd=temp_cwd)

        # Existing directory should pass
        assert validator.validate("test_dir", "dir")

        # File should fail
        assert not validator.validate("test_file.txt", "dir")

        # Non-existent path should fail
        assert not validator.validate("nonexistent", "dir")

    def test_validate_path_type(self, temp_cwd: Path) -> None:
        """Test path type validation."""
        validator = TypeValidator(cwd=temp_cwd)

        # Existing directory should pass
        assert validator.validate("test_dir", "path")

        # Existing file should pass
        assert validator.validate("test_file.txt", "path")

        # Non-existent path should fail
        assert not validator.validate("nonexistent", "path")

    def test_validate_str_type(self, temp_cwd: Path) -> None:
        """Test str type validation."""
        validator = TypeValidator(cwd=temp_cwd)

        # Any string should pass
        assert validator.validate("anything", "str")
        assert validator.validate("123", "str")
        assert validator.validate("", "str")

    def test_validate_int_type(self, temp_cwd: Path) -> None:
        """Test int type validation."""
        validator = TypeValidator(cwd=temp_cwd)

        assert validator.validate("123", "int")
        assert validator.validate("-456", "int")
        assert validator.validate("0", "int")
        assert not validator.validate("abc", "int")
        assert not validator.validate("12.34", "int")

    def test_validate_args_type(self, temp_cwd: Path) -> None:
        """Test args type validation (always passes)."""
        validator = TypeValidator(cwd=temp_cwd)

        assert validator.validate("any value", "args")
        assert validator.validate("", "args")

    def test_validate_read_type_with_permission_manager(self, temp_cwd: Path) -> None:
        """Test read type validation with permission manager."""
        # Create permission manager that allows reading test_file.txt
        config = SandboxConfig(allow_cwd=True)
        perm_manager = PermissionManager.from_config(str(temp_cwd), config)

        validator = TypeValidator(cwd=temp_cwd, permission_manager=perm_manager)

        # File exists and is readable
        assert validator.validate("test_file.txt", "r")
        assert validator.validate("test_file.txt", "read")

    def test_validate_write_type_with_permission_manager(self, temp_cwd: Path) -> None:
        """Test write type validation with permission manager."""
        # Create permission manager that allows writing
        config = SandboxConfig(allow_cwd=True, allow_cwd_write=True)
        perm_manager = PermissionManager.from_config(str(temp_cwd), config)

        validator = TypeValidator(cwd=temp_cwd, permission_manager=perm_manager)

        # File should be writable
        assert validator.validate("test_file.txt", "w")
        assert validator.validate("test_file.txt", "write")

    def test_validate_mdir_type(self, temp_cwd: Path) -> None:
        """Test mdir (mutable directory) type validation."""
        # Create permission manager that allows writing
        config = SandboxConfig(allow_cwd=True, allow_cwd_write=True)
        perm_manager = PermissionManager.from_config(str(temp_cwd), config)

        validator = TypeValidator(cwd=temp_cwd, permission_manager=perm_manager)

        # Directory with write permission
        assert validator.validate("test_dir", "mdir")

        # File should fail (not a directory)
        assert not validator.validate("test_file.txt", "mdir")

    def test_validate_unknown_type(self, temp_cwd: Path) -> None:
        """Test unknown type validation fails."""
        validator = TypeValidator(cwd=temp_cwd)

        assert not validator.validate("anything", "unknown_type")


class TestPatternMatcher:
    """Test the PatternMatcher class."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "test_dir").mkdir()
        (tmp_path / "test_file.txt").write_text("test content")
        return tmp_path

    def test_match_literal_pattern(self, temp_cwd: Path) -> None:
        """Test matching a literal pattern."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("git status", "git", ["status"])
        assert result.matched
        assert result.captures == {}

        result = matcher.match("git status", "git", ["commit"])
        assert not result.matched

    def test_match_glob_pattern(self, temp_cwd: Path) -> None:
        """Test matching a legacy glob pattern."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("git *", "git", ["status"])
        assert result.matched

        result = matcher.match("git *", "git", ["commit", "-m", "test"])
        assert result.matched

        result = matcher.match("npm *", "git", ["status"])
        assert not result.matched

    def test_match_dir_placeholder(self, temp_cwd: Path) -> None:
        """Test matching a dir type placeholder."""
        matcher = PatternMatcher(cwd=temp_cwd)

        # Should match when target is a directory
        result = matcher.match("rm -rf {target:dir}", "rm", ["-rf", "test_dir"])
        assert result.matched
        assert result.captures == {"target": "test_dir"}

        # Should not match when target is a file
        result = matcher.match("rm -rf {target:dir}", "rm", ["-rf", "test_file.txt"])
        assert not result.matched

        # Should not match when target doesn't exist
        result = matcher.match("rm -rf {target:dir}", "rm", ["-rf", "nonexistent"])
        assert not result.matched

    def test_match_str_placeholder(self, temp_cwd: Path) -> None:
        """Test matching a str type placeholder."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("echo {msg:str}", "echo", ["hello"])
        assert result.matched
        assert result.captures == {"msg": "hello"}

    def test_match_anonymous_placeholder(self, temp_cwd: Path) -> None:
        """Test matching an anonymous placeholder."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("echo {:str}", "echo", ["hello"])
        assert result.matched
        assert result.captures == {}  # Anonymous placeholders don't capture

    def test_match_args_placeholder(self, temp_cwd: Path) -> None:
        """Test matching an args type placeholder."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("git add {files:args}", "git", ["add", "a.txt", "b.txt"])
        assert result.matched
        assert result.captures == {"files": "a.txt b.txt"}

    def test_match_args_with_trailing_literal(self, temp_cwd: Path) -> None:
        """Test args placeholder followed by literal."""
        matcher = PatternMatcher(cwd=temp_cwd)

        # Pattern: command {:args} --flag
        result = matcher.match("cmd {:args} --end", "cmd", ["a", "b", "--end"])
        assert result.matched

    def test_match_int_placeholder(self, temp_cwd: Path) -> None:
        """Test matching an int type placeholder."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("kill {pid:int}", "kill", ["1234"])
        assert result.matched
        assert result.captures == {"pid": "1234"}

        result = matcher.match("kill {pid:int}", "kill", ["abc"])
        assert not result.matched

    def test_match_multiple_placeholders(self, temp_cwd: Path) -> None:
        """Test matching multiple placeholders."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match(
            "cp {src:path} {dst:str}", "cp", ["test_file.txt", "output.txt"]
        )
        assert result.matched
        assert result.captures == {"src": "test_file.txt", "dst": "output.txt"}

    def test_match_too_few_tokens(self, temp_cwd: Path) -> None:
        """Test matching fails with too few tokens."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("rm -rf {target:str}", "rm", ["-rf"])
        assert not result.matched

    def test_match_too_many_tokens(self, temp_cwd: Path) -> None:
        """Test matching fails with too many tokens."""
        matcher = PatternMatcher(cwd=temp_cwd)

        result = matcher.match("git status", "git", ["status", "extra"])
        assert not result.matched


class TestShellPermissionManagerTypedPatterns:
    """Test ShellPermissionManager with typed patterns."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        """Create a temporary working directory with test files."""
        (tmp_path / "test_dir").mkdir()
        (tmp_path / "test_file.txt").write_text("test content")
        return tmp_path

    def test_typed_pattern_matches_directory(self, temp_cwd: Path) -> None:
        """Test typed pattern matches only directories."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="rm -rf {target:dir}", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        # Should match directory
        assert manager.check_access("rm", ["-rf", "test_dir"])

        # Should not match file
        assert not manager.check_access("rm", ["-rf", "test_file.txt"])

        # Should not match non-existent path
        assert not manager.check_access("rm", ["-rf", "nonexistent"])

    def test_typed_pattern_returns_captures(self, temp_cwd: Path) -> None:
        """Test typed pattern returns captured values."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="rm -rf {target:dir}", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        allowed, captures = manager.check_access_with_captures("rm", ["-rf", "test_dir"])
        assert allowed
        assert captures == {"target": "test_dir"}

    def test_legacy_glob_still_works(self, temp_cwd: Path) -> None:
        """Test legacy glob patterns still work."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        assert manager.check_access("git", ["status"])
        assert manager.check_access("git", ["commit", "-m", "test"])
        assert not manager.check_access("npm", ["install"])

    def test_mixed_patterns(self, temp_cwd: Path) -> None:
        """Test mixing typed and glob patterns."""
        config = SandboxConfig(
            shell_permissions=[
                # Typed pattern: only delete directories
                ShellPermissionConfig(pattern="rm -rf {target:dir}", allow=True),
                # Legacy glob: any git command
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        # rm -rf should only work on directories
        assert manager.check_access("rm", ["-rf", "test_dir"])
        assert not manager.check_access("rm", ["-rf", "test_file.txt"])

        # git commands should work
        assert manager.check_access("git", ["status"])

    def test_path_type_placeholder(self, temp_cwd: Path) -> None:
        """Test path type placeholder accepts files and directories."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="ls {target:path}", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        # Should match directory
        assert manager.check_access("ls", ["test_dir"])

        # Should match file
        assert manager.check_access("ls", ["test_file.txt"])

        # Should not match non-existent
        assert not manager.check_access("ls", ["nonexistent"])

    def test_args_placeholder_captures_remaining(self, temp_cwd: Path) -> None:
        """Test args placeholder captures remaining tokens."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="git add {files:args}", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        allowed, captures = manager.check_access_with_captures(
            "git", ["add", "file1.txt", "file2.txt"]
        )
        assert allowed
        assert captures == {"files": "file1.txt file2.txt"}

    def test_file_permission_integration(self, temp_cwd: Path) -> None:
        """Test file permission integration with read/write types."""
        # Create permission manager that allows reading but not writing
        file_config = SandboxConfig(allow_cwd=True, allow_cwd_write=False)
        perm_manager = PermissionManager.from_config(str(temp_cwd), file_config)

        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="cat {file:r}", allow=True),
                ShellPermissionConfig(pattern="write_to {file:w}", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(
            config, cwd=temp_cwd, permission_manager=perm_manager
        )

        # cat should work (file is readable)
        assert manager.check_access("cat", ["test_file.txt"])

        # write_to should fail (file is not writable per file permissions)
        assert not manager.check_access("write_to", ["test_file.txt"])

    def test_no_cwd_falls_back_to_fnmatch(self) -> None:
        """Test without cwd, typed patterns fall back to fnmatch."""
        config = SandboxConfig(
            shell_permissions=[
                # This pattern won't work as typed without cwd
                ShellPermissionConfig(pattern="rm -rf {target:dir}", allow=True),
                # This glob should still work
                ShellPermissionConfig(pattern="git *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        # No cwd provided
        manager = ShellPermissionManager.from_config(config, cwd=None)

        # Typed pattern won't match (no matcher)
        # But it will fall back to fnmatch which won't match either
        # because fnmatch doesn't know about {target:dir}
        assert not manager.check_access("rm", ["-rf", "somedir"])

        # Legacy glob should still work
        assert manager.check_access("git", ["status"])

    def test_first_match_wins_with_typed(self, temp_cwd: Path) -> None:
        """Test first matching rule wins with typed patterns."""
        config = SandboxConfig(
            shell_permissions=[
                # Deny rm -rf on directories
                ShellPermissionConfig(pattern="rm -rf {target:dir}", allow=False),
                # Allow rm on anything (but this won't match rm -rf since first rule matched)
                ShellPermissionConfig(pattern="rm *", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        # rm -rf on directory should be denied (first rule)
        assert not manager.check_access("rm", ["-rf", "test_dir"])

        # rm without -rf should be allowed (second rule)
        assert manager.check_access("rm", ["test_file.txt"])

    def test_int_type_placeholder(self, temp_cwd: Path) -> None:
        """Test int type placeholder validates numeric values."""
        config = SandboxConfig(
            shell_permissions=[
                ShellPermissionConfig(pattern="kill {pid:int}", allow=True),
            ],
            shell_deny_by_default=True,
        )
        manager = ShellPermissionManager.from_config(config, cwd=temp_cwd)

        # Should match numeric PID
        assert manager.check_access("kill", ["1234"])
        assert manager.check_access("kill", ["-9"])  # Negative works

        # Should not match non-numeric
        assert not manager.check_access("kill", ["all"])


# =============================================================================
# Website Permission Tests
# =============================================================================


class TestURLTypeValidator:
    """Test URL type validators."""

    def test_domain_valid(self) -> None:
        """Test valid domain names."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("github.com", "domain")
        assert URLTypeValidator.validate("api.example.com", "domain")
        assert URLTypeValidator.validate("sub.domain.example.com", "domain")
        assert URLTypeValidator.validate("example-site.com", "domain")

    def test_domain_invalid(self) -> None:
        """Test invalid domain names."""
        from activecontext.session.permissions import URLTypeValidator

        assert not URLTypeValidator.validate("-invalid.com", "domain")
        assert not URLTypeValidator.validate("domain-.com", "domain")
        assert not URLTypeValidator.validate("192.168.1.1", "domain")

    def test_host_valid_domain(self) -> None:
        """Test host validator with valid domains."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("github.com", "host")
        assert URLTypeValidator.validate("api.example.com", "host")

    def test_host_valid_ipv4(self) -> None:
        """Test host validator with IPv4 addresses."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("192.168.1.1", "host")
        assert URLTypeValidator.validate("127.0.0.1", "host")
        assert URLTypeValidator.validate("8.8.8.8", "host")

    def test_host_valid_ipv6(self) -> None:
        """Test host validator with IPv6 addresses."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("::1", "host")
        assert URLTypeValidator.validate("2001:0db8:85a3::8a2e:0370:7334", "host")
        assert URLTypeValidator.validate("[::1]", "host")  # Bracketed
        assert URLTypeValidator.validate("2001:db8::1", "host")

    def test_host_invalid(self) -> None:
        """Test host validator with invalid values."""
        from activecontext.session.permissions import URLTypeValidator

        assert not URLTypeValidator.validate("999.999.999.999", "host")
        assert not URLTypeValidator.validate("not a host!", "host")

    def test_url_valid(self) -> None:
        """Test valid URLs."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("https://example.com", "url")
        assert URLTypeValidator.validate("http://localhost:8080/api", "url")
        assert URLTypeValidator.validate("https://api.github.com/repos?page=1", "url")
        assert URLTypeValidator.validate("ws://localhost:3000/socket", "url")

    def test_url_invalid(self) -> None:
        """Test invalid URLs."""
        from activecontext.session.permissions import URLTypeValidator

        assert not URLTypeValidator.validate("not-a-url", "url")
        assert not URLTypeValidator.validate("://missing-scheme", "url")
        assert not URLTypeValidator.validate("http://", "url")  # No netloc

    def test_port_valid(self) -> None:
        """Test valid port numbers."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("80", "port")
        assert URLTypeValidator.validate("8080", "port")
        assert URLTypeValidator.validate("65535", "port")
        assert URLTypeValidator.validate("1", "port")

    def test_port_invalid(self) -> None:
        """Test invalid port numbers."""
        from activecontext.session.permissions import URLTypeValidator

        assert not URLTypeValidator.validate("0", "port")
        assert not URLTypeValidator.validate("65536", "port")
        assert not URLTypeValidator.validate("abc", "port")

    def test_protocol_valid(self) -> None:
        """Test valid protocols."""
        from activecontext.session.permissions import URLTypeValidator

        assert URLTypeValidator.validate("http", "protocol")
        assert URLTypeValidator.validate("https", "protocol")
        assert URLTypeValidator.validate("ws", "protocol")
        assert URLTypeValidator.validate("wss", "protocol")
        assert URLTypeValidator.validate("HTTP", "protocol")  # Case insensitive

    def test_protocol_invalid(self) -> None:
        """Test invalid protocols."""
        from activecontext.session.permissions import URLTypeValidator

        assert not URLTypeValidator.validate("ftp", "protocol")
        assert not URLTypeValidator.validate("ssh", "protocol")


class TestURLPatternMatcher:
    """Test URL pattern matching."""

    def test_exact_match(self) -> None:
        """Test exact URL matching."""
        from activecontext.session.permissions import URLPatternMatcher

        matcher = URLPatternMatcher()
        result = matcher.match(
            "https://api.github.com/repos", "https://api.github.com/repos"
        )
        assert result.matched is True

    def test_glob_pattern(self) -> None:
        """Test glob pattern matching."""
        from activecontext.session.permissions import URLPatternMatcher

        matcher = URLPatternMatcher()
        result = matcher.match("https://api.github.com/*", "https://api.github.com/repos")
        assert result.matched is True

    def test_host_placeholder_with_domain(self) -> None:
        """Test host placeholder with domain."""
        from activecontext.session.permissions import URLPatternMatcher

        matcher = URLPatternMatcher()
        result = matcher.match("https://{host:host}/api", "https://github.com/api")
        assert result.matched is True
        assert result.captures["host"] == "github.com"

    def test_host_placeholder_with_ipv4(self) -> None:
        """Test host placeholder with IPv4."""
        from activecontext.session.permissions import URLPatternMatcher

        matcher = URLPatternMatcher()
        result = matcher.match("http://{host:host}:{port:port}/api", "http://192.168.1.1:8080/api")
        assert result.matched is True
        assert result.captures["host"] == "192.168.1.1"
        assert result.captures["port"] == "8080"

    def test_url_placeholder(self) -> None:
        """Test url placeholder."""
        from activecontext.session.permissions import URLPatternMatcher

        matcher = URLPatternMatcher()
        result = matcher.match(
            "Redirect to {target:url}", "Redirect to https://example.com/callback"
        )
        assert result.matched is True
        assert result.captures["target"] == "https://example.com/callback"

    def test_multiple_placeholders(self) -> None:
        """Test multiple placeholders."""
        from activecontext.session.permissions import URLPatternMatcher

        matcher = URLPatternMatcher()
        result = matcher.match(
            "{protocol:protocol}://{host:host}/{endpoint:path}",
            "https://api.example.com/v1/users",
        )
        assert result.matched is True
        assert result.captures["protocol"] == "https"
        assert result.captures["host"] == "api.example.com"
        assert result.captures["endpoint"] == "v1/users"


class TestWebsitePermissionManager:
    """Test website permission manager."""

    def test_default_config_denies_all(self, tmp_path: Path) -> None:
        """Test default config denies all requests."""
        from activecontext.session.permissions import WebsitePermissionManager

        manager = WebsitePermissionManager.from_config(None, tmp_path)
        assert manager.deny_by_default is True
        assert not manager.check_access("https://example.com", "GET")

    def test_explicit_allow_rule(self, tmp_path: Path) -> None:
        """Test explicit allow rule."""
        from activecontext.config.schema import WebsitePermissionConfig

        config = SandboxConfig(
            website_permissions=[
                WebsitePermissionConfig(pattern="https://api.github.com/*", methods=["GET"]),
            ],
        )
        from activecontext.session.permissions import WebsitePermissionManager

        manager = WebsitePermissionManager.from_config(config, tmp_path)

        assert manager.check_access("https://api.github.com/repos", "GET")
        assert not manager.check_access("https://api.github.com/repos", "POST")
        assert not manager.check_access("https://example.com", "GET")

    def test_host_placeholder_allows_ip_and_domain(self, tmp_path: Path) -> None:
        """Test host placeholder with IP and domain."""
        from activecontext.config.schema import WebsitePermissionConfig

        config = SandboxConfig(
            website_permissions=[
                WebsitePermissionConfig(
                    pattern="http://{host:host}/api",
                    methods=["GET"],
                ),
            ],
        )
        from activecontext.session.permissions import WebsitePermissionManager

        manager = WebsitePermissionManager.from_config(config, tmp_path)

        assert manager.check_access("http://example.com/api", "GET")
        assert manager.check_access("http://192.168.1.1/api", "GET")
        assert manager.check_access("http://[::1]/api", "GET")

    def test_allow_localhost(self, tmp_path: Path) -> None:
        """Test allow_localhost setting."""
        from activecontext.config.schema import WebsitePermissionConfig

        config = SandboxConfig(
            website_permissions=[],
            website_deny_by_default=True,
            allow_localhost=True,
        )
        from activecontext.session.permissions import WebsitePermissionManager

        manager = WebsitePermissionManager.from_config(config, tmp_path)

        assert manager.check_access("http://localhost:8080/api", "GET")
        assert manager.check_access("http://127.0.0.1/api", "GET")
        assert not manager.check_access("http://example.com/api", "GET")

    def test_all_methods(self, tmp_path: Path) -> None:
        """Test ALL methods wildcard."""
        from activecontext.config.schema import WebsitePermissionConfig

        config = SandboxConfig(
            website_permissions=[
                WebsitePermissionConfig(
                    pattern="https://api.example.com/*",
                    methods=["ALL"],
                ),
            ],
        )
        from activecontext.session.permissions import WebsitePermissionManager

        manager = WebsitePermissionManager.from_config(config, tmp_path)

        assert manager.check_access("https://api.example.com/users", "GET")
        assert manager.check_access("https://api.example.com/users", "POST")
        assert manager.check_access("https://api.example.com/users", "PUT")
        assert manager.check_access("https://api.example.com/users", "DELETE")

    def test_temporary_grants(self, tmp_path: Path) -> None:
        """Test temporary grants."""
        from activecontext.session.permissions import WebsitePermissionManager

        config = SandboxConfig(website_deny_by_default=True)
        manager = WebsitePermissionManager.from_config(config, tmp_path)

        url = "https://example.com/api"
        assert not manager.check_access(url, "GET")

        manager.grant_temporary(url, "GET")
        assert manager.check_access(url, "GET")
        assert not manager.check_access(url, "POST")

    def test_clear_temporary_grants(self, tmp_path: Path) -> None:
        """Test clearing temporary grants."""
        from activecontext.session.permissions import WebsitePermissionManager

        config = SandboxConfig(website_deny_by_default=True)
        manager = WebsitePermissionManager.from_config(config, tmp_path)

        url = "https://example.com/api"
        manager.grant_temporary(url, "GET")
        assert manager.check_access(url, "GET")

        manager.clear_temporary_grants()
        assert not manager.check_access(url, "GET")


class TestWriteWebsitePermissionToConfig:
    """Test write_website_permission_to_config functionality."""

    def test_creates_config_file_if_not_exists(self, tmp_path: Path) -> None:
        """Test creating a new config file with website permission."""
        from activecontext.session.permissions import write_website_permission_to_config

        write_website_permission_to_config(
            tmp_path, "https://api.github.com/*", "GET"
        )

        config_path = tmp_path / ".ac" / "config.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "sandbox" in data
        assert "website_permissions" in data["sandbox"]
        assert len(data["sandbox"]["website_permissions"]) == 1
        assert data["sandbox"]["website_permissions"][0]["pattern"] == "https://api.github.com/*"
        assert data["sandbox"]["website_permissions"][0]["methods"] == ["GET"]

    def test_merges_methods_for_existing_pattern(self, tmp_path: Path) -> None:
        """Test merging methods for existing pattern."""
        from activecontext.session.permissions import write_website_permission_to_config

        # Add first permission
        write_website_permission_to_config(tmp_path, "https://api.example.com/*", "GET")

        # Add same pattern with different method
        write_website_permission_to_config(tmp_path, "https://api.example.com/*", "POST")

        config_path = tmp_path / ".ac" / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Should have merged methods
        assert len(data["sandbox"]["website_permissions"]) == 1
        methods = data["sandbox"]["website_permissions"][0]["methods"]
        assert set(methods) == {"GET", "POST"}


class TestWebsitePermissionConfigParsing:
    """Test website permission config parsing from YAML."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary config directory."""
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        return config_dir

    def test_website_config_from_yaml(self, temp_config_dir: Path) -> None:
        """Test loading website config from YAML file."""
        from activecontext.config import load_config, reset_config

        reset_config()

        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
sandbox:
  website_deny_by_default: true
  allow_localhost: false
  website_permissions:
    - pattern: "https://api.github.com/*"
      methods: [GET, POST]
    - pattern: "http://{host:host}/api"
      methods: [GET]
"""
        )

        config = load_config(session_root=str(temp_config_dir.parent))

        assert config.sandbox.website_deny_by_default is True
        assert config.sandbox.allow_localhost is False
        assert len(config.sandbox.website_permissions) == 2
        assert config.sandbox.website_permissions[0].pattern == "https://api.github.com/*"
        assert set(config.sandbox.website_permissions[0].methods) == {"GET", "POST"}
        assert config.sandbox.website_permissions[1].pattern == "http://{host:host}/api"
