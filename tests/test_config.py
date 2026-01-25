"""Tests for the configuration module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from activecontext.config import (
    Config,
    get_config,
    load_config,
    reset_config,
)
from activecontext.config.merge import deep_merge, merge_configs
from activecontext.config.paths import (
    get_config_paths,
    get_project_config_path,
    get_system_config_path,
    get_user_config_path,
)


class TestDeepMerge:
    """Test the deep merge algorithm."""

    def test_simple_override(self) -> None:
        """Test that override values replace base values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Test that nested dicts are recursively merged."""
        base = {"llm": {"model": "gpt-4", "temperature": 0.5}}
        override = {"llm": {"temperature": 0.0}}
        result = deep_merge(base, override)
        assert result["llm"]["model"] == "gpt-4"
        assert result["llm"]["temperature"] == 0.0

    def test_none_does_not_override(self) -> None:
        """Test that None values in override don't replace base values."""
        base = {"a": 1}
        override = {"a": None}
        result = deep_merge(base, override)
        assert result["a"] == 1

    def test_list_replaced_not_merged(self) -> None:
        """Test that lists are replaced, not concatenated."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result["items"] == [4, 5]

    def test_deeply_nested(self) -> None:
        """Test deeply nested merging."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 3}}}
        result = deep_merge(base, override)
        assert result["a"]["b"]["c"] == 3
        assert result["a"]["b"]["d"] == 2

    def test_merge_configs_multiple(self) -> None:
        """Test merging multiple configs in order."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3}
        config3 = {"c": 4}
        result = merge_configs(config1, config2, config3)
        assert result == {"a": 1, "b": 3, "c": 4}


class TestConfigPaths:
    """Test platform-aware path resolution."""

    def test_windows_system_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test system config path on Windows."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("PROGRAMDATA", "C:\\ProgramData")

        path = get_system_config_path()
        assert path is not None
        assert "ProgramData" in str(path)
        assert "activecontext" in str(path)
        assert "config.yaml" in str(path)

    def test_windows_user_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test user config path on Windows."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")

        path = get_user_config_path()
        assert path is not None
        assert "AppData" in str(path)
        assert "activecontext" in str(path)

    def test_unix_system_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test system config path on Unix."""
        monkeypatch.setattr(sys, "platform", "linux")

        path = get_system_config_path()
        assert path == Path("/etc/activecontext/config.yaml")

    def test_unix_user_path_xdg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test user config path respects XDG_CONFIG_HOME."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("XDG_CONFIG_HOME", "/home/test/.config-custom")

        path = get_user_config_path()
        assert path is not None
        assert ".config-custom" in str(path)

    def test_project_config_path(self) -> None:
        """Test project config path construction."""
        path = get_project_config_path("/home/user/myproject")
        assert path == Path("/home/user/myproject/.ac/config.yaml")

    def test_get_config_paths_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that config paths are in correct order."""
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        paths = get_config_paths(session_root="/project")
        assert len(paths) == 3
        # System should be first (use path parts for cross-platform check)
        assert "etc" in paths[0].parts
        # User should be second
        assert ".ac" in str(paths[1]) or ".config" in str(paths[1])
        # Project should be last
        assert "project" in paths[2].parts


class TestConfigLoading:
    """Test configuration loading."""

    @pytest.fixture(autouse=True)
    def reset_global_config(self) -> None:
        """Reset global config cache before each test."""
        reset_config()

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create a temporary config directory."""
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        return config_dir

    def test_load_yaml_config(self, temp_config_dir: Path) -> None:
        """Test loading a valid YAML config file."""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
llm:
  role: coding
  provider: anthropic
"""
        )
        config = load_config(session_root=str(temp_config_dir.parent))
        assert config.llm.role == "coding"
        assert config.llm.provider == "anthropic"

    def test_env_overrides_config(
        self, temp_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that config file values persist (env vars handled separately via fetch_secret)."""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
llm:
  role: coding
  provider: openai
"""
        )
        # API keys are handled via fetch_secret(), not config
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")

        config = load_config(session_root=str(temp_config_dir.parent))
        # Config file values should persist
        assert config.llm.role == "coding"
        assert config.llm.provider == "openai"

    def test_invalid_yaml_uses_defaults(
        self, temp_config_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid YAML falls back to defaults."""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text("invalid: yaml: :")

        config = load_config(session_root=str(temp_config_dir.parent))
        # Should get default values
        assert config.llm.role is None

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        """Test that missing config files use defaults."""
        config = load_config(session_root=str(tmp_path))
        assert isinstance(config, Config)
        assert config.llm.role is None

    def test_session_modes_from_config(self, temp_config_dir: Path) -> None:
        """Test loading session modes from config."""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
session:
  default_mode: plan
  modes:
    - id: normal
      name: Normal
      description: Standard mode
    - id: plan
      name: Plan
      description: Think before acting
"""
        )
        config = load_config(session_root=str(temp_config_dir.parent))
        assert config.session.default_mode == "plan"
        assert len(config.session.modes) == 2
        assert config.session.modes[0].id == "normal"
        assert config.session.modes[1].id == "plan"

    def test_extra_fields_preserved(self, temp_config_dir: Path) -> None:
        """Test that unknown config fields are preserved in extra."""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
custom_field: custom_value
nested:
  field: value
"""
        )
        config = load_config(session_root=str(temp_config_dir.parent))
        assert "custom_field" in config.extra
        assert config.extra["custom_field"] == "custom_value"
        assert config.extra["nested"]["field"] == "value"

    def test_mcp_servers_from_config(self, temp_config_dir: Path) -> None:
        """Test loading MCP server config with extra_args."""
        config_file = temp_config_dir / "config.yaml"
        config_file.write_text(
            """
mcp:
  allow_dynamic_servers: true
  servers:
    - name: task-graph
      transport: stdio
      command: ["task-graph-mcp"]
      extra_args: ["--log", "/tmp/task-graph.log"]
      connect: auto
    - name: rider
      url: "http://127.0.0.1:64342/sse"
      transport: sse
      connect: manual
"""
        )
        config = load_config(session_root=str(temp_config_dir.parent))
        assert config.mcp.allow_dynamic_servers is True
        assert len(config.mcp.servers) == 2

        # Check stdio server with extra_args
        task_graph = config.mcp.servers[0]
        assert task_graph.name == "task-graph"
        assert task_graph.command == ["task-graph-mcp"]
        assert task_graph.extra_args == ["--log", "/tmp/task-graph.log"]
        assert task_graph.transport == "stdio"

        # Check SSE server
        rider = config.mcp.servers[1]
        assert rider.name == "rider"
        assert rider.url == "http://127.0.0.1:64342/sse"
        assert rider.transport == "sse"


class TestBackwardCompatibility:
    """Test backward compatibility with environment variables."""

    @pytest.fixture(autouse=True)
    def reset_global_config(self) -> None:
        """Reset global config cache before each test."""
        reset_config()

    def test_env_only_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that env-var-only users get default config (API keys via fetch_secret)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        config = load_config()
        # API keys are now fetched via fetch_secret(), not stored in config
        # Config should have defaults
        assert config.llm.role is None
        assert config.llm.provider is None

        # But API key should be available via fetch_secret
        from activecontext.config import fetch_secret, clear_secret_cache
        clear_secret_cache()  # Clear cache to pick up monkeypatched env
        assert fetch_secret("ANTHROPIC_API_KEY") == "test-key"

    def test_activecontext_log_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test ACTIVECONTEXT_LOG env var is respected."""
        monkeypatch.setenv("ACTIVECONTEXT_LOG", "/tmp/test.log")

        config = load_config()
        assert config.logging.file == "/tmp/test.log"


class TestConfigCaching:
    """Test config caching behavior."""

    @pytest.fixture(autouse=True)
    def reset_global_config(self) -> None:
        """Reset global config cache before each test."""
        reset_config()

    def test_get_config_caches(self) -> None:
        """Test that get_config returns cached config."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reset_clears_cache(self) -> None:
        """Test that reset_config clears the cache."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        assert config1 is not config2

    def test_project_config_not_cached(self, tmp_path: Path) -> None:
        """Test that project-specific config is not globally cached."""
        # Load project config
        project_config = load_config(session_root=str(tmp_path))
        # Load global config
        global_config = get_config()
        # Should be different objects (project not cached as global)
        assert project_config is not global_config
