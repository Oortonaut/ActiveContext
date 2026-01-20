"""Tests for the agent type registry."""

import pytest
from pathlib import Path

from activecontext.agents.registry import AgentTypeRegistry
from activecontext.agents.schema import AgentType


class TestBuiltinTypes:
    """Tests for built-in agent types."""

    def test_default_type_exists(self):
        registry = AgentTypeRegistry()
        agent_type = registry.get("default")

        assert agent_type is not None
        assert agent_type.id == "default"
        assert agent_type.name == "Default Agent"

    def test_explorer_type_exists(self):
        registry = AgentTypeRegistry()
        agent_type = registry.get("explorer")

        assert agent_type is not None
        assert agent_type.id == "explorer"
        assert "read" in agent_type.capabilities
        assert "search" in agent_type.capabilities

    def test_summarizer_type_exists(self):
        registry = AgentTypeRegistry()
        agent_type = registry.get("summarizer")

        assert agent_type is not None
        assert agent_type.id == "summarizer"
        assert "read" in agent_type.capabilities
        assert "summarize" in agent_type.capabilities

    def test_get_nonexistent_returns_none(self):
        registry = AgentTypeRegistry()
        assert registry.get("nonexistent") is None


class TestRegistration:
    """Tests for agent type registration."""

    def test_register_new_type(self):
        registry = AgentTypeRegistry()
        custom_type = AgentType(
            id="custom",
            name="Custom Agent",
            system_prompt="You are a custom agent.",
            capabilities=["custom_cap"],
        )

        registry.register(custom_type)

        retrieved = registry.get("custom")
        assert retrieved is not None
        assert retrieved.id == "custom"
        assert retrieved.capabilities == ["custom_cap"]

    def test_register_overwrites_existing(self):
        registry = AgentTypeRegistry()

        # Register custom type
        v1 = AgentType(id="test", name="Test V1", system_prompt="V1")
        registry.register(v1)

        # Overwrite with new version
        v2 = AgentType(id="test", name="Test V2", system_prompt="V2")
        registry.register(v2)

        retrieved = registry.get("test")
        assert retrieved.name == "Test V2"

    def test_unregister_removes_type(self):
        registry = AgentTypeRegistry()
        custom_type = AgentType(id="temp", name="Temp", system_prompt="Temp")
        registry.register(custom_type)

        result = registry.unregister("temp")

        assert result is True
        assert registry.get("temp") is None

    def test_unregister_nonexistent_returns_false(self):
        registry = AgentTypeRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_unregister_builtin_type(self):
        registry = AgentTypeRegistry()

        result = registry.unregister("default")

        assert result is True
        assert registry.get("default") is None


class TestListTypes:
    """Tests for listing agent types."""

    def test_list_includes_builtins(self):
        registry = AgentTypeRegistry()
        types = registry.list_types()

        type_ids = {t.id for t in types}
        assert "default" in type_ids
        assert "explorer" in type_ids
        assert "summarizer" in type_ids

    def test_list_includes_registered(self):
        registry = AgentTypeRegistry()
        custom = AgentType(id="custom", name="Custom", system_prompt="Custom")
        registry.register(custom)

        types = registry.list_types()
        type_ids = {t.id for t in types}

        assert "custom" in type_ids

    def test_list_returns_copies(self):
        registry = AgentTypeRegistry()
        types1 = registry.list_types()
        types2 = registry.list_types()

        # Lists should be different objects
        assert types1 is not types2


class TestLoadFromDirectory:
    """Tests for loading agent types from YAML files."""

    def test_load_from_nonexistent_directory(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        nonexistent = tmp_path / "nonexistent"

        loaded = registry.load_from_directory(nonexistent)

        assert loaded == 0

    def test_load_from_empty_directory(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        loaded = registry.load_from_directory(empty_dir)

        assert loaded == 0

    def test_load_valid_yaml(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        yaml_content = """
id: yaml_agent
name: YAML Agent
system_prompt: You are a YAML-defined agent.
default_mode: plan
capabilities:
  - read
  - write
"""
        (agents_dir / "yaml_agent.yaml").write_text(yaml_content)

        loaded = registry.load_from_directory(agents_dir)

        assert loaded == 1
        agent = registry.get("yaml_agent")
        assert agent is not None
        assert agent.name == "YAML Agent"
        assert agent.default_mode == "plan"
        assert "read" in agent.capabilities

    def test_load_multiple_yaml_files(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create two valid YAML files
        yaml1 = "id: agent1\nname: Agent 1\nsystem_prompt: Prompt 1\n"
        yaml2 = "id: agent2\nname: Agent 2\nsystem_prompt: Prompt 2\n"

        (agents_dir / "agent1.yaml").write_text(yaml1)
        (agents_dir / "agent2.yaml").write_text(yaml2)

        loaded = registry.load_from_directory(agents_dir)

        assert loaded == 2
        assert registry.get("agent1") is not None
        assert registry.get("agent2") is not None

    def test_skip_invalid_yaml(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Valid YAML
        valid = "id: valid\nname: Valid\nsystem_prompt: Prompt\n"
        (agents_dir / "valid.yaml").write_text(valid)

        # Invalid YAML (syntax error)
        invalid = "id: invalid\nname: [unclosed bracket\n"
        (agents_dir / "invalid.yaml").write_text(invalid)

        loaded = registry.load_from_directory(agents_dir)

        assert loaded == 1
        assert registry.get("valid") is not None
        assert registry.get("invalid") is None

    def test_skip_non_dict_yaml(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # YAML that parses to a list, not dict
        list_yaml = "- item1\n- item2\n"
        (agents_dir / "list.yaml").write_text(list_yaml)

        loaded = registry.load_from_directory(agents_dir)

        assert loaded == 0

    def test_ignores_non_yaml_files(self, tmp_path: Path):
        registry = AgentTypeRegistry()
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()

        # Create non-YAML files
        (agents_dir / "readme.txt").write_text("This is a readme")
        (agents_dir / "data.json").write_text('{"id": "json_agent"}')

        loaded = registry.load_from_directory(agents_dir)

        assert loaded == 0


class TestLoadUserTypes:
    """Tests for loading user-defined agent types."""

    def test_load_user_types_from_ac_directory(self, tmp_path: Path):
        registry = AgentTypeRegistry()

        # Create .ac/agents directory structure
        agents_dir = tmp_path / ".ac" / "agents"
        agents_dir.mkdir(parents=True)

        yaml_content = "id: user_agent\nname: User Agent\nsystem_prompt: User defined\n"
        (agents_dir / "user_agent.yaml").write_text(yaml_content)

        loaded = registry.load_user_types(tmp_path)

        assert loaded == 1
        assert registry.get("user_agent") is not None

    def test_load_user_types_no_ac_directory(self, tmp_path: Path):
        registry = AgentTypeRegistry()

        # No .ac directory exists
        loaded = registry.load_user_types(tmp_path)

        assert loaded == 0

    def test_load_user_types_empty_agents_directory(self, tmp_path: Path):
        registry = AgentTypeRegistry()

        # Create empty .ac/agents directory
        agents_dir = tmp_path / ".ac" / "agents"
        agents_dir.mkdir(parents=True)

        loaded = registry.load_user_types(tmp_path)

        assert loaded == 0
