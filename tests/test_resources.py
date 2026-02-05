"""Tests for the resources package and import_script DSL function."""

from __future__ import annotations

import pytest

from activecontext.resources import list_prompts, load_prompt, load_resource


class TestLoadResource:
    """Tests for load_resource()."""

    def test_load_prompt_file(self):
        """Load a known prompt file via load_resource."""
        content = load_resource("prompts/system.md")
        assert len(content) > 0
        assert "ActiveContext" in content

    def test_load_config_file(self):
        """Load providers.yaml via load_resource."""
        content = load_resource("config/providers.yaml")
        assert len(content) > 0
        assert "providers" in content

    def test_missing_resource_raises(self):
        """Non-existent resource raises FileNotFoundError or similar."""
        with pytest.raises(FileNotFoundError):
            load_resource("nonexistent/file.txt")


class TestLoadPrompt:
    """Tests for load_prompt()."""

    def test_load_by_name(self):
        """Load a prompt by name without extension."""
        content = load_prompt("system")
        assert len(content) > 0
        assert "ActiveContext" in content

    def test_load_by_name_with_extension(self):
        """Load a prompt by name with .md extension."""
        content = load_prompt("system.md")
        assert len(content) > 0

    def test_load_subdirectory_prompt(self):
        """Load a prompt from modes/ subdirectory."""
        content = load_prompt("modes/normal")
        assert len(content) > 0

    def test_load_subdirectory_with_extension(self):
        """Load a subdirectory prompt with .md extension."""
        content = load_prompt("modes/normal.md")
        assert len(content) > 0

    def test_all_known_prompts_load(self):
        """Verify all expected prompts load without error."""
        expected = [
            "system",
            "dsl_reference",
            "node_states",
            "context_graph",
            "context_guide",
            "work_coordination",
            "mcp",
            "startup",
        ]
        for name in expected:
            content = load_prompt(name)
            assert len(content) > 0, f"Prompt '{name}' is empty"

    def test_mode_prompts_load(self):
        """Verify all mode prompts load."""
        for mode in ("normal", "plan", "brave"):
            content = load_prompt(f"modes/{mode}")
            assert len(content) > 0, f"Mode prompt '{mode}' is empty"


class TestListPrompts:
    """Tests for list_prompts()."""

    def test_returns_list(self):
        """list_prompts returns a list of strings."""
        prompts = list_prompts()
        assert isinstance(prompts, list)
        assert all(isinstance(p, str) for p in prompts)

    def test_contains_known_prompts(self):
        """list_prompts includes known prompt names."""
        prompts = list_prompts()
        assert "system" in prompts
        assert "dsl_reference" in prompts
        assert "startup" in prompts

    def test_no_extensions(self):
        """Prompt names should not include .md extension."""
        prompts = list_prompts()
        assert not any(p.endswith(".md") for p in prompts)


class TestStartupMdParsing:
    """Tests that startup.md parses to the expected statements."""

    def test_statement_count(self):
        """startup.md should produce exactly 6 reference documentation statements."""
        from activecontext.config.schema import PACKAGE_DEFAULT_STARTUP

        assert len(PACKAGE_DEFAULT_STARTUP) == 6

    def test_reference_documentation_statements(self):
        """All 6 statements should load reference documentation."""
        from activecontext.config.schema import PACKAGE_DEFAULT_STARTUP

        expected_prompts = [
            "context_guide",
            "dsl_reference",
            "node_states",
            "context_graph",
            "work_coordination",
            "mcp",
        ]
        for i, prompt_name in enumerate(expected_prompts):
            assert f"@prompts/{prompt_name}.md" in PACKAGE_DEFAULT_STARTUP[i]
            assert "Expansion.ALL" in PACKAGE_DEFAULT_STARTUP[i]

    def test_roundtrip_matches_original(self):
        """Parsed statements should exactly match the original hardcoded list."""
        from activecontext.config.schema import PACKAGE_DEFAULT_STARTUP

        original = [
            'markdown("@prompts/context_guide.md", expansion=Expansion.ALL)',
            'markdown("@prompts/dsl_reference.md", expansion=Expansion.ALL)',
            'markdown("@prompts/node_states.md", expansion=Expansion.ALL)',
            'markdown("@prompts/context_graph.md", expansion=Expansion.ALL)',
            'markdown("@prompts/work_coordination.md", expansion=Expansion.ALL)',
            'markdown("@prompts/mcp.md", expansion=Expansion.ALL)',
        ]
        assert original == PACKAGE_DEFAULT_STARTUP
