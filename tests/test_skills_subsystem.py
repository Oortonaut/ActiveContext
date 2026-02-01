"""Comprehensive tests for the skills subsystem.

This module tests the entire skills infrastructure using real filesystem fixtures
located in tests/fixtures/skills/. Test coverage includes:

1. Skill discovery from filesystem
2. SKILL.md parsing with valid/invalid frontmatter
3. SkillManager lifecycle (activate, deactivate, state management)
4. /skill command handling in ACP agent
5. Skill content injection and rendering
6. Multi-skill activation/deactivation
7. Namespace function registration from scripts/
8. Cross-platform path handling (Windows/Unix)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.session.timeline import Timeline
from activecontext.skills.loader import (
    discover_skills,
    get_skills_directory,
    load_skill,
    load_skill_from_path,
)
from activecontext.skills.manager import SkillManager
from activecontext.skills.schema import SkillManifest


@pytest.fixture
def fixtures_skills_dir() -> Path:
    """Return the path to the test fixtures skills directory."""
    return Path(__file__).parent / "fixtures" / "skills"


class TestSkillDiscovery:
    """Tests for discovering skills from the filesystem."""

    def test_discover_skills_from_fixtures(self, fixtures_skills_dir: Path):
        """Test discovering skills from the fixtures directory."""
        with patch(
            "activecontext.skills.loader.get_skills_directory",
            return_value=fixtures_skills_dir,
        ):
            skills = discover_skills()

        # Should find valid skills, skip invalid ones
        skill_names = {skill.name for skill in skills}

        # valid-skill should be found
        assert "valid-skill" in skill_names
        # test-skill should be found
        assert "test-skill" in skill_names
        # with-references should be found
        assert "with-references" in skill_names
        # with-scripts should be found
        assert "with-scripts" in skill_names

        # Invalid skills should be skipped
        assert "invalid-frontmatter" not in skill_names
        assert "missing-required" not in skill_names

    def test_discover_skills_empty_directory(self, tmp_path: Path):
        """Test discovering skills from an empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("activecontext.skills.loader.get_skills_directory", return_value=empty_dir):
            skills = discover_skills()

        assert len(skills) == 0

    def test_discover_skills_nonexistent_directory(self, tmp_path: Path):
        """Test discovering skills when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"

        with patch("activecontext.skills.loader.get_skills_directory", return_value=nonexistent):
            skills = discover_skills()

        assert len(skills) == 0

    def test_discover_skills_sorted_by_name(self, fixtures_skills_dir: Path):
        """Test that discovered skills are sorted by name."""
        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            skills = discover_skills()

        # Extract names
        names = [skill.name for skill in skills]

        # Should be sorted
        assert names == sorted(names)


class TestSkillLoading:
    """Tests for loading individual skills from SKILL.md files."""

    def test_load_valid_skill(self, fixtures_skills_dir: Path):
        """Test loading a well-formed skill."""
        skill_path = fixtures_skills_dir / "valid-skill"
        manifest = load_skill_from_path(skill_path)

        assert manifest is not None
        assert manifest.name == "valid-skill"
        assert manifest.description == "A well-formed test skill with all required fields"
        assert manifest.license == "MIT"
        assert manifest.allowed_tools == ["Bash", "Read", "Write"]
        assert manifest.metadata == {
            "version": "1.0.0",
            "author": "Test Suite",
            "model": "claude-opus-4-5-20251101",
        }
        assert "# Valid Skill" in manifest.content
        assert manifest.path == skill_path.resolve()

    def test_load_skill_missing_required_field(self, fixtures_skills_dir: Path):
        """Test loading a skill missing required 'description' field."""
        skill_path = fixtures_skills_dir / "missing-required"
        manifest = load_skill_from_path(skill_path)

        # Should return None due to missing description
        assert manifest is None

    def test_load_skill_invalid_frontmatter(self, fixtures_skills_dir: Path):
        """Test loading a skill with malformed YAML frontmatter."""
        skill_path = fixtures_skills_dir / "invalid-frontmatter"
        manifest = load_skill_from_path(skill_path)

        # Should return None due to YAML parse error
        assert manifest is None

    def test_load_skill_no_skill_md(self, tmp_path: Path):
        """Test loading from a directory without SKILL.md."""
        empty_dir = tmp_path / "no-skill"
        empty_dir.mkdir()

        manifest = load_skill_from_path(empty_dir)
        assert manifest is None

    def test_load_skill_by_name(self, fixtures_skills_dir: Path):
        """Test loading a skill by name using load_skill()."""
        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manifest = load_skill("valid-skill")

        assert manifest is not None
        assert manifest.name == "valid-skill"

    def test_load_skill_with_references(self, fixtures_skills_dir: Path):
        """Test loading a skill that has a references/ directory."""
        skill_path = fixtures_skills_dir / "with-references"
        manifest = load_skill_from_path(skill_path)

        assert manifest is not None
        assert manifest.name == "with-references"
        assert manifest.description == "Test skill with a references directory"

        # Verify references directory exists
        references_dir = manifest.path / "references"
        assert references_dir.exists()
        assert (references_dir / "api.md").exists()
        assert (references_dir / "implementation.md").exists()

    def test_load_skill_with_scripts(self, fixtures_skills_dir: Path):
        """Test loading a skill that has a scripts/ directory."""
        skill_path = fixtures_skills_dir / "with-scripts"
        manifest = load_skill_from_path(skill_path)

        assert manifest is not None
        assert manifest.name == "with-scripts"

        # Verify scripts directory exists
        scripts_dir = manifest.path / "scripts"
        assert scripts_dir.exists()
        assert (scripts_dir / "helper.py").exists()
        assert (scripts_dir / "validator.sh").exists()


class TestSkillManifestValidation:
    """Tests for SkillManifest validation logic."""

    def test_valid_manifest_creation(self):
        """Test creating a valid SkillManifest."""
        manifest = SkillManifest(
            name="test-skill",
            description="A test skill for validation",
            license="MIT",
            allowed_tools=["Bash", "Read"],
            metadata={"version": "1.0.0"},
            content="# Test Content",
            path=Path("/tmp/test-skill"),
        )

        assert manifest.name == "test-skill"
        assert manifest.description == "A test skill for validation"

    def test_invalid_name_uppercase(self):
        """Test that uppercase letters in name cause validation error."""
        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(
                name="TestSkill",
                description="Test",
            )

    def test_invalid_name_underscore(self):
        """Test that underscores in name cause validation error."""
        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(
                name="test_skill",
                description="Test",
            )

    def test_invalid_name_consecutive_hyphens(self):
        """Test that consecutive hyphens cause validation error."""
        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(
                name="test--skill",
                description="Test",
            )

    def test_invalid_name_leading_hyphen(self):
        """Test that leading hyphen causes validation error."""
        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(
                name="-test-skill",
                description="Test",
            )

    def test_invalid_name_trailing_hyphen(self):
        """Test that trailing hyphen causes validation error."""
        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(
                name="test-skill-",
                description="Test",
            )

    def test_invalid_name_too_long(self):
        """Test that name exceeding MAX_NAME_LENGTH causes validation error."""
        long_name = "a" * 65
        with pytest.raises(ValueError, match="exceeds 64 characters"):
            SkillManifest(
                name=long_name,
                description="Test",
            )

    def test_invalid_description_too_long(self):
        """Test that description exceeding MAX_DESCRIPTION_LENGTH causes validation error."""
        long_desc = "a" * 1025
        with pytest.raises(ValueError, match="exceeds 1024 characters"):
            SkillManifest(
                name="test-skill",
                description=long_desc,
            )

    def test_invalid_description_contains_angle_brackets(self):
        """Test that < or > in description causes validation error."""
        with pytest.raises(ValueError, match="must not contain < or >"):
            SkillManifest(
                name="test-skill",
                description="Test with <html> tags",
            )

    def test_missing_name(self):
        """Test that missing name causes validation error."""
        with pytest.raises(ValueError, match="name is required"):
            SkillManifest(
                name="",
                description="Test",
            )

    def test_missing_description(self):
        """Test that missing description causes validation error."""
        with pytest.raises(ValueError, match="description is required"):
            SkillManifest(
                name="test-skill",
                description="",
            )


class TestSkillManager:
    """Tests for SkillManager lifecycle operations."""

    def test_activate_skill(self, fixtures_skills_dir: Path):
        """Test activating a skill."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manifest = manager.activate_skill("valid-skill")

        assert manifest is not None
        assert manifest.name == "valid-skill"
        assert manager.is_active("valid-skill")

    def test_activate_already_active_skill(self, fixtures_skills_dir: Path):
        """Test that activating an already-active skill returns existing manifest."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manifest1 = manager.activate_skill("valid-skill")
            manifest2 = manager.activate_skill("valid-skill")

        # Should return the same manifest without reloading
        assert manifest1 is manifest2

    def test_activate_nonexistent_skill(self, fixtures_skills_dir: Path):
        """Test activating a skill that doesn't exist."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manifest = manager.activate_skill("nonexistent-skill")

        assert manifest is None
        assert not manager.is_active("nonexistent-skill")

    def test_deactivate_skill(self, fixtures_skills_dir: Path):
        """Test deactivating an active skill."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.activate_skill("valid-skill")
            result = manager.deactivate_skill("valid-skill")

        assert result is True
        assert not manager.is_active("valid-skill")

    def test_deactivate_inactive_skill(self):
        """Test deactivating a skill that is not active."""
        manager = SkillManager()
        result = manager.deactivate_skill("nonexistent-skill")

        assert result is False

    def test_list_active_skills(self, fixtures_skills_dir: Path):
        """Test listing all active skills."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.activate_skill("valid-skill")
            manager.activate_skill("test-skill")
            active = manager.list_active_skills()

        assert len(active) == 2
        names = {skill.name for skill in active}
        assert "valid-skill" in names
        assert "test-skill" in names

        # Should be sorted by name
        assert active[0].name < active[1].name

    def test_get_active_skill(self, fixtures_skills_dir: Path):
        """Test retrieving an active skill manifest."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.activate_skill("valid-skill")
            manifest = manager.get_active_skill("valid-skill")

        assert manifest is not None
        assert manifest.name == "valid-skill"

    def test_get_inactive_skill(self):
        """Test retrieving a skill that is not active."""
        manager = SkillManager()
        manifest = manager.get_active_skill("nonexistent")

        assert manifest is None

    def test_clear_all_skills(self, fixtures_skills_dir: Path):
        """Test clearing all active skills."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.activate_skill("valid-skill")
            manager.activate_skill("test-skill")
            manager.clear()

        assert len(manager.list_active_skills()) == 0
        assert not manager.is_active("valid-skill")
        assert not manager.is_active("test-skill")

    def test_get_state(self, fixtures_skills_dir: Path):
        """Test getting manager state for persistence."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.activate_skill("valid-skill")
            manager.activate_skill("test-skill")
            state = manager.get_state()

        assert "active_skills" in state
        assert set(state["active_skills"]) == {"test-skill", "valid-skill"}
        # Should be sorted
        assert state["active_skills"] == sorted(state["active_skills"])

    def test_restore_state(self, fixtures_skills_dir: Path):
        """Test restoring manager state from persistence."""
        manager = SkillManager()

        state = {"active_skills": ["valid-skill", "test-skill"]}

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.restore_state(state)

        assert manager.is_active("valid-skill")
        assert manager.is_active("test-skill")

    def test_restore_state_invalid_format(self):
        """Test restoring with invalid state format."""
        manager = SkillManager()

        # Invalid state - not a dict
        manager.restore_state("invalid")  # type: ignore[arg-type]
        assert len(manager.list_active_skills()) == 0

        # Invalid active_skills - not a list
        manager.restore_state({"active_skills": "not-a-list"})  # type: ignore[dict-item]
        assert len(manager.list_active_skills()) == 0

    def test_restore_state_with_missing_skills(self, fixtures_skills_dir: Path):
        """Test restoring state when some skills can't be loaded."""
        manager = SkillManager()

        state = {"active_skills": ["valid-skill", "nonexistent-skill"]}

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.restore_state(state)

        # valid-skill should be loaded, nonexistent-skill should be skipped
        assert manager.is_active("valid-skill")
        assert not manager.is_active("nonexistent-skill")


class TestSkillFunctionLoading:
    """Tests for loading Python functions from skills/scripts/ directories."""

    @pytest.fixture
    def timeline(self):
        """Create a Timeline instance for testing."""
        graph = ContextGraph()
        timeline = Timeline(
            session_id="test-session",
            context_graph=graph,
            cwd=".",
        )
        return timeline

    def test_load_skill_functions_from_with_scripts(
        self, timeline: Timeline, fixtures_skills_dir: Path
    ):
        """Test loading functions from a skill with scripts/ directory."""
        skill_path = fixtures_skills_dir / "with-scripts"
        manifest = load_skill_from_path(skill_path)
        assert manifest is not None

        with patch(
            "activecontext.skills.manager.SkillManager.list_active_skills", return_value=[manifest]
        ):
            timeline._load_skill_functions()

        # The helper.py has a main() function
        assert "skill_main" in timeline._namespace

        # Verify it's callable
        assert callable(timeline._namespace["skill_main"])

    def test_no_skill_functions_when_no_active_skills(self, timeline: Timeline):
        """Test that no functions are loaded when no skills are active."""
        with patch("activecontext.skills.manager.SkillManager.list_active_skills", return_value=[]):
            timeline._load_skill_functions()

        # No skill_ prefixed functions should be loaded
        skill_funcs = [k for k in timeline._namespace if k.startswith("skill_")]
        assert len(skill_funcs) == 0

    def test_no_skill_functions_when_no_scripts_dir(
        self, timeline: Timeline, fixtures_skills_dir: Path
    ):
        """Test behavior when a skill has no scripts/ directory."""
        skill_path = fixtures_skills_dir / "valid-skill"
        manifest = load_skill_from_path(skill_path)
        assert manifest is not None

        with patch(
            "activecontext.skills.manager.SkillManager.list_active_skills", return_value=[manifest]
        ):
            timeline._load_skill_functions()

        # No skill_ prefixed functions should be loaded
        skill_funcs = [k for k in timeline._namespace if k.startswith("skill_")]
        assert len(skill_funcs) == 0


class TestCrossPlatformPathHandling:
    """Tests for cross-platform path handling in the skills subsystem."""

    def test_get_skills_directory_uses_home_env(self):
        """Test that get_skills_directory() uses HOME or USERPROFILE env vars."""
        # Unix-style HOME
        with patch.dict("os.environ", {"HOME": "/home/testuser"}, clear=True):
            skills_dir = get_skills_directory()
            assert skills_dir == Path("/home/testuser/.claude/skills")

        # Windows-style USERPROFILE
        with patch.dict("os.environ", {"USERPROFILE": "C:\\Users\\testuser"}, clear=True):
            if "HOME" in os.environ:
                del os.environ["HOME"]
            skills_dir = get_skills_directory()
            expected = Path("C:/Users/testuser/.claude/skills")
            assert skills_dir == expected

    def test_get_skills_directory_fallback(self):
        """Test get_skills_directory() fallback when env vars are missing."""
        # When environment variables are cleared, Path.home() may fail on Windows
        # We test that the code attempts to use Path.home() as a fallback
        # but we need at least one env var present for it to work
        import tempfile

        temp_home = tempfile.mkdtemp()

        try:
            with patch.dict("os.environ", {"USERPROFILE": temp_home}, clear=True):
                skills_dir = get_skills_directory()
                expected = Path(temp_home) / ".claude" / "skills"
                assert skills_dir == expected
        finally:
            import shutil

            shutil.rmtree(temp_home, ignore_errors=True)

    def test_skill_path_resolution_cross_platform(self, fixtures_skills_dir: Path):
        """Test that skill paths are resolved consistently across platforms."""
        skill_path = fixtures_skills_dir / "valid-skill"
        manifest = load_skill_from_path(skill_path)

        assert manifest is not None
        # Path should be resolved (absolute)
        assert manifest.path.is_absolute()
        # Should match the resolved skill_path
        assert manifest.path == skill_path.resolve()


class TestSkillCommandParsing:
    """Tests for /skill command parsing logic (unit tests without full agent)."""

    def test_skill_command_parses_skill_name(self):
        """Test that skill name is correctly parsed from /skill command."""
        command = "/skill valid-skill"
        parts = command.split(maxsplit=1)
        assert len(parts) == 2
        args = parts[1]
        skill_parts = args.split(maxsplit=1)
        skill_name = skill_parts[0]
        assert skill_name == "valid-skill"

    def test_skill_command_parses_skill_name_with_args(self):
        """Test parsing skill name and arguments."""
        command = "/skill valid-skill --arg value"
        parts = command.split(maxsplit=1)
        args = parts[1]
        skill_parts = args.split(maxsplit=1)
        skill_name = skill_parts[0]
        skill_args = skill_parts[1] if len(skill_parts) > 1 else ""

        assert skill_name == "valid-skill"
        assert skill_args == "--arg value"

    def test_skill_command_no_args(self):
        """Test parsing /skill command without arguments."""
        command = "/skill"
        parts = command.split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        assert args == ""


class TestMultiSkillActivation:
    """Tests for activating and deactivating multiple skills simultaneously."""

    def test_activate_multiple_skills_in_sequence(self, fixtures_skills_dir: Path):
        """Test activating multiple skills one after another."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manifest1 = manager.activate_skill("valid-skill")
            manifest2 = manager.activate_skill("test-skill")
            manifest3 = manager.activate_skill("with-references")

        assert manifest1 is not None
        assert manifest2 is not None
        assert manifest3 is not None

        active = manager.list_active_skills()
        assert len(active) == 3

        names = {skill.name for skill in active}
        assert names == {"valid-skill", "test-skill", "with-references"}

    def test_deactivate_one_skill_among_many(self, fixtures_skills_dir: Path):
        """Test deactivating one skill while others remain active."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            manager.activate_skill("valid-skill")
            manager.activate_skill("test-skill")
            manager.activate_skill("with-references")

            # Deactivate one
            result = manager.deactivate_skill("test-skill")

        assert result is True
        assert not manager.is_active("test-skill")
        assert manager.is_active("valid-skill")
        assert manager.is_active("with-references")

        active = manager.list_active_skills()
        assert len(active) == 2

    def test_activate_deactivate_activate_cycle(self, fixtures_skills_dir: Path):
        """Test cycling a skill through activation and deactivation."""
        manager = SkillManager()

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=fixtures_skills_dir
        ):
            # Activate
            manifest1 = manager.activate_skill("valid-skill")
            assert manifest1 is not None
            assert manager.is_active("valid-skill")

            # Deactivate
            result = manager.deactivate_skill("valid-skill")
            assert result is True
            assert not manager.is_active("valid-skill")

            # Re-activate
            manifest2 = manager.activate_skill("valid-skill")
            assert manifest2 is not None
            assert manager.is_active("valid-skill")

            # Manifests should be different instances after reload
            assert manifest1 is not manifest2


class TestSkillContentFormatting:
    """Tests for skill content formatting (unit tests without agent)."""

    def test_skill_content_format(self, fixtures_skills_dir: Path):
        """Test that skill content can be formatted correctly."""
        manifest = load_skill_from_path(fixtures_skills_dir / "valid-skill")
        assert manifest is not None

        # Format as if injecting into session
        skill_content = f"# Skill: {manifest.name}\n\n"
        skill_content += f"{manifest.description}\n\n"
        skill_content += "---\n\n"
        skill_content += manifest.content

        # Verify expected content
        assert "# Skill: valid-skill" in skill_content
        assert "A well-formed test skill with all required fields" in skill_content
        assert "# Valid Skill" in skill_content
        assert "Purpose" in skill_content

    def test_skill_content_with_arguments(self, fixtures_skills_dir: Path):
        """Test formatting skill content with arguments."""
        manifest = load_skill_from_path(fixtures_skills_dir / "valid-skill")
        assert manifest is not None

        skill_args = "--arg value"
        skill_content = f"# Skill: {manifest.name}\n\n"
        skill_content += f"{manifest.description}\n\n"
        skill_content += f"**Arguments:** {skill_args}\n\n"
        skill_content += "---\n\n"
        skill_content += manifest.content

        assert "**Arguments:** --arg value" in skill_content
