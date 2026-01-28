"""Tests for the skills module."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from activecontext.skills import (
    SkillManifest,
    discover_skills,
    get_skills_directory,
    load_skill,
    load_skill_from_path,
)


class TestSkillManifest:
    """Test the SkillManifest dataclass."""

    def test_valid_manifest(self) -> None:
        """Test creating a valid manifest."""
        manifest = SkillManifest(
            name="my-skill",
            description="A test skill for testing purposes",
        )
        assert manifest.name == "my-skill"
        assert manifest.description == "A test skill for testing purposes"
        assert manifest.license is None
        assert manifest.allowed_tools is None
        assert manifest.metadata == {}
        assert manifest.content == ""
        assert manifest.path is None

    def test_full_manifest(self) -> None:
        """Test creating a manifest with all fields."""
        manifest = SkillManifest(
            name="full-skill",
            description="A fully specified skill",
            license="MIT",
            allowed_tools=["Bash", "Read", "Write"],
            metadata={"version": "1.0.0", "author": "Test"},
            content="# Skill Content\n\nInstructions here.",
            path=Path("/some/path"),
        )
        assert manifest.name == "full-skill"
        assert manifest.license == "MIT"
        assert manifest.allowed_tools == ["Bash", "Read", "Write"]
        assert manifest.metadata["version"] == "1.0.0"
        assert "Instructions" in manifest.content
        assert manifest.path == Path("/some/path")

    def test_empty_name_rejected(self) -> None:
        """Test that empty name is rejected."""
        with pytest.raises(ValueError, match="name is required"):
            SkillManifest(name="", description="Valid description")

    def test_empty_description_rejected(self) -> None:
        """Test that empty description is rejected."""
        with pytest.raises(ValueError, match="description is required"):
            SkillManifest(name="valid-name", description="")

    def test_name_max_length(self) -> None:
        """Test that name exceeding max length is rejected."""
        long_name = "a" * 65
        with pytest.raises(ValueError, match="exceeds 64 characters"):
            SkillManifest(name=long_name, description="Valid description")

    def test_description_max_length(self) -> None:
        """Test that description exceeding max length is rejected."""
        long_desc = "a" * 1025
        with pytest.raises(ValueError, match="exceeds 1024 characters"):
            SkillManifest(name="valid-name", description=long_desc)

    def test_description_no_angle_brackets(self) -> None:
        """Test that description with < or > is rejected."""
        with pytest.raises(ValueError, match="must not contain < or >"):
            SkillManifest(name="valid-name", description="Contains <tag>")

        with pytest.raises(ValueError, match="must not contain < or >"):
            SkillManifest(name="valid-name", description="Has > arrow")

        with pytest.raises(ValueError, match="must not contain < or >"):
            SkillManifest(name="valid-name", description="Has < arrow")

    def test_hyphen_case_validation(self) -> None:
        """Test that name must be hyphen-case."""
        # Valid names
        SkillManifest(name="simple", description="OK")
        SkillManifest(name="with-hyphen", description="OK")
        SkillManifest(name="multi-word-name", description="OK")
        SkillManifest(name="with-123-numbers", description="OK")
        SkillManifest(name="skill1", description="OK")

        # Invalid names
        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="CamelCase", description="Invalid")

        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="UPPER", description="Invalid")

        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="with_underscore", description="Invalid")

        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="-starts-with-hyphen", description="Invalid")

        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="ends-with-hyphen-", description="Invalid")

        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="double--hyphen", description="Invalid")

        with pytest.raises(ValueError, match="hyphen-case"):
            SkillManifest(name="has space", description="Invalid")


class TestGetSkillsDirectory:
    """Test the get_skills_directory function."""

    def test_uses_home_env(self) -> None:
        """Test that HOME environment variable is used."""
        with patch.dict("os.environ", {"HOME": "/mock/home", "USERPROFILE": ""}):
            path = get_skills_directory()
            assert path == Path("/mock/home/.claude/skills")

    def test_uses_userprofile_fallback(self) -> None:
        """Test that USERPROFILE is used when HOME is not set."""
        with patch.dict(
            "os.environ", {"HOME": "", "USERPROFILE": "C:\\Users\\Test"}, clear=False
        ):
            path = get_skills_directory()
            assert path == Path("C:\\Users\\Test/.claude/skills")

    def test_fallback_to_path_home(self) -> None:
        """Test fallback to Path.home() when env vars are empty."""
        with patch.dict("os.environ", {"HOME": "", "USERPROFILE": ""}, clear=False):
            path = get_skills_directory()
            assert path == Path.home() / ".claude" / "skills"


class TestLoadSkillFromPath:
    """Test the load_skill_from_path function."""

    def test_load_valid_skill(self, tmp_path: Path) -> None:
        """Test loading a valid skill from a directory."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: my-skill
description: A test skill
license: MIT
allowed-tools:
  - Bash
  - Read
metadata:
  version: "1.0.0"
---

# My Skill

Instructions for using this skill.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is not None
        assert manifest.name == "my-skill"
        assert manifest.description == "A test skill"
        assert manifest.license == "MIT"
        assert manifest.allowed_tools == ["Bash", "Read"]
        assert manifest.metadata == {"version": "1.0.0"}
        assert "# My Skill" in manifest.content
        assert manifest.path == skill_dir.resolve()

    def test_load_minimal_skill(self, tmp_path: Path) -> None:
        """Test loading a skill with only required fields."""
        skill_dir = tmp_path / "minimal"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: minimal
description: Minimal skill
---

Content here.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is not None
        assert manifest.name == "minimal"
        assert manifest.description == "Minimal skill"
        assert manifest.license is None
        assert manifest.allowed_tools is None
        assert manifest.metadata == {}
        assert manifest.content == "Content here."

    def test_missing_skill_file(self, tmp_path: Path) -> None:
        """Test loading from directory without SKILL.md."""
        skill_dir = tmp_path / "no-skill"
        skill_dir.mkdir()

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None

    def test_missing_frontmatter(self, tmp_path: Path) -> None:
        """Test loading skill without frontmatter."""
        skill_dir = tmp_path / "no-frontmatter"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("Just content, no frontmatter.", encoding="utf-8")

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None

    def test_missing_name(self, tmp_path: Path) -> None:
        """Test loading skill without name field."""
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
description: Has description but no name
---

Content.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None

    def test_missing_description(self, tmp_path: Path) -> None:
        """Test loading skill without description field."""
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: no-desc
---

Content.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Test loading skill with invalid YAML."""
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: [unclosed bracket
description: oops
---

Content.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None

    def test_invalid_name_format(self, tmp_path: Path) -> None:
        """Test loading skill with invalid name format."""
        skill_dir = tmp_path / "bad-name"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: Invalid_Name
description: Has invalid name
---

Content.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None

    def test_allowed_tools_not_list(self, tmp_path: Path) -> None:
        """Test that allowed-tools must be a list."""
        skill_dir = tmp_path / "bad-tools"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: bad-tools
description: Has non-list allowed-tools
allowed-tools: not-a-list
---

Content.
""",
            encoding="utf-8",
        )

        manifest = load_skill_from_path(skill_dir)

        assert manifest is None


class TestLoadSkill:
    """Test the load_skill function."""

    def test_load_by_name(self, tmp_path: Path) -> None:
        """Test loading a skill by name."""
        skills_dir = tmp_path / ".claude" / "skills"
        skill_dir = skills_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: test-skill
description: Test skill loaded by name
---

Instructions.
""",
            encoding="utf-8",
        )

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifest = load_skill("test-skill")

        assert manifest is not None
        assert manifest.name == "test-skill"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading a skill that doesn't exist."""
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifest = load_skill("nonexistent")

        assert manifest is None


class TestDiscoverSkills:
    """Test the discover_skills function."""

    def test_discover_multiple_skills(self, tmp_path: Path) -> None:
        """Test discovering multiple skills."""
        skills_dir = tmp_path / ".claude" / "skills"

        # Create skill-a
        skill_a = skills_dir / "skill-a"
        skill_a.mkdir(parents=True)
        (skill_a / "SKILL.md").write_text(
            """\
---
name: skill-a
description: First skill
---

A content.
""",
            encoding="utf-8",
        )

        # Create skill-b
        skill_b = skills_dir / "skill-b"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text(
            """\
---
name: skill-b
description: Second skill
---

B content.
""",
            encoding="utf-8",
        )

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifests = discover_skills()

        assert len(manifests) == 2
        assert manifests[0].name == "skill-a"
        assert manifests[1].name == "skill-b"

    def test_discover_skips_malformed(self, tmp_path: Path) -> None:
        """Test that discover_skills skips malformed skills."""
        skills_dir = tmp_path / ".claude" / "skills"

        # Create valid skill
        valid = skills_dir / "valid"
        valid.mkdir(parents=True)
        (valid / "SKILL.md").write_text(
            """\
---
name: valid
description: Valid skill
---
""",
            encoding="utf-8",
        )

        # Create malformed skill
        malformed = skills_dir / "malformed"
        malformed.mkdir()
        (malformed / "SKILL.md").write_text("No frontmatter here", encoding="utf-8")

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifests = discover_skills()

        assert len(manifests) == 1
        assert manifests[0].name == "valid"

    def test_discover_skips_files(self, tmp_path: Path) -> None:
        """Test that discover_skills only looks at directories."""
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        # Create a file (not directory)
        (skills_dir / "readme.md").write_text("Not a skill", encoding="utf-8")

        # Create valid skill
        skill = skills_dir / "real-skill"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            """\
---
name: real-skill
description: The only real skill
---
""",
            encoding="utf-8",
        )

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifests = discover_skills()

        assert len(manifests) == 1
        assert manifests[0].name == "real-skill"

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        """Test discovering skills from empty directory."""
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifests = discover_skills()

        assert manifests == []

    def test_discover_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test discovering skills from nonexistent directory."""
        skills_dir = tmp_path / ".claude" / "skills"  # Don't create it

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifests = discover_skills()

        assert manifests == []

    def test_discover_sorted_by_name(self, tmp_path: Path) -> None:
        """Test that discovered skills are sorted by name."""
        skills_dir = tmp_path / ".claude" / "skills"

        # Create skills in reverse order
        for name in ["zebra", "alpha", "middle"]:
            skill = skills_dir / name
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text(
                f"""\
---
name: {name}
description: Skill {name}
---
""",
                encoding="utf-8",
            )

        with patch(
            "activecontext.skills.loader.get_skills_directory", return_value=skills_dir
        ):
            manifests = discover_skills()

        names = [m.name for m in manifests]
        assert names == ["alpha", "middle", "zebra"]
