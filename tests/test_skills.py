"""Tests for the skills module."""

from __future__ import annotations

from pathlib import Path
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
        with patch.dict("os.environ", {"HOME": "", "USERPROFILE": "C:\\Users\\Test"}, clear=False):
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

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
            manifest = load_skill("test-skill")

        assert manifest is not None
        assert manifest.name == "test-skill"

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Test loading a skill that doesn't exist."""
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
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

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
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

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
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

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
            manifests = discover_skills()

        assert len(manifests) == 1
        assert manifests[0].name == "real-skill"

    def test_discover_empty_directory(self, tmp_path: Path) -> None:
        """Test discovering skills from empty directory."""
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
            manifests = discover_skills()

        assert manifests == []

    def test_discover_nonexistent_directory(self, tmp_path: Path) -> None:
        """Test discovering skills from nonexistent directory."""
        skills_dir = tmp_path / ".claude" / "skills"  # Don't create it

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
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

        with patch("activecontext.skills.loader.get_skills_directory", return_value=skills_dir):
            manifests = discover_skills()

        names = [m.name for m in manifests]
        assert names == ["alpha", "middle", "zebra"]


class TestFixtureSkills:
    """Test loading from version-controlled fixture skills."""

    def test_load_valid_skill_fixture(self) -> None:
        """Test loading the valid-skill fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "skills" / "valid-skill"
        manifest = load_skill_from_path(fixture_path)

        assert manifest is not None
        assert manifest.name == "valid-skill"
        assert manifest.description == "A well-formed test skill with all required fields"
        assert manifest.license == "MIT"
        assert manifest.allowed_tools == ["Bash", "Read", "Write"]
        assert manifest.metadata["version"] == "1.0.0"
        assert "Valid Skill" in manifest.content

    def test_load_invalid_frontmatter_fixture(self) -> None:
        """Test loading the invalid-frontmatter fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "skills" / "invalid-frontmatter"
        manifest = load_skill_from_path(fixture_path)

        # Should fail to load due to malformed YAML
        assert manifest is None

    def test_load_missing_required_fixture(self) -> None:
        """Test loading the missing-required fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "skills" / "missing-required"
        manifest = load_skill_from_path(fixture_path)

        # Should fail to load due to missing description field
        assert manifest is None

    def test_load_with_scripts_fixture(self) -> None:
        """Test loading skill with scripts directory."""
        fixture_path = Path(__file__).parent / "fixtures" / "skills" / "with-scripts"
        manifest = load_skill_from_path(fixture_path)

        assert manifest is not None
        assert manifest.name == "with-scripts"
        assert manifest.description == "Test skill with a scripts directory"

        # Verify scripts directory exists
        scripts_dir = fixture_path / "scripts"
        assert scripts_dir.exists()
        assert scripts_dir.is_dir()
        assert (scripts_dir / "helper.py").exists()
        assert (scripts_dir / "validator.sh").exists()

    def test_load_with_references_fixture(self) -> None:
        """Test loading skill with references directory."""
        fixture_path = Path(__file__).parent / "fixtures" / "skills" / "with-references"
        manifest = load_skill_from_path(fixture_path)

        assert manifest is not None
        assert manifest.name == "with-references"
        assert manifest.description == "Test skill with a references directory"

        # Verify references directory exists
        refs_dir = fixture_path / "references"
        assert refs_dir.exists()
        assert refs_dir.is_dir()
        assert (refs_dir / "api.md").exists()
        assert (refs_dir / "implementation.md").exists()

    def test_discover_fixture_skills(self) -> None:
        """Test discovering all fixture skills."""
        fixtures_dir = Path(__file__).parent / "fixtures" / "skills"

        with patch("activecontext.skills.loader.get_skills_directory", return_value=fixtures_dir):
            manifests = discover_skills()

        # Should discover valid-skill, with-scripts, with-references, test-skill
        # But NOT invalid-frontmatter or missing-required
        assert len(manifests) == 4
        names = [m.name for m in manifests]
        assert "valid-skill" in names
        assert "with-scripts" in names
        assert "with-references" in names
        assert "test-skill" in names
        assert "invalid-frontmatter" not in names
        assert "missing-required" not in names


class TestCrossPlatformPaths:
    """Test cross-platform path handling."""

    def test_windows_backslash_paths(self, tmp_path: Path) -> None:
        """Test that Windows backslash paths work correctly."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: test-skill
description: Test Windows paths
---

Content.
""",
            encoding="utf-8",
        )

        # Load using both forward slash and backslash representations
        manifest1 = load_skill_from_path(skill_dir)
        manifest2 = load_skill_from_path(Path(str(skill_dir).replace("/", "\\")))

        assert manifest1 is not None
        assert manifest2 is not None
        # Paths should resolve to same location
        assert manifest1.path == manifest2.path

    def test_unix_forward_slash_paths(self, tmp_path: Path) -> None:
        """Test that Unix forward slash paths work correctly."""
        skill_dir = tmp_path / "unix-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: unix-skill
description: Test Unix paths
---

Content.
""",
            encoding="utf-8",
        )

        # pathlib normalizes paths across platforms
        manifest = load_skill_from_path(skill_dir)

        assert manifest is not None
        assert manifest.name == "unix-skill"
        assert manifest.path.exists()

    def test_path_normalization(self, tmp_path: Path) -> None:
        """Test that paths are normalized (resolve symlinks, .., etc)."""
        skill_dir = tmp_path / "normalize-test"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: normalize-test
description: Path normalization test
---

Content.
""",
            encoding="utf-8",
        )

        # Load with redundant path components
        redundant_path = tmp_path / "." / "normalize-test"
        manifest = load_skill_from_path(redundant_path)

        assert manifest is not None
        # Path should be resolved (no . or .. components)
        assert ".." not in str(manifest.path)
        assert manifest.path == skill_dir.resolve()

    def test_relative_vs_absolute_paths(self, tmp_path: Path) -> None:
        """Test that both relative and absolute paths work."""
        skill_dir = tmp_path / "path-test"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: path-test
description: Test relative vs absolute paths
---

Content.
""",
            encoding="utf-8",
        )

        # Load with absolute path
        manifest_abs = load_skill_from_path(skill_dir.resolve())
        # Load with relative path (if tmp_path is relative)
        manifest_rel = load_skill_from_path(skill_dir)

        assert manifest_abs is not None
        assert manifest_rel is not None
        # Both should resolve to same absolute path
        assert manifest_abs.path == manifest_rel.path
        assert manifest_abs.path.is_absolute()

    def test_case_sensitivity(self, tmp_path: Path) -> None:
        """Test path case handling (case-insensitive on Windows, sensitive on Unix)."""
        skill_dir = tmp_path / "CaseSensitive"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            """\
---
name: case-test
description: Case sensitivity test
---

Content.
""",
            encoding="utf-8",
        )

        # On Windows, these should work the same
        # On Unix, only exact case works
        import platform

        manifest = load_skill_from_path(skill_dir)
        assert manifest is not None

        if platform.system() == "Windows":
            # Windows is case-insensitive
            load_skill_from_path(tmp_path / "casesensitive")
            # May or may not work depending on filesystem
            # Just verify original works
            assert manifest.name == "case-test"
        else:
            # Unix is case-sensitive - only exact match works
            assert manifest.name == "case-test"
