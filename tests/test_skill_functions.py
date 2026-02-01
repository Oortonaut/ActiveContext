"""Tests for skill function loading in Timeline._setup_namespace().

Tests cover:
1. Loading Python functions from skills/scripts/ directories
2. Registering with skill_ prefix
3. Skipping private modules/functions
4. Error handling for import failures
5. Respecting allowed_tools (future enhancement)
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.session.timeline import Timeline
from activecontext.skills.schema import SkillManifest


@pytest.fixture
def timeline():
    """Create a Timeline instance for testing."""
    graph = ContextGraph()
    timeline = Timeline(
        session_id="test-session",
        context_graph=graph,
        cwd=".",
    )
    return timeline


@pytest.fixture
def mock_skill_manifest(tmp_path: Path):
    """Create a mock skill manifest with scripts directory."""
    skill_path = tmp_path / "test-skill"
    skill_path.mkdir()

    # Create scripts directory
    scripts_dir = skill_path / "scripts"
    scripts_dir.mkdir()

    # Create a sample script file
    script_file = scripts_dir / "helpers.py"
    script_file.write_text(
        dedent("""
        def calculate_sum(a, b):
            '''Add two numbers.'''
            return a + b

        def calculate_product(a, b):
            '''Multiply two numbers.'''
            return a * b

        def _private_function():
            '''This should not be loaded.'''
            return "private"

        # This is a variable, not a function
        CONSTANT = 42
        """),
        encoding="utf-8",
    )

    manifest = SkillManifest(
        name="test-skill",
        description="A test skill",
        path=skill_path,
    )

    return manifest


class TestSkillFunctionLoading:
    """Tests for loading skill-provided Python functions."""

    def test_load_skill_functions_basic(
        self, timeline: Timeline, mock_skill_manifest: SkillManifest
    ):
        """Test loading functions from a skill's scripts directory."""
        from activecontext.skills.manager import SkillManager

        # Mock the SkillManager to return our test skill
        with patch.object(SkillManager, "list_active_skills", return_value=[mock_skill_manifest]):
            timeline._load_skill_functions()

        # Check that functions are loaded with skill_ prefix
        assert "skill_calculate_sum" in timeline._namespace
        assert "skill_calculate_product" in timeline._namespace

        # Test the loaded functions work
        assert timeline._namespace["skill_calculate_sum"](2, 3) == 5
        assert timeline._namespace["skill_calculate_product"](2, 3) == 6

    def test_skip_private_functions(self, timeline: Timeline, mock_skill_manifest: SkillManifest):
        """Test that private functions (starting with _) are skipped."""
        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[mock_skill_manifest]):
            timeline._load_skill_functions()

        # Private function should not be loaded
        assert "skill__private_function" not in timeline._namespace
        assert "_private_function" not in timeline._namespace

    def test_skip_non_callable_attributes(
        self, timeline: Timeline, mock_skill_manifest: SkillManifest
    ):
        """Test that non-callable attributes (variables, classes) are skipped."""
        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[mock_skill_manifest]):
            timeline._load_skill_functions()

        # Constant should not be loaded
        assert "skill_CONSTANT" not in timeline._namespace
        assert "CONSTANT" not in timeline._namespace

    def test_no_active_skills(self, timeline: Timeline):
        """Test behavior when no skills are active."""
        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[]):
            # Should not raise an error
            timeline._load_skill_functions()

        # No skill functions should be loaded
        skill_funcs = [k for k in timeline._namespace if k.startswith("skill_")]
        assert len(skill_funcs) == 0

    def test_missing_scripts_directory(self, timeline: Timeline, tmp_path: Path):
        """Test behavior when a skill has no scripts/ directory."""
        skill_path = tmp_path / "no-scripts-skill"
        skill_path.mkdir()

        manifest = SkillManifest(
            name="no-scripts-skill",
            description="A skill without scripts",
            path=skill_path,
        )

        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[manifest]):
            # Should not raise an error
            timeline._load_skill_functions()

        # No skill functions should be loaded
        skill_funcs = [k for k in timeline._namespace if k.startswith("skill_")]
        assert len(skill_funcs) == 0

    def test_import_error_handling(self, timeline: Timeline, tmp_path: Path):
        """Test graceful handling of import errors."""
        skill_path = tmp_path / "broken-skill"
        skill_path.mkdir()

        scripts_dir = skill_path / "scripts"
        scripts_dir.mkdir()

        # Create a script with a syntax error
        broken_script = scripts_dir / "broken.py"
        broken_script.write_text("def broken(\n  # Missing closing parenthesis")

        manifest = SkillManifest(
            name="broken-skill",
            description="A skill with broken scripts",
            path=skill_path,
        )

        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[manifest]):
            # Should not raise an error, just log a warning
            timeline._load_skill_functions()

        # No functions from the broken script should be loaded
        skill_funcs = [k for k in timeline._namespace if k.startswith("skill_broken")]
        assert len(skill_funcs) == 0

    def test_multiple_skills(self, timeline: Timeline, tmp_path: Path):
        """Test loading functions from multiple active skills."""
        # Create first skill
        skill1_path = tmp_path / "skill-one"
        skill1_path.mkdir()
        scripts1 = skill1_path / "scripts"
        scripts1.mkdir()
        (scripts1 / "utils.py").write_text("def helper1(): return 'one'")

        manifest1 = SkillManifest(
            name="skill-one",
            description="First skill",
            path=skill1_path,
        )

        # Create second skill
        skill2_path = tmp_path / "skill-two"
        skill2_path.mkdir()
        scripts2 = skill2_path / "scripts"
        scripts2.mkdir()
        (scripts2 / "utils.py").write_text("def helper2(): return 'two'")

        manifest2 = SkillManifest(
            name="skill-two",
            description="Second skill",
            path=skill2_path,
        )

        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[manifest1, manifest2]):
            timeline._load_skill_functions()

        # Functions from both skills should be loaded
        assert "skill_helper1" in timeline._namespace
        assert "skill_helper2" in timeline._namespace
        assert timeline._namespace["skill_helper1"]() == "one"
        assert timeline._namespace["skill_helper2"]() == "two"

    def test_skip_private_modules(self, timeline: Timeline, tmp_path: Path):
        """Test that modules starting with _ are skipped."""
        skill_path = tmp_path / "test-skill"
        skill_path.mkdir()

        scripts_dir = skill_path / "scripts"
        scripts_dir.mkdir()

        # Create a private module
        private_script = scripts_dir / "_internal.py"
        private_script.write_text("def internal_func(): return 'internal'")

        manifest = SkillManifest(
            name="test-skill",
            description="A test skill",
            path=skill_path,
        )

        from activecontext.skills.manager import SkillManager

        with patch.object(SkillManager, "list_active_skills", return_value=[manifest]):
            timeline._load_skill_functions()

        # Private module's functions should not be loaded
        assert "skill_internal_func" not in timeline._namespace
