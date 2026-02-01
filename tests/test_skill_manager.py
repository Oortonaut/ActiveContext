"""Tests for SkillManager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from activecontext.skills.manager import SkillManager
from activecontext.skills.schema import SkillManifest


@pytest.fixture
def skill_manager():
    """Create a fresh SkillManager instance."""
    return SkillManager()


@pytest.fixture
def mock_skill_manifest():
    """Create a mock SkillManifest."""
    return SkillManifest(
        name="test-skill",
        description="A test skill for testing",
        license="MIT",
        content="# Test Skill\n\nThis is test content.",
        path=Path("/fake/path/test-skill"),
    )


@pytest.fixture
def another_mock_skill_manifest():
    """Create another mock SkillManifest."""
    return SkillManifest(
        name="another-skill",
        description="Another test skill",
        content="# Another Skill\n\nMore test content.",
        path=Path("/fake/path/another-skill"),
    )


class TestSkillManagerActivation:
    """Tests for skill activation."""

    def test_activate_skill_success(self, skill_manager, mock_skill_manifest):
        """Test successfully activating a skill."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            result = skill_manager.activate_skill("test-skill")

            assert result is mock_skill_manifest
            assert skill_manager.is_active("test-skill")
            mock_load.assert_called_once_with("test-skill")

    def test_activate_skill_not_found(self, skill_manager):
        """Test activating a skill that doesn't exist."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = None

            result = skill_manager.activate_skill("nonexistent")

            assert result is None
            assert not skill_manager.is_active("nonexistent")

    def test_activate_skill_already_active(self, skill_manager, mock_skill_manifest):
        """Test activating a skill that's already active returns cached manifest."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            # First activation
            result1 = skill_manager.activate_skill("test-skill")
            assert result1 is mock_skill_manifest

            # Second activation should not reload
            result2 = skill_manager.activate_skill("test-skill")
            assert result2 is mock_skill_manifest
            assert result2 is result1

            # load_skill should only be called once
            mock_load.assert_called_once_with("test-skill")

    def test_activate_multiple_skills(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test activating multiple different skills."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            result1 = skill_manager.activate_skill("test-skill")
            result2 = skill_manager.activate_skill("another-skill")

            assert result1 is mock_skill_manifest
            assert result2 is another_mock_skill_manifest
            assert skill_manager.is_active("test-skill")
            assert skill_manager.is_active("another-skill")


class TestSkillManagerDeactivation:
    """Tests for skill deactivation."""

    def test_deactivate_active_skill(self, skill_manager, mock_skill_manifest):
        """Test deactivating an active skill."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            skill_manager.activate_skill("test-skill")
            assert skill_manager.is_active("test-skill")

            result = skill_manager.deactivate_skill("test-skill")

            assert result is True
            assert not skill_manager.is_active("test-skill")

    def test_deactivate_inactive_skill(self, skill_manager):
        """Test deactivating a skill that's not active."""
        result = skill_manager.deactivate_skill("test-skill")

        assert result is False
        assert not skill_manager.is_active("test-skill")

    def test_deactivate_one_of_many(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test deactivating one skill while others remain active."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            skill_manager.activate_skill("test-skill")
            skill_manager.activate_skill("another-skill")

            result = skill_manager.deactivate_skill("test-skill")

            assert result is True
            assert not skill_manager.is_active("test-skill")
            assert skill_manager.is_active("another-skill")


class TestSkillManagerListing:
    """Tests for listing active skills."""

    def test_list_active_skills_empty(self, skill_manager):
        """Test listing skills when none are active."""
        result = skill_manager.list_active_skills()

        assert result == []

    def test_list_active_skills_single(self, skill_manager, mock_skill_manifest):
        """Test listing skills with one active skill."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            skill_manager.activate_skill("test-skill")
            result = skill_manager.list_active_skills()

            assert len(result) == 1
            assert result[0] is mock_skill_manifest

    def test_list_active_skills_multiple_sorted(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test listing skills returns sorted list."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            # Activate in reverse alphabetical order
            skill_manager.activate_skill("test-skill")
            skill_manager.activate_skill("another-skill")

            result = skill_manager.list_active_skills()

            # Should be sorted alphabetically
            assert len(result) == 2
            assert result[0].name == "another-skill"
            assert result[1].name == "test-skill"

    def test_is_active_returns_true_for_active(self, skill_manager, mock_skill_manifest):
        """Test is_active returns True for active skills."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            skill_manager.activate_skill("test-skill")

            assert skill_manager.is_active("test-skill") is True

    def test_is_active_returns_false_for_inactive(self, skill_manager):
        """Test is_active returns False for inactive skills."""
        assert skill_manager.is_active("test-skill") is False

    def test_get_active_skill_returns_manifest(self, skill_manager, mock_skill_manifest):
        """Test get_active_skill returns the manifest for active skills."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            skill_manager.activate_skill("test-skill")
            result = skill_manager.get_active_skill("test-skill")

            assert result is mock_skill_manifest

    def test_get_active_skill_returns_none_for_inactive(self, skill_manager):
        """Test get_active_skill returns None for inactive skills."""
        result = skill_manager.get_active_skill("test-skill")

        assert result is None


class TestSkillManagerClear:
    """Tests for clearing all active skills."""

    def test_clear_empty(self, skill_manager):
        """Test clearing when no skills are active."""
        skill_manager.clear()

        assert skill_manager.list_active_skills() == []

    def test_clear_with_active_skills(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test clearing removes all active skills."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            skill_manager.activate_skill("test-skill")
            skill_manager.activate_skill("another-skill")

            skill_manager.clear()

            assert skill_manager.list_active_skills() == []
            assert not skill_manager.is_active("test-skill")
            assert not skill_manager.is_active("another-skill")


class TestSkillManagerStatePersistence:
    """Tests for state persistence functionality."""

    def test_get_state_empty(self, skill_manager):
        """Test get_state when no skills are active."""
        state = skill_manager.get_state()

        assert state == {"active_skills": []}

    def test_get_state_with_active_skills(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test get_state returns sorted list of active skill names."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            # Activate in reverse order
            skill_manager.activate_skill("test-skill")
            skill_manager.activate_skill("another-skill")

            state = skill_manager.get_state()

            # Should be sorted
            assert state == {"active_skills": ["another-skill", "test-skill"]}

    def test_restore_state_empty(self, skill_manager):
        """Test restoring state with no active skills."""
        state = {"active_skills": []}

        skill_manager.restore_state(state)

        assert skill_manager.list_active_skills() == []

    def test_restore_state_with_skills(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test restoring state activates skills."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            state = {"active_skills": ["test-skill", "another-skill"]}

            skill_manager.restore_state(state)

            assert skill_manager.is_active("test-skill")
            assert skill_manager.is_active("another-skill")
            assert len(skill_manager.list_active_skills()) == 2

    def test_restore_state_with_missing_skill(self, skill_manager, mock_skill_manifest):
        """Test restoring state when some skills can't be loaded."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                return None  # "missing-skill" fails to load

            mock_load.side_effect = side_effect

            state = {"active_skills": ["test-skill", "missing-skill"]}

            skill_manager.restore_state(state)

            # Only test-skill should be active
            assert skill_manager.is_active("test-skill")
            assert not skill_manager.is_active("missing-skill")
            assert len(skill_manager.list_active_skills()) == 1

    def test_restore_state_invalid_dict(self, skill_manager):
        """Test restoring state with invalid type."""
        skill_manager.restore_state("not a dict")  # type: ignore

        assert skill_manager.list_active_skills() == []

    def test_restore_state_invalid_active_skills(self, skill_manager):
        """Test restoring state with invalid active_skills value."""
        skill_manager.restore_state({"active_skills": "not a list"})

        assert skill_manager.list_active_skills() == []

    def test_restore_state_invalid_skill_name(self, skill_manager, mock_skill_manifest):
        """Test restoring state with non-string skill names."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:
            mock_load.return_value = mock_skill_manifest

            state = {"active_skills": ["test-skill", 123, None]}  # type: ignore

            skill_manager.restore_state(state)

            # Only valid string should be processed
            assert skill_manager.is_active("test-skill")
            assert len(skill_manager.list_active_skills()) == 1


class TestSkillManagerRoundTrip:
    """Tests for state persistence round-trip."""

    def test_state_round_trip(
        self, skill_manager, mock_skill_manifest, another_mock_skill_manifest
    ):
        """Test saving and restoring state produces identical results."""
        with patch("activecontext.skills.manager.load_skill") as mock_load:

            def side_effect(name):
                if name == "test-skill":
                    return mock_skill_manifest
                elif name == "another-skill":
                    return another_mock_skill_manifest
                return None

            mock_load.side_effect = side_effect

            # Activate skills
            skill_manager.activate_skill("test-skill")
            skill_manager.activate_skill("another-skill")

            # Save state
            state = skill_manager.get_state()

            # Create new manager and restore
            new_manager = SkillManager()
            new_manager.restore_state(state)

            # Should have same active skills
            assert set(s.name for s in new_manager.list_active_skills()) == {
                "test-skill",
                "another-skill",
            }
            assert new_manager.is_active("test-skill")
            assert new_manager.is_active("another-skill")
