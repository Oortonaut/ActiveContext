"""Skill manager for tracking and managing active skills per session.

The SkillManager tracks which skills are currently active in a session,
handles activation/deactivation, and provides lifecycle integration with
SessionManager.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from activecontext.skills.loader import load_skill
from activecontext.skills.schema import SkillManifest

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


class SkillManager:
    """Manages active skills for a session.

    Tracks which skills are currently active, handles activation/deactivation,
    and prevents duplicate activations. Optionally persists active skill state
    for session restoration.

    Example usage:
        manager = SkillManager()

        # Activate a skill
        manifest = manager.activate_skill("memory-system")
        if manifest:
            print(f"Activated: {manifest.name}")

        # List active skills
        for skill in manager.list_active_skills():
            print(f"Active: {skill.name}")

        # Deactivate a skill
        if manager.deactivate_skill("memory-system"):
            print("Deactivated memory-system")

        # Get skill state for persistence
        state = manager.get_state()
    """

    def __init__(self) -> None:
        """Initialize the skill manager."""
        self._active_skills: dict[str, SkillManifest] = {}

    def activate_skill(self, name: str) -> SkillManifest | None:
        """Activate a skill by name.

        Loads the skill manifest and adds it to the active skills set.
        If the skill is already active, returns the existing manifest without
        reloading.

        Args:
            name: The skill name to activate.

        Returns:
            SkillManifest if the skill was loaded successfully, None if the
            skill could not be found or loaded.
        """
        # Check if already active
        if name in self._active_skills:
            _log.debug("Skill '%s' is already active", name)
            return self._active_skills[name]

        # Load the skill
        manifest = load_skill(name)
        if not manifest:
            _log.warning("Could not load skill '%s'", name)
            return None

        # Add to active skills
        self._active_skills[name] = manifest
        _log.info("Activated skill: %s", name)
        return manifest

    def deactivate_skill(self, name: str) -> bool:
        """Deactivate a skill by name.

        Removes the skill from the active skills set.

        Args:
            name: The skill name to deactivate.

        Returns:
            True if the skill was deactivated, False if it was not active.
        """
        if name not in self._active_skills:
            _log.debug("Skill '%s' is not active", name)
            return False

        del self._active_skills[name]
        _log.info("Deactivated skill: %s", name)
        return True

    def list_active_skills(self) -> list[SkillManifest]:
        """List all currently active skills.

        Returns:
            List of SkillManifest objects for active skills, sorted by name.
        """
        manifests = list(self._active_skills.values())
        manifests.sort(key=lambda m: m.name)
        return manifests

    def is_active(self, name: str) -> bool:
        """Check if a skill is currently active.

        Args:
            name: The skill name to check.

        Returns:
            True if the skill is active, False otherwise.
        """
        return name in self._active_skills

    def get_active_skill(self, name: str) -> SkillManifest | None:
        """Get the manifest for an active skill.

        Args:
            name: The skill name.

        Returns:
            SkillManifest if the skill is active, None otherwise.
        """
        return self._active_skills.get(name)

    def clear(self) -> None:
        """Deactivate all skills."""
        count = len(self._active_skills)
        self._active_skills.clear()
        _log.info("Deactivated all skills (%d total)", count)

    def get_state(self) -> dict[str, list[str]]:
        """Get the current state for persistence.

        Returns a dictionary containing the names of all active skills,
        suitable for serialization to session state.

        Returns:
            Dictionary with "active_skills" key containing list of skill names.
        """
        return {"active_skills": sorted(self._active_skills.keys())}

    def restore_state(self, state: dict[str, list[str]]) -> None:
        """Restore skill state from persisted data.

        Attempts to activate all skills in the state. Skills that fail to load
        are logged but do not cause the restoration to fail.

        Args:
            state: Dictionary with "active_skills" key containing skill names.
        """
        if not isinstance(state, dict):
            _log.warning("Invalid skill state: expected dict, got %s", type(state))
            return

        skill_names = state.get("active_skills", [])
        if not isinstance(skill_names, list):
            _log.warning("Invalid active_skills: expected list, got %s", type(skill_names))
            return

        for name in skill_names:
            if not isinstance(name, str):
                _log.warning("Invalid skill name in state: %s", name)
                continue

            manifest = self.activate_skill(name)
            if not manifest:
                _log.warning("Could not restore skill '%s'", name)
