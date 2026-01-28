"""Skill discovery and loading for ActiveContext.

This module provides functionality to discover and load Claude Code skills
from the ~/.claude/skills/ directory. Each skill is defined by a SKILL.md
file with YAML frontmatter containing metadata.

Example usage:
    from activecontext.skills import discover_skills, load_skill, SkillManifest

    # Discover all available skills
    skills = discover_skills()
    for skill in skills:
        print(f"{skill.name}: {skill.description}")

    # Load a specific skill by name
    manifest = load_skill("memory-system")
    if manifest:
        print(manifest.description)
        print(manifest.content)  # Full SKILL.md content after frontmatter
"""

from activecontext.skills.loader import (
    discover_skills,
    get_skills_directory,
    load_skill,
    load_skill_from_path,
)
from activecontext.skills.schema import SkillManifest

__all__ = [
    # Schema
    "SkillManifest",
    # Loader functions
    "discover_skills",
    "load_skill",
    "load_skill_from_path",
    "get_skills_directory",
]
