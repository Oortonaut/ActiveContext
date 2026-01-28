"""Skill discovery and loading functionality.

Provides functions to scan the ~/.claude/skills/ directory for available skills
and parse SKILL.md files with YAML frontmatter.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from activecontext.skills.schema import SkillManifest

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

# YAML frontmatter pattern: starts with ---, ends with ---
_FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?(.*)$",
    re.DOTALL,
)


def get_skills_directory() -> Path:
    """Get the path to the skills directory.

    Returns:
        Path to ~/.claude/skills/ directory.
    """
    # Use HOME on Unix, USERPROFILE on Windows
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or ""
    if not home:
        _log.warning("Could not determine home directory")
        return Path.home() / ".claude" / "skills"
    return Path(home) / ".claude" / "skills"


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: The full file content starting with ---.

    Returns:
        Tuple of (frontmatter dict, remaining content).

    Raises:
        ValueError: If frontmatter is malformed or missing.
    """
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        raise ValueError("Missing or malformed YAML frontmatter (must start with ---)")

    frontmatter_yaml = match.group(1)
    body = match.group(2)

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml)
        if not isinstance(frontmatter, dict):
            raise ValueError("Frontmatter must be a YAML mapping")
        return frontmatter, body.strip()
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in frontmatter: {e}") from e


def load_skill_from_path(skill_path: Path) -> SkillManifest | None:
    """Load a skill manifest from a directory path.

    Args:
        skill_path: Path to the skill directory containing SKILL.md.

    Returns:
        SkillManifest if successfully loaded, None if file doesn't exist
        or is malformed.
    """
    skill_file = skill_path / "SKILL.md"

    if not skill_file.exists():
        _log.debug("SKILL.md not found in %s", skill_path)
        return None

    try:
        content = skill_file.read_text(encoding="utf-8")
    except PermissionError:
        _log.warning("Permission denied reading %s", skill_file)
        return None
    except OSError as e:
        _log.warning("Error reading %s: %s", skill_file, e)
        return None

    try:
        frontmatter, body = _parse_frontmatter(content)
    except ValueError as e:
        _log.warning("Error parsing %s: %s", skill_file, e)
        return None

    # Extract required fields
    name = frontmatter.get("name")
    description = frontmatter.get("description")

    if not name:
        _log.warning("Missing required 'name' field in %s", skill_file)
        return None
    if not description:
        _log.warning("Missing required 'description' field in %s", skill_file)
        return None

    # Extract optional fields
    license_str = frontmatter.get("license")
    allowed_tools = frontmatter.get("allowed-tools")
    metadata = frontmatter.get("metadata", {})

    # Validate allowed_tools is a list if present
    if allowed_tools is not None and not isinstance(allowed_tools, list):
        _log.warning("'allowed-tools' must be a list in %s", skill_file)
        return None

    try:
        return SkillManifest(
            name=str(name),
            description=str(description),
            license=str(license_str) if license_str else None,
            allowed_tools=allowed_tools,
            metadata=metadata if isinstance(metadata, dict) else {},
            content=body,
            path=skill_path.resolve(),
        )
    except ValueError as e:
        _log.warning("Invalid skill manifest in %s: %s", skill_file, e)
        return None


def load_skill(name: str) -> SkillManifest | None:
    """Load a skill by name from the skills directory.

    Args:
        name: The skill name (directory name under ~/.claude/skills/).

    Returns:
        SkillManifest if found and valid, None otherwise.
    """
    skills_dir = get_skills_directory()
    skill_path = skills_dir / name
    return load_skill_from_path(skill_path)


def discover_skills() -> list[SkillManifest]:
    """Discover all available skills in the skills directory.

    Scans ~/.claude/skills/ for subdirectories containing SKILL.md files.
    Malformed skills are logged and skipped.

    Returns:
        List of successfully loaded SkillManifest objects, sorted by name.
    """
    skills_dir = get_skills_directory()

    if not skills_dir.exists():
        _log.debug("Skills directory does not exist: %s", skills_dir)
        return []

    if not skills_dir.is_dir():
        _log.warning("Skills path is not a directory: %s", skills_dir)
        return []

    manifests: list[SkillManifest] = []

    try:
        entries = list(skills_dir.iterdir())
    except PermissionError:
        _log.warning("Permission denied listing %s", skills_dir)
        return []
    except OSError as e:
        _log.warning("Error listing %s: %s", skills_dir, e)
        return []

    for entry in entries:
        if not entry.is_dir():
            continue

        manifest = load_skill_from_path(entry)
        if manifest:
            manifests.append(manifest)

    # Sort by name for consistent ordering
    manifests.sort(key=lambda m: m.name)
    return manifests
