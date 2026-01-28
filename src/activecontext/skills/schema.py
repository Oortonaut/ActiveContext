"""Skill manifest schema definitions.

Defines the SkillManifest dataclass that represents the parsed YAML frontmatter
from a SKILL.md file.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillManifest:
    """Represents a parsed skill manifest from SKILL.md frontmatter.

    The frontmatter follows this format:
        ---
        name: skill-name          # Required: hyphen-case, max 64 chars
        description: What it does # Required: max 1024 chars, no < or >
        license: MIT              # Optional
        allowed-tools:            # Optional: restrict tool access
          - Bash
          - Read
        metadata:                 # Optional: custom fields
          version: 1.0.0
          model: claude-opus-4-5-20251101
        ---

    Attributes:
        name: The skill identifier in hyphen-case. Required, max 64 characters.
        description: Human-readable description. Required, max 1024 characters.
        license: Optional license identifier (e.g., "MIT", "Apache-2.0").
        allowed_tools: Optional list of tools the skill can access. If None,
            all tools are available.
        metadata: Optional dictionary for custom extension fields.
        content: The full content of SKILL.md after the frontmatter.
        path: The absolute path to the skill directory containing SKILL.md.
    """

    name: str
    description: str
    license: str | None = None
    allowed_tools: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    path: Path | None = None

    # Validation constants
    MAX_NAME_LENGTH: int = 64
    MAX_DESCRIPTION_LENGTH: int = 1024

    def __post_init__(self) -> None:
        """Validate the manifest fields."""
        # Validate name
        if not self.name:
            raise ValueError("Skill name is required")
        if len(self.name) > self.MAX_NAME_LENGTH:
            raise ValueError(
                f"Skill name exceeds {self.MAX_NAME_LENGTH} characters: {len(self.name)}"
            )
        if not self._is_valid_hyphen_case(self.name):
            raise ValueError(
                f"Skill name must be hyphen-case (lowercase letters, numbers, hyphens): {self.name}"
            )

        # Validate description
        if not self.description:
            raise ValueError("Skill description is required")
        if len(self.description) > self.MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Skill description exceeds {self.MAX_DESCRIPTION_LENGTH} "
                f"characters: {len(self.description)}"
            )
        if "<" in self.description or ">" in self.description:
            raise ValueError("Skill description must not contain < or > characters")

    @staticmethod
    def _is_valid_hyphen_case(name: str) -> bool:
        """Check if name is valid hyphen-case.

        Args:
            name: The name to validate.

        Returns:
            True if the name is valid hyphen-case.
        """
        if not name:
            return False
        # Must start and end with alphanumeric
        if name.startswith("-") or name.endswith("-"):
            return False
        # No consecutive hyphens
        if "--" in name:
            return False
        # Only lowercase letters, numbers, and single hyphens
        return all(char.islower() or char.isdigit() or char == "-" for char in name)
