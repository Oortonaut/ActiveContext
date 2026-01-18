"""Agent type registry for managing agent definitions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from activecontext.agents.schema import AgentType

if TYPE_CHECKING:
    pass


class AgentTypeRegistry:
    """Registry for agent type definitions.

    Manages built-in and user-defined agent types. Agent types define
    the system prompt and capabilities for a class of agents.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in types."""
        self._types: dict[str, AgentType] = {}
        self._load_builtin_types()

    def _load_builtin_types(self) -> None:
        """Load built-in agent types."""
        # Define minimal built-in types
        # More sophisticated types can be loaded from YAML files later
        self._types["default"] = AgentType(
            id="default",
            name="Default Agent",
            system_prompt="You are an assistant agent.",
            default_mode="normal",
            capabilities=[],
        )

        self._types["explorer"] = AgentType(
            id="explorer",
            name="Explorer Agent",
            system_prompt=(
                "You are an explorer agent. Your task is to search the codebase "
                "and gather information. Use view() to examine files, and report "
                "your findings back to your parent agent."
            ),
            default_mode="normal",
            capabilities=["read", "search"],
        )

        self._types["summarizer"] = AgentType(
            id="summarizer",
            name="Summarizer Agent",
            system_prompt=(
                "You are a summarizer agent. Your task is to read context nodes "
                "and produce concise summaries. Focus on extracting key information "
                "relevant to the task."
            ),
            default_mode="normal",
            capabilities=["read", "summarize"],
        )

    def get(self, type_id: str) -> AgentType | None:
        """Get an agent type by ID.

        Args:
            type_id: The agent type ID

        Returns:
            The agent type, or None if not found
        """
        return self._types.get(type_id)

    def register(self, agent_type: AgentType) -> None:
        """Register a new agent type.

        Args:
            agent_type: The agent type to register
        """
        self._types[agent_type.id] = agent_type

    def unregister(self, type_id: str) -> bool:
        """Unregister an agent type.

        Args:
            type_id: The agent type ID to remove

        Returns:
            True if removed, False if not found
        """
        if type_id in self._types:
            del self._types[type_id]
            return True
        return False

    def list_types(self) -> list[AgentType]:
        """List all registered agent types.

        Returns:
            List of all agent types
        """
        return list(self._types.values())

    def load_from_directory(self, directory: Path) -> int:
        """Load agent types from YAML files in a directory.

        Args:
            directory: Directory containing .yaml agent type files

        Returns:
            Number of types loaded
        """
        if not directory.exists():
            return 0

        loaded = 0
        for yaml_file in directory.glob("*.yaml"):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    agent_type = AgentType.from_dict(data)
                    self.register(agent_type)
                    loaded += 1
            except Exception:
                # Skip invalid files
                continue

        return loaded

    def load_user_types(self, project_root: Path) -> int:
        """Load user-defined agent types from project .ac/agents/ directory.

        Args:
            project_root: Project root directory

        Returns:
            Number of types loaded
        """
        agents_dir = project_root / ".ac" / "agents"
        return self.load_from_directory(agents_dir)
