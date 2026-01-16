"""Checkpoint system for context DAG organizational structure.

Checkpoints capture the edge structure (parent/child links) of the DAG,
allowing the same content nodes to participate in different organizational
contexts with different summarization intents.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroupState:
    """Captured state of a GroupNode at checkpoint time.

    Attributes:
        node_id: The group node's ID
        summary_prompt: Custom summarization prompt
        cached_summary: LLM-generated summary at checkpoint time
        last_child_versions: Version tracking for child nodes
    """

    node_id: str
    summary_prompt: str | None = None
    cached_summary: str | None = None
    last_child_versions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "summary_prompt": self.summary_prompt,
            "cached_summary": self.cached_summary,
            "last_child_versions": self.last_child_versions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupState:
        """Deserialize from dictionary."""
        return cls(
            node_id=data["node_id"],
            summary_prompt=data.get("summary_prompt"),
            cached_summary=data.get("cached_summary"),
            last_child_versions=data.get("last_child_versions", {}),
        )


@dataclass
class Checkpoint:
    """Snapshot of DAG organizational structure.

    Captures the edge structure (which nodes link to which) and group-specific
    state, allowing restoration of a particular organizational view of the
    content nodes.

    Attributes:
        checkpoint_id: Unique identifier (8-char UUID suffix)
        name: Human-readable name for the checkpoint
        created_at: Unix timestamp of creation
        edges: List of (child_id, parent_id) tuples
        group_states: Captured state for each GroupNode
        root_ids: Node IDs that were roots at checkpoint time
    """

    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    created_at: float = field(default_factory=time.time)
    edges: list[tuple[str, str]] = field(default_factory=list)
    group_states: dict[str, GroupState] = field(default_factory=dict)
    root_ids: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "name": self.name,
            "created_at": self.created_at,
            "edges": self.edges,
            "group_states": {
                node_id: state.to_dict()
                for node_id, state in self.group_states.items()
            },
            "root_ids": list(self.root_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            name=data["name"],
            created_at=data["created_at"],
            edges=[tuple(edge) for edge in data["edges"]],
            group_states={
                node_id: GroupState.from_dict(state_data)
                for node_id, state_data in data.get("group_states", {}).items()
            },
            root_ids=set(data.get("root_ids", [])),
        )

    def get_digest(self) -> dict[str, Any]:
        """Return a brief summary of the checkpoint."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "name": self.name,
            "created_at": self.created_at,
            "edge_count": len(self.edges),
            "group_count": len(self.group_states),
            "root_count": len(self.root_ids),
        }
