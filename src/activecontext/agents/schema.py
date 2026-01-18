"""Data schemas for multi-agent coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AgentState(Enum):
    """Lifecycle state of an agent."""

    SPAWNED = "spawned"  # Created but not yet running
    RUNNING = "running"  # Actively processing turns
    WAITING = "waiting"  # Blocked on wait condition
    PAUSED = "paused"  # Manually paused
    DONE = "done"  # Completed task
    TERMINATED = "terminated"  # Forcibly stopped


@dataclass
class AgentMessage:
    """A message between agents."""

    id: str  # Unique message ID
    sender: str  # Agent ID
    recipient: str  # Agent ID
    content: str  # Message content (may contain {node_id} refs)
    node_refs: list[str] = field(default_factory=list)  # Node IDs referenced
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: datetime | None = None
    status: str = "pending"  # pending, delivered, read
    reply_to: str | None = None  # For threading
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "node_refs": self.node_refs,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "status": self.status,
            "reply_to": self.reply_to,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        return cls(
            id=data["id"],
            sender=data["sender"],
            recipient=data["recipient"],
            content=data["content"],
            node_refs=data.get("node_refs", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            delivered_at=(
                datetime.fromisoformat(data["delivered_at"])
                if data.get("delivered_at")
                else None
            ),
            status=data.get("status", "pending"),
            reply_to=data.get("reply_to"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentEntry:
    """An agent registration in the scratchpad."""

    id: str  # Agent ID (8-char UUID prefix)
    session_id: str  # Underlying session ID
    agent_type: str  # Type ID (explorer, summarizer, etc.)
    task: str  # Task description
    parent_id: str | None = None  # Parent agent ID
    state: AgentState = AgentState.SPAWNED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    heartbeat_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "task": self.task,
            "parent_id": self.parent_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "heartbeat_at": self.heartbeat_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEntry:
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            agent_type=data["agent_type"],
            task=data["task"],
            parent_id=data.get("parent_id"),
            state=AgentState(data.get("state", "spawned")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            heartbeat_at=datetime.fromisoformat(data["heartbeat_at"]),
        )


@dataclass
class AgentType:
    """Definition of an agent type."""

    id: str  # e.g., "explorer", "summarizer"
    name: str  # Human-readable name
    system_prompt: str  # System prompt for this agent type
    default_mode: str = "normal"  # Default session mode
    capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.system_prompt,
            "default_mode": self.default_mode,
            "capabilities": self.capabilities,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentType:
        return cls(
            id=data["id"],
            name=data["name"],
            system_prompt=data["system_prompt"],
            default_mode=data.get("default_mode", "normal"),
            capabilities=data.get("capabilities", []),
        )
