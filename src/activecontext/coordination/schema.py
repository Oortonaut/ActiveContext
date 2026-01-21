"""Data schemas for agent work coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.agents.schema import AgentEntry, AgentMessage


@dataclass
class FileAccess:
    """A file or pattern the agent is accessing."""

    path: str
    mode: str = "read"  # "read" or "write"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "mode": self.mode}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileAccess:
        return cls(path=data["path"], mode=data.get("mode", "read"))


@dataclass
class WorkEntry:
    """A single agent's work registration."""

    id: str  # Agent ID (8-char UUID prefix)
    session_id: str
    intent: str
    status: str = "active"  # active, paused, done
    files: list[FileAccess] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    heartbeat_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "intent": self.intent,
            "status": self.status,
            "files": [f.to_dict() for f in self.files],
            "dependencies": self.dependencies,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "heartbeat_at": self.heartbeat_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkEntry:
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            intent=data["intent"],
            status=data.get("status", "active"),
            files=[FileAccess.from_dict(f) for f in data.get("files", [])],
            dependencies=data.get("dependencies", []),
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            heartbeat_at=datetime.fromisoformat(data["heartbeat_at"]),
        )


@dataclass
class Scratchpad:
    """The full scratchpad file.

    Version history:
    - v1: Work coordination only (entries)
    - v2: Multi-agent support (agents, messages)
    """

    version: int = 2
    entries: list[WorkEntry] = field(default_factory=list)
    # Multi-agent fields (v2)
    agents: list[AgentEntry] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "entries": [e.to_dict() for e in self.entries],
            "agents": [a.to_dict() for a in self.agents],
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scratchpad:
        # Import here to avoid circular dependency
        from activecontext.agents.schema import AgentEntry, AgentMessage

        return cls(
            version=data.get("version", 1),
            entries=[WorkEntry.from_dict(e) for e in data.get("entries", [])],
            agents=[AgentEntry.from_dict(a) for a in data.get("agents", [])],
            messages=[AgentMessage.from_dict(m) for m in data.get("messages", [])],
        )


@dataclass
class Conflict:
    """A potential conflict with another agent."""

    agent_id: str
    file: str
    their_mode: str  # read/write
    their_intent: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "file": self.file,
            "their_mode": self.their_mode,
            "their_intent": self.their_intent,
        }
