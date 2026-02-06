"""WorkNode - Multi-agent work coordination."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency, WorkStatus
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class WorkNode(ContextNode):
    """Represents this agent's work coordination entry.

    This node shows what files the agent is working on and any conflicts
    with other agents working on the same project. It integrates with the
    ScratchpadManager to provide visibility into multi-agent coordination.

    Attributes:
        intent: Human-readable description of current work
        work_status: Current work status (active, paused, done)
        files: List of file paths being accessed with mode (read/write)
        dependencies: Files needed but not modified
        conflicts: Detected conflicts with other agents
        agent_id: This agent's unique ID
    """

    intent: str = ""
    work_status: WorkStatus = WorkStatus.ACTIVE
    files: list[dict[str, str]] = field(default_factory=list)  # [{path, mode}]
    dependencies: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(
        default_factory=list
    )  # [{agent_id, file, their_mode, their_intent}]
    agent_id: str = ""

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "intent": self.intent,
            "status": self.work_status.value,
            "file_count": len(self.files),
            "conflict_count": len(self.conflicts),
            "agent_id": self.agent_id,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render agent, files, dependencies, and conflicts."""
        parts: list[str] = []
        parts.append(f"Agent: {self.agent_id}\n")

        # Show files being worked on
        if self.files:
            parts.append("\nFiles:\n")
            for f in self.files:
                mode_indicator = "[W]" if f.get("mode") == "write" else "[R]"
                parts.append(f"  {mode_indicator} {f.get('path', '')}\n")

        # Show dependencies
        if self.dependencies:
            parts.append("\nDependencies:\n")
            for dep in self.dependencies:
                parts.append(f"  [R] {dep}\n")

        # Show conflicts
        if self.conflicts:
            parts.append("\n--- CONFLICTS ---\n")
            for c in self.conflicts:
                parts.append(
                    f"  Agent {c.get('agent_id', '?')}: {c.get('file', '?')} "
                    f"[{c.get('their_mode', '?')}] - {c.get('their_intent', '?')}\n"
                )

        return "".join(parts)

    def set_intent(self, intent: str) -> WorkNode:
        """Update work intent."""
        old_intent = self.intent
        self.intent = intent
        if old_intent != intent:
            self.mark_changed(f"Intent: {old_intent[:30]}... → {intent[:30]}...")
        return self

    def set_files(self, files: list[dict[str, str]]) -> WorkNode:
        """Update files being worked on."""
        old_count = len(self.files)
        self.files = files
        if old_count != len(files):
            self.mark_changed(f"Files: {old_count} → {len(files)}")
        return self

    def set_conflicts(self, conflicts: list[dict[str, str]]) -> WorkNode:
        """Update detected conflicts."""
        old_count = len(self.conflicts)
        self.conflicts = conflicts
        if len(conflicts) != old_count:
            self.mark_changed(
                f"Work conflicts: {old_count} → {len(conflicts)}",
            )
        return self

    def render_digest(self) -> str:
        """Return 'Work: intent [status] Nf Nc' format."""
        intent_display = self.intent[:30] + "..." if len(self.intent) > 30 else self.intent
        nf, nc = len(self.files), len(self.conflicts)
        return f"Work: {intent_display} [{self.work_status.value}] {nf}f {nc}c"

    def to_dict(self) -> dict[str, Any]:
        """Serialize WorkNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "intent": self.intent,
                "work_status": self.work_status.value,
                "files": self.files,
                "dependencies": self.dependencies,
                "conflicts": self.conflicts,
                "agent_id": self.agent_id,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> WorkNode:
        """Deserialize WorkNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            intent=data.get("intent", ""),
            work_status=WorkStatus(data.get("work_status", "active")),
            files=data.get("files", []),
            dependencies=data.get("dependencies", []),
            conflicts=data.get("conflicts", []),
            agent_id=data.get("agent_id", ""),
        )
        return node
