"""TaskNode - Task tracking for concurrent operations."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.state import Expansion, IOMode, TaskStatus, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class TaskNode(ContextNode):
    """Represents a task in the context graph.

    TaskNodes track the status and metadata of concurrent tasks
    within a session. They provide visibility into running agents,
    blocking conversations, and streaming tasks.

    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (e.g., "agent", "mcp_menu", "shell_session")
        io_mode: I/O mode ("sync", "async", "streaming")
        status: Current task status (pending, running, paused, done, failed)
        created_at: When the task was created
        started_at: When the task started running (if applicable)
        completed_at: When the task completed (if applicable)
        metadata: Additional task-specific metadata
    """

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    task_type: str = "script"
    io_mode: IOMode = IOMode.ASYNC
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=lambda: time.time())
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def GetDigest(self) -> dict[str, Any]:
        """Return metadata digest for this task."""
        duration = None
        if self.started_at:
            if self.completed_at:
                duration = self.completed_at - self.started_at
            else:
                duration = time.time() - self.started_at

        return {
            "id": self.node_id,
            "type": self.node_type,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "io_mode": self.io_mode.value,
            "status": self.status.value,
            "duration": duration,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
        }

    def render_content(self) -> str:
        """Render full task info with timing and metadata."""
        lines = [f"Task: {self.task_id}"]
        lines.append(f"  Type: {self.task_type}")
        lines.append(f"  Status: {self.status.value}")
        lines.append(f"  I/O Mode: {self.io_mode.value}")

        # Timing info
        if self.created_at:
            from datetime import datetime

            created = datetime.fromtimestamp(self.created_at)
            lines.append(f"  Created: {created.isoformat()}")

        if self.started_at:
            from datetime import datetime

            started = datetime.fromtimestamp(self.started_at)
            lines.append(f"  Started: {started.isoformat()}")

        if self.completed_at:
            from datetime import datetime

            completed = datetime.fromtimestamp(self.completed_at)
            lines.append(f"  Completed: {completed.isoformat()}")

            if self.started_at:
                duration = self.completed_at - self.started_at
                lines.append(f"  Duration: {duration:.2f}s")

        if self.metadata:
            lines.append("  Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def update_status(self, status: TaskStatus) -> None:
        """Update task status with timing."""
        old_status = self.status
        self.status = status

        if status == TaskStatus.RUNNING and not self.started_at:
            self.started_at = time.time()
        elif status in (TaskStatus.DONE, TaskStatus.FAILED) and not self.completed_at:
            self.completed_at = time.time()

        self.mark_changed(f"status: {old_status.value} -> {status.value}")

    def render_digest(self) -> str:
        """Display name for the task with type and status."""
        return f"Task: {self.task_type} | {self.status.value}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "io_mode": self.io_mode.value,
                "status": self.status.value,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "metadata": self.metadata,
            }
        )
        return base

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TaskNode:
        """Deserialize from dictionary."""
        tick_freq_data = data.get("tick_frequency")
        tick_frequency = TickFrequency.from_dict(tick_freq_data) if tick_freq_data else None

        return cls(
            node_id=data.get("node_id", ""),
            default_expansion=Expansion(data.get("expansion", Expansion.CONTENT.value)),
            mode=data.get("mode", "running"),
            tick_frequency=tick_frequency,
            task_id=data.get("task_id", ""),
            task_type=data.get("task_type", "script"),
            io_mode=IOMode(data.get("io_mode", "async")),
            status=TaskStatus(data.get("status", "pending")),
            created_at=data.get("created_at", 0.0),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )
