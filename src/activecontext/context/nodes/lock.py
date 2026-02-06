"""LockNode - File lock acquisition."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode
from .enums import LockStatus


@trace_all_fields
@dataclass(kw_only=True)
class LockNode(ContextNode):
    """Represents an async file lock acquisition.

    The lock_file() DSL function creates a LockNode and starts the lock
    acquisition in the background. The node's status changes as the lock
    is acquired or times out, and change notifications propagate up the DAG.

    Attributes:
        lockfile: Path to the lock file
        lock_status: Current lock status (PENDING, ACQUIRED, TIMEOUT, etc.)
        timeout: Maximum time to wait for lock acquisition (seconds)
        error_message: Error details if lock failed
        acquired_at: When the lock was acquired
        holder_pid: PID holding the lock (this process when acquired)
    """

    lockfile: str = ""
    lock_status: LockStatus = LockStatus.PENDING
    timeout: float = 30.0
    error_message: str | None = None
    acquired_at: float | None = None
    holder_pid: int | None = None

    @property
    def is_complete(self) -> bool:
        """True if lock operation has finished (acquired, timeout, released, or error)."""
        return self.lock_status in (
            LockStatus.ACQUIRED,
            LockStatus.TIMEOUT,
            LockStatus.RELEASED,
            LockStatus.ERROR,
        )

    @property
    def is_held(self) -> bool:
        """True if lock is currently held by this process."""
        return self.lock_status == LockStatus.ACQUIRED

    @property
    def error(self) -> str | None:
        """Error message if lock failed, None otherwise."""
        if self.lock_status == LockStatus.TIMEOUT:
            return self.error_message or f"Timed out after {self.timeout}s"
        if self.lock_status == LockStatus.ERROR:
            return self.error_message or "Lock error"
        return None

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "lockfile": self.lockfile,
            "status": self.lock_status.value,
            "timeout": self.timeout,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render timeout, holder, error, and acquired_at."""
        parts: list[str] = []
        parts.append(f"Timeout: {self.timeout}s\n")

        if self.holder_pid:
            parts.append(f"Holder PID: {self.holder_pid}\n")

        if self.error_message:
            parts.append(f"Error: {self.error_message}\n")

        if self.acquired_at:
            parts.append(f"Acquired at: {self.acquired_at:.3f}\n")

        return "".join(parts)

    def set_acquired(self, pid: int) -> LockNode:
        """Mark lock as acquired."""
        self.lock_status = LockStatus.ACQUIRED
        self.acquired_at = time.time()
        self.holder_pid = pid
        self.mark_changed(
            f"Lock '{self.lockfile}' acquired by PID {pid}",
        )
        return self

    def set_timeout(self) -> LockNode:
        """Mark lock acquisition as timed out."""
        self.lock_status = LockStatus.TIMEOUT
        self.error_message = f"Timed out after {self.timeout}s"
        self.mark_changed(
            f"Lock '{self.lockfile}' timed out",
        )
        return self

    def set_released(self) -> LockNode:
        """Mark lock as released."""
        self.lock_status = LockStatus.RELEASED
        self.mark_changed(
            f"Lock '{self.lockfile}' released",
        )
        return self

    def set_error(self, message: str) -> LockNode:
        """Mark lock operation as failed with error."""
        self.lock_status = LockStatus.ERROR
        self.error_message = message
        self.mark_changed(
            f"Lock '{self.lockfile}' error: {message}",
        )
        return self

    def render_digest(self) -> str:
        """Return 'Lock: file [STATUS]' format."""
        return f"Lock: {self.lockfile} [{self.lock_status.value.upper()}]"

    def get_wake_data(self) -> dict[str, Any]:
        """Return data dict for wake prompt template formatting."""
        return {
            "node_id": self.node_id,
            "lockfile": self.lockfile,
            "status": self.lock_status.value,
            "error": self.error_message or "",
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize LockNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "lockfile": self.lockfile,
                "lock_status": self.lock_status.value,
                "timeout": self.timeout,
                "error_message": self.error_message,
                "acquired_at": self.acquired_at,
                "holder_pid": self.holder_pid,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LockNode:
        """Deserialize LockNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "header")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            lockfile=data.get("lockfile", ""),
            lock_status=LockStatus(data.get("lock_status", "pending")),
            timeout=data.get("timeout", 30.0),
            error_message=data.get("error_message"),
            acquired_at=data.get("acquired_at"),
            holder_pid=data.get("holder_pid"),
        )
        return node
