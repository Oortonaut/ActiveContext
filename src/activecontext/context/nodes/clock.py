"""ClockNode - Timer/countdown node with tick-driven updates."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class ClockNode(ContextNode):
    """Timer/countdown node with tick-driven updates.

    Attributes:
        start_time: Unix timestamp when timer started
        duration_seconds: Duration for countdown (None = stopwatch mode)
        is_running: Whether the timer is currently running
        elapsed_seconds: Cached elapsed time
    """

    start_time: float = field(default_factory=time.time)
    duration_seconds: float | None = None
    is_running: bool = True
    elapsed_seconds: float = 0.0

    def GetDigest(self) -> dict[str, Any]:
        remaining = self.get_remaining()
        return {
            "id": self.node_id,
            "type": self.node_type,
            "mode": "countdown" if self.duration_seconds else "stopwatch",
            "elapsed": f"{self.elapsed_seconds:.1f}s",
            "remaining": f"{remaining:.1f}s" if remaining is not None else None,
            "is_running": self.is_running,
            "expansion": self.default_expansion.value,
        }

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if not self.is_running:
            return self.elapsed_seconds
        return time.time() - self.start_time + self.elapsed_seconds

    def get_remaining(self) -> float | None:
        """Get remaining time in countdown mode (None in stopwatch mode)."""
        if self.duration_seconds is None:
            return None
        remaining = self.duration_seconds - self.get_elapsed()
        return max(0.0, remaining)

    def is_complete(self) -> bool:
        """Check if countdown has completed."""
        if self.duration_seconds is None:
            return False
        return self.get_elapsed() >= self.duration_seconds

    def start(self) -> ClockNode:
        """Start or resume the timer."""
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True
            self.mark_changed("Timer started")
        return self

    def pause(self) -> ClockNode:
        """Pause the timer."""
        if self.is_running:
            self.elapsed_seconds = self.get_elapsed()
            self.is_running = False
            self.mark_changed("Timer paused")
        return self

    def reset(self) -> ClockNode:
        """Reset the timer to zero."""
        self.start_time = time.time()
        self.elapsed_seconds = 0.0
        self.is_running = False
        self.mark_changed("Timer reset")
        return self

    def Recompute(self) -> None:
        """Update elapsed time on tick."""
        if self.is_running and self.is_complete():
            self.pause()
            self.mark_changed("Countdown completed")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def render_content(self) -> str:
        """Render time and status."""
        elapsed = self.get_elapsed()
        status = "⏸" if not self.is_running else "▶"

        if self.duration_seconds:
            remaining = self.get_remaining()
            elapsed_str = self._format_time(elapsed)
            duration_str = self._format_time(self.duration_seconds)
            remaining_str = self._format_time(remaining or 0)
            return f"{status} {elapsed_str} / {duration_str} (remaining: {remaining_str})\n"
        else:
            return f"{status} {self._format_time(elapsed)}\n"

    def render_digest(self) -> str:
        """Return clock type indicator with elapsed time."""
        elapsed = self._format_time(self.get_elapsed())
        if self.duration_seconds:
            duration = self._format_time(self.duration_seconds)
            return f"COUNTDOWN: {elapsed} / {duration}"
        return f"STOPWATCH: {elapsed}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize ClockNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "start_time": self.start_time,
                "duration_seconds": self.duration_seconds,
                "is_running": self.is_running,
                "elapsed_seconds": self.elapsed_seconds,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ClockNode:
        """Deserialize ClockNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
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
            start_time=data.get("start_time", time.time()),
            duration_seconds=data.get("duration_seconds"),
            is_running=data.get("is_running", True),
            elapsed_seconds=data.get("elapsed_seconds", 0.0),
        )
