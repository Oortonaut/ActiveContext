"""State and tick frequency types for context nodes.

This module defines:
- Expansion: Semantic rendering states (replaces numeric LOD)
- TickFrequency: Typed tick frequency specification (replaces string parsing)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Visibility(Enum):
    """Visibility state for context nodes.

    Controls whether a node appears in projections:
    - HIDE: Not shown in projection (but still ticked if running)
    - SHOW: Shown in projection (rendering controlled by Expansion)
    """

    HIDE = "hide"
    SHOW = "show"

    def __str__(self) -> str:
        return self.value


class Expansion(Enum):
    """Rendering state for context nodes.

    States control how visible nodes appear in projections:
    - COLLAPSED: Title and metadata only (size, trace count)
    - SUMMARY: LLM-generated summary (default for groups)
    - DETAILS: Full view with child settings (default for views)
    """

    COLLAPSED = "collapsed"
    SUMMARY = "summary"
    DETAILS = "details"

    def __str__(self) -> str:
        return self.value


class NotificationLevel(Enum):
    """Controls notification behavior when a node changes.

    Notification levels determine how changes are communicated to the agent:
    - IGNORE: Changes propagate via notify_parents(), but no notification message generated
    - HOLD: Notification queued, delivered at tick boundary
    - WAKE: Notification queued, delivered at tick boundary, AND agent woken immediately
    """

    IGNORE = "ignore"
    HOLD = "hold"
    WAKE = "wake"

    def __str__(self) -> str:
        return self.value


@dataclass(slots=True)
class Notification:
    """A notification about a node change.

    Notifications are generated when nodes with notification_level != IGNORE
    are changed. They are collected at subscription points and delivered
    to the agent via the Alerts group.

    Attributes:
        node_id: Source node that changed
        trace_id: Unique ID for deduplication (node_id:version)
        header: Brief description (e.g., "text_3: (-5/+12 lines at 100)")
        level: NotificationLevel value ("hold" or "wake")
        originator: Who/what caused the change (node ID, filename, or arbitrary string)
        timestamp: When the notification was generated
    """

    node_id: str
    trace_id: str
    header: str
    level: str  # "hold" or "wake" - string to avoid issues with enum serialization
    originator: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TickFrequency:
    """Tick frequency specification for node recomputation.

    Modes:
    - turn: Execute every turn (replaces "Sync")
    - async: Async execution
    - never: No execution
    - periodic: Execute at intervals

    Attributes:
        mode: Execution mode
        interval: Interval in seconds (for periodic mode)
    """

    mode: str  # "turn", "async", "never", "periodic"
    interval: float | None = None  # For periodic mode

    @staticmethod
    def turn() -> TickFrequency:
        """Execute every turn (replaces "Sync")."""
        return TickFrequency(mode="turn")

    @staticmethod
    def async_() -> TickFrequency:
        """Async execution."""
        return TickFrequency(mode="async")

    @staticmethod
    def never() -> TickFrequency:
        """No execution."""
        return TickFrequency(mode="never")

    @staticmethod
    def period(seconds: float) -> TickFrequency:
        """Periodic execution at given interval.

        Args:
            seconds: Interval in seconds
        """
        return TickFrequency(mode="periodic", interval=seconds)

    @staticmethod
    def from_string(s: str) -> TickFrequency:
        """Parse tick frequency from string format.

        Supported formats:
        - "Sync" or "turn" -> turn()
        - "async" -> async_()
        - "never" -> never()
        - "Periodic:5s" or "period:5.0" -> period(5.0)
        - "Periodic:2m" -> period(120.0)

        Args:
            s: String representation

        Returns:
            TickFrequency instance

        Raises:
            ValueError: If string format is invalid
        """
        s = s.strip()

        if s.lower() in ("sync", "turn"):
            return TickFrequency.turn()
        elif s.lower() == "async":
            return TickFrequency.async_()
        elif s.lower() == "never":
            return TickFrequency.never()
        elif s.lower().startswith("periodic:") or s.lower().startswith("period:"):
            # Parse "Periodic:5s", "period:120.0", etc.
            parts = s.split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid periodic format: {s}")

            time_str = parts[1].strip()

            # Check for time units
            if time_str.endswith("s"):
                seconds = float(time_str[:-1])
            elif time_str.endswith("m"):
                seconds = float(time_str[:-1]) * 60
            elif time_str.endswith("h"):
                seconds = float(time_str[:-1]) * 3600
            else:
                # No unit, assume seconds
                seconds = float(time_str)

            return TickFrequency.period(seconds)
        else:
            raise ValueError(f"Unknown tick frequency format: {s}")

    def to_string(self) -> str:
        """Convert to string representation.

        Returns:
            String format suitable for parsing with from_string()
        """
        if self.mode == "turn":
            return "turn"
        elif self.mode == "async":
            return "async"
        elif self.mode == "never":
            return "never"
        elif self.mode == "periodic":
            if self.interval is None:
                raise ValueError("Periodic mode requires interval")
            return f"period:{self.interval}"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def __str__(self) -> str:
        return self.to_string()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML persistence."""
        result: dict[str, Any] = {"mode": self.mode}
        if self.interval is not None:
            result["interval"] = self.interval
        return result

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TickFrequency:
        """Deserialize from dict."""
        return TickFrequency(
            mode=data["mode"],
            interval=data.get("interval"),
        )
