"""State and tick frequency types for context nodes.

This module defines:
- NodeState: Semantic rendering states (replaces numeric LOD)
- TickFrequency: Typed tick frequency specification (replaces string parsing)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class NodeState(Enum):
    """Rendering state for context nodes.

    States control how nodes appear in projections:
    - HIDDEN: Not shown in projection (but still ticked if running)
    - COLLAPSED: Title and metadata only (size, trace count)
    - SUMMARY: LLM-generated summary (default for groups)
    - DETAILS: Full view with child settings (default for views)
    - ALL: Everything including full traces
    """

    HIDDEN = "hidden"
    COLLAPSED = "collapsed"
    SUMMARY = "summary"
    DETAILS = "details"
    ALL = "all"

    def __str__(self) -> str:
        return self.value


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
    interval: Optional[float] = None  # For periodic mode

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

    def to_dict(self) -> dict[str, any]:
        """Serialize to dict for YAML persistence."""
        result: dict[str, any] = {"mode": self.mode}
        if self.interval is not None:
            result["interval"] = self.interval
        return result

    @staticmethod
    def from_dict(data: dict[str, any]) -> TickFrequency:
        """Deserialize from dict."""
        return TickFrequency(
            mode=data["mode"],
            interval=data.get("interval"),
        )
