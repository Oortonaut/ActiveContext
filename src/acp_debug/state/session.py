"""Session state tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionState:
    """Tracked state for a session."""

    session_id: str
    mode: str | None = None
    model: str | None = None
    cwd: str | None = None

    # Message history (optional, for extensions that need it)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Pending tool calls
    pending_tools: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Custom extension data
    metadata: dict[str, Any] = field(default_factory=dict)
