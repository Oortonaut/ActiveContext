"""Session persistence storage.

Handles saving and loading sessions to/from YAML files in:
  $PROJECT/.ac/sessions/<session-id>.yaml

Session files contain:
- session_id: Unique identifier
- title: Human-readable title
- created_at: ISO timestamp
- updated_at: ISO timestamp
- conversation: List of messages
- timeline: List of executed statements (sources only)
- context_graph: Serialized DAG with nodes and edges
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from activecontext.logging import get_logger

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.core.llm.provider import Message

log = get_logger("storage")


@dataclass
class SessionMetadata:
    """Lightweight session metadata for listing."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    cwd: str


@dataclass
class SessionData:
    """Full session data for persistence."""

    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    cwd: str
    conversation: list[dict[str, Any]]
    timeline: list[str]
    context_graph: dict[str, Any]


def get_sessions_dir(cwd: str) -> Path:
    """Get the sessions directory for a project.

    Args:
        cwd: Project working directory.

    Returns:
        Path to .ac/sessions/ directory.
    """
    return Path(cwd) / ".ac" / "sessions"


def ensure_sessions_dir(cwd: str) -> Path:
    """Ensure the sessions directory exists.

    Args:
        cwd: Project working directory.

    Returns:
        Path to .ac/sessions/ directory.
    """
    sessions_dir = get_sessions_dir(cwd)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def get_session_path(cwd: str, session_id: str) -> Path:
    """Get the path to a session file.

    Args:
        cwd: Project working directory.
        session_id: Session identifier.

    Returns:
        Path to the session YAML file.
    """
    return get_sessions_dir(cwd) / f"{session_id}.yaml"


def generate_default_title() -> str:
    """Generate default session title with current date/time.

    Returns:
        Title like "Session 2026-01-17 10:30"
    """
    return f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"


def save_session(
    cwd: str,
    session_id: str,
    title: str,
    created_at: datetime,
    message_history: list[Message],
    timeline_sources: list[str],
    context_graph: ContextGraph,
) -> Path:
    """Save a session to YAML file.

    Performs atomic write by writing to a temp file first.

    Args:
        cwd: Project working directory.
        session_id: Session identifier.
        title: Session title.
        created_at: When session was created.
        message_history: List of Message objects.
        timeline_sources: List of statement source strings.
        context_graph: The context graph.

    Returns:
        Path to the saved session file.
    """
    sessions_dir = ensure_sessions_dir(cwd)
    session_path = sessions_dir / f"{session_id}.yaml"
    temp_path = sessions_dir / f"{session_id}.yaml.tmp"

    # Serialize message history
    msg_data = []
    for msg in message_history:
        msg_dict: dict[str, Any] = {
            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
            "content": msg.content,
        }
        if hasattr(msg, "originator") and msg.originator:
            msg_dict["originator"] = msg.originator
        msg_data.append(msg_dict)

    # Build session data
    # Note: YAML key "conversation" kept for backward compatibility with saved sessions
    data = {
        "session_id": session_id,
        "title": title,
        "created_at": created_at.isoformat(),
        "updated_at": datetime.now().isoformat(),
        "cwd": cwd,
        "conversation": msg_data,
        "timeline": timeline_sources,
        "context_graph": context_graph.to_dict(),
    }

    # Atomic write
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # On Windows, need to remove existing file before rename
        if session_path.exists():
            session_path.unlink()
        temp_path.rename(session_path)

        log.debug("Saved session %s to %s", session_id, session_path)
        return session_path
    except Exception as e:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to save session: {e}") from e


def load_session_metadata(session_path: Path) -> SessionMetadata | None:
    """Load just the metadata from a session file.

    This is faster than loading the full session data.

    Args:
        session_path: Path to the session YAML file.

    Returns:
        SessionMetadata, or None if file doesn't exist or is invalid.
    """
    if not session_path.exists():
        return None

    try:
        with open(session_path, encoding="utf-8") as f:
            # Load full data but only return metadata
            data = yaml.safe_load(f)

        return SessionMetadata(
            session_id=data["session_id"],
            title=data.get("title", "Untitled"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            cwd=data.get("cwd", ""),
        )
    except Exception as e:
        log.warning("Failed to load session metadata from %s: %s", session_path, e)
        return None


def load_session_data(session_path: Path) -> SessionData | None:
    """Load full session data from a YAML file.

    Args:
        session_path: Path to the session YAML file.

    Returns:
        SessionData, or None if file doesn't exist or is invalid.
    """
    if not session_path.exists():
        return None

    try:
        with open(session_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return SessionData(
            session_id=data["session_id"],
            title=data.get("title", "Untitled"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            cwd=data.get("cwd", ""),
            conversation=data.get("conversation", []),
            timeline=data.get("timeline", []),
            context_graph=data.get("context_graph", {}),
        )
    except Exception as e:
        log.warning("Failed to load session data from %s: %s", session_path, e)
        return None


def list_sessions(cwd: str) -> list[SessionMetadata]:
    """List all sessions for a project.

    Args:
        cwd: Project working directory.

    Returns:
        List of SessionMetadata, sorted by updated_at (newest first).
    """
    sessions_dir = get_sessions_dir(cwd)
    if not sessions_dir.exists():
        return []

    sessions: list[SessionMetadata] = []
    for path in sessions_dir.glob("*.yaml"):
        # Skip temp files
        if path.name.endswith(".yaml.tmp"):
            continue

        metadata = load_session_metadata(path)
        if metadata:
            sessions.append(metadata)

    # Sort by updated_at, newest first
    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions


def delete_session(cwd: str, session_id: str) -> bool:
    """Delete a session file.

    Args:
        cwd: Project working directory.
        session_id: Session identifier.

    Returns:
        True if deleted, False if file didn't exist.
    """
    session_path = get_session_path(cwd, session_id)
    if session_path.exists():
        session_path.unlink()
        log.debug("Deleted session %s", session_id)
        return True
    return False


def session_exists(cwd: str, session_id: str) -> bool:
    """Check if a session file exists.

    Args:
        cwd: Project working directory.
        session_id: Session identifier.

    Returns:
        True if session file exists.
    """
    return get_session_path(cwd, session_id).exists()
