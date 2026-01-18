"""Session management layer: protocols, Timeline, SessionManager."""

from activecontext.session.protocols import (
    ExecutionResult,
    ExecutionStatus,
    NamespaceTrace,
    Projection,
    SessionManagerProtocol,
    SessionProtocol,
    SessionUpdate,
    Statement,
    TimelineProtocol,
    UpdateKind,
)
from activecontext.session.session_manager import Session, SessionManager
from activecontext.session.timeline import Timeline

__all__ = [
    "ExecutionResult",
    "ExecutionStatus",
    "NamespaceTrace",
    "Projection",
    "Session",
    "SessionManager",
    "SessionManagerProtocol",
    "SessionProtocol",
    "SessionUpdate",
    "Statement",
    "Timeline",
    "TimelineProtocol",
    "UpdateKind",
]
