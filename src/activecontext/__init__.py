"""ActiveContext: Agent loop with executable context as a Python statement timeline."""

__version__ = "0.1.0"

# Public API
from activecontext.core import LiteLLMProvider, LLMProvider, Message, Role
from activecontext.session import (
    ExecutionResult,
    ExecutionStatus,
    Projection,
    Session,
    SessionManager,
    SessionUpdate,
    Statement,
    Timeline,
    UpdateKind,
)
from activecontext.transport import ActiveContext, AsyncSession

__all__ = [
    # Main entry points
    "ActiveContext",
    "AsyncSession",
    # LLM
    "LLMProvider",
    "LiteLLMProvider",
    "Message",
    "Role",
    # Session
    "ExecutionResult",
    "ExecutionStatus",
    "Projection",
    "Session",
    "SessionManager",
    "SessionUpdate",
    "Statement",
    "Timeline",
    "UpdateKind",
]
