"""ActiveContext: Agent loop with executable context as a Python statement timeline."""

__version__ = "0.1.0"

# Public API
from activecontext.config import Config, get_config, load_config
from activecontext.core import LiteLLMProvider, LLMProvider, Message, Role
from activecontext.mcp import (
    MCPClientManager,
    MCPConnectionStatus,
    MCPToolInfo,
    MCPToolResult,
)
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
from activecontext.context.state import Expansion, NotificationLevel, TickFrequency, Visibility
from activecontext.terminal import ShellResult
from activecontext.transport import ActiveContext, AsyncSession

__all__ = [
    # Main entry points
    "ActiveContext",
    "AsyncSession",
    # Context state
    "Expansion",
    "NotificationLevel",
    "TickFrequency",
    "Visibility",
    # Config
    "Config",
    "load_config",
    "get_config",
    # LLM
    "LLMProvider",
    "LiteLLMProvider",
    "Message",
    "Role",
    # MCP
    "MCPClientManager",
    "MCPConnectionStatus",
    "MCPToolInfo",
    "MCPToolResult",
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
    # Terminal
    "ShellResult",
]
