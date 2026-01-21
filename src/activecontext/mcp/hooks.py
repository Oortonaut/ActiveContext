"""MCP invocation hooks for real-time feedback.

Provides a contextvar-based mechanism for transports to receive
notifications before MCP tool calls begin, enabling UI feedback
during potentially slow operations.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Type for the pre-call hook: async (server_name, tool_name, arguments) -> None
MCPPreCallHook = Callable[[str, str, dict[str, Any]], Awaitable[None]]

# Contextvar holding the current pre-call hook (if any)
_mcp_pre_call_hook: ContextVar[MCPPreCallHook | None] = ContextVar(
    "mcp_pre_call_hook", default=None
)


def get_pre_call_hook() -> MCPPreCallHook | None:
    """Get the current MCP pre-call hook, if any."""
    return _mcp_pre_call_hook.get()


def set_pre_call_hook(hook: MCPPreCallHook | None) -> None:
    """Set the MCP pre-call hook for the current context."""
    _mcp_pre_call_hook.set(hook)
