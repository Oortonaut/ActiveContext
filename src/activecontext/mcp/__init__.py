"""MCP client support for ActiveContext.

This module provides MCP (Model Context Protocol) client functionality,
allowing ActiveContext sessions to connect to MCP servers and expose
their tools to the LLM.

Example usage in a session:

    # Connect to a configured server
    fs = mcp_connect("filesystem")

    # Or connect dynamically
    gh = mcp_connect("github", command=["npx", "-y", "@mcp/server-github"])

    # Call tools (async, auto-awaited)
    result = fs.read_file(path="/home/user/data.txt")

    # Disconnect when done
    mcp_disconnect("filesystem")
"""

from activecontext.mcp.client import (
    MCPClientManager,
    MCPConnection,
    ServerProxy,
)
from activecontext.mcp.hooks import (
    get_pre_call_hook,
    set_pre_call_hook,
)
from activecontext.mcp.permissions import (
    MCPPermissionDenied,
    MCPPermissionManager,
    MCPPermissionRule,
)
from activecontext.mcp.types import (
    MCPConnectionStatus,
    MCPPromptInfo,
    MCPResourceInfo,
    MCPToolInfo,
    MCPToolResult,
)

__all__ = [
    # Client
    "MCPClientManager",
    "MCPConnection",
    "ServerProxy",
    # Permissions
    "MCPPermissionDenied",
    "MCPPermissionManager",
    "MCPPermissionRule",
    # Types
    "MCPConnectionStatus",
    "MCPPromptInfo",
    "MCPResourceInfo",
    "MCPToolInfo",
    "MCPToolResult",
    # Hooks
    "get_pre_call_hook",
    "set_pre_call_hook",
]
