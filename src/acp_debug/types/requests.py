"""ACP request types (Client → Agent and Agent → Client)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from acp_debug.types.common import (
    ClientCapabilities,
    ClientInfo,
    ContentBlock,
    PermissionOption,
    ToolCall,
)

# === Agent Methods (Client → Agent) ===


class InitializeRequest(BaseModel):
    """Initialize request from client to agent."""

    protocol_version: int = Field(alias="protocolVersion")
    client_capabilities: ClientCapabilities = Field(alias="clientCapabilities")
    client_info: ClientInfo = Field(alias="clientInfo")


class NewSessionRequest(BaseModel):
    """Create new session request."""

    cwd: str
    mcp_servers: list[dict[str, Any]] | None = Field(default=None, alias="mcpServers")


class LoadSessionRequest(BaseModel):
    """Load existing session request."""

    session_id: str = Field(alias="sessionId")
    cwd: str
    mcp_servers: list[dict[str, Any]] | None = Field(default=None, alias="mcpServers")


class ListSessionsRequest(BaseModel):
    """List available sessions request."""

    cwd: str
    cursor: str | None = None


class PromptRequest(BaseModel):
    """Send prompt to agent."""

    session_id: str = Field(alias="sessionId")
    prompt: list[ContentBlock]


class SetModeRequest(BaseModel):
    """Set session mode request."""

    session_id: str = Field(alias="sessionId")
    mode_id: str = Field(alias="modeId")


class SetModelRequest(BaseModel):
    """Set session model request."""

    session_id: str = Field(alias="sessionId")
    model_id: str = Field(alias="modelId")


# === Client Methods (Agent → Client) ===


class PermissionRequest(BaseModel):
    """Request permission from user."""

    session_id: str = Field(alias="sessionId")
    tool_call: ToolCall = Field(alias="toolCall")
    options: list[PermissionOption]


class ReadFileRequest(BaseModel):
    """Read file content request."""

    session_id: str = Field(alias="sessionId")
    path: str
    line: int | None = None
    limit: int | None = None


class WriteFileRequest(BaseModel):
    """Write file content request."""

    session_id: str = Field(alias="sessionId")
    path: str
    content: str


class CreateTerminalRequest(BaseModel):
    """Create terminal request."""

    session_id: str = Field(alias="sessionId")
    command: str
    args: list[str] | None = None
    cwd: str | None = None
    output_byte_limit: int | None = Field(default=None, alias="outputByteLimit")


class TerminalOutputRequest(BaseModel):
    """Poll terminal output request."""

    terminal_id: str = Field(alias="terminalId")


class WaitForExitRequest(BaseModel):
    """Wait for terminal exit request."""

    terminal_id: str = Field(alias="terminalId")


class KillTerminalRequest(BaseModel):
    """Kill terminal request."""

    terminal_id: str = Field(alias="terminalId")


class ReleaseTerminalRequest(BaseModel):
    """Release terminal resources request."""

    terminal_id: str = Field(alias="terminalId")
