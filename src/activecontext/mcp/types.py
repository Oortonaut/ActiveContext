"""MCP client type definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MCPConnectionStatus(Enum):
    """Status of an MCP server connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


@dataclass
class MCPResourceInfo:
    """Information about an MCP resource."""

    uri: str
    name: str
    description: str | None
    mime_type: str | None
    server_name: str


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""

    success: bool
    content: list[dict[str, Any]]
    structured_content: dict[str, Any] | None = None
    is_error: bool = False
    error_message: str | None = None

    def text(self) -> str:
        """Extract text content from the result."""
        texts = []
        for item in self.content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)

    def __repr__(self) -> str:
        if self.is_error:
            return f"MCPToolResult(error={self.error_message!r})"
        if self.structured_content:
            return f"MCPToolResult(structured={self.structured_content!r})"
        return f"MCPToolResult(content={self.content!r})"


@dataclass
class MCPPromptInfo:
    """Information about an MCP prompt."""

    name: str
    description: str | None
    arguments: list[dict[str, Any]]
    server_name: str
