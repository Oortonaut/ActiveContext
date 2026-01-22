"""Common ACP types shared across requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ToolResultContent(BaseModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    content: list[TextContent]


ContentBlock = TextContent | ToolResultContent | dict[str, Any]


class FSCapabilities(BaseModel):
    """Filesystem capabilities."""

    read_text_file: bool = Field(default=False, alias="readTextFile")
    write_text_file: bool = Field(default=False, alias="writeTextFile")


class PromptCapabilities(BaseModel):
    """Prompt content capabilities."""

    image: bool = False
    audio: bool = False
    embedded_context: bool = Field(default=False, alias="embeddedContext")


class ClientCapabilities(BaseModel):
    """Client capabilities sent during initialization."""

    fs: FSCapabilities | None = None
    terminal: bool = False


class ClientInfo(BaseModel):
    """Client identification."""

    name: str
    title: str | None = None
    version: str | None = None


class AgentCapabilities(BaseModel):
    """Agent capabilities advertised during initialization."""

    load_session: bool = Field(default=False, alias="loadSession")
    prompt_capabilities: PromptCapabilities | None = Field(
        default=None, alias="promptCapabilities"
    )
    session_capabilities: dict[str, Any] = Field(
        default_factory=dict, alias="sessionCapabilities"
    )


class AgentInfo(BaseModel):
    """Agent identification."""

    name: str
    title: str | None = None
    version: str | None = None


class ModelInfo(BaseModel):
    """Model information."""

    model_id: str = Field(alias="modelId")
    name: str
    description: str | None = None


class ModeInfo(BaseModel):
    """Operating mode information."""

    id: str
    name: str
    description: str | None = None


class ToolCallStatus(str, Enum):
    """Status of a tool call."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCall(BaseModel):
    """Tool call information for permission requests and updates."""

    tool_call_id: str = Field(alias="toolCallId")
    title: str | None = None
    kind: str | None = None
    status: ToolCallStatus = ToolCallStatus.PENDING
    content: list[ContentBlock] | None = None


class PermissionOptionKind(str, Enum):
    """Permission option kinds."""

    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    REJECT_ONCE = "reject_once"
    REJECT_ALWAYS = "reject_always"


class PermissionOption(BaseModel):
    """Permission option presented to user."""

    option_id: str = Field(alias="optionId")
    kind: PermissionOptionKind
    name: str


class PlanEntry(BaseModel):
    """Entry in an execution plan."""

    content: str
    priority: str | None = None
    status: str | None = None


class AvailableCommand(BaseModel):
    """Slash command advertised by agent."""

    name: str
    description: str | None = None
