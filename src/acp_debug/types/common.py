"""Common ACP types shared across requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AcpModel(BaseModel):
    """Base model for ACP types with populate_by_name enabled."""

    model_config = ConfigDict(populate_by_name=True)


class TextContent(AcpModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str


class ToolResultContent(AcpModel):
    """Tool result content block."""

    type: Literal["tool_result"] = "tool_result"
    content: list[TextContent]


ContentBlock = TextContent | ToolResultContent | dict[str, Any]


class FSCapabilities(AcpModel):
    """Filesystem capabilities."""

    read_text_file: bool = Field(default=False, alias="readTextFile")
    write_text_file: bool = Field(default=False, alias="writeTextFile")


class PromptCapabilities(AcpModel):
    """Prompt content capabilities."""

    image: bool = False
    audio: bool = False
    embedded_context: bool = Field(default=False, alias="embeddedContext")


class ClientCapabilities(AcpModel):
    """Client capabilities sent during initialization."""

    fs: FSCapabilities | None = None
    terminal: bool = False


class ClientInfo(AcpModel):
    """Client identification."""

    name: str
    title: str | None = None
    version: str | None = None


class AgentCapabilities(AcpModel):
    """Agent capabilities advertised during initialization."""

    load_session: bool = Field(default=False, alias="loadSession")
    prompt_capabilities: PromptCapabilities | None = Field(
        default=None, alias="promptCapabilities"
    )
    session_capabilities: dict[str, Any] = Field(
        default_factory=dict, alias="sessionCapabilities"
    )


class AgentInfo(AcpModel):
    """Agent identification."""

    name: str
    title: str | None = None
    version: str | None = None


class ModelInfo(AcpModel):
    """Model information."""

    model_id: str = Field(alias="modelId")
    name: str
    description: str | None = None


class ModeInfo(AcpModel):
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


class ToolCall(AcpModel):
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


class PermissionOption(AcpModel):
    """Permission option presented to user."""

    option_id: str = Field(alias="optionId")
    kind: PermissionOptionKind
    name: str


class PlanEntry(AcpModel):
    """Entry in an execution plan."""

    content: str
    priority: str | None = None
    status: str | None = None


class AvailableCommand(AcpModel):
    """Slash command advertised by agent."""

    name: str
    description: str | None = None
