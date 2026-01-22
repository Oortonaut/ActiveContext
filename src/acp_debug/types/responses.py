"""ACP response types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from acp_debug.types.common import AgentCapabilities, AgentInfo, ModeInfo, ModelInfo

# === Agent Method Responses ===


class InitializeResponse(BaseModel):
    """Initialize response from agent."""

    protocol_version: int = Field(alias="protocolVersion")
    agent_capabilities: AgentCapabilities = Field(alias="agentCapabilities")
    agent_info: AgentInfo = Field(alias="agentInfo")
    auth_methods: list[dict[str, Any]] = Field(default_factory=list, alias="authMethods")


class ModelsInfo(BaseModel):
    """Available models information."""

    available_models: list[ModelInfo] = Field(alias="availableModels")
    current_model_id: str | None = Field(default=None, alias="currentModelId")


class ModesInfo(BaseModel):
    """Available modes information."""

    available_modes: list[ModeInfo] = Field(alias="availableModes")
    current_mode_id: str | None = Field(default=None, alias="currentModeId")


class NewSessionResponse(BaseModel):
    """New session response."""

    session_id: str = Field(alias="sessionId")
    models: ModelsInfo | None = None
    modes: ModesInfo | None = None


class LoadSessionResponse(BaseModel):
    """Load session response."""

    session_id: str = Field(alias="sessionId")
    models: ModelsInfo | None = None
    modes: ModesInfo | None = None


class SessionInfo(BaseModel):
    """Session information for listing."""

    session_id: str = Field(alias="sessionId")
    title: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")


class ListSessionsResponse(BaseModel):
    """List sessions response."""

    sessions: list[SessionInfo]
    cursor: str | None = None


class StopReason(str, Enum):
    """Prompt stop reasons."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    MAX_TURN_REQUESTS = "max_turn_requests"
    REFUSAL = "refusal"
    CANCELLED = "cancelled"


class PromptResponse(BaseModel):
    """Prompt response."""

    stop_reason: StopReason = Field(alias="stopReason")


class SetModeResponse(BaseModel):
    """Set mode response."""

    mode_id: str = Field(alias="modeId")


class SetModelResponse(BaseModel):
    """Set model response."""

    model_id: str = Field(alias="modelId")


# === Client Method Responses ===


class PermissionOutcome(BaseModel):
    """Permission decision outcome."""

    outcome: str  # "selected" or "dismissed"
    option_id: str | None = Field(default=None, alias="optionId")


class PermissionResponse(BaseModel):
    """Permission response from client."""

    outcome: PermissionOutcome


class ReadFileResponse(BaseModel):
    """Read file response."""

    content: str


class CreateTerminalResponse(BaseModel):
    """Create terminal response."""

    terminal_id: str = Field(alias="terminalId")


class TerminalOutputResponse(BaseModel):
    """Terminal output response."""

    output: str
    truncated: bool = False
    exit_status: dict[str, Any] | None = Field(default=None, alias="exitStatus")


class WaitForExitResponse(BaseModel):
    """Wait for exit response."""

    exit_code: int | None = Field(default=None, alias="exitCode")
    signal: str | None = None
