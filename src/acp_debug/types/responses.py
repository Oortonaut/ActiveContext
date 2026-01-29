"""ACP response types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from acp_debug.types.common import AcpModel, AgentCapabilities, AgentInfo, ModeInfo, ModelInfo

# === Agent Method Responses ===


class InitializeResponse(AcpModel):
    """Initialize response from agent."""

    protocol_version: int = Field(alias="protocolVersion")
    agent_capabilities: AgentCapabilities = Field(alias="agentCapabilities")
    agent_info: AgentInfo = Field(alias="agentInfo")
    auth_methods: list[dict[str, Any]] = Field(default_factory=list, alias="authMethods")


class ModelsInfo(AcpModel):
    """Available models information."""

    available_models: list[ModelInfo] = Field(alias="availableModels")
    current_model_id: str | None = Field(default=None, alias="currentModelId")


class ModesInfo(AcpModel):
    """Available modes information."""

    available_modes: list[ModeInfo] = Field(alias="availableModes")
    current_mode_id: str | None = Field(default=None, alias="currentModeId")


class NewSessionResponse(AcpModel):
    """New session response."""

    session_id: str = Field(alias="sessionId")
    models: ModelsInfo | None = None
    modes: ModesInfo | None = None


class LoadSessionResponse(AcpModel):
    """Load session response."""

    session_id: str = Field(alias="sessionId")
    models: ModelsInfo | None = None
    modes: ModesInfo | None = None


class SessionInfo(AcpModel):
    """Session information for listing."""

    session_id: str = Field(alias="sessionId")
    title: str | None = None
    created_at: str | None = Field(default=None, alias="createdAt")


class ListSessionsResponse(AcpModel):
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


class PromptResponse(AcpModel):
    """Prompt response."""

    stop_reason: StopReason = Field(alias="stopReason")


class SetModeResponse(AcpModel):
    """Set mode response."""

    mode_id: str = Field(alias="modeId")


class SetModelResponse(AcpModel):
    """Set model response."""

    model_id: str = Field(alias="modelId")


# === Client Method Responses ===


class PermissionOutcome(AcpModel):
    """Permission decision outcome."""

    outcome: str  # "selected" or "dismissed"
    option_id: str | None = Field(default=None, alias="optionId")


class PermissionResponse(AcpModel):
    """Permission response from client."""

    outcome: PermissionOutcome


class ReadFileResponse(AcpModel):
    """Read file response."""

    content: str


class CreateTerminalResponse(AcpModel):
    """Create terminal response."""

    terminal_id: str = Field(alias="terminalId")


class TerminalOutputResponse(AcpModel):
    """Terminal output response."""

    output: str
    truncated: bool = False
    exit_status: dict[str, Any] | None = Field(default=None, alias="exitStatus")


class WaitForExitResponse(AcpModel):
    """Wait for exit response."""

    exit_code: int | None = Field(default=None, alias="exitCode")
    signal: str | None = None
