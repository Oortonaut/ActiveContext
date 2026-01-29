"""ACP notification types."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from acp_debug.types.common import AcpModel, AvailableCommand, PlanEntry, ToolCallStatus


class CancelNotification(AcpModel):
    """Cancel in-progress prompt notification."""

    session_id: str = Field(alias="sessionId")


class SessionUpdateType(str, Enum):
    """Session update types."""

    AGENT_MESSAGE_TEXT = "agent_message_text_update"
    AGENT_THOUGHT_TEXT = "agent_thought_text_update"
    TOOL_CALL = "tool_call_update"
    PLAN = "plan"
    CURRENT_MODE = "current_mode_update"
    AVAILABLE_COMMANDS = "available_commands_update"


class SessionUpdate(AcpModel):
    """Session update notification (polymorphic based on session_update field)."""

    session_id: str = Field(alias="sessionId")
    session_update: SessionUpdateType = Field(alias="sessionUpdate")

    # agent_message_text_update / agent_thought_text_update
    text: str | None = None

    # tool_call_update
    tool_call_id: str | None = Field(default=None, alias="toolCallId")
    title: str | None = None
    kind: str | None = None
    status: ToolCallStatus | None = None
    content: list[dict[str, Any]] | None = None

    # plan
    entries: list[PlanEntry] | None = None

    # current_mode_update
    mode_id: str | None = Field(default=None, alias="modeId")

    # available_commands_update
    available_commands: list[AvailableCommand] | None = Field(
        default=None, alias="availableCommands"
    )
