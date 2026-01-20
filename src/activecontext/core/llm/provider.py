"""LLM provider protocol and base types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class Role(Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """A message in an LLM conversation.

    Attributes:
        role: The role (system, user, assistant)
        content: The message content
        originator: Who produced this message. Examples:
            - "user" - direct user input
            - "agent" - main agent response
            - "agent:plan" - agent in plan mode
            - "subagent:{name}" - a subagent
            - "tool:{name}" - tool execution result
            - None - unspecified (legacy)
    """

    role: Role
    content: str
    originator: str | None = None


@dataclass(slots=True)
class StreamChunk:
    """A chunk from streaming LLM response."""

    text: str
    is_final: bool = False
    finish_reason: str | None = None


@dataclass(slots=True)
class CompletionResult:
    """Result from a non-streaming completion."""

    content: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Implementations should support both streaming and non-streaming completions.
    """

    @property
    def model(self) -> str:
        """The model identifier being used."""
        ...

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Generate a completion (non-streaming).

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            CompletionResult with the generated content
        """
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Yields:
            StreamChunk objects as they arrive
        """
        ...
