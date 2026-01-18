"""LiteLLM provider implementation.

Supports 100+ LLM providers through litellm:
- Anthropic: "claude-3-5-sonnet-20241022"
- OpenAI: "gpt-4o", "gpt-4-turbo"
- Local: "ollama/llama3", "ollama/codellama"
- And many more...

See https://docs.litellm.ai/docs/providers for full list.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import litellm

from activecontext.core.llm.provider import (
    CompletionResult,
    Message,
    StreamChunk,
)
from activecontext.core.llm.providers import PROVIDER_CONFIGS


class LiteLLMProvider:
    """LLM provider using litellm for multi-provider support.

    Usage:
        # Anthropic
        provider = LiteLLMProvider("claude-3-5-sonnet-20241022")

        # OpenAI
        provider = LiteLLMProvider("gpt-4o")

        # Local Ollama
        provider = LiteLLMProvider("ollama/llama3")

        # With custom base URL
        provider = LiteLLMProvider("gpt-4", api_base="http://localhost:8000/v1")
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the provider.

        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022", "gpt-4o")
            api_key: API key (uses env vars if not provided)
            api_base: Custom API base URL
            **kwargs: Additional litellm options
        """
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._kwargs = kwargs

    @property
    def model(self) -> str:
        return self._model

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build kwargs for litellm call."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            "max_tokens": max_tokens,
            "stream": stream,
            **self._kwargs,
        }

        # Only include temperature if model supports it
        # Check model config for temperature support
        model_temp = self._get_model_temperature()
        if model_temp is not None and model_temp >= 0:
            # Use provided temperature or model default
            kwargs["temperature"] = temperature if temperature is not None else model_temp
        elif temperature is not None and self._model_supports_temperature():
            # Model not in config but temperature explicitly provided
            kwargs["temperature"] = temperature

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if stop:
            kwargs["stop"] = stop

        return kwargs

    def _get_model_temperature(self) -> float | None:
        """Get the configured temperature for this model, or None if not configured."""
        for provider_config in PROVIDER_CONFIGS.values():
            for model in provider_config.models:
                if model.id == self._model:
                    return model.temperature
        return None

    def _model_supports_temperature(self) -> bool:
        """Check if model is in our config (if not, assume it supports temperature)."""
        for provider_config in PROVIDER_CONFIGS.values():
            for model in provider_config.models:
                if model.id == self._model:
                    # Model is in config - only supports temp if configured
                    return model.temperature is not None
        # Model not in our config - assume it supports temperature
        return True

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """Generate a completion (non-streaming).

        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (None = use model default, or omit if unsupported)
            stop: Stop sequences
        """
        kwargs = self._build_kwargs(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=False,
        )

        response = await litellm.acompletion(**kwargs)

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return CompletionResult(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming completion.

        Args:
            messages: Conversation messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (None = use model default, or omit if unsupported)
            stop: Stop sequences
        """
        kwargs = self._build_kwargs(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        )

        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                text = delta.content or ""
                finish_reason = chunk.choices[0].finish_reason

                yield StreamChunk(
                    text=text,
                    is_final=finish_reason is not None,
                    finish_reason=finish_reason,
                )


# Convenience function for quick setup
def create_provider(
    model: str = "claude-sonnet-4-20250514",
    **kwargs: Any,
) -> LiteLLMProvider:
    """Create an LLM provider with sensible defaults.

    Args:
        model: Model to use (default: Claude Sonnet 4)
        **kwargs: Additional provider options

    Returns:
        Configured LiteLLMProvider
    """
    return LiteLLMProvider(model, **kwargs)
