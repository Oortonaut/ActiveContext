"""Tests for LLM provider discovery and LiteLLM integration.

Tests coverage for:
- src/activecontext/core/llm/discovery.py
- src/activecontext/core/llm/litellm_provider.py
- src/activecontext/core/llm/provider.py
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, patch

from activecontext.core.llm.discovery import (
    ModelInfo,
    RoleModelEntry,
    get_all_role_models,
    get_available_models,
    get_available_providers,
    get_default_model,
    get_model_for_role,
    get_role_models,
)
from activecontext.core.llm.providers import (
    DEFAULT_ROLES,
    PROVIDER_CONFIGS,
    PROVIDER_PRIORITY,
    ROLE_MODEL_DEFAULTS,
)
from activecontext.core.llm.litellm_provider import LiteLLMProvider, create_provider
from activecontext.core.llm.provider import (
    CompletionResult,
    Message,
    Role,
    StreamChunk,
)
from tests.utils import create_mock_llm_response, create_mock_llm_stream_chunk


# =============================================================================
# Provider Discovery Tests
# =============================================================================


class TestProviderDiscovery:
    """Tests for provider discovery based on API keys."""

    def test_no_api_keys(self, monkeypatch):
        """Test discovery with no API keys set."""
        # Clear all API keys
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)

        providers = get_available_providers()
        assert providers == []

        models = get_available_models()
        assert models == []

        default_model = get_default_model()
        assert default_model is None

    def test_anthropic_only(self, monkeypatch):
        """Test discovery with only Anthropic API key."""
        # Clear all, then set Anthropic
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")

        providers = get_available_providers()
        assert providers == ["anthropic"]

        models = get_available_models()
        assert len(models) == 3  # Opus, Sonnet, Haiku
        assert all(isinstance(m, ModelInfo) for m in models)
        assert all(m.provider == "anthropic" for m in models)
        assert models[0].model_id == "claude-opus-4-5-20251101"
        assert models[0].name == "Claude Opus 4.5"

        # Default is "balanced" role = sonnet for Anthropic
        default_model = get_default_model()
        assert default_model == "claude-sonnet-4-5-20250929"

    def test_openai_only(self, monkeypatch):
        """Test discovery with only OpenAI API key."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")

        providers = get_available_providers()
        assert providers == ["openai"]

        models = get_available_models()
        assert len(models) == 5  # GPT-5.2, 5.2-codex, 4.1, 4.1-mini, 4.1-nano
        assert all(m.provider == "openai" for m in models)

        # Default is "balanced" role = gpt-4.1 for OpenAI
        default_model = get_default_model()
        assert default_model == "gpt-4.1"

    def test_multiple_providers(self, monkeypatch):
        """Test discovery with multiple API keys."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-456")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-789")

        providers = get_available_providers()
        assert set(providers) == {"anthropic", "openai", "deepseek"}

        models = get_available_models()
        assert len(models) == 3 + 5 + 2  # Anthropic + OpenAI + DeepSeek

        # Default is "balanced" role, Anthropic has priority = sonnet
        default_model = get_default_model()
        assert default_model == "claude-sonnet-4-5-20250929"

    def test_priority_order(self, monkeypatch):
        """Test default model priority: Anthropic > OpenAI > others (using balanced role)."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)

        # Only Groq - balanced role = llama-3.3-70b-versatile
        monkeypatch.setenv("GROQ_API_KEY", "sk-groq-123")
        assert get_default_model() == "groq/llama-3.3-70b-versatile"

        # Add OpenAI (higher priority) - balanced role = gpt-4.1
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-456")
        assert get_default_model() == "gpt-4.1"

        # Add Anthropic (highest priority) - balanced role = sonnet
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-789")
        assert get_default_model() == "claude-sonnet-4-5-20250929"

    def test_config_override_default_model(self, monkeypatch):
        """Test that config.llm.role overrides default 'balanced' role."""
        # Set up API key
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")

        # Mock config to use "reasoning" role instead of default "balanced"
        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.llm.role = "reasoning"
        mock_config.llm.provider = None  # Use priority order
        mock_config.llm.role_providers = []

        with patch("activecontext.config.get_config", return_value=mock_config):
            default_model = get_default_model()
            # reasoning role for anthropic = opus
            assert default_model == "claude-opus-4-5-20251101"

    def test_model_info_attributes(self, monkeypatch):
        """Test ModelInfo dataclass attributes."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        models = get_available_models()
        model = models[0]

        assert isinstance(model, ModelInfo)
        assert model.model_id == "claude-opus-4-5-20251101"
        assert model.name == "Claude Opus 4.5"
        assert model.provider == "anthropic"
        assert model.description == "Most capable, hybrid reasoning"


# =============================================================================
# Role-Based Model Selection Tests
# =============================================================================


class TestRoleBasedSelection:
    """Tests for role-based model selection."""

    def test_default_roles_constant(self):
        """Test DEFAULT_ROLES contains expected roles."""
        assert "coding" in DEFAULT_ROLES
        assert "reasoning" in DEFAULT_ROLES
        assert "writing" in DEFAULT_ROLES
        assert "balanced" in DEFAULT_ROLES
        assert "fast" in DEFAULT_ROLES
        assert "cheap" in DEFAULT_ROLES

    def test_provider_priority_constant(self):
        """Test PROVIDER_PRIORITY order."""
        assert PROVIDER_PRIORITY[0] == "anthropic"
        assert PROVIDER_PRIORITY[1] == "openai"
        assert "groq" in PROVIDER_PRIORITY
        assert "deepseek" in PROVIDER_PRIORITY

    def test_role_model_defaults_coverage(self):
        """Test ROLE_MODEL_DEFAULTS has entries for all role/provider combinations."""
        for role in DEFAULT_ROLES:
            for provider in ["anthropic", "openai"]:
                key = (role, provider)
                assert key in ROLE_MODEL_DEFAULTS, f"Missing {key}"

    def test_get_role_models(self, monkeypatch):
        """Test get_role_models returns correct entries."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")

        models = get_role_models("coding")

        assert len(models) == 2  # Anthropic and OpenAI
        assert all(isinstance(m, RoleModelEntry) for m in models)
        assert all(m.role == "coding" for m in models)

        # Check display_name format
        anthropic_model = next(m for m in models if m.provider == "anthropic")
        assert "Coding" in anthropic_model.display_name
        assert "anthropic" in anthropic_model.display_name
        assert anthropic_model.model_id == "claude-sonnet-4-5-20250929"

    def test_get_role_models_ordering(self, monkeypatch):
        """Test get_role_models respects provider priority."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        models = get_role_models("fast")

        # Anthropic should come first (higher priority)
        assert models[0].provider == "anthropic"
        assert models[1].provider == "openai"

    def test_get_all_role_models(self, monkeypatch):
        """Test get_all_role_models returns all roles."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        all_models = get_all_role_models()

        assert "coding" in all_models
        assert "reasoning" in all_models
        assert "fast" in all_models
        assert len(all_models) == len(DEFAULT_ROLES)

    def test_get_model_for_role(self, monkeypatch):
        """Test get_model_for_role returns correct model."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        model = get_model_for_role("coding")
        assert model == "claude-sonnet-4-5-20250929"

        model = get_model_for_role("reasoning")
        assert model == "claude-opus-4-5-20251101"

        model = get_model_for_role("fast")
        assert model == "claude-haiku-4-5-20251001"

    def test_get_model_for_role_with_provider(self, monkeypatch):
        """Test get_model_for_role with explicit provider."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")

        # Explicit provider
        model = get_model_for_role("coding", provider="openai")
        assert model == "gpt-5.2-codex"

        model = get_model_for_role("fast", provider="openai")
        assert model == "gpt-4.1-mini"

    def test_get_model_for_role_unavailable_provider(self, monkeypatch):
        """Test get_model_for_role with unavailable provider returns None."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        # OpenAI not available
        model = get_model_for_role("coding", provider="openai")
        assert model is None

    def test_get_model_for_role_uses_config_preference(self, monkeypatch):
        """Test get_model_for_role uses saved config preferences."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")

        # Mock config with saved role preference (provider only - model looked up)
        mock_role_provider = Mock()
        mock_role_provider.role = "coding"
        mock_role_provider.provider = "openai"
        mock_role_provider.model = None  # No override, lookup from ROLE_MODEL_DEFAULTS

        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.llm.role = None
        mock_config.llm.provider = None
        mock_config.llm.role_providers = [mock_role_provider]

        with patch("activecontext.config.get_config", return_value=mock_config):
            # Should use saved preference (OpenAI) instead of priority (Anthropic)
            model = get_model_for_role("coding")
            assert model == "gpt-5.2-codex"

    def test_get_model_for_role_uses_model_override(self, monkeypatch):
        """Test get_model_for_role uses model override when specified."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")

        # Mock config with explicit model override
        mock_role_provider = Mock()
        mock_role_provider.role = "fast"
        mock_role_provider.provider = "openai"
        mock_role_provider.model = "gpt-5-mini-custom"  # Custom model override

        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.llm.role = None
        mock_config.llm.provider = None
        mock_config.llm.role_providers = [mock_role_provider]

        with patch("activecontext.config.get_config", return_value=mock_config):
            # Should use the custom model override
            model = get_model_for_role("fast")
            assert model == "gpt-5-mini-custom"

    def test_get_default_model_uses_role_from_config(self, monkeypatch):
        """Test get_default_model uses role from config."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        mock_config = Mock()
        mock_config.llm = Mock()
        mock_config.llm.role = "coding"
        mock_config.llm.provider = None
        mock_config.llm.role_providers = []

        with patch("activecontext.config.get_config", return_value=mock_config):
            model = get_default_model()
            # coding role for anthropic = sonnet
            assert model == "claude-sonnet-4-5-20250929"

    def test_role_model_entry_display_name(self, monkeypatch):
        """Test RoleModelEntry display_name format."""
        for _, config in PROVIDER_CONFIGS.items():
            monkeypatch.delenv(config.env_var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

        models = get_role_models("fast")
        entry = models[0]

        # Format: "Role (provider/ModelName)"
        assert entry.display_name == "Fast (anthropic/Claude Haiku 4.5)"


# =============================================================================
# LiteLLM Provider Tests
# =============================================================================


class TestLiteLLMProvider:
    """Tests for LiteLLM provider implementation."""

    def test_initialization(self):
        """Test provider initialization with various configurations."""
        # Basic initialization
        provider = LiteLLMProvider("claude-sonnet-4-20250514")
        assert provider.model == "claude-sonnet-4-20250514"

        # With API key
        provider = LiteLLMProvider("gpt-4o", api_key="sk-custom-key")
        assert provider.model == "gpt-4o"
        assert provider._api_key == "sk-custom-key"

        # With custom API base
        provider = LiteLLMProvider(
            "gpt-4", api_base="http://localhost:8000/v1"
        )
        assert provider._api_base == "http://localhost:8000/v1"

        # With additional kwargs
        provider = LiteLLMProvider(
            "claude-sonnet-4-20250514", timeout=120, custom_param="value"
        )
        assert provider._kwargs["timeout"] == 120
        assert provider._kwargs["custom_param"] == "value"

    def test_build_kwargs(self):
        """Test _build_kwargs method constructs correct arguments."""
        provider = LiteLLMProvider(
            "claude-sonnet-4-20250514",
            api_key="sk-test",
            api_base="http://localhost:8000",
        )

        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello!"),
        ]

        kwargs = provider._build_kwargs(
            messages=messages,
            max_tokens=1000,
            temperature=0.5,
            stop=["STOP"],
            stream=True,
        )

        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["max_tokens"] == 1000
        assert kwargs["temperature"] == 0.5
        assert kwargs["stop"] == ["STOP"]
        assert kwargs["stream"] is True
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["api_base"] == "http://localhost:8000"

        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["content"] == "You are helpful."
        assert kwargs["messages"][1]["role"] == "user"
        assert kwargs["messages"][1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful non-streaming completion."""
        provider = LiteLLMProvider("claude-sonnet-4-20250514")

        mock_response = create_mock_llm_response("This is the response!")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [Message(role=Role.USER, content="Test prompt")]
            result = await provider.complete(messages, max_tokens=100, temperature=0.7)

            assert isinstance(result, CompletionResult)
            assert result.content == "This is the response!"
            assert result.finish_reason == "stop"
            assert result.usage["prompt_tokens"] == 10
            assert result.usage["completion_tokens"] == 20
            assert result.usage["total_tokens"] == 30

            # Verify litellm was called correctly
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "claude-sonnet-4-20250514"
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["stream"] is False

    @pytest.mark.asyncio
    async def test_complete_no_usage(self):
        """Test completion when usage data is not provided."""
        provider = LiteLLMProvider("gpt-4o")

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = None  # No usage data

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [Message(role=Role.USER, content="Test")]
            result = await provider.complete(messages)

            assert result.content == "Response text"
            assert result.usage == {}

    @pytest.mark.asyncio
    async def test_stream_success(self):
        """Test successful streaming completion."""
        provider = LiteLLMProvider("claude-sonnet-4-20250514")

        # Create mock streaming response
        chunks = [
            create_mock_llm_stream_chunk("Hello", False),
            create_mock_llm_stream_chunk(" world", False),
            create_mock_llm_stream_chunk("!", True),
        ]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        mock_response = async_chunks()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [Message(role=Role.USER, content="Test")]
            stream_iter = provider.stream(messages, max_tokens=50)

            collected_chunks = []
            async for chunk in stream_iter:
                assert isinstance(chunk, StreamChunk)
                collected_chunks.append(chunk)

            assert len(collected_chunks) == 3
            assert collected_chunks[0].text == "Hello"
            assert collected_chunks[0].is_final is False
            assert collected_chunks[1].text == " world"
            assert collected_chunks[2].text == "!"
            assert collected_chunks[2].is_final is True
            assert collected_chunks[2].finish_reason == "stop"

            # Verify litellm was called with stream=True
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_empty_deltas(self):
        """Test streaming handles empty delta content gracefully."""
        provider = LiteLLMProvider("gpt-4o")

        chunks = [
            create_mock_llm_stream_chunk("Text", False),
            create_mock_llm_stream_chunk("", False),  # Empty delta
            create_mock_llm_stream_chunk("More", True),
        ]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = async_chunks()

            messages = [Message(role=Role.USER, content="Test")]
            collected = [chunk async for chunk in provider.stream(messages)]

            # All chunks should be yielded, including empty ones
            assert len(collected) == 3
            assert collected[1].text == ""

    @pytest.mark.asyncio
    async def test_complete_with_stop_sequences(self):
        """Test completion with stop sequences."""
        provider = LiteLLMProvider("claude-sonnet-4-20250514")

        mock_response = create_mock_llm_response("Response until STOP")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [Message(role=Role.USER, content="Test")]
            await provider.complete(messages, stop=["STOP", "END"])

            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["stop"] == ["STOP", "END"]

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for API failures."""
        provider = LiteLLMProvider("invalid-model")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.side_effect = Exception("API Error: Invalid model")

            messages = [Message(role=Role.USER, content="Test")]

            with pytest.raises(Exception) as exc_info:
                await provider.complete(messages)

            assert "API Error: Invalid model" in str(exc_info.value)


# =============================================================================
# Data Types Tests
# =============================================================================


class TestProviderProtocol:
    """Tests for provider protocol data types."""

    def test_role_enum(self):
        """Test Role enum values."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_message_creation(self):
        """Test Message dataclass creation and immutability."""
        msg = Message(role=Role.USER, content="Hello", actor="user")

        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.actor == "user"

        # Test immutability (frozen dataclass)
        with pytest.raises(AttributeError):
            msg.content = "Modified"  # type: ignore

    def test_message_no_actor(self):
        """Test Message with no actor (legacy)."""
        msg = Message(role=Role.ASSISTANT, content="Response")

        assert msg.actor is None

    def test_stream_chunk(self):
        """Test StreamChunk dataclass."""
        # Non-final chunk
        chunk = StreamChunk(text="Hello")
        assert chunk.text == "Hello"
        assert chunk.is_final is False
        assert chunk.finish_reason is None

        # Final chunk
        chunk = StreamChunk(text="Done", is_final=True, finish_reason="stop")
        assert chunk.is_final is True
        assert chunk.finish_reason == "stop"

    def test_completion_result(self):
        """Test CompletionResult dataclass."""
        result = CompletionResult(
            content="Response text",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        assert result.content == "Response text"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10

        # Test default usage
        result = CompletionResult(content="Text")
        assert result.usage == {}

    def test_message_equality(self):
        """Test Message equality comparison."""
        msg1 = Message(role=Role.USER, content="Test", actor="user")
        msg2 = Message(role=Role.USER, content="Test", actor="user")
        msg3 = Message(role=Role.USER, content="Different", actor="user")

        assert msg1 == msg2
        assert msg1 != msg3


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_provider_defaults(self):
        """Test create_provider with default model."""
        provider = create_provider()

        assert isinstance(provider, LiteLLMProvider)
        assert provider.model == "claude-sonnet-4-20250514"

    def test_create_provider_custom_model(self):
        """Test create_provider with custom model."""
        provider = create_provider(model="gpt-4o")

        assert provider.model == "gpt-4o"

    def test_create_provider_with_kwargs(self):
        """Test create_provider with additional kwargs."""
        provider = create_provider(
            model="claude-opus-4-5-20251101",
            api_key="sk-custom",
            timeout=60,
        )

        assert provider.model == "claude-opus-4-5-20251101"
        assert provider._api_key == "sk-custom"
        assert provider._kwargs["timeout"] == 60
