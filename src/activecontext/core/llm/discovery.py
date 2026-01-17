"""LLM model discovery based on environment variables and roles.

Detects which LLM providers are available by checking for API keys
in the environment, and provides role-based model selection.

Roles represent use-case categories (coding, reasoning, fast, etc.)
that map to the best model for each provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from activecontext.core.llm.providers import (
    DEFAULT_ROLES,
    PROVIDER_CONFIGS,
    PROVIDER_PRIORITY,
    ROLE_MODEL_DEFAULTS,
    ProviderConfig,
)


@dataclass
class ModelInfo:
    """Information about an available model."""

    model_id: str
    name: str
    provider: str
    description: str | None = None


@dataclass
class RoleModelEntry:
    """A model assigned to a role for a provider."""

    role: str  # "coding", "fast", etc.
    provider: str  # "anthropic", "openai", etc.
    model_id: str  # "claude-sonnet-4-5-20250929"
    display_name: str  # "Coding (anthropic/claude-sonnet-4.5)"


def _get_provider_api_key(config: ProviderConfig) -> str | None:
    """Get API key for a provider, checking direct key then env var."""
    if config.api_key:
        return config.api_key
    return os.environ.get(config.env_var)


def _get_model_display_name(model_id: str, provider: str) -> str:
    """Get short display name for a model."""
    if provider in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[provider]
        for mid, name, _ in config.models:
            if mid == model_id:
                return name
    return model_id


def get_available_providers() -> list[str]:
    """Return list of providers with API keys configured."""
    available = []
    for provider, config in PROVIDER_CONFIGS.items():
        if _get_provider_api_key(config):
            available.append(provider)
    return available


def get_available_models() -> list[ModelInfo]:
    """Return models for all providers with API keys configured."""
    models: list[ModelInfo] = []
    for provider, config in PROVIDER_CONFIGS.items():
        if _get_provider_api_key(config):
            for model_id, name, description in config.models:
                models.append(
                    ModelInfo(
                        model_id=model_id,
                        name=name,
                        provider=provider,
                        description=description,
                    )
                )
    return models


def get_provider_config(provider: str) -> ProviderConfig | None:
    """Get configuration for a provider."""
    return PROVIDER_CONFIGS.get(provider)


def get_role_models(role: str) -> list[RoleModelEntry]:
    """Get all available models for a role across providers with API keys.

    Args:
        role: The role name (e.g., "coding", "fast", "reasoning")

    Returns:
        List of RoleModelEntry for each available provider that supports this role.
        Ordered by provider priority.

    Example:
        >>> get_role_models("fast")
        [RoleModelEntry("fast", "anthropic", "claude-haiku-4-5", "Fast (anthropic/Claude Haiku 4.5)"),
         RoleModelEntry("fast", "openai", "gpt-4.1-mini", "Fast (openai/GPT-4.1 Mini)")]
    """
    available_providers = get_available_providers()
    entries: list[RoleModelEntry] = []

    for provider in PROVIDER_PRIORITY:
        if provider not in available_providers:
            continue

        key = (role, provider)
        if key in ROLE_MODEL_DEFAULTS:
            model_id = ROLE_MODEL_DEFAULTS[key]
            model_name = _get_model_display_name(model_id, provider)
            display_name = f"{role.capitalize()} ({provider}/{model_name})"
            entries.append(
                RoleModelEntry(
                    role=role,
                    provider=provider,
                    model_id=model_id,
                    display_name=display_name,
                )
            )

    return entries


def get_all_role_models() -> dict[str, list[RoleModelEntry]]:
    """Get all roles with their available models.

    Returns:
        Dict mapping role names to lists of RoleModelEntry.
        Only includes roles that have at least one available model.

    Example:
        >>> get_all_role_models()
        {"coding": [...], "fast": [...], "reasoning": [...], ...}
    """
    result: dict[str, list[RoleModelEntry]] = {}
    for role in DEFAULT_ROLES:
        models = get_role_models(role)
        if models:
            result[role] = models
    return result


def get_model_for_role(role: str, provider: str | None = None) -> str | None:
    """Get the best model for a role, optionally from a specific provider.

    Selection priority:
    1. Check config.llm.role_models for user's saved choice (if no provider specified)
    2. Use specified provider or first available by priority
    3. Lookup in ROLE_MODEL_DEFAULTS

    Args:
        role: The role name (e.g., "coding", "fast")
        provider: Optional provider to use. If None, uses priority order.

    Returns:
        Model ID string, or None if no model available for this role.
    """
    # Check config for saved role preferences (only when provider not explicitly specified)
    if provider is None:
        try:
            from activecontext.config import get_config

            config = get_config()
            if config.llm.role_models:
                for rm in config.llm.role_models:
                    if rm.role == role:
                        # Verify the provider is available
                        if rm.provider in get_available_providers():
                            return rm.model_id
        except ImportError:
            pass  # Config not available

    # Get available providers
    available = get_available_providers()
    if not available:
        return None

    # If provider specified, use it directly
    if provider:
        if provider not in available:
            return None
        key = (role, provider)
        return ROLE_MODEL_DEFAULTS.get(key)

    # Find first available provider with this role
    for p in PROVIDER_PRIORITY:
        if p in available:
            key = (role, p)
            if key in ROLE_MODEL_DEFAULTS:
                return ROLE_MODEL_DEFAULTS[key]

    return None


def get_default_model() -> str | None:
    """Return the default model ID based on config and role system.

    Selection priority:
    1. config.llm.model (explicit model override)
    2. config.llm.default_role → get_model_for_role()
    3. "balanced" role with priority provider
    """
    # Check config first
    try:
        from activecontext.config import get_config

        config = get_config()

        # Explicit model takes highest priority
        if config.llm.model:
            return config.llm.model

        # Use default_role if configured
        if config.llm.default_role:
            model = get_model_for_role(config.llm.default_role)
            if model:
                return model

    except ImportError:
        pass  # Config not available, fall through

    # Fall back to "balanced" role
    model = get_model_for_role("balanced")
    if model:
        return model

    # Ultimate fallback: first model from first available provider
    providers = get_available_providers()
    if not providers:
        return None

    for p in PROVIDER_PRIORITY:
        if p in providers:
            provider_config = PROVIDER_CONFIGS[p]
            if provider_config.models:
                return provider_config.models[0][0]

    return None
