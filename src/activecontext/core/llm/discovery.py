"""LLM model discovery based on environment variables and roles.

Detects which LLM providers are available by checking for API keys
in the environment, and provides role-based model selection.

Roles represent use-case categories (coding, thinking, fast, etc.)
that map to the best model for each provider.
"""

from __future__ import annotations

from dataclasses import dataclass

from activecontext.config.secrets import fetch_secret
from activecontext.core.llm.providers import (
    DEFAULT_ROLES,
    PROVIDER_CONFIGS,
    ROLE_DESCRIPTIONS,
    ROLE_MODEL_DEFAULTS,
    ModelConfig,
    ProviderConfig,
)


@dataclass
class ModelInfo:
    """Information about an available model."""

    model_id: str
    name: str
    provider: str
    context_length: int
    description: str | None = None
    capabilities: list[str] | None = None


@dataclass
class RoleModelEntry:
    """A model assigned to a role for a provider."""

    role: str  # "coding", "fast", etc.
    provider: str  # "anthropic", "openai", etc.
    model_id: str  # "claude-sonnet-4-5-20250929"
    display_name: str  # "anthropic/Claude Sonnet 4.5"
    context_length: int  # 200000
    description: str | None = None  # "Code generation... - Best for coding & agents"
    capabilities: list[str] | None = None  # ["tool_use", "image_input", "chat"]


def _get_provider_api_key(config: ProviderConfig) -> str | None:
    """Get API key for a provider using fetch_secret.

    Priority:
    1. Direct api_key in ProviderConfig (for programmatic override)
    2. fetch_secret() (.env file > os.environ)
    """
    if config.api_key:
        return config.api_key
    return fetch_secret(config.env_var)


def _get_model_config(model_id: str, provider: str) -> ModelConfig | None:
    """Get ModelConfig for a model ID."""
    if provider in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[provider]
        for model in config.models:
            if model.id == model_id:
                return model
    return None


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
            for model in config.models:
                models.append(
                    ModelInfo(
                        model_id=model.id,
                        name=model.name,
                        provider=provider,
                        context_length=model.context_length,
                        description=model.description,
                        capabilities=model.capabilities if model.capabilities else None,
                    )
                )
    return models


def get_provider_config(provider: str) -> ProviderConfig | None:
    """Get configuration for a provider."""
    return PROVIDER_CONFIGS.get(provider)


def _sort_key(entry: RoleModelEntry) -> tuple[int, str]:
    """Sort key: descending context_length, then ascending display_name."""
    return (-entry.context_length, entry.display_name)


def get_role_models(role: str) -> list[RoleModelEntry]:
    """Get all available models for a role across providers with API keys.

    Args:
        role: The role name (e.g., "coding", "fast", "thinking")

    Returns:
        List of RoleModelEntry for each available provider that supports this role.
        Ordered by context_length (descending), then display_name (ascending).

    Example:
        >>> get_role_models("fast")
        [RoleModelEntry("fast", "gemini", "gemini/gemini-3-flash", "gemini/Gemini 3 Flash", 1000000, ...),
         RoleModelEntry("fast", "anthropic", "claude-haiku-4-5", "anthropic/Claude Haiku 4.5", 200000, ...)]
    """
    available_providers = get_available_providers()
    entries: list[RoleModelEntry] = []

    role_desc = ROLE_DESCRIPTIONS.get(role, "")

    for provider in available_providers:
        key = (role, provider)
        if key not in ROLE_MODEL_DEFAULTS:
            continue

        model_id = ROLE_MODEL_DEFAULTS[key]
        model_config = _get_model_config(model_id, provider)
        if not model_config:
            continue

        display_name = f"{provider}/{model_config.name}"

        # Build description: role description + model description
        description: str | None
        if role_desc and model_config.description:
            description = f"{role_desc} - {model_config.description}"
        elif model_config.description:
            description = model_config.description
        else:
            description = role_desc or None

        entries.append(
            RoleModelEntry(
                role=role,
                provider=provider,
                model_id=model_id,
                display_name=display_name,
                context_length=model_config.context_length,
                description=description,
                capabilities=model_config.capabilities if model_config.capabilities else None,
            )
        )

    # Sort by context_length descending, then display_name ascending
    entries.sort(key=_sort_key)
    return entries


def get_all_role_models() -> dict[str, list[RoleModelEntry]]:
    """Get all roles with their available models.

    Returns:
        Dict mapping role names to lists of RoleModelEntry.
        Only includes roles that have at least one available model.

    Example:
        >>> get_all_role_models()
        {"coding": [...], "fast": [...], "thinking": [...], ...}
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
    1. Check config.llm.role_providers for user's saved choice (if no provider specified)
       - If entry has model override, use it directly
       - Otherwise lookup provider in ROLE_MODEL_DEFAULTS
    2. If provider specified, use that provider
    3. Use first model from get_role_models (sorted by context_length, then name)

    Args:
        role: The role name (e.g., "coding", "fast")
        provider: Optional provider to use. If None, uses sorted order.

    Returns:
        Model ID string, or None if no model available for this role.
    """
    # Check config for saved role preferences (only when provider not explicitly specified)
    if provider is None:
        try:
            from activecontext.config import get_config

            config = get_config()
            if config.llm.role_providers:
                for rp in config.llm.role_providers:
                    if rp.role == role:
                        # Verify the provider is available
                        if rp.provider in get_available_providers():
                            # Use model override if specified, else lookup default
                            if rp.model:
                                return rp.model
                            key = (role, rp.provider)
                            return ROLE_MODEL_DEFAULTS.get(key)
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

    # Use first model from sorted list (highest context length)
    models = get_role_models(role)
    if models:
        return models[0].model_id

    return None


def get_default_model() -> str | None:
    """Return the default model ID based on config and role system.

    Selection priority:
    1. config.llm.role + config.llm.provider → get_model_for_role()
    2. "coding" role (best for agent tasks)
    3. "thinking" role (deep reasoning)
    4. First model from first available provider (sorted by context_length)

    Note: Model is always derived from role/provider, not stored directly.
    """
    # Check config first
    try:
        from activecontext.config import get_config

        config = get_config()

        # Use saved role/provider if configured
        if config.llm.role:
            model = get_model_for_role(config.llm.role, config.llm.provider)
            if model:
                return model

    except ImportError:
        pass  # Config not available, fall through

    # Try "coding" role first (best for agent tasks)
    model = get_model_for_role("coding")
    if model:
        return model

    # Try "thinking" role (deep reasoning)
    model = get_model_for_role("thinking")
    if model:
        return model

    # Ultimate fallback: first model from any available provider
    available = get_available_providers()
    if not available:
        return None

    # Collect all models and sort by context_length
    all_models: list[tuple[int, str, str]] = []  # (context_length, name, model_id)
    for provider in available:
        provider_config = PROVIDER_CONFIGS[provider]
        for model_config in provider_config.models:
            all_models.append((model_config.context_length, model_config.name, model_config.id))

    if all_models:
        # Sort by context_length descending, then name ascending
        all_models.sort(key=lambda x: (-x[0], x[1]))
        return all_models[0][2]

    return None