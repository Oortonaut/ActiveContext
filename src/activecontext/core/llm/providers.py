"""LLM provider configurations.

This file contains the provider definitions and role-to-model mappings.
It can be managed separately in source control for easier updates.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    env_var: str  # Environment variable name for API key (e.g., "ANTHROPIC_API_KEY")
    models: list[tuple[str, str, str]]  # (model_id, display_name, description)
    api_key: str | None = None  # Direct API key (overrides env var)
    auth_prefix: str | None = None  # Auth prefix (e.g., "Bearer", "X-API-Key"). None = provider default


# Provider configurations
# Updated January 2026
PROVIDER_CONFIGS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig(
        env_var="ANTHROPIC_API_KEY",
        models=[
            ("claude-opus-4-5-20251101", "Claude Opus 4.5", "Most capable, hybrid reasoning"),
            ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "Best for coding & agents"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "Fast, cost-efficient"),
        ],
    ),
    "openai": ProviderConfig(
        env_var="OPENAI_API_KEY",
        models=[
            ("gpt-5.2", "GPT-5.2", "Most capable, reasoning"),
            ("gpt-5.2-codex", "GPT-5.2 Codex", "Optimized for agentic coding"),
            ("gpt-4.1", "GPT-4.1", "Flagship, 1M context"),
            ("gpt-4.1-mini", "GPT-4.1 Mini", "Fast and affordable"),
            ("gpt-4.1-nano", "GPT-4.1 Nano", "Fastest, cheapest"),
        ],
    ),
    "groq": ProviderConfig(
        env_var="GROQ_API_KEY",
        models=[
            ("groq/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick", "Latest multimodal"),
            ("groq/llama-4-scout-17b-16e-instruct", "Llama 4 Scout", "Fast multimodal"),
            ("groq/llama-3.3-70b-versatile", "Llama 3.3 70B", "Versatile, tool use"),
        ],
    ),
    "deepseek": ProviderConfig(
        env_var="DEEPSEEK_API_KEY",
        models=[
            ("deepseek/deepseek-chat", "DeepSeek V3.2", "General purpose, tool use"),
            ("deepseek/deepseek-reasoner", "DeepSeek R1", "Chain-of-thought reasoning"),
        ],
    ),
    "mistral": ProviderConfig(
        env_var="MISTRAL_API_KEY",
        models=[
            ("mistral/mistral-large-latest", "Mistral Large 3", "41B params, 256K context"),
            ("mistral/codestral-latest", "Codestral", "Code completion specialist"),
            ("mistral/magistral-medium-2506", "Magistral Medium", "Reasoning specialist"),
        ],
    ),
    "gemini": ProviderConfig(
        env_var="GEMINI_API_KEY",
        models=[
            ("gemini/gemini-3-flash", "Gemini 3 Flash", "Latest multimodal"),
            ("gemini/gemini-3-pro", "Gemini 3 Pro", "Reasoning-first, agentic"),
            ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", "Deep Think, long context"),
        ],
    ),
    "openrouter": ProviderConfig(
        env_var="OPENROUTER_API_KEY",
        models=[
            ("openrouter/auto", "OpenRouter Auto", "Auto-selects best model"),
        ],
    ),
}

# Provider priority order for automatic selection
PROVIDER_PRIORITY = ["anthropic", "openai", "groq", "deepseek", "mistral", "gemini", "openrouter"]

# Default roles available in the system
DEFAULT_ROLES = ["coding", "reasoning", "writing", "balanced", "fast", "cheap"]

# Role to model mapping: (role, provider) → model_id
# Maps each role to the best model for that use case per provider
ROLE_MODEL_DEFAULTS: dict[tuple[str, str], str] = {
    # Anthropic
    ("coding", "anthropic"): "claude-sonnet-4-5-20250929",
    ("reasoning", "anthropic"): "claude-opus-4-5-20251101",
    ("writing", "anthropic"): "claude-opus-4-5-20251101",
    ("balanced", "anthropic"): "claude-sonnet-4-5-20250929",
    ("fast", "anthropic"): "claude-haiku-4-5-20251001",
    ("cheap", "anthropic"): "claude-haiku-4-5-20251001",
    # OpenAI
    ("coding", "openai"): "gpt-5.2-codex",
    ("reasoning", "openai"): "gpt-5.2",
    ("writing", "openai"): "gpt-5.2",
    ("balanced", "openai"): "gpt-4.1",
    ("fast", "openai"): "gpt-4.1-mini",
    ("cheap", "openai"): "gpt-4.1-nano",
    # Groq
    ("coding", "groq"): "groq/llama-4-maverick-17b-128e-instruct",
    ("reasoning", "groq"): "groq/llama-4-maverick-17b-128e-instruct",
    ("writing", "groq"): "groq/llama-4-maverick-17b-128e-instruct",
    ("balanced", "groq"): "groq/llama-3.3-70b-versatile",
    ("fast", "groq"): "groq/llama-4-scout-17b-16e-instruct",
    ("cheap", "groq"): "groq/llama-4-scout-17b-16e-instruct",
    # DeepSeek
    ("coding", "deepseek"): "deepseek/deepseek-chat",
    ("reasoning", "deepseek"): "deepseek/deepseek-reasoner",
    ("writing", "deepseek"): "deepseek/deepseek-chat",
    ("balanced", "deepseek"): "deepseek/deepseek-chat",
    ("fast", "deepseek"): "deepseek/deepseek-chat",
    ("cheap", "deepseek"): "deepseek/deepseek-chat",
    # Mistral
    ("coding", "mistral"): "mistral/codestral-latest",
    ("reasoning", "mistral"): "mistral/magistral-medium-2506",
    ("writing", "mistral"): "mistral/mistral-large-latest",
    ("balanced", "mistral"): "mistral/mistral-large-latest",
    ("fast", "mistral"): "mistral/mistral-large-latest",
    ("cheap", "mistral"): "mistral/codestral-latest",
    # Gemini
    ("coding", "gemini"): "gemini/gemini-3-pro",
    ("reasoning", "gemini"): "gemini/gemini-3-pro",
    ("writing", "gemini"): "gemini/gemini-2.5-pro",
    ("balanced", "gemini"): "gemini/gemini-2.5-pro",
    ("fast", "gemini"): "gemini/gemini-3-flash",
    ("cheap", "gemini"): "gemini/gemini-3-flash",
    # OpenRouter (uses auto for all roles)
    ("coding", "openrouter"): "openrouter/auto",
    ("reasoning", "openrouter"): "openrouter/auto",
    ("writing", "openrouter"): "openrouter/auto",
    ("balanced", "openrouter"): "openrouter/auto",
    ("fast", "openrouter"): "openrouter/auto",
    ("cheap", "openrouter"): "openrouter/auto",
}
