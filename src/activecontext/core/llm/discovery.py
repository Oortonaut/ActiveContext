"""LLM model discovery based on environment variables.

Detects which LLM providers are available by checking for API keys
in the environment, and returns curated model lists for each.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an available model."""

    model_id: str
    name: str
    provider: str
    description: str | None = None


# Provider configurations: (env_var, curated_models)
# Curated models are (model_id, display_name, description)
# Updated January 2026
PROVIDER_CONFIGS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "anthropic": (
        "ANTHROPIC_API_KEY",
        [
            ("claude-opus-4-5-20251101", "Claude Opus 4.5", "Most capable, hybrid reasoning"),
            ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "Best for coding & agents"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "Fast, cost-efficient"),
        ],
    ),
    "openai": (
        "OPENAI_API_KEY",
        [
            ("gpt-5.2", "GPT-5.2", "Most capable, reasoning"),
            ("gpt-5.2-codex", "GPT-5.2 Codex", "Optimized for agentic coding"),
            ("gpt-4.1", "GPT-4.1", "Flagship, 1M context"),
            ("gpt-4.1-mini", "GPT-4.1 Mini", "Fast and affordable"),
            ("gpt-4.1-nano", "GPT-4.1 Nano", "Fastest, cheapest"),
        ],
    ),
    "groq": (
        "GROQ_API_KEY",
        [
            ("groq/llama-4-maverick-17b-128e-instruct", "Llama 4 Maverick", "Latest multimodal"),
            ("groq/llama-4-scout-17b-16e-instruct", "Llama 4 Scout", "Fast multimodal"),
            ("groq/llama-3.3-70b-versatile", "Llama 3.3 70B", "Versatile, tool use"),
        ],
    ),
    "deepseek": (
        "DEEPSEEK_API_KEY",
        [
            ("deepseek/deepseek-chat", "DeepSeek V3.2", "General purpose, tool use"),
            ("deepseek/deepseek-reasoner", "DeepSeek R1", "Chain-of-thought reasoning"),
        ],
    ),
    "mistral": (
        "MISTRAL_API_KEY",
        [
            ("mistral/mistral-large-latest", "Mistral Large 3", "41B params, 256K context"),
            ("mistral/codestral-latest", "Codestral", "Code completion specialist"),
            ("mistral/magistral-medium-2506", "Magistral Medium", "Reasoning specialist"),
        ],
    ),
    "gemini": (
        "GEMINI_API_KEY",
        [
            ("gemini/gemini-3-flash", "Gemini 3 Flash", "Latest multimodal"),
            ("gemini/gemini-3-pro", "Gemini 3 Pro", "Reasoning-first, agentic"),
            ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", "Deep Think, long context"),
        ],
    ),
    "openrouter": (
        "OPENROUTER_API_KEY",
        [
            ("openrouter/auto", "OpenRouter Auto", "Auto-selects best model"),
        ],
    ),
}


def get_available_providers() -> list[str]:
    """Return list of providers with API keys configured."""
    available = []
    for provider, (env_var, _) in PROVIDER_CONFIGS.items():
        if os.environ.get(env_var):
            available.append(provider)
    return available


def get_available_models() -> list[ModelInfo]:
    """Return models for all providers with API keys configured."""
    models: list[ModelInfo] = []
    for provider, (env_var, model_list) in PROVIDER_CONFIGS.items():
        if os.environ.get(env_var):
            for model_id, name, description in model_list:
                models.append(
                    ModelInfo(
                        model_id=model_id,
                        name=name,
                        provider=provider,
                        description=description,
                    )
                )
    return models


def get_default_model() -> str | None:
    """Return the default model ID based on available providers.

    Priority: Anthropic > OpenAI > others
    """
    providers = get_available_providers()
    if not providers:
        return None

    # Priority order
    priority = ["anthropic", "openai", "groq", "deepseek", "mistral", "gemini"]
    for p in priority:
        if p in providers:
            _, model_list = PROVIDER_CONFIGS[p]
            if model_list:
                return model_list[0][0]  # First model's ID

    # Fallback to first available
    first_provider = providers[0]
    _, model_list = PROVIDER_CONFIGS[first_provider]
    return model_list[0][0] if model_list else None
