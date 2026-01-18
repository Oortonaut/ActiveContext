"""LLM provider configurations.

Loads provider definitions and role-to-model mappings from providers.yaml.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    id: str
    name: str
    description: str
    context_length: int
    capabilities: list[str] = field(default_factory=list)  # tool_use, image_input, image_generation, chat, thinking


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    env_var: str
    models: list[ModelConfig] = field(default_factory=list)
    role_models: dict[str, str] = field(default_factory=dict)
    api_key: str | None = None
    auth_prefix: str | None = None


@dataclass
class RoleConfig:
    """Configuration for a role."""

    name: str
    description: str


@lru_cache(maxsize=1)
def _load_providers_yaml() -> dict[str, Any]:
    """Load providers.yaml from package resources."""
    files = importlib.resources.files("activecontext.core.llm")
    yaml_path = files.joinpath("providers.yaml")
    with importlib.resources.as_file(yaml_path) as path:
        with open(path) as f:
            return yaml.safe_load(f)


def _build_provider_configs() -> dict[str, ProviderConfig]:
    """Build ProviderConfig objects from YAML data."""
    data = _load_providers_yaml()
    configs: dict[str, ProviderConfig] = {}

    for provider_name, provider_data in data.get("providers", {}).items():
        models = [
            ModelConfig(
                id=m["id"],
                name=m["name"],
                description=m["description"],
                context_length=m["context_length"],
                capabilities=m.get("capabilities", []),
            )
            for m in provider_data.get("models", [])
        ]
        configs[provider_name] = ProviderConfig(
            env_var=provider_data["env_var"],
            models=models,
            role_models=provider_data.get("role_models", {}),
        )

    return configs


def _build_role_configs() -> dict[str, RoleConfig]:
    """Build RoleConfig objects from YAML data."""
    data = _load_providers_yaml()
    configs: dict[str, RoleConfig] = {}

    for role_name, role_data in data.get("roles", {}).items():
        configs[role_name] = RoleConfig(
            name=role_name,
            description=role_data.get("description", ""),
        )

    return configs


def _build_role_model_defaults() -> dict[tuple[str, str], str]:
    """Build (role, provider) -> model_id mapping from YAML data."""
    data = _load_providers_yaml()
    defaults: dict[tuple[str, str], str] = {}

    for provider_name, provider_data in data.get("providers", {}).items():
        for role, model_id in provider_data.get("role_models", {}).items():
            defaults[(role, provider_name)] = model_id

    return defaults


# Exported constants - loaded lazily on first access
PROVIDER_CONFIGS: dict[str, ProviderConfig] = _build_provider_configs()
ROLE_CONFIGS: dict[str, RoleConfig] = _build_role_configs()
ROLE_MODEL_DEFAULTS: dict[tuple[str, str], str] = _build_role_model_defaults()
DEFAULT_ROLES: list[str] = list(ROLE_CONFIGS.keys())
ROLE_DESCRIPTIONS: dict[str, str] = {r.name: r.description for r in ROLE_CONFIGS.values()}