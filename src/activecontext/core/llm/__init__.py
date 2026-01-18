"""LLM provider abstraction."""

from activecontext.core.llm.discovery import (
    ModelInfo,
    RoleModelEntry,
    get_all_role_models,
    get_available_models,
    get_available_providers,
    get_default_model,
    get_model_for_role,
    get_provider_config,
    get_role_models,
)
from activecontext.core.llm.litellm_provider import LiteLLMProvider
from activecontext.core.llm.provider import LLMProvider, Message, Role
from activecontext.core.llm.providers import (
    DEFAULT_ROLES,
    PROVIDER_CONFIGS,
    ROLE_CONFIGS,
    ROLE_DESCRIPTIONS,
    ROLE_MODEL_DEFAULTS,
    ModelConfig,
    ProviderConfig,
    RoleConfig,
)

__all__ = [
    # Provider protocol and implementations
    "LLMProvider",
    "LiteLLMProvider",
    "Message",
    "Role",
    # Model info types
    "ModelInfo",
    "RoleModelEntry",
    "ProviderConfig",
    # Discovery functions
    "get_available_models",
    "get_available_providers",
    "get_default_model",
    "get_provider_config",
    # Role-based selection
    "get_role_models",
    "get_all_role_models",
    "get_model_for_role",
    # Constants
    "DEFAULT_ROLES",
    "PROVIDER_CONFIGS",
    "ROLE_CONFIGS",
    "ROLE_DESCRIPTIONS",
    "ROLE_MODEL_DEFAULTS",
    # Config types
    "ModelConfig",
    "RoleConfig",
]
