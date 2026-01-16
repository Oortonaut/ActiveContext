"""Configuration file loading and caching.

Handles:
- YAML file parsing
- Environment variable overrides
- Config caching with reload support
- Conversion from dict to typed Config dataclass
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from activecontext.config.merge import merge_configs
from activecontext.config.paths import get_config_paths
from activecontext.config.schema import (
    Config,
    LLMConfig,
    LoggingConfig,
    ProjectionConfig,
    SessionConfig,
    SessionModeConfig,
)

# Module logger (may not be configured yet at import time)
_log = logging.getLogger("activecontext.config")

# Global cached config
_cached_config: Config | None = None

# Callbacks to notify on config reload
_reload_callbacks: list[Callable[[Config], None]] = []


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found or invalid.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML as dict, or empty dict on error.
    """
    if not path.exists():
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except yaml.YAMLError as e:
        _log.warning("Invalid YAML in %s: %s", path, e)
        return {}
    except PermissionError:
        _log.debug("Permission denied reading %s", path)
        return {}
    except OSError as e:
        _log.warning("Error reading %s: %s", path, e)
        return {}


def env_overrides() -> dict[str, Any]:
    """Build config dict from environment variables.

    Environment variables take highest priority for backward compatibility.

    Returns:
        Config dict with values from environment.
    """
    overrides: dict[str, Any] = {}

    # API keys - check all providers
    api_key_env_vars = [
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("GROQ_API_KEY", "groq"),
        ("DEEPSEEK_API_KEY", "deepseek"),
        ("MISTRAL_API_KEY", "mistral"),
        ("GEMINI_API_KEY", "gemini"),
        ("OPENROUTER_API_KEY", "openrouter"),
    ]

    for env_var, provider in api_key_env_vars:
        api_key = os.environ.get(env_var)
        if api_key:
            if "llm" not in overrides:
                overrides["llm"] = {}
            # Only set if not already set by a higher-priority env var
            if overrides["llm"].get("api_key") is None:
                overrides["llm"]["api_key"] = api_key
            if overrides["llm"].get("provider") is None:
                overrides["llm"]["provider"] = provider
            break  # First found wins for default provider

    # Logging from ACTIVECONTEXT_LOG
    log_path = os.environ.get("ACTIVECONTEXT_LOG")
    if log_path:
        if "logging" not in overrides:
            overrides["logging"] = {}
        overrides["logging"]["file"] = log_path

    return overrides


def dict_to_config(data: dict[str, Any]) -> Config:
    """Convert merged dict to typed Config dataclass.

    Args:
        data: Merged configuration dictionary.

    Returns:
        Typed Config object.
    """
    # LLM config
    llm_data = data.get("llm", {})
    llm = LLMConfig(
        provider=llm_data.get("provider"),
        model=llm_data.get("model"),
        api_key=llm_data.get("api_key"),
        api_base=llm_data.get("api_base"),
        temperature=llm_data.get("temperature"),
        max_tokens=llm_data.get("max_tokens"),
    )

    # Session config
    session_data = data.get("session", {})
    modes_data = session_data.get("modes", [])
    modes = [
        SessionModeConfig(
            id=m.get("id", ""),
            name=m.get("name", ""),
            description=m.get("description", ""),
        )
        for m in modes_data
        if isinstance(m, dict)
    ]
    session = SessionConfig(
        modes=modes,
        default_mode=session_data.get("default_mode"),
    )

    # Projection config
    proj_data = data.get("projection", {})
    projection = ProjectionConfig(
        total_budget=proj_data.get("total_budget"),
        conversation_ratio=proj_data.get("conversation_ratio"),
        views_ratio=proj_data.get("views_ratio"),
        groups_ratio=proj_data.get("groups_ratio"),
    )

    # Logging config
    log_data = data.get("logging", {})
    logging_config = LoggingConfig(
        level=log_data.get("level"),
        file=log_data.get("file"),
    )

    # Extra fields for extensibility
    known_keys = {"llm", "session", "projection", "logging"}
    extra = {k: v for k, v in data.items() if k not in known_keys}

    return Config(
        llm=llm,
        session=session,
        projection=projection,
        logging=logging_config,
        extra=extra,
    )


def load_config(session_root: str | None = None, reload: bool = False) -> Config:
    """Load and merge config from all sources.

    Priority order (highest to lowest):
    1. Environment variables (backward compat)
    2. Project config ($session_root/.ac/config.yaml)
    3. User config (~/.ac/config.yaml or %APPDATA%)
    4. System config (/etc/activecontext/ or %PROGRAMDATA%)

    Args:
        session_root: Project directory for project-level config.
        reload: Force reload even if cached.

    Returns:
        Merged Config object.
    """
    global _cached_config

    # Return cached global config if available and not reloading
    if _cached_config is not None and not reload and session_root is None:
        return _cached_config

    configs: list[dict[str, Any]] = []

    # Load file configs in order (system -> user -> project)
    for path in get_config_paths(session_root):
        config_data = load_yaml_file(path)
        if config_data:
            _log.debug("Loaded config from %s", path)
            configs.append(config_data)

    # Environment overrides (highest priority)
    env_config = env_overrides()
    if env_config:
        configs.append(env_config)

    # Merge all configs
    merged = merge_configs(*configs)

    # Convert to typed Config
    config = dict_to_config(merged)

    # Cache only global config (no session_root)
    if session_root is None:
        _cached_config = config

    return config


def get_config() -> Config:
    """Get the cached global config.

    Loads config if not yet loaded.

    Returns:
        The global Config object.
    """
    global _cached_config
    if _cached_config is None:
        return load_config()
    return _cached_config


def reset_config() -> None:
    """Reset cached config.

    Useful for testing or forcing a reload.
    """
    global _cached_config
    _cached_config = None


def reload_config(session_root: str | None = None) -> Config:
    """Reload config from files and notify callbacks.

    Args:
        session_root: Optional project directory.

    Returns:
        The newly loaded Config.
    """
    config = load_config(session_root=session_root, reload=True)

    # Notify registered callbacks
    for callback in _reload_callbacks:
        try:
            callback(config)
        except Exception as e:
            _log.warning("Config reload callback error: %s", e)

    return config


def on_config_reload(callback: Callable[[Config], None]) -> Callable[[], None]:
    """Register a callback to be called when config is reloaded.

    Args:
        callback: Function to call with the new Config.

    Returns:
        A function to unregister the callback.
    """
    _reload_callbacks.append(callback)

    def unregister() -> None:
        if callback in _reload_callbacks:
            _reload_callbacks.remove(callback)

    return unregister
