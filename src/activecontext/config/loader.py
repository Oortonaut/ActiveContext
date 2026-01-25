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
    FilePermissionConfig,
    ImportConfig,
    LLMConfig,
    LoggingConfig,
    MCPConfig,
    MCPConnectMode,
    MCPServerConfig,
    ProjectionConfig,
    RoleProviderConfig,
    SandboxConfig,
    SessionConfig,
    SessionModeConfig,
    ShellPermissionConfig,
    StartupConfig,
    UserConfig,
    WebsitePermissionConfig,
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
    Note: API keys are NOT loaded here - use fetch_secret() for secrets.

    Returns:
        Config dict with values from environment.
    """
    overrides: dict[str, Any] = {}

    # Logging from AC_LOG
    log_path = os.environ.get("AC_LOG")
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
    role_providers_data = llm_data.get("role_providers", [])
    role_providers = [
        RoleProviderConfig(
            role=rp.get("role", ""),
            provider=rp.get("provider", ""),
            model=rp.get("model"),
        )
        for rp in role_providers_data
        if isinstance(rp, dict) and rp.get("role") and rp.get("provider")
    ]
    llm = LLMConfig(
        role=llm_data.get("role"),
        provider=llm_data.get("provider"),
        role_providers=role_providers,
        api_base=llm_data.get("api_base"),
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

    # Startup config
    startup_data = session_data.get("startup", {})
    startup_statements = startup_data.get("statements", [])
    startup_additional = startup_data.get("additional", [])
    startup = StartupConfig(
        statements=[s for s in startup_statements if isinstance(s, str)],
        additional=[s for s in startup_additional if isinstance(s, str)],
        skip_default_context=startup_data.get("skip_default_context", False),
    )

    session = SessionConfig(
        modes=modes,
        default_mode=session_data.get("default_mode"),
        startup=startup,
    )

    # Projection config (budget removed - agent manages via node visibility)
    projection = ProjectionConfig()

    # Logging config
    log_data = data.get("logging", {})
    logging_config = LoggingConfig(
        level=log_data.get("level"),
        file=log_data.get("file"),
    )

    # Sandbox config
    sandbox_data = data.get("sandbox", {})
    perms_data = sandbox_data.get("file_permissions", [])
    file_permissions = [
        FilePermissionConfig(
            pattern=p.get("pattern", ""),
            mode=p.get("mode", "read"),
        )
        for p in perms_data
        if isinstance(p, dict) and p.get("pattern")
    ]

    # Import whitelist config
    imports_data = sandbox_data.get("imports", {})
    imports = ImportConfig(
        allowed_modules=imports_data.get("allowed_modules", []),
        allow_submodules=imports_data.get("allow_submodules", True),
        allow_all=imports_data.get("allow_all", False),
    )

    # Shell permission config
    shell_perms_data = sandbox_data.get("shell_permissions", [])
    shell_permissions = [
        ShellPermissionConfig(
            pattern=p.get("pattern", ""),
            allow=p.get("allow", True),
        )
        for p in shell_perms_data
        if isinstance(p, dict) and p.get("pattern")
    ]

    # Website permission config
    website_perms_data = sandbox_data.get("website_permissions", [])
    website_permissions = [
        WebsitePermissionConfig(
            pattern=p.get("pattern", ""),
            methods=p.get("methods", ["GET"]),
            allow=p.get("allow", True),
        )
        for p in website_perms_data
        if isinstance(p, dict) and p.get("pattern")
    ]

    sandbox = SandboxConfig(
        file_permissions=file_permissions,
        allow_cwd=sandbox_data.get("allow_cwd", True),
        allow_cwd_write=sandbox_data.get("allow_cwd_write", False),
        deny_by_default=sandbox_data.get("deny_by_default", True),
        allow_absolute=sandbox_data.get("allow_absolute", False),
        imports=imports,
        shell_permissions=shell_permissions,
        shell_deny_by_default=sandbox_data.get("shell_deny_by_default", True),
        website_permissions=website_permissions,
        website_deny_by_default=sandbox_data.get("website_deny_by_default", True),
        allow_localhost=sandbox_data.get("allow_localhost", False),
    )

    # MCP config
    mcp_data = data.get("mcp", {})
    servers_data = mcp_data.get("servers", [])
    mcp_servers = []
    for s in servers_data:
        if not isinstance(s, dict) or not s.get("name"):
            continue

        # Parse connect mode
        connect_str = s.get("connect", "manual")
        connect = MCPConnectMode(connect_str)

        mcp_servers.append(
            MCPServerConfig(
                name=s.get("name", ""),
                command=s.get("command"),
                extra_args=s.get("extra_args", []),
                env=s.get("env", {}),
                url=s.get("url"),
                headers=s.get("headers", {}),
                transport=s.get("transport", "stdio"),
                connect=connect,
                timeout=s.get("timeout", 30.0),
            )
        )
    mcp = MCPConfig(
        servers=mcp_servers,
        allow_dynamic_servers=mcp_data.get("allow_dynamic_servers", True),
    )

    # User config
    user_data = data.get("user", {})
    user = UserConfig(
        display_name=user_data.get("display_name"),
    )

    # Extra fields for extensibility
    known_keys = {"llm", "session", "projection", "logging", "sandbox", "mcp", "user"}
    extra = {k: v for k, v in data.items() if k not in known_keys}

    return Config(
        llm=llm,
        session=session,
        projection=projection,
        logging=logging_config,
        sandbox=sandbox,
        user=user,
        mcp=mcp,
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
