"""Configuration management for ActiveContext.

Provides hierarchical YAML-based configuration with:
- System-level config (/etc/activecontext/ or %PROGRAMDATA%)
- User-level config (~/.ac/ or %APPDATA%)
- Project-level config ($session_root/.ac/)
- Environment variable overrides (highest priority)

Example usage:
    from activecontext.config import load_config, get_config, Config

    # Load with project-specific config
    config = load_config(session_root="/path/to/project")

    # Access typed configuration
    print(config.llm.role)
    print(config.projection.total_budget)

    # Get cached global config
    config = get_config()

    # Watch for config file changes
    from activecontext.config import start_watching
    start_watching(session_root="/path/to/project")
"""

from activecontext.config.loader import (
    get_config,
    load_config,
    on_config_reload,
    reload_config,
    reset_config,
)
from activecontext.config.paths import (
    get_config_paths,
    get_project_config_path,
    get_system_config_path,
    get_user_config_path,
)
from activecontext.config.schema import (
    Config,
    LLMConfig,
    LoggingConfig,
    ProjectionConfig,
    RoleProviderConfig,
    SessionConfig,
    SessionModeConfig,
)
from activecontext.config.secrets import (
    clear_secret_cache,
    fetch_secret,
)
from activecontext.config.watcher import (
    ConfigWatcher,
    start_watching,
    stop_watching,
)

__all__ = [
    # Main API
    "Config",
    "load_config",
    "get_config",
    "reload_config",
    "reset_config",
    "on_config_reload",
    # Schema types
    "LLMConfig",
    "RoleProviderConfig",
    "SessionConfig",
    "SessionModeConfig",
    "ProjectionConfig",
    "LoggingConfig",
    # Secret management
    "fetch_secret",
    "clear_secret_cache",
    # Path utilities
    "get_config_paths",
    "get_system_config_path",
    "get_user_config_path",
    "get_project_config_path",
    # Watcher
    "ConfigWatcher",
    "start_watching",
    "stop_watching",
]
