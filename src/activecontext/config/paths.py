"""Platform-aware configuration path resolution.

Handles config file locations for:
- Windows: %PROGRAMDATA% (system), %APPDATA% (user)
- Unix: /etc/ (system), ~/.ac/ or ~/.config/activecontext/ (user)
- Project: $session_root/.ac/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

CONFIG_FILENAME = "config.yaml"
APP_NAME = "activecontext"
SHORT_NAME = ".ac"


def get_system_config_path() -> Path | None:
    """Get system-level config path.

    Returns:
        Path to system config file, or None if not determinable.
        The file may not exist.
    """
    if sys.platform == "win32":
        # %PROGRAMDATA%\activecontext\config.yaml
        program_data = os.environ.get("PROGRAMDATA")
        if program_data:
            return Path(program_data) / APP_NAME / CONFIG_FILENAME
    else:
        # /etc/activecontext/config.yaml
        return Path("/etc") / APP_NAME / CONFIG_FILENAME
    return None


def get_user_config_path() -> Path | None:
    """Get user-level config path.

    Returns:
        Path to user config file, or None if not determinable.
        The file may not exist.
    """
    if sys.platform == "win32":
        # %APPDATA%\activecontext\config.yaml
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / APP_NAME / CONFIG_FILENAME
    else:
        # Try XDG_CONFIG_HOME first, then ~/.config, then ~/.ac
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config:
            return Path(xdg_config) / APP_NAME / CONFIG_FILENAME

        home = Path.home()

        # Prefer ~/.config/activecontext if ~/.config exists
        xdg_default = home / ".config"
        if xdg_default.exists():
            return xdg_default / APP_NAME / CONFIG_FILENAME

        # Fall back to ~/.ac
        return home / SHORT_NAME / CONFIG_FILENAME

    return None


def get_project_config_path(session_root: str) -> Path:
    """Get project-level config path.

    Args:
        session_root: The project/session root directory.

    Returns:
        Path to project config file (may not exist).
    """
    return Path(session_root) / SHORT_NAME / CONFIG_FILENAME


def get_config_paths(session_root: str | None = None) -> list[Path]:
    """Get all config paths in priority order (lowest to highest).

    Args:
        session_root: Optional project directory for project-level config.

    Returns:
        List of config paths in order: system, user, project.
        Later paths override earlier ones when merging.
    """
    paths: list[Path] = []

    system_path = get_system_config_path()
    if system_path:
        paths.append(system_path)

    user_path = get_user_config_path()
    if user_path:
        paths.append(user_path)

    if session_root:
        paths.append(get_project_config_path(session_root))

    return paths


def get_config_dirs(session_root: str | None = None) -> list[Path]:
    """Get all config directories (for watching).

    Args:
        session_root: Optional project directory.

    Returns:
        List of config directories that exist.
    """
    dirs: list[Path] = []

    for path in get_config_paths(session_root):
        parent = path.parent
        if parent.exists():
            dirs.append(parent)

    return dirs
