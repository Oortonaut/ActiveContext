"""Secret management for ActiveContext.

Provides centralized secret fetching with dotenv support.
Secrets are loaded from .env files and cached for performance.

Priority order:
1. .env file in project root (cached)
2. Environment variables (os.environ)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values


@lru_cache(maxsize=1)
def _load_dotenv(env_path: Path | None = None) -> dict[str, str | None]:
    """Load and cache .env file.

    Args:
        env_path: Optional path to .env file. If None, searches current directory.

    Returns:
        Dict of environment variable names to values.
    """
    if env_path:
        return dotenv_values(env_path)
    return dotenv_values()


def fetch_secret(
    key: str,
    default: str | None = None,
    env_path: Path | None = None,
) -> str | None:
    """Fetch a secret from .env or environment.

    Priority:
    1. .env file (project-local secrets, cached)
    2. Environment variable (system/shell secrets)

    Args:
        key: Environment variable name (e.g., "ANTHROPIC_API_KEY")
        default: Default value if not found
        env_path: Optional path to .env file

    Returns:
        Secret value or default if not found.

    Example:
        >>> fetch_secret("ANTHROPIC_API_KEY")
        'sk-ant-...'
    """
    # Check .env first (cached)
    env_values = _load_dotenv(env_path)
    if key in env_values and env_values[key] is not None:
        return env_values[key]

    # Fall back to os.environ
    return os.environ.get(key, default)


def clear_secret_cache() -> None:
    """Clear the dotenv cache.

    Call this when:
    - .env file has changed
    - Running tests that modify environment
    - Reloading configuration
    """
    _load_dotenv.cache_clear()
