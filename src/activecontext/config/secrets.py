"""Secret management for ActiveContext.

Provides centralized secret fetching with dotenv support.
Secrets are loaded from .env.secrets files and cached for performance.

Priority order:
1. Environment variables (os.environ) - set by shell, IDE, or .env
2. .env.secrets file in project root (cached) - for API keys not in env
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

# Default secrets file name
SECRETS_FILE = ".env.secrets"


@lru_cache(maxsize=1)
def _load_secrets(secrets_path: Path | None = None) -> dict[str, str | None]:
    """Load and cache .env.secrets file.

    Args:
        secrets_path: Optional path to secrets file. If None, searches for .env.secrets.

    Returns:
        Dict of environment variable names to values.
    """
    if secrets_path:
        if secrets_path.exists():
            return dotenv_values(secrets_path)
        return {}

    # Search for .env.secrets in current directory
    default_path = Path(SECRETS_FILE)
    if default_path.exists():
        return dotenv_values(default_path)
    return {}


def fetch_secret(
    key: str,
    default: str | None = None,
    secrets_path: Path | None = None,
) -> str | None:
    """Fetch a secret from environment or .env.secrets.

    Priority:
    1. Environment variable (os.environ) - real env vars, shell, IDE, .env
    2. .env.secrets file (project-local secrets, cached)

    This ordering ensures that:
    - Tests can use monkeypatch.delenv() to clear env vars
    - Real environment variables take precedence over secrets file
    - API keys can be stored in .env.secrets for local development

    Args:
        key: Environment variable name (e.g., "ANTHROPIC_API_KEY")
        default: Default value if not found
        secrets_path: Optional path to .env.secrets file

    Returns:
        Secret value or default if not found.

    Example:
        >>> fetch_secret("ANTHROPIC_API_KEY")
        'sk-ant-...'
    """
    # Check os.environ first (allows test mocking via monkeypatch)
    value = os.environ.get(key)
    if value is not None:
        return value

    # Fall back to .env.secrets file (cached)
    secrets = _load_secrets(secrets_path)
    if key in secrets and secrets[key] is not None:
        return secrets[key]

    return default


def clear_secret_cache() -> None:
    """Clear the secrets cache.

    Call this when:
    - .env.secrets file has changed
    - Running tests that modify environment
    - Reloading configuration
    """
    _load_secrets.cache_clear()
