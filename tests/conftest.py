"""Root pytest configuration for all tests."""

from __future__ import annotations

import pytest

# Configure pytest-asyncio to use auto mode
# This is redundant with pyproject.toml but ensures it's set
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def anyio_backend():
    """Set anyio backend to asyncio."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _clean_file_watcher_registry():
    """Clear the module-level file watcher registry before and after each test.

    TextNode.__post_init__ auto-registers with the global _file_watchers dict.
    Without this cleanup, TextNodes created in one test leak into subsequent
    tests across modules, causing spurious failures in tests that inspect
    the registry contents directly.
    """
    from activecontext.context.nodes import _file_watchers

    _file_watchers.clear()
    yield
    _file_watchers.clear()
