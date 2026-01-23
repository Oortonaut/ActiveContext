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
