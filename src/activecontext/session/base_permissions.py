"""Base classes and utilities for permission managers.

This module provides common functionality shared by all permission managers:
- BasePermissionManager: Protocol for permission manager common interface
- TemporaryGrantMixin: Mixin providing temporary grant management
- write_permission_to_config_generic: Generic config file writer
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import yaml

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

# Type variables for grant types
GrantT = TypeVar("GrantT")


class TemporaryGrantMixin(Generic[GrantT]):
    """Mixin providing temporary grant management for permission managers.

    Subclasses must define `_temporary_grants: set[GrantT]` as a dataclass field.
    Override `clear_temporary_grants()` if additional cleanup is needed.
    """

    _temporary_grants: set[GrantT]
    _grant_type_name: str = "grants"  # Override in subclass for better logging

    def clear_temporary_grants(self) -> None:
        """Clear all temporary grants."""
        self._temporary_grants.clear()
        _log.debug("Temporary %s cleared", self._grant_type_name)


class ReloadablePermissionManager(ABC):
    """Abstract base for permission managers that can reload from config."""

    @classmethod
    @abstractmethod
    def from_config(cls, *args: Any, **kwargs: Any) -> ReloadablePermissionManager:
        """Create a permission manager from configuration."""
        ...

    @abstractmethod
    def reload(self, config: Any) -> None:
        """Reload permissions from a new configuration."""
        ...


@dataclass
class ConfigWriteStrategy:
    """Strategy for writing a permission entry to config.yaml.

    Attributes:
        section_path: Path to the config section (e.g., ["sandbox", "file_permissions"])
        key_field: Field name used to check for duplicates (e.g., "pattern")
        build_entry: Function that builds the dict entry from kwargs
    """

    section_path: list[str]
    key_field: str
    build_entry: Callable[..., dict[str, Any]]


def write_permission_to_config_generic(
    cwd: Path,
    strategy: ConfigWriteStrategy,
    **entry_kwargs: Any,
) -> None:
    """Write a permission entry to the project config file.

    Args:
        cwd: Project working directory
        strategy: Strategy defining how to write the entry
        **entry_kwargs: Arguments passed to strategy.build_entry()
    """
    config_path = cwd / ".ac" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # Navigate to the target section, creating intermediate dicts as needed
    section = data
    for key in strategy.section_path[:-1]:
        section = section.setdefault(key, {})

    list_key = strategy.section_path[-1]
    section.setdefault(list_key, [])

    # Build the entry
    entry = strategy.build_entry(**entry_kwargs)
    key_value = entry.get(strategy.key_field)

    # Check for duplicates
    existing_keys = {p.get(strategy.key_field) for p in section[list_key] if isinstance(p, dict)}
    if key_value in existing_keys:
        _log.debug("Permission already exists: %s=%s", strategy.key_field, key_value)
        return

    # Add the new entry
    section[list_key].append(entry)

    # Write back
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    _log.debug("Permission written to config: %s", entry)


# Pre-defined strategies for each permission type


def _build_file_permission_entry(pattern: str, access: str) -> dict[str, Any]:
    """Build a file permission entry."""
    return {"pattern": pattern, "access": access}


def _build_shell_permission_entry(pattern: str, allow: bool) -> dict[str, Any]:
    """Build a shell permission entry."""
    return {"pattern": pattern, "allow": allow}


def _build_import_entry(module: str, allow_submodules: bool = False) -> dict[str, Any]:
    """Build an import permission entry."""
    entry: dict[str, Any] = {"module": module}
    if allow_submodules:
        entry["allow_submodules"] = True
    return entry


def _build_website_permission_entry(
    pattern: str, allow: bool, methods: list[str] | None = None
) -> dict[str, Any]:
    """Build a website permission entry."""
    entry: dict[str, Any] = {"pattern": pattern, "allow": allow}
    if methods:
        entry["methods"] = methods
    return entry


# Strategy instances for easy reuse
FILE_PERMISSION_STRATEGY = ConfigWriteStrategy(
    section_path=["sandbox", "file_permissions"],
    key_field="pattern",
    build_entry=_build_file_permission_entry,
)

SHELL_PERMISSION_STRATEGY = ConfigWriteStrategy(
    section_path=["sandbox", "shell_permissions"],
    key_field="pattern",
    build_entry=_build_shell_permission_entry,
)

IMPORT_PERMISSION_STRATEGY = ConfigWriteStrategy(
    section_path=["sandbox", "imports", "allowed"],
    key_field="module",
    build_entry=_build_import_entry,
)

WEBSITE_PERMISSION_STRATEGY = ConfigWriteStrategy(
    section_path=["sandbox", "website_permissions"],
    key_field="pattern",
    build_entry=_build_website_permission_entry,
)
