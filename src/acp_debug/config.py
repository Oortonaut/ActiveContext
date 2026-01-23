"""Configuration loading for acp-debug."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ShutdownConfig:
    """Shutdown timeout configuration."""

    interrupt_timeout: float = 2.0
    """Seconds to wait after sending interrupt (SIGINT/Ctrl+Break)."""

    terminate_timeout: float = 3.0
    """Seconds to wait after sending terminate (SIGTERM)."""


@dataclass
class Config:
    """ACP Debug configuration."""

    extensions: list[Path] = field(default_factory=list)
    extensions_path: list[Path] = field(default_factory=list)
    verbose: int = 0
    quiet: bool = False

    # Logging config per method
    log_config: dict[str, str] = field(default_factory=dict)

    # Shutdown configuration
    shutdown: ShutdownConfig = field(default_factory=ShutdownConfig)


def load_config(
    config_path: Path | None = None,
    extensions: list[Path] | None = None,
    extensions_path: list[Path] | None = None,
) -> Config:
    """Load configuration from file and CLI overrides."""
    config = Config()

    # Try to load config file
    if config_path is None:
        # Look for default config files
        for name in ["acp-debug.yaml", ".acp-debug.yaml", "acp-debug.yml", ".acp-debug.yml"]:
            if Path(name).exists():
                config_path = Path(name)
                break

    if config_path and config_path.exists():
        config = _load_yaml_config(config_path)

    # CLI overrides (extend, don't replace)
    if extensions:
        config.extensions.extend(extensions)
    if extensions_path:
        config.extensions_path.extend(extensions_path)

    return config


def _load_yaml_config(path: Path) -> Config:
    """Load config from YAML file."""
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    # Parse shutdown config
    shutdown_data = data.get("shutdown", {})
    shutdown = ShutdownConfig(
        interrupt_timeout=shutdown_data.get("interrupt_timeout", 2.0),
        terminate_timeout=shutdown_data.get("terminate_timeout", 3.0),
    )

    return Config(
        extensions=[Path(p) for p in data.get("extensions", [])],
        extensions_path=[Path(p) for p in data.get("extensions_path", [])],
        verbose=data.get("verbose", 0),
        quiet=data.get("quiet", False),
        log_config=data.get("log", {}),
        shutdown=shutdown,
    )
