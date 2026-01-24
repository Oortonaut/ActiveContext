"""Logging configuration for ActiveContext.

Uses Python's standard logging module with support for:
- File logging via config or AC_LOG environment variable
- Log level configuration via config
- Stderr fallback when no log file is configured
- Structured format with timestamps
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activecontext.config.schema import LoggingConfig

# Module-level logger
logger = logging.getLogger("activecontext")

_initialized = False

# Map string level names to logging constants
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Initialize logging based on configuration.

    Config values take precedence, with env var fallback for backward compat.
    Call this once at startup. Subsequent calls are no-ops.

    Args:
        config: Optional LoggingConfig with level and file settings.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Determine log level (config -> default DEBUG)
    log_level = logging.DEBUG
    if config and config.level:
        level_str = config.level.upper()
        log_level = _LEVEL_MAP.get(level_str, logging.DEBUG)

    logger.setLevel(log_level)

    # Format: [HH:MM:SS] message
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # Determine log file path (config already includes env var via loader)
    # Fallback to env var directly if no config provided
    log_path = config.file if config and config.file else os.environ.get("AC_LOG")

    if log_path:
        # Expand ~ in path
        log_path = os.path.expanduser(log_path)
        try:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fall back to stderr if file can't be opened (only if real console)
            if sys.stderr.isatty():
                print(f"[activecontext] Failed to open log file: {e}", file=sys.stderr)
                _add_stderr_handler(formatter, log_level)
    elif sys.stderr.isatty():
        # Only log to stderr if it's a real console, not pipes from IDE
        _add_stderr_handler(formatter, log_level)


def _add_stderr_handler(formatter: logging.Formatter, level: int = logging.DEBUG) -> None:
    """Add a stderr handler to the logger."""
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional name for a child logger (e.g., "acp", "session").
              If None, returns the root activecontext logger.

    Returns:
        A configured logger instance.
    """
    if name:
        return logger.getChild(name)
    return logger
