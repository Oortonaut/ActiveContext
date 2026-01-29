"""Logging configuration for ActiveContext.

Uses Python's standard logging module with support for:
- File logging via config or AC_LOG environment variable
- Verbosity levels: error(0), warning(1), info(2), verbose(3), trace(4)
- Stderr fallback when no log file is configured
- Structured format with timestamps and level names
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activecontext.config.schema import LoggingConfig

# Custom log levels
TRACE = 5
VERBOSE = 15

logging.addLevelName(TRACE, "TRACE")
logging.addLevelName(VERBOSE, "VERBOSE")

# Module-level logger
logger = logging.getLogger("activecontext")

_initialized = False

# Map string level names to logging constants
_LEVEL_MAP = {
    "TRACE": TRACE,
    "DEBUG": logging.DEBUG,
    "VERBOSE": VERBOSE,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Map --verbose=N to log levels (0=errors only, 4=everything)
_VERBOSITY_MAP = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: VERBOSE,
    4: TRACE,
}


class _LowercaseLevelFormatter(logging.Formatter):
    """Formatter that emits lowercase level names."""

    def format(self, record: logging.LogRecord) -> str:
        record.levelname = record.levelname.lower()
        return super().format(record)


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Initialize logging based on configuration.

    Config values take precedence, with env var fallback for backward compat.
    Call this once at startup. Subsequent calls are no-ops.

    Verbosity levels (--verbose / config.logging.verbose):
        0 = error   - errors only
        1 = warning  - errors + warnings
        2 = info     - normal operation (default)
        3 = verbose  - detailed diagnostics
        4 = trace    - everything

    Args:
        config: Optional LoggingConfig with level, verbose, and file settings.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Determine log level: verbose (int) takes precedence over level (str)
    log_level = logging.INFO  # default
    if config:
        if config.verbose is not None:
            log_level = _VERBOSITY_MAP.get(config.verbose, TRACE)
        elif config.level:
            level_str = config.level.upper()
            log_level = _LEVEL_MAP.get(level_str, logging.INFO)

    logger.setLevel(log_level)

    # Format: HH:MM:SS level: message
    formatter = _LowercaseLevelFormatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

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
