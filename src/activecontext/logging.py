"""Logging configuration for ActiveContext.

Uses Python's standard logging module with support for:
- File logging via ACTIVECONTEXT_LOG environment variable
- Stderr fallback when no log file is configured
- Structured format with timestamps
"""

import logging
import os
import sys

# Module-level logger
logger = logging.getLogger("activecontext")

_initialized = False


def setup_logging() -> None:
    """Initialize logging based on environment configuration.

    Call this once at startup. Subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    logger.setLevel(logging.DEBUG)

    # Format: [HH:MM:SS] message
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    # Check for file logging
    log_path = os.environ.get("ACTIVECONTEXT_LOG")
    if log_path:
        try:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fall back to stderr if file can't be opened
            print(f"[activecontext] Failed to open log file: {e}", file=sys.stderr)
            _add_stderr_handler(formatter)
    else:
        _add_stderr_handler(formatter)


def _add_stderr_handler(formatter: logging.Formatter) -> None:
    """Add a stderr handler to the logger."""
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG)
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
