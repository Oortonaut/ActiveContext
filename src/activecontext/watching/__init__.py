"""File watching module for ActiveContext.

Provides polling-based file watching to detect external changes to files
referenced by TextNodes. Changes are reported through the event system
and can wake the agent or be queued for batch processing.
"""

from activecontext.watching.watcher import (
    FileChangeEvent,
    FileWatcher,
    WatchedFile,
)

__all__ = [
    "FileChangeEvent",
    "FileWatcher",
    "WatchedFile",
]
