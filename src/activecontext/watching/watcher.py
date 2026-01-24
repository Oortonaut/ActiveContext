"""File watching implementation using polling.

This module provides polling-based file watching, similar to the ConfigWatcher
pattern used elsewhere in ActiveContext. Polling is preferred over native
file watchers for cross-platform reliability.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from activecontext.logging import get_logger

if TYPE_CHECKING:
    from typing import Any

log = get_logger("watching")


@dataclass
class WatchedFile:
    """Tracks a watched file's state."""

    path: Path
    mtime: float | None = None
    size: int | None = None
    exists: bool = True


@dataclass
class FileChangeEvent:
    """Represents a detected file change."""

    path: Path
    change_type: str  # "modified", "created", "deleted"
    old_mtime: float | None
    new_mtime: float | None
    timestamp: float = field(default_factory=time.time)
    node_ids: set[str] = field(default_factory=set)  # TextNodes watching this file

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for event payloads."""
        return {
            "path": str(self.path),
            "change_type": self.change_type,
            "old_mtime": self.old_mtime,
            "new_mtime": self.new_mtime,
            "timestamp": self.timestamp,
            "node_ids": list(self.node_ids),
        }


class FileWatcher:
    """Watches files for external changes using polling.

    The FileWatcher polls registered file paths at a configurable interval
    and detects changes (modifications, creations, deletions). When a change
    is detected, the registered callback is invoked.

    This integrates with the Session tick cycle:
    1. TextNodes register their file paths on creation
    2. FileWatcher polls for changes at the configured interval
    3. Changes are reported via callback (typically fires events)
    4. Events can wake the agent or be queued for batch processing

    Example:
        watcher = FileWatcher(cwd=Path("/project"), poll_interval=1.0)
        watcher.register_path(Path("/project/main.py"), "text_1")

        async def on_change(event: FileChangeEvent) -> None:
            print(f"File changed: {event.path}")

        await watcher.start(on_change)
    """

    def __init__(
        self,
        cwd: Path,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize the file watcher.

        Args:
            cwd: Working directory for resolving relative paths
            poll_interval: Seconds between polling cycles (default 1.0)
        """
        self._cwd = cwd
        self._poll_interval = poll_interval

        # Watched files: path -> WatchedFile
        self._watched: dict[Path, WatchedFile] = {}

        # Registry: path -> set of node_ids watching this path
        self._registry: dict[Path, set[str]] = {}

        # Control state
        self._running = False
        self._task: asyncio.Task[None] | None = None

    @property
    def poll_interval(self) -> float:
        """Get the polling interval in seconds."""
        return self._poll_interval

    @poll_interval.setter
    def poll_interval(self, value: float) -> None:
        """Set the polling interval in seconds."""
        self._poll_interval = max(0.1, value)  # Minimum 100ms

    @property
    def watched_count(self) -> int:
        """Get the number of watched files."""
        return len(self._watched)

    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to cwd.

        Args:
            path: Path string or Path object (can be relative or absolute)

        Returns:
            Absolute resolved path
        """
        p = Path(path)
        if not p.is_absolute():
            p = self._cwd / p
        return p.resolve()

    def register_path(self, path: str | Path, node_id: str) -> None:
        """Register a node's interest in a file path.

        Args:
            path: Path to watch (will be resolved relative to cwd)
            node_id: ID of the node watching this path
        """
        resolved = self.resolve_path(path)

        # Add to registry
        if resolved not in self._registry:
            self._registry[resolved] = set()
        self._registry[resolved].add(node_id)

        # Initialize watched file if new
        if resolved not in self._watched:
            try:
                stat = resolved.stat()
                self._watched[resolved] = WatchedFile(
                    path=resolved,
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                    exists=True,
                )
            except FileNotFoundError:
                self._watched[resolved] = WatchedFile(
                    path=resolved,
                    mtime=None,
                    size=None,
                    exists=False,
                )

        log.debug("Registered %s for node %s", resolved, node_id)

    def unregister_path(self, path: str | Path, node_id: str) -> None:
        """Unregister a node's interest in a file path.

        Args:
            path: Path to stop watching
            node_id: ID of the node to unregister
        """
        resolved = self.resolve_path(path)

        if resolved in self._registry:
            self._registry[resolved].discard(node_id)

            # If no more watchers, remove from watched files
            if not self._registry[resolved]:
                del self._registry[resolved]
                self._watched.pop(resolved, None)
                log.debug("Removed watch for %s", resolved)

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from all watched paths.

        Args:
            node_id: ID of the node to unregister
        """
        paths_to_remove: list[Path] = []

        for path, nodes in self._registry.items():
            nodes.discard(node_id)
            if not nodes:
                paths_to_remove.append(path)

        for path in paths_to_remove:
            del self._registry[path]
            self._watched.pop(path, None)

    def get_watchers(self, path: str | Path) -> set[str]:
        """Get the node IDs watching a specific path.

        Args:
            path: Path to check

        Returns:
            Set of node IDs watching this path
        """
        resolved = self.resolve_path(path)
        return self._registry.get(resolved, set()).copy()

    def check_changes(self) -> list[FileChangeEvent]:
        """Check all watched files for changes.

        This is a synchronous check suitable for calling from tick().

        Returns:
            List of FileChangeEvent for any changed files
        """
        events: list[FileChangeEvent] = []

        for path, watched in list(self._watched.items()):
            event = self._check_file(path, watched)
            if event:
                events.append(event)

        return events

    def _check_file(self, path: Path, watched: WatchedFile) -> FileChangeEvent | None:
        """Check a single file for changes.

        Args:
            path: Path to check
            watched: Current watched state

        Returns:
            FileChangeEvent if changed, None otherwise
        """
        node_ids = self._registry.get(path, set())

        try:
            stat = path.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size

            if not watched.exists:
                # File was created
                watched.exists = True
                watched.mtime = current_mtime
                watched.size = current_size
                return FileChangeEvent(
                    path=path,
                    change_type="created",
                    old_mtime=None,
                    new_mtime=current_mtime,
                    node_ids=node_ids.copy(),
                )

            # Check for modification (mtime or size changed)
            if current_mtime != watched.mtime or current_size != watched.size:
                old_mtime = watched.mtime
                watched.mtime = current_mtime
                watched.size = current_size
                return FileChangeEvent(
                    path=path,
                    change_type="modified",
                    old_mtime=old_mtime,
                    new_mtime=current_mtime,
                    node_ids=node_ids.copy(),
                )

        except FileNotFoundError:
            if watched.exists:
                # File was deleted
                old_mtime = watched.mtime
                watched.exists = False
                watched.mtime = None
                watched.size = None
                return FileChangeEvent(
                    path=path,
                    change_type="deleted",
                    old_mtime=old_mtime,
                    new_mtime=None,
                    node_ids=node_ids.copy(),
                )

        except (PermissionError, OSError) as e:
            log.warning("Error checking %s: %s", path, e)

        return None

    async def start(
        self,
        callback: Callable[[FileChangeEvent], None],
    ) -> None:
        """Start the polling loop.

        Args:
            callback: Function to call when a file change is detected.
                     The callback receives a FileChangeEvent.
        """
        if self._running:
            log.warning("FileWatcher already running")
            return

        self._running = True
        log.info("FileWatcher started (interval: %.1fs)", self._poll_interval)

        try:
            while self._running:
                # Check all files
                for event in self.check_changes():
                    try:
                        callback(event)
                    except Exception as e:
                        log.error("Error in file change callback: %s", e)

                # Wait for next poll
                await asyncio.sleep(self._poll_interval)

        except asyncio.CancelledError:
            log.info("FileWatcher cancelled")
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        log.info("FileWatcher stopped")

    def is_running(self) -> bool:
        """Check if the watcher is currently running."""
        return self._running
