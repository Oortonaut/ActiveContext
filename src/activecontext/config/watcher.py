"""Config file watcher for automatic reload on changes.

Uses polling-based approach for cross-platform compatibility without
additional dependencies. Monitors config file modification times.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from activecontext.config.loader import reload_config
from activecontext.config.paths import get_config_paths

if TYPE_CHECKING:
    pass

_log = logging.getLogger("activecontext.config.watcher")

# Default poll interval in seconds
DEFAULT_POLL_INTERVAL = 2.0


class ConfigWatcher:
    """Watches config files for changes and triggers reload.

    Uses polling to check file modification times. This is more portable
    than inotify/FSEvents and doesn't require additional dependencies.
    """

    def __init__(
        self,
        session_root: str | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        """Initialize the config watcher.

        Args:
            session_root: Optional project directory to watch.
            poll_interval: How often to check for changes (seconds).
        """
        self._session_root = session_root
        self._poll_interval = poll_interval
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._mtimes: dict[Path, float] = {}

    def _get_watched_paths(self) -> list[Path]:
        """Get list of config files to watch."""
        return get_config_paths(self._session_root)

    def _check_mtimes(self) -> dict[Path, float]:
        """Get current modification times for all config files."""
        mtimes: dict[Path, float] = {}
        for path in self._get_watched_paths():
            if path.exists():
                with contextlib.suppress(OSError):
                    mtimes[path] = path.stat().st_mtime
        return mtimes

    def _detect_changes(self) -> list[Path]:
        """Detect which files have changed since last check.

        Returns:
            List of paths that were created, modified, or deleted.
        """
        current = self._check_mtimes()
        changed: list[Path] = []

        # Check for modified or deleted files
        for path, old_mtime in self._mtimes.items():
            new_mtime = current.get(path)
            if new_mtime is None:
                # File was deleted
                changed.append(path)
            elif new_mtime != old_mtime:
                # File was modified
                changed.append(path)

        # Check for new files
        for path in current:
            if path not in self._mtimes:
                changed.append(path)

        # Update stored mtimes
        self._mtimes = current

        return changed

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        # Initialize mtimes
        self._mtimes = self._check_mtimes()

        while self._running:
            await asyncio.sleep(self._poll_interval)

            if not self._running:
                break

            changed = self._detect_changes()
            if changed:
                _log.info("Config changed: %s", [str(p) for p in changed])
                try:
                    reload_config(session_root=self._session_root)
                except Exception as e:
                    _log.error("Error reloading config: %s", e)

    def start(self) -> None:
        """Start watching for config changes.

        Creates an async task that polls for changes.
        Must be called from within an async context.
        """
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        _log.debug("Config watcher started (interval=%.1fs)", self._poll_interval)

    def stop(self) -> None:
        """Stop watching for config changes."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        _log.debug("Config watcher stopped")

    async def __aenter__(self) -> ConfigWatcher:
        """Async context manager entry."""
        self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        """Async context manager exit."""
        self.stop()


# Global watcher instance
_global_watcher: ConfigWatcher | None = None


def start_watching(
    session_root: str | None = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> ConfigWatcher:
    """Start the global config watcher.

    Args:
        session_root: Optional project directory.
        poll_interval: How often to check for changes.

    Returns:
        The ConfigWatcher instance.
    """
    global _global_watcher

    if _global_watcher is not None:
        _global_watcher.stop()

    _global_watcher = ConfigWatcher(
        session_root=session_root,
        poll_interval=poll_interval,
    )
    _global_watcher.start()
    return _global_watcher


def stop_watching() -> None:
    """Stop the global config watcher."""
    global _global_watcher

    if _global_watcher is not None:
        _global_watcher.stop()
        _global_watcher = None
