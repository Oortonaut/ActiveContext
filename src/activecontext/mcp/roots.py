"""MCP roots management for ActiveContext.

Manages filesystem roots that are advertised to connected MCP servers
per the MCP roots protocol (2025-06-18 spec).

Roots define boundaries for where servers can operate within the filesystem.
The client advertises roots via the roots/list callback and sends
notifications/roots/list_changed when the set changes.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_log = logging.getLogger("activecontext.mcp.roots")


@dataclass(frozen=True)
class Root:
    """A filesystem root advertised to MCP servers.

    Attributes:
        uri: file:// URI for the root directory.
        name: Optional human-readable display name.
    """

    uri: str
    name: str | None = None


def path_to_file_uri(path: str | Path) -> str:
    """Convert an OS path to a file:// URI.

    Handles both Unix and Windows paths:
        /home/user/project  -> file:///home/user/project
        C:\\Users\\Ace\\proj -> file:///C:/Users/Ace/proj

    Args:
        path: OS filesystem path (absolute).

    Returns:
        Normalized file:// URI string.
    """
    p = Path(path).resolve()
    return p.as_uri()


def file_uri_to_path(uri: str) -> str:
    """Convert a file:// URI back to an OS-native path.

    Strips the file:// scheme and returns a platform-appropriate path:
        file:///home/user/project  -> /home/user/project      (Unix)
        file:///C:/Users/Ace/proj  -> C:\\Users\\Ace\\proj     (Windows)

    Also handles bare paths passed in without scheme (no-op normalization).

    Args:
        uri: file:// URI or bare path.

    Returns:
        OS-native absolute path string.
    """
    if not uri.startswith("file://"):
        # Bare path — just normalize
        return os.path.normpath(uri)

    # Strip file:// prefix
    # file:///home/user -> /home/user  (Unix, 3 slashes -> path starts with /)
    # file:///C:/Users  -> C:/Users    (Windows, drive letter after third slash)
    path_part = uri[7:]  # Remove "file://"

    # Handle Windows drive letter: /C:/... -> C:/...
    if len(path_part) >= 3 and path_part[0] == "/" and path_part[2] == ":":
        path_part = path_part[1:]

    # Decode percent-encoded characters
    from urllib.parse import unquote

    path_part = unquote(path_part)

    # Convert forward slashes to native separator
    return os.path.normpath(path_part)


def normalize_to_file_uri(path_or_uri: str) -> str:
    """Accept either a path or file:// URI and return a normalized file:// URI.

    This is the canonical entry point for user-provided root paths — it handles
    bare paths, file:// URIs, and normalizes to a consistent format.

    Args:
        path_or_uri: A filesystem path or file:// URI.

    Returns:
        Normalized file:// URI.
    """
    if path_or_uri.startswith("file://"):
        # Already a URI — round-trip to normalize
        return path_to_file_uri(file_uri_to_path(path_or_uri))
    return path_to_file_uri(path_or_uri)


# Callback type for change notifications
RootsChangedCallback = Callable[[], Any]


@dataclass
class RootsManager:
    """Manages the set of filesystem roots advertised to MCP servers.

    Usage:
        manager = RootsManager()
        manager.add("/home/user/project", name="My Project")
        manager.add("C:\\\\Work\\\\repo", name="Work Repo")

        # Get roots for MCP protocol
        roots = manager.list_roots()

        # Register callback for changes
        manager.on_change(notify_servers)
    """

    _roots: dict[str, Root] = field(default_factory=dict)
    _change_callbacks: list[RootsChangedCallback] = field(default_factory=list)

    def add(self, path_or_uri: str, name: str | None = None) -> Root:
        """Register a new root.

        Args:
            path_or_uri: Filesystem path or file:// URI.
            name: Optional display name.

        Returns:
            The created Root object.
        """
        uri = normalize_to_file_uri(path_or_uri)

        if uri in self._roots:
            existing = self._roots[uri]
            if existing.name == name:
                return existing
            # Update name
            _log.debug("Updating root name: %s -> %s", existing.name, name)

        root = Root(uri=uri, name=name)
        self._roots[uri] = root
        _log.info("Registered root: %s (%s)", uri, name or "unnamed")
        self._notify_changed()
        return root

    def remove(self, path_or_uri: str) -> bool:
        """Remove a root by path or URI.

        Args:
            path_or_uri: Path or URI to remove.

        Returns:
            True if removed, False if not found.
        """
        uri = normalize_to_file_uri(path_or_uri)
        if uri in self._roots:
            removed = self._roots.pop(uri)
            _log.info("Removed root: %s (%s)", uri, removed.name or "unnamed")
            self._notify_changed()
            return True
        return False

    def list_roots(self) -> list[Root]:
        """Get all registered roots."""
        return list(self._roots.values())

    def has_roots(self) -> bool:
        """Check if any roots are registered."""
        return len(self._roots) > 0

    def on_change(self, callback: RootsChangedCallback) -> Callable[[], None]:
        """Register a callback for root changes.

        Args:
            callback: Called (no args) whenever roots are added/removed.

        Returns:
            Unregister function.
        """
        self._change_callbacks.append(callback)

        def unregister() -> None:
            if callback in self._change_callbacks:
                self._change_callbacks.remove(callback)

        return unregister

    def _notify_changed(self) -> None:
        """Invoke all change callbacks."""
        for cb in self._change_callbacks:
            try:
                cb()
            except Exception:
                _log.debug("Root change callback failed", exc_info=True)
