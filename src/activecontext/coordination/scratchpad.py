"""Scratchpad manager for agent work coordination."""

from __future__ import annotations

import fnmatch
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml
from filelock import FileLock

from activecontext.coordination.schema import (
    Conflict,
    FileAccess,
    Scratchpad,
    WorkEntry,
)

if TYPE_CHECKING:
    pass


class ScratchpadManager:
    """Manages the shared work coordination file.

    The scratchpad is a YAML file at `.ac/scratchpad.yaml` where agents
    register their work areas. It is advisory only - agents use it to
    inform each other but are not blocked by conflicts.
    """

    def __init__(self, cwd: str) -> None:
        """Initialize the scratchpad manager.

        Args:
            cwd: Working directory (project root)
        """
        self._cwd = Path(cwd)
        self._path = self._cwd / ".ac" / "scratchpad.yaml"
        self._lock_path = self._path.with_suffix(".lock")
        self._agent_id: str | None = None
        self._session_id: str | None = None

    @property
    def agent_id(self) -> str | None:
        """The agent ID for this manager (set after register)."""
        return self._agent_id

    def _load(self) -> Scratchpad:
        """Load scratchpad from file."""
        if not self._path.exists():
            return Scratchpad()
        try:
            with open(self._path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return Scratchpad.from_dict(data)
        except Exception:
            # If file is corrupted, start fresh
            return Scratchpad()

    def _save(self, scratchpad: Scratchpad) -> None:
        """Save scratchpad to file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.safe_dump(scratchpad.to_dict(), f, default_flow_style=False)

    def _atomic_update(
        self, modifier: Callable[[Scratchpad], Scratchpad]
    ) -> Scratchpad:
        """Read-modify-write with file locking."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self._lock_path, timeout=10)
        with lock:
            scratchpad = self._load()
            scratchpad = modifier(scratchpad)
            self._save(scratchpad)
            return scratchpad

    def register(
        self,
        session_id: str,
        intent: str,
        files: list[FileAccess] | None = None,
        dependencies: list[str] | None = None,
    ) -> WorkEntry:
        """Register this agent's work area.

        Creates a new entry or updates existing one for this agent.
        Also runs stale cleanup opportunistically.

        Args:
            session_id: The session ID
            intent: Human-readable description of work
            files: Files/patterns being accessed
            dependencies: Files needed (read-only)

        Returns:
            The created/updated entry
        """
        if self._agent_id is None:
            self._agent_id = uuid.uuid4().hex[:8]
        self._session_id = session_id

        now = datetime.now(timezone.utc)
        entry = WorkEntry(
            id=self._agent_id,
            session_id=session_id,
            intent=intent,
            status="active",
            files=files or [],
            dependencies=dependencies or [],
            started_at=now,
            updated_at=now,
            heartbeat_at=now,
        )

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            # Remove stale entries
            threshold = now - timedelta(seconds=300)
            scratchpad.entries = [
                e for e in scratchpad.entries if e.heartbeat_at > threshold
            ]
            # Remove existing entry for this agent
            scratchpad.entries = [
                e for e in scratchpad.entries if e.id != self._agent_id
            ]
            # Add new entry
            scratchpad.entries.append(entry)
            return scratchpad

        self._atomic_update(modifier)
        return entry

    def update(
        self,
        intent: str | None = None,
        files: list[FileAccess] | None = None,
        dependencies: list[str] | None = None,
        status: str | None = None,
    ) -> WorkEntry | None:
        """Update this agent's entry.

        Args:
            intent: New intent description
            files: New file list
            dependencies: New dependencies
            status: New status (active/paused/done)

        Returns:
            The updated entry, or None if not registered
        """
        if self._agent_id is None:
            return None

        now = datetime.now(timezone.utc)
        updated_entry: WorkEntry | None = None

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            nonlocal updated_entry
            for entry in scratchpad.entries:
                if entry.id == self._agent_id:
                    if intent is not None:
                        entry.intent = intent
                    if files is not None:
                        entry.files = files
                    if dependencies is not None:
                        entry.dependencies = dependencies
                    if status is not None:
                        entry.status = status
                    entry.updated_at = now
                    entry.heartbeat_at = now
                    updated_entry = entry
                    break
            return scratchpad

        self._atomic_update(modifier)
        return updated_entry

    def heartbeat(self) -> None:
        """Update heartbeat timestamp to show agent is alive."""
        if self._agent_id is None:
            return

        now = datetime.now(timezone.utc)

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            for entry in scratchpad.entries:
                if entry.id == self._agent_id:
                    entry.heartbeat_at = now
                    break
            return scratchpad

        self._atomic_update(modifier)

    def unregister(self) -> None:
        """Remove this agent's entry (called on session close)."""
        if self._agent_id is None:
            return

        agent_id = self._agent_id

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            scratchpad.entries = [e for e in scratchpad.entries if e.id != agent_id]
            return scratchpad

        self._atomic_update(modifier)
        self._agent_id = None
        self._session_id = None

    def get_conflicts(
        self, paths: list[str], mode: str = "write"
    ) -> list[Conflict]:
        """Check if given paths conflict with other agents.

        A conflict occurs when:
        - Another agent has write access to a file we want to access
        - Another agent has any access to a file we want to write

        This is advisory - does not block.

        Args:
            paths: File paths to check
            mode: Access mode we want ("read" or "write")

        Returns:
            List of conflicts with other agents
        """
        scratchpad = self._load()
        conflicts: list[Conflict] = []

        for entry in scratchpad.entries:
            # Skip self
            if entry.id == self._agent_id:
                continue
            # Skip inactive entries
            if entry.status != "active":
                continue

            for file_access in entry.files:
                for path in paths:
                    if self._paths_match(path, file_access.path):
                        # Conflict if they're writing, or we're writing
                        if file_access.mode == "write" or mode == "write":
                            conflicts.append(
                                Conflict(
                                    agent_id=entry.id,
                                    file=file_access.path,
                                    their_mode=file_access.mode,
                                    their_intent=entry.intent,
                                )
                            )

        return conflicts

    def _paths_match(self, path1: str, path2: str) -> bool:
        """Check if two paths match (supports glob patterns)."""
        # Normalize paths
        p1 = Path(path1).as_posix()
        p2 = Path(path2).as_posix()

        # Direct match
        if p1 == p2:
            return True

        # Check if either is a glob pattern matching the other
        if fnmatch.fnmatch(p1, p2) or fnmatch.fnmatch(p2, p1):
            return True

        return False

    def get_all_entries(self) -> list[WorkEntry]:
        """Get all current work entries."""
        scratchpad = self._load()
        return scratchpad.entries

    def cleanup_stale(self, max_age_seconds: float = 300) -> int:
        """Remove entries with heartbeats older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds (default 5 minutes)

        Returns:
            Number of entries removed
        """
        now = datetime.now(timezone.utc)
        threshold = now - timedelta(seconds=max_age_seconds)
        removed_count = 0

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            nonlocal removed_count
            original_count = len(scratchpad.entries)
            scratchpad.entries = [
                e for e in scratchpad.entries if e.heartbeat_at > threshold
            ]
            removed_count = original_count - len(scratchpad.entries)
            return scratchpad

        self._atomic_update(modifier)
        return removed_count
