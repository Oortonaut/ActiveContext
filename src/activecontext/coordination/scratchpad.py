"""Scratchpad manager for agent work coordination."""

from __future__ import annotations

import fnmatch
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from filelock import FileLock

from activecontext.coordination.schema import (
    Conflict,
    FileAccess,
    Scratchpad,
    WorkEntry,
)

if TYPE_CHECKING:
    from activecontext.agents.schema import AgentEntry, AgentMessage, AgentState


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

    def _atomic_update(self, modifier: Callable[[Scratchpad], Scratchpad]) -> Scratchpad:
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
            scratchpad.entries = [e for e in scratchpad.entries if e.heartbeat_at > threshold]
            # Remove existing entry for this agent
            scratchpad.entries = [e for e in scratchpad.entries if e.id != self._agent_id]
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

    def get_conflicts(self, paths: list[str], mode: str = "write") -> list[Conflict]:
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

        # Build indexes for O(1) exact-match lookups
        # Separate exact paths from glob patterns
        exact_files: dict[str, list[tuple[WorkEntry, FileAccess]]] = {}
        glob_files: list[tuple[WorkEntry, FileAccess]] = []

        for entry in scratchpad.entries:
            # Skip self and inactive entries
            if entry.id == self._agent_id or entry.status != "active":
                continue
            for file_access in entry.files:
                normalized = Path(file_access.path).as_posix()
                if "*" in normalized or "?" in normalized:
                    glob_files.append((entry, file_access))
                else:
                    exact_files.setdefault(normalized, []).append((entry, file_access))

        # Normalize query paths once
        normalized_paths = [(p, Path(p).as_posix()) for p in paths]

        for original_path, norm_path in normalized_paths:
            is_glob = "*" in norm_path or "?" in norm_path

            # O(1) exact match lookup for non-glob query paths
            if not is_glob and norm_path in exact_files:
                for entry, fa in exact_files[norm_path]:
                    if fa.mode == "write" or mode == "write":
                        conflicts.append(
                            Conflict(
                                agent_id=entry.id,
                                file=fa.path,
                                their_mode=fa.mode,
                                their_intent=entry.intent,
                            )
                        )

            # Check against glob patterns from other agents
            for entry, fa in glob_files:
                if self._paths_match(original_path, fa.path):
                    if fa.mode == "write" or mode == "write":
                        conflicts.append(
                            Conflict(
                                agent_id=entry.id,
                                file=fa.path,
                                their_mode=fa.mode,
                                their_intent=entry.intent,
                            )
                        )

            # If query path is a glob, check against all exact paths
            if is_glob:
                for exact_path, entries_fas in exact_files.items():
                    if fnmatch.fnmatch(exact_path, norm_path):
                        for entry, fa in entries_fas:
                            if fa.mode == "write" or mode == "write":
                                conflicts.append(
                                    Conflict(
                                        agent_id=entry.id,
                                        file=fa.path,
                                        their_mode=fa.mode,
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
            scratchpad.entries = [e for e in scratchpad.entries if e.heartbeat_at > threshold]
            removed_count = original_count - len(scratchpad.entries)
            return scratchpad

        self._atomic_update(modifier)
        return removed_count

    # =========================================================================
    # Agent Management (v2)
    # =========================================================================

    def register_agent(
        self,
        agent_id: str,
        session_id: str,
        agent_type: str,
        task: str,
        parent_id: str | None = None,
    ) -> AgentEntry:
        """Register an agent in the scratchpad.

        Args:
            agent_id: Unique agent ID
            session_id: Underlying session ID
            agent_type: Type ID (explorer, summarizer, etc.)
            task: Task description
            parent_id: Parent agent ID if spawned by another agent

        Returns:
            The created entry
        """
        from activecontext.agents.schema import AgentEntry, AgentState

        now = datetime.now(timezone.utc)
        entry = AgentEntry(
            id=agent_id,
            session_id=session_id,
            agent_type=agent_type,
            task=task,
            parent_id=parent_id,
            state=AgentState.SPAWNED,
            created_at=now,
            updated_at=now,
            heartbeat_at=now,
        )

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            # Remove existing entry for this agent
            scratchpad.agents = [a for a in scratchpad.agents if a.id != agent_id]
            scratchpad.agents.append(entry)
            return scratchpad

        self._atomic_update(modifier)
        return entry

    def update_agent(
        self,
        agent_id: str,
        state: AgentState | None = None,
        task: str | None = None,
    ) -> AgentEntry | None:
        """Update an agent's entry.

        Args:
            agent_id: Agent ID to update
            state: New state
            task: New task description

        Returns:
            Updated entry, or None if not found
        """

        now = datetime.now(timezone.utc)
        updated_entry: AgentEntry | None = None

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            nonlocal updated_entry
            for entry in scratchpad.agents:
                if entry.id == agent_id:
                    if state is not None:
                        entry.state = state
                    if task is not None:
                        entry.task = task
                    entry.updated_at = now
                    entry.heartbeat_at = now
                    updated_entry = entry
                    break
            return scratchpad

        self._atomic_update(modifier)
        return updated_entry

    def agent_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat timestamp."""
        now = datetime.now(timezone.utc)

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            for entry in scratchpad.agents:
                if entry.id == agent_id:
                    entry.heartbeat_at = now
                    break
            return scratchpad

        self._atomic_update(modifier)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent's entry."""

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            scratchpad.agents = [a for a in scratchpad.agents if a.id != agent_id]
            # Also clean up messages for this agent
            scratchpad.messages = [
                m for m in scratchpad.messages if m.sender != agent_id and m.recipient != agent_id
            ]
            return scratchpad

        self._atomic_update(modifier)

    def get_agent(self, agent_id: str) -> AgentEntry | None:
        """Get an agent entry by ID."""
        scratchpad = self._load()
        for entry in scratchpad.agents:
            if entry.id == agent_id:
                return entry
        return None

    def get_all_agents(self) -> list[AgentEntry]:
        """Get all current agent entries."""
        scratchpad = self._load()
        return scratchpad.agents

    # =========================================================================
    # Message Management (v2)
    # =========================================================================

    def send_message(
        self,
        sender: str,
        recipient: str,
        content: str,
        node_refs: list[str] | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        """Send a message between agents.

        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            content: Message content
            node_refs: Node IDs referenced in content
            reply_to: Message ID this is replying to
            metadata: Additional metadata

        Returns:
            The created message
        """
        from activecontext.agents.schema import AgentMessage

        now = datetime.now(timezone.utc)
        message = AgentMessage(
            id=uuid.uuid4().hex[:12],
            sender=sender,
            recipient=recipient,
            content=content,
            node_refs=node_refs or [],
            created_at=now,
            status="pending",
            reply_to=reply_to,
            metadata=metadata or {},
        )

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            scratchpad.messages.append(message)
            return scratchpad

        self._atomic_update(modifier)
        return message

    def get_messages(
        self,
        recipient: str,
        status: str | None = "pending",
    ) -> list[AgentMessage]:
        """Get messages for an agent.

        Args:
            recipient: Recipient agent ID
            status: Filter by status (None for all)

        Returns:
            List of matching messages
        """
        scratchpad = self._load()
        messages = [m for m in scratchpad.messages if m.recipient == recipient]
        if status is not None:
            messages = [m for m in messages if m.status == status]
        return messages

    def mark_message_status(self, message_id: str, status: str) -> None:
        """Update a message's status.

        Args:
            message_id: Message ID
            status: New status (pending, delivered, read)
        """
        now = datetime.now(timezone.utc)

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            for message in scratchpad.messages:
                if message.id == message_id:
                    message.status = status
                    if status in ("delivered", "read") and message.delivered_at is None:
                        message.delivered_at = now
                    break
            return scratchpad

        self._atomic_update(modifier)

    def delete_old_messages(self, max_age: timedelta) -> int:
        """Delete messages older than max_age.

        Args:
            max_age: Maximum age for messages

        Returns:
            Number of messages deleted
        """
        now = datetime.now(timezone.utc)
        threshold = now - max_age
        deleted_count = 0

        def modifier(scratchpad: Scratchpad) -> Scratchpad:
            nonlocal deleted_count
            original_count = len(scratchpad.messages)
            scratchpad.messages = [m for m in scratchpad.messages if m.created_at > threshold]
            deleted_count = original_count - len(scratchpad.messages)
            return scratchpad

        self._atomic_update(modifier)
        return deleted_count
