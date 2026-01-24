"""Work coordination manager extracted from Timeline.

This module handles multi-agent work coordination through a project-wide
scratchpad that tracks which agents are working on which files.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import WorkNode
from activecontext.context.state import Expansion

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.coordination.scratchpad import ScratchpadManager


class WorkCoordinator:
    """Manages work coordination for a Timeline.

    Extracted from Timeline to reduce class size and improve cohesion.
    Handles registration, conflict detection, and status updates for
    multi-agent file access coordination.
    """

    def __init__(
        self,
        session_id: str,
        context_graph: ContextGraph,
        scratchpad_manager: ScratchpadManager | None,
    ) -> None:
        """Initialize WorkCoordinator.

        Args:
            session_id: Session identifier for this agent
            context_graph: Graph for adding WorkNodes
            scratchpad_manager: Manager for cross-agent coordination
        """
        self._session_id = session_id
        self._context_graph = context_graph
        self._scratchpad_manager = scratchpad_manager
        self._work_node: WorkNode | None = None

    @property
    def work_node(self) -> WorkNode | None:
        """Get the current WorkNode, if any."""
        return self._work_node

    def work_on(
        self,
        intent: str,
        *files: str,
        mode: str = "write",
        dependencies: list[str] | None = None,
    ) -> WorkNode:
        """Register intent and files being worked on.

        Creates a WorkNode in the context graph to display coordination status.
        Also registers with the project-wide scratchpad for cross-agent visibility.

        Args:
            intent: Human-readable description of work
            *files: File paths being accessed
            mode: Access mode for files ("read" or "write")
            dependencies: Additional files needed (read-only)

        Returns:
            WorkNode for displaying coordination status

        Example:
            work_on("Implementing OAuth2", "src/auth/oauth.py", "src/auth/config.py")
        """
        from activecontext.coordination.schema import FileAccess

        if not self._scratchpad_manager:
            raise RuntimeError("Scratchpad manager not configured")

        # Build file access list
        file_accesses = [FileAccess(path=f, mode=mode) for f in files]

        # Register with scratchpad
        entry = self._scratchpad_manager.register(
            session_id=self._session_id,
            intent=intent,
            files=file_accesses,
            dependencies=dependencies,
        )

        # Check for conflicts
        all_paths = list(files) + (dependencies or [])
        conflicts = self._scratchpad_manager.get_conflicts(all_paths, mode)

        # Create or update WorkNode
        if self._work_node is None:
            node_id = f"work_{uuid.uuid4().hex[:8]}"
            self._work_node = WorkNode(
                node_id=node_id,
                tokens=200,
                expansion=Expansion.ALL,
                intent=entry.intent,
                work_status=entry.status,
                files=[f.to_dict() for f in file_accesses],
                dependencies=dependencies or [],
                conflicts=[c.to_dict() for c in conflicts],
                agent_id=entry.id,
            )
            self._context_graph.add_node(self._work_node)
        else:
            self._work_node.intent = entry.intent
            self._work_node.work_status = entry.status
            self._work_node.files = [f.to_dict() for f in file_accesses]
            self._work_node.dependencies = dependencies or []
            self._work_node.set_conflicts([c.to_dict() for c in conflicts])
            self._work_node.agent_id = entry.id

        return self._work_node

    def work_check(self, *files: str, mode: str = "write") -> list[dict[str, str]]:
        """Check for conflicts on files before modifying.

        Args:
            *files: File paths to check
            mode: Access mode we want ("read" or "write")

        Returns:
            List of conflicts: [{agent_id, file, their_mode, their_intent}, ...]

        Example:
            conflicts = work_check("src/auth/utils.py")
            if conflicts:
                print(f"Warning: {conflicts[0]['agent_id']} is working on this")
        """
        if not self._scratchpad_manager:
            return []

        conflicts = self._scratchpad_manager.get_conflicts(list(files), mode)
        return [c.to_dict() for c in conflicts]

    def work_update(
        self,
        intent: str | None = None,
        files: list[str] | None = None,
        mode: str = "write",
        dependencies: list[str] | None = None,
        status: str | None = None,
    ) -> WorkNode | None:
        """Update current work registration.

        Args:
            intent: New intent description
            files: New file list (replaces existing)
            mode: Access mode for new files
            dependencies: New dependencies
            status: New status (active/paused/done)

        Returns:
            Updated WorkNode, or None if not registered
        """
        from activecontext.coordination.schema import FileAccess

        if not self._scratchpad_manager:
            return None

        # Build file access list if provided
        file_accesses: list[FileAccess] | None = None
        if files is not None:
            file_accesses = [FileAccess(path=f, mode=mode) for f in files]

        # Update scratchpad entry
        entry = self._scratchpad_manager.update(
            intent=intent,
            files=file_accesses,
            dependencies=dependencies,
            status=status,
        )

        if entry is None:
            return None

        # Update WorkNode
        if self._work_node:
            if intent is not None:
                self._work_node.intent = intent
            if file_accesses is not None:
                self._work_node.files = [f.to_dict() for f in file_accesses]
            if dependencies is not None:
                self._work_node.dependencies = dependencies
            if status is not None:
                self._work_node.work_status = status

            # Refresh conflicts
            all_paths = [f["path"] for f in self._work_node.files]
            conflicts = self._scratchpad_manager.get_conflicts(all_paths, mode)
            self._work_node.set_conflicts([c.to_dict() for c in conflicts])

        return self._work_node

    def work_done(self) -> None:
        """Mark work as complete and unregister.

        Removes this agent's entry from the scratchpad and hides the WorkNode.
        """
        if not self._scratchpad_manager:
            return

        self._scratchpad_manager.unregister()

        if self._work_node:
            self._work_node.work_status = "done"
            self._work_node.expansion = Expansion.HEADER

    def work_list(self) -> list[dict[str, Any]]:
        """List all active work entries from all agents.

        Returns:
            List of work entries with files, intent, status, etc.
        """
        if not self._scratchpad_manager:
            return []

        entries = self._scratchpad_manager.get_all_entries()
        return [
            {
                "agent_id": e.id,
                "session_id": e.session_id,
                "intent": e.intent,
                "status": e.status,
                "files": [f.to_dict() for f in e.files],
                "dependencies": e.dependencies,
                "started_at": e.started_at.isoformat(),
                "updated_at": e.updated_at.isoformat(),
            }
            for e in entries
        ]
