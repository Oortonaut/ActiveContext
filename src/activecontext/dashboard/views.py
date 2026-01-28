"""View management for dashboard.

ViewManager provides named snapshots of view state (hide/expand per node)
that can be saved, restored, and synchronized with the agent's live state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.context.state import Expansion

if TYPE_CHECKING:
    from activecontext.session.session_manager import Session

_log = logging.getLogger(__name__)


@dataclass
class ViewSnapshot:
    """A named snapshot of view state.

    Captures the hide/expand state for all nodes at a point in time.
    """

    name: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # node_id -> {"hide": bool, "expand": str}
    node_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "node_states": self.node_states,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewSnapshot:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            node_states=data.get("node_states", {}),
        )


class ViewManager:
    """Manages named view snapshots for a session.

    View snapshots capture the hide/expand state of all nodes,
    allowing users to save and restore different "views" of the context.

    Operations:
    - clone: Snapshot current state with a new name
    - read: Copy agent's live state TO a saved view (update existing)
    - write: Apply a saved view's state TO the agent (restore)
    - delete: Remove a saved view
    - list: Get all saved views
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize the view manager.

        Args:
            storage_path: Optional path for persistent storage.
                         If None, views are only stored in memory.
        """
        self._views: dict[str, ViewSnapshot] = {}
        self._storage_path = storage_path

        # Load existing views if storage path exists
        if self._storage_path and self._storage_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Load views from disk storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, encoding="utf-8") as f:
                data = json.load(f)
                for view_data in data.get("views", []):
                    snapshot = ViewSnapshot.from_dict(view_data)
                    self._views[snapshot.name] = snapshot
            _log.debug(f"Loaded {len(self._views)} views from {self._storage_path}")
        except Exception as e:
            _log.warning(f"Failed to load views from {self._storage_path}: {e}")

    def _save_to_disk(self) -> None:
        """Save views to disk storage."""
        if not self._storage_path:
            return

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"views": [v.to_dict() for v in self._views.values()]}
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            _log.debug(f"Saved {len(self._views)} views to {self._storage_path}")
        except Exception as e:
            _log.warning(f"Failed to save views to {self._storage_path}: {e}")

    def list_views(self) -> list[dict[str, Any]]:
        """List all saved views.

        Returns:
            List of view metadata dicts with name, created_at, updated_at, node_count.
        """
        return [
            {
                "name": v.name,
                "created_at": v.created_at,
                "updated_at": v.updated_at,
                "node_count": len(v.node_states),
            }
            for v in sorted(self._views.values(), key=lambda x: x.created_at)
        ]

    def get_view(self, name: str) -> ViewSnapshot | None:
        """Get a view by name.

        Args:
            name: The view name.

        Returns:
            The ViewSnapshot or None if not found.
        """
        return self._views.get(name)

    def clone_view(self, name: str, session: Session) -> ViewSnapshot:
        """Snapshot current agent state with a new name.

        Args:
            name: Name for the new view.
            session: The session to snapshot.

        Returns:
            The created ViewSnapshot.

        Raises:
            ValueError: If a view with this name already exists.
        """
        if name in self._views:
            raise ValueError(f"View '{name}' already exists. Use read_view to update.")

        snapshot = self._capture_current_state(session, name)
        self._views[name] = snapshot
        self._save_to_disk()
        return snapshot

    def read_view(self, name: str, session: Session) -> ViewSnapshot:
        """Copy agent's live state TO an existing view (update).

        Args:
            name: Name of the view to update.
            session: The session to read state from.

        Returns:
            The updated ViewSnapshot.

        Raises:
            ValueError: If view doesn't exist.
        """
        if name not in self._views:
            raise ValueError(f"View '{name}' not found. Use clone_view to create.")

        snapshot = self._capture_current_state(session, name)
        snapshot.created_at = self._views[name].created_at  # Preserve original
        self._views[name] = snapshot
        self._save_to_disk()
        return snapshot

    def write_view(self, name: str, session: Session) -> int:
        """Apply saved view's state TO the agent (restore).

        Args:
            name: Name of the view to apply.
            session: The session to apply state to.

        Returns:
            Number of nodes updated.

        Raises:
            ValueError: If view doesn't exist.
        """
        if name not in self._views:
            raise ValueError(f"View '{name}' not found.")

        snapshot = self._views[name]
        return self._apply_state_to_session(snapshot, session)

    def delete_view(self, name: str) -> bool:
        """Delete a saved view.

        Args:
            name: Name of the view to delete.

        Returns:
            True if deleted, False if not found.
        """
        if name not in self._views:
            return False

        del self._views[name]
        self._save_to_disk()
        return True

    def _capture_current_state(self, session: Session, name: str) -> ViewSnapshot:
        """Capture the current view state from a session.

        Args:
            session: The session to capture.
            name: Name for the snapshot.

        Returns:
            ViewSnapshot with current state.
        """
        node_states: dict[str, dict[str, Any]] = {}

        try:
            views = session.timeline.views
            for node_id, view in views.items():
                node_states[node_id] = {
                    "hide": view.hide,
                    "expand": view.expand.value,
                }
        except Exception as e:
            _log.warning(f"Failed to capture view state: {e}")

        return ViewSnapshot(
            name=name,
            node_states=node_states,
        )

    def _apply_state_to_session(self, snapshot: ViewSnapshot, session: Session) -> int:
        """Apply a snapshot's state to a session.

        Args:
            snapshot: The snapshot to apply.
            session: The session to update.

        Returns:
            Number of nodes updated.
        """
        updated_count = 0

        try:
            views = session.timeline.views
            for node_id, state in snapshot.node_states.items():
                if node_id in views:
                    view = views[node_id]

                    # Update hide state
                    if "hide" in state:
                        view.hide = state["hide"]

                    # Update expand state
                    if "expand" in state:
                        try:
                            view.expand = Expansion(state["expand"])
                        except ValueError:
                            _log.warning(f"Invalid expansion value: {state['expand']}")

                    updated_count += 1
        except Exception as e:
            _log.warning(f"Failed to apply view state: {e}")

        return updated_count


# Global view managers per session
_view_managers: dict[str, ViewManager] = {}


def get_view_manager(session_id: str, storage_dir: Path | None = None) -> ViewManager:
    """Get or create a ViewManager for a session.

    Args:
        session_id: The session ID.
        storage_dir: Optional directory for persistent storage.

    Returns:
        The ViewManager for this session.
    """
    if session_id not in _view_managers:
        storage_path = None
        if storage_dir:
            storage_path = storage_dir / f"{session_id}_views.json"
        _view_managers[session_id] = ViewManager(storage_path)

    return _view_managers[session_id]


def cleanup_view_manager(session_id: str) -> None:
    """Remove the ViewManager for a session.

    Args:
        session_id: The session ID.
    """
    if session_id in _view_managers:
        del _view_managers[session_id]
