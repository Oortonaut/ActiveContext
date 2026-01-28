"""Tests for src/activecontext/dashboard/views.py"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from activecontext.context.state import Expansion
from activecontext.dashboard.views import (
    ViewSnapshot,
    ViewManager,
    get_view_manager,
    cleanup_view_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock session with timeline and views."""
    session = MagicMock()
    
    # Create mock views
    view1 = MagicMock()
    view1.hide = False
    view1.expand = Expansion.ALL
    
    view2 = MagicMock()
    view2.hide = True
    view2.expand = Expansion.HEADER
    
    session.timeline.views = {
        "node-1": view1,
        "node-2": view2,
    }
    
    return session


@pytest.fixture
def temp_storage():
    """Create a temporary file for storage testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"views": []}')
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink(missing_ok=True)


# =============================================================================
# Tests for ViewSnapshot
# =============================================================================


class TestViewSnapshot:
    """Tests for ViewSnapshot dataclass."""

    def test_creates_with_defaults(self):
        """Should create snapshot with default values."""
        snapshot = ViewSnapshot(name="test")
        
        assert snapshot.name == "test"
        assert snapshot.created_at > 0
        assert snapshot.updated_at > 0
        assert snapshot.node_states == {}

    def test_to_dict(self):
        """Should serialize to dictionary."""
        snapshot = ViewSnapshot(
            name="my-view",
            created_at=1000.0,
            updated_at=2000.0,
            node_states={"node-1": {"hide": True, "expand": "header"}},
        )
        
        result = snapshot.to_dict()
        
        assert result["name"] == "my-view"
        assert result["created_at"] == 1000.0
        assert result["updated_at"] == 2000.0
        assert result["node_states"]["node-1"]["hide"] is True

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "name": "restored",
            "created_at": 500.0,
            "updated_at": 600.0,
            "node_states": {"node-2": {"hide": False, "expand": "all"}},
        }
        
        snapshot = ViewSnapshot.from_dict(data)
        
        assert snapshot.name == "restored"
        assert snapshot.created_at == 500.0
        assert snapshot.node_states["node-2"]["expand"] == "all"


# =============================================================================
# Tests for ViewManager
# =============================================================================


class TestViewManager:
    """Tests for ViewManager class."""

    def test_creates_empty(self):
        """Should create with no views."""
        manager = ViewManager()
        
        assert manager.list_views() == []

    def test_clone_view(self, mock_session):
        """Should clone current view state with a name."""
        manager = ViewManager()
        
        snapshot = manager.clone_view("my-view", mock_session)
        
        assert snapshot.name == "my-view"
        assert "node-1" in snapshot.node_states
        assert snapshot.node_states["node-1"]["hide"] is False
        assert snapshot.node_states["node-1"]["expand"] == "all"
        assert snapshot.node_states["node-2"]["hide"] is True

    def test_clone_view_fails_for_existing_name(self, mock_session):
        """Should raise error when cloning to existing name."""
        manager = ViewManager()
        manager.clone_view("existing", mock_session)
        
        with pytest.raises(ValueError, match="already exists"):
            manager.clone_view("existing", mock_session)

    def test_list_views(self, mock_session):
        """Should list all saved views with metadata."""
        manager = ViewManager()
        manager.clone_view("view-1", mock_session)
        manager.clone_view("view-2", mock_session)
        
        views = manager.list_views()
        
        assert len(views) == 2
        assert views[0]["name"] == "view-1"
        assert views[1]["name"] == "view-2"
        assert "created_at" in views[0]
        assert "node_count" in views[0]

    def test_get_view(self, mock_session):
        """Should get view by name."""
        manager = ViewManager()
        manager.clone_view("test-view", mock_session)
        
        snapshot = manager.get_view("test-view")
        
        assert snapshot is not None
        assert snapshot.name == "test-view"

    def test_get_view_returns_none_for_unknown(self):
        """Should return None for unknown view name."""
        manager = ViewManager()
        
        assert manager.get_view("unknown") is None

    def test_read_view_updates_existing(self, mock_session):
        """Should update existing view from current state."""
        manager = ViewManager()
        
        # Clone initial state
        manager.clone_view("test", mock_session)
        original = manager.get_view("test")
        original_created_at = original.created_at
        
        # Modify the mock session views
        mock_session.timeline.views["node-1"].hide = True
        mock_session.timeline.views["node-1"].expand = Expansion.CONTENT
        
        # Read (update) the view
        snapshot = manager.read_view("test", mock_session)
        
        assert snapshot.node_states["node-1"]["hide"] is True
        assert snapshot.node_states["node-1"]["expand"] == "content"
        # created_at should be preserved
        assert snapshot.created_at == original_created_at

    def test_read_view_fails_for_unknown(self, mock_session):
        """Should raise error for unknown view name."""
        manager = ViewManager()
        
        with pytest.raises(ValueError, match="not found"):
            manager.read_view("unknown", mock_session)

    def test_write_view_applies_state(self, mock_session):
        """Should apply saved view state to session."""
        manager = ViewManager()
        
        # Clone current state
        manager.clone_view("saved", mock_session)
        
        # Change the session views
        mock_session.timeline.views["node-1"].hide = True
        mock_session.timeline.views["node-1"].expand = Expansion.HEADER
        
        # Apply saved state
        updated = manager.write_view("saved", mock_session)
        
        assert updated == 2  # Both nodes updated
        assert mock_session.timeline.views["node-1"].hide is False
        assert mock_session.timeline.views["node-1"].expand == Expansion.ALL

    def test_write_view_fails_for_unknown(self, mock_session):
        """Should raise error for unknown view name."""
        manager = ViewManager()
        
        with pytest.raises(ValueError, match="not found"):
            manager.write_view("unknown", mock_session)

    def test_delete_view(self, mock_session):
        """Should delete existing view."""
        manager = ViewManager()
        manager.clone_view("to-delete", mock_session)
        
        result = manager.delete_view("to-delete")
        
        assert result is True
        assert manager.get_view("to-delete") is None

    def test_delete_view_returns_false_for_unknown(self):
        """Should return False for unknown view name."""
        manager = ViewManager()
        
        result = manager.delete_view("unknown")
        
        assert result is False


class TestViewManagerPersistence:
    """Tests for ViewManager disk persistence."""

    def test_saves_to_disk(self, mock_session, temp_storage):
        """Should save views to disk on changes."""
        manager = ViewManager(storage_path=temp_storage)
        manager.clone_view("persisted", mock_session)
        
        # Read the file directly
        with open(temp_storage) as f:
            data = json.load(f)
        
        assert len(data["views"]) == 1
        assert data["views"][0]["name"] == "persisted"

    def test_loads_from_disk(self, temp_storage):
        """Should load existing views from disk."""
        # Write views directly
        with open(temp_storage, "w") as f:
            json.dump({
                "views": [
                    {
                        "name": "loaded-view",
                        "created_at": 100.0,
                        "updated_at": 100.0,
                        "node_states": {},
                    }
                ]
            }, f)
        
        manager = ViewManager(storage_path=temp_storage)
        
        views = manager.list_views()
        assert len(views) == 1
        assert views[0]["name"] == "loaded-view"

    def test_handles_missing_file(self):
        """Should handle missing storage file gracefully."""
        manager = ViewManager(storage_path=Path("/nonexistent/path.json"))
        
        assert manager.list_views() == []


# =============================================================================
# Tests for Module Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_view_manager_creates_new(self):
        """Should create new manager for unknown session."""
        # Clean up first
        cleanup_view_manager("test-session-new")
        
        manager = get_view_manager("test-session-new")
        
        assert manager is not None
        assert manager.list_views() == []
        
        # Clean up
        cleanup_view_manager("test-session-new")

    def test_get_view_manager_returns_existing(self, mock_session):
        """Should return same manager for same session."""
        cleanup_view_manager("test-session-same")
        
        manager1 = get_view_manager("test-session-same")
        manager1.clone_view("test", mock_session)
        
        manager2 = get_view_manager("test-session-same")
        
        assert manager1 is manager2
        assert len(manager2.list_views()) == 1
        
        cleanup_view_manager("test-session-same")

    def test_cleanup_view_manager(self, mock_session):
        """Should remove manager for session."""
        manager = get_view_manager("to-cleanup")
        manager.clone_view("test", mock_session)
        
        cleanup_view_manager("to-cleanup")
        
        # Should get a fresh manager
        new_manager = get_view_manager("to-cleanup")
        assert new_manager.list_views() == []
        
        cleanup_view_manager("to-cleanup")
