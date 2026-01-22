"""Tests for the agent work coordination scratchpad system."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from activecontext.context.nodes import WorkNode
from activecontext.context.state import Expansion, TickFrequency
from activecontext.coordination import (
    Conflict,
    FileAccess,
    Scratchpad,
    ScratchpadManager,
    WorkEntry,
)


class TestFileAccess:
    """Tests for FileAccess dataclass."""

    def test_to_dict(self) -> None:
        fa = FileAccess(path="src/main.py", mode="write")
        assert fa.to_dict() == {"path": "src/main.py", "mode": "write"}

    def test_from_dict(self) -> None:
        fa = FileAccess.from_dict({"path": "src/main.py", "mode": "read"})
        assert fa.path == "src/main.py"
        assert fa.mode == "read"

    def test_from_dict_default_mode(self) -> None:
        fa = FileAccess.from_dict({"path": "src/main.py"})
        assert fa.mode == "read"


class TestWorkEntry:
    """Tests for WorkEntry dataclass."""

    def test_to_dict(self) -> None:
        entry = WorkEntry(
            id="abc12345",
            session_id="session-1",
            intent="Refactoring auth",
            status="active",
            files=[FileAccess("src/auth.py", "write")],
            dependencies=["tests/test_auth.py"],
        )
        d = entry.to_dict()
        assert d["id"] == "abc12345"
        assert d["session_id"] == "session-1"
        assert d["intent"] == "Refactoring auth"
        assert d["status"] == "active"
        assert len(d["files"]) == 1
        assert d["files"][0]["path"] == "src/auth.py"
        assert d["dependencies"] == ["tests/test_auth.py"]

    def test_from_dict(self) -> None:
        d = {
            "id": "abc12345",
            "session_id": "session-1",
            "intent": "Refactoring auth",
            "status": "paused",
            "files": [{"path": "src/auth.py", "mode": "write"}],
            "dependencies": [],
            "started_at": "2026-01-17T10:00:00+00:00",
            "updated_at": "2026-01-17T10:30:00+00:00",
            "heartbeat_at": "2026-01-17T10:30:00+00:00",
        }
        entry = WorkEntry.from_dict(d)
        assert entry.id == "abc12345"
        assert entry.status == "paused"
        assert len(entry.files) == 1


class TestScratchpad:
    """Tests for Scratchpad dataclass."""

    def test_to_dict(self) -> None:
        sp = Scratchpad(
            version=1,
            entries=[
                WorkEntry(
                    id="abc12345",
                    session_id="session-1",
                    intent="Test",
                )
            ],
        )
        d = sp.to_dict()
        assert d["version"] == 1
        assert len(d["entries"]) == 1

    def test_from_dict_empty(self) -> None:
        sp = Scratchpad.from_dict({})
        assert sp.version == 1
        assert sp.entries == []


class TestScratchpadManager:
    """Tests for ScratchpadManager."""

    def test_register_creates_entry(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        entry = manager.register(
            session_id="session-1",
            intent="Testing feature X",
            files=[FileAccess("src/x.py", "write")],
        )

        assert entry.intent == "Testing feature X"
        assert entry.status == "active"
        assert manager.agent_id is not None
        assert len(manager.agent_id) == 8

        # Check file was created
        scratchpad_path = tmp_path / ".ac" / "scratchpad.yaml"
        assert scratchpad_path.exists()

    def test_register_twice_updates(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        entry1 = manager.register(
            session_id="session-1",
            intent="First intent",
        )
        agent_id = manager.agent_id

        entry2 = manager.register(
            session_id="session-1",
            intent="Second intent",
        )

        # Same agent ID
        assert manager.agent_id == agent_id
        assert entry2.intent == "Second intent"

        # Should only have one entry
        entries = manager.get_all_entries()
        assert len(entries) == 1

    def test_update_entry(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        manager.register(
            session_id="session-1",
            intent="Initial intent",
        )

        updated = manager.update(intent="Updated intent", status="paused")
        assert updated is not None
        assert updated.intent == "Updated intent"
        assert updated.status == "paused"

    def test_update_without_register_returns_none(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        result = manager.update(intent="Something")
        assert result is None

    def test_heartbeat(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        manager.register(session_id="session-1", intent="Test")

        entries_before = manager.get_all_entries()
        old_heartbeat = entries_before[0].heartbeat_at

        time.sleep(0.01)  # Small delay
        manager.heartbeat()

        entries_after = manager.get_all_entries()
        new_heartbeat = entries_after[0].heartbeat_at

        assert new_heartbeat > old_heartbeat

    def test_unregister(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        manager.register(session_id="session-1", intent="Test")

        entries = manager.get_all_entries()
        assert len(entries) == 1

        manager.unregister()

        entries = manager.get_all_entries()
        assert len(entries) == 0
        assert manager.agent_id is None

    def test_get_conflicts_write_write(self, tmp_path: Path) -> None:
        # Agent 1 registers
        manager1 = ScratchpadManager(str(tmp_path))
        manager1.register(
            session_id="session-1",
            intent="Agent 1 work",
            files=[FileAccess("src/shared.py", "write")],
        )

        # Agent 2 checks for conflicts
        manager2 = ScratchpadManager(str(tmp_path))
        conflicts = manager2.get_conflicts(["src/shared.py"], mode="write")

        assert len(conflicts) == 1
        assert conflicts[0].file == "src/shared.py"
        assert conflicts[0].their_mode == "write"
        assert conflicts[0].their_intent == "Agent 1 work"

    def test_get_conflicts_read_write(self, tmp_path: Path) -> None:
        # Agent 1 writes to file
        manager1 = ScratchpadManager(str(tmp_path))
        manager1.register(
            session_id="session-1",
            intent="Agent 1 writing",
            files=[FileAccess("src/shared.py", "write")],
        )

        # Agent 2 wants to read - conflict because agent 1 is writing
        manager2 = ScratchpadManager(str(tmp_path))
        conflicts = manager2.get_conflicts(["src/shared.py"], mode="read")

        assert len(conflicts) == 1

    def test_get_conflicts_read_read_no_conflict(self, tmp_path: Path) -> None:
        # Agent 1 reads file
        manager1 = ScratchpadManager(str(tmp_path))
        manager1.register(
            session_id="session-1",
            intent="Agent 1 reading",
            files=[FileAccess("src/shared.py", "read")],
        )

        # Agent 2 also reads - no conflict
        manager2 = ScratchpadManager(str(tmp_path))
        conflicts = manager2.get_conflicts(["src/shared.py"], mode="read")

        assert len(conflicts) == 0

    def test_get_conflicts_glob_pattern(self, tmp_path: Path) -> None:
        # Agent 1 claims *.py in src/auth/
        manager1 = ScratchpadManager(str(tmp_path))
        manager1.register(
            session_id="session-1",
            intent="Working on auth module",
            files=[FileAccess("src/auth/*.py", "write")],
        )

        # Agent 2 wants specific file
        manager2 = ScratchpadManager(str(tmp_path))
        conflicts = manager2.get_conflicts(["src/auth/login.py"], mode="write")

        assert len(conflicts) == 1

    def test_get_conflicts_ignores_self(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        manager.register(
            session_id="session-1",
            intent="My work",
            files=[FileAccess("src/main.py", "write")],
        )

        # Same manager checking its own file - no conflict
        conflicts = manager.get_conflicts(["src/main.py"], mode="write")
        assert len(conflicts) == 0

    def test_get_conflicts_ignores_inactive(self, tmp_path: Path) -> None:
        # Agent 1 paused
        manager1 = ScratchpadManager(str(tmp_path))
        manager1.register(
            session_id="session-1",
            intent="Paused work",
            files=[FileAccess("src/main.py", "write")],
        )
        manager1.update(status="paused")

        # Agent 2 checks - no conflict with paused agent
        manager2 = ScratchpadManager(str(tmp_path))
        conflicts = manager2.get_conflicts(["src/main.py"], mode="write")

        assert len(conflicts) == 0

    def test_cleanup_stale(self, tmp_path: Path) -> None:
        manager = ScratchpadManager(str(tmp_path))
        manager.register(session_id="session-1", intent="Test")

        # Manually set heartbeat to past
        scratchpad = manager._load()
        scratchpad.entries[0].heartbeat_at = datetime.now(timezone.utc) - timedelta(
            seconds=600
        )
        manager._save(scratchpad)

        # Cleanup with 5 min threshold
        removed = manager.cleanup_stale(max_age_seconds=300)
        assert removed == 1

        entries = manager.get_all_entries()
        assert len(entries) == 0

    def test_get_all_entries_multiple_agents(self, tmp_path: Path) -> None:
        manager1 = ScratchpadManager(str(tmp_path))
        manager1.register(session_id="session-1", intent="Agent 1")

        manager2 = ScratchpadManager(str(tmp_path))
        manager2.register(session_id="session-2", intent="Agent 2")

        entries = manager1.get_all_entries()
        assert len(entries) == 2
        intents = {e.intent for e in entries}
        assert intents == {"Agent 1", "Agent 2"}


class TestWorkNode:
    """Tests for WorkNode context node."""

    def test_node_type(self) -> None:
        node = WorkNode(node_id="work_123", intent="Test work")
        assert node.node_type == "work"

    def test_get_digest(self) -> None:
        node = WorkNode(
            node_id="work_123",
            intent="Implementing feature",
            work_status="active",
            agent_id="abc12345",
            files=[{"path": "src/main.py", "mode": "write"}],
            conflicts=[{"agent_id": "def67890", "file": "src/main.py", "their_mode": "write", "their_intent": "Also editing"}],
        )
        digest = node.GetDigest()
        assert digest["type"] == "work"
        assert digest["intent"] == "Implementing feature"
        assert digest["status"] == "active"
        assert digest["file_count"] == 1
        assert digest["conflict_count"] == 1
        assert digest["agent_id"] == "abc12345"

    def test_render_collapsed(self) -> None:
        node = WorkNode(
            node_id="work_123",
            intent="Implementing a very long feature description that should be truncated",
            expansion=Expansion.COLLAPSED,
            files=[{"path": "src/main.py", "mode": "write"}],
        )
        rendered = node.Render()
        assert "Work:" in rendered  # Display name in new header format
        assert "(tokens:" in rendered  # New token breakdown format

    def test_render_details(self) -> None:
        node = WorkNode(
            node_id="work_123",
            intent="Test work",
            expansion=Expansion.DETAILS,
            agent_id="abc12345",
            files=[
                {"path": "src/main.py", "mode": "write"},
                {"path": "src/util.py", "mode": "read"},
            ],
        )
        rendered = node.Render()
        assert "Work: Test work [active]" in rendered  # New header format
        assert "(tokens:" in rendered  # Token breakdown
        assert "Agent: abc12345" in rendered
        assert "[W] src/main.py" in rendered
        assert "[R] src/util.py" in rendered

    def test_render_with_conflicts(self) -> None:
        node = WorkNode(
            node_id="work_123",
            intent="Test",
            expansion=Expansion.DETAILS,
            conflicts=[
                {
                    "agent_id": "def67890",
                    "file": "src/shared.py",
                    "their_mode": "write",
                    "their_intent": "Also editing",
                }
            ],
        )
        rendered = node.Render()
        assert "CONFLICTS" in rendered
        assert "def67890" in rendered
        assert "src/shared.py" in rendered

    def test_render_hidden(self) -> None:
        node = WorkNode(
            node_id="work_123",
            intent="Test",
            expansion=Expansion.HIDDEN,
        )
        assert node.Render() == ""

    def test_to_dict(self) -> None:
        node = WorkNode(
            node_id="work_123",
            intent="Test",
            work_status="paused",
            files=[{"path": "src/main.py", "mode": "write"}],
            dependencies=["tests/test_main.py"],
            conflicts=[],
            agent_id="abc12345",
        )
        d = node.to_dict()
        assert d["node_type"] == "work"
        assert d["intent"] == "Test"
        assert d["work_status"] == "paused"
        assert d["files"] == [{"path": "src/main.py", "mode": "write"}]
        assert d["dependencies"] == ["tests/test_main.py"]
        assert d["agent_id"] == "abc12345"

    def test_from_dict(self) -> None:
        d = {
            "node_id": "work_123",
            "node_type": "work",
            "intent": "Test",
            "work_status": "active",
            "files": [{"path": "src/main.py", "mode": "write"}],
            "dependencies": [],
            "conflicts": [],
            "agent_id": "abc12345",
            "state": "details",
            "tokens": 200,
        }
        node = WorkNode._from_dict(d)
        assert node.node_id == "work_123"
        assert node.intent == "Test"
        assert node.work_status == "active"
        assert node.agent_id == "abc12345"

    def test_from_dict_via_factory(self) -> None:
        """Test that ContextNode.from_dict dispatches to WorkNode."""
        from activecontext.context.nodes import ContextNode

        d = {
            "node_id": "work_123",
            "node_type": "work",
            "intent": "Test",
            "work_status": "active",
        }
        node = ContextNode.from_dict(d)
        assert isinstance(node, WorkNode)
        assert node.intent == "Test"

    def test_set_conflicts(self) -> None:
        node = WorkNode(node_id="work_123", intent="Test")
        assert node.version == 0

        node.set_conflicts([{"agent_id": "other", "file": "x.py", "their_mode": "write", "their_intent": "Test"}])
        assert len(node.conflicts) == 1
        assert node.version == 1  # Should have incremented

    def test_set_intent(self) -> None:
        node = WorkNode(node_id="work_123", intent="Original")
        node.set_intent("Updated")
        assert node.intent == "Updated"


class TestConflict:
    """Tests for Conflict dataclass."""

    def test_to_dict(self) -> None:
        c = Conflict(
            agent_id="abc12345",
            file="src/main.py",
            their_mode="write",
            their_intent="Editing main",
        )
        d = c.to_dict()
        assert d["agent_id"] == "abc12345"
        assert d["file"] == "src/main.py"
        assert d["their_mode"] == "write"
        assert d["their_intent"] == "Editing main"
