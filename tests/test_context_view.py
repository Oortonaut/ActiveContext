"""Tests for the context view module."""

from __future__ import annotations

import time
import pytest
from unittest.mock import Mock
from dataclasses import dataclass

from activecontext.context.view import AgentView, ViewRegistry
from activecontext.context.state import Expansion


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_content():
    """Create a mock ContentData."""
    content = Mock()
    content.content_type = "file"
    content.token_count = 100
    content.raw_content = "This is the file content.\nLine 2.\nLine 3."
    content.summary = "Summary of the file."
    content.summary_tokens = 10
    content.source_info = {"path": "/test/file.py"}
    return content


@pytest.fixture
def mock_shell_content():
    """Create a mock ContentData for shell output."""
    content = Mock()
    content.content_type = "shell"
    content.token_count = 50
    content.raw_content = "Command output here"
    content.summary = None
    content.summary_tokens = 0
    content.source_info = {"command": "ls -la /home/user/documents"}
    return content


@pytest.fixture
def mock_content_registry():
    """Create a mock ContentRegistry."""
    registry = Mock()
    return registry


@pytest.fixture
def view_registry(mock_content_registry):
    """Create a ViewRegistry with mock content registry."""
    return ViewRegistry(mock_content_registry)


# =============================================================================
# AgentView Initialization Tests
# =============================================================================


class TestAgentViewInit:
    """Tests for AgentView initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        view = AgentView()

        assert view.view_id is not None
        assert len(view.view_id) == 8
        assert view.agent_id == ""
        assert view.content_id == ""
        assert view.hidden is False
        assert view.expansion == Expansion.DETAILS
        assert view.tokens == 1000
        assert view.mode == "paused"
        assert view.parent_ids == set()
        assert view.children_ids == set()

    def test_post_init_sets_node_id(self):
        """Test that __post_init__ sets node_id from view_id."""
        view = AgentView(view_id="abc12345")

        assert view.node_id == "abc12345"

    def test_explicit_node_id_preserved(self):
        """Test that explicit node_id is preserved."""
        view = AgentView(view_id="view1234", node_id="custom_node")

        assert view.node_id == "custom_node"

    def test_custom_values(self):
        """Test initialization with custom values."""
        view = AgentView(
            view_id="test1234",
            agent_id="agent1",
            content_id="content1",
            hidden=True,
            expansion=Expansion.SUMMARY,
            tokens=500,
            mode="running",
        )

        assert view.view_id == "test1234"
        assert view.agent_id == "agent1"
        assert view.content_id == "content1"
        assert view.hidden is True
        assert view.expansion == Expansion.SUMMARY
        assert view.tokens == 500
        assert view.mode == "running"


# =============================================================================
# AgentView Render Tests
# =============================================================================


class TestAgentViewRender:
    """Tests for AgentView render method."""

    def test_render_hidden_shows_placeholder(self, mock_content):
        """Test render when hidden shows placeholder."""
        view = AgentView(hidden=True)

        result = view.render(mock_content)

        assert "[file: 100 tokens]" in result

    def test_render_collapsed_file(self, mock_content):
        """Test render collapsed view for file content."""
        view = AgentView(expansion=Expansion.COLLAPSED)

        result = view.render(mock_content)

        assert "File" in result
        assert "/test/file.py" in result
        assert "100 tokens" in result

    def test_render_collapsed_shell(self, mock_shell_content):
        """Test render collapsed view for shell content."""
        view = AgentView(expansion=Expansion.COLLAPSED)

        result = view.render(mock_shell_content)

        assert "Shell" in result
        assert "ls -la" in result
        assert "50 tokens" in result

    def test_render_collapsed_generic(self):
        """Test render collapsed view for generic content."""
        content = Mock()
        content.content_type = "artifact"
        content.token_count = 25
        content.source_info = {}

        view = AgentView(expansion=Expansion.COLLAPSED)

        result = view.render(content)

        assert "Artifact" in result
        assert "25 tokens" in result

    def test_render_summary(self, mock_content):
        """Test render summary view."""
        view = AgentView(expansion=Expansion.SUMMARY)

        result = view.render(mock_content)

        assert "Summary of the file." in result
        assert "100 tokens total" in result

    def test_render_summary_no_summary(self, mock_shell_content):
        """Test render summary view when no summary available."""
        view = AgentView(expansion=Expansion.SUMMARY)

        result = view.render(mock_shell_content)

        assert "no summary" in result

    def test_render_summary_truncation(self, mock_content):
        """Test render summary view with truncation."""
        mock_content.summary = "A" * 1000
        mock_content.summary_tokens = 200
        view = AgentView(expansion=Expansion.SUMMARY, tokens=50)

        result = view.render(mock_content)

        assert len(result) < 1000
        assert "..." in result

    def test_render_details(self, mock_content):
        """Test render details view."""
        view = AgentView(expansion=Expansion.DETAILS, tokens=200)

        result = view.render(mock_content)

        assert result == mock_content.raw_content

    def test_render_details_truncation(self, mock_content):
        """Test render details view with truncation."""
        mock_content.raw_content = "X" * 1000
        mock_content.token_count = 200
        view = AgentView(expansion=Expansion.DETAILS, tokens=50)

        result = view.render(mock_content)

        assert "truncated" in result
        assert len(result) < 1000

    def test_render_details_full(self, mock_content):
        """Test render DETAILS view shows full content."""
        view = AgentView(expansion=Expansion.DETAILS, tokens=200)

        result = view.render(mock_content)

        # DETAILS shows raw content without summary section
        assert mock_content.raw_content in result

    def test_render_details_shell(self, mock_shell_content):
        """Test render DETAILS view with shell content."""
        view = AgentView(expansion=Expansion.DETAILS, tokens=200)

        result = view.render(mock_shell_content)

        # DETAILS shows raw content
        assert mock_shell_content.raw_content in result

    def test_render_hidden_expansion_legacy(self):
        """Test render with HIDDEN expansion (legacy)."""
        content = Mock()
        content.content_type = "file"
        content.token_count = 50

        view = AgentView(expansion=Expansion.HIDDEN)

        result = view.render(content)

        assert "[file: 50 tokens]" in result

    def test_render_uses_budget_parameter(self, mock_content):
        """Test render uses budget parameter over self.tokens."""
        view = AgentView(expansion=Expansion.DETAILS, tokens=1000)

        mock_content.raw_content = "X" * 100
        mock_content.token_count = 50

        result = view.render(mock_content, budget=10)

        # Should truncate based on budget, not self.tokens
        assert "truncated" in result


# =============================================================================
# AgentView Fluent API Tests
# =============================================================================


class TestAgentViewFluentAPI:
    """Tests for AgentView fluent API methods."""

    def test_set_hidden_returns_self(self):
        """Test SetHidden returns self for chaining."""
        view = AgentView()

        result = view.SetHidden(True)

        assert result is view
        assert view.hidden is True

    def test_set_hidden_updates_timestamp(self):
        """Test SetHidden updates updated_at."""
        view = AgentView()
        original_time = view.updated_at

        time.sleep(0.01)
        view.SetHidden(True)

        assert view.updated_at > original_time

    def test_set_expansion_returns_self(self):
        """Test SetState returns self for chaining."""
        view = AgentView()

        result = view.SetExpansion(Expansion.SUMMARY)

        assert result is view
        assert view.expansion == Expansion.SUMMARY

    def test_set_expansion_updates_timestamp(self):
        """Test SetState updates updated_at."""
        view = AgentView()
        original_time = view.updated_at

        time.sleep(0.01)
        view.SetExpansion(Expansion.DETAILS)

        assert view.updated_at > original_time

    def test_set_tokens_returns_self(self):
        """Test SetTokens returns self for chaining."""
        view = AgentView()

        result = view.SetTokens(500)

        assert result is view
        assert view.tokens == 500

    def test_set_tokens_updates_timestamp(self):
        """Test SetTokens updates updated_at."""
        view = AgentView()
        original_time = view.updated_at

        time.sleep(0.01)
        view.SetTokens(2000)

        assert view.updated_at > original_time

    def test_run_returns_self(self):
        """Test Run returns self for chaining."""
        view = AgentView()

        result = view.Run()

        assert result is view
        assert view.mode == "running"

    def test_pause_returns_self(self):
        """Test Pause returns self for chaining."""
        view = AgentView(mode="running")

        result = view.Pause()

        assert result is view
        assert view.mode == "paused"

    def test_chaining_multiple_methods(self):
        """Test chaining multiple fluent methods."""
        view = AgentView()

        result = (
            view.SetHidden(False)
            .SetExpansion(Expansion.SUMMARY)
            .SetTokens(750)
            .Run()
        )

        assert result is view
        assert view.hidden is False
        assert view.expansion == Expansion.SUMMARY
        assert view.tokens == 750
        assert view.mode == "running"


# =============================================================================
# AgentView Serialization Tests
# =============================================================================


class TestAgentViewSerialization:
    """Tests for AgentView to_dict/from_dict."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        view = AgentView(
            view_id="test1234",
            agent_id="agent1",
            content_id="content1",
            node_id="node1",
            hidden=True,
            expansion=Expansion.SUMMARY,
            tokens=500,
            parent_ids={"p1", "p2"},
            children_ids={"c1"},
            mode="running",
            tags={"key": "value"},
        )

        result = view.to_dict()

        assert result["view_id"] == "test1234"
        assert result["agent_id"] == "agent1"
        assert result["content_id"] == "content1"
        assert result["node_id"] == "node1"
        assert result["hidden"] is True
        assert result["expansion"] == "summary"
        assert result["tokens"] == 500
        assert set(result["parent_ids"]) == {"p1", "p2"}
        assert result["children_ids"] == ["c1"]
        assert result["mode"] == "running"
        assert result["tags"] == {"key": "value"}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "view_id": "test1234",
            "agent_id": "agent1",
            "content_id": "content1",
            "node_id": "node1",
            "hidden": True,
            "expansion": "summary",
            "tokens": 500,
            "parent_ids": ["p1", "p2"],
            "children_ids": ["c1"],
            "mode": "running",
            "created_at": 1000.0,
            "updated_at": 2000.0,
            "tags": {"key": "value"},
        }

        view = AgentView.from_dict(data)

        assert view.view_id == "test1234"
        assert view.agent_id == "agent1"
        assert view.content_id == "content1"
        assert view.node_id == "node1"
        assert view.hidden is True
        assert view.expansion == Expansion.SUMMARY
        assert view.tokens == 500
        assert view.parent_ids == {"p1", "p2"}
        assert view.children_ids == {"c1"}
        assert view.mode == "running"
        assert view.tags == {"key": "value"}

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing keys."""
        data = {"view_id": "min12345"}

        view = AgentView.from_dict(data)

        assert view.view_id == "min12345"
        assert view.agent_id == ""
        assert view.hidden is False
        assert view.expansion == Expansion.DETAILS
        assert view.tokens == 1000

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = AgentView(
            view_id="round123",
            agent_id="agent1",
            expansion=Expansion.DETAILS,
            hidden=True,
            parent_ids={"p1"},
        )

        data = original.to_dict()
        restored = AgentView.from_dict(data)

        assert restored.view_id == original.view_id
        assert restored.agent_id == original.agent_id
        assert restored.expansion == original.expansion
        assert restored.hidden == original.hidden
        assert restored.parent_ids == original.parent_ids


# =============================================================================
# ViewRegistry Tests
# =============================================================================


class TestViewRegistryInit:
    """Tests for ViewRegistry initialization."""

    def test_init(self, mock_content_registry):
        """Test initialization."""
        registry = ViewRegistry(mock_content_registry)

        assert registry._content_registry is mock_content_registry
        assert len(registry) == 0


class TestViewRegistryRegister:
    """Tests for ViewRegistry.register method."""

    def test_register_returns_view_id(self, view_registry):
        """Test register returns view_id."""
        view = AgentView(view_id="test1234", agent_id="agent1")

        result = view_registry.register(view)

        assert result == "test1234"

    def test_register_indexes_by_agent(self, view_registry):
        """Test register indexes by agent_id."""
        view = AgentView(view_id="test1234", agent_id="agent1")

        view_registry.register(view)

        assert "agent1" in view_registry._by_agent
        assert "test1234" in view_registry._by_agent["agent1"]

    def test_register_indexes_by_node(self, view_registry):
        """Test register indexes by node_id."""
        view = AgentView(view_id="test1234", node_id="node5678")

        view_registry.register(view)

        assert view_registry._by_node["node5678"] == "test1234"


class TestViewRegistryGet:
    """Tests for ViewRegistry.get method."""

    def test_get_existing(self, view_registry):
        """Test get returns existing view."""
        view = AgentView(view_id="test1234")
        view_registry.register(view)

        result = view_registry.get("test1234")

        assert result is view

    def test_get_nonexistent(self, view_registry):
        """Test get returns None for nonexistent view."""
        result = view_registry.get("nonexistent")

        assert result is None


class TestViewRegistryGetByNode:
    """Tests for ViewRegistry.get_by_node method."""

    def test_get_by_node_existing(self, view_registry):
        """Test get_by_node returns view by node_id."""
        view = AgentView(view_id="view1234", node_id="node5678")
        view_registry.register(view)

        result = view_registry.get_by_node("node5678")

        assert result is view

    def test_get_by_node_nonexistent(self, view_registry):
        """Test get_by_node returns None for nonexistent node_id."""
        result = view_registry.get_by_node("nonexistent")

        assert result is None


class TestViewRegistryGetAgentViews:
    """Tests for ViewRegistry.get_agent_views method."""

    def test_get_agent_views(self, view_registry):
        """Test get_agent_views returns all views for agent."""
        view1 = AgentView(view_id="view1", agent_id="agent1")
        view2 = AgentView(view_id="view2", agent_id="agent1")
        view3 = AgentView(view_id="view3", agent_id="agent2")

        view_registry.register(view1)
        view_registry.register(view2)
        view_registry.register(view3)

        result = view_registry.get_agent_views("agent1")

        assert len(result) == 2
        assert view1 in result
        assert view2 in result
        assert view3 not in result

    def test_get_agent_views_empty(self, view_registry):
        """Test get_agent_views returns empty for unknown agent."""
        result = view_registry.get_agent_views("unknown")

        assert result == []


class TestViewRegistryRemove:
    """Tests for ViewRegistry.remove method."""

    def test_remove_existing(self, view_registry):
        """Test remove existing view."""
        view = AgentView(view_id="test1234", agent_id="agent1", node_id="node5678")
        view_registry.register(view)

        result = view_registry.remove("test1234")

        assert result is True
        assert view_registry.get("test1234") is None
        assert view_registry.get_by_node("node5678") is None

    def test_remove_nonexistent(self, view_registry):
        """Test remove nonexistent view."""
        result = view_registry.remove("nonexistent")

        assert result is False


class TestViewRegistryRenderView:
    """Tests for ViewRegistry.render_view method."""

    def test_render_view_success(self, view_registry, mock_content_registry, mock_content):
        """Test render_view returns rendered content."""
        view = AgentView(
            view_id="test1234",
            content_id="content1",
            expansion=Expansion.DETAILS,
        )
        view_registry.register(view)
        mock_content_registry.get.return_value = mock_content

        result = view_registry.render_view("test1234")

        assert result == mock_content.raw_content

    def test_render_view_missing_view(self, view_registry):
        """Test render_view returns None for missing view."""
        result = view_registry.render_view("nonexistent")

        assert result is None

    def test_render_view_missing_content(self, view_registry, mock_content_registry):
        """Test render_view handles missing content."""
        view = AgentView(view_id="test1234", content_id="missing")
        view_registry.register(view)
        mock_content_registry.get.return_value = None

        result = view_registry.render_view("test1234")

        assert "not found" in result

    def test_render_view_with_budget(self, view_registry, mock_content_registry, mock_content):
        """Test render_view passes budget parameter."""
        mock_content.raw_content = "X" * 1000
        mock_content.token_count = 200
        view = AgentView(
            view_id="test1234",
            content_id="content1",
            expansion=Expansion.DETAILS,
            tokens=1000,
        )
        view_registry.register(view)
        mock_content_registry.get.return_value = mock_content

        result = view_registry.render_view("test1234", budget=50)

        assert "truncated" in result


class TestViewRegistryLen:
    """Tests for ViewRegistry.__len__."""

    def test_len_empty(self, view_registry):
        """Test len of empty registry."""
        assert len(view_registry) == 0

    def test_len_with_views(self, view_registry):
        """Test len with registered views."""
        view_registry.register(AgentView(view_id="v1"))
        view_registry.register(AgentView(view_id="v2"))
        view_registry.register(AgentView(view_id="v3"))

        assert len(view_registry) == 3
