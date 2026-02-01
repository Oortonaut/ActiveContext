"""Tests for plugin management DSL functions in Timeline.

Tests cover:
1. plugin_load() - Load a plugin by node_type
2. plugin_unload() - Unload a plugin
3. plugin_available() - List available but not loaded plugins
4. plugin_info() - Get info about a registered plugin
5. Error handling for each function
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.registry import NodeTypeRegistry, PluginInfo
from activecontext.session.timeline import (
    PluginInUseError,
    PluginLoadError,
    PluginNotFoundError,
    Timeline,
)


@pytest.fixture
def timeline():
    """Create a Timeline instance for testing."""
    graph = ContextGraph()
    timeline = Timeline(
        session_id="test-session",
        context_graph=graph,
        cwd=".",
    )
    return timeline


@pytest.fixture
def mock_registry():
    """Create a mock NodeTypeRegistry."""
    registry = MagicMock(spec=NodeTypeRegistry)
    return registry


class TestPluginLoad:
    """Tests for plugin_load() DSL function."""

    def test_load_plugin_success(self, timeline: Timeline, mock_registry: MagicMock):
        """Test successfully loading a plugin."""
        plugin_info = PluginInfo(
            node_type="custom_lint",
            name="Custom Linter",
            description="A custom linting plugin",
            version="1.0.0",
            author="Test Author",
            source_path=Path("/path/to/plugin.py"),
            is_builtin=False,
            is_loaded=True,
        )

        mock_registry.get.return_value = None  # Not loaded initially
        mock_registry.load_from_module.return_value = "custom_lint"
        mock_registry.get_plugin_info.return_value = plugin_info

        with (
            patch("activecontext.context.registry.get_node_registry", return_value=mock_registry),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("custom_lint.py")]),
        ):
            result = timeline._plugin_load("custom_lint")

        assert result == plugin_info
        mock_registry.load_from_module.assert_called_once()

    def test_load_already_loaded_plugin(self, timeline: Timeline, mock_registry: MagicMock):
        """Test loading a plugin that is already loaded."""
        plugin_info = PluginInfo(
            node_type="custom_lint",
            name="Custom Linter",
            description="A custom linting plugin",
            version="1.0.0",
            author="Test Author",
            source_path=None,
            is_builtin=False,
            is_loaded=True,
        )

        mock_registry.get.return_value = MagicMock()  # Already loaded
        mock_registry.get_plugin_info.return_value = plugin_info

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_load("custom_lint")

        assert result == plugin_info
        mock_registry.load_from_module.assert_not_called()

    def test_load_plugin_not_found(self, timeline: Timeline, mock_registry: MagicMock):
        """Test loading a plugin that doesn't exist."""
        mock_registry.get.return_value = None

        with (
            patch("activecontext.context.registry.get_node_registry", return_value=mock_registry),
            patch("pathlib.Path.exists", return_value=False),
            pytest.raises(PluginNotFoundError, match="not found in plugin directories"),
        ):
            timeline._plugin_load("nonexistent")

    def test_load_plugin_import_error(self, timeline: Timeline, mock_registry: MagicMock):
        """Test loading a plugin that fails to import."""
        mock_registry.get.return_value = None
        mock_registry.load_from_module.side_effect = ImportError("Module not found")

        with (
            patch("activecontext.context.registry.get_node_registry", return_value=mock_registry),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.glob", return_value=[Path("broken.py")]),
            pytest.raises(PluginLoadError, match="Failed to load plugin"),
        ):
            timeline._plugin_load("broken")


class TestPluginUnload:
    """Tests for plugin_unload() DSL function."""

    def test_unload_plugin_success(self, timeline: Timeline, mock_registry: MagicMock):
        """Test successfully unloading a plugin."""
        mock_registry.unload_plugin.return_value = True

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_unload("custom_lint")

        assert result is True
        mock_registry.unload_plugin.assert_called_once_with(
            "custom_lint", force=False, has_active_nodes=False
        )

    def test_unload_plugin_not_found(self, timeline: Timeline, mock_registry: MagicMock):
        """Test unloading a plugin that isn't loaded."""
        mock_registry.unload_plugin.return_value = False

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_unload("nonexistent")

        assert result is False

    def test_unload_plugin_in_use(self, timeline: Timeline, mock_registry: MagicMock):
        """Test unloading a plugin with active nodes."""
        # Add a node of the custom type to the graph
        from activecontext.context.nodes import TextNode

        node = TextNode(node_id="custom_lint_1", path="test.py")
        # Override node_type to simulate a plugin node
        node._node_type = "custom_lint"  # Hack for testing
        timeline._context_graph.add_node(node)

        mock_registry.unload_plugin.side_effect = RuntimeError(
            "Cannot unload 'custom_lint': active nodes exist"
        )

        with (
            patch("activecontext.context.registry.get_node_registry", return_value=mock_registry),
            pytest.raises(PluginInUseError, match="active nodes exist"),
        ):
            timeline._plugin_unload("custom_lint")

    def test_unload_builtin_raises_error(self, timeline: Timeline, mock_registry: MagicMock):
        """Test unloading a builtin type raises ValueError."""
        mock_registry.unload_plugin.side_effect = ValueError("Cannot unload builtin node type")

        with (
            patch("activecontext.context.registry.get_node_registry", return_value=mock_registry),
            pytest.raises(ValueError, match="Cannot unload builtin"),
        ):
            timeline._plugin_unload("text")


class TestPluginAvailable:
    """Tests for plugin_available() DSL function."""

    def test_list_available_plugins(self, timeline: Timeline, mock_registry: MagicMock):
        """Test listing available but not loaded plugins."""
        discovered = [
            PluginInfo(
                node_type="custom_lint",
                name="Custom Linter",
                description="Linting plugin",
                version="1.0.0",
                author="Test",
                source_path=Path("/path/to/lint.py"),
                is_builtin=False,
                is_loaded=False,
            ),
            PluginInfo(
                node_type="text",
                name="Text Node",
                description="Builtin text node",
                version="1.0.0",
                author="ActiveContext",
                source_path=None,
                is_builtin=True,
                is_loaded=True,
            ),
        ]

        mock_registry.discover_plugins.return_value = discovered

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_available()

        # Should only return custom_lint (not loaded)
        assert result == ["custom_lint"]

    def test_list_available_empty(self, timeline: Timeline, mock_registry: MagicMock):
        """Test listing available plugins when all are loaded."""
        discovered = [
            PluginInfo(
                node_type="text",
                name="Text",
                description="Text node",
                version="1.0.0",
                author="ActiveContext",
                source_path=None,
                is_builtin=True,
                is_loaded=True,
            ),
        ]

        mock_registry.discover_plugins.return_value = discovered

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_available()

        assert result == []


class TestPluginInfo:
    """Tests for plugin_info() DSL function."""

    def test_get_plugin_info_success(self, timeline: Timeline, mock_registry: MagicMock):
        """Test getting info for a registered plugin."""
        plugin_info = PluginInfo(
            node_type="custom_lint",
            name="Custom Linter",
            description="A linting plugin",
            version="1.0.0",
            author="Test Author",
            source_path=Path("/path/to/lint.py"),
            is_builtin=False,
            is_loaded=True,
        )

        mock_registry.get_plugin_info.return_value = plugin_info

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_info("custom_lint")

        assert result == plugin_info
        mock_registry.get_plugin_info.assert_called_once_with("custom_lint")

    def test_get_plugin_info_not_found(self, timeline: Timeline, mock_registry: MagicMock):
        """Test getting info for an unregistered plugin."""
        mock_registry.get_plugin_info.return_value = None

        with (
            patch("activecontext.context.registry.get_node_registry", return_value=mock_registry),
            pytest.raises(PluginNotFoundError, match="is not registered"),
        ):
            timeline._plugin_info("nonexistent")

    def test_get_builtin_info(self, timeline: Timeline, mock_registry: MagicMock):
        """Test getting info for a builtin node type."""
        plugin_info = PluginInfo(
            node_type="text",
            name="Text Node",
            description="Builtin text node",
            version="1.0.0",
            author="ActiveContext",
            source_path=None,
            is_builtin=True,
            is_loaded=True,
        )

        mock_registry.get_plugin_info.return_value = plugin_info

        with patch("activecontext.context.registry.get_node_registry", return_value=mock_registry):
            result = timeline._plugin_info("text")

        assert result == plugin_info
        assert result.is_builtin is True
