"""Tests for plugin node type integration with help system.

Covers:
1. Plugin nodes with plugin_info dict get help content from dict
2. Plugin nodes without plugin_info fall back to docstrings
3. Plugin help content includes version and author metadata
4. .help() method works for plugin nodes same as builtin nodes
5. Help content is cached after first generation
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import pytest

from activecontext.context.exposed import exposed
from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import ContextNode, HelpNode, _extract_help_content
from activecontext.context.registry import get_node_registry

# ---------------------------------------------------------------------------
# Test Plugin Node Types
# ---------------------------------------------------------------------------


@dataclass
class CustomLintNode(ContextNode):
    """A custom linting node for testing.

    This node performs custom linting on code files.
    """

    file_path: str = ""
    severity: str = "warning"

    @property
    def node_type(self) -> str:
        return "custom_lint"

    @exposed
    def SetSeverity(self, severity: str) -> CustomLintNode:
        """Set the lint severity level."""
        self.severity = severity
        return self

    @exposed
    @property
    def is_complete(self) -> bool:
        """Check if linting is complete."""
        return True

    def render_digest(self) -> str:
        """Render digest for the node."""
        return f"CustomLint: {self.file_path}"

    def get_token_breakdown(self) -> dict[str, int]:
        """Get token usage breakdown."""
        return {"total": 100, "header": 20, "content": 80}

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "file_path": self.file_path,
            "severity": self.severity,
        }

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        return (
            f"CustomLint: {self.file_path}\nSeverity: {self.severity}\nComplete: {self.is_complete}"
        )


@dataclass
class NoPluginInfoNode(ContextNode):
    """A plugin node without plugin_info dict.

    Falls back to docstring extraction.
    """

    value: int = 0

    @property
    def node_type(self) -> str:
        return "no_plugin_info"

    def render_digest(self) -> str:
        """Render digest for the node."""
        return f"NoPluginInfo: {self.value}"

    def get_token_breakdown(self) -> dict[str, int]:
        """Get token usage breakdown."""
        return {"total": 50, "header": 10, "content": 40}

    def GetDigest(self) -> dict[str, Any]:
        return {"id": self.node_id, "type": self.node_type, "value": self.value}

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        return f"NoPluginInfo: {self.value}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph() -> ContextGraph:
    """Create a fresh ContextGraph for testing."""
    return ContextGraph()


@pytest.fixture
def mock_plugin_module():
    """Create a mock module with plugin_info dict."""
    import types

    # Create a mock module with plugin_info
    mock_module = types.ModuleType("test_plugin_module")
    mock_module.plugin_info = {
        "node_type": "custom_lint",
        "name": "Custom Linter",
        "description": "A custom linting plugin for code quality checks",
        "version": "1.0.0",
        "author": "Test Author",
    }
    mock_module.CustomLintNode = CustomLintNode

    # Register it in sys.modules
    sys.modules["test_plugin_module"] = mock_module

    # Update the CustomLintNode class to use this module
    CustomLintNode.__module__ = "test_plugin_module"

    yield mock_module

    # Cleanup
    del sys.modules["test_plugin_module"]
    # Reset module back (though it will be test module anyway)


@pytest.fixture
def custom_lint_node(graph: ContextGraph, mock_plugin_module) -> CustomLintNode:
    """Create a CustomLintNode with plugin_info available."""
    node = CustomLintNode(file_path="main.py", severity="error")
    graph.add_node(node)
    return node


@pytest.fixture
def no_info_node(graph: ContextGraph) -> NoPluginInfoNode:
    """Create a NoPluginInfoNode without plugin_info."""
    node = NoPluginInfoNode(value=42)
    graph.add_node(node)
    return node


# ---------------------------------------------------------------------------
# 1. Plugin nodes with plugin_info dict get help content from dict
# ---------------------------------------------------------------------------


class TestPluginInfoExtraction:
    """Tests for plugin_info dict priority in help extraction."""

    def test_extract_uses_plugin_info_name(self, mock_plugin_module) -> None:
        """Help extraction uses plugin_info name instead of class name."""
        content = _extract_help_content(CustomLintNode)
        assert "Custom Linter" in content
        # Should not use class name as header
        assert "# Custom Linter" in content

    def test_extract_uses_plugin_info_description(self, mock_plugin_module) -> None:
        """Help extraction uses plugin_info description."""
        content = _extract_help_content(CustomLintNode)
        assert "custom linting plugin for code quality checks" in content

    def test_extract_includes_version(self, mock_plugin_module) -> None:
        """Help extraction includes plugin version from plugin_info."""
        content = _extract_help_content(CustomLintNode)
        assert "Version" in content
        assert "1.0.0" in content

    def test_extract_includes_author(self, mock_plugin_module) -> None:
        """Help extraction includes plugin author from plugin_info."""
        content = _extract_help_content(CustomLintNode)
        assert "Author" in content
        assert "Test Author" in content

    def test_extract_includes_plugin_information_section(self, mock_plugin_module) -> None:
        """Help extraction includes Plugin Information section."""
        content = _extract_help_content(CustomLintNode)
        assert "## Plugin Information" in content


# ---------------------------------------------------------------------------
# 2. Plugin nodes without plugin_info fall back to docstrings
# ---------------------------------------------------------------------------


class TestFallbackToDocstring:
    """Tests for fallback to docstring when plugin_info is missing."""

    def test_extract_uses_class_name_without_plugin_info(self) -> None:
        """Without plugin_info, uses class name as header."""
        content = _extract_help_content(NoPluginInfoNode)
        assert "# NoPluginInfoNode" in content

    def test_extract_uses_docstring_without_plugin_info(self) -> None:
        """Without plugin_info, extracts description from docstring."""
        content = _extract_help_content(NoPluginInfoNode)
        assert "plugin node without plugin_info" in content

    def test_no_plugin_information_section_without_info(self) -> None:
        """Without plugin_info, no Plugin Information section is added."""
        content = _extract_help_content(NoPluginInfoNode)
        assert "## Plugin Information" not in content


# ---------------------------------------------------------------------------
# 3. Plugin help content includes constructor parameters and methods
# ---------------------------------------------------------------------------


class TestPluginHelpContentStructure:
    """Tests for complete help content structure for plugins."""

    def test_includes_constructor_parameters(self, mock_plugin_module) -> None:
        """Plugin help includes constructor parameters."""
        content = _extract_help_content(CustomLintNode)
        assert "## Constructor Parameters" in content
        assert "file_path" in content
        assert "severity" in content

    def test_includes_methods(self, mock_plugin_module) -> None:
        """Plugin help includes method documentation."""
        content = _extract_help_content(CustomLintNode)
        assert "## Methods" in content
        # Check for exposed method
        assert "SetSeverity" in content

    def test_includes_properties(self, mock_plugin_module) -> None:
        """Plugin help includes property documentation."""
        content = _extract_help_content(CustomLintNode)
        assert "## Properties" in content
        assert "is_complete" in content

    def test_includes_exposed_api_section(self, mock_plugin_module) -> None:
        """Plugin help includes @exposed API section."""
        content = _extract_help_content(CustomLintNode)
        assert "## Agent-Facing API (@exposed)" in content
        assert "SetSeverity" in content
        assert "is_complete" in content


# ---------------------------------------------------------------------------
# 4. .help() method works for plugin nodes same as builtin nodes
# ---------------------------------------------------------------------------


class TestPluginNodeHelpMethod:
    """Tests that .help() method works for plugin nodes."""

    def test_help_creates_helpnode_for_plugin(
        self, custom_lint_node: CustomLintNode, mock_plugin_module
    ) -> None:
        """Calling .help() on plugin node creates HelpNode child."""
        help_node = custom_lint_node.help()
        assert isinstance(help_node, HelpNode)
        assert help_node.parent_node_type == "custom_lint"
        assert help_node.node_id in custom_lint_node.children_ids

    def test_help_content_uses_plugin_info(
        self, custom_lint_node: CustomLintNode, mock_plugin_module
    ) -> None:
        """Help content for plugin node uses plugin_info."""
        help_node = custom_lint_node.help()
        assert "Custom Linter" in help_node._help_content
        assert "1.0.0" in help_node._help_content
        assert "Test Author" in help_node._help_content

    def test_help_idempotent_for_plugin(
        self, custom_lint_node: CustomLintNode, mock_plugin_module
    ) -> None:
        """Calling .help() multiple times returns same HelpNode for plugin."""
        first = custom_lint_node.help()
        second = custom_lint_node.help()
        assert first is second

    def test_help_for_plugin_without_info(self, no_info_node: NoPluginInfoNode) -> None:
        """Plugin node without plugin_info still gets help from docstring."""
        help_node = no_info_node.help()
        assert isinstance(help_node, HelpNode)
        assert "NoPluginInfoNode" in help_node._help_content
        assert "without plugin_info" in help_node._help_content


# ---------------------------------------------------------------------------
# 5. Help content is cached after first generation
# ---------------------------------------------------------------------------


class TestHelpCaching:
    """Tests that help content is generated once and cached."""

    def test_help_content_cached_in_helpnode(
        self, custom_lint_node: CustomLintNode, mock_plugin_module
    ) -> None:
        """HelpNode stores and caches the help content."""
        help_node = custom_lint_node.help()
        first_content = help_node._help_content

        # Get help again
        help_node2 = custom_lint_node.help()
        second_content = help_node2._help_content

        # Should be same instance and content
        assert help_node is help_node2
        assert first_content == second_content

    def test_help_rendering_uses_cached_content(
        self, custom_lint_node: CustomLintNode, mock_plugin_module
    ) -> None:
        """Help rendering uses the cached content."""
        help_node = custom_lint_node.help()
        content = help_node.render_content()

        # Should contain plugin info from cached content
        assert "Custom Linter" in content or "custom_lint" in content


# ---------------------------------------------------------------------------
# 6. Registry integration for plugin help lookup
# ---------------------------------------------------------------------------


class TestRegistryPluginHelp:
    """Tests for registry-based plugin help lookup."""

    def test_can_extract_help_for_registered_plugin(self) -> None:
        """Can extract help content for plugin types via registry lookup."""
        registry = get_node_registry()

        # Register the plugin type temporarily
        try:
            registry.register("custom_lint", CustomLintNode)

            # Look up via registry
            cls = registry.get("custom_lint")
            assert cls is not None

            # Extract help
            content = _extract_help_content(cls)
            assert "CustomLintNode" in content or "Custom Linter" in content

        finally:
            # Cleanup
            registry.unregister("custom_lint")

    def test_plugin_help_preserves_exposed_members(self) -> None:
        """Plugin help correctly identifies @exposed members."""
        content = _extract_help_content(CustomLintNode)

        # Should mark exposed methods
        assert "SetSeverity" in content
        # Exposed marker should be present in the agent-facing API section
        lines = content.split("\n")
        in_exposed_section = False
        found_set_severity = False
        for line in lines:
            if "## Agent-Facing API" in line:
                in_exposed_section = True
            elif line.startswith("## ") and in_exposed_section:
                break
            elif in_exposed_section and "SetSeverity" in line:
                found_set_severity = True
        assert found_set_severity


# ---------------------------------------------------------------------------
# 7. Edge cases and error handling
# ---------------------------------------------------------------------------


class TestPluginHelpEdgeCases:
    """Tests for edge cases in plugin help extraction."""

    def test_extract_handles_missing_module(self) -> None:
        """Extraction handles case where class module is not in sys.modules."""

        # Create a class with a non-existent module name
        @dataclass
        class OrphanNode(ContextNode):
            """An orphan node."""

            @property
            def node_type(self) -> str:
                return "orphan"

            def render_digest(self) -> str:
                return "Orphan"

            def get_token_breakdown(self) -> dict[str, int]:
                return {"total": 10, "header": 5, "content": 5}

            def GetDigest(self) -> dict[str, Any]:
                return {"id": self.node_id, "type": self.node_type}

            def render_content(
                self, cwd: str = ".", text_buffers: dict[str, Any] | None = None
            ) -> str:
                return "Orphan"

        OrphanNode.__module__ = "nonexistent_module_xyz"

        # Should not crash, fall back to docstring
        content = _extract_help_content(OrphanNode)
        assert "OrphanNode" in content
        assert "orphan node" in content

    def test_extract_handles_invalid_plugin_info(self) -> None:
        """Extraction handles invalid plugin_info (not a dict)."""
        import types

        # Create a module with invalid plugin_info
        mock_module = types.ModuleType("invalid_info_module")
        mock_module.plugin_info = "not a dict"  # Invalid type
        sys.modules["invalid_info_module"] = mock_module

        @dataclass
        class InvalidInfoNode(ContextNode):
            """A node with invalid plugin_info."""

            @property
            def node_type(self) -> str:
                return "invalid_info"

            def render_digest(self) -> str:
                return "Invalid"

            def get_token_breakdown(self) -> dict[str, int]:
                return {"total": 10, "header": 5, "content": 5}

            def GetDigest(self) -> dict[str, Any]:
                return {"id": self.node_id, "type": self.node_type}

            def render_content(
                self, cwd: str = ".", text_buffers: dict[str, Any] | None = None
            ) -> str:
                return "Invalid"

        InvalidInfoNode.__module__ = "invalid_info_module"

        try:
            # Should fall back to docstring
            content = _extract_help_content(InvalidInfoNode)
            assert "InvalidInfoNode" in content
            # Should not use "not a dict" as description
            assert "not a dict" not in content
        finally:
            del sys.modules["invalid_info_module"]

    def test_extract_handles_partial_plugin_info(self) -> None:
        """Extraction handles plugin_info with only some fields."""
        import types

        # Create a module with partial plugin_info
        mock_module = types.ModuleType("partial_info_module")
        mock_module.plugin_info = {
            "name": "Partial Plugin",
            # Missing: description, version, author
        }
        sys.modules["partial_info_module"] = mock_module

        @dataclass
        class PartialInfoNode(ContextNode):
            """A node with partial plugin_info."""

            @property
            def node_type(self) -> str:
                return "partial_info"

            def render_digest(self) -> str:
                return "Partial"

            def get_token_breakdown(self) -> dict[str, int]:
                return {"total": 10, "header": 5, "content": 5}

            def GetDigest(self) -> dict[str, Any]:
                return {"id": self.node_id, "type": self.node_type}

            def render_content(
                self, cwd: str = ".", text_buffers: dict[str, Any] | None = None
            ) -> str:
                return "Partial"

        PartialInfoNode.__module__ = "partial_info_module"

        try:
            content = _extract_help_content(PartialInfoNode)
            # Should use name from plugin_info
            assert "Partial Plugin" in content
            # Should use class docstring for description
            assert "partial plugin_info" in content
            # Should not crash on missing fields
        finally:
            del sys.modules["partial_info_module"]
