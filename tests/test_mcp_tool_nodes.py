"""Tests for MCPToolNode and MCPServerNode tool child management."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from activecontext.context.nodes import (
    MCPToolNode,
    MCPServerNode,
    Expansion,
    ContextNode,
)
from activecontext.context.graph import ContextGraph
from activecontext.mcp.types import MCPConnectionStatus, MCPToolInfo


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def context_graph():
    """Create a ContextGraph instance."""
    return ContextGraph()


@pytest.fixture
def mcp_server_node(context_graph):
    """Create an MCPServerNode added to the graph."""
    node = MCPServerNode(
        server_name="test-server",
        tokens=1000,
        expansion=Expansion.ALL,
    )
    context_graph.add_node(node)
    return node


@pytest.fixture
def mock_connection():
    """Create a mock MCPConnection with tools."""
    conn = Mock()
    conn.status = MCPConnectionStatus.CONNECTED
    conn.error_message = None
    conn.tools = [
        MCPToolInfo(
            name="read_file",
            description="Read contents of a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
            server_name="test-server",
        ),
        MCPToolInfo(
            name="write_file",
            description="Write content to a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            server_name="test-server",
        ),
    ]
    conn.resources = []
    conn.prompts = []
    return conn


# =============================================================================
# MCPToolNode Tests
# =============================================================================


class TestMCPToolNodeInit:
    """Tests for MCPToolNode initialization."""

    def test_default_values(self):
        """Test MCPToolNode has correct defaults."""
        node = MCPToolNode()
        assert node.tool_name == ""
        assert node.server_name == ""
        assert node.description == ""
        assert node.input_schema == {}
        assert node.expansion == Expansion.ALL  # Inherited from ContextNode
        assert node.node_type == "mcp_tool"

    def test_custom_values(self):
        """Test MCPToolNode with custom values."""
        node = MCPToolNode(
            tool_name="read_file",
            server_name="filesystem",
            description="Read a file",
            input_schema={"type": "object"},
            tokens=200,
            expansion=Expansion.ALL,
        )
        assert node.tool_name == "read_file"
        assert node.server_name == "filesystem"
        assert node.description == "Read a file"
        assert node.input_schema == {"type": "object"}
        assert node.tokens == 200
        assert node.expansion == Expansion.ALL


class TestMCPToolNodeRender:
    """Tests for MCPToolNode rendering at different states."""

    @pytest.fixture
    def tool_node(self):
        """Create a tool node with realistic data."""
        return MCPToolNode(
            tool_name="read_file",
            server_name="filesystem",
            description="Read contents of a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "description": "File encoding"},
                },
                "required": ["path"],
            },
            tokens=200,
        )

    def test_render_collapsed(self, tool_node):
        """Test COLLAPSED state shows tool name + brief + ID + tokens."""
        tool_node.expansion = Expansion.HEADER
        result = tool_node.Render()
        # Format: ### `tool_name` description... | {#id} (X/Y tokens)
        assert "### `read_file`" in result
        assert "Read contents of a file" in result
        assert f"{{#{tool_node.node_id}}}" in result
        assert "tokens)" in result

    def test_render_summary(self, tool_node):
        """Test SUMMARY state shows name and truncated description."""
        tool_node.expansion = Expansion.CONTENT
        result = tool_node.Render()
        assert "**read_file**:" in result
        assert "Read contents" in result

    def test_render_summary_truncates_long_description(self):
        """Test SUMMARY truncates descriptions over 80 chars."""
        node = MCPToolNode(
            tool_name="long_tool",
            description="A" * 100,
            expansion=Expansion.CONTENT,
        )
        result = node.Render()
        assert "..." in result
        assert len(result) < 150  # Should be truncated

    def test_render_details(self, tool_node):
        """Test DETAILS state shows name, description, and required params."""
        tool_node.expansion = Expansion.ALL
        result = tool_node.Render()
        assert "### `read_file`" in result
        assert "Read contents of a file" in result
        assert "`path`*" in result  # Required param marked with *
        assert "`encoding`" in result  # Optional param without *



class TestMCPToolNodeDigest:
    """Tests for MCPToolNode.GetDigest()."""

    def test_digest_contents(self):
        """Test GetDigest returns expected fields."""
        node = MCPToolNode(
            tool_name="read_file",
            server_name="filesystem",
            input_schema={"properties": {"path": {}}},
            tokens=200,
            expansion=Expansion.ALL,
        )
        digest = node.GetDigest()
        assert digest["type"] == "mcp_tool"
        assert digest["tool_name"] == "read_file"
        assert digest["server_name"] == "filesystem"
        assert digest["has_schema"] is True
        assert digest["tokens"] == 200
        assert digest["expansion"] == "all"

    def test_digest_no_schema(self):
        """Test GetDigest with empty schema."""
        node = MCPToolNode(tool_name="simple")
        digest = node.GetDigest()
        assert digest["has_schema"] is False


class TestMCPToolNodeSerialization:
    """Tests for MCPToolNode serialization/deserialization."""

    def test_to_dict(self):
        """Test MCPToolNode.to_dict() includes all fields."""
        node = MCPToolNode(
            tool_name="read_file",
            server_name="filesystem",
            description="Read a file",
            input_schema={"type": "object"},
            tokens=200,
        )
        data = node.to_dict()
        assert data["node_type"] == "mcp_tool"
        assert data["tool_name"] == "read_file"
        assert data["server_name"] == "filesystem"
        assert data["description"] == "Read a file"
        assert data["input_schema"] == {"type": "object"}

    def test_from_dict(self):
        """Test MCPToolNode._from_dict() restores node."""
        original = MCPToolNode(
            tool_name="write_file",
            server_name="filesystem",
            description="Write a file",
            input_schema={"properties": {"path": {}}},
            tokens=300,
            expansion=Expansion.ALL,
        )
        data = original.to_dict()
        restored = MCPToolNode._from_dict(data)
        assert restored.tool_name == "write_file"
        assert restored.server_name == "filesystem"
        assert restored.description == "Write a file"
        assert restored.input_schema == {"properties": {"path": {}}}
        assert restored.tokens == 300
        assert restored.expansion == Expansion.ALL

    def test_from_dict_via_factory(self):
        """Test ContextNode.from_dict() dispatches to MCPToolNode."""
        node = MCPToolNode(tool_name="test_tool", server_name="test")
        data = node.to_dict()
        restored = ContextNode.from_dict(data)
        assert isinstance(restored, MCPToolNode)
        assert restored.tool_name == "test_tool"


# =============================================================================
# MCPServerNode Tool Child Management Tests
# =============================================================================


class TestMCPServerNodeToolChildren:
    """Tests for MCPServerNode creating/managing tool children."""

    def test_update_from_connection_creates_tool_nodes(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test update_from_connection creates MCPToolNode children."""
        mcp_server_node.update_from_connection(mock_connection)

        # Should have tool nodes tracked
        assert len(mcp_server_node._tool_nodes) == 2
        assert "read_file" in mcp_server_node._tool_nodes
        assert "write_file" in mcp_server_node._tool_nodes

        # Tool nodes should exist in graph
        for tool_name, node_id in mcp_server_node._tool_nodes.items():
            node = context_graph.get_node(node_id)
            assert node is not None
            assert isinstance(node, MCPToolNode)
            assert node.tool_name == tool_name
            assert node.server_name == "test-server"

    def test_update_from_connection_links_children(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test tool nodes are linked as children in the graph."""
        mcp_server_node.update_from_connection(mock_connection)

        for node_id in mcp_server_node._tool_nodes.values():
            node = context_graph.get_node(node_id)
            assert mcp_server_node.node_id in node.parent_ids
            assert node_id in mcp_server_node.children_ids

    def test_tool_method_returns_tool_node(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test MCPServerNode.tool() returns the correct MCPToolNode."""
        mcp_server_node.update_from_connection(mock_connection)

        tool = mcp_server_node.tool("read_file")
        assert tool is not None
        assert isinstance(tool, MCPToolNode)
        assert tool.tool_name == "read_file"

    def test_tool_method_returns_none_for_unknown(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test MCPServerNode.tool() returns None for unknown tool."""
        mcp_server_node.update_from_connection(mock_connection)

        tool = mcp_server_node.tool("nonexistent")
        assert tool is None

    def test_tool_nodes_property(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test MCPServerNode.tool_nodes returns all tool children."""
        mcp_server_node.update_from_connection(mock_connection)

        nodes = mcp_server_node.tool_nodes
        assert len(nodes) == 2
        assert all(isinstance(n, MCPToolNode) for n in nodes)
        tool_names = {n.tool_name for n in nodes}
        assert tool_names == {"read_file", "write_file"}


class TestMCPServerNodeToolDiff:
    """Tests for tool diff/update on reconnection."""

    def test_reconnect_adds_new_tools(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test reconnection adds new tools."""
        # Initial connection with 2 tools
        mcp_server_node.update_from_connection(mock_connection)
        assert len(mcp_server_node._tool_nodes) == 2

        # Reconnect with 3 tools
        mock_connection.tools.append(
            MCPToolInfo(
                name="delete_file",
                description="Delete a file",
                input_schema={"type": "object"},
                server_name="test-server",
            )
        )
        mcp_server_node.update_from_connection(mock_connection)

        assert len(mcp_server_node._tool_nodes) == 3
        assert "delete_file" in mcp_server_node._tool_nodes
        tool = mcp_server_node.tool("delete_file")
        assert tool is not None
        assert tool.description == "Delete a file"

    def test_reconnect_removes_old_tools(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test reconnection removes tools that no longer exist."""
        # Initial connection with 2 tools
        mcp_server_node.update_from_connection(mock_connection)
        write_node_id = mcp_server_node._tool_nodes["write_file"]
        assert context_graph.get_node(write_node_id) is not None

        # Reconnect with only 1 tool
        mock_connection.tools = [mock_connection.tools[0]]  # Keep only read_file
        mcp_server_node.update_from_connection(mock_connection)

        assert len(mcp_server_node._tool_nodes) == 1
        assert "write_file" not in mcp_server_node._tool_nodes
        assert context_graph.get_node(write_node_id) is None  # Removed from graph

    def test_reconnect_updates_changed_tools(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test reconnection updates tools with changed schema."""
        # Initial connection
        mcp_server_node.update_from_connection(mock_connection)
        tool = mcp_server_node.tool("read_file")
        old_version = tool.version

        # Reconnect with updated schema
        mock_connection.tools[0] = MCPToolInfo(
            name="read_file",
            description="Read contents of a file (updated)",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "description": "Max bytes to read"},
                },
                "required": ["path"],
            },
            server_name="test-server",
        )
        mcp_server_node.update_from_connection(mock_connection)

        tool = mcp_server_node.tool("read_file")
        assert tool.description == "Read contents of a file (updated)"
        assert "limit" in tool.input_schema["properties"]
        assert tool.version > old_version  # Version bumped from change

    def test_reconnect_preserves_unchanged_tools(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test reconnection preserves tools that haven't changed."""
        # Initial connection
        mcp_server_node.update_from_connection(mock_connection)
        tool = mcp_server_node.tool("read_file")
        old_node_id = mcp_server_node._tool_nodes["read_file"]
        old_version = tool.version

        # Reconnect with same tools
        mcp_server_node.update_from_connection(mock_connection)

        # Same node ID should be kept
        assert mcp_server_node._tool_nodes["read_file"] == old_node_id
        # Version should not change (no actual changes)
        tool = mcp_server_node.tool("read_file")
        assert tool.version == old_version


class TestMCPServerNodeSerialization:
    """Tests for MCPServerNode serialization with tool nodes."""

    def test_to_dict_includes_tool_nodes(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test MCPServerNode.to_dict() includes _tool_nodes mapping."""
        mcp_server_node.update_from_connection(mock_connection)
        data = mcp_server_node.to_dict()

        assert "_tool_nodes" in data
        assert len(data["_tool_nodes"]) == 2
        assert "read_file" in data["_tool_nodes"]
        assert "write_file" in data["_tool_nodes"]

    def test_from_dict_restores_tool_nodes_mapping(self, context_graph):
        """Test MCPServerNode._from_dict() restores _tool_nodes mapping."""
        original = MCPServerNode(
            server_name="test",
            _tool_nodes={"read_file": "node_123", "write_file": "node_456"},
        )
        data = original.to_dict()
        restored = MCPServerNode._from_dict(data)

        assert restored._tool_nodes == {"read_file": "node_123", "write_file": "node_456"}


# =============================================================================
# Integration Tests
# =============================================================================


class TestMCPToolNodeIntegration:
    """Integration tests for MCPToolNode with the full system."""

    def test_tool_node_direct_assignment(self, context_graph, mcp_server_node, mock_connection):
        """Test MCPToolNode supports direct field assignment for state changes."""
        mcp_server_node.update_from_connection(mock_connection)
        tool = mcp_server_node.tool("read_file")

        # Test direct assignment
        tool.expansion = Expansion.ALL
        assert tool.expansion == Expansion.ALL

    def test_tool_node_display_name(self):
        """Test MCPToolNode.get_display_name() format."""
        node = MCPToolNode(tool_name="read_file", server_name="filesystem")
        assert node.get_display_name() == "filesystem.read_file"

    def test_child_order_populated_on_link(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test that tool nodes are added to child_order for projection rendering."""
        mcp_server_node.update_from_connection(mock_connection)

        # child_order should be populated by graph.link()
        assert len(mcp_server_node.child_order) == 2
        # All tool node IDs should be in child_order
        for node_id in mcp_server_node._tool_nodes.values():
            assert node_id in mcp_server_node.child_order

    def test_projection_includes_tool_nodes(
        self, context_graph, mcp_server_node, mock_connection
    ):
        """Test that projection engine includes tool nodes in render path."""
        from activecontext.core.projection_engine import ProjectionEngine

        mcp_server_node.update_from_connection(mock_connection)
        mcp_server_node.expansion = Expansion.ALL  # DETAILS/ALL render children

        engine = ProjectionEngine()
        projection = engine.build(context_graph=context_graph, cwd=".")

        # Tool node content should appear in projection
        rendered = projection.render()
        assert "read_file" in rendered
        assert "write_file" in rendered
