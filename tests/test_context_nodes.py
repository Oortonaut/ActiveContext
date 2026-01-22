"""Tests for context node serialization.

Tests coverage for:
- src/activecontext/context/nodes.py (to_dict/from_dict round-trips)
"""

from __future__ import annotations

import pytest

from activecontext.context.nodes import (
    AgentNode,
    ArtifactNode,
    ContextNode,
    GroupNode,
    LockNode,
    MCPManagerNode,
    MCPServerNode,
    SessionNode,
    ShellNode,
    TopicNode,
    TextNode,
)
from activecontext.context.state import Expansion


# =============================================================================
# TextNode Serialization Tests
# =============================================================================


class TestTextNodeSerialization:
    """Tests for TextNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test TextNode serialization to dict."""
        node = TextNode(
            node_id="view1",
            path="src/main.py",
            state=Expansion.DETAILS,
        )

        data = node.to_dict()

        assert data["node_type"] == "text"
        assert data["node_id"] == "view1"
        assert data["path"] == "src/main.py"
        assert data["state"] == "details"
        # TextNode doesn't serialize _content - it's loaded from disk

    def test_from_dict_basic(self):
        """Test TextNode deserialization from dict."""
        data = {
            "node_type": "text",
            "node_id": "view1",
            "path": "src/main.py",
            "state": "details",
            "tokens": 100,
            "pos": "1:0",
        }

        node = TextNode._from_dict(data)

        assert node.node_id == "view1"
        assert node.path == "src/main.py"
        assert node.state == Expansion.DETAILS

    def test_roundtrip(self):
        """Test TextNode serialization round-trip."""
        original = TextNode(
            node_id="view1",
            path="src/main.py",
            state=Expansion.DETAILS,
            pos="10:5",
        )

        data = original.to_dict()
        restored = TextNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.path == original.path
        assert restored.state == original.state
        assert restored.pos == original.pos

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to TextNode."""
        data = {
            "node_type": "text",
            "node_id": "view1",
            "path": "test.py",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, TextNode)
        assert node.node_id == "view1"


# =============================================================================
# GroupNode Serialization Tests
# =============================================================================


class TestGroupNodeSerialization:
    """Tests for GroupNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test GroupNode serialization to dict."""
        node = GroupNode(
            node_id="group1",
            summary_prompt="Summarize these files",
            state=Expansion.SUMMARY,
            cached_summary="A group of related files",
        )

        data = node.to_dict()

        assert data["node_type"] == "group"
        assert data["node_id"] == "group1"
        assert data["summary_prompt"] == "Summarize these files"
        assert data["state"] == "summary"
        assert data["cached_summary"] == "A group of related files"

    def test_from_dict_basic(self):
        """Test GroupNode deserialization from dict."""
        data = {
            "node_type": "group",
            "node_id": "group1",
            "summary_prompt": "Test prompt",
            "state": "summary",
            "cached_summary": "Test summary",
        }

        node = GroupNode._from_dict(data)

        assert node.node_id == "group1"
        assert node.summary_prompt == "Test prompt"
        assert node.state == Expansion.SUMMARY
        assert node.cached_summary == "Test summary"

    def test_roundtrip(self):
        """Test GroupNode serialization round-trip."""
        original = GroupNode(
            node_id="group1",
            summary_prompt="Summarize the auth module",
            state=Expansion.DETAILS,
            cached_summary="Authentication implementation",
        )

        data = original.to_dict()
        restored = GroupNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.summary_prompt == original.summary_prompt
        assert restored.state == original.state
        assert restored.cached_summary == original.cached_summary

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to GroupNode."""
        data = {
            "node_type": "group",
            "node_id": "group1",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, GroupNode)


# =============================================================================
# TopicNode Serialization Tests
# =============================================================================


class TestTopicNodeSerialization:
    """Tests for TopicNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test TopicNode serialization to dict."""
        node = TopicNode(
            node_id="topic1",
            title="Authentication Implementation",
            state=Expansion.COLLAPSED,
        )

        data = node.to_dict()

        assert data["node_type"] == "topic"
        assert data["node_id"] == "topic1"
        assert data["title"] == "Authentication Implementation"

    def test_roundtrip(self):
        """Test TopicNode serialization round-trip."""
        original = TopicNode(
            node_id="topic1",
            title="Bug Fix Discussion",
            state=Expansion.DETAILS,
        )

        data = original.to_dict()
        restored = TopicNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.title == original.title
        assert restored.state == original.state

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to TopicNode."""
        data = {
            "node_type": "topic",
            "node_id": "topic1",
            "title": "Test Topic",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, TopicNode)


# =============================================================================
# ArtifactNode Serialization Tests
# =============================================================================


class TestArtifactNodeSerialization:
    """Tests for ArtifactNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test ArtifactNode serialization to dict."""
        node = ArtifactNode(
            node_id="artifact1",
            content="def foo(): pass",
            artifact_type="code",
            language="python",
            state=Expansion.DETAILS,
        )

        data = node.to_dict()

        assert data["node_type"] == "artifact"
        assert data["node_id"] == "artifact1"
        assert data["content"] == "def foo(): pass"
        assert data["artifact_type"] == "code"
        assert data["language"] == "python"

    def test_roundtrip(self):
        """Test ArtifactNode serialization round-trip."""
        original = ArtifactNode(
            node_id="artifact1",
            content="Error: Connection refused",
            artifact_type="error",
            language="text",
            state=Expansion.DETAILS,
        )

        data = original.to_dict()
        restored = ArtifactNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.content == original.content
        assert restored.artifact_type == original.artifact_type
        assert restored.language == original.language

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to ArtifactNode."""
        data = {
            "node_type": "artifact",
            "node_id": "artifact1",
            "content": "test",
            "artifact_type": "output",
            "state": "details",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, ArtifactNode)


# =============================================================================
# ShellNode Serialization Tests
# =============================================================================


class TestShellNodeSerialization:
    """Tests for ShellNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test ShellNode serialization to dict."""
        node = ShellNode(
            node_id="shell1",
            command="pytest",
            args=["-v", "tests/"],
            state=Expansion.DETAILS,
            output="All tests passed",
            exit_code=0,
        )

        data = node.to_dict()

        assert data["node_type"] == "shell"
        assert data["node_id"] == "shell1"
        assert data["command"] == "pytest"
        assert data["args"] == ["-v", "tests/"]
        assert data["output"] == "All tests passed"
        assert data["exit_code"] == 0

    def test_roundtrip(self):
        """Test ShellNode serialization round-trip."""
        original = ShellNode(
            node_id="shell1",
            command="git",
            args=["status"],
            state=Expansion.DETAILS,
            output="On branch main",
            exit_code=0,
        )

        data = original.to_dict()
        restored = ShellNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.command == original.command
        assert restored.args == original.args
        assert restored.output == original.output
        assert restored.exit_code == original.exit_code

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to ShellNode."""
        data = {
            "node_type": "shell",
            "node_id": "shell1",
            "command": "ls",
            "args": ["-la"],
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, ShellNode)


# =============================================================================
# LockNode Serialization Tests
# =============================================================================


class TestLockNodeSerialization:
    """Tests for LockNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test LockNode serialization to dict."""
        node = LockNode(
            node_id="lock1",
            lockfile="src/config.py.lock",
            state=Expansion.COLLAPSED,
        )

        data = node.to_dict()

        assert data["node_type"] == "lock"
        assert data["node_id"] == "lock1"
        assert data["lockfile"] == "src/config.py.lock"

    def test_roundtrip(self):
        """Test LockNode serialization round-trip."""
        original = LockNode(
            node_id="lock1",
            lockfile="src/main.py.lock",
            timeout=60.0,
            state=Expansion.DETAILS,
        )

        data = original.to_dict()
        restored = LockNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.lockfile == original.lockfile
        assert restored.timeout == original.timeout

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to LockNode."""
        data = {
            "node_type": "lock",
            "node_id": "lock1",
            "lockfile": "test.py.lock",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, LockNode)


# =============================================================================
# SessionNode Serialization Tests
# =============================================================================


class TestSessionNodeSerialization:
    """Tests for SessionNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test SessionNode serialization to dict."""
        node = SessionNode(
            node_id="session1",
            state=Expansion.COLLAPSED,
            turn_count=5,
            total_statements_executed=25,
        )

        data = node.to_dict()

        assert data["node_type"] == "session"
        assert data["node_id"] == "session1"
        assert data["turn_count"] == 5
        assert data["total_statements_executed"] == 25

    def test_roundtrip(self):
        """Test SessionNode serialization round-trip."""
        original = SessionNode(
            node_id="session1",
            state=Expansion.DETAILS,
            turn_count=10,
            total_tokens_consumed=5000,
        )

        data = original.to_dict()
        restored = SessionNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.turn_count == original.turn_count

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to SessionNode."""
        data = {
            "node_type": "session",
            "node_id": "session1",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, SessionNode)


# =============================================================================
# MCPServerNode Serialization Tests
# =============================================================================


class TestMCPServerNodeSerialization:
    """Tests for MCPServerNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test MCPServerNode serialization to dict."""
        node = MCPServerNode(
            node_id="mcp1",
            server_name="filesystem",
            state=Expansion.COLLAPSED,
            tools=[{"name": "read_file", "description": "Read a file"}],
        )

        data = node.to_dict()

        assert data["node_type"] == "mcp_server"
        assert data["node_id"] == "mcp1"
        assert data["server_name"] == "filesystem"
        assert len(data["tools"]) == 1

    def test_roundtrip(self):
        """Test MCPServerNode serialization round-trip."""
        original = MCPServerNode(
            node_id="mcp1",
            server_name="github",
            state=Expansion.DETAILS,
            tools=[
                {"name": "list_repos", "description": "List repositories"},
                {"name": "create_issue", "description": "Create an issue"},
            ],
        )

        data = original.to_dict()
        restored = MCPServerNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.server_name == original.server_name
        assert len(restored.tools) == len(original.tools)

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to MCPServerNode."""
        data = {
            "node_type": "mcp_server",
            "node_id": "mcp1",
            "server_name": "test",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, MCPServerNode)


# =============================================================================
# MCPManagerNode Serialization Tests
# =============================================================================


class TestMCPManagerNodeSerialization:
    """Tests for MCPManagerNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test MCPManagerNode serialization to dict."""
        node = MCPManagerNode(
            node_id="mcp_manager",
            state=Expansion.COLLAPSED,
        )

        data = node.to_dict()

        assert data["node_type"] == "mcp_manager"
        assert data["node_id"] == "mcp_manager"

    def test_roundtrip(self):
        """Test MCPManagerNode serialization round-trip."""
        original = MCPManagerNode(
            node_id="mcp_manager",
            state=Expansion.DETAILS,
        )

        data = original.to_dict()
        restored = MCPManagerNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.state == original.state

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to MCPManagerNode."""
        data = {
            "node_type": "mcp_manager",
            "node_id": "mcp_manager",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, MCPManagerNode)


# =============================================================================
# AgentNode Serialization Tests
# =============================================================================


class TestAgentNodeSerialization:
    """Tests for AgentNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test AgentNode serialization to dict."""
        node = AgentNode(
            node_id="agent1",
            agent_id="child-agent-123",
            agent_type="researcher",
            state=Expansion.DETAILS,
            agent_state="running",
            task="Research the API",
        )

        data = node.to_dict()

        assert data["node_type"] == "agent"
        assert data["node_id"] == "agent1"
        assert data["agent_id"] == "child-agent-123"
        assert data["agent_type"] == "researcher"
        assert data["agent_state"] == "running"
        assert data["task"] == "Research the API"

    def test_roundtrip(self):
        """Test AgentNode serialization round-trip."""
        original = AgentNode(
            node_id="agent1",
            agent_id="worker-456",
            agent_type="coder",
            state=Expansion.DETAILS,
            agent_state="completed",
            task="Implement feature X",
        )

        data = original.to_dict()
        restored = AgentNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.agent_id == original.agent_id
        assert restored.agent_type == original.agent_type
        assert restored.agent_state == original.agent_state
        assert restored.task == original.task

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to AgentNode."""
        data = {
            "node_type": "agent",
            "node_id": "agent1",
            "agent_id": "test-agent",
            "agent_type": "helper",
            "state": "collapsed",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, AgentNode)


# =============================================================================
# ContextGraph Serialization Tests
# =============================================================================


class TestContextGraphSerialization:
    """Tests for ContextGraph to_dict/from_dict round-trip."""

    def test_graph_roundtrip(self):
        """Test ContextGraph serialization round-trip preserves structure."""
        from activecontext.context.graph import ContextGraph

        # Create graph with various node types
        graph = ContextGraph()

        view = TextNode(node_id="view1", path="main.py", state=Expansion.DETAILS)
        graph.add_node(view)

        group = GroupNode(node_id="group1", state=Expansion.SUMMARY, cached_summary="Code files")
        graph.add_node(group)

        topic = TopicNode(node_id="topic1", title="Discussion", state=Expansion.COLLAPSED)
        graph.add_node(topic)
        graph.link("topic1", "group1")  # topic is child of group (link(child, parent))

        # Serialize
        data = graph.to_dict()

        # Deserialize
        restored = ContextGraph.from_dict(data)

        # Verify structure
        assert len(restored) == 3
        assert "view1" in restored
        assert "group1" in restored
        assert "topic1" in restored

        # Verify node types
        assert isinstance(restored.get_node("view1"), TextNode)
        assert isinstance(restored.get_node("group1"), GroupNode)
        assert isinstance(restored.get_node("topic1"), TopicNode)

        # Verify content preserved
        assert restored.get_node("group1").cached_summary == "Code files"
        assert restored.get_node("topic1").title == "Discussion"

        # Verify roots (view1 and group1 are roots, topic1 is not)
        root_ids = {n.node_id for n in restored.get_roots()}
        assert "view1" in root_ids
        assert "group1" in root_ids
        assert "topic1" not in root_ids  # has parent
