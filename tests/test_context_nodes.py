"""Tests for context node serialization.

Tests coverage for:
- src/activecontext/context/nodes.py (to_dict/from_dict round-trips)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from activecontext.context.nodes import (
    AgentNode,
    ArtifactNode,
    ContextNode,
    GroupNode,
    LockNode,
    MCPManagerNode,
    MCPServerNode,
    PluginManagerNode,
    SessionNode,
    ShellNode,
    StatementNode,
    StatementResultNode,
    TextNode,
    TopicNode,
)
from activecontext.agents.schema import AgentState
from activecontext.context.state import Expansion, WorkStatus

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
            default_expansion=Expansion.ALL,
        )

        data = node.to_dict()

        # node_type is now the class name
        assert data["node_type"] == "TextNode"
        assert data["node_id"] == "view1"
        assert data["path"] == "src/main.py"
        assert data["expansion"] == "all"
        # TextNode doesn't serialize _content - it's loaded from disk

    def test_from_dict_basic(self):
        """Test TextNode deserialization from dict."""
        data = {
            "node_type": "TextNode",
            "node_id": "view1",
            "path": "src/main.py",
            "expansion": "all",
            "tokens": 100,
            "pos": "1:0",
        }

        node = TextNode._from_dict(data)

        assert node.node_id == "view1"
        assert node.path == "src/main.py"
        assert node.default_expansion == Expansion.ALL

    def test_roundtrip(self):
        """Test TextNode serialization round-trip."""
        original = TextNode(
            node_id="view1",
            path="src/main.py",
            default_expansion=Expansion.ALL,
            pos="10:5",
        )

        data = original.to_dict()
        restored = TextNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.path == original.path
        assert restored.default_expansion == original.default_expansion
        assert restored.pos == original.pos

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to TextNode."""
        data = {
            "node_type": "TextNode",
            "node_id": "view1",
            "path": "test.py",
            "expansion": "header",
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
            default_expansion=Expansion.CONTENT,
        )

        data = node.to_dict()

        assert data["node_type"] == "GroupNode"
        assert data["node_id"] == "group1"
        assert data["summary_prompt"] == "Summarize these files"
        assert data["expansion"] == "content"

    def test_from_dict_basic(self):
        """Test GroupNode deserialization from dict."""
        data = {
            "node_type": "GroupNode",
            "node_id": "group1",
            "summary_prompt": "Test prompt",
            "expansion": "content",
        }

        node = GroupNode._from_dict(data)

        assert node.node_id == "group1"
        assert node.summary_prompt == "Test prompt"
        assert node.default_expansion == Expansion.CONTENT

    def test_roundtrip(self):
        """Test GroupNode serialization round-trip."""
        original = GroupNode(
            node_id="group1",
            summary_prompt="Summarize the auth module",
            default_expansion=Expansion.ALL,
        )

        data = original.to_dict()
        restored = GroupNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.summary_prompt == original.summary_prompt
        assert restored.default_expansion == original.default_expansion

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to GroupNode."""
        data = {
            "node_type": "GroupNode",
            "node_id": "group1",
            "expansion": "header",
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
            default_expansion=Expansion.HEADER,
        )

        data = node.to_dict()

        assert data["node_type"] == "TopicNode"
        assert data["node_id"] == "topic1"
        assert data["title"] == "Authentication Implementation"

    def test_roundtrip(self):
        """Test TopicNode serialization round-trip."""
        original = TopicNode(
            node_id="topic1",
            title="Bug Fix Discussion",
            default_expansion=Expansion.ALL,
        )

        data = original.to_dict()
        restored = TopicNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.title == original.title
        assert restored.default_expansion == original.default_expansion

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to TopicNode."""
        data = {
            "node_type": "TopicNode",
            "node_id": "topic1",
            "title": "Test Topic",
            "expansion": "header",
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
            default_expansion=Expansion.ALL,
        )

        data = node.to_dict()

        assert data["node_type"] == "ArtifactNode"
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
            default_expansion=Expansion.ALL,
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
            "node_type": "ArtifactNode",
            "node_id": "artifact1",
            "content": "test",
            "artifact_type": "output",
            "expansion": "all",
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
            default_expansion=Expansion.ALL,
            output="All tests passed",
            exit_code=0,
        )

        data = node.to_dict()

        assert data["node_type"] == "ShellNode"
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
            default_expansion=Expansion.ALL,
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
            "node_type": "ShellNode",
            "node_id": "shell1",
            "command": "ls",
            "args": ["-la"],
            "expansion": "header",
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
            default_expansion=Expansion.HEADER,
        )

        data = node.to_dict()

        assert data["node_type"] == "LockNode"
        assert data["node_id"] == "lock1"
        assert data["lockfile"] == "src/config.py.lock"

    def test_roundtrip(self):
        """Test LockNode serialization round-trip."""
        original = LockNode(
            node_id="lock1",
            lockfile="src/main.py.lock",
            timeout=60.0,
            default_expansion=Expansion.ALL,
        )

        data = original.to_dict()
        restored = LockNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.lockfile == original.lockfile
        assert restored.timeout == original.timeout

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to LockNode."""
        data = {
            "node_type": "LockNode",
            "node_id": "lock1",
            "lockfile": "test.py.lock",
            "expansion": "header",
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
            default_expansion=Expansion.HEADER,
            turn_count=5,
            total_statements_executed=25,
        )

        data = node.to_dict()

        assert data["node_type"] == "SessionNode"
        assert data["node_id"] == "session1"
        assert data["turn_count"] == 5
        assert data["total_statements_executed"] == 25

    def test_roundtrip(self):
        """Test SessionNode serialization round-trip."""
        original = SessionNode(
            node_id="session1",
            default_expansion=Expansion.ALL,
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
            "node_type": "SessionNode",
            "node_id": "session1",
            "expansion": "header",
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
            default_expansion=Expansion.HEADER,
            tools=[{"name": "read_file", "description": "Read a file"}],
        )

        data = node.to_dict()

        assert data["node_type"] == "MCPServerNode"
        assert data["node_id"] == "mcp1"
        assert data["server_name"] == "filesystem"
        assert len(data["tools"]) == 1

    def test_roundtrip(self):
        """Test MCPServerNode serialization round-trip."""
        original = MCPServerNode(
            node_id="mcp1",
            server_name="github",
            default_expansion=Expansion.ALL,
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
            "node_type": "MCPServerNode",
            "node_id": "mcp1",
            "server_name": "test",
            "expansion": "header",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, MCPServerNode)


class TestMCPServerNodeGetattr:
    """Tests for MCPServerNode __getattr__ delegation to server proxy."""

    def test_delegates_to_server_proxy(self):
        """Test that __getattr__ delegates to _server_proxy."""
        proxy = MagicMock()
        proxy.list_directory_tree = MagicMock(return_value="tree result")

        node = MCPServerNode(server_name="rider", status="connected")
        node._server_proxy = proxy

        result = node.list_directory_tree()
        assert result == "tree result"
        proxy.list_directory_tree.assert_called_once()

    def test_raises_attribute_error_without_proxy(self):
        """Test that __getattr__ raises AttributeError when no proxy is set."""
        node = MCPServerNode(server_name="test")
        with pytest.raises(AttributeError, match="MCPServerNode"):
            _ = node.nonexistent_method

    def test_raises_attribute_error_for_unknown_tool(self):
        """Test that __getattr__ raises AttributeError for tools not on proxy."""
        proxy = MagicMock(spec=[])  # Empty spec = no attributes
        node = MCPServerNode(server_name="test")
        node._server_proxy = proxy

        with pytest.raises(AttributeError, match="MCPServerNode"):
            _ = node.unknown_tool

    def test_dataclass_fields_unaffected(self):
        """Test that normal dataclass field access still works."""
        node = MCPServerNode(server_name="test", status="connected")
        assert node.server_name == "test"
        assert node.status == "connected"
        assert node.tools == []

    def test_serialization_excludes_server_proxy(self):
        """Test that _server_proxy is not included in to_dict output."""
        proxy = MagicMock()
        node = MCPServerNode(server_name="test")
        node._server_proxy = proxy

        data = node.to_dict()
        assert "_server_proxy" not in data

    def test_proxy_none_by_default(self):
        """Test that _server_proxy defaults to None."""
        node = MCPServerNode(server_name="test")
        assert node._server_proxy is None


# =============================================================================
# MCPManagerNode Serialization Tests
# =============================================================================


class TestMCPManagerNodeSerialization:
    """Tests for MCPManagerNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test MCPManagerNode serialization to dict."""
        node = MCPManagerNode(
            node_id="mcp_manager",
            default_expansion=Expansion.HEADER,
        )

        data = node.to_dict()

        assert data["node_type"] == "MCPManagerNode"
        assert data["node_id"] == "mcp_manager"

    def test_roundtrip(self):
        """Test MCPManagerNode serialization round-trip."""
        original = MCPManagerNode(
            node_id="mcp_manager",
            default_expansion=Expansion.ALL,
        )

        data = original.to_dict()
        restored = MCPManagerNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.default_expansion == original.default_expansion

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to MCPManagerNode."""
        data = {
            "node_type": "MCPManagerNode",
            "node_id": "mcp_manager",
            "expansion": "header",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, MCPManagerNode)


# =============================================================================
# PluginManagerNode Tests
# =============================================================================


class TestPluginManagerNodeSerialization:
    """Tests for PluginManagerNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test PluginManagerNode serialization to dict."""
        node = PluginManagerNode(
            node_id="plugin_manager",
            default_expansion=Expansion.HEADER,
            builtin_count=16,
            loaded_count=2,
        )

        data = node.to_dict()

        assert data["node_type"] == "PluginManagerNode"
        assert data["node_id"] == "plugin_manager"
        assert data["builtin_count"] == 16
        assert data["loaded_count"] == 2

    def test_roundtrip(self):
        """Test PluginManagerNode serialization round-trip."""

        original = PluginManagerNode(
            node_id="plugin_manager",
            default_expansion=Expansion.ALL,
            builtin_count=16,
            loaded_count=2,
            plugin_states={"test_plugin": "connected"},
            plugin_types={"test_plugin": ["custom_type_a", "custom_type_b"]},
        )

        data = original.to_dict()
        restored = PluginManagerNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.default_expansion == original.default_expansion
        assert restored.builtin_count == original.builtin_count
        assert restored.loaded_count == original.loaded_count
        assert restored.plugin_states == original.plugin_states
        assert restored.plugin_types == original.plugin_types

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to PluginManagerNode."""
        data = {
            "node_type": "PluginManagerNode",
            "node_id": "plugin_manager",
            "expansion": "header",
            "builtin_count": 16,
            "loaded_count": 0,
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, PluginManagerNode)


class TestPluginManagerNodeRendering:
    """Tests for PluginManagerNode rendering methods."""

    def test_render_summary_no_plugins(self):
        """Test summary rendering with no loaded plugins."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=0,
        )

        result = node.render_content()

        assert "No plugin servers loaded" in result

    def test_render_summary_with_plugins(self):
        """Test summary rendering with loaded plugins."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=2,
            plugin_states={"serena": "connected", "test_plugin": "error"},
            plugin_types={
                "serena": ["semantic_search", "memory"],
                "test_plugin": ["custom_lint"],
            },
        )

        result = node.render_content()

        assert "Loaded Plugins" in result
        assert "serena" in result
        assert "test_plugin" in result
        assert "[OK]" in result  # connected status
        assert "[ERR]" in result  # error status

    def test_render_content_shows_overview(self):
        """Test content rendering includes builtin and loaded counts."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=2,
            plugin_states={"serena": "connected"},
            plugin_types={"serena": ["semantic_search", "memory"]},
        )

        result = node.render_content()

        assert "Builtin types: 16" in result
        assert "Loaded plugin servers: 2" in result

    def test_render_content_with_events(self):
        """Test content rendering includes events."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=1,
        )
        node.connection_events = [{"time": "10:30:00", "message": "serena: new -> connected"}]

        result = node.render_content()

        assert "Recent Events" in result
        assert "serena: new -> connected" in result

    def test_render_digest(self):
        """Test digest formatting."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=2,
        )

        name = node.render_digest()

        assert "Plugin Manager" in name
        assert "16 builtin" in name
        assert "2 loaded" in name


class TestPluginManagerNodeStateUpdates:
    """Tests for PluginManagerNode state update methods."""

    def test_update_plugin_state_new(self):
        """Test updating state for a new plugin."""

        node = PluginManagerNode(node_id="plugin_manager", builtin_count=16)

        node.update_plugin_state("serena", "connected", ["semantic_search", "memory"])

        assert node.plugin_states["serena"] == "connected"
        assert node.plugin_types["serena"] == ["semantic_search", "memory"]
        assert node.loaded_count == 1

    def test_update_plugin_state_change(self):
        """Test updating state of existing plugin."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            plugin_states={"serena": "connecting"},
        )

        node.update_plugin_state("serena", "connected")

        assert node.plugin_states["serena"] == "connected"
        assert len(node.connection_events) == 1
        assert "connecting -> connected" in node.connection_events[0]["message"]

    def test_update_plugin_state_disconnected_updates_count(self):
        """Test that disconnected status decreases loaded count."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=2,
            plugin_states={"serena": "connected", "test": "connected"},
        )

        node.update_plugin_state("serena", "disconnected")

        assert node.loaded_count == 1

    def test_unregister_plugin(self):
        """Test unregistering a plugin removes it from tracking."""

        node = PluginManagerNode(
            node_id="plugin_manager",
            builtin_count=16,
            loaded_count=2,
            plugin_states={"serena": "connected", "test": "connected"},
            plugin_types={"serena": ["memory"], "test": ["lint"]},
        )

        node.unregister_plugin("serena")

        assert "serena" not in node.plugin_states
        assert "serena" not in node.plugin_types
        assert node.loaded_count == 1

    def test_connection_events_limited(self):
        """Test that connection events are limited to max_events."""

        node = PluginManagerNode(node_id="plugin_manager", builtin_count=16, max_events=3)

        # Add more events than max_events
        for i in range(5):
            node.update_plugin_state(f"plugin_{i}", "connected")

        # Should only keep the last 3
        assert len(node.connection_events) == 3
        assert "plugin_2" in node.connection_events[0]["message"]
        assert "plugin_4" in node.connection_events[2]["message"]


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
            default_expansion=Expansion.ALL,
            agent_state=AgentState.RUNNING,
            task="Research the API",
        )

        data = node.to_dict()

        assert data["node_type"] == "AgentNode"
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
            default_expansion=Expansion.ALL,
            agent_state=AgentState.DONE,
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
            "node_type": "AgentNode",
            "node_id": "agent1",
            "agent_id": "test-agent",
            "agent_type": "helper",
            "expansion": "header",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, AgentNode)


# =============================================================================
# StatementNode Serialization Tests
# =============================================================================


class TestStatementNodeSerialization:
    """Tests for StatementNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test StatementNode serialization to dict."""
        node = StatementNode(
            node_id="stmt1",
            statement_id="uuid-123",
            source="v = text('main.py')",
            index=0,
            timestamp=1234567890.0,
            status="ok",
            default_expansion=Expansion.ALL,
        )

        data = node.to_dict()

        assert data["node_type"] == "StatementNode"
        assert data["node_id"] == "stmt1"
        assert data["statement_id"] == "uuid-123"
        assert data["source"] == "v = text('main.py')"
        assert data["index"] == 0
        assert data["timestamp"] == 1234567890.0
        assert data["status"] == "ok"

    def test_roundtrip(self):
        """Test StatementNode serialization round-trip."""
        original = StatementNode(
            node_id="stmt1",
            statement_id="uuid-456",
            source="g = group(v1, v2)",
            index=5,
            timestamp=1234567890.5,
            status="error",
            default_expansion=Expansion.CONTENT,
        )
        original._result_nodes["exec-1"] = "result-node-1"

        data = original.to_dict()
        restored = StatementNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.statement_id == original.statement_id
        assert restored.source == original.source
        assert restored.index == original.index
        assert restored.timestamp == original.timestamp
        assert restored.status == original.status
        assert restored._result_nodes == original._result_nodes

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to StatementNode."""
        data = {
            "node_type": "StatementNode",
            "node_id": "stmt1",
            "statement_id": "test-stmt",
            "source": "x = 1",
            "expansion": "header",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, StatementNode)

    def test_render_content(self):
        """Test StatementNode renders source in fenced code block."""
        node = StatementNode(
            node_id="stmt1",
            source="v = text('main.py')",
        )

        content = node.render_content()

        assert "```python" in content
        assert "v = text('main.py')" in content
        assert "```" in content

    def test_render_digest(self):
        """Test StatementNode digest format."""
        node = StatementNode(
            node_id="stmt1",
            source="some_long_function_call_that_exceeds_forty_characters()",
            status="ok",
        )

        digest = node.render_digest()

        assert "[ok]" in digest
        assert "..." in digest  # Long source should be truncated

    def test_add_result_tracking(self):
        """Test StatementNode tracks result children."""
        node = StatementNode(node_id="stmt1")

        node.add_result("result-1", "exec-uuid-1")
        node.add_result("result-2", "exec-uuid-2")

        assert "exec-uuid-1" in node._result_nodes
        assert "exec-uuid-2" in node._result_nodes
        assert node._result_nodes["exec-uuid-1"] == "result-1"


# =============================================================================
# StatementResultNode Serialization Tests
# =============================================================================


class TestStatementResultNodeSerialization:
    """Tests for StatementResultNode to_dict/from_dict."""

    def test_to_dict_basic(self):
        """Test StatementResultNode serialization to dict."""
        node = StatementResultNode(
            node_id="result1",
            execution_id="exec-uuid-1",
            statement_id="stmt-uuid-1",
            status="ok",
            stdout="hello\n",
            stderr="",
            duration_ms=15.5,
            default_expansion=Expansion.ALL,
        )

        data = node.to_dict()

        assert data["node_type"] == "StatementResultNode"
        assert data["node_id"] == "result1"
        assert data["execution_id"] == "exec-uuid-1"
        assert data["statement_id"] == "stmt-uuid-1"
        assert data["status"] == "ok"
        assert data["stdout"] == "hello\n"
        assert data["duration_ms"] == 15.5

    def test_roundtrip(self):
        """Test StatementResultNode serialization round-trip."""
        original = StatementResultNode(
            node_id="result1",
            execution_id="exec-uuid-2",
            statement_id="stmt-uuid-2",
            status="error",
            stdout="",
            stderr="syntax error",
            exception={"type": "SyntaxError", "message": "invalid syntax"},
            state_trace={"added": {"v": "TextNode"}, "changed": {}, "deleted": []},
            duration_ms=25.0,
            default_expansion=Expansion.CONTENT,
        )

        data = original.to_dict()
        restored = StatementResultNode._from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.execution_id == original.execution_id
        assert restored.statement_id == original.statement_id
        assert restored.status == original.status
        assert restored.stdout == original.stdout
        assert restored.stderr == original.stderr
        assert restored.exception == original.exception
        assert restored.state_trace == original.state_trace
        assert restored.duration_ms == original.duration_ms

    def test_factory_dispatch(self):
        """Test that ContextNode.from_dict dispatches to StatementResultNode."""
        data = {
            "node_type": "StatementResultNode",
            "node_id": "result1",
            "execution_id": "test-exec",
            "statement_id": "test-stmt",
            "status": "ok",
            "expansion": "header",
        }

        node = ContextNode.from_dict(data)

        assert isinstance(node, StatementResultNode)

    def test_render_content_success(self):
        """Test StatementResultNode renders success output."""
        node = StatementResultNode(
            node_id="result1",
            status="ok",
            stdout="output text",
            state_trace={"added": {"v": "TextNode"}, "changed": {}, "deleted": []},
        )

        content = node.render_content()

        assert "stdout: output text" in content
        assert "created: v" in content

    def test_render_content_error(self):
        """Test StatementResultNode renders error details."""
        node = StatementResultNode(
            node_id="result1",
            status="error",
            stderr="error output",
            exception={"type": "ValueError", "message": "invalid value"},
        )

        content = node.render_content()

        assert "stderr: error output" in content
        assert "ValueError: invalid value" in content

    def test_render_content_minimal(self):
        """Test StatementResultNode renders 'ok' when no output."""
        node = StatementResultNode(
            node_id="result1",
            status="ok",
        )

        content = node.render_content()

        assert content == "ok"

    def test_render_digest(self):
        """Test StatementResultNode digest format with timing."""
        node = StatementResultNode(
            node_id="result1",
            status="ok",
            duration_ms=42.5,
        )

        digest = node.render_digest()

        assert "[ok]" in digest
        assert "42.5ms" in digest


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

        view = TextNode(node_id="view1", path="main.py", default_expansion=Expansion.ALL)
        graph.add_node(view)

        group = GroupNode(
            node_id="group1", default_expansion=Expansion.CONTENT
        )
        graph.add_node(group)

        topic = TopicNode(node_id="topic1", title="Discussion", default_expansion=Expansion.HEADER)
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
        assert restored.get_node("topic1").title == "Discussion"

        # Verify roots (view1 and group1 are roots, topic1 is not)
        root_ids = {n.node_id for n in restored.get_roots()}
        assert "view1" in root_ids
        assert "group1" in root_ids
        assert "topic1" not in root_ids  # has parent
