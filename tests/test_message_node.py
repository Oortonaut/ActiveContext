"""Tests for MessageNode and conversation rendering with IDs."""

import time

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import MessageNode
from activecontext.context.state import Expansion
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine


class TestMessageNodeBasics:
    """Test MessageNode creation and properties."""

    def test_create_message_node(self) -> None:
        """Test basic MessageNode creation."""
        node = MessageNode(
            role="user",
            content="Hello, world!",
            originator="user",
        )
        assert node.node_type == "message"
        assert node.role == "user"
        assert node.content == "Hello, world!"
        assert node.originator == "user"

    def test_effective_role_user(self) -> None:
        """Test effective role is USER for user messages."""
        node = MessageNode(role="user", content="test", originator="user")
        assert node.effective_role == "USER"

    def test_effective_role_assistant(self) -> None:
        """Test effective role is ASSISTANT for non-user messages."""
        node = MessageNode(role="assistant", content="test", originator="agent")
        assert node.effective_role == "ASSISTANT"

        # Tool messages are also ASSISTANT role
        tool_node = MessageNode(role="tool_call", content="", originator="tool:grep")
        assert tool_node.effective_role == "ASSISTANT"

    def test_display_label_user(self) -> None:
        """Test display label for user messages."""
        node = MessageNode(role="user", content="test", originator="user")
        assert node.display_label == "User"  # Default, overridden at render time

    def test_display_label_agent(self) -> None:
        """Test display label for agent messages."""
        node = MessageNode(role="assistant", content="test", originator="agent")
        assert node.display_label == "Agent"

    def test_display_label_agent_plan(self) -> None:
        """Test display label for agent in plan mode."""
        node = MessageNode(role="assistant", content="test", originator="agent:plan")
        assert node.display_label == "Agent (Plan)"

    def test_display_label_child_agent(self) -> None:
        """Test display label for child agents."""
        node = MessageNode(role="assistant", content="test", originator="agent:explorer")
        assert node.display_label == "Child: explorer"

    def test_display_label_tool_call(self) -> None:
        """Test display label for tool calls."""
        node = MessageNode(role="tool_call", content="", originator="tool:grep")
        assert node.display_label == "Tool Call: grep"

    def test_display_label_tool_result(self) -> None:
        """Test display label for tool results."""
        node = MessageNode(role="tool_result", content="output", originator="tool:grep")
        assert node.display_label == "Tool Result"


class TestMessageNodeSerialization:
    """Test MessageNode serialization and deserialization."""

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        node = MessageNode(
            node_id="abc12345",
            role="user",
            content="Hello",
            originator="user",
            tool_name=None,
            tool_args={},
        )
        data = node.to_dict()

        assert data["node_type"] == "message"
        assert data["node_id"] == "abc12345"
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["originator"] == "user"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "node_type": "message",
            "node_id": "test1234",
            "role": "assistant",
            "content": "Response",
            "originator": "agent",
            "tokens": 500,
            "state": "details",
            "mode": "paused",
        }
        node = MessageNode._from_dict(data)

        assert node.node_id == "test1234"
        assert node.role == "assistant"
        assert node.content == "Response"
        assert node.originator == "agent"
        assert node.tokens == 500

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization roundtrips correctly."""
        original = MessageNode(
            role="tool_call",
            content="",
            originator="tool:read_file",
            tool_name="read_file",
            tool_args={"path": "main.py"},
        )
        data = original.to_dict()
        restored = MessageNode._from_dict(data)

        assert restored.role == original.role
        assert restored.originator == original.originator
        assert restored.tool_name == original.tool_name
        assert restored.tool_args == original.tool_args


class TestMessageNodeRender:
    """Test MessageNode rendering."""

    def test_render_basic_message(self) -> None:
        """Test rendering a basic message."""
        node = MessageNode(
            role="user",
            content="Hello, how are you?",
            originator="user",
        )
        rendered = node.Render()
        assert "Hello, how are you?" in rendered

    def test_render_hidden_state(self) -> None:
        """Test that hidden messages render empty."""
        node = MessageNode(
            role="user",
            content="Hello",
            originator="user",
            state=Expansion.HIDDEN,
        )
        assert node.Render() == ""

    def test_render_collapsed_state(self) -> None:
        """Test collapsed state shows metadata only."""
        node = MessageNode(
            role="user",
            content="Hello, world!",
            originator="user",
            state=Expansion.COLLAPSED,
        )
        rendered = node.Render()
        assert "User" in rendered  # Title case in new header format
        assert "(tokens:" in rendered  # New token breakdown format

    def test_render_tool_call(self) -> None:
        """Test rendering a tool call message."""
        node = MessageNode(
            role="tool_call",
            content="",
            originator="tool:grep",
            tool_name="grep",
            tool_args={"pattern": "test", "path": "src/"},
        )
        rendered = node.Render()
        assert "[Tool: grep]" in rendered
        assert 'pattern="test"' in rendered

    def test_render_tool_result(self) -> None:
        """Test rendering a tool result message."""
        node = MessageNode(
            role="tool_result",
            content="line1: test found\nline2: test again",
            originator="tool:grep",
        )
        rendered = node.Render()
        assert "test found" in rendered


class TestMessageNodeInGraph:
    """Test MessageNode integration with ContextGraph."""

    def test_add_message_to_graph(self) -> None:
        """Test adding a MessageNode to the graph."""
        graph = ContextGraph()
        node = MessageNode(
            role="user",
            content="Hello",
            originator="user",
        )
        node_id = graph.add_node(node)

        assert graph.get_node(node_id) is node
        assert node_id in graph

    def test_get_messages_by_type(self) -> None:
        """Test retrieving message nodes by type."""
        graph = ContextGraph()
        msg1 = MessageNode(role="user", content="Q1", originator="user")
        msg2 = MessageNode(role="assistant", content="A1", originator="agent")
        msg3 = MessageNode(role="user", content="Q2", originator="user")

        graph.add_node(msg1)
        graph.add_node(msg2)
        graph.add_node(msg3)

        message_nodes = graph.get_nodes_by_type("message")
        assert len(message_nodes) == 3


