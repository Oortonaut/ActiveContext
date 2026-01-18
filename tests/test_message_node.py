"""Tests for MessageNode and conversation rendering with IDs."""

import time

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import MessageNode
from activecontext.context.state import NodeState
from activecontext.core.projection_engine import ProjectionConfig, ProjectionEngine


class TestMessageNodeBasics:
    """Test MessageNode creation and properties."""

    def test_create_message_node(self) -> None:
        """Test basic MessageNode creation."""
        node = MessageNode(
            role="user",
            content="Hello, world!",
            actor="user",
        )
        assert node.node_type == "message"
        assert node.role == "user"
        assert node.content == "Hello, world!"
        assert node.actor == "user"

    def test_effective_role_user(self) -> None:
        """Test effective role is USER for user messages."""
        node = MessageNode(role="user", content="test", actor="user")
        assert node.effective_role == "USER"

    def test_effective_role_assistant(self) -> None:
        """Test effective role is ASSISTANT for non-user messages."""
        node = MessageNode(role="assistant", content="test", actor="agent")
        assert node.effective_role == "ASSISTANT"

        # Tool messages are also ASSISTANT role
        tool_node = MessageNode(role="tool_call", content="", actor="tool:grep")
        assert tool_node.effective_role == "ASSISTANT"

    def test_display_label_user(self) -> None:
        """Test display label for user messages."""
        node = MessageNode(role="user", content="test", actor="user")
        assert node.display_label == "User"  # Default, overridden at render time

    def test_display_label_agent(self) -> None:
        """Test display label for agent messages."""
        node = MessageNode(role="assistant", content="test", actor="agent")
        assert node.display_label == "Agent"

    def test_display_label_agent_plan(self) -> None:
        """Test display label for agent in plan mode."""
        node = MessageNode(role="assistant", content="test", actor="agent:plan")
        assert node.display_label == "Agent (Plan)"

    def test_display_label_child_agent(self) -> None:
        """Test display label for child agents."""
        node = MessageNode(role="assistant", content="test", actor="agent:explorer")
        assert node.display_label == "Child: explorer"

    def test_display_label_tool_call(self) -> None:
        """Test display label for tool calls."""
        node = MessageNode(role="tool_call", content="", actor="tool:grep")
        assert node.display_label == "Tool Call: grep"

    def test_display_label_tool_result(self) -> None:
        """Test display label for tool results."""
        node = MessageNode(role="tool_result", content="output", actor="tool:grep")
        assert node.display_label == "Tool Result"


class TestMessageNodeSerialization:
    """Test MessageNode serialization and deserialization."""

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        node = MessageNode(
            node_id="abc12345",
            role="user",
            content="Hello",
            actor="user",
            tool_name=None,
            tool_args={},
        )
        data = node.to_dict()

        assert data["node_type"] == "message"
        assert data["node_id"] == "abc12345"
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["actor"] == "user"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "node_type": "message",
            "node_id": "test1234",
            "role": "assistant",
            "content": "Response",
            "actor": "agent",
            "tokens": 500,
            "state": "details",
            "mode": "paused",
        }
        node = MessageNode._from_dict(data)

        assert node.node_id == "test1234"
        assert node.role == "assistant"
        assert node.content == "Response"
        assert node.actor == "agent"
        assert node.tokens == 500

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization roundtrips correctly."""
        original = MessageNode(
            role="tool_call",
            content="",
            actor="tool:read_file",
            tool_name="read_file",
            tool_args={"path": "main.py"},
        )
        data = original.to_dict()
        restored = MessageNode._from_dict(data)

        assert restored.role == original.role
        assert restored.actor == original.actor
        assert restored.tool_name == original.tool_name
        assert restored.tool_args == original.tool_args


class TestMessageNodeRender:
    """Test MessageNode rendering."""

    def test_render_basic_message(self) -> None:
        """Test rendering a basic message."""
        node = MessageNode(
            role="user",
            content="Hello, how are you?",
            actor="user",
        )
        rendered = node.Render()
        assert "Hello, how are you?" in rendered

    def test_render_hidden_state(self) -> None:
        """Test that hidden messages render empty."""
        node = MessageNode(
            role="user",
            content="Hello",
            actor="user",
            state=NodeState.HIDDEN,
        )
        assert node.Render() == ""

    def test_render_collapsed_state(self) -> None:
        """Test collapsed state shows metadata only."""
        node = MessageNode(
            role="user",
            content="Hello, world!",
            actor="user",
            state=NodeState.COLLAPSED,
        )
        rendered = node.Render()
        assert "USER" in rendered
        assert "13 chars" in rendered

    def test_render_tool_call(self) -> None:
        """Test rendering a tool call message."""
        node = MessageNode(
            role="tool_call",
            content="",
            actor="tool:grep",
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
            actor="tool:grep",
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
            actor="user",
        )
        node_id = graph.add_node(node)

        assert graph.get_node(node_id) is node
        assert node_id in graph

    def test_get_messages_by_type(self) -> None:
        """Test retrieving message nodes by type."""
        graph = ContextGraph()
        msg1 = MessageNode(role="user", content="Q1", actor="user")
        msg2 = MessageNode(role="assistant", content="A1", actor="agent")
        msg3 = MessageNode(role="user", content="Q2", actor="user")

        graph.add_node(msg1)
        graph.add_node(msg2)
        graph.add_node(msg3)

        message_nodes = graph.get_nodes_by_type("message")
        assert len(message_nodes) == 3


class TestMessageRendering:
    """Test _render_messages() in ProjectionEngine."""

    def test_render_messages_basic(self) -> None:
        """Test rendering messages with block merging."""
        graph = ContextGraph()

        # Create messages with increasing created_at times
        base_time = time.time()
        user_msg = MessageNode(
            role="user",
            content="What is Python?",
            actor="user",
            created_at=base_time,
        )
        agent_msg = MessageNode(
            role="assistant",
            content="Python is a programming language.",
            actor="agent",
            created_at=base_time + 1,
        )

        graph.add_node(user_msg)
        graph.add_node(agent_msg)

        engine = ProjectionEngine(ProjectionConfig())
        section = engine._render_messages(graph, 8000, "Ace")

        assert section is not None
        assert "## Conversation" in section.content
        assert "[msg:" in section.content  # Has message IDs
        assert "**Ace**" in section.content  # User display name
        assert "**Agent**" in section.content  # Agent label

    def test_render_messages_block_merging(self) -> None:
        """Test that adjacent same-role messages merge into blocks."""
        graph = ContextGraph()

        base_time = time.time()
        # User message
        user_msg = MessageNode(
            role="user",
            content="Run a test",
            actor="user",
            created_at=base_time,
        )
        # Agent with tool call (both ASSISTANT role)
        agent_msg = MessageNode(
            role="assistant",
            content="Let me run the test.",
            actor="agent",
            created_at=base_time + 1,
        )
        tool_call = MessageNode(
            role="tool_call",
            content="",
            actor="tool:pytest",
            tool_name="pytest",
            tool_args={"path": "tests/"},
            created_at=base_time + 2,
        )

        graph.add_node(user_msg)
        graph.add_node(agent_msg)
        graph.add_node(tool_call)

        engine = ProjectionEngine(ProjectionConfig())
        section = engine._render_messages(graph, 8000, "User")

        assert section is not None
        # Should have 2 blocks: USER and ASSISTANT
        # Count [msg: occurrences for block count
        msg_id_count = section.content.count("[msg:")
        assert msg_id_count == 2  # One for user block, one for agent block

    def test_render_messages_empty_graph(self) -> None:
        """Test rendering with no messages returns None."""
        graph = ContextGraph()
        engine = ProjectionEngine(ProjectionConfig())
        section = engine._render_messages(graph, 8000, "User")
        assert section is None

    def test_render_messages_preserves_order(self) -> None:
        """Test that messages are rendered in creation order."""
        graph = ContextGraph()

        base_time = time.time()
        msg1 = MessageNode(role="user", content="First", actor="user", created_at=base_time)
        msg2 = MessageNode(role="assistant", content="Second", actor="agent", created_at=base_time + 1)
        msg3 = MessageNode(role="user", content="Third", actor="user", created_at=base_time + 2)

        # Add in random order
        graph.add_node(msg3)
        graph.add_node(msg1)
        graph.add_node(msg2)

        engine = ProjectionEngine(ProjectionConfig())
        section = engine._render_messages(graph, 8000, "User")

        assert section is not None
        content = section.content
        # Verify order
        assert content.index("First") < content.index("Second")
        assert content.index("Second") < content.index("Third")


class TestUserDisplayName:
    """Test user display name resolution."""

    def test_get_user_display_name_default(self) -> None:
        """Test default user display name from environment."""
        engine = ProjectionEngine(ProjectionConfig())
        name = engine._get_user_display_name()
        # Should return something (USER, USERNAME, or "User")
        assert name is not None
        assert len(name) > 0


class TestDynamicToolLabels:
    """Test dynamic label generation for tool results."""

    def test_view_tool_shows_filename(self) -> None:
        """Test that view/read tools show the filename."""
        engine = ProjectionEngine(ProjectionConfig())
        node = MessageNode(
            role="tool_result",
            content="file contents here",
            actor="tool:read_file",
            tool_name="read_file",
            tool_args={"file_path": "/path/to/main.py"},
        )
        label = engine._get_tool_result_label(node)
        assert label == "main.py"

    def test_grep_tool_shows_pattern(self) -> None:
        """Test that grep tools show the search pattern."""
        engine = ProjectionEngine(ProjectionConfig())
        node = MessageNode(
            role="tool_result",
            content="search results",
            actor="tool:grep",
            tool_name="grep",
            tool_args={"pattern": "TODO"},
        )
        label = engine._get_tool_result_label(node)
        assert label == "grep: TODO"

    def test_shell_tool_shows_command(self) -> None:
        """Test that shell tools show the command."""
        engine = ProjectionEngine(ProjectionConfig())
        node = MessageNode(
            role="tool_result",
            content="command output",
            actor="tool:bash",
            tool_name="bash",
            tool_args={"command": "git status"},
        )
        label = engine._get_tool_result_label(node)
        assert label == "git status"

    def test_unknown_tool_shows_tool_name(self) -> None:
        """Test that unknown tools show tool name + result."""
        engine = ProjectionEngine(ProjectionConfig())
        node = MessageNode(
            role="tool_result",
            content="output",
            actor="tool:custom_tool",
            tool_name="custom_tool",
            tool_args={},
        )
        label = engine._get_tool_result_label(node)
        assert label == "custom_tool result"

    def test_no_tool_name_shows_result(self) -> None:
        """Test fallback to 'Result' when no tool info."""
        engine = ProjectionEngine(ProjectionConfig())
        node = MessageNode(
            role="tool_result",
            content="output",
            actor="tool:unknown",
        )
        label = engine._get_tool_result_label(node)
        assert label == "Result"
