"""Tests for Session context stack and tool use scoping."""

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import GroupNode, MessageNode, TextNode
from activecontext.session.timeline import Timeline


class MockSession:
    """Minimal Session-like object for testing context stack operations."""

    def __init__(self) -> None:
        self._timeline = Timeline(
            session_id="test",
            cwd=".",
            context_graph=ContextGraph(),
        )
        self._context_stack: list[str] = []

    @property
    def current_group(self) -> str | None:
        return self._context_stack[-1] if self._context_stack else None

    def push_group(self, group_id: str) -> None:
        self._context_stack.append(group_id)

    def pop_group(self) -> str | None:
        return self._context_stack.pop() if self._context_stack else None

    def add_node(self, node: object) -> str:
        from activecontext.context.nodes import ContextNode

        if not isinstance(node, ContextNode):
            raise TypeError(f"Expected ContextNode, got {type(node).__name__}")

        self._timeline.context_graph.add_node(node)

        if self.current_group:
            self._timeline.context_graph.link(node.node_id, self.current_group)

        return node.node_id

    def begin_tool_use(self, tool_name: str, args: dict | None = None) -> str:
        group = GroupNode(summary_prompt=f"Tool: {tool_name}")
        self.add_node(group)

        tool_call = MessageNode(
            role="tool_call",
            content="",
            actor=f"tool:{tool_name}",
            tool_name=tool_name,
            tool_args=args or {},
        )
        self._timeline.context_graph.add_node(tool_call)
        self._timeline.context_graph.link(tool_call.node_id, group.node_id)

        self.push_group(group.node_id)
        return group.node_id

    def end_tool_use(self, summary: str | None = None) -> str | None:
        group_id = self.pop_group()

        if group_id and summary:
            group = self._timeline.context_graph.get_node(group_id)
            if isinstance(group, GroupNode):
                group.cached_summary = summary

        return group_id


class TestContextStackBasics:
    """Test basic context stack operations."""

    def test_empty_stack_returns_none(self) -> None:
        """Test that empty stack returns None for current_group."""
        session = MockSession()
        assert session.current_group is None

    def test_push_sets_current_group(self) -> None:
        """Test that push_group sets current_group."""
        session = MockSession()
        group = GroupNode()
        session._timeline.context_graph.add_node(group)

        session.push_group(group.node_id)
        assert session.current_group == group.node_id

    def test_pop_returns_group_id(self) -> None:
        """Test that pop_group returns the group ID."""
        session = MockSession()
        group = GroupNode()
        session._timeline.context_graph.add_node(group)

        session.push_group(group.node_id)
        popped = session.pop_group()

        assert popped == group.node_id
        assert session.current_group is None

    def test_pop_empty_returns_none(self) -> None:
        """Test that pop on empty stack returns None."""
        session = MockSession()
        assert session.pop_group() is None

    def test_nested_push_pop(self) -> None:
        """Test nested push/pop operations."""
        session = MockSession()
        group1 = GroupNode(node_id="group1")
        group2 = GroupNode(node_id="group2")
        session._timeline.context_graph.add_node(group1)
        session._timeline.context_graph.add_node(group2)

        session.push_group("group1")
        assert session.current_group == "group1"

        session.push_group("group2")
        assert session.current_group == "group2"

        session.pop_group()
        assert session.current_group == "group1"

        session.pop_group()
        assert session.current_group is None


class TestAddNodeLinking:
    """Test that add_node links to current group."""

    def test_add_node_at_root(self) -> None:
        """Test adding node at root level (no group)."""
        session = MockSession()
        node = MessageNode(role="user", content="Hello", actor="user")

        session.add_node(node)

        # Node should have no parents (is a root)
        assert len(node.parent_ids) == 0
        assert node.node_id in session._timeline.context_graph._root_ids

    def test_add_node_in_group(self) -> None:
        """Test adding node inside a group."""
        session = MockSession()
        group = GroupNode(node_id="group1")
        session._timeline.context_graph.add_node(group)
        session.push_group("group1")

        node = MessageNode(role="user", content="Hello", actor="user")
        session.add_node(node)

        # Node should be child of the group
        assert "group1" in node.parent_ids
        assert node.node_id in group.children_ids

    def test_add_multiple_nodes_in_group(self) -> None:
        """Test adding multiple nodes to same group."""
        session = MockSession()
        group = GroupNode(node_id="group1")
        session._timeline.context_graph.add_node(group)
        session.push_group("group1")

        node1 = MessageNode(role="user", content="First", actor="user")
        node2 = TextNode(path="test.py")

        session.add_node(node1)
        session.add_node(node2)

        # Both should be children of the group
        assert "group1" in node1.parent_ids
        assert "group1" in node2.parent_ids
        assert node1.node_id in group.children_ids
        assert node2.node_id in group.children_ids

    def test_add_node_rejects_non_context_node(self) -> None:
        """Test that add_node rejects non-ContextNode objects."""
        session = MockSession()

        with pytest.raises(TypeError, match="Expected ContextNode"):
            session.add_node("not a node")  # type: ignore


class TestBeginEndToolUse:
    """Test begin_tool_use and end_tool_use methods."""

    def test_begin_tool_use_creates_group(self) -> None:
        """Test that begin_tool_use creates a group."""
        session = MockSession()
        group_id = session.begin_tool_use("view", {"path": "main.py"})

        group = session._timeline.context_graph.get_node(group_id)
        assert isinstance(group, GroupNode)
        assert group.summary_prompt == "Tool: view"

    def test_begin_tool_use_creates_tool_call(self) -> None:
        """Test that begin_tool_use creates a tool_call MessageNode."""
        session = MockSession()
        group_id = session.begin_tool_use("grep", {"pattern": "TODO"})

        group = session._timeline.context_graph.get_node(group_id)
        assert group is not None
        children = session._timeline.context_graph.get_children(group_id)

        # Should have one child: the tool_call message
        assert len(children) == 1
        tool_call = children[0]
        assert isinstance(tool_call, MessageNode)
        assert tool_call.role == "tool_call"
        assert tool_call.tool_name == "grep"
        assert tool_call.tool_args == {"pattern": "TODO"}

    def test_begin_tool_use_pushes_group(self) -> None:
        """Test that begin_tool_use pushes group to stack."""
        session = MockSession()
        group_id = session.begin_tool_use("view")

        assert session.current_group == group_id

    def test_nodes_added_after_begin_are_children(self) -> None:
        """Test that nodes added after begin_tool_use become children."""
        session = MockSession()
        group_id = session.begin_tool_use("view")

        view = TextNode(path="main.py")
        session.add_node(view)

        assert group_id in view.parent_ids

    def test_end_tool_use_pops_stack(self) -> None:
        """Test that end_tool_use pops the stack."""
        session = MockSession()
        session.begin_tool_use("view")
        session.end_tool_use()

        assert session.current_group is None

    def test_end_tool_use_sets_summary(self) -> None:
        """Test that end_tool_use sets the group summary."""
        session = MockSession()
        group_id = session.begin_tool_use("view")
        session.end_tool_use(summary="Opened main.py")

        group = session._timeline.context_graph.get_node(group_id)
        assert isinstance(group, GroupNode)
        assert group.cached_summary == "Opened main.py"

    def test_end_tool_use_returns_group_id(self) -> None:
        """Test that end_tool_use returns the group ID."""
        session = MockSession()
        group_id = session.begin_tool_use("view")
        returned_id = session.end_tool_use()

        assert returned_id == group_id


class TestNestedToolUse:
    """Test nested tool use scenarios."""

    def test_nested_tool_calls(self) -> None:
        """Test that nested tool calls work correctly."""
        session = MockSession()

        # Outer tool call
        outer_id = session.begin_tool_use("search")
        assert session.current_group == outer_id

        # Inner tool call (nested)
        inner_id = session.begin_tool_use("view")
        assert session.current_group == inner_id

        # Inner group should be child of outer group
        inner_group = session._timeline.context_graph.get_node(inner_id)
        assert inner_group is not None
        assert outer_id in inner_group.parent_ids

        # End inner
        session.end_tool_use(summary="Inner done")
        assert session.current_group == outer_id

        # End outer
        session.end_tool_use(summary="Outer done")
        assert session.current_group is None

    def test_nodes_in_nested_tool_go_to_inner_group(self) -> None:
        """Test that nodes added in nested context go to inner group."""
        session = MockSession()

        outer_id = session.begin_tool_use("search")
        inner_id = session.begin_tool_use("view")

        # Add a view node while in inner group
        view = TextNode(path="found.py")
        session.add_node(view)

        # View should be child of inner, not outer
        assert inner_id in view.parent_ids
        assert outer_id not in view.parent_ids


class TestToolUseDAGStructure:
    """Test the DAG structure created by tool use."""

    def test_complete_tool_use_structure(self) -> None:
        """Test the complete structure of a tool use."""
        session = MockSession()

        # Simulate: user asks, agent uses view tool
        user_msg = MessageNode(role="user", content="Show main.py", actor="user")
        session.add_node(user_msg)

        group_id = session.begin_tool_use("view", {"path": "main.py"})

        view = TextNode(path="main.py", tokens=2000)
        session.add_node(view)

        session.end_tool_use(summary="Opened main.py")

        agent_msg = MessageNode(role="assistant", content="Here it is", actor="agent")
        session.add_node(agent_msg)

        # Verify structure
        graph = session._timeline.context_graph

        # User message and agent message should be roots
        assert user_msg.node_id in graph._root_ids
        assert agent_msg.node_id in graph._root_ids

        # Group should be a root (not child of user message)
        assert group_id in graph._root_ids

        # View should be child of group
        assert group_id in view.parent_ids

        # Group should have tool_call as child
        children = graph.get_children(group_id)
        tool_calls = [c for c in children if isinstance(c, MessageNode) and c.role == "tool_call"]
        assert len(tool_calls) == 1
