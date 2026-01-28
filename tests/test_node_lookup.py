"""Tests for Node Lookup functionality (repl-001).

These tests cover the get() DSL function and nodes accessor for looking up
nodes by name with fuzzy matching support.
"""

import asyncio
from pathlib import Path

import pytest

from activecontext.session.timeline import Timeline, NodeLookup
from activecontext.context.graph import ContextGraph
from activecontext.context.view import NodeView


class TestGetFunction:
    """Test get() function for node lookup."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        # Create a test file
        test_file = tmp_path / "main.py"
        test_file.write_text("# Test file\n")
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_by_variable_name(self, temp_cwd: Path) -> None:
        """Test that get() finds nodes by variable name."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a node with a variable name
            await timeline.execute_statement('my_view = text("main.py")')

            # Look up by variable name
            result = await timeline.execute_statement('found = get("my_view")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            found = ns["found"]
            assert found is not None
            assert isinstance(found, NodeView)
            assert found.node_id == ns["my_view"].node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_by_node_id(self, temp_cwd: Path) -> None:
        """Test that get() finds nodes by node_id."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a node
            await timeline.execute_statement('v = text("main.py")')

            ns = timeline.get_namespace()
            node_id = ns["v"].node_id

            # Look up by node_id
            result = await timeline.execute_statement(f'found = get("{node_id}")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            found = ns["found"]
            assert found is not None
            assert found.node_id == node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_by_file_path(self, temp_cwd: Path) -> None:
        """Test that get() finds TextNodes by file path."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a text node
            await timeline.execute_statement('v = text("main.py")')

            # Look up by file path
            result = await timeline.execute_statement('found = get("main.py")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            found = ns["found"]
            assert found is not None
            assert found.node_id == ns["v"].node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_fuzzy_match_filename(self, temp_cwd: Path) -> None:
        """Test that get() supports fuzzy matching on filename."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a text node
            await timeline.execute_statement('v = text("main.py")')

            # Fuzzy match on partial filename
            result = await timeline.execute_statement('found = get("main")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            found = ns["found"]
            assert found is not None
            assert found.node_id == ns["v"].node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_returns_none_for_no_match(self, temp_cwd: Path) -> None:
        """Test that get() returns None when no match is found."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Look up nonexistent node
            result = await timeline.execute_statement('found = get("nonexistent")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["found"] is None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_by_title(self, temp_cwd: Path) -> None:
        """Test that get() finds nodes by title."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a topic with a title - title is stored in TopicNode.title
            await timeline.execute_statement('t = topic("My Important Topic")')

            # Look up by partial title match
            result = await timeline.execute_statement('found = get("Important")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            found = ns["found"]
            assert found is not None
            assert found.node_id == ns["t"].node_id
        finally:
            await timeline.close()


class TestNodesAccessor:
    """Test nodes dict-like accessor."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        test_file = tmp_path / "main.py"
        test_file.write_text("# Test file\n")
        return tmp_path

    @pytest.mark.asyncio
    async def test_nodes_indexing(self, temp_cwd: Path) -> None:
        """Test that nodes[name] works for exact lookup."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a node
            await timeline.execute_statement('v = text("main.py")')

            # Look up by indexing
            result = await timeline.execute_statement('found = nodes["main.py"]')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            found = ns["found"]
            assert found is not None
            assert found.node_id == ns["v"].node_id
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nodes_raises_keyerror_for_no_match(self, temp_cwd: Path) -> None:
        """Test that nodes[name] raises KeyError when no match found."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('nodes["nonexistent"]')
            assert result.status.value == "error"
            assert "KeyError" in str(result.exception)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nodes_contains(self, temp_cwd: Path) -> None:
        """Test 'in' operator with nodes accessor."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("main.py")')

            result = await timeline.execute_statement('exists = "main.py" in nodes')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["exists"] is True

            result = await timeline.execute_statement('exists = "nonexistent" in nodes')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["exists"] is False
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nodes_len(self, temp_cwd: Path) -> None:
        """Test len(nodes) returns node count."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('count = len(nodes)')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            initial_count = ns["count"]

            await timeline.execute_statement('v = text("main.py")')
            await timeline.execute_statement('t = topic("Test")')

            result = await timeline.execute_statement('count = len(nodes)')
            ns = timeline.get_namespace()
            assert ns["count"] == initial_count + 2
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nodes_keys(self, temp_cwd: Path) -> None:
        """Test nodes.keys() returns node identifiers."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("main.py")')
            await timeline.execute_statement('t = topic("Test")')

            result = await timeline.execute_statement('k = nodes.keys()')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            keys = ns["k"]
            assert isinstance(keys, list)
            # Should include variable names and node IDs
            assert "v" in keys or ns["v"].node_id in keys
            assert "t" in keys or ns["t"].node_id in keys
        finally:
            await timeline.close()


class TestNodeLookupClass:
    """Unit tests for NodeLookup class."""

    def test_score_match_exact(self) -> None:
        """Test exact match scoring."""
        lookup = NodeLookup(
            namespace_getter=lambda: {},
            graph_getter=lambda: None,
            views_getter=lambda: {},
        )
        assert lookup._score_match("test", "test") == 1.0
        assert lookup._score_match("TEST", "test") == 1.0  # Case insensitive

    def test_score_match_prefix(self) -> None:
        """Test prefix match scoring."""
        lookup = NodeLookup(
            namespace_getter=lambda: {},
            graph_getter=lambda: None,
            views_getter=lambda: {},
        )
        score = lookup._score_match("main", "main.py")
        assert score > 0.5  # High score for prefix match
        assert score < 1.0  # But not exact

    def test_score_match_substring(self) -> None:
        """Test substring match scoring."""
        lookup = NodeLookup(
            namespace_getter=lambda: {},
            graph_getter=lambda: None,
            views_getter=lambda: {},
        )
        score = lookup._score_match("view", "my_view_node")
        assert score > 0  # Some score for substring
        assert score < 0.9  # Lower than prefix

    def test_score_match_no_match(self) -> None:
        """Test no match scoring."""
        lookup = NodeLookup(
            namespace_getter=lambda: {},
            graph_getter=lambda: None,
            views_getter=lambda: {},
        )
        assert lookup._score_match("xyz", "abc") == 0.0

    def test_repr(self) -> None:
        """Test repr shows node count."""
        lookup = NodeLookup(
            namespace_getter=lambda: {},
            graph_getter=lambda: None,
            views_getter=lambda: {},
        )
        assert "0 nodes" in repr(lookup)


class TestNodeLookupUserMistakes:
    """Test that common user mistakes give helpful errors."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_empty_string_returns_none(self, temp_cwd: Path) -> None:
        """Test that get('') returns None, not an error."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('found = get("")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["found"] is None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nodes_empty_string_raises_keyerror(self, temp_cwd: Path) -> None:
        """Test that nodes[''] raises KeyError with clear message."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('nodes[""]')
            assert result.status.value == "error"
            assert "KeyError" in str(result.exception)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_get_with_none_type_error(self, temp_cwd: Path) -> None:
        """Test that get(None) gives a type error."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement('found = get(None)')
            assert result.status.value == "error"
            # Should get a type error since None is not a string
            assert "TypeError" in str(result.exception) or "attribute" in str(result.exception).lower()
        finally:
            await timeline.close()
