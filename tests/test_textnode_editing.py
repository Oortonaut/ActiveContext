"""Tests for TextNode live editing: replace_lines() and file change tracking.

Tests coverage for:
- src/activecontext/context/nodes.py (replace_lines, file watcher registry,
  on_file_change propagation)
"""

from __future__ import annotations

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    LineChange,
    TextNode,
    _file_watchers,
    get_watchers,
    on_file_change,
    register_file_watcher,
    unregister_file_watcher,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    node_id: str = "txt1",
    path: str = "src/main.py",
    lines: list[str] | None = None,
    pos: str = "1:0",
    end_pos: str | None = None,
) -> TextNode:
    """Create a TextNode with optional pre-populated lines."""
    node = TextNode(node_id=node_id, path=path, pos=pos, end_pos=end_pos)
    if lines is not None:
        node._lines = list(lines)
    return node


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the module-level file watcher registry before each test."""
    _file_watchers.clear()
    yield
    _file_watchers.clear()


# ===================================================================
# replace_lines() tests
# ===================================================================


class TestReplaceLines:
    """Tests for TextNode.replace_lines()."""

    def test_basic_replacement(self):
        """Replace a single line in the middle of content."""
        node = _make_node(lines=["aaa", "bbb", "ccc", "ddd"])
        node.replace_lines(line_no=2, num_removed=1, new_lines=["BBB"])
        assert node._lines == ["aaa", "BBB", "ccc", "ddd"]

    def test_multi_line_replacement(self):
        """Replace multiple lines with a different number of lines."""
        node = _make_node(lines=["aaa", "bbb", "ccc", "ddd"])
        node.replace_lines(line_no=2, num_removed=2, new_lines=["XXX"])
        assert node._lines == ["aaa", "XXX", "ddd"]

    def test_insert_only(self):
        """Insert lines without removing any (num_removed=0)."""
        node = _make_node(lines=["aaa", "bbb"])
        node.replace_lines(line_no=2, num_removed=0, new_lines=["NEW1", "NEW2"])
        assert node._lines == ["aaa", "NEW1", "NEW2", "bbb"]

    def test_delete_only(self):
        """Delete lines without inserting any (new_lines=[])."""
        node = _make_node(lines=["aaa", "bbb", "ccc"])
        node.replace_lines(line_no=2, num_removed=1, new_lines=[])
        assert node._lines == ["aaa", "ccc"]

    def test_replace_at_beginning(self):
        """Replace lines starting at line 1."""
        node = _make_node(lines=["aaa", "bbb", "ccc"])
        node.replace_lines(line_no=1, num_removed=1, new_lines=["FIRST"])
        assert node._lines == ["FIRST", "bbb", "ccc"]

    def test_replace_at_end(self):
        """Replace the last line."""
        node = _make_node(lines=["aaa", "bbb", "ccc"])
        node.replace_lines(line_no=3, num_removed=1, new_lines=["LAST"])
        assert node._lines == ["aaa", "bbb", "LAST"]

    def test_append_at_end(self):
        """Insert at one past the last line (append)."""
        node = _make_node(lines=["aaa", "bbb"])
        node.replace_lines(line_no=3, num_removed=0, new_lines=["ccc"])
        assert node._lines == ["aaa", "bbb", "ccc"]

    def test_out_of_range_raises_index_error(self):
        """line_no beyond content + 1 should raise IndexError."""
        node = _make_node(lines=["aaa", "bbb"])
        with pytest.raises(IndexError, match="beyond the end"):
            node.replace_lines(line_no=5, num_removed=0, new_lines=["x"])

    def test_zero_line_no_raises_value_error(self):
        """line_no < 1 should raise ValueError."""
        node = _make_node(lines=["aaa"])
        with pytest.raises(ValueError, match="must be >= 1"):
            node.replace_lines(line_no=0, num_removed=0, new_lines=["x"])

    def test_negative_line_no_raises_value_error(self):
        """Negative line_no should raise ValueError."""
        node = _make_node(lines=["aaa"])
        with pytest.raises(ValueError, match="must be >= 1"):
            node.replace_lines(line_no=-1, num_removed=0, new_lines=["x"])

    def test_version_incremented(self):
        """replace_lines should bump the node version."""
        node = _make_node(lines=["aaa", "bbb"])
        v0 = node.version
        node.replace_lines(line_no=1, num_removed=1, new_lines=["AAA"])
        assert node.version == v0 + 1

    def test_empty_content_insert(self):
        """Insert into a node with no lines."""
        node = _make_node(lines=[])
        node.replace_lines(line_no=1, num_removed=0, new_lines=["first"])
        assert node._lines == ["first"]

    def test_remove_more_than_available(self):
        """Removing more lines than exist past line_no removes what is there."""
        node = _make_node(lines=["aaa", "bbb"])
        # Remove 5 lines starting at line 2, but only 1 exists there
        node.replace_lines(line_no=2, num_removed=5, new_lines=["XXX"])
        assert node._lines == ["aaa", "XXX"]


# ===================================================================
# File watcher registry tests
# ===================================================================


class TestFileWatcherRegistry:
    """Tests for the module-level file watcher registry."""

    def test_register_and_get(self):
        """Registering a watcher should make it retrievable."""
        register_file_watcher("src/main.py", "node_a")
        assert "node_a" in get_watchers("src/main.py")

    def test_register_multiple_nodes(self):
        """Multiple nodes can watch the same file."""
        register_file_watcher("src/main.py", "node_a")
        register_file_watcher("src/main.py", "node_b")
        watchers = get_watchers("src/main.py")
        assert watchers == {"node_a", "node_b"}

    def test_unregister_removes_node(self):
        """Unregistering a watcher removes it from the set."""
        register_file_watcher("src/main.py", "node_a")
        register_file_watcher("src/main.py", "node_b")
        unregister_file_watcher("src/main.py", "node_a")
        assert get_watchers("src/main.py") == {"node_b"}

    def test_unregister_last_removes_path(self):
        """Unregistering the last watcher removes the path entry entirely."""
        register_file_watcher("src/main.py", "node_a")
        unregister_file_watcher("src/main.py", "node_a")
        assert get_watchers("src/main.py") == set()
        assert "src/main.py" not in _file_watchers

    def test_unregister_nonexistent_is_noop(self):
        """Unregistering a non-existent watcher should not raise."""
        unregister_file_watcher("nonexistent.py", "node_x")
        # No error expected

    def test_get_watchers_returns_copy(self):
        """get_watchers() should return a copy, not the internal set."""
        register_file_watcher("src/main.py", "node_a")
        watchers = get_watchers("src/main.py")
        watchers.add("node_z")
        # Internal state should not be affected
        assert "node_z" not in get_watchers("src/main.py")

    def test_get_watchers_unknown_path(self):
        """Querying an unknown path returns an empty set."""
        assert get_watchers("does_not_exist.py") == set()


# ===================================================================
# TextNode auto-registration tests
# ===================================================================


class TestTextNodeAutoRegistration:
    """Tests for TextNode auto-registering / unregistering with the watcher."""

    def test_node_auto_registers_on_creation(self):
        """TextNode with a path should auto-register in __post_init__."""
        TextNode(node_id="reg1", path="src/auto.py")
        assert "reg1" in get_watchers("src/auto.py")

    def test_node_without_path_does_not_register(self):
        """TextNode without a path should not register."""
        TextNode(node_id="reg2", path="")
        # No watchers expected for empty path
        assert get_watchers("") == set()

    def test_unregister_watcher(self):
        """Calling unregister_watcher() removes the node from the registry."""
        node = TextNode(node_id="reg3", path="src/cleanup.py")
        assert "reg3" in get_watchers("src/cleanup.py")
        node.unregister_watcher()
        assert "reg3" not in get_watchers("src/cleanup.py")


# ===================================================================
# on_file_change() propagation tests
# ===================================================================


class TestOnFileChange:
    """Tests for on_file_change() propagating changes to watching nodes."""

    def _make_graph_with_node(
        self,
        node_id: str = "fc1",
        path: str = "src/target.py",
        lines: list[str] | None = None,
        pos: str = "1:0",
        end_pos: str | None = None,
    ) -> tuple[ContextGraph, TextNode]:
        """Create a ContextGraph containing a single TextNode."""
        graph = ContextGraph()
        node = TextNode(
            node_id=node_id,
            path=path,
            pos=pos,
            end_pos=end_pos,
        )
        if lines is not None:
            node._lines = list(lines)
        graph.add_node(node)
        return graph, node

    def test_propagates_to_watcher(self):
        """on_file_change should apply changes to watching TextNodes."""
        graph, node = self._make_graph_with_node(
            lines=["aaa", "bbb", "ccc"],
        )
        changes = [LineChange(line_no=2, num_removed=1, new_lines=["BBB"])]
        notified = on_file_change("src/target.py", changes, graph=graph)
        assert node.node_id in notified
        assert node._lines == ["aaa", "BBB", "ccc"]

    def test_no_graph_returns_ids_only(self):
        """Without a graph, on_file_change returns IDs but cannot apply changes."""
        register_file_watcher("src/test.py", "noid1")
        notified = on_file_change(
            "src/test.py",
            [LineChange(line_no=1, num_removed=0, new_lines=["x"])],
        )
        assert "noid1" in notified

    def test_no_watchers_returns_empty(self):
        """If no nodes watch a file, on_file_change returns empty."""
        graph = ContextGraph()
        notified = on_file_change(
            "src/unwatched.py",
            [LineChange(line_no=1, num_removed=0, new_lines=["x"])],
            graph=graph,
        )
        assert notified == []

    def test_change_outside_range_ignored(self):
        """Changes outside the node's displayed range should not modify content."""
        graph, node = self._make_graph_with_node(
            lines=["line1", "line2", "line3"],
            pos="10:0",
            end_pos="12:0",
        )
        # Change at line 5 is before the node's range (10-12)
        changes = [LineChange(line_no=5, num_removed=1, new_lines=["x"])]
        on_file_change("src/target.py", changes, graph=graph)
        # Content should be unchanged
        assert node._lines == ["line1", "line2", "line3"]

    def test_change_within_range_applied(self):
        """Changes within the node's displayed range should be applied."""
        graph, node = self._make_graph_with_node(
            lines=["line10", "line11", "line12"],
            pos="10:0",
            end_pos="12:0",
        )
        # Change at line 11 is within the node's range (10-12)
        changes = [LineChange(line_no=11, num_removed=1, new_lines=["REPLACED"])]
        on_file_change("src/target.py", changes, graph=graph)
        # Line 11 maps to local line 2 (11 - 10 + 1)
        assert node._lines == ["line10", "REPLACED", "line12"]

    def test_multiple_watchers(self):
        """Multiple nodes watching the same file all get updated."""
        graph = ContextGraph()
        node_a = TextNode(node_id="ma", path="src/shared.py", pos="1:0")
        node_a._lines = ["A1", "A2", "A3"]
        graph.add_node(node_a)

        node_b = TextNode(node_id="mb", path="src/shared.py", pos="1:0")
        node_b._lines = ["B1", "B2", "B3"]
        graph.add_node(node_b)

        changes = [LineChange(line_no=1, num_removed=1, new_lines=["FIRST"])]
        notified = on_file_change("src/shared.py", changes, graph=graph)

        assert "ma" in notified
        assert "mb" in notified
        assert node_a._lines == ["FIRST", "A2", "A3"]
        assert node_b._lines == ["FIRST", "B2", "B3"]

    def test_node_cleanup_unregisters(self):
        """After unregister_watcher(), on_file_change should not find the node."""
        graph, node = self._make_graph_with_node(
            lines=["aaa", "bbb"],
        )
        node.unregister_watcher()
        changes = [LineChange(line_no=1, num_removed=1, new_lines=["x"])]
        notified = on_file_change("src/target.py", changes, graph=graph)
        assert notified == []
        # Content should be untouched
        assert node._lines == ["aaa", "bbb"]


# ===================================================================
# TextNode expansion state and rendering tests
# ===================================================================


class TestTextNodeExpansionStates:
    """Tests for TextNode rendering across all Expansion states."""

    def test_render_collapsed(self):
        """Test HEADER (collapsed) rendering shows only header."""
        from activecontext.context.state import Expansion

        graph = ContextGraph()
        node = TextNode(
            node_id="txt1",
            path="src/test.py",
            title="Test File",
        )
        node._lines = ["line 1", "line 2", "line 3"]
        graph.add_node(node)
        node.expansion = Expansion.HEADER

        result = node.Render(cwd=".")

        # Should have header but not content
        assert "txt1" in result or "Test File" in result
        assert "line 1" not in result
        assert "line 2" not in result

    def test_render_content(self):
        """Test CONTENT rendering shows summary (or header if no summary)."""
        from activecontext.context.state import Expansion

        graph = ContextGraph()
        node = TextNode(
            node_id="txt2",
            path="src/test.py",
            title="Test File",
        )
        node._lines = ["line 1", "line 2", "line 3"]
        graph.add_node(node)
        node.expansion = Expansion.CONTENT
        node.cached_summary = "This is a test file summary."
        node.summary_stale = False

        result = node.Render(cwd=".")

        # Should include summary
        assert "This is a test file summary." in result

    def test_render_content_without_summary(self):
        """Test CONTENT without cached summary shows header only."""
        from activecontext.context.state import Expansion

        graph = ContextGraph()
        node = TextNode(
            node_id="txt3",
            path="src/test.py",
        )
        node._lines = ["line 1", "line 2", "line 3"]
        graph.add_node(node)
        node.expansion = Expansion.CONTENT
        node.cached_summary = None

        result = node.Render(cwd=".")

        # Should show header but no content
        assert "line 1" not in result

    def test_render_all_shows_detail(self):
        """Test ALL rendering shows full content."""
        import tempfile
        from pathlib import Path

        from activecontext.context.state import Expansion

        # Create a temporary file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("line 1\nline 2\nline 3\n")

            graph = ContextGraph()
            node = TextNode(
                node_id="txt4",
                path=str(test_file),
            )
            graph.add_node(node)
            node.expansion = Expansion.ALL

            result = node.Render(cwd=tmpdir)

            # Should include all content lines
            assert "line 1" in result
            assert "line 2" in result
            assert "line 3" in result

    def test_all_expansion_states_side_by_side(self):
        """Demonstrate all TextNode visibility states side-by-side."""
        import tempfile
        from pathlib import Path

        from activecontext.context.state import Expansion

        # Create a temporary file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "example.py"
            test_file.write_text("def foo():\n    return 42\n")

            graph = ContextGraph()

            # Create identical nodes with different expansion states
            nodes = {}
            for exp in [
                Expansion.HEADER,
                Expansion.CONTENT,
                Expansion.INDEX,
                Expansion.ALL,
            ]:
                node = TextNode(
                    node_id=f"node_{exp.value}",
                    path=str(test_file),
                    title=f"Example {exp.value}",
                )
                node.cached_summary = f"Summary for {exp.value} state"
                node.summary_stale = False
                node.expansion = exp
                graph.add_node(node)
                nodes[exp] = node

            # Render each and verify expected content
            header_output = nodes[Expansion.HEADER].Render(cwd=tmpdir)
            content_output = nodes[Expansion.CONTENT].Render(cwd=tmpdir)
            index_output = nodes[Expansion.INDEX].Render(cwd=tmpdir)
            all_output = nodes[Expansion.ALL].Render(cwd=tmpdir)

            # HEADER: Only metadata
            assert "def foo():" not in header_output
            assert "Summary for" not in header_output

            # CONTENT: Summary + full content (CAP2: no content/detail split)
            assert "Summary for content state" in content_output
            assert "def foo():" in content_output

            # INDEX: Same content as CONTENT for leaf nodes
            assert "def foo():" in index_output
            assert "return 42" in index_output

            # ALL: Same content as CONTENT for leaf nodes
            assert "def foo():" in all_output
            assert "return 42" in all_output

    def test_render_header_format(self):
        """Verify render_header() produces expected format."""
        from activecontext.context.state import Expansion

        graph = ContextGraph()
        node = TextNode(
            node_id="hdr1",
            path="src/test.py",
            title="My Test File",
        )
        graph.add_node(node)
        node.expansion = Expansion.HEADER

        header = node.render_header(cwd=".")

        # Header should include node ID or title
        assert "hdr1" in header or "My Test File" in header

    def test_stale_summary_not_shown(self):
        """Test that stale cached summary is not rendered."""
        from activecontext.context.state import Expansion

        graph = ContextGraph()
        node = TextNode(
            node_id="stale1",
            path="src/test.py",
        )
        node._lines = ["line 1"]
        node.cached_summary = "Old stale summary"
        node.summary_stale = True
        node.expansion = Expansion.CONTENT
        graph.add_node(node)

        result = node.Render(cwd=".")

        # Stale summary should not appear
        assert "Old stale summary" not in result
