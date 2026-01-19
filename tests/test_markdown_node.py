"""Tests for MarkdownNode context node.

Tests coverage for:
- src/activecontext/context/nodes.py (MarkdownNode class)
- Markdown parsing and tree construction
- Rendering in different states
- Sibling navigation
- Serialization/deserialization
"""

from __future__ import annotations

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import MarkdownNode
from activecontext.context.state import NodeState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_markdown():
    """Simple markdown with one h1 heading (no preamble, so root is elided)."""
    return """# Example Markdown Text

This text (trimmed) will be the summary of the markdown node.

## Child Node

This text belongs to the child.

### Grandchild

Even deeper content here.

## Second Child

Another section.
"""


@pytest.fixture
def markdown_with_preamble():
    """Markdown with preamble (document root kept)."""
    return """Some introductory text before any heading.

# Main Heading

Main content here.

## Subheading

Sub content.
"""


@pytest.fixture
def multi_h1_markdown():
    """Markdown with multiple h1 headings (document root kept)."""
    return """Preamble text.

# First Section

Content of first section.

## Subsection

Subsection content.

# Second Section

Content of second section.
"""


@pytest.fixture
def no_heading_markdown():
    """Markdown with no headings."""
    return """Just some plain text content.

With multiple paragraphs.

And more text here.
"""


@pytest.fixture
def h2_only_markdown():
    """Markdown with only h2+ headings (no h1)."""
    return """Introduction text.

## First Topic

Content for first topic.

## Second Topic

Content for second topic.
"""


@pytest.fixture
def graph_with_markdown():
    """Create a graph with markdown nodes added."""
    graph = ContextGraph()
    root, all_nodes = MarkdownNode.from_markdown(
        path="test.md",
        content="""# Main

Summary paragraph here.

## Section A

Section A content.

## Section B

Section B content.
""",
        tokens=2000,
        state=NodeState.DETAILS,
    )
    for node in all_nodes:
        graph.add_node(node)
    return graph, root, all_nodes


# =============================================================================
# Parsing Tests
# =============================================================================


class TestMarkdownParsing:
    """Tests for markdown parsing and tree construction."""

    def test_no_preamble_elides_document_root(self, simple_markdown):
        """Test that no preamble + <=1 h1 elides the document root."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown
        )

        # Root should be the first heading (h1), keeping its original level
        assert root.heading == "Example Markdown Text"
        assert root.level == 1  # Original level preserved
        assert root.path == "test.md"
        assert "This text (trimmed)" in root.content

    def test_preamble_keeps_document_root(self, markdown_with_preamble):
        """Test that preamble content preserves document root."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=markdown_with_preamble
        )

        # Root should be the document node (level 0)
        assert root.heading == ""
        assert root.level == 0
        assert "introductory text" in root.content

        # h1 should be a child
        assert len(root.child_order) == 1

    def test_multiple_h1_keeps_document_root(self, multi_h1_markdown):
        """Test that multiple h1 headings preserve document root."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=multi_h1_markdown
        )

        # Root should be the document node
        assert root.heading == ""
        assert root.level == 0
        assert "Preamble" in root.content

        # Should have 2 h1 children
        assert len(root.child_order) == 2

    def test_no_headings_single_root(self, no_heading_markdown):
        """Test markdown with no headings creates single root."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=no_heading_markdown
        )

        assert len(nodes) == 1
        assert root.heading == ""
        assert "Just some plain text" in root.content
        assert len(root.child_order) == 0

    def test_h2_only_creates_hierarchy(self, h2_only_markdown):
        """Test markdown with only h2+ creates proper hierarchy."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=h2_only_markdown
        )

        # Root should hold the introduction
        assert "Introduction text" in root.content

        # Should have 2 children (## headings)
        assert len(root.child_order) == 2

    def test_nested_headings_hierarchy(self, simple_markdown):
        """Test that heading levels create proper parent-child structure."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown
        )

        # Find the "Child Node" (## = level 2, preserved)
        child_nodes = [n for n in nodes if n.heading == "Child Node"]
        assert len(child_nodes) == 1
        child = child_nodes[0]
        assert child.level == 2  # Original ## level preserved

        # Find the "Grandchild" (### = level 3, preserved)
        grandchild_nodes = [n for n in nodes if n.heading == "Grandchild"]
        assert len(grandchild_nodes) == 1
        grandchild = grandchild_nodes[0]
        assert grandchild.level == 3  # Original ### level preserved

        # Grandchild should be child of Child Node
        assert grandchild.node_id in child.child_order

    def test_node_content_assignment(self, simple_markdown):
        """Test that content is correctly assigned to each node."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown
        )

        # Root content should be text between # and ##
        assert "summary of the markdown node" in root.content

        # Child content should be text between ## and ###
        child = [n for n in nodes if n.heading == "Child Node"][0]
        assert "This text belongs to the child" in child.content


# =============================================================================
# Rendering Tests
# =============================================================================


class TestMarkdownRendering:
    """Tests for MarkdownNode rendering in different states."""

    def test_render_hidden_returns_empty(self, simple_markdown):
        """Test HIDDEN state returns empty string."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown, state=NodeState.HIDDEN
        )

        result = root.Render()
        assert result == ""

    def test_render_collapsed_shows_token_count(self, simple_markdown):
        """Test COLLAPSED state shows heading with token count."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown, state=NodeState.COLLAPSED
        )

        result = root.Render()
        assert "Example Markdown Text" in result
        assert "tokens" in result

    def test_render_summary_shows_first_paragraph(self, simple_markdown):
        """Test SUMMARY state shows heading and first paragraph."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown, state=NodeState.SUMMARY
        )

        result = root.Render()
        assert "Example Markdown Text" in result
        assert "summary of the markdown node" in result
        # Should not include child content
        assert "Child Node" not in result

    def test_render_details_collapses_children(self, simple_markdown):
        """Test DETAILS state shows full content with collapsed children."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown, state=NodeState.DETAILS
        )

        # Add nodes to graph for rendering
        graph = ContextGraph()
        for node in nodes:
            graph.add_node(node)

        result = root.Render()

        # Should have root content
        assert "summary of the markdown node" in result

        # Children should appear as collapsed placeholders
        # ## = level 2, so renders with ##
        assert "## Child Node" in result
        assert "tokens)" in result  # Token count in placeholder

    def test_render_all_includes_children(self, simple_markdown):
        """Test ALL state renders children according to their own states."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown, state=NodeState.ALL
        )

        # Add nodes to graph for rendering
        graph = ContextGraph()
        for node in nodes:
            graph.add_node(node)

        result = root.Render()

        # Should have all content (since all nodes are ALL state)
        assert "summary of the markdown node" in result
        assert "This text belongs to the child" in result
        assert "Even deeper content" in result
        assert "Another section" in result

    def test_render_all_respects_child_states(self, simple_markdown):
        """Test ALL state respects each child's individual state."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md", content=simple_markdown, state=NodeState.ALL
        )

        # Add nodes to graph
        graph = ContextGraph()
        for node in nodes:
            graph.add_node(node)

        # Set one child to COLLAPSED
        child_node = [n for n in nodes if n.heading == "Child Node"][0]
        child_node.SetState(NodeState.COLLAPSED)

        result = root.Render()

        # Root content should be present
        assert "summary of the markdown node" in result

        # Child Node should be collapsed (not showing its content)
        assert "Child Node" in result
        assert "(tokens:" in result  # Collapsed format with token breakdown
        # Child's full content should NOT appear
        assert "This text belongs to the child" not in result

        # Second Child (still ALL) should render fully
        assert "Another section" in result


# =============================================================================
# Navigation Tests
# =============================================================================


class TestMarkdownNavigation:
    """Tests for sibling navigation in markdown trees."""

    def test_next_sibling(self, graph_with_markdown):
        """Test next_sibling returns correct sibling."""
        graph, root, all_nodes = graph_with_markdown

        # Find Section A and Section B
        section_a = [n for n in all_nodes if n.heading == "Section A"][0]
        section_b = [n for n in all_nodes if n.heading == "Section B"][0]

        # Section A's next sibling should be Section B
        next_sib = section_a.next_sibling()
        assert next_sib is not None
        assert next_sib.node_id == section_b.node_id

    def test_prev_sibling(self, graph_with_markdown):
        """Test prev_sibling returns correct sibling."""
        graph, root, all_nodes = graph_with_markdown

        section_a = [n for n in all_nodes if n.heading == "Section A"][0]
        section_b = [n for n in all_nodes if n.heading == "Section B"][0]

        # Section B's prev sibling should be Section A
        prev_sib = section_b.prev_sibling()
        assert prev_sib is not None
        assert prev_sib.node_id == section_a.node_id

    def test_first_sibling_has_no_prev(self, graph_with_markdown):
        """Test first sibling returns None for prev_sibling."""
        graph, root, all_nodes = graph_with_markdown

        section_a = [n for n in all_nodes if n.heading == "Section A"][0]

        prev_sib = section_a.prev_sibling()
        assert prev_sib is None

    def test_last_sibling_has_no_next(self, graph_with_markdown):
        """Test last sibling returns None for next_sibling."""
        graph, root, all_nodes = graph_with_markdown

        section_b = [n for n in all_nodes if n.heading == "Section B"][0]

        next_sib = section_b.next_sibling()
        assert next_sib is None

    def test_root_has_no_siblings(self, graph_with_markdown):
        """Test root node has no siblings."""
        graph, root, all_nodes = graph_with_markdown

        assert root.next_sibling() is None
        assert root.prev_sibling() is None


# =============================================================================
# Serialization Tests
# =============================================================================


class TestMarkdownSerialization:
    """Tests for MarkdownNode serialization/deserialization."""

    def test_to_dict_roundtrip(self, simple_markdown):
        """Test that to_dict/from_dict preserves all fields."""
        root, nodes = MarkdownNode.from_markdown(
            path="test.md",
            content=simple_markdown,
            summary_tokens=150,
        )

        # Serialize
        data = root.to_dict()

        # Verify key fields
        assert data["node_type"] == "markdown"
        assert data["path"] == "test.md"
        assert data["heading"] == "Example Markdown Text"
        assert data["level"] == 1  # Original # level preserved
        assert data["summary_tokens"] == 150
        assert len(data["child_order"]) > 0

        # Deserialize
        restored = MarkdownNode._from_dict(data)

        assert restored.node_id == root.node_id
        assert restored.path == root.path
        assert restored.heading == root.heading
        assert restored.level == root.level
        assert restored.content == root.content
        assert restored.child_order == root.child_order
        assert restored.summary_tokens == root.summary_tokens

    def test_from_dict_in_factory(self, simple_markdown):
        """Test that ContextNode.from_dict dispatches to MarkdownNode."""
        from activecontext.context.nodes import ContextNode

        root, _ = MarkdownNode.from_markdown(path="test.md", content=simple_markdown)
        data = root.to_dict()

        restored = ContextNode.from_dict(data)

        assert isinstance(restored, MarkdownNode)
        assert restored.heading == root.heading


# =============================================================================
# Node Properties Tests
# =============================================================================


class TestMarkdownNodeProperties:
    """Tests for MarkdownNode properties and methods."""

    def test_node_type(self, simple_markdown):
        """Test node_type returns 'markdown'."""
        root, _ = MarkdownNode.from_markdown(path="test.md", content=simple_markdown)
        assert root.node_type == "markdown"

    def test_get_digest(self, simple_markdown):
        """Test GetDigest returns expected fields."""
        root, _ = MarkdownNode.from_markdown(path="test.md", content=simple_markdown)

        digest = root.GetDigest()

        assert digest["type"] == "markdown"
        assert digest["heading"] == "Example Markdown Text"
        assert digest["level"] == 1  # Original # level preserved
        assert "child_count" in digest
        assert "content_tokens" in digest

    def test_estimate_tokens(self, simple_markdown):
        """Test token estimation."""
        root, _ = MarkdownNode.from_markdown(path="test.md", content=simple_markdown)

        # ~4 chars per token
        estimated = root._estimate_tokens("This is a test string.")  # 22 chars
        assert estimated == 5  # 22 // 4 = 5

    def test_get_summary_truncates(self):
        """Test _get_summary truncates to summary_tokens."""
        long_content = "A" * 1000  # 1000 chars

        root = MarkdownNode(
            heading="Test",
            content=long_content,
            summary_tokens=50,  # 50 * 4 = 200 char budget
        )

        summary = root._get_summary()

        # Should be truncated to ~200 chars + "..."
        assert len(summary) < len(long_content)
        assert summary.endswith("...")

    def test_get_summary_uses_first_paragraph(self):
        """Test _get_summary extracts first paragraph."""
        content = """First paragraph here.

Second paragraph is ignored for summary.

Third paragraph too.
"""
        root = MarkdownNode(heading="Test", content=content, summary_tokens=200)

        summary = root._get_summary()

        assert "First paragraph" in summary
        assert "Second paragraph" not in summary

    def test_total_tokens_includes_children(self, graph_with_markdown):
        """Test _total_tokens includes descendant content."""
        graph, root, all_nodes = graph_with_markdown

        total = root._total_tokens()

        # Should include root + all children
        individual_sum = sum(n._estimate_tokens(n.content) for n in all_nodes)
        assert total == individual_sum
