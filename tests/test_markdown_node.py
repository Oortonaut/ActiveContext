"""Tests for MarkdownNode and MarkdownListItemNode."""

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import MarkdownListItemNode, MarkdownNode
from activecontext.context.state import Expansion
from activecontext.context.view import NodeView


class TestMarkdownListItemNode:
    """Tests for MarkdownListItemNode."""

    def test_creation(self):
        """Test creating a list item node."""
        item = MarkdownListItemNode(
            content="First item",
            is_ordered=False,
            indent_level=0,
            marker="-",
        )
        assert item.node_type == "markdown_list_item"
        assert item.content == "First item"
        assert not item.is_ordered
        assert item.indent_level == 0
        assert item.marker == "-"

    def test_ordered_list_item(self):
        """Test ordered list item."""
        item = MarkdownListItemNode(
            content="Numbered item",
            is_ordered=True,
            indent_level=0,
            marker="1.",
        )
        assert item.is_ordered
        assert item.marker == "1."

    def test_nested_item(self):
        """Test nested list item."""
        item = MarkdownListItemNode(
            content="Nested item",
            is_ordered=False,
            indent_level=1,
            marker="-",
        )
        assert item.indent_level == 1

    def test_render_detail(self):
        """Test rendering detail view."""
        item = MarkdownListItemNode(
            content="Test item",
            is_ordered=False,
            indent_level=1,
            marker="-",
        )
        result = item.render_content()
        assert "- Test item" in result

    def test_get_digest(self):
        """Test digest generation."""
        item = MarkdownListItemNode(content="Short", is_ordered=False)
        digest = item.GetDigest()
        assert digest["type"] == "markdown_list_item"
        assert digest["content_preview"] == "Short"
        assert not digest["is_ordered"]

    def test_serialization(self):
        """Test to_dict and from_dict."""
        item = MarkdownListItemNode(
            content="Test",
            is_ordered=True,
            indent_level=2,
            marker="3.",
        )
        data = item.to_dict()
        restored = MarkdownListItemNode._from_dict(data)
        assert restored.content == "Test"
        assert restored.is_ordered
        assert restored.indent_level == 2
        assert restored.marker == "3."


class TestMarkdownNode:
    """Tests for MarkdownNode."""

    def test_creation(self):
        """Test creating a markdown node."""
        md = MarkdownNode(content="# Hello\n- Item 1\n- Item 2")
        assert md.node_type == "markdown"
        assert md.auto_parse

    def test_parse_unordered_list(self):
        """Test parsing unordered list items."""
        md = MarkdownNode()
        md.content = "- First\n- Second\n- Third"
        items = md._parse_lists()
        assert len(items) == 3
        assert items[0] == ("First", False, 0, "-")
        assert items[1] == ("Second", False, 0, "-")
        assert items[2] == ("Third", False, 0, "-")

    def test_parse_ordered_list(self):
        """Test parsing ordered list items."""
        md = MarkdownNode()
        md.content = "1. First\n2. Second\n3. Third"
        items = md._parse_lists()
        assert len(items) == 3
        assert items[0] == ("First", True, 0, "1.")
        assert items[1] == ("Second", True, 0, "2.")
        assert items[2] == ("Third", True, 0, "3.")

    def test_parse_nested_list(self):
        """Test parsing nested list items."""
        md = MarkdownNode()
        md.content = "- Parent\n  - Child 1\n  - Child 2\n- Parent 2"
        items = md._parse_lists()
        assert len(items) == 4
        assert items[0] == ("Parent", False, 0, "-")
        assert items[1] == ("Child 1", False, 1, "-")
        assert items[2] == ("Child 2", False, 1, "-")
        assert items[3] == ("Parent 2", False, 0, "-")

    def test_parse_mixed_markers(self):
        """Test parsing lists with different markers."""
        md = MarkdownNode()
        md.content = "* Star\n- Dash\n+ Plus"
        items = md._parse_lists()
        assert len(items) == 3
        assert items[0][3] == "*"
        assert items[1][3] == "-"
        assert items[2][3] == "+"

    def test_parse_with_non_list_content(self):
        """Test that non-list content is ignored."""
        md = MarkdownNode()
        md.content = "# Header\n\nParagraph\n\n- Item 1\n- Item 2\n\nMore text"
        items = md._parse_lists()
        assert len(items) == 2
        assert items[0] == ("Item 1", False, 0, "-")
        assert items[1] == ("Item 2", False, 0, "-")

    def test_create_children_simple(self):
        """Test creating child nodes from simple list."""
        graph = ContextGraph()
        md = MarkdownNode(content="- Item 1\n- Item 2")
        graph.add_node(md)
        md.parse_and_create_children()

        assert len(md.child_order) == 2
        # Use child_order for proper ordering
        children = [graph.get_node(child_id) for child_id in md.child_order]
        assert all(isinstance(child, MarkdownListItemNode) for child in children)
        assert children[0].content == "Item 1"
        assert children[1].content == "Item 2"

    def test_create_children_nested(self):
        """Test creating nested child nodes."""
        graph = ContextGraph()
        md = MarkdownNode(content="- Parent\n  - Child 1\n  - Child 2")
        graph.add_node(md)
        md.parse_and_create_children()

        # Should have 1 direct child (Parent)
        assert len(md.child_order) == 1

        parent_node = graph.get_node(list(md.child_order)[0])
        assert isinstance(parent_node, MarkdownListItemNode)
        assert parent_node.content == "Parent"

        # Parent should have 2 children
        assert len(parent_node.child_order) == 2
        grandchildren = [graph.get_node(child_id) for child_id in parent_node.child_order]
        assert grandchildren[0].content == "Child 1"
        assert grandchildren[1].content == "Child 2"

    def test_create_children_three_levels(self):
        """Test creating three levels of nested lists."""
        graph = ContextGraph()
        md = MarkdownNode(content="- Level 1\n  - Level 2\n    - Level 3\n  - Level 2b")
        graph.add_node(md)
        md.parse_and_create_children()

        # Check the structure
        level1 = graph.get_node(list(md.child_order)[0])
        assert level1.content == "Level 1"
        assert level1.indent_level == 0

        level2_nodes = [graph.get_node(cid) for cid in level1.child_order]
        assert len(level2_nodes) == 2
        assert level2_nodes[0].content == "Level 2"
        assert level2_nodes[1].content == "Level 2b"

        # Check level 3
        assert len(level2_nodes[0].child_order) == 1
        level3 = graph.get_node(list(level2_nodes[0].child_order)[0])
        assert level3.content == "Level 3"
        assert level3.indent_level == 2

    def test_set_content_with_auto_parse(self):
        """Test that set_content triggers auto-parsing."""
        graph = ContextGraph()
        md = MarkdownNode(auto_parse=True)
        graph.add_node(md)

        md.set_content("- New item 1\n- New item 2")

        assert len(md.child_order) == 2
        children = [graph.get_node(child_id) for child_id in md.child_order]
        assert children[0].content == "New item 1"
        assert children[1].content == "New item 2"

    def test_set_content_without_auto_parse(self):
        """Test that auto_parse=False prevents automatic parsing."""
        graph = ContextGraph()
        md = MarkdownNode(auto_parse=False)
        graph.add_node(md)

        md.set_content("- Item 1\n- Item 2")

        # Should not create children
        assert len(md.child_order) == 0

    def test_reparse_replaces_old_children(self):
        """Test that re-parsing removes old children and creates new ones."""
        graph = ContextGraph()
        md = MarkdownNode(content="- Old 1\n- Old 2")
        graph.add_node(md)
        md.parse_and_create_children()

        old_child_ids = set(md.child_order)
        assert len(old_child_ids) == 2

        # Change content and reparse
        md.set_content("- New 1\n- New 2\n- New 3")

        # Should have different children
        new_child_ids = set(md.child_order)
        assert len(new_child_ids) == 3
        assert old_child_ids.isdisjoint(new_child_ids)

        # Verify new content
        children = [graph.get_node(child_id) for child_id in md.child_order]
        contents = [child.content for child in children]
        assert contents == ["New 1", "New 2", "New 3"]

    def test_individual_list_item_expansion(self):
        """Test that individual list items can have independent NodeState."""
        graph = ContextGraph()
        md = MarkdownNode(content="- Item 1\n- Item 2\n- Item 3")
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(child_id) for child_id in md.child_order]

        # Set different expansion states
        children[0].default_expansion = Expansion.HEADER
        children[1].default_expansion = Expansion.CONTENT
        children[2].default_expansion = Expansion.ALL

        assert children[0].default_expansion == Expansion.HEADER
        assert children[1].default_expansion == Expansion.CONTENT
        assert children[2].default_expansion == Expansion.ALL

    def test_get_digest(self):
        """Test digest generation."""
        md = MarkdownNode(content="Test content")
        digest = md.GetDigest()
        assert digest["type"] == "markdown"
        assert digest["content_length"] == 12

    def test_serialization(self):
        """Test to_dict and from_dict."""
        md = MarkdownNode(
            content="# Test\n- Item",
            buffer_id="buf123",
            auto_parse=False,
        )
        data = md.to_dict()
        restored = MarkdownNode._from_dict(data)
        assert restored.content == "# Test\n- Item"
        assert restored.buffer_id == "buf123"
        assert not restored.auto_parse

    def test_empty_content(self):
        """Test that empty content doesn't create children."""
        graph = ContextGraph()
        md = MarkdownNode(content="")
        graph.add_node(md)
        md.parse_and_create_children()
        assert len(md.child_order) == 0

    def test_no_graph_reference(self):
        """Test that parse_and_create_children without graph doesn't crash."""
        md = MarkdownNode(content="- Item 1")
        # Should not crash, just return self
        result = md.parse_and_create_children()
        assert result is md
        assert len(md.child_order) == 0

    def test_markdown_list_item_child_expansion_states(self):
        """Test MarkdownListItemNode children can have independent expansion states."""
        graph = ContextGraph()
        md = MarkdownNode(content="- First item\n- Second item\n- Third item")
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(cid) for cid in md.child_order]
        assert len(children) == 3

        # Set different expansion states on list items
        children[0].default_expansion = Expansion.HEADER
        children[1].default_expansion = Expansion.CONTENT
        children[2].default_expansion = Expansion.ALL

        # Verify states are independent
        assert children[0].default_expansion == Expansion.HEADER
        assert children[1].default_expansion == Expansion.CONTENT
        assert children[2].default_expansion == Expansion.ALL

        # Verify each renders appropriately
        render0 = NodeView(children[0]).render()
        render1 = NodeView(children[1]).render()
        render2 = NodeView(children[2]).render()

        # HEADER should only show metadata, not content
        assert "markdown_list_item" in render0
        assert "First item" not in render0

        # CONTENT and ALL should show the item content
        assert "Second item" in render1
        assert "Third item" in render2

    def test_nested_list_item_traversal(self):
        """Test that nested list items form correct parent-child relationships."""
        graph = ContextGraph()
        md = MarkdownNode(
            content=("- Parent 1\n  - Child 1A\n  - Child 1B\n- Parent 2\n  - Child 2A\n")
        )
        graph.add_node(md)
        md.parse_and_create_children()

        # MarkdownNode should have 2 direct children (Parent 1, Parent 2)
        assert len(md.child_order) == 2
        parents = [graph.get_node(cid) for cid in md.child_order]

        # First parent
        assert parents[0].content == "Parent 1"
        assert len(parents[0].child_order) == 2
        p1_children = [graph.get_node(cid) for cid in parents[0].child_order]
        assert p1_children[0].content == "Child 1A"
        assert p1_children[1].content == "Child 1B"

        # Second parent
        assert parents[1].content == "Parent 2"
        assert len(parents[1].child_order) == 1
        p2_children = [graph.get_node(cid) for cid in parents[1].child_order]
        assert p2_children[0].content == "Child 2A"

    def test_list_item_with_complex_content(self):
        """Test list items with special characters and formatting."""
        graph = ContextGraph()
        md = MarkdownNode(
            content=("- Item with **bold** text\n- Item with `code`\n- Item with [link](url)\n")
        )
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(cid) for cid in md.child_order]
        assert len(children) == 3

        # Content should preserve markdown formatting
        assert "**bold**" in children[0].content
        assert "`code`" in children[1].content
        assert "[link](url)" in children[2].content

    def test_mixed_ordered_unordered_lists(self):
        """Test parsing mixed ordered and unordered lists."""
        graph = ContextGraph()
        md = MarkdownNode(content=("- Unordered 1\n1. Ordered 1\n2. Ordered 2\n- Unordered 2\n"))
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(cid) for cid in md.child_order]
        assert len(children) == 4

        assert not children[0].is_ordered
        assert children[1].is_ordered
        assert children[2].is_ordered
        assert not children[3].is_ordered

    def test_list_item_indent_levels(self):
        """Test that indent levels are correctly set on list items."""
        graph = ContextGraph()
        md = MarkdownNode(content=("- Level 0\n  - Level 1\n    - Level 2\n      - Level 3\n"))
        graph.add_node(md)
        md.parse_and_create_children()

        # Get all nodes in depth-first order
        level0 = graph.get_node(list(md.child_order)[0])
        assert level0.indent_level == 0

        level1 = graph.get_node(list(level0.child_order)[0])
        assert level1.indent_level == 1

        level2 = graph.get_node(list(level1.child_order)[0])
        assert level2.indent_level == 2

        level3 = graph.get_node(list(level2.child_order)[0])
        assert level3.indent_level == 3

    def test_list_item_markers_preserved(self):
        """Test that list item markers are correctly preserved."""
        graph = ContextGraph()
        md = MarkdownNode(content=("* Asterisk\n- Dash\n+ Plus\n1. Numbered\n"))
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(cid) for cid in md.child_order]
        assert children[0].marker == "*"
        assert children[1].marker == "-"
        assert children[2].marker == "+"
        assert children[3].marker == "1."

    def test_empty_list_items(self):
        """Test parsing list with empty items."""
        graph = ContextGraph()
        md = MarkdownNode(content=("- Item 1\n- \n- Item 3\n"))
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(cid) for cid in md.child_order]
        assert len(children) == 3
        assert children[0].content == "Item 1"
        assert children[1].content == ""
        assert children[2].content == "Item 3"

    def test_list_items_with_multiple_paragraphs(self):
        """Test list items that contain multiple lines of content."""
        graph = ContextGraph()
        # This tests basic multi-line item content
        md = MarkdownNode(content=("- First item first line\n  continuation line\n- Second item\n"))
        graph.add_node(md)
        md.parse_and_create_children()

        children = [graph.get_node(cid) for cid in md.child_order]
        # Parser treats continuation as nested item due to indentation
        # This documents current behavior
        assert len(children) >= 2
