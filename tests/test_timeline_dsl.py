"""Tests for Timeline DSL functions.

These tests cover the DSL functions exposed by Timeline that aren't covered
by other test modules.
"""

import asyncio
from pathlib import Path

import pytest

from activecontext.session.timeline import Timeline
from activecontext.context.graph import ContextGraph


class TestLsHandles:
    """Test ls() function for listing context object handles."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_ls_returns_context_objects(self, temp_cwd: Path) -> None:
        """Test that ls() returns context object digests."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Initially empty
            result = await timeline.execute_statement("handles = ls()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            handles = ns["handles"]
            assert isinstance(handles, list)
            assert len(handles) == 0

            # Create some objects
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement('t = topic("Test Topic")')

            # Now ls() should return them
            result = await timeline.execute_statement("handles = ls()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            handles = ns["handles"]
            assert len(handles) == 2

            # Check structure (GetDigest format)
            types = [h["type"] for h in handles]
            assert "text" in types
            assert "topic" in types
        finally:
            await timeline.close()


class TestSetTitle:
    """Test set_title() function."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_set_title_with_callback(self, temp_cwd: Path) -> None:
        """Test that set_title() calls the registered callback."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            captured_title = None

            def title_callback(title: str) -> None:
                nonlocal captured_title
                captured_title = title

            # Register the callback
            timeline.set_title_callback(title_callback)

            result = await timeline.execute_statement('set_title("My Test Session")')
            assert result.status.value == "ok"

            assert captured_title == "My Test Session"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_set_title_without_callback(self, temp_cwd: Path) -> None:
        """Test that set_title() warns when no callback is registered."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # No callback registered, should still succeed but print warning
            result = await timeline.execute_statement('set_title("No Callback")')
            assert result.status.value == "ok"
            # Warning is printed to stdout, which is captured in result
        finally:
            await timeline.close()


class TestMarkdownFunction:
    """Test markdown() function for creating TextNode tree from markdown."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_create_markdown_tree_from_file(self, temp_cwd: Path) -> None:
        """Test creating a markdown tree from a file."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a markdown file
            md_file = temp_cwd / "README.md"
            md_content = """# Project Overtext

## Features

- Feature 1
- Feature 2

## Installation

Run pip install"""
            md_file.write_text(md_content)

            result = await timeline.execute_statement('m = markdown("README.md")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert "m" in ns

            # markdown() now returns a TextNode (root of tree)
            digest = ns["m"].GetDigest()
            assert digest["type"] == "text"
            assert digest["media_type"] == "markdown"

            # Root should have children (the ## sections)
            root = ns["m"]
            assert len(root.children_ids) == 2  # Features and Installation
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_create_markdown_tree_with_content(self, temp_cwd: Path) -> None:
        """Test creating a markdown tree with inline content."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            md_content = "# Hello\\n\\nWorld"

            result = await timeline.execute_statement(
                f'm = markdown("inline.md", content={md_content!r})'
            )
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert "m" in ns

            # Should be a TextNode
            digest = ns["m"].GetDigest()
            assert digest["type"] == "text"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_markdown_nodes_render_with_node_id(self, temp_cwd: Path) -> None:
        """Test that markdown nodes render headings with display-friendly node IDs.

        Node IDs have format {node_type}_{sequence} (e.g., text_1, text_2, etc.).
        """
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a markdown file matching the golden fixture
            md_file = temp_cwd / "test.md"
            md_content = """# Main Title

Introduction text.

## Section One

Content for section one.

## Section Two

Content for section two.
"""
            md_file.write_text(md_content)

            result = await timeline.execute_statement('m = markdown("test.md")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            root = ns["m"]

            # Get all nodes in order (root + children sorted by display_sequence)
            all_nodes = [root]
            children = []
            context_objects = timeline.get_context_objects()
            for child_id in root.children_ids:
                child = context_objects.get(child_id)
                if child:
                    children.append(child)
            # Sort children by display_sequence for deterministic order
            children.sort(key=lambda n: n.display_sequence or 0)
            all_nodes.extend(children)

            # Verify node_id format is text_N
            for node in all_nodes:
                assert node.node_id.startswith("text_")
                num = int(node.node_id.split("_")[1])
                assert num >= 1

            # Render all nodes and concatenate
            rendered_parts = []
            for node in all_nodes:
                rendered = node.Render(
                    cwd=str(temp_cwd),
                    text_buffers=timeline._text_buffers,
                )
                rendered_parts.append(rendered.rstrip())

            full_rendered = "\n\n".join(rendered_parts) + "\n"

            # Compare to golden file
            golden_path = Path(__file__).parent / "fixtures" / "example_node_golden.md"
            golden = golden_path.read_text(encoding="utf-8")

            assert full_rendered == golden
        finally:
            await timeline.close()


class TestLsPermissions:
    """Test ls_permissions() and related permission inspection functions."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_ls_permissions_without_manager(self, temp_cwd: Path) -> None:
        """Test ls_permissions() when no permission manager is set."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("perms = ls_permissions()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            perms = ns["perms"]

            # Without a manager, should return empty list or minimal info
            assert isinstance(perms, list)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_ls_imports(self, temp_cwd: Path) -> None:
        """Test ls_imports() lists import configuration."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("imports = ls_imports()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            imports = ns["imports"]

            # Returns dict with allow_all/allow_submodules/allowed_modules keys
            assert isinstance(imports, dict)
            # Check for expected keys from ImportGuard
            assert "allow_all" in imports or "allowed_modules" in imports
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_ls_shell_permissions(self, temp_cwd: Path) -> None:
        """Test ls_shell_permissions() lists shell permission configuration."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("shell_perms = ls_shell_permissions()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            shell_perms = ns["shell_perms"]

            # Returns dict with deny_by_default/rules keys
            assert isinstance(shell_perms, dict)
            assert "deny_by_default" in shell_perms or "rules" in shell_perms
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_ls_website_permissions(self, temp_cwd: Path) -> None:
        """Test ls_website_permissions() lists website permission configuration."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("web_perms = ls_website_permissions()")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            web_perms = ns["web_perms"]

            # Returns dict with deny_by_default/allow_localhost/rules keys
            assert isinstance(web_perms, dict)
            assert "deny_by_default" in web_perms
        finally:
            await timeline.close()


class TestWaitConditions:
    """Test wait condition DSL functions."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_wait_all_sets_condition(self, temp_cwd: Path) -> None:
        """Test that wait_all() sets up a wait condition for multiple nodes."""
        import sys

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create two shell commands
            if sys.platform == "win32":
                cmd1 = 'shell("cmd", ["/c", "echo", "1"])'
                cmd2 = 'shell("cmd", ["/c", "echo", "2"])'
            else:
                cmd1 = 'shell("echo", ["1"])'
                cmd2 = 'shell("echo", ["2"])'

            await timeline.execute_statement(f"s1 = {cmd1}")
            await timeline.execute_statement(f"s2 = {cmd2}")

            # Setup wait_all condition
            result = await timeline.execute_statement("wait_all(s1, s2)")
            assert result.status.value == "ok"

            # Check that wait condition was set
            assert timeline._wait_condition is not None
            assert len(timeline._wait_condition.node_ids) == 2
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_wait_any_sets_condition(self, temp_cwd: Path) -> None:
        """Test that wait_any() sets up a wait condition for first completion."""
        import sys

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create two shell commands
            if sys.platform == "win32":
                cmd1 = 'shell("cmd", ["/c", "echo", "1"])'
                cmd2 = 'shell("cmd", ["/c", "echo", "2"])'
            else:
                cmd1 = 'shell("echo", ["1"])'
                cmd2 = 'shell("echo", ["2"])'

            await timeline.execute_statement(f"s1 = {cmd1}")
            await timeline.execute_statement(f"s2 = {cmd2}")

            # Setup wait_any condition
            result = await timeline.execute_statement("wait_any(s1, s2)")
            assert result.status.value == "ok"

            # Check that wait condition was set with ANY mode
            assert timeline._wait_condition is not None
            from activecontext.session.timeline import WaitMode

            assert timeline._wait_condition.mode == WaitMode.ANY
        finally:
            await timeline.close()


class TestShowFunction:
    """Test show() function for texting object details."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_show_text_node(self, temp_cwd: Path) -> None:
        """Test show() returns rendered string about a text node."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')

            result = await timeline.execute_statement("info = show(v)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            info = ns["info"]

            # show() returns a string representation
            assert isinstance(info, str)
            assert "text" in info or "test.py" in info
        finally:
            await timeline.close()


class TestReplayFrom:
    """Test replay_from() statement replay functionality."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_replay_from_index(self, temp_cwd: Path) -> None:
        """Test replaying statements from a given index."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Execute several statements
            await timeline.execute_statement("x = 1")
            await timeline.execute_statement("y = 2")
            await timeline.execute_statement("z = x + y")

            # Get current timeline state
            original_statements = len(timeline.get_statements())
            assert original_statements == 3

            # replay_from is an async generator, consume all results
            async for result in timeline.replay_from(1):
                pass  # Just consume the results

            # Should still have same number of statements
            # but namespace reflects replayed state
            ns = timeline.get_namespace()
            # z should still equal 3 since we replayed y=2 and z=x+y
            assert ns["z"] == 3
        finally:
            await timeline.close()


class TestNamespaceSnapshot:
    """Test namespace snapshot functionality."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_get_namespace_returns_copy(self, temp_cwd: Path) -> None:
        """Test that get_namespace returns a snapshot copy."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement("x = 1")

            ns1 = timeline.get_namespace()
            ns1["x"] = 999  # Modify the copy

            ns2 = timeline.get_namespace()
            # Original should be unchanged
            assert ns2["x"] == 1
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_namespace_excludes_dsl_functions(self, temp_cwd: Path) -> None:
        """Test that get_namespace excludes DSL functions but includes user vars."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Initially namespace should be empty (DSL funcs are excluded)
            ns = timeline.get_namespace()
            assert len(ns) == 0

            # Add user variables
            await timeline.execute_statement("my_var = 42")
            await timeline.execute_statement('v = text("test.py")')

            ns = timeline.get_namespace()
            # User variables should be present
            assert "my_var" in ns
            assert ns["my_var"] == 42
            # Context nodes should be present
            assert "v" in ns
            # DSL functions should NOT be in the snapshot
            assert "text" not in ns
            assert "group" not in ns
        finally:
            await timeline.close()


class TestHideUnhide:
    """Test hide() and unhide() functions for traversal control."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_hide_single_node(self, temp_cwd: Path) -> None:
        """Test hide() sets node expansion to HIDDEN."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')

            ns = timeline.get_namespace()
            v = ns["v"]
            assert v.expand == Expansion.ALL  # Default
            assert v.hide is False

            result = await timeline.execute_statement("count = hide(v)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 1
            assert v.hide is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_hide_multiple_nodes(self, temp_cwd: Path) -> None:
        """Test hide() with multiple nodes."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v1 = text("test1.py")')
            await timeline.execute_statement('v2 = text("test2.py")')

            result = await timeline.execute_statement("count = hide(v1, v2)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 2
            assert ns["v1"].hide is True
            assert ns["v2"].hide is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_hide_by_id(self, temp_cwd: Path) -> None:
        """Test hide() with node ID string."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')

            ns = timeline.get_namespace()
            node_id = ns["v"].node_id

            result = await timeline.execute_statement(f'count = hide("{node_id}")')
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 1
            assert ns["v"].hide is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_hide_stores_previous_expansion(self, temp_cwd: Path) -> None:
        """Test that hide() stores previous expansion for restoration."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("v.expand = Expansion.CONTENT")

            ns = timeline.get_namespace()
            v = ns["v"]
            assert v.expand == Expansion.CONTENT

            await timeline.execute_statement("hide(v)")

            assert v.hide is True
            assert v.tags.get("_hidden_expand") == "content"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_hide_skips_already_hidden(self, temp_cwd: Path) -> None:
        """Test that hide() skips nodes already hidden."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("hide(v)")  # Hide first

            result = await timeline.execute_statement("count = hide(v)")  # Try again
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 0  # No change (already hidden)
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unhide_restores_previous_expansion(self, temp_cwd: Path) -> None:
        """Test that unhide() restores the expansion from before hide()."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("v.expand = Expansion.CONTENT")

            ns = timeline.get_namespace()
            v = ns["v"]

            await timeline.execute_statement("hide(v)")
            assert v.hide is True

            result = await timeline.execute_statement("count = unhide(v)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 1
            assert v.hide is False
            assert v.expand == Expansion.CONTENT  # Restored
            assert "_hidden_expand" not in v.tags  # Cleaned up
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unhide_with_explicit_expansion(self, temp_cwd: Path) -> None:
        """Test unhide() with explicit expand parameter."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("hide(v)")

            ns = timeline.get_namespace()
            v = ns["v"]
            assert v.hide is True

            result = await timeline.execute_statement("count = unhide(v, expand=Expansion.HEADER)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 1
            assert v.hide is False
            assert v.expand == Expansion.HEADER
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unhide_defaults_to_details(self, temp_cwd: Path) -> None:
        """Test unhide() defaults to DETAILS if no stored expansion."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            # Manually set hide without going through hide()
            await timeline.execute_statement("v.hide = True")

            ns = timeline.get_namespace()
            v = ns["v"]

            result = await timeline.execute_statement("count = unhide(v)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 1
            assert v.hide is False
            assert v.expand == Expansion.ALL  # Default
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_unhide_multiple_nodes(self, temp_cwd: Path) -> None:
        """Test unhide() with multiple nodes."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v1 = text("test1.py")')
            await timeline.execute_statement('v2 = text("test2.py")')
            await timeline.execute_statement("hide(v1, v2)")

            ns = timeline.get_namespace()
            assert ns["v1"].hide is True
            assert ns["v2"].hide is True

            result = await timeline.execute_statement("count = unhide(v1, v2)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            assert ns["count"] == 2
            assert ns["v1"].hide is False
            assert ns["v2"].hide is False
            assert ns["v1"].expand == Expansion.ALL
            assert ns["v2"].expand == Expansion.ALL
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_hide_unhide_roundtrip(self, temp_cwd: Path) -> None:
        """Test full hide/unhide roundtrip preserves expansion."""
        from activecontext.context.state import Expansion

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            await timeline.execute_statement('v = text("test.py")')
            await timeline.execute_statement("v.expand = Expansion.HEADER")

            ns = timeline.get_namespace()
            v = ns["v"]

            # Hide and verify
            await timeline.execute_statement("hide(v)")
            assert v.hide is True

            # Unhide and verify restored
            await timeline.execute_statement("unhide(v)")
            assert v.hide is False
            assert v.expand == Expansion.HEADER
        finally:
            await timeline.close()



class TestMultiLineExpressionResult:
    """Test that multi-line code blocks return the final expression result.

    When a code block contains multiple statements with a final expression,
    the REPL should print the value of that expression (like Python's REPL).
    """

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_final_expression_printed(self, temp_cwd: Path) -> None:
        """Test that final expression in multi-line block is captured in stdout."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Multi-line block: assignment followed by expression
            result = await timeline.execute_statement("x = 42\nx")
            assert result.status.value == "ok"
            assert "42" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_assignment_only_no_output(self, temp_cwd: Path) -> None:
        """Test that assignment-only blocks produce no stdout."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("x = 42")
            assert result.status.value == "ok"
            # Assignment should not produce output
            assert result.stdout == ""
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_multiple_statements_with_final_expression(self, temp_cwd: Path) -> None:
        """Test multiple statements followed by expression."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            code = """a = 10
b = 20
c = a + b
c"""
            result = await timeline.execute_statement(code)
            assert result.status.value == "ok"
            assert "30" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_single_expression_still_works(self, temp_cwd: Path) -> None:
        """Test that single expressions still work."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("2 + 2")
            assert result.status.value == "ok"
            assert "4" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_final_statement_is_not_expression(self, temp_cwd: Path) -> None:
        """Test that if final statement is not expression, no output produced."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            code = """x = 1
y = x + 1"""
            result = await timeline.execute_statement(code)
            assert result.status.value == "ok"
            assert result.stdout == ""
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_ls_with_handles_assignment(self, temp_cwd: Path) -> None:
        """Test the original bug case: handles = ls(); handles."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create a context object first
            await timeline.execute_statement('v = text("test.py")')

            # Now the bug case
            code = """handles = ls()
handles"""
            result = await timeline.execute_statement(code)
            assert result.status.value == "ok"
            # Should output the handles list
            assert "text" in result.stdout or "[" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_multiline_with_print_and_expression(self, temp_cwd: Path) -> None:
        """Test that explicit print AND final expression both appear."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            code = """print("hello")
42"""
            result = await timeline.execute_statement(code)
            assert result.status.value == "ok"
            assert "hello" in result.stdout
            assert "42" in result.stdout
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_exception_in_statements(self, temp_cwd: Path) -> None:
        """Test that exception in statements is properly reported."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            code = """x = 1 / 0
x"""
            result = await timeline.execute_statement(code)
            assert result.status.value == "error"
            assert result.exception is not None
            assert result.exception["type"] == "ZeroDivisionError"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_exception_in_final_expression(self, temp_cwd: Path) -> None:
        """Test that exception in final expression is properly reported."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            code = """x = 10
undefined_var"""
            result = await timeline.execute_statement(code)
            assert result.status.value == "error"
            assert result.exception is not None
            assert result.exception["type"] == "NameError"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_async_multiline_falls_back(self, temp_cwd: Path) -> None:
        """Test that async multi-line blocks execute correctly (but don't capture result)."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            code = """async def get_val():
    return 42
x = await get_val()
x"""
            result = await timeline.execute_statement(code)
            # Should execute successfully even though result isn't captured
            assert result.status.value == "ok"
            # Verify x was set correctly
            ns = timeline.get_namespace()
            assert ns["x"] == 42
        finally:
            await timeline.close()


class TestTopLevelAwait:
    """Test that top-level 'await' works in DSL code.

    The DSL uses PyCF_ALLOW_TOP_LEVEL_AWAIT to support await expressions
    and statements directly, without requiring users to define async functions.
    """

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_await_expression_works(self, temp_cwd: Path) -> None:
        """Test that 'await' in expression context works."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Add an async function to the namespace
            async def async_func():
                return 42

            timeline._namespace["async_func"] = async_func

            result = await timeline.execute_statement("x = await async_func()")

            assert result.status.value == "ok"
            assert timeline._namespace.get("x") == 42
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_with_asyncio_sleep(self, temp_cwd: Path) -> None:
        """Test that 'await asyncio.sleep()' works."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Import asyncio first (or add to namespace)
            timeline._namespace["asyncio"] = asyncio

            result = await timeline.execute_statement(
                "result = await asyncio.sleep(0.001, result='done')"
            )

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == "done"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_multiline_with_await(self, temp_cwd: Path) -> None:
        """Test that multiline code with 'await' works."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:

            async def get_value():
                return "hello"

            timeline._namespace["get_value"] = get_value

            source = """x = await get_value()
y = x.upper()"""

            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("x") == "hello"
            assert timeline._namespace.get("y") == "HELLO"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_undefined_function_gives_name_error(self, temp_cwd: Path) -> None:
        """Test that awaiting undefined function gives clear NameError."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("x = await undefined_func()")

            assert result.status.value == "error"
            assert result.exception is not None
            assert result.exception["type"] == "NameError"
            assert "undefined_func" in result.exception["message"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_inside_function_definition(self, temp_cwd: Path) -> None:
        """Test that 'await' inside async function definition works."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Add asyncio to namespace
            timeline._namespace["asyncio"] = asyncio

            # Define an async function that uses await internally
            source = """
async def my_async_func():
    return await asyncio.sleep(0.001, result='inner')
"""
            result = await timeline.execute_statement(source)
            assert result.status.value == "ok"

            # Now call it with top-level await
            result = await timeline.execute_statement("result = await my_async_func()")
            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == "inner"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_partial_execution_on_await_failure(self, temp_cwd: Path) -> None:
        """Test that side effects before a failing await persist."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def failing_func():
                raise ValueError("intentional failure")

            timeline._namespace["failing_func"] = failing_func

            source = """
x = 'before_await'
y = await failing_func()
z = 'after_await'
"""
            result = await timeline.execute_statement(source)

            # Should fail
            assert result.status.value == "error"
            assert "intentional failure" in result.exception["message"]

            # x should be set (ran before await)
            assert timeline._namespace.get("x") == "before_await"
            # y and z should not be set
            assert timeline._namespace.get("y") is None
            assert timeline._namespace.get("z") is None
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_nested_await(self, temp_cwd: Path) -> None:
        """Test nested await: await (await func())."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def outer():
                async def inner():
                    return 42
                return inner()  # Returns coroutine, not result

            timeline._namespace["outer"] = outer

            # This requires two awaits: one for outer, one for the returned coroutine
            result = await timeline.execute_statement("result = await (await outer())")

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == 42
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_async_for_loop(self, temp_cwd: Path) -> None:
        """Test async for loop at top level."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def async_gen():
                for i in range(5):
                    yield i * 2

            timeline._namespace["async_gen"] = async_gen

            source = """
result = []
async for x in async_gen():
    result.append(x)
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == [0, 2, 4, 6, 8]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_async_with_context_manager(self, temp_cwd: Path) -> None:
        """Test async with at top level."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            class AsyncCM:
                def __init__(self):
                    self.entered = False
                    self.exited = False

                async def __aenter__(self):
                    self.entered = True
                    return "context_value"

                async def __aexit__(self, *args):
                    self.exited = True

            cm_instance = AsyncCM()
            timeline._namespace["cm"] = cm_instance

            source = """
async with cm as val:
    result = val
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == "context_value"
            assert cm_instance.entered is True
            assert cm_instance.exited is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_async_with_exception_in_body(self, temp_cwd: Path) -> None:
        """Test that async with properly exits on exception in body."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            exit_called = {"value": False}

            class AsyncCM:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    exit_called["value"] = True
                    return False  # Don't suppress exception

            timeline._namespace["AsyncCM"] = AsyncCM
            timeline._namespace["exit_called"] = exit_called

            source = """
async with AsyncCM():
    raise ValueError("body error")
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "error"
            assert "body error" in result.exception["message"]
            # __aexit__ should still be called
            assert exit_called["value"] is True
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_non_coroutine_gives_error(self, temp_cwd: Path) -> None:
        """Test that awaiting a non-coroutine gives clear TypeError."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            def sync_func():
                return 42

            timeline._namespace["sync_func"] = sync_func

            result = await timeline.execute_statement("result = await sync_func()")

            assert result.status.value == "error"
            assert result.exception["type"] == "TypeError"
            # Should mention can't await or not awaitable
            assert "await" in result.exception["message"].lower() or "awaitable" in result.exception["message"].lower()
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_multiple_awaits_middle_fails(self, temp_cwd: Path) -> None:
        """Test multiple awaits where the middle one fails."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            call_count = {"value": 0}

            async def counting_func():
                call_count["value"] += 1
                if call_count["value"] == 2:
                    raise ValueError("second call failed")
                return call_count["value"]

            timeline._namespace["counting_func"] = counting_func
            timeline._namespace["call_count"] = call_count

            source = """
a = await counting_func()
b = await counting_func()
c = await counting_func()
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "error"
            assert "second call failed" in result.exception["message"]

            # First call succeeded
            assert timeline._namespace.get("a") == 1
            # Second and third didn't complete
            assert timeline._namespace.get("b") is None
            assert timeline._namespace.get("c") is None
            # Only 2 calls were made
            assert call_count["value"] == 2
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_with_sync_code_interleaved(self, temp_cwd: Path) -> None:
        """Test mixing sync code with awaits."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            timeline._namespace["asyncio"] = asyncio

            source = """
a = 1
b = await asyncio.sleep(0.001, result='async1')
c = a + 1
d = await asyncio.sleep(0.001, result='async2')
e = c + 1
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("a") == 1
            assert timeline._namespace.get("b") == "async1"
            assert timeline._namespace.get("c") == 2
            assert timeline._namespace.get("d") == "async2"
            assert timeline._namespace.get("e") == 3
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_in_list_comprehension(self, temp_cwd: Path) -> None:
        """Test await in async list comprehension."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def async_double(x):
                return x * 2

            timeline._namespace["async_double"] = async_double

            # Async comprehension
            source = "result = [await async_double(i) for i in range(5)]"
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == [0, 2, 4, 6, 8]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_concurrent_statements_dont_interfere(self, temp_cwd: Path) -> None:
        """Test that concurrent statement execution doesn't corrupt namespace."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            timeline._namespace["asyncio"] = asyncio

            # Execute two statements concurrently
            task1 = asyncio.create_task(
                timeline.execute_statement("x = await asyncio.sleep(0.01, result='first')")
            )
            task2 = asyncio.create_task(
                timeline.execute_statement("y = await asyncio.sleep(0.01, result='second')")
            )

            result1, result2 = await asyncio.gather(task1, task2)

            # Both should succeed
            assert result1.status.value == "ok"
            assert result2.status.value == "ok"

            # Both values should be set correctly
            assert timeline._namespace.get("x") == "first"
            assert timeline._namespace.get("y") == "second"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_none_gives_error(self, temp_cwd: Path) -> None:
        """Test that await None gives clear error."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            result = await timeline.execute_statement("result = await None")

            assert result.status.value == "error"
            assert result.exception["type"] == "TypeError"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_async_generator_expression(self, temp_cwd: Path) -> None:
        """Test async generator expression at top level."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def async_range(n):
                for i in range(n):
                    yield i

            timeline._namespace["async_range"] = async_range

            source = """
result = []
async for x in (i * 2 async for i in async_range(5)):
    result.append(x)
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == [0, 2, 4, 6, 8]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_in_try_except_finally(self, temp_cwd: Path) -> None:
        """Test await in try/except/finally blocks."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def might_fail(should_fail: bool):
                if should_fail:
                    raise ValueError("failed")
                return "success"

            timeline._namespace["might_fail"] = might_fail

            source = """
results = []
try:
    results.append(await might_fail(False))
    results.append(await might_fail(True))
except ValueError as e:
    results.append(f'caught: {e}')
finally:
    results.append('finally')
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            results = timeline._namespace.get("results")
            assert results == ["success", "caught: failed", "finally"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_with_walrus_operator(self, temp_cwd: Path) -> None:
        """Test await with walrus operator (:=)."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def get_value():
                return 42

            timeline._namespace["get_value"] = get_value

            source = """
if (x := await get_value()) > 40:
    result = f'got {x}'
else:
    result = 'too small'
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("x") == 42
            assert timeline._namespace.get("result") == "got 42"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_in_conditional_expression(self, temp_cwd: Path) -> None:
        """Test await in ternary conditional expression."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def async_true():
                return "yes"

            async def async_false():
                return "no"

            timeline._namespace["async_true"] = async_true
            timeline._namespace["async_false"] = async_false

            source = "result = await async_true() if True else await async_false()"
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == "yes"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_multiple_async_context_managers(self, temp_cwd: Path) -> None:
        """Test multiple async context managers in single with statement."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            order = []

            class AsyncCM:
                def __init__(self, name):
                    self.name = name

                async def __aenter__(self):
                    order.append(f"enter_{self.name}")
                    return self.name

                async def __aexit__(self, *args):
                    order.append(f"exit_{self.name}")

            timeline._namespace["AsyncCM"] = AsyncCM
            timeline._namespace["order"] = order

            source = """
async with AsyncCM('a') as a, AsyncCM('b') as b:
    order.append(f'body_{a}_{b}')
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert order == ["enter_a", "enter_b", "body_a_b", "exit_b", "exit_a"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_recursive_async_calls(self, temp_cwd: Path) -> None:
        """Test recursive async function calls."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            source = """
async def factorial(n):
    if n <= 1:
        return 1
    return n * await factorial(n - 1)
"""
            result = await timeline.execute_statement(source)
            assert result.status.value == "ok"

            result = await timeline.execute_statement("result = await factorial(5)")
            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == 120
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_exception_in_async_generator(self, temp_cwd: Path) -> None:
        """Test exception raised during async iteration."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def bad_gen():
                yield 1
                yield 2
                raise ValueError("generator failed")
                yield 3  # Never reached

            timeline._namespace["bad_gen"] = bad_gen

            source = """
result = []
try:
    async for x in bad_gen():
        result.append(x)
except ValueError as e:
    result.append(f'error: {e}')
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == [1, 2, "error: generator failed"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_chained_coroutines(self, temp_cwd: Path) -> None:
        """Test awaiting a chain of coroutines."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def step1():
                return "a"

            async def step2(prev):
                return prev + "b"

            async def step3(prev):
                return prev + "c"

            timeline._namespace["step1"] = step1
            timeline._namespace["step2"] = step2
            timeline._namespace["step3"] = step3

            source = "result = await step3(await step2(await step1()))"
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == "abc"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_in_dict_comprehension(self, temp_cwd: Path) -> None:
        """Test await in async dict comprehension."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def async_key(i):
                return f"key_{i}"

            async def async_val(i):
                return i * 10

            timeline._namespace["async_key"] = async_key
            timeline._namespace["async_val"] = async_val

            source = "result = {await async_key(i): await async_val(i) for i in range(3)}"
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("result") == {
                "key_0": 0,
                "key_1": 10,
                "key_2": 20,
            }
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_preserves_exception_traceback(self, temp_cwd: Path) -> None:
        """Test that exceptions preserve their traceback through await."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def inner():
                raise ValueError("inner error")

            async def outer():
                return await inner()

            timeline._namespace["outer"] = outer

            result = await timeline.execute_statement("await outer()")

            assert result.status.value == "error"
            assert "inner error" in result.exception["message"]
            # Traceback should mention both functions
            assert "inner" in result.exception["traceback"]
            assert "outer" in result.exception["traceback"]
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_with_default_argument(self, temp_cwd: Path) -> None:
        """Test async function with default arguments."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            source = """
async def greet(name, greeting='Hello'):
    return f'{greeting}, {name}!'
"""
            result = await timeline.execute_statement(source)
            assert result.status.value == "ok"

            result = await timeline.execute_statement("r1 = await greet('World')")
            assert result.status.value == "ok"
            assert timeline._namespace.get("r1") == "Hello, World!"

            result = await timeline.execute_statement("r2 = await greet('World', 'Hi')")
            assert result.status.value == "ok"
            assert timeline._namespace.get("r2") == "Hi, World!"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_await_modifying_shared_state(self, temp_cwd: Path) -> None:
        """Test multiple awaits modifying shared mutable state."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            shared = {"counter": 0}

            async def increment():
                shared["counter"] += 1
                return shared["counter"]

            timeline._namespace["increment"] = increment
            timeline._namespace["shared"] = shared

            source = """
a = await increment()
b = await increment()
c = await increment()
"""
            result = await timeline.execute_statement(source)

            assert result.status.value == "ok"
            assert timeline._namespace.get("a") == 1
            assert timeline._namespace.get("b") == 2
            assert timeline._namespace.get("c") == 3
            assert shared["counter"] == 3
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_returned_coroutine_is_auto_awaited(self, temp_cwd: Path) -> None:
        """Test that coroutines returned and stored in namespace are auto-awaited.

        The Timeline's _await_namespace_coroutines() automatically awaits any
        coroutine objects stored in the namespace. This is the feature that allows
        `x = mcp_connect("server")` to work without explicit await.
        """
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            async def inner():
                return "final"

            async def returns_coro():
                return inner()  # Returns coroutine, doesn't await

            timeline._namespace["returns_coro"] = returns_coro

            # await returns_coro() returns a coroutine, but Timeline auto-awaits it
            result = await timeline.execute_statement("result = await returns_coro()")
            assert result.status.value == "ok"

            # The coroutine was auto-awaited, so we get the final value directly
            assert timeline._namespace.get("result") == "final"
        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_cancellation_cleanup(self, temp_cwd: Path) -> None:
        """Test that cancellation properly cleans up."""
        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            cleanup_called = {"value": False}

            async def slow_with_cleanup():
                try:
                    await asyncio.sleep(10)  # Long sleep
                except asyncio.CancelledError:
                    cleanup_called["value"] = True
                    raise

            timeline._namespace["slow_with_cleanup"] = slow_with_cleanup
            timeline._namespace["cleanup_called"] = cleanup_called
            timeline._namespace["asyncio"] = asyncio

            # Start the statement execution
            task = asyncio.create_task(
                timeline.execute_statement("await slow_with_cleanup()")
            )

            # Give it a moment to start
            await asyncio.sleep(0.01)

            # Cancel it
            task.cancel()

            # Wait for cancellation to complete
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Cleanup should have been called
            assert cleanup_called["value"] is True
        finally:
            await timeline.close()


class TestChoiceDSL:
    """Test choice() DSL function for dropdown-like selection."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_choice_creates_choice_view(self, temp_cwd: Path) -> None:
        """Test that choice() creates a ChoiceView wrapping a GroupNode."""
        from activecontext.context.view import ChoiceView

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create child nodes using markdown with content parameter
            await timeline.execute_statement(
                'v1 = markdown("option_a.md", content="# Option A", tokens=100)'
            )
            await timeline.execute_statement(
                'v2 = markdown("option_b.md", content="# Option B", tokens=100)'
            )
            await timeline.execute_statement(
                'v3 = markdown("option_c.md", content="# Option C", tokens=100)'
            )

            # Create choice view
            result = await timeline.execute_statement("c = choice(v1, v2, v3)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            c = ns["c"]

            # Verify it's a ChoiceView
            assert isinstance(c, ChoiceView)

            # Verify the first child is selected by default
            assert c.selected_id == ns["v1"].node_id

        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_choice_with_explicit_selection(self, temp_cwd: Path) -> None:
        """Test that choice() respects explicit selected parameter."""
        from activecontext.context.view import ChoiceView

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create child nodes
            await timeline.execute_statement(
                'v1 = markdown("opt_a.md", content="# Option A", tokens=100)'
            )
            await timeline.execute_statement(
                'v2 = markdown("opt_b.md", content="# Option B", tokens=100)'
            )

            # Create choice with explicit selection
            result = await timeline.execute_statement(
                "c = choice(v1, v2, selected=v2.node_id)"
            )
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            c = ns["c"]

            # Verify v2 is selected
            assert isinstance(c, ChoiceView)
            assert c.selected_id == ns["v2"].node_id

        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_choice_select_method(self, temp_cwd: Path) -> None:
        """Test that ChoiceView.select() switches selection."""
        from activecontext.context.view import ChoiceView

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))

        try:
            # Create child nodes
            await timeline.execute_statement(
                'v1 = markdown("a.md", content="# Option A", tokens=100)'
            )
            await timeline.execute_statement(
                'v2 = markdown("b.md", content="# Option B", tokens=100)'
            )

            # Create choice
            result = await timeline.execute_statement("c = choice(v1, v2)")
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            c = ns["c"]
            v1_id = ns["v1"].node_id
            v2_id = ns["v2"].node_id

            # Initial selection is v1
            assert c.selected_id == v1_id

            # Switch to v2
            c.select(v2_id)
            assert c.selected_id == v2_id

            # Switch back to v1
            c.select(v1_id)
            assert c.selected_id == v1_id

        finally:
            await timeline.close()


class TestSessionModeIntegration:
    """Test session mode integration with ChoiceView."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_session_mode_property(self, temp_cwd: Path) -> None:
        """Test Session.mode property and set_mode method."""
        from activecontext.session.session_manager import Session

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        session = Session(
            session_id="test-session",
            cwd=str(temp_cwd),
            timeline=timeline,
        )

        try:
            # Default mode is "normal"
            assert session.mode == "normal"

            # Set mode updates the property
            session.set_mode("plan")
            assert session.mode == "plan"

            session.set_mode("brave")
            assert session.mode == "brave"

        finally:
            await timeline.close()

    @pytest.mark.asyncio
    async def test_session_mode_choice_view_integration(self, temp_cwd: Path) -> None:
        """Test that set_mode_choice_view builds mode mapping correctly."""
        from activecontext.context.view import ChoiceView
        from activecontext.session.session_manager import Session

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=str(temp_cwd))
        session = Session(
            session_id="test-session",
            cwd=str(temp_cwd),
            timeline=timeline,
        )

        try:
            # Create mode script nodes with paths matching the pattern
            await timeline.execute_statement(
                'normal_mode = markdown("modes/normal.md", content="# Normal", tokens=100)'
            )
            await timeline.execute_statement(
                'plan_mode = markdown("modes/plan.md", content="# Plan", tokens=100)'
            )
            await timeline.execute_statement(
                'brave_mode = markdown("modes/brave.md", content="# Brave", tokens=100)'
            )

            # Create choice view
            result = await timeline.execute_statement(
                "mode_choice = choice(normal_mode, plan_mode, brave_mode)"
            )
            assert result.status.value == "ok"

            ns = timeline.get_namespace()
            mode_choice = ns["mode_choice"]
            assert isinstance(mode_choice, ChoiceView)

            # Register with session
            session.set_mode_choice_view(mode_choice)

            # Verify mapping was built
            assert "normal" in session._mode_node_ids
            assert "plan" in session._mode_node_ids
            assert "brave" in session._mode_node_ids

            # Verify mode switching updates ChoiceView selection
            normal_id = session._mode_node_ids["normal"]
            plan_id = session._mode_node_ids["plan"]

            # Initial selection
            assert mode_choice.selected_id == normal_id

            # Switch to plan mode
            session.set_mode("plan")
            assert session.mode == "plan"
            assert mode_choice.selected_id == plan_id

        finally:
            await timeline.close()
