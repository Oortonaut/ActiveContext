"""Tests for Timeline DSL functions.

These tests cover the DSL functions exposed by Timeline that aren't covered
by other test modules.
"""

import asyncio
from pathlib import Path

import pytest

from activecontext.session.timeline import Timeline


class TestLsHandles:
    """Test ls() function for listing context object handles."""

    @pytest.fixture
    def temp_cwd(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.asyncio
    async def test_ls_returns_context_objects(self, temp_cwd: Path) -> None:
        """Test that ls() returns context object digests."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
    async def test_markdown_nodes_render_with_display_id(self, temp_cwd: Path) -> None:
        """Test that markdown nodes render headings with {#text_N} annotations."""
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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

            # Verify display_id format is text_N
            for node in all_nodes:
                assert node.display_id.startswith("text_")
                num = int(node.display_id.split("_")[1])
                assert num >= 0

            # Render all nodes and concatenate
            rendered_parts = []
            for node in all_nodes:
                rendered = node.Render(
                    tokens=500,
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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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

        timeline = Timeline("test-session", cwd=str(temp_cwd))

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

        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
        timeline = Timeline("test-session", cwd=str(temp_cwd))

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
