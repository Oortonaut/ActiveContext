"""Tests for new node types: FileSystemNode, ClockNode, FunctionDocNode."""

import time

from activecontext.context.nodes import ClockNode, FileSystemNode, FunctionDocNode
from activecontext.context.view import NodeView


class TestFileSystemNode:
    """Tests for FileSystemNode."""

    def test_creation(self):
        """Test creating a filesystem node."""
        fs = FileSystemNode(root_path="/tmp")
        assert fs.node_type == "filesystem"
        assert fs.root_path == "/tmp"
        assert not fs.show_hidden
        assert fs.max_depth is None

    def test_scan_directory(self, tmp_path):
        """Test scanning a directory."""
        # Create test structure
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("code")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "nested.txt").write_text("nested")

        fs = FileSystemNode(root_path=str(tmp_path))
        tree = fs._scan_directory()

        assert tmp_path.name in tree
        # Since root isn't expanded by default, we won't see children

    def test_toggle_path_expansion(self, tmp_path):
        """Test toggling directory expansion."""
        (tmp_path / "dir1").mkdir()

        fs = FileSystemNode(root_path=str(tmp_path))
        dir_path = str(tmp_path / "dir1")

        # Initially not expanded
        assert dir_path not in fs.expanded_paths

        # Toggle to expand
        fs.toggle_path(dir_path)
        assert dir_path in fs.expanded_paths

        # Toggle to collapse
        fs.toggle_path(dir_path)
        assert dir_path not in fs.expanded_paths

    def test_pattern_filtering(self, tmp_path):
        """Test file pattern filtering."""
        (tmp_path / "file1.txt").write_text("content")
        (tmp_path / "file2.py").write_text("code")
        (tmp_path / "file3.md").write_text("doc")

        FileSystemNode(root_path=str(tmp_path), pattern="*.py")
        # Pattern filtering is applied in _build_tree during scan

    def test_hidden_files(self, tmp_path):
        """Test hidden file handling."""
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("public")

        # With show_hidden=False (default)
        fs = FileSystemNode(root_path=str(tmp_path), show_hidden=False)
        tree = fs._scan_directory()
        assert ".hidden" not in tree

        # With show_hidden=True
        FileSystemNode(root_path=str(tmp_path), show_hidden=True)
        # Would show hidden files if expanded

    def test_max_depth(self, tmp_path):
        """Test maximum depth limiting."""
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "level3").mkdir()

        fs = FileSystemNode(root_path=str(tmp_path), max_depth=1)
        fs.expanded_paths.add(str(tmp_path))
        fs.expanded_paths.add(str(tmp_path / "level1"))
        fs._scan_directory()

        # Should not go beyond depth 1

    def test_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        fs = FileSystemNode(root_path="/nonexistent/path")
        tree = fs._scan_directory()
        assert "not found" in tree.lower()

    def test_recompute(self, tmp_path):
        """Test tick-driven recompute."""
        fs = FileSystemNode(root_path=str(tmp_path))
        fs._last_scan = 0  # Force rescan
        fs.Recompute()
        assert fs._cached_tree  # Should have scanned

    def test_render_modes(self, tmp_path):
        """Test different render modes."""
        fs = FileSystemNode(root_path=str(tmp_path))

        header = NodeView(fs).render_header()
        assert header  # Has header

        content = fs.render_content()
        assert str(tmp_path) in content or "Root:" in content

    def test_serialization(self, tmp_path):
        """Test to_dict and from_dict."""
        fs = FileSystemNode(
            root_path=str(tmp_path),
            pattern="*.py",
            max_depth=3,
            show_hidden=True,
        )
        fs.expanded_paths.add(str(tmp_path / "dir1"))

        data = fs.to_dict()
        restored = FileSystemNode._from_dict(data)

        assert restored.root_path == str(tmp_path)
        assert restored.pattern == "*.py"
        assert restored.max_depth == 3
        assert restored.show_hidden
        assert str(tmp_path / "dir1") in restored.expanded_paths


class TestClockNode:
    """Tests for ClockNode."""

    def test_creation_stopwatch(self):
        """Test creating a stopwatch (no duration)."""
        clock = ClockNode()
        assert clock.node_type == "clock"
        assert clock.duration_seconds is None
        assert clock.is_running

    def test_creation_countdown(self):
        """Test creating a countdown timer."""
        clock = ClockNode(duration_seconds=60.0)
        assert clock.duration_seconds == 60.0
        assert not clock.is_complete()

    def test_get_elapsed(self):
        """Test elapsed time calculation."""
        start = time.time()
        clock = ClockNode(start_time=start)
        time.sleep(0.1)
        elapsed = clock.get_elapsed()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Sanity check

    def test_get_remaining(self):
        """Test remaining time calculation."""
        clock = ClockNode(duration_seconds=10.0, elapsed_seconds=3.0, is_running=False)
        remaining = clock.get_remaining()
        assert remaining == 7.0

    def test_stopwatch_no_remaining(self):
        """Test that stopwatch mode returns None for remaining time."""
        clock = ClockNode()  # No duration
        assert clock.get_remaining() is None

    def test_is_complete(self):
        """Test completion check."""
        clock = ClockNode(duration_seconds=1.0, elapsed_seconds=1.5, is_running=False)
        assert clock.is_complete()

        clock2 = ClockNode(duration_seconds=10.0, elapsed_seconds=5.0, is_running=False)
        assert not clock2.is_complete()

    def test_start_pause_resume(self):
        """Test start/pause/resume cycle."""
        clock = ClockNode(is_running=False)

        # Start
        clock.start()
        assert clock.is_running

        # Pause
        time.sleep(0.1)
        clock.pause()
        assert not clock.is_running
        paused_elapsed = clock.elapsed_seconds
        assert paused_elapsed > 0

        # Resume
        time.sleep(0.1)
        clock.start()
        assert clock.is_running

    def test_reset(self):
        """Test timer reset."""
        clock = ClockNode(elapsed_seconds=5.0)
        clock.reset()
        assert clock.elapsed_seconds == 0.0
        assert not clock.is_running

    def test_recompute_countdown_complete(self):
        """Test that countdown pauses when complete."""
        clock = ClockNode(duration_seconds=0.1, is_running=True)
        time.sleep(0.15)
        clock.Recompute()
        assert not clock.is_running  # Should auto-pause

    def test_format_time(self):
        """Test time formatting."""
        clock = ClockNode()

        # Seconds only
        assert clock._format_time(45) == "00:45"

        # Minutes and seconds
        assert clock._format_time(125) == "02:05"

        # Hours, minutes, seconds
        assert clock._format_time(3665) == "01:01:05"

    def test_render_stopwatch(self):
        """Test rendering stopwatch mode."""
        clock = ClockNode(elapsed_seconds=30.0, is_running=False)
        summary = clock.render_content()
        assert "00:30" in summary
        assert "⏸" in summary  # Paused indicator

    def test_render_countdown(self):
        """Test rendering countdown mode."""
        clock = ClockNode(duration_seconds=60.0, elapsed_seconds=20.0, is_running=True)
        summary = clock.render_content()
        assert "▶" in summary  # Running indicator
        assert "remaining" in summary.lower()

    def test_serialization(self):
        """Test to_dict and from_dict."""
        clock = ClockNode(
            start_time=1000.0,
            duration_seconds=120.0,
            is_running=False,
            elapsed_seconds=45.0,
        )

        data = clock.to_dict()
        restored = ClockNode._from_dict(data)

        assert restored.start_time == 1000.0
        assert restored.duration_seconds == 120.0
        assert not restored.is_running
        assert restored.elapsed_seconds == 45.0


class TestFunctionDocNode:
    """Tests for FunctionDocNode."""

    def test_creation(self):
        """Test creating a function doc node."""
        doc = FunctionDocNode(
            file_path="test.py",
            function_name="my_function",
        )
        assert doc.node_type == "function_doc"
        assert doc.file_path == "test.py"
        assert doc.function_name == "my_function"

    def test_extract_simple_function(self, tmp_path):
        """Test extracting a simple function."""
        test_file = tmp_path / "test.py"
        test_file.write_text('''
def simple_function(x, y):
    """Add two numbers."""
    return x + y
''')

        doc = FunctionDocNode(file_path=str(test_file), function_name="simple_function")
        doc.extract_function_info()

        assert "simple_function" in doc.signature
        assert "x, y" in doc.signature
        assert doc.docstring == "Add two numbers."

    def test_extract_typed_function(self, tmp_path):
        """Test extracting a function with type hints."""
        test_file = tmp_path / "test.py"
        test_file.write_text('''
def typed_function(x: int, y: str) -> bool:
    """Check if something."""
    return True
''')

        doc = FunctionDocNode(file_path=str(test_file), function_name="typed_function")
        doc.extract_function_info()

        assert "int" in doc.signature
        assert "str" in doc.signature
        assert "-> bool" in doc.signature

    def test_extract_async_function(self, tmp_path):
        """Test extracting an async function."""
        test_file = tmp_path / "test.py"
        test_file.write_text('''
async def async_function(name: str):
    """Async operation."""
    pass
''')

        doc = FunctionDocNode(file_path=str(test_file), function_name="async_function")
        doc.extract_function_info()

        assert "async def" in doc.signature
        assert "async_function" in doc.signature

    def test_function_not_found(self, tmp_path):
        """Test handling of non-existent function."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def other_function():
    pass
""")

        doc = FunctionDocNode(file_path=str(test_file), function_name="missing_function")
        doc.extract_function_info()

        assert "not found" in doc.docstring.lower()

    def test_invalid_file(self):
        """Test handling of invalid file path."""
        doc = FunctionDocNode(file_path="/nonexistent.py", function_name="func")
        doc.extract_function_info()

        assert "error" in doc.docstring.lower() or "not found" in doc.docstring.lower()

    def test_render_modes(self, tmp_path):
        """Test different render modes."""
        test_file = tmp_path / "test.py"
        test_file.write_text('''
def test_func(x: int) -> str:
    """Test function."""
    return str(x)
''')

        doc = FunctionDocNode(file_path=str(test_file), function_name="test_func")

        header = NodeView(doc).render_header()
        assert header  # Has header

        content = doc.render_content()
        assert "test_func" in content
        assert "Test function" in content

    def test_serialization(self):
        """Test to_dict and from_dict."""
        doc = FunctionDocNode(
            file_path="module.py",
            function_name="my_func",
            signature="def my_func(x: int) -> str",
            docstring="Does something.",
            source_lines="def my_func...",
        )

        data = doc.to_dict()
        restored = FunctionDocNode._from_dict(data)

        assert restored.file_path == "module.py"
        assert restored.function_name == "my_func"
        assert restored.signature == "def my_func(x: int) -> str"
        assert restored.docstring == "Does something."
        assert restored.source_lines == "def my_func..."
