"""Tests for the clean module."""

import pytest
from pathlib import Path


class TestShouldSkip:
    """Tests for _should_skip function."""

    def test_skip_venv(self):
        from activecontext.clean import _should_skip

        path = Path(".venv/lib/python3.12/__pycache__")
        assert _should_skip(path) is True

    def test_skip_git(self):
        from activecontext.clean import _should_skip

        path = Path(".git/objects/pack")
        assert _should_skip(path) is True

    def test_skip_node_modules(self):
        from activecontext.clean import _should_skip

        path = Path("node_modules/package/__pycache__")
        assert _should_skip(path) is True

    def test_allow_regular_path(self):
        from activecontext.clean import _should_skip

        path = Path("src/activecontext/__pycache__")
        assert _should_skip(path) is False

    def test_allow_nested_path(self):
        from activecontext.clean import _should_skip

        path = Path("tests/fixtures/__pycache__")
        assert _should_skip(path) is False


class TestClean:
    """Tests for clean function."""

    def test_clean_removes_pycache(self, tmp_path: Path):
        from activecontext.clean import clean

        # Create __pycache__ directory with a file
        pycache = tmp_path / "src" / "__pycache__"
        pycache.mkdir(parents=True)
        (pycache / "module.cpython-312.pyc").write_bytes(b"fake bytecode")

        clean(tmp_path)

        assert not pycache.exists()

    def test_clean_removes_pytest_cache(self, tmp_path: Path):
        from activecontext.clean import clean

        pytest_cache = tmp_path / ".pytest_cache"
        pytest_cache.mkdir()
        (pytest_cache / "v" / "cache").mkdir(parents=True)
        (pytest_cache / "README.md").write_text("pytest cache")

        clean(tmp_path)

        assert not pytest_cache.exists()

    def test_clean_removes_mypy_cache(self, tmp_path: Path):
        from activecontext.clean import clean

        mypy_cache = tmp_path / ".mypy_cache"
        mypy_cache.mkdir()
        (mypy_cache / "3.12").mkdir()

        clean(tmp_path)

        assert not mypy_cache.exists()

    def test_clean_removes_pyc_files(self, tmp_path: Path):
        from activecontext.clean import clean

        pyc_file = tmp_path / "module.pyc"
        pyc_file.write_bytes(b"bytecode")

        clean(tmp_path)

        assert not pyc_file.exists()

    def test_clean_removes_pyo_files(self, tmp_path: Path):
        from activecontext.clean import clean

        pyo_file = tmp_path / "module.pyo"
        pyo_file.write_bytes(b"optimized bytecode")

        clean(tmp_path)

        assert not pyo_file.exists()

    def test_clean_preserves_venv(self, tmp_path: Path):
        from activecontext.clean import clean

        # Create __pycache__ inside .venv
        venv_cache = tmp_path / ".venv" / "lib" / "__pycache__"
        venv_cache.mkdir(parents=True)
        (venv_cache / "test.pyc").write_bytes(b"bytecode")

        clean(tmp_path)

        # Should NOT be removed
        assert venv_cache.exists()

    def test_clean_preserves_git(self, tmp_path: Path):
        from activecontext.clean import clean

        # Create cache-like dir inside .git
        git_objects = tmp_path / ".git" / "objects"
        git_objects.mkdir(parents=True)

        clean(tmp_path)

        assert git_objects.exists()

    def test_clean_preserves_node_modules(self, tmp_path: Path):
        from activecontext.clean import clean

        nm_cache = tmp_path / "node_modules" / ".cache"
        nm_cache.mkdir(parents=True)

        clean(tmp_path)

        assert nm_cache.exists()

    def test_clean_nested_pycache(self, tmp_path: Path):
        from activecontext.clean import clean

        # Create nested __pycache__ directories
        cache1 = tmp_path / "pkg" / "__pycache__"
        cache2 = tmp_path / "pkg" / "sub" / "__pycache__"
        cache1.mkdir(parents=True)
        cache2.mkdir(parents=True)

        clean(tmp_path)

        assert not cache1.exists()
        assert not cache2.exists()

    def test_clean_empty_directory(self, tmp_path: Path, capsys):
        from activecontext.clean import clean

        clean(tmp_path)

        captured = capsys.readouterr()
        assert "Nothing to clean" in captured.out

    def test_clean_reports_removed_items(self, tmp_path: Path, capsys):
        from activecontext.clean import clean

        pycache = tmp_path / "__pycache__"
        pycache.mkdir()

        clean(tmp_path)

        captured = capsys.readouterr()
        assert "Removed" in captured.out

    def test_clean_uses_cwd_by_default(self, tmp_path: Path, monkeypatch):
        from activecontext.clean import clean

        monkeypatch.chdir(tmp_path)
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()

        clean()  # No argument

        assert not pycache.exists()


class TestMain:
    """Tests for main entry point."""

    def test_main_runs(self, tmp_path: Path, monkeypatch):
        from activecontext.clean import main

        monkeypatch.chdir(tmp_path)
        # Should not raise
        main()
