#!/usr/bin/env python3
"""Clean build artifacts and cache files from the project."""

import shutil
from pathlib import Path

DIRS_TO_REMOVE = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "*.egg-info",
]

FILES_TO_REMOVE = [
    "*.pyc",
    "*.pyo",
]

EXCLUDE_DIRS = {".venv", ".git", "node_modules"}


def _should_skip(path: Path) -> bool:
    """Check if path is inside an excluded directory."""
    return any(part in EXCLUDE_DIRS for part in path.parts)


def clean(root: Path | None = None) -> None:
    """Remove all build artifacts and cache directories."""
    root = root or Path.cwd()
    removed = []

    # Remove directories
    for pattern in DIRS_TO_REMOVE:
        for path in root.rglob(pattern):
            if path.is_dir() and not _should_skip(path):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(path)

    # Remove files
    for pattern in FILES_TO_REMOVE:
        for path in root.rglob(pattern):
            if path.is_file() and not _should_skip(path):
                path.unlink(missing_ok=True)
                removed.append(path)

    if removed:
        print(f"Removed {len(removed)} items:")
        for p in removed[:10]:
            print(f"  {p}")
        if len(removed) > 10:
            print(f"  ... and {len(removed) - 10} more")
    else:
        print("Nothing to clean.")


def main() -> None:
    clean()


if __name__ == "__main__":
    main()
