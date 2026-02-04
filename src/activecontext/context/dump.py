"""Context dump writer for logging projection snapshots to markdown files.

Provides:
- ``ContextDumpWriter`` — write numbered context-NNNNNN.md files with rotation
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from activecontext.config.schema import Config
    from activecontext.session.protocols import Projection

_log = logging.getLogger(__name__)

# Pattern for dump filenames: context-000001.md
_DUMP_RE = re.compile(r"^context-(\d{6})\.md$")





class ContextDumpWriter:
    """Writes numbered context dump markdown files with optional rotation.

    Files are named ``context-NNNNNN.md`` in the configured directory.
    Numbering continues from the highest existing file (survives restarts).

    Args:
        directory: Path to the dump directory.
        max_files: Maximum files to keep. ``None`` means unlimited.
    """

    def __init__(self, directory: str | Path, max_files: int | None = None) -> None:
        self._directory = Path(directory)
        self._max_files = max_files
        self._counter: int | None = None  # lazy-init on first write

    @classmethod
    def from_config(cls, config: Config) -> ContextDumpWriter | None:
        """Create a writer from config, or None if not configured.

        Args:
            config: The application config.

        Returns:
            A ContextDumpWriter if ``config.logging.context_dir`` is set,
            otherwise None.
        """
        context_dir = config.logging.context_dir
        if not isinstance(context_dir, str):
            return None
        return cls(
            directory=context_dir,
            max_files=config.logging.context_n,
        )

    def write(self, projection: Projection) -> Path:
        """Write a context dump file and rotate if needed.

        Args:
            projection: The Projection to dump.

        Returns:
            Path to the written file.
        """
        self._directory.mkdir(parents=True, exist_ok=True)

        if self._counter is None:
            existing = self._scan_existing()
            self._counter = max(existing) if existing else 0

        self._counter += 1
        filename = f"context-{self._counter:06d}.md"
        path = self._directory / filename

        content = projection.frame_context()
        path.write_text(content, encoding="utf-8")
        _log.debug("Wrote context dump: %s", path)

        if self._max_files is not None:
            self._rotate()

        return path

    def _scan_existing(self) -> list[int]:
        """Find existing dump file numbers in the directory.

        Returns:
            Sorted list of existing file numbers.
        """
        if not self._directory.exists():
            return []

        numbers: list[int] = []
        for child in self._directory.iterdir():
            m = _DUMP_RE.match(child.name)
            if m:
                numbers.append(int(m.group(1)))
        numbers.sort()
        return numbers

    def _rotate(self) -> None:
        """Delete oldest files if count exceeds max_files."""
        if self._max_files is None:
            return

        existing = self._scan_existing()
        to_delete = len(existing) - self._max_files

        for num in existing[:to_delete]:
            path = self._directory / f"context-{num:06d}.md"
            try:
                path.unlink()
                _log.debug("Rotated context dump: %s", path)
            except OSError as e:
                _log.warning("Failed to delete context dump %s: %s", path, e)
