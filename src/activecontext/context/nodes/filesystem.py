"""FileSystemNode - Directory tree view with filtering and expand/collapse support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class FileSystemNode(ContextNode):
    """Directory tree view with filtering and expand/collapse support.

    Attributes:
        root_path: Root directory path to display
        pattern: Glob pattern for filtering (e.g., "*.py", "**/*.md")
        max_depth: Maximum directory depth to display (None = unlimited)
        show_hidden: Whether to show hidden files/directories
        expanded_paths: Set of expanded directory paths
    """

    root_path: str = "."
    pattern: str | None = None
    max_depth: int | None = None
    show_hidden: bool = False
    expanded_paths: set[str] = field(default_factory=set)
    _cached_tree: str = field(default="", init=False, repr=False)
    _last_scan: float = field(default=0.0, init=False, repr=False)

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "root_path": self.root_path,
            "pattern": self.pattern,
            "max_depth": self.max_depth,
            "expansion": self.default_expansion.value,
        }

    def _scan_directory(self) -> str:
        """Scan directory and build tree representation."""
        try:
            root = Path(self.root_path).resolve()
            if not root.exists():
                return f"[Directory not found: {self.root_path}]\n"

            lines: list[str] = []
            self._build_tree(root, "", lines, 0)
            return "\\n".join(lines)
        except Exception as e:
            return f"[Error scanning directory: {e}]\n"

    def _build_tree(
        self,
        path: Path,
        prefix: str,
        lines: list[str],
        depth: int,
    ) -> None:
        """Recursively build tree structure."""
        if self.max_depth is not None and depth > self.max_depth:
            return

        # Filter hidden files
        if not self.show_hidden and path.name.startswith("."):
            return

        # Apply pattern filter - skip non-matching files (but still recurse into dirs)
        if self.pattern and not path.match(self.pattern) and not path.is_dir():
            return

        # Add current item
        is_dir = path.is_dir()
        marker = "📁" if is_dir else "📄"
        is_expanded = str(path) in self.expanded_paths

        if is_dir:
            marker = "📂" if is_expanded else "📁"

        lines.append(f"{prefix}{marker} {path.name}")

        # Recurse into directories if expanded
        if is_dir and (is_expanded or depth == 0):
            try:
                children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
                for i, child in enumerate(children):
                    is_last = i == len(children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    self._build_tree(child, new_prefix, lines, depth + 1)
            except PermissionError:
                lines.append(f"{prefix}    [Permission denied]")

    def toggle_path(self, path: str) -> FileSystemNode:
        """Toggle expansion state of a directory path."""
        if path in self.expanded_paths:
            self.expanded_paths.remove(path)
        else:
            self.expanded_paths.add(path)
        self._last_scan = 0  # Force rescan
        self.mark_changed(f"Toggled {path}")
        return self

    def Recompute(self) -> None:
        """Recompute directory tree on tick."""
        current_time = time.time()
        # Rescan every 5 seconds or on expansion change
        if current_time - self._last_scan > 5.0:
            self._cached_tree = self._scan_directory()
            self._last_scan = current_time

    def render_content(self) -> str:
        """Render root path and directory tree."""
        if not self._cached_tree:
            self._cached_tree = self._scan_directory()
        return f"Root: {self.root_path}\n{self._cached_tree}"

    def render_digest(self) -> str:
        """Return filesystem node indicator with full path."""
        return f"FS: {self.root_path}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize FileSystemNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "root_path": self.root_path,
                "pattern": self.pattern,
                "max_depth": self.max_depth,
                "show_hidden": self.show_hidden,
                "expanded_paths": list(self.expanded_paths),
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> FileSystemNode:
        """Deserialize FileSystemNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            root_path=data.get("root_path", "."),
            pattern=data.get("pattern"),
            max_depth=data.get("max_depth"),
            show_hidden=data.get("show_hidden", False),
            expanded_paths=set(data.get("expanded_paths", [])),
        )
