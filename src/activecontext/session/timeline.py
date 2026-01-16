"""Timeline: wrapper around StatementLog and PythonExec.

The Timeline is the canonical history of executed Python statements
for a session. It manages:
- Statement recording and indexing
- Python namespace execution
- Replay/re-execution from any point
"""

from __future__ import annotations

import time
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any

from activecontext.session.protocols import (
    ExecutionResult,
    ExecutionStatus,
    NamespaceDiff,
    Statement,
)


@dataclass
class _ExecutionRecord:
    """Internal record of a statement execution."""

    execution_id: str
    statement_id: str
    started_at: float
    ended_at: float
    status: ExecutionStatus
    stdout: str
    stderr: str
    exception: dict[str, Any] | None
    state_diff: NamespaceDiff


class Timeline:
    """Statement timeline with controlled Python execution.

    Each session has one Timeline that tracks all executed statements
    and maintains the Python namespace.
    """

    def __init__(self, session_id: str, cwd: str = ".") -> None:
        self._session_id = session_id
        self._cwd = cwd
        self._statements: list[Statement] = []
        self._executions: dict[str, list[_ExecutionRecord]] = {}  # statement_id -> executions

        # Controlled Python namespace
        self._namespace: dict[str, Any] = {}
        self._setup_namespace()

        # Tracked context objects (ViewHandle, GroupHandle, etc.)
        self._context_objects: dict[str, Any] = {}

        # Max output capture per statement
        self._max_stdout = 50000
        self._max_stderr = 10000

        # Done signal from agent
        self._done_called = False
        self._done_message: str | None = None

    @property
    def cwd(self) -> str:
        return self._cwd

    def _setup_namespace(self) -> None:
        """Initialize the Python namespace with injected functions."""
        # Import builtins we want to expose
        import builtins

        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if not name.startswith("_")
        }

        self._namespace = {
            "__builtins__": safe_builtins,
            "__name__": "__activecontext__",
            "__session_id__": self._session_id,
            # Inject context object constructors
            "view": self._make_view,
            "group": self._make_group,
            # Utility functions
            "ls": self._ls_handles,
            "show": self._show_handle,
            # Agent control
            "done": self._done,
        }

    def _make_view(
        self,
        path: str,
        *,
        pos: str = "0:0",
        tokens: int = 2000,
        lod: int = 0,
        mode: str = "paused",
    ) -> Any:
        """Create a ViewHandle (placeholder until context module is implemented)."""
        # TODO: Replace with actual ViewHandle once implemented
        from dataclasses import dataclass as dc
        from pathlib import Path

        timeline_cwd = self._cwd  # Capture for closure

        @dc
        class _PlaceholderView:
            path: str
            pos: str
            tokens: int
            lod: int
            mode: str
            _id: str = ""
            _cwd: str = ""

            def __post_init__(self) -> None:
                self._id = str(uuid.uuid4())[:8]
                self._cwd = timeline_cwd

            def SetPos(self, pos: str) -> _PlaceholderView:
                self.pos = pos
                return self

            def SetTokens(self, n: int) -> _PlaceholderView:
                self.tokens = n
                return self

            def SetLod(self, k: int) -> _PlaceholderView:
                self.lod = k
                return self

            def Run(self, freq: str = "Sync") -> _PlaceholderView:
                self.mode = "running"
                return self

            def Pause(self) -> _PlaceholderView:
                self.mode = "paused"
                return self

            def GetDigest(self) -> dict[str, Any]:
                return {
                    "id": self._id,
                    "type": "view",
                    "path": self.path,
                    "pos": self.pos,
                    "tokens": self.tokens,
                    "lod": self.lod,
                    "mode": self.mode,
                }

            def Render(self, tokens: int | None = None) -> str:
                """Render file content with line numbers, respecting token budget."""
                effective_tokens = tokens if tokens is not None else self.tokens

                # Resolve path relative to cwd
                full_path = Path(self._cwd) / self.path if self._cwd else Path(self.path)
                try:
                    content = full_path.read_text(encoding="utf-8")
                except Exception as e:
                    return f"[Error reading {self.path}: {e}]"

                # Parse position
                pos_parts = self.pos.split(":")
                start_line = int(pos_parts[0]) if pos_parts[0] else 0

                lines = content.splitlines()

                # Estimate token budget as ~4 chars per token
                char_budget = effective_tokens * 4

                numbered: list[str] = []
                char_count = 0
                for i, line in enumerate(lines[start_line:], start=start_line + 1):
                    numbered_line = f"{i:4d} | {line}"
                    if char_count + len(numbered_line) > char_budget:
                        remaining = len(lines) - start_line - len(numbered)
                        if remaining > 0:
                            numbered.append(f"     | ... [{remaining} more lines]")
                        break
                    numbered.append(numbered_line)
                    char_count += len(numbered_line) + 1

                end_line = start_line + len(numbered)
                header = f"=== {self.path} (lines {start_line + 1}-{end_line}) ===\n"
                return header + "\n".join(numbered)

        v = _PlaceholderView(path=path, pos=pos, tokens=tokens, lod=lod, mode=mode)
        self._context_objects[v._id] = v
        return v

    def _make_group(
        self,
        *members: Any,
        tokens: int = 500,
        lod: int = 1,
        mode: str = "paused",
    ) -> Any:
        """Create a GroupHandle (placeholder until context module is implemented)."""
        from dataclasses import dataclass as dc

        @dc
        class _PlaceholderGroup:
            members: tuple[Any, ...]
            tokens: int
            lod: int
            mode: str
            _id: str = ""

            def __post_init__(self) -> None:
                self._id = str(uuid.uuid4())[:8]

            def SetTokens(self, n: int) -> _PlaceholderGroup:
                self.tokens = n
                return self

            def SetLod(self, k: int) -> _PlaceholderGroup:
                self.lod = k
                return self

            def Run(self, freq: str = "Sync") -> _PlaceholderGroup:
                self.mode = "running"
                return self

            def Pause(self) -> _PlaceholderGroup:
                self.mode = "paused"
                return self

            def GetDigest(self) -> dict[str, Any]:
                return {
                    "id": self._id,
                    "type": "group",
                    "member_count": len(self.members),
                    "tokens": self.tokens,
                    "lod": self.lod,
                    "mode": self.mode,
                }

        g = _PlaceholderGroup(members=members, tokens=tokens, lod=lod, mode=mode)
        self._context_objects[g._id] = g
        return g

    def _ls_handles(self) -> list[dict[str, Any]]:
        """List all context object handles with brief digests."""
        return [obj.GetDigest() for obj in self._context_objects.values()]

    def _show_handle(self, obj: Any, *, lod: int | None = None, tokens: int | None = None) -> str:
        """Force render a handle (placeholder)."""
        digest = obj.GetDigest() if hasattr(obj, "GetDigest") else str(obj)
        return f"[{digest}]"

    def _done(self, message: str = "") -> None:
        """Signal that the agent has completed its task.

        Args:
            message: Final message to send to the user.
        """
        self._done_called = True
        self._done_message = message
        if message:
            print(message)

    def is_done(self) -> bool:
        """Check if done() was called."""
        return self._done_called

    def get_done_message(self) -> str | None:
        """Get the message passed to done(), if any."""
        return self._done_message

    def reset_done(self) -> None:
        """Reset the done signal (call at start of each prompt)."""
        self._done_called = False
        self._done_message = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def _capture_namespace_diff(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> NamespaceDiff:
        """Compute the diff between two namespace snapshots."""
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added = {k: type(after[k]).__name__ for k in after_keys - before_keys}
        deleted = list(before_keys - after_keys)

        changed = {}
        for k in before_keys & after_keys:
            if before[k] is not after[k]:
                changed[k] = f"{type(before[k]).__name__} -> {type(after[k]).__name__}"

        return NamespaceDiff(added=added, changed=changed, deleted=deleted)

    def _snapshot_namespace(self) -> dict[str, Any]:
        """Create a shallow snapshot of user-defined namespace entries."""
        return {
            k: v
            for k, v in self._namespace.items()
            if not k.startswith("__") and k not in ("view", "group", "ls", "show")
        }

    async def execute_statement(self, source: str) -> ExecutionResult:
        """Execute a Python statement and record it."""
        statement_id = str(uuid.uuid4())
        execution_id = str(uuid.uuid4())
        timestamp = time.time()

        # Record the statement
        stmt = Statement(
            statement_id=statement_id,
            index=len(self._statements),
            source=source,
            timestamp=timestamp,
        )
        self._statements.append(stmt)

        # Capture namespace before
        ns_before = self._snapshot_namespace()

        # Execute with output capture
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        started_at = time.time()
        status = ExecutionStatus.OK
        exception_info: dict[str, Any] | None = None

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Use exec for statements, eval for expressions
                try:
                    # Try as expression first
                    result = eval(source, self._namespace)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    # Fall back to exec for statements
                    exec(source, self._namespace)
        except Exception as e:
            status = ExecutionStatus.ERROR
            exception_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

        ended_at = time.time()

        # Capture outputs (truncated)
        stdout_val = stdout_capture.getvalue()[: self._max_stdout]
        stderr_val = stderr_capture.getvalue()[: self._max_stderr]

        # Compute namespace diff
        ns_after = self._snapshot_namespace()
        state_diff = self._capture_namespace_diff(ns_before, ns_after)

        # Record execution
        record = _ExecutionRecord(
            execution_id=execution_id,
            statement_id=statement_id,
            started_at=started_at,
            ended_at=ended_at,
            status=status,
            stdout=stdout_val,
            stderr=stderr_val,
            exception=exception_info,
            state_diff=state_diff,
        )
        self._executions.setdefault(statement_id, []).append(record)

        return ExecutionResult(
            execution_id=execution_id,
            statement_id=statement_id,
            status=status,
            stdout=stdout_val,
            stderr=stderr_val,
            exception=exception_info,
            state_diff=state_diff,
            duration_ms=(ended_at - started_at) * 1000,
        )

    async def replay_from(self, statement_index: int) -> AsyncIterator[ExecutionResult]:
        """Re-execute statements from a given index."""
        if statement_index < 0 or statement_index >= len(self._statements):
            return

        # Reset namespace
        self._namespace.clear()
        self._context_objects.clear()
        self._setup_namespace()

        # Replay statements from start to get to clean state, then from index
        for stmt in self._statements[:statement_index]:
            # Execute silently to rebuild state
            await self.execute_statement(stmt.source)

        # Now replay from index, yielding results
        for stmt in self._statements[statement_index:]:
            result = await self.execute_statement(stmt.source)
            yield result

    def get_statements(self) -> list[Statement]:
        """Get all statements in the timeline."""
        return list(self._statements)

    def get_namespace(self) -> dict[str, Any]:
        """Get current Python namespace snapshot."""
        return self._snapshot_namespace()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all tracked context objects."""
        return dict(self._context_objects)
