"""PTY session manager.

Manages long-lived interactive PTY sessions with Nagle-batched output
that is applied at tick boundaries (same mutation model as ShellManager).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from activecontext.context.nodes import PtyNode, PtyStatus
from activecontext.context.state import Expansion
from activecontext.terminal.pty_backend import PtyBackend, create_pty_backend
from activecontext.util.nagle import NagleBuffer

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph

log = logging.getLogger(__name__)

# Sentinel used to signal "process exited" through the pending queue.
_EXIT_SENTINEL = object()


class PtyManager:
    """Manages interactive PTY sessions for the timeline.

    Lifecycle mirrors ShellManager:
    1. ``spawn()`` creates a PtyNode + starts background read loop
    2. Background loop writes output into a NagleBuffer
    3. NagleBuffer flushes batched text into ``_pending_pty_output``
    4. ``process_pending_output()`` is called at tick to apply mutations

    Thread-safety: only ``process_pending_output`` mutates PtyNode
    (from the event-loop thread at tick). The read loop and NagleBuffer
    callbacks only append to ``_pending_pty_output`` which is a list
    (append is thread-safe on CPython).
    """

    def __init__(
        self,
        *,
        context_graph: ContextGraph,
        cwd: str,
        flush_interval: float = 0.05,
        flush_threshold: int = 200,
    ):
        self._context_graph = context_graph
        self._cwd = cwd

        # Backends keyed by node_id
        self._backends: dict[str, PtyBackend] = {}

        # Read-loop tasks keyed by node_id
        self._read_tasks: dict[str, asyncio.Task[None]] = {}

        # Pending output (node_id, text) or (node_id, _EXIT_SENTINEL)
        # to be processed at tick boundaries.
        self._pending_pty_output: list[tuple[str, str | object]] = []

        # Nagle buffer batches raw PTY bytes before flushing.
        self._nagle = NagleBuffer(
            flush_callback=self._on_nagle_flush,
            flush_interval=flush_interval,
            flush_threshold=flush_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def spawn(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        columns: int = 80,
        rows: int = 24,
        *,
        expansion: Expansion = Expansion.CONTENT,
    ) -> PtyNode:
        """Create a PtyNode and begin spawning the process.

        The node is added to the context graph immediately (PENDING).
        A background task handles the actual ``backend.spawn()`` and
        starts the read loop once the process is alive.

        Returns:
            The PtyNode (status=PENDING until spawn completes).
        """
        node = PtyNode(
            command=command,
            args=args or [],
            default_expansion=expansion,
        )
        self._context_graph.add_node(node)

        task = asyncio.create_task(
            self._spawn_and_read(
                node_id=node.node_id,
                command=command,
                args=args,
                cwd=cwd or self._cwd,
                env=env,
                columns=columns,
                rows=rows,
            )
        )
        self._read_tasks[node.node_id] = task

        return node

    def send_input(self, node_id: str, text: str) -> bool:
        """Write *text* to a running PTY's stdin.

        Also records the input on the PtyNode for display.

        Returns:
            True if written, False if no such backend or not alive.
        """
        backend = self._backends.get(node_id)
        if backend is None or not backend.is_alive:
            return False

        backend.write(text.encode("utf-8", errors="replace"))

        node = self._context_graph.get_node(node_id)
        if isinstance(node, PtyNode):
            node.record_input(text)

        return True

    def process_pending_output(self) -> list[str]:
        """Drain pending output and apply to PtyNodes.

        Called during tick.  Returns list of updated node IDs.
        """
        updated: list[str] = []
        seen: set[str] = set()

        while self._pending_pty_output:
            node_id, payload = self._pending_pty_output.pop(0)

            node = self._context_graph.get_node(node_id)
            if not isinstance(node, PtyNode):
                continue

            if payload is _EXIT_SENTINEL:
                # Process exited — finalize the node.
                backend = self._backends.get(node_id)
                exit_code = -1
                if backend:
                    # Exit code was set by _read_loop before sentinel.
                    stored = getattr(backend, "_exit_code", None)
                    if stored is not None:
                        exit_code = stored

                if node.pty_status not in (PtyStatus.EXITED, PtyStatus.KILLED, PtyStatus.ERROR):
                    node.set_exited(exit_code)

                self._cleanup_backend(node_id)
            else:
                # Normal output text.
                node.append_output(payload)  # type: ignore[arg-type]

            if node_id not in seen:
                updated.append(node_id)
                seen.add(node_id)

        return updated

    def close(self, node_id: str) -> None:
        """Terminate a PTY session and clean up resources."""
        backend = self._backends.get(node_id)
        if backend and backend.is_alive:
            backend.kill()
        self._cleanup_backend(node_id)

        node = self._context_graph.get_node(node_id)
        if isinstance(node, PtyNode) and not node.is_complete:
            node.set_exited(-1, signal_name="SIGKILL")

    def close_all(self) -> None:
        """Terminate all PTY sessions (called on timeline/session shutdown)."""
        for node_id in list(self._backends):
            self.close(node_id)
        asyncio.ensure_future(self._nagle.close_all())

    def resize(self, node_id: str, columns: int, rows: int) -> None:
        """Resize a running PTY's window."""
        backend = self._backends.get(node_id)
        if backend:
            backend.resize(columns, rows)

    def has_pending_tasks(self) -> bool:
        """True if any PTY sessions are still active."""
        return bool(self._backends) or bool(self._read_tasks)

    # ------------------------------------------------------------------
    # Internal: spawn + read loop
    # ------------------------------------------------------------------

    async def _spawn_and_read(
        self,
        node_id: str,
        command: str,
        args: list[str] | None,
        cwd: str,
        env: dict[str, str] | None,
        columns: int,
        rows: int,
    ) -> None:
        """Background task: create backend, spawn, run read loop."""
        node = self._context_graph.get_node(node_id)
        try:
            backend = create_pty_backend(columns, rows)
            await backend.spawn(command, args or [], cwd=cwd, env=env)
            self._backends[node_id] = backend

            if isinstance(node, PtyNode):
                node.set_running()

            await self._read_loop(node_id, backend)
        except Exception as exc:
            log.error("PTY spawn/read failed for %s: %s", node_id, exc)
            if isinstance(node, PtyNode) and not node.is_complete:
                node.set_error(str(exc))
        finally:
            self._read_tasks.pop(node_id, None)

    async def _read_loop(self, node_id: str, backend: PtyBackend) -> None:
        """Continuously read from the PTY and feed into the NagleBuffer."""
        try:
            while True:
                data = await backend.read(4096)
                if not data:
                    if not backend.is_alive:
                        break
                    # Empty read but process is alive — yield and retry.
                    await asyncio.sleep(0.01)
                    continue

                text = data.decode("utf-8", errors="replace")
                await self._nagle.write(node_id, text)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.debug("PTY read loop error for %s: %s", node_id, exc)
        finally:
            # Flush any remaining buffered output.
            await self._nagle.flush(node_id)

            # Collect exit code.
            try:
                exit_code = await asyncio.wait_for(backend.wait(), timeout=2.0)
                # Store exit code on backend for retrieval in process_pending_output.
                backend._exit_code = exit_code  # type: ignore[attr-defined]
            except (asyncio.TimeoutError, Exception):
                pass

            # Queue the exit sentinel.
            self._pending_pty_output.append((node_id, _EXIT_SENTINEL))

    # ------------------------------------------------------------------
    # Internal: NagleBuffer callback + cleanup
    # ------------------------------------------------------------------

    async def _on_nagle_flush(self, node_id: str, text: str) -> None:
        """Called by NagleBuffer when a batch is ready."""
        self._pending_pty_output.append((node_id, text))

    def _cleanup_backend(self, node_id: str) -> None:
        """Cancel read task and close backend for *node_id*."""
        task = self._read_tasks.pop(node_id, None)
        if task and not task.done():
            task.cancel()

        backend = self._backends.pop(node_id, None)
        if backend:
            backend.close()

        # Cancel any pending NagleBuffer timers for this key.
        asyncio.ensure_future(self._nagle.close(node_id))
