"""File lock manager for coordinating access across processes.

Manages async file lock acquisition, release, and background tasks.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import LockNode, LockStatus
from activecontext.context.state import Expansion

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph


class LockManager:
    """Manages file lock acquisition and release with background tasks.

    Responsibilities:
    - Acquire file locks asynchronously with timeout
    - Track running lock acquisition tasks
    - Process lock results at tick time
    - Release held locks
    - Cancel pending lock acquisitions
    """

    def __init__(
        self,
        *,
        context_graph: ContextGraph,
        cwd: str,
    ):
        """Initialize lock manager.

        Args:
            context_graph: The session's context graph for adding LockNodes
            cwd: Working directory for resolving relative lock file paths
        """
        self._context_graph = context_graph
        self._cwd = cwd

        # Pending results from background tasks (processed at tick)
        self._pending_lock_results: list[tuple[str, LockStatus, str | None]] = []

        # Running background tasks keyed by node_id
        self._lock_tasks: dict[str, asyncio.Task[None]] = {}

        # Active file locks keyed by node_id (stores file handle for releasing)
        self._active_locks: dict[str, Any] = {}

    def acquire(
        self,
        lockfile: str,
        timeout: float = 30.0,
        *,
        tokens: int = 200,
        expansion: Expansion = Expansion.HEADER,
    ) -> LockNode:
        """Acquire an exclusive file lock asynchronously, returning a LockNode.

        Creates a LockNode in the context graph and starts lock acquisition
        in the background. The node's status updates when the lock is acquired
        or times out, and change notifications propagate up the DAG at tick time.

        The lock uses OS-level file locking (fcntl on Unix, msvcrt on Windows).
        The lockfile is created if it doesn't exist.

        Args:
            lockfile: Path to the lock file (will be created if needed).
            timeout: Timeout in seconds for acquisition (default: 30).
            tokens: Token budget for rendering (default: 200).
            state: Initial rendering state (default: COLLAPSED).

        Returns:
            LockNode that tracks the lock status.

        Example:
            lock = lock_file(".mylock", timeout=10)
            wait(lock, wake_prompt="Lock acquired, proceeding...")
        """
        # Resolve path relative to cwd
        if not os.path.isabs(lockfile):
            lockfile = os.path.join(self._cwd, lockfile)

        # Create the LockNode
        node = LockNode(
            lockfile=lockfile,
            timeout=timeout,
            tokens=tokens,
            expansion=expansion,
        )

        # Add to context graph
        self._context_graph.add_node(node)

        # Start background lock acquisition
        task = asyncio.create_task(
            self._background_task(
                node_id=node.node_id,
                lockfile=lockfile,
                timeout=timeout,
            )
        )
        self._lock_tasks[node.node_id] = task

        return node

    async def _background_task(
        self,
        node_id: str,
        lockfile: str,
        timeout: float,
    ) -> None:
        """Background task that acquires a file lock with timeout.

        This runs in the background while the agent continues. When complete,
        it stores the result in _pending_lock_results for processing at tick.
        """
        start_time = time.time()
        poll_interval = 0.1  # 100ms between lock attempts

        try:
            # Create the lock file if it doesn't exist
            Path(lockfile).parent.mkdir(parents=True, exist_ok=True)

            # Open file for locking (create if needed)
            lock_fd = open(lockfile, "w")

            # Try to acquire lock with timeout
            while True:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    # Timeout - close file and report
                    lock_fd.close()
                    self._pending_lock_results.append((node_id, LockStatus.TIMEOUT, None))
                    return

                try:
                    # Platform-specific locking
                    if sys.platform == "win32":
                        import msvcrt

                        msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Lock acquired - store file handle for later release
                    self._active_locks[node_id] = lock_fd
                    self._pending_lock_results.append((node_id, LockStatus.ACQUIRED, None))
                    return

                except OSError:
                    # Lock held by another process - wait and retry
                    await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            self._pending_lock_results.append((node_id, LockStatus.ERROR, "Cancelled"))
        except Exception as e:
            self._pending_lock_results.append((node_id, LockStatus.ERROR, str(e)))

    def process_pending_results(self) -> list[str]:
        """Process pending lock results and update nodes.

        Called during tick to apply async lock results. This triggers
        node change notifications that propagate up the DAG.

        Returns:
            List of node IDs that were updated.
        """
        updated_nodes: list[str] = []

        while self._pending_lock_results:
            node_id, status, error_msg = self._pending_lock_results.pop(0)

            node = self._context_graph.get_node(node_id)
            if not isinstance(node, LockNode):
                continue

            # Apply the result - this triggers _mark_changed() and notifications
            if status == LockStatus.ACQUIRED:
                node.set_acquired(os.getpid())
            elif status == LockStatus.TIMEOUT:
                node.set_timeout()
            elif status == LockStatus.ERROR:
                node.set_error(error_msg or "Unknown error")

            updated_nodes.append(node_id)

            # Clean up task reference
            self._lock_tasks.pop(node_id, None)

        return updated_nodes

    def release(self, lock: LockNode | str) -> bool:
        """Release a file lock and remove the lock file.

        Args:
            lock: The LockNode or node_id to release.

        Returns:
            True if lock was released, False if not found or not held.

        Example:
            lock_release(lock)  # or lock_release(lock.node_id)
        """
        node_id = lock.node_id if isinstance(lock, LockNode) else lock

        # Get the lock node
        node = self._context_graph.get_node(node_id)
        if not isinstance(node, LockNode):
            return False

        # Check if we hold the lock
        lock_fd = self._active_locks.get(node_id)
        if lock_fd is None:
            return False

        try:
            # Release the lock
            if sys.platform == "win32":
                import msvcrt

                try:
                    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # May already be released
            else:
                import fcntl

                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass  # May already be released

            # Close the file handle
            lock_fd.close()

            # Remove the lock file
            try:
                os.unlink(node.lockfile)
            except OSError:
                pass  # File may already be removed

            # Update node state
            node.set_released()

            # Clean up
            del self._active_locks[node_id]

            # Cancel any pending acquisition task
            task = self._lock_tasks.pop(node_id, None)
            if task and not task.done():
                task.cancel()

            return True

        except Exception:
            return False

    def cancel(self, node_id: str) -> bool:
        """Cancel a pending lock acquisition.

        Args:
            node_id: The LockNode ID to cancel.

        Returns:
            True if a task was cancelled, False if not found/already done.
        """
        task = self._lock_tasks.get(node_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel all pending lock acquisitions.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        for node_id, task in list(self._lock_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def release_all(self) -> int:
        """Release all held locks.

        Returns:
            Number of locks released.
        """
        released = 0
        for node_id in list(self._active_locks.keys()):
            if self.release(node_id):
                released += 1
        return released

    def has_pending_tasks(self) -> bool:
        """Check if there are any running lock acquisition tasks.

        Returns:
            True if any lock tasks are still running.
        """
        return len(self._lock_tasks) > 0
