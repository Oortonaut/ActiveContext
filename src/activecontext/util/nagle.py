"""Reusable Nagle-style batching buffer.

Accumulates writes per key and flushes them via a callback when either:
- A **size threshold** is reached (immediate flush), or
- A **time interval** elapses since the first un-flushed write (delayed flush).

This is the same algorithm used by TCP's Nagle algorithm to coalesce small
packets.  Both the ACP response-chunk transport and PTY output batching share
this implementation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

log = logging.getLogger(__name__)

# Type alias: async callback receiving (key, accumulated_text)
FlushCallback = Callable[[str, str], Awaitable[Any]]


class NagleBuffer:
    """Time + size triggered batch buffer.

    Parameters
    ----------
    flush_callback:
        ``async (key, text) -> None`` called when a batch is ready.
    flush_interval:
        Seconds to wait before a timer-based flush (default 50 ms).
    flush_threshold:
        Character count that triggers an immediate flush (default 100).
    """

    def __init__(
        self,
        flush_callback: FlushCallback,
        *,
        flush_interval: float = 0.05,
        flush_threshold: int = 100,
    ) -> None:
        self._flush_callback = flush_callback
        self.flush_interval = flush_interval
        self.flush_threshold = flush_threshold

        self._buffers: dict[str, str] = {}
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._closed_keys: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def write(self, key: str, text: str) -> None:
        """Accumulate *text* for *key*, flushing when the threshold is hit.

        If *key* has been closed, the write is silently discarded.
        """
        async with self._lock:
            if key in self._closed_keys:
                return
            self._buffers[key] = self._buffers.get(key, "") + text
            buffer_len = len(self._buffers[key])

        # Size threshold → immediate flush (cancel pending timer)
        if buffer_len >= self.flush_threshold:
            async with self._lock:
                task = self._flush_tasks.pop(key, None)
                if task and not task.done():
                    task.cancel()
            await self.flush(key)
            return

        # Schedule a delayed flush if none is pending
        async with self._lock:
            if key not in self._flush_tasks and key not in self._closed_keys:
                self._flush_tasks[key] = asyncio.create_task(
                    self._delayed_flush(key)
                )

    async def flush(self, key: str) -> None:
        """Force-flush buffered text for *key*.

        Safe to call even when the buffer is empty or the key is closed.
        """
        async with self._lock:
            if key in self._closed_keys:
                self._buffers.pop(key, None)
                self._flush_tasks.pop(key, None)
                return
            text = self._buffers.pop(key, "")
            self._flush_tasks.pop(key, None)

        if text:
            await self._flush_callback(key, text)

    async def flush_all(self) -> None:
        """Force-flush every key that has buffered text."""
        async with self._lock:
            keys = list(self._buffers.keys())
        for key in keys:
            await self.flush(key)

    async def close(self, key: str) -> None:
        """Mark *key* as closed: cancel pending timer, discard buffered text."""
        async with self._lock:
            self._closed_keys.add(key)
            self._buffers.pop(key, None)
            task = self._flush_tasks.pop(key, None)
            if task and not task.done():
                task.cancel()

    async def close_all(self) -> None:
        """Close every tracked key."""
        async with self._lock:
            keys = list(self._buffers.keys()) + list(self._flush_tasks.keys())
        for key in set(keys):
            await self.close(key)

    def reopen(self, key: str) -> None:
        """Remove *key* from the closed set so it can accept writes again."""
        self._closed_keys.discard(key)

    # ------------------------------------------------------------------
    # Introspection (useful for tests and the ACP agent)
    # ------------------------------------------------------------------

    def has_buffered(self, key: str) -> bool:
        """Return True if there is un-flushed text for *key*."""
        return key in self._buffers

    def has_pending_flush(self, key: str) -> bool:
        """Return True if a delayed flush task is scheduled for *key*."""
        return key in self._flush_tasks

    @property
    def buffered_keys(self) -> set[str]:
        """Return keys that currently have buffered text."""
        return set(self._buffers.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _delayed_flush(self, key: str) -> None:
        """Sleep for ``flush_interval``, then flush."""
        try:
            await asyncio.sleep(self.flush_interval)
            await self.flush(key)
        except asyncio.CancelledError:
            pass
