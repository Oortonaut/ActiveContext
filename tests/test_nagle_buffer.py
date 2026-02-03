"""Tests for NagleBuffer – the reusable Nagle-style batching utility."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from activecontext.util.nagle import NagleBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buffer(
    *,
    flush_interval: float = 10.0,  # long default so timers don't fire
    flush_threshold: int = 100,
    callback: AsyncMock | None = None,
) -> tuple[NagleBuffer, AsyncMock]:
    cb = callback or AsyncMock()
    buf = NagleBuffer(cb, flush_interval=flush_interval, flush_threshold=flush_threshold)
    return buf, cb


# ---------------------------------------------------------------------------
# write() accumulation
# ---------------------------------------------------------------------------


class TestWrite:
    @pytest.mark.asyncio
    async def test_accumulates_text(self):
        buf, cb = _make_buffer()
        await buf.write("k", "hello ")
        await buf.write("k", "world")

        assert buf.has_buffered("k")
        cb.assert_not_called()

        # Clean up timer
        await buf.close("k")

    @pytest.mark.asyncio
    async def test_schedules_delayed_flush(self):
        buf, cb = _make_buffer()
        await buf.write("k", "short")

        assert buf.has_pending_flush("k")
        cb.assert_not_called()

        await buf.close("k")

    @pytest.mark.asyncio
    async def test_discards_for_closed_key(self):
        buf, cb = _make_buffer()
        await buf.close("k")
        await buf.write("k", "dropped")

        assert not buf.has_buffered("k")
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# Threshold-based immediate flush
# ---------------------------------------------------------------------------


class TestThresholdFlush:
    @pytest.mark.asyncio
    async def test_flushes_when_threshold_exceeded(self):
        buf, cb = _make_buffer(flush_threshold=5)
        await buf.write("k", "abcdef")  # 6 > 5

        assert not buf.has_buffered("k")
        cb.assert_awaited_once_with("k", "abcdef")

    @pytest.mark.asyncio
    async def test_accumulates_then_flushes_at_threshold(self):
        buf, cb = _make_buffer(flush_threshold=10)
        await buf.write("k", "hello")  # 5 chars, below threshold
        await buf.write("k", "world!")  # 11 total, above threshold

        cb.assert_awaited_once_with("k", "helloworld!")
        assert not buf.has_buffered("k")

    @pytest.mark.asyncio
    async def test_threshold_cancels_pending_timer(self):
        buf, cb = _make_buffer(flush_threshold=5, flush_interval=10.0)

        await buf.write("k", "ab")   # below threshold, schedules timer
        assert buf.has_pending_flush("k")

        await buf.write("k", "cdef")  # 6 total, above threshold
        assert not buf.has_pending_flush("k")
        cb.assert_awaited_once_with("k", "abcdef")


# ---------------------------------------------------------------------------
# Timer-based delayed flush
# ---------------------------------------------------------------------------


class TestDelayedFlush:
    @pytest.mark.asyncio
    async def test_flushes_after_interval(self):
        buf, cb = _make_buffer(flush_interval=0.01)
        await buf.write("k", "delayed")

        # Wait for the timer to fire
        await asyncio.sleep(0.05)

        cb.assert_awaited_once_with("k", "delayed")
        assert not buf.has_buffered("k")


# ---------------------------------------------------------------------------
# flush() / flush_all()
# ---------------------------------------------------------------------------


class TestExplicitFlush:
    @pytest.mark.asyncio
    async def test_flush_sends_and_clears(self):
        buf, cb = _make_buffer()
        await buf.write("k", "text")
        await buf.flush("k")

        cb.assert_awaited_once_with("k", "text")
        assert not buf.has_buffered("k")

    @pytest.mark.asyncio
    async def test_flush_noop_when_empty(self):
        buf, cb = _make_buffer()
        await buf.flush("k")

        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_skips_closed_key(self):
        buf, cb = _make_buffer()
        buf._buffers["k"] = "leftover"
        await buf.close("k")
        await buf.flush("k")

        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_all(self):
        buf, cb = _make_buffer()
        await buf.write("a", "one")
        await buf.write("b", "two")
        await buf.flush_all()

        assert cb.await_count == 2
        assert not buf.has_buffered("a")
        assert not buf.has_buffered("b")


# ---------------------------------------------------------------------------
# close() / close_all()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_discards_buffer_and_timer(self):
        buf, cb = _make_buffer()
        await buf.write("k", "data")
        assert buf.has_buffered("k")
        assert buf.has_pending_flush("k")

        await buf.close("k")

        assert not buf.has_buffered("k")
        assert not buf.has_pending_flush("k")
        cb.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_all(self):
        buf, cb = _make_buffer()
        await buf.write("a", "x")
        await buf.write("b", "y")
        await buf.close_all()

        assert not buf.has_buffered("a")
        assert not buf.has_buffered("b")
        cb.assert_not_called()


# ---------------------------------------------------------------------------
# reopen()
# ---------------------------------------------------------------------------


class TestReopen:
    @pytest.mark.asyncio
    async def test_reopen_allows_new_writes(self):
        buf, cb = _make_buffer()
        await buf.close("k")
        await buf.write("k", "dropped")
        assert not buf.has_buffered("k")

        buf.reopen("k")
        await buf.write("k", "accepted")
        assert buf.has_buffered("k")
        await buf.flush("k")
        cb.assert_awaited_once_with("k", "accepted")


# ---------------------------------------------------------------------------
# Multiple keys
# ---------------------------------------------------------------------------


class TestMultipleKeys:
    @pytest.mark.asyncio
    async def test_independent_buffers(self):
        buf, cb = _make_buffer(flush_threshold=5)

        await buf.write("a", "abc")  # below threshold
        await buf.write("b", "12345678")  # above threshold

        # Only b should have flushed
        cb.assert_awaited_once_with("b", "12345678")
        assert buf.has_buffered("a")
        assert not buf.has_buffered("b")

        await buf.close("a")

    @pytest.mark.asyncio
    async def test_close_one_key_does_not_affect_other(self):
        buf, cb = _make_buffer()
        await buf.write("a", "aaa")
        await buf.write("b", "bbb")
        await buf.close("a")

        assert not buf.has_buffered("a")
        assert buf.has_buffered("b")

        await buf.flush("b")
        cb.assert_awaited_once_with("b", "bbb")
