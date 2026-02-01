"""Multiplexed transport for ACP and LSP on shared stdio.

Protocol detection:
- LSP: starts with "Content-Length:" header
- ACP: newline-delimited JSON (no header)

This module enables running both ACP and LSP protocols over a single
stdio connection. Each incoming message is inspected to determine which
protocol it belongs to, then dispatched to the appropriate handler.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from activecontext.transport.lsp.framing import (
    CONTENT_ENCODING,
    LSPFramingError,
)
from activecontext.transport.lsp.framing import (
    read_message as lsp_read_message,
)
from activecontext.transport.lsp.framing import (
    write_message as lsp_write_message,
)

logger = logging.getLogger(__name__)

# The LSP header prefix used for protocol detection.
_LSP_HEADER_PREFIX = b"Content-Length:"


class ProtocolMode(Enum):
    """Transport protocol mode."""

    ACP = "acp"
    LSP = "lsp"
    MULTIPLEXED = "multiplexed"


class MultiplexedTransport:
    """Routes stdin/stdout between ACP and LSP based on message format.

    In MULTIPLEXED mode, each incoming message is inspected:
    - If it starts with ``Content-Length:``, it is treated as LSP.
    - If it starts with ``{``, it is treated as ACP (newline-delimited JSON).

    In ACP or LSP mode, all messages are routed to the single handler
    without protocol detection.
    """

    def __init__(
        self,
        mode: ProtocolMode = ProtocolMode.MULTIPLEXED,
        on_acp_message: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_lsp_message: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._mode = mode
        self._on_acp_message = on_acp_message
        self._on_lsp_message = on_lsp_message
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._stopped = False
        # Lock to serialise writes so interleaved messages stay intact.
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Start reading and routing messages.

        Args:
            reader: Async stream reader (typically stdin).
            writer: Async stream writer (typically stdout).
        """
        self._reader = reader
        self._writer = writer
        self._stopped = False
        self._read_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Stop the transport, cancelling the read loop."""
        self._stopped = True
        if self._read_task is not None and not self._read_task.done():
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None

    # ------------------------------------------------------------------
    # Protocol detection
    # ------------------------------------------------------------------

    def detect_protocol(self, data: bytes) -> ProtocolMode:
        """Detect whether *data* is the start of an LSP or ACP message.

        Args:
            data: The first bytes peeked from the stream.

        Returns:
            ``ProtocolMode.LSP`` if the data starts with ``Content-Length:``,
            ``ProtocolMode.ACP`` if it starts with ``{``.

        Raises:
            ValueError: If the data does not match either protocol.
        """
        stripped = data.lstrip()
        if stripped.startswith(_LSP_HEADER_PREFIX):
            return ProtocolMode.LSP
        if stripped.startswith(b"{"):
            return ProtocolMode.ACP
        raise ValueError(f"Cannot detect protocol from data: {data[:40]!r}")

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send_acp(self, msg: dict[str, Any]) -> None:
        """Send an ACP message (newline-delimited JSON).

        Args:
            msg: JSON-serialisable dictionary to send.

        Raises:
            RuntimeError: If the transport has not been started.
        """
        if self._writer is None:
            raise RuntimeError("Transport not started")
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        async with self._write_lock:
            self._writer.write(line.encode(CONTENT_ENCODING))
            await self._writer.drain()

    async def send_lsp(self, msg: dict[str, Any]) -> None:
        """Send an LSP message (Content-Length framed).

        Args:
            msg: JSON-serialisable dictionary to send.

        Raises:
            RuntimeError: If the transport has not been started.
        """
        if self._writer is None:
            raise RuntimeError("Transport not started")
        async with self._write_lock:
            await lsp_write_message(self._writer, msg)

    # ------------------------------------------------------------------
    # Read loop
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        """Main read loop -- peek at bytes, detect protocol, dispatch."""
        assert self._reader is not None
        reader = self._reader

        while not self._stopped:
            try:
                # Peek ahead to determine the protocol without consuming
                # the data.  We read up to 32 bytes which is enough to
                # see ``Content-Length:`` or ``{``.
                peeked = await self._peek(reader)
                if peeked is None:
                    # EOF
                    logger.debug("EOF on reader, stopping read loop")
                    break

                # Determine which protocol this message belongs to.
                if self._mode == ProtocolMode.ACP:
                    await self._read_acp_message(reader)
                elif self._mode == ProtocolMode.LSP:
                    await self._read_lsp_message(reader)
                else:
                    # Multiplexed -- detect from content.
                    try:
                        detected = self.detect_protocol(peeked)
                    except ValueError:
                        # Unrecognised data -- skip the line and log.
                        line = await reader.readline()
                        logger.warning(
                            "Skipping unrecognised data: %r",
                            line[:80],
                        )
                        continue

                    if detected == ProtocolMode.LSP:
                        await self._read_lsp_message(reader)
                    else:
                        await self._read_acp_message(reader)

            except asyncio.CancelledError:
                break
            except Exception:
                if self._stopped:
                    break
                logger.exception("Error in multiplexer read loop")
                # Small sleep to avoid tight-looping on persistent errors.
                await asyncio.sleep(0.05)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _peek(self, reader: asyncio.StreamReader) -> bytes | None:
        """Non-destructive peek at the stream's internal buffer.

        Waits until at least 1 byte is available, then returns the
        buffered data *without* consuming it.  Returns ``None`` on EOF.
        """
        # Wait for data or EOF.  ``_wait_for_data`` is internal but the
        # only reliable way to wait without consuming.  We fall back to
        # a tiny read + buffer push if it is unavailable.
        while not reader._buffer and not reader.at_eof():  # type: ignore[attr-defined]
            try:
                # Read a small chunk -- this blocks until data arrives.
                chunk = await reader.read(1)
                if not chunk:
                    return None
                # Put it back so the protocol readers can consume it.
                # ``_buffer`` is a bytearray exposed by StreamReader.
                reader._buffer[0:0] = chunk  # type: ignore[attr-defined]
                break
            except asyncio.CancelledError:
                raise

        if reader.at_eof() and not reader._buffer:  # type: ignore[attr-defined]
            return None

        return bytes(reader._buffer)  # type: ignore[attr-defined]

    async def _read_acp_message(self, reader: asyncio.StreamReader) -> None:
        """Read a single newline-delimited JSON message and dispatch."""
        line = await reader.readline()
        if not line:
            return  # EOF

        line = line.strip()
        if not line:
            return  # blank line

        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid ACP JSON: %s (line: %r)", exc, line[:80])
            return

        if not isinstance(msg, dict):
            logger.warning("ACP message must be a JSON object, got %s", type(msg).__name__)
            return

        if self._on_acp_message is not None:
            await self._on_acp_message(msg)

    async def _read_lsp_message(self, reader: asyncio.StreamReader) -> None:
        """Read a single LSP-framed message and dispatch."""
        try:
            msg = await lsp_read_message(reader)
        except LSPFramingError as exc:
            logger.warning("LSP framing error: %s", exc)
            return

        if msg is None:
            return  # EOF

        if self._on_lsp_message is not None:
            await self._on_lsp_message(msg)
