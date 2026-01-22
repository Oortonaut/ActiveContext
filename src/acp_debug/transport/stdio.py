"""Stdio JSON-RPC transport."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JsonRpcMessage:
    """Parsed JSON-RPC message."""

    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any = None
    error: dict[str, Any] | None = None

    def is_request(self) -> bool:
        """Check if this is a request (has method and id)."""
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        """Check if this is a notification (has method but no id)."""
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        """Check if this is a response (has result or error)."""
        return self.result is not None or self.error is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.id is not None:
            d["id"] = self.id
        if self.method is not None:
            d["method"] = self.method
        if self.params is not None:
            d["params"] = self.params
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JsonRpcMessage:
        """Parse from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class StdioTransport:
    """Async JSON-RPC transport over stdio.

    Handles newline-delimited JSON messages over stdin/stdout.
    """

    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    _read_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    async def from_stdio(cls) -> StdioTransport:
        """Create transport from stdin/stdout."""
        loop = asyncio.get_event_loop()

        # Create reader for stdin
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        # Create writer for stdout
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

        return cls(reader=reader, writer=writer)

    @classmethod
    async def from_process(
        cls,
        process: asyncio.subprocess.Process,
    ) -> StdioTransport:
        """Create transport from subprocess stdin/stdout."""
        if process.stdin is None or process.stdout is None:
            raise ValueError("Process must have stdin and stdout pipes")

        # Wrap process streams
        return cls(
            reader=process.stdout,
            writer=process.stdin,
        )

    async def read_message(self) -> JsonRpcMessage | None:
        """Read a single JSON-RPC message.

        Returns None on EOF.
        """
        if self.reader is None:
            return None

        async with self._read_lock:
            try:
                line = await self.reader.readline()
                if not line:
                    return None

                data = json.loads(line.decode("utf-8"))
                return JsonRpcMessage.from_dict(data)

            except json.JSONDecodeError:
                return None
            except Exception:
                return None

    async def write_message(self, msg: JsonRpcMessage) -> None:
        """Write a JSON-RPC message."""
        if self.writer is None:
            return

        async with self._write_lock:
            data = json.dumps(msg.to_dict(), separators=(",", ":"))
            self.writer.write(f"{data}\n".encode())
            await self.writer.drain()

    async def messages(self) -> AsyncIterator[JsonRpcMessage]:
        """Iterate over incoming messages until EOF."""
        while True:
            msg = await self.read_message()
            if msg is None:
                break
            yield msg

    async def close(self) -> None:
        """Close the transport."""
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except Exception:
                pass
