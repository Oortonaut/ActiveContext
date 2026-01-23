"""Test utilities for ACP protocol tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ACPTestClient:
    """Test client for sending JSON-RPC messages to agent.

    Uses inline response reading for simplicity and pytest-asyncio compatibility.
    """

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    _next_id: int = field(default=0, init=False)
    _notifications: list[dict] = field(default_factory=list, init=False)
    _read_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def start_handler(self) -> None:
        """No-op for API compatibility. Response reading is done inline."""
        pass

    async def _write_raw(self, msg: dict) -> None:
        """Write raw message to agent."""
        data = json.dumps(msg, separators=(",", ":"))
        self.writer.write(f"{data}\n".encode())
        await self.writer.drain()

    async def _read_until_response(self, request_id: int | str, timeout: float) -> dict[str, Any]:
        """Read messages until we get the response for request_id.

        Collects notifications along the way and auto-responds to agent requests.
        """
        deadline = asyncio.get_event_loop().time() + timeout

        async with self._read_lock:
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Timeout waiting for response to request {request_id}")

                try:
                    line = await asyncio.wait_for(self.reader.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Timeout waiting for response to request {request_id}")

                if not line:
                    raise ConnectionError("Agent closed connection")

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue  # Skip malformed messages

                # Check if this is our response
                if msg.get("id") == request_id:
                    return msg

                # Handle notifications
                if "method" in msg and "id" not in msg:
                    self._notifications.append(msg)
                    continue

                # Handle agent requests (e.g., permission requests) - auto-approve
                if "method" in msg and "id" in msg:
                    response = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
                    await self._write_raw(response)
                    continue

                # Response for different ID - shouldn't happen in simple tests
                continue

    async def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: int | str | None = None,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response.

        Args:
            method: RPC method name
            params: Optional parameters
            request_id: Optional custom request ID
            timeout: Response timeout in seconds

        Returns:
            Full JSON-RPC response dict
        """
        if request_id is None:
            request_id = self._next_id
            self._next_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            request["params"] = params

        await self._write_raw(request)
        return await self._read_until_response(request_id, timeout)

    async def send_notification(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        notification = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notification["params"] = params

        await self._write_raw(notification)

    async def send_raw(self, data: str) -> None:
        """Send raw data (for testing parse errors)."""
        self.writer.write(f"{data}\n".encode())
        await self.writer.drain()

    def get_notifications(self) -> list[dict]:
        """Get list of received notifications."""
        return list(self._notifications)

    def clear_notifications(self) -> None:
        """Clear notification buffer."""
        self._notifications.clear()

    async def close(self) -> None:
        """Close the client."""
        pass  # No background tasks to cancel


async def initialize_agent(client: ACPTestClient) -> dict[str, Any]:
    """Send initialize request with standard client info.

    Returns:
        Full JSON-RPC response dict
    """
    response = await client.send_request(
        "initialize",
        {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": True, "writeTextFile": True},
                "terminal": True,
            },
            "clientInfo": {
                "name": "pytest",
                "title": "ACP Protocol Tests",
                "version": "1.0.0",
            },
        },
    )
    return response
