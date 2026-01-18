"""WebSocket connection manager for real-time dashboard updates."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

log = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for dashboard clients.

    Tracks connections per session_id and provides broadcast functionality.
    """

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, session_id: str) -> None:
        """Accept a new WebSocket connection for a session."""
        await websocket.accept()
        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = set()
            self._connections[session_id].add(websocket)
        log.debug(f"Dashboard client connected for session {session_id}")

    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if session_id in self._connections:
                self._connections[session_id].discard(websocket)
                if not self._connections[session_id]:
                    del self._connections[session_id]
        log.debug(f"Dashboard client disconnected from session {session_id}")

    async def broadcast(self, session_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all clients watching a session."""
        async with self._lock:
            connections = self._connections.get(session_id, set()).copy()

        if not connections:
            return

        dead_connections: list[WebSocket] = []
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception:
                dead_connections.append(websocket)

        # Clean up dead connections
        if dead_connections:
            async with self._lock:
                for ws in dead_connections:
                    if session_id in self._connections:
                        self._connections[session_id].discard(ws)

    async def broadcast_all(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        async with self._lock:
            all_sessions = list(self._connections.keys())

        for session_id in all_sessions:
            await self.broadcast(session_id, message)

    def get_connection_count(self, session_id: str | None = None) -> int:
        """Get the number of active connections."""
        if session_id:
            return len(self._connections.get(session_id, set()))
        return sum(len(conns) for conns in self._connections.values())

    async def close_all(self, reason: str = "Server shutting down") -> None:
        """Close all WebSocket connections gracefully."""
        async with self._lock:
            all_connections = [
                (session_id, ws)
                for session_id, conns in self._connections.items()
                for ws in conns
            ]
            self._connections.clear()

        for _session_id, websocket in all_connections:
            with contextlib.suppress(Exception):
                await websocket.close(code=1001, reason=reason)
        log.info(f"Closed {len(all_connections)} dashboard connections")
