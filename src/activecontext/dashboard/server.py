"""Dashboard web server lifecycle management."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.session.session_manager import SessionManager

log = logging.getLogger(__name__)

# Server state
_server_task: asyncio.Task[None] | None = None
_server_port: int | None = None
_start_time: float | None = None

# Shared state for routes
_manager: SessionManager | None = None
_get_current_model: Callable[[], str | None] | None = None
_sessions_model: dict[str, str] | None = None
_sessions_mode: dict[str, str] | None = None


def get_manager() -> SessionManager | None:
    """Get the shared SessionManager instance."""
    return _manager


def get_current_model_id() -> str | None:
    """Get the current model ID."""
    if _get_current_model:
        return _get_current_model()
    return None


def get_session_model(session_id: str) -> str | None:
    """Get the model for a specific session."""
    if _sessions_model:
        return _sessions_model.get(session_id)
    return None


def get_session_mode(session_id: str) -> str | None:
    """Get the mode for a specific session."""
    if _sessions_mode:
        return _sessions_mode.get(session_id)
    return None


def is_dashboard_running() -> bool:
    """Check if the dashboard server is running."""
    return _server_task is not None and not _server_task.done()


def get_dashboard_status() -> dict[str, Any]:
    """Get dashboard server status."""
    # Import the connection manager from routes if available
    try:
        from activecontext.dashboard.routes import connection_manager

        connections = connection_manager.get_connection_count()
    except Exception:
        connections = 0

    return {
        "running": is_dashboard_running(),
        "port": _server_port,
        "uptime": time.time() - _start_time if _start_time else 0,
        "connections": connections,
    }


async def start_dashboard(
    port: int,
    manager: SessionManager,
    get_current_model: Callable[[], str | None],
    sessions_model: dict[str, str],
    sessions_mode: dict[str, str],
) -> None:
    """Start the dashboard web server.

    Args:
        port: Port to listen on
        manager: SessionManager instance for data access
        get_current_model: Callable to get current model ID
        sessions_model: Dict mapping session_id -> model_id
        sessions_mode: Dict mapping session_id -> mode
    """
    global _server_task, _server_port, _start_time
    global _manager, _get_current_model, _sessions_model, _sessions_mode

    if is_dashboard_running():
        raise RuntimeError(f"Dashboard already running on port {_server_port}")

    # Store shared state
    _manager = manager
    _get_current_model = get_current_model
    _sessions_model = sessions_model
    _sessions_mode = sessions_mode

    # Import here to avoid circular imports and startup overhead
    import uvicorn

    from activecontext.dashboard.routes import create_app

    app = create_app()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    # Start server in background task
    _server_task = asyncio.create_task(server.serve())
    _server_port = port
    _start_time = time.time()

    log.info(f"Dashboard started on http://127.0.0.1:{port}")


async def stop_dashboard() -> None:
    """Stop the dashboard web server."""
    global _server_task, _server_port, _start_time
    global _manager, _get_current_model, _sessions_model, _sessions_mode

    if not _server_task:
        return

    # Close all WebSocket connections
    try:
        from activecontext.dashboard.routes import connection_manager

        await connection_manager.close_all("Server shutting down")
    except Exception:
        pass

    # Cancel the server task
    _server_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _server_task

    log.info(f"Dashboard stopped (was on port {_server_port})")

    # Clear state
    _server_task = None
    _server_port = None
    _start_time = None
    _manager = None
    _get_current_model = None
    _sessions_model = None
    _sessions_mode = None


def get_static_dir() -> Path:
    """Get the path to the static files directory."""
    return Path(__file__).parent / "static"
