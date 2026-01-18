"""FastAPI routes for dashboard REST API and WebSocket."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from activecontext.dashboard.data import (
    format_session_update,
    get_context_data,
    get_conversation_data,
    get_llm_status,
    get_projection_data,
    get_rendered_projection_data,
    get_session_summary,
    get_timeline_data,
)
from activecontext.dashboard.server import (
    get_current_model_id,
    get_dashboard_status,
    get_manager,
    get_session_mode,
    get_session_model,
    get_static_dir,
)
from activecontext.dashboard.websocket import ConnectionManager

log = logging.getLogger(__name__)

# Global WebSocket connection manager
connection_manager = ConnectionManager()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ActiveContext Dashboard",
        description="Real-time monitoring dashboard for ActiveContext sessions",
        version="0.1.0",
    )

    # Mount static files
    static_dir = get_static_dir()
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    @app.get("/")
    async def index() -> FileResponse:
        """Serve the dashboard HTML page."""
        static_dir = get_static_dir()
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(index_path)

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        """Get dashboard health status."""
        status = get_dashboard_status()
        return {
            "status": "ok",
            "uptime": status["uptime"],
            "connections": status["connections"],
        }

    @app.get("/api/llm")
    async def api_llm() -> dict[str, Any]:
        """Get LLM provider and model status."""
        current_model = get_current_model_id()
        return get_llm_status(current_model)

    @app.get("/api/sessions")
    async def api_sessions() -> list[dict[str, Any]]:
        """List all active sessions."""
        manager = get_manager()
        if not manager:
            return []

        sessions = await manager.list_sessions()
        result: list[dict[str, Any]] = []

        for session_id in sessions:
            session = await manager.get_session(session_id)
            if session:
                model = get_session_model(session_id)
                mode = get_session_mode(session_id)
                result.append(get_session_summary(session, model, mode))

        return result

    @app.get("/api/sessions/{session_id}/context")
    async def api_session_context(session_id: str) -> dict[str, Any]:
        """Get context objects for a session."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return get_context_data(session)

    @app.get("/api/sessions/{session_id}/timeline")
    async def api_session_timeline(session_id: str) -> dict[str, Any]:
        """Get statement execution timeline for a session."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return get_timeline_data(session)

    @app.get("/api/sessions/{session_id}/projection")
    async def api_session_projection(session_id: str) -> dict[str, Any]:
        """Get token budget and projection breakdown for a session."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return get_projection_data(session)

    @app.get("/api/sessions/{session_id}/conversation")
    async def api_session_conversation(session_id: str) -> dict[str, Any]:
        """Get conversation messages for a session."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return get_conversation_data(session)

    @app.get("/api/sessions/{session_id}/rendered")
    async def api_session_rendered(session_id: str) -> dict[str, Any]:
        """Get the full rendered projection content sent to the LLM."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return get_rendered_projection_data(session)

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
        """WebSocket endpoint for real-time session updates."""
        await connection_manager.connect(websocket, session_id)

        try:
            # Send initial state with error handling
            manager = get_manager()
            if manager:
                session = await manager.get_session(session_id)
                if session:
                    try:
                        init_data = {
                            "type": "init",
                            "session_id": session_id,
                            "context": get_context_data(session),
                            "timeline": get_timeline_data(session),
                            "projection": get_projection_data(session),
                            "conversation": get_conversation_data(session),
                            "rendered": get_rendered_projection_data(session),
                        }
                        await websocket.send_json(init_data)
                    except Exception as e:
                        log.exception(f"Failed to send initial state for session {session_id}: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Failed to load session data: {e}",
                        })

            # Keep connection alive and handle incoming messages
            while True:
                try:
                    # Wait for messages (pings, etc.)
                    data = await websocket.receive_text()
                    # Handle ping/pong
                    if data == "ping":
                        await websocket.send_text("pong")
                except WebSocketDisconnect:
                    break

        finally:
            await connection_manager.disconnect(websocket, session_id)


async def broadcast_update(
    session_id: str,
    kind: str,
    payload: dict[str, Any],
    timestamp: float,
) -> None:
    """Broadcast a session update to all connected dashboard clients.

    This function is called from the agent when session state changes.
    """
    message = format_session_update(kind, session_id, payload, timestamp)
    await connection_manager.broadcast(session_id, message)
