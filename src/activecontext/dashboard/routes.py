"""FastAPI routes for dashboard REST API and WebSocket."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from activecontext.context.state import NodeState
from activecontext.dashboard.data import (
    format_session_update,
    get_client_capabilities_data,
    get_context_data,
    get_llm_status,
    get_message_history_data,
    get_projection_data,
    get_rendered_projection_data,
    get_session_features_data,
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

    @app.get("/api/sessions/{session_id}/message-history")
    async def api_session_message_history(session_id: str) -> dict[str, Any]:
        """Get message history for a session."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        return get_message_history_data(session)

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

    @app.get("/api/client")
    async def api_client() -> dict[str, Any]:
        """Get ACP client capabilities and transport info."""
        return get_client_capabilities_data()

    @app.get("/api/sessions/{session_id}/features")
    async def api_session_features(session_id: str) -> dict[str, Any]:
        """Get session features including mode, model, and permissions."""
        manager = get_manager()
        if not manager:
            raise HTTPException(status_code=503, detail="Dashboard not initialized")

        session = await manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        model = get_session_model(session_id)
        mode = get_session_mode(session_id)
        return get_session_features_data(session, model, mode)

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
                        model = get_session_model(session_id)
                        mode = get_session_mode(session_id)
                        init_data = {
                            "type": "init",
                            "session_id": session_id,
                            "client": get_client_capabilities_data(),
                            "features": get_session_features_data(session, model, mode),
                            "context": get_context_data(session),
                            "timeline": get_timeline_data(session),
                            "projection": get_projection_data(session),
                            "message_history": get_message_history_data(session),
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
                    # Wait for messages (pings, commands, etc.)
                    data = await websocket.receive_text()
                    # Handle ping/pong
                    if data == "ping":
                        await websocket.send_text("pong")
                    else:
                        # Try to parse as JSON command
                        try:
                            cmd = json.loads(data)
                            await _handle_websocket_command(websocket, session_id, cmd)
                        except json.JSONDecodeError:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Invalid JSON",
                            })
                except WebSocketDisconnect:
                    break

        finally:
            await connection_manager.disconnect(websocket, session_id)


async def _handle_websocket_command(
    websocket: WebSocket,
    session_id: str,
    cmd: dict[str, Any],
) -> None:
    """Handle incoming WebSocket commands."""
    cmd_type = cmd.get("type")

    if cmd_type == "set_state":
        await _handle_set_state(websocket, session_id, cmd)
    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown command type: {cmd_type}",
        })


async def _handle_set_state(
    websocket: WebSocket,
    session_id: str,
    cmd: dict[str, Any],
) -> None:
    """Handle node state change request."""
    manager = get_manager()
    if not manager:
        await websocket.send_json({
            "type": "error",
            "message": "Dashboard not initialized",
        })
        return

    session = await manager.get_session(session_id)
    if not session:
        await websocket.send_json({
            "type": "error",
            "message": f"Session not found: {session_id}",
        })
        return

    node_id = cmd.get("node_id")
    new_state_str = cmd.get("state")

    if not node_id or not new_state_str:
        await websocket.send_json({
            "type": "error",
            "message": "Missing node_id or state",
        })
        return

    # Validate state
    try:
        new_state = NodeState(new_state_str)
    except ValueError:
        valid_states = ", ".join(s.value for s in NodeState)
        await websocket.send_json({
            "type": "error",
            "message": f"Invalid state: {new_state_str}. Valid: {valid_states}",
        })
        return

    # Get node and apply state change
    graph = session.get_context_graph()
    node = graph.get_node(node_id)

    if not node:
        await websocket.send_json({
            "type": "error",
            "message": f"Node not found: {node_id}",
        })
        return

    old_state = node.state
    node.SetState(new_state)

    # Broadcast update to all dashboard clients
    await broadcast_update(
        session_id,
        "node_changed",
        {
            "node_id": node_id,
            "node_type": node.node_type,
            "change": "state_changed",
            "old_state": old_state.value,
            "new_state": new_state.value,
            "digest": node.GetDigest(),
        },
        time.time(),
    )

    # Confirm to requesting client
    await websocket.send_json({
        "type": "state_changed",
        "node_id": node_id,
        "old_state": old_state.value,
        "new_state": new_state.value,
    })


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
