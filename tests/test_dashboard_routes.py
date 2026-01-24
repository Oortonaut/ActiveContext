"""Comprehensive tests for dashboard routes module."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def mock_session():
    """Create a mock session object."""
    session = MagicMock()
    session.session_id = "session-1"
    session.cwd = Path("/test/path")
    session.title = "Test Session"

    # Mock LLM
    llm = MagicMock()
    llm.model = "claude-3-opus"
    session.llm = llm

    # Mock timeline
    timeline = MagicMock()
    timeline.get_statements.return_value = []
    session.timeline = timeline

    # Mock context graph
    graph = MagicMock()
    graph.__iter__ = MagicMock(return_value=iter([]))
    graph.get_node.return_value = None
    session.get_context_graph.return_value = graph

    # Mock projection
    projection = MagicMock()
    projection.sections = []
    projection.render.return_value = "Rendered content"
    session.get_projection.return_value = projection

    # Mock message history
    session._message_history = []

    return session


@pytest.fixture
def mock_manager(mock_session):
    """Create a mock session manager."""
    manager = MagicMock()
    manager.list_sessions = AsyncMock(return_value=["session-1", "session-2"])
    manager.get_session = AsyncMock(return_value=mock_session)
    return manager


@pytest.fixture
def mock_context_data():
    """Mock context data response."""
    return {
        "nodes_by_type": {
            "text": [{"id": "node-1", "type": "text", "path": "main.py"}],
        },
        "total": 1,
    }


@pytest.fixture
def mock_timeline_data():
    """Mock timeline data response."""
    return {
        "statements": [
            {
                "statement_id": "stmt-1",
                "index": 0,
                "source": "v = text('main.py')",
                "timestamp": time.time(),
                "status": "completed",
                "duration_ms": 100,
                "has_error": False,
            }
        ],
        "count": 1,
    }


@pytest.fixture
def mock_projection_data():
    """Mock projection data response."""
    return {
        "total_used": 500,
        "sections": [
            {
                "type": "context",
                "source_id": "node-1",
                "tokens_used": 500,
                "state": "all",
            }
        ],
    }


@pytest.fixture
def mock_message_history_data():
    """Mock message history response."""
    return {
        "messages": [
            {"id": "msg_0", "role": "user", "content": "Hello"},
            {"id": "msg_1", "role": "assistant", "content": "Hi there!"},
        ],
        "count": 2,
    }


@pytest.fixture
def mock_rendered_projection_data():
    """Mock rendered projection response."""
    return {
        "rendered": "# Context\n\nRendered content here",
        "total_tokens": 100,
        "sections": [],
        "section_count": 0,
    }


@pytest.fixture
def mock_client_capabilities_data():
    """Mock client capabilities response."""
    return {
        "transport": {"type": "direct", "is_acp": False},
        "protocol_version": None,
        "client": None,
    }


@pytest.fixture
def mock_session_features_data():
    """Mock session features response."""
    return {
        "model": "claude-3-opus",
        "mode": "normal",
        "cwd": "/test/path",
        "title": "Test Session",
        "transport": {"type": "direct", "is_acp": False},
        "protocol_version": None,
        "client": None,
    }


@pytest.fixture
def mock_llm_status():
    """Mock LLM status response."""
    return {
        "current_model": "claude-3-opus",
        "available_providers": ["anthropic"],
        "available_models": [
            {
                "model_id": "claude-3-opus",
                "name": "Claude 3 Opus",
                "provider": "anthropic",
                "description": "Most capable model",
            }
        ],
        "configured": True,
    }


@pytest.fixture
def mock_session_summary():
    """Mock session summary response."""
    return {
        "session_id": "session-1",
        "cwd": "/test/path",
        "model": "claude-3-opus",
        "mode": "normal",
    }


@pytest.fixture
def mock_dashboard_status():
    """Mock dashboard status response."""
    return {
        "running": True,
        "port": 8080,
        "uptime": 100.0,
        "connections": 0,
    }


# ============================================================================
# App Fixture with All Mocks
# ============================================================================


@pytest.fixture
def app_with_mocks(
    mock_manager,
    mock_session,
    mock_context_data,
    mock_timeline_data,
    mock_projection_data,
    mock_message_history_data,
    mock_rendered_projection_data,
    mock_client_capabilities_data,
    mock_session_features_data,
    mock_llm_status,
    mock_session_summary,
    mock_dashboard_status,
):
    """Create FastAPI app with all necessary mocks configured."""
    # Create static dir mock that reports as not existing (to skip static mount)
    mock_static_dir = MagicMock(spec=Path)
    mock_static_dir.exists.return_value = False
    mock_index = MagicMock()
    mock_index.exists.return_value = False
    mock_static_dir.__truediv__ = lambda self, x: mock_index

    with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
         patch("activecontext.dashboard.routes.get_dashboard_status") as mock_get_status, \
         patch("activecontext.dashboard.routes.get_current_model_id") as mock_get_model, \
         patch("activecontext.dashboard.routes.get_session_model") as mock_get_session_model, \
         patch("activecontext.dashboard.routes.get_session_mode") as mock_get_session_mode, \
         patch("activecontext.dashboard.routes.get_static_dir") as mock_get_static, \
         patch("activecontext.dashboard.routes.get_context_data") as mock_get_context, \
         patch("activecontext.dashboard.routes.get_timeline_data") as mock_get_timeline, \
         patch("activecontext.dashboard.routes.get_projection_data") as mock_get_projection, \
         patch("activecontext.dashboard.routes.get_message_history_data") as mock_get_messages, \
         patch("activecontext.dashboard.routes.get_rendered_projection_data") as mock_get_rendered, \
         patch("activecontext.dashboard.routes.get_client_capabilities_data") as mock_get_client, \
         patch("activecontext.dashboard.routes.get_session_features_data") as mock_get_features, \
         patch("activecontext.dashboard.routes.get_llm_status") as mock_get_llm, \
         patch("activecontext.dashboard.routes.get_session_summary") as mock_get_summary:

        # Configure mocks
        mock_get_manager.return_value = mock_manager
        mock_get_status.return_value = mock_dashboard_status
        mock_get_model.return_value = "claude-3-opus"
        mock_get_session_model.return_value = "claude-3-opus"
        mock_get_session_mode.return_value = "normal"
        mock_get_static.return_value = mock_static_dir
        mock_get_context.return_value = mock_context_data
        mock_get_timeline.return_value = mock_timeline_data
        mock_get_projection.return_value = mock_projection_data
        mock_get_messages.return_value = mock_message_history_data
        mock_get_rendered.return_value = mock_rendered_projection_data
        mock_get_client.return_value = mock_client_capabilities_data
        mock_get_features.return_value = mock_session_features_data
        mock_get_llm.return_value = mock_llm_status
        mock_get_summary.return_value = mock_session_summary

        # Store mocks for test access
        mocks = {
            "get_manager": mock_get_manager,
            "get_dashboard_status": mock_get_status,
            "get_current_model_id": mock_get_model,
            "get_session_model": mock_get_session_model,
            "get_session_mode": mock_get_session_mode,
            "get_static_dir": mock_get_static,
            "get_context_data": mock_get_context,
            "get_timeline_data": mock_get_timeline,
            "get_projection_data": mock_get_projection,
            "get_message_history_data": mock_get_messages,
            "get_rendered_projection_data": mock_get_rendered,
            "get_client_capabilities_data": mock_get_client,
            "get_session_features_data": mock_get_features,
            "get_llm_status": mock_get_llm,
            "get_session_summary": mock_get_summary,
            "manager": mock_manager,
            "session": mock_session,
        }

        from activecontext.dashboard.routes import create_app
        app = create_app()

        yield app, mocks


@pytest.fixture
def client(app_with_mocks):
    """Create test client with mocks attached."""
    app, mocks = app_with_mocks
    with TestClient(app) as client:
        client.mocks = mocks  # type: ignore
        yield client


# ============================================================================
# Tests for create_app()
# ============================================================================


class TestCreateApp:
    """Tests for create_app() function."""

    def test_create_app_returns_fastapi_instance(self):
        """create_app should return a FastAPI application."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()
            assert isinstance(app, FastAPI)

    def test_create_app_has_title_and_version(self):
        """create_app should set title and version."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()
            assert app.title == "ActiveContext Dashboard"
            assert app.version == "0.1.0"

    def test_create_app_has_api_routes(self):
        """create_app should configure API routes."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            route_paths = [route.path for route in app.routes]
            assert "/" in route_paths
            assert "/api/status" in route_paths
            assert "/api/sessions" in route_paths
            assert "/api/llm" in route_paths
            assert "/api/client" in route_paths

    def test_create_app_has_session_routes(self):
        """create_app should configure session-specific routes."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            route_paths = [route.path for route in app.routes]
            assert "/api/sessions/{session_id}/context" in route_paths
            assert "/api/sessions/{session_id}/timeline" in route_paths
            assert "/api/sessions/{session_id}/projection" in route_paths
            assert "/api/sessions/{session_id}/message-history" in route_paths
            assert "/api/sessions/{session_id}/rendered" in route_paths
            assert "/api/sessions/{session_id}/features" in route_paths

    def test_create_app_has_websocket_route(self):
        """create_app should configure WebSocket route."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            route_paths = [route.path for route in app.routes]
            assert "/ws/{session_id}" in route_paths

    def test_create_app_mounts_static_files_when_dir_exists(self):
        """create_app should mount static files if directory exists."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = True
            mock_dir.__str__ = lambda x: "/path/to/static"
            mock_static.return_value = mock_dir

            with patch("activecontext.dashboard.routes.StaticFiles"):
                from activecontext.dashboard.routes import create_app
                app = create_app()

                # Check static mount exists
                route_paths = [route.path for route in app.routes]
                assert "/static" in route_paths or any("/static" in str(r) for r in app.routes)


# ============================================================================
# Tests for GET / (index.html)
# ============================================================================


class TestIndexRoute:
    """Tests for GET / endpoint."""

    def test_index_returns_404_when_no_index_html(self):
        """GET / should return 404 if index.html doesn't exist."""
        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_index = MagicMock()
            mock_index.exists.return_value = False
            mock_dir.__truediv__ = lambda self, x: mock_index
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/")
                assert response.status_code == 404
                assert "Dashboard not found" in response.json()["detail"]

    def test_index_returns_200_when_index_html_exists(self, tmp_path):
        """GET / should return index.html content when file exists."""
        # Create a temp index.html
        index_content = "<html><body>Dashboard</body></html>"
        index_file = tmp_path / "index.html"
        index_file.write_text(index_content)

        with patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_static.return_value = tmp_path

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/")
                assert response.status_code == 200
                assert "Dashboard" in response.text


# ============================================================================
# Tests for GET /api/status
# ============================================================================


class TestStatusEndpoint:
    """Tests for GET /api/status endpoint."""

    def test_status_returns_200(self, client):
        """GET /api/status should return 200 OK."""
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_status_returns_json(self, client):
        """GET /api/status should return JSON response."""
        response = client.get("/api/status")
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert isinstance(data, dict)

    def test_status_contains_required_fields(self, client, mock_dashboard_status):
        """GET /api/status should contain status, uptime, connections."""
        response = client.get("/api/status")
        data = response.json()

        assert "status" in data
        assert data["status"] == "ok"
        assert "uptime" in data
        assert data["uptime"] == mock_dashboard_status["uptime"]
        assert "connections" in data
        assert data["connections"] == mock_dashboard_status["connections"]

    def test_status_calls_get_dashboard_status(self, client):
        """GET /api/status should call get_dashboard_status()."""
        response = client.get("/api/status")
        assert response.status_code == 200
        client.mocks["get_dashboard_status"].assert_called_once()


# ============================================================================
# Tests for GET /api/llm
# ============================================================================


class TestLLMEndpoint:
    """Tests for GET /api/llm endpoint."""

    def test_llm_returns_200(self, client):
        """GET /api/llm should return 200 OK."""
        response = client.get("/api/llm")
        assert response.status_code == 200

    def test_llm_returns_status(self, client, mock_llm_status):
        """GET /api/llm should return LLM status."""
        response = client.get("/api/llm")
        data = response.json()

        assert data["current_model"] == mock_llm_status["current_model"]
        assert data["available_providers"] == mock_llm_status["available_providers"]
        assert data["configured"] == mock_llm_status["configured"]

    def test_llm_calls_get_current_model_id(self, client):
        """GET /api/llm should call get_current_model_id()."""
        response = client.get("/api/llm")
        assert response.status_code == 200
        client.mocks["get_current_model_id"].assert_called_once()

    def test_llm_calls_get_llm_status(self, client):
        """GET /api/llm should call get_llm_status() with model."""
        response = client.get("/api/llm")
        assert response.status_code == 200
        client.mocks["get_llm_status"].assert_called_once_with("claude-3-opus")


# ============================================================================
# Tests for GET /api/sessions
# ============================================================================


class TestSessionsListEndpoint:
    """Tests for GET /api/sessions endpoint."""

    def test_sessions_returns_200(self, client):
        """GET /api/sessions should return 200 OK."""
        response = client.get("/api/sessions")
        assert response.status_code == 200

    def test_sessions_returns_list(self, client):
        """GET /api/sessions should return a list."""
        response = client.get("/api/sessions")
        data = response.json()
        assert isinstance(data, list)

    def test_sessions_returns_empty_when_no_manager(self):
        """GET /api/sessions should return empty list when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions")
                assert response.status_code == 200
                assert response.json() == []

    def test_sessions_uses_manager_list_sessions(self, client):
        """GET /api/sessions should use manager.list_sessions()."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        client.mocks["manager"].list_sessions.assert_called()


# ============================================================================
# Tests for GET /api/sessions/{id}/context
# ============================================================================


class TestSessionContextEndpoint:
    """Tests for GET /api/sessions/{id}/context endpoint."""

    def test_context_returns_200(self, client):
        """GET /api/sessions/{id}/context should return 200 OK."""
        response = client.get("/api/sessions/session-1/context")
        assert response.status_code == 200

    def test_context_returns_context_data(self, client, mock_context_data):
        """GET /api/sessions/{id}/context should return context data."""
        response = client.get("/api/sessions/session-1/context")
        data = response.json()

        assert "nodes_by_type" in data
        assert "total" in data
        assert data["total"] == mock_context_data["total"]

    def test_context_calls_get_context_data(self, client):
        """GET /api/sessions/{id}/context should call get_context_data()."""
        response = client.get("/api/sessions/session-1/context")
        assert response.status_code == 200
        client.mocks["get_context_data"].assert_called_once()

    def test_context_returns_503_when_no_manager(self):
        """GET /api/sessions/{id}/context should return 503 when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/session-1/context")
                assert response.status_code == 503
                assert "Dashboard not initialized" in response.json()["detail"]

    def test_context_returns_404_when_session_not_found(self):
        """GET /api/sessions/{id}/context should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/nonexistent/context")
                assert response.status_code == 404
                assert "not found" in response.json()["detail"]


# ============================================================================
# Tests for GET /api/sessions/{id}/timeline
# ============================================================================


class TestSessionTimelineEndpoint:
    """Tests for GET /api/sessions/{id}/timeline endpoint."""

    def test_timeline_returns_200(self, client):
        """GET /api/sessions/{id}/timeline should return 200 OK."""
        response = client.get("/api/sessions/session-1/timeline")
        assert response.status_code == 200

    def test_timeline_returns_timeline_data(self, client, mock_timeline_data):
        """GET /api/sessions/{id}/timeline should return timeline data."""
        response = client.get("/api/sessions/session-1/timeline")
        data = response.json()

        assert "statements" in data
        assert "count" in data
        assert data["count"] == mock_timeline_data["count"]

    def test_timeline_calls_get_timeline_data(self, client):
        """GET /api/sessions/{id}/timeline should call get_timeline_data()."""
        response = client.get("/api/sessions/session-1/timeline")
        assert response.status_code == 200
        client.mocks["get_timeline_data"].assert_called_once()

    def test_timeline_returns_503_when_no_manager(self):
        """GET /api/sessions/{id}/timeline should return 503 when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/session-1/timeline")
                assert response.status_code == 503

    def test_timeline_returns_404_when_session_not_found(self):
        """GET /api/sessions/{id}/timeline should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/nonexistent/timeline")
                assert response.status_code == 404


# ============================================================================
# Tests for GET /api/sessions/{id}/projection
# ============================================================================


class TestSessionProjectionEndpoint:
    """Tests for GET /api/sessions/{id}/projection endpoint."""

    def test_projection_returns_200(self, client):
        """GET /api/sessions/{id}/projection should return 200 OK."""
        response = client.get("/api/sessions/session-1/projection")
        assert response.status_code == 200

    def test_projection_returns_projection_data(self, client, mock_projection_data):
        """GET /api/sessions/{id}/projection should return projection data."""
        response = client.get("/api/sessions/session-1/projection")
        data = response.json()

        assert "total_used" in data
        assert "sections" in data
        assert data["total_used"] == mock_projection_data["total_used"]

    def test_projection_calls_get_projection_data(self, client):
        """GET /api/sessions/{id}/projection should call get_projection_data()."""
        response = client.get("/api/sessions/session-1/projection")
        assert response.status_code == 200
        client.mocks["get_projection_data"].assert_called_once()

    def test_projection_returns_503_when_no_manager(self):
        """GET /api/sessions/{id}/projection should return 503 when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/session-1/projection")
                assert response.status_code == 503

    def test_projection_returns_404_when_session_not_found(self):
        """GET /api/sessions/{id}/projection should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/nonexistent/projection")
                assert response.status_code == 404


# ============================================================================
# Tests for GET /api/sessions/{id}/message-history
# ============================================================================


class TestSessionMessageHistoryEndpoint:
    """Tests for GET /api/sessions/{id}/message-history endpoint."""

    def test_message_history_returns_200(self, client):
        """GET /api/sessions/{id}/message-history should return 200 OK."""
        response = client.get("/api/sessions/session-1/message-history")
        assert response.status_code == 200

    def test_message_history_returns_messages(self, client, mock_message_history_data):
        """GET /api/sessions/{id}/message-history should return message list."""
        response = client.get("/api/sessions/session-1/message-history")
        data = response.json()

        assert "messages" in data
        assert "count" in data
        assert data["count"] == mock_message_history_data["count"]

    def test_message_history_calls_get_message_history_data(self, client):
        """GET /api/sessions/{id}/message-history should call get_message_history_data()."""
        response = client.get("/api/sessions/session-1/message-history")
        assert response.status_code == 200
        client.mocks["get_message_history_data"].assert_called_once()

    def test_message_history_returns_503_when_no_manager(self):
        """GET /api/sessions/{id}/message-history should return 503 when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/session-1/message-history")
                assert response.status_code == 503

    def test_message_history_returns_404_when_session_not_found(self):
        """GET /api/sessions/{id}/message-history should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/nonexistent/message-history")
                assert response.status_code == 404


# ============================================================================
# Tests for GET /api/sessions/{id}/rendered
# ============================================================================


class TestSessionRenderedEndpoint:
    """Tests for GET /api/sessions/{id}/rendered endpoint."""

    def test_rendered_returns_200(self, client):
        """GET /api/sessions/{id}/rendered should return 200 OK."""
        response = client.get("/api/sessions/session-1/rendered")
        assert response.status_code == 200

    def test_rendered_returns_projection_content(self, client, mock_rendered_projection_data):
        """GET /api/sessions/{id}/rendered should return rendered projection."""
        response = client.get("/api/sessions/session-1/rendered")
        data = response.json()

        assert "rendered" in data
        assert "total_tokens" in data
        assert "sections" in data
        assert "section_count" in data

    def test_rendered_calls_get_rendered_projection_data(self, client):
        """GET /api/sessions/{id}/rendered should call get_rendered_projection_data()."""
        response = client.get("/api/sessions/session-1/rendered")
        assert response.status_code == 200
        client.mocks["get_rendered_projection_data"].assert_called_once()

    def test_rendered_returns_503_when_no_manager(self):
        """GET /api/sessions/{id}/rendered should return 503 when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/session-1/rendered")
                assert response.status_code == 503

    def test_rendered_returns_404_when_session_not_found(self):
        """GET /api/sessions/{id}/rendered should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/nonexistent/rendered")
                assert response.status_code == 404


# ============================================================================
# Tests for GET /api/client
# ============================================================================


class TestClientEndpoint:
    """Tests for GET /api/client endpoint."""

    def test_client_returns_200(self, client):
        """GET /api/client should return 200 OK."""
        response = client.get("/api/client")
        assert response.status_code == 200

    def test_client_returns_capabilities(self, client, mock_client_capabilities_data):
        """GET /api/client should return client capabilities."""
        response = client.get("/api/client")
        data = response.json()

        assert "transport" in data
        assert "protocol_version" in data
        assert "client" in data
        assert data["transport"]["type"] == mock_client_capabilities_data["transport"]["type"]

    def test_client_calls_get_client_capabilities_data(self, client):
        """GET /api/client should call get_client_capabilities_data()."""
        response = client.get("/api/client")
        assert response.status_code == 200
        client.mocks["get_client_capabilities_data"].assert_called_once()


# ============================================================================
# Tests for GET /api/sessions/{id}/features
# ============================================================================


class TestSessionFeaturesEndpoint:
    """Tests for GET /api/sessions/{id}/features endpoint."""

    def test_features_returns_200(self, client):
        """GET /api/sessions/{id}/features should return 200 OK."""
        response = client.get("/api/sessions/session-1/features")
        assert response.status_code == 200

    def test_features_returns_feature_data(self, client, mock_session_features_data):
        """GET /api/sessions/{id}/features should return session features."""
        response = client.get("/api/sessions/session-1/features")
        data = response.json()

        assert "model" in data
        assert "mode" in data
        assert "cwd" in data
        assert "transport" in data
        assert data["model"] == mock_session_features_data["model"]

    def test_features_calls_get_session_features_data(self, client):
        """GET /api/sessions/{id}/features should call get_session_features_data()."""
        response = client.get("/api/sessions/session-1/features")
        assert response.status_code == 200
        client.mocks["get_session_features_data"].assert_called_once()

    def test_features_returns_503_when_no_manager(self):
        """GET /api/sessions/{id}/features should return 503 when no manager."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/session-1/features")
                assert response.status_code == 503

    def test_features_returns_404_when_session_not_found(self):
        """GET /api/sessions/{id}/features should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions/nonexistent/features")
                assert response.status_code == 404


# ============================================================================
# Tests for WebSocket /ws/{session_id}
# ============================================================================


class TestWebSocketEndpoint:
    """Tests for WS /ws/{session_id} endpoint."""

    def test_websocket_connect_and_receive_init(self, app_with_mocks):
        """WebSocket should connect and receive initial data."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Should receive init message
                data = websocket.receive_json()
                assert data["type"] == "init"
                assert data["session_id"] == "session-1"
                assert "context" in data
                assert "timeline" in data
                assert "projection" in data
                assert "message_history" in data
                assert "rendered" in data
                assert "features" in data
                assert "client" in data

    def test_websocket_ping_text_returns_pong(self, app_with_mocks):
        """WebSocket should respond to 'ping' text with 'pong'."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Drain init message
                websocket.receive_json()

                # Send ping as text
                websocket.send_text("ping")

                # Should receive pong as text
                response = websocket.receive_text()
                assert response == "pong"

    def test_websocket_invalid_json_returns_error(self, app_with_mocks):
        """WebSocket should return error for invalid JSON."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Drain init message
                websocket.receive_json()

                # Send invalid JSON
                websocket.send_text("not valid json {{{")

                # Should receive error
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "Invalid JSON" in response["message"]

    def test_websocket_unknown_command_returns_error(self, app_with_mocks):
        """WebSocket should return error for unknown command type."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Drain init message
                websocket.receive_json()

                # Send unknown command
                websocket.send_json({"type": "unknown_command"})

                # Should receive error
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "Unknown command type" in response["message"]

    def test_websocket_set_expansion_missing_params(self, app_with_mocks):
        """WebSocket set_expansion should error on missing parameters."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Drain init message
                websocket.receive_json()

                # Send set_expansion without node_id or expansion
                websocket.send_json({"type": "set_expansion"})

                # Should receive error
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "Missing node_id or expansion" in response["message"]

    def test_websocket_set_expansion_invalid_expansion(self, app_with_mocks):
        """WebSocket set_expansion should error on invalid expansion value."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Drain init message
                websocket.receive_json()

                # Send set_expansion with invalid expansion
                websocket.send_json({
                    "type": "set_expansion",
                    "node_id": "node-1",
                    "expansion": "invalid_value",
                })

                # Should receive error
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "Invalid expansion" in response["message"]

    def test_websocket_set_expansion_node_not_found(self, app_with_mocks):
        """WebSocket set_expansion should error when node not found."""
        app, mocks = app_with_mocks

        # Configure graph to return None for node lookup
        mock_graph = mocks["session"].get_context_graph.return_value
        mock_graph.get_node.return_value = None

        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Drain init message
                websocket.receive_json()

                # Send set_expansion for non-existent node
                websocket.send_json({
                    "type": "set_expansion",
                    "node_id": "nonexistent",
                    "expansion": "header",
                })

                # Should receive error
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "Node not found" in response["message"]

    def test_websocket_set_expansion_success(self):
        """WebSocket set_expansion should succeed with valid parameters."""
        from activecontext.context.state import Expansion

        # Configure mock node
        mock_node = MagicMock()
        mock_node.expansion = Expansion.ALL
        mock_node.node_type = "text"
        mock_node.GetDigest.return_value = {"id": "node-1", "type": "text"}

        # Configure mock session with graph
        mock_session = MagicMock()
        mock_graph = MagicMock()
        mock_graph.get_node.return_value = mock_node
        mock_session.get_context_graph.return_value = mock_graph

        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=mock_session)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static, \
             patch("activecontext.dashboard.routes.get_session_model") as mock_get_model, \
             patch("activecontext.dashboard.routes.get_session_mode") as mock_get_mode, \
             patch("activecontext.dashboard.routes.get_client_capabilities_data") as mock_get_client, \
             patch("activecontext.dashboard.routes.get_session_features_data") as mock_get_features, \
             patch("activecontext.dashboard.routes.get_context_data") as mock_get_context, \
             patch("activecontext.dashboard.routes.get_timeline_data") as mock_get_timeline, \
             patch("activecontext.dashboard.routes.get_projection_data") as mock_get_projection, \
             patch("activecontext.dashboard.routes.get_message_history_data") as mock_get_messages, \
             patch("activecontext.dashboard.routes.get_rendered_projection_data") as mock_get_rendered, \
             patch("activecontext.dashboard.routes.broadcast_update") as mock_broadcast:

            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir
            mock_get_model.return_value = "claude-3"
            mock_get_mode.return_value = "normal"
            mock_get_client.return_value = {"transport": {"type": "direct"}}
            mock_get_features.return_value = {"model": "claude-3"}
            mock_get_context.return_value = {"nodes_by_type": {}, "total": 0}
            mock_get_timeline.return_value = {"statements": [], "count": 0}
            mock_get_projection.return_value = {"total_used": 0, "sections": []}
            mock_get_messages.return_value = {"messages": [], "count": 0}
            mock_get_rendered.return_value = {"rendered": "", "total_tokens": 0, "sections": [], "section_count": 0}
            mock_broadcast.return_value = None

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                with client.websocket_connect("/ws/session-1") as websocket:
                    # Drain init message
                    websocket.receive_json()

                    # Send set_expansion
                    websocket.send_json({
                        "type": "set_expansion",
                        "node_id": "node-1",
                        "expansion": "header",
                    })

                    # Should receive confirmation
                    response = websocket.receive_json()
                    assert response["type"] == "expansion_changed"
                    assert response["node_id"] == "node-1"
                    assert response["old_expansion"] == "all"
                    assert response["new_expansion"] == "header"

    def test_websocket_handles_session_not_found_on_init(self):
        """WebSocket should handle session not found during init."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                # Should connect but not receive init data
                with client.websocket_connect("/ws/nonexistent") as websocket:
                    # WebSocket connects but manager can't find session
                    # Should still be able to receive messages (just not init)
                    websocket.send_text("ping")
                    response = websocket.receive_text()
                    assert response == "pong"

    def test_websocket_sends_error_when_init_data_fails(self):
        """WebSocket should send error message when gathering init data fails."""
        mock_session = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=mock_session)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static, \
             patch("activecontext.dashboard.routes.get_context_data") as mock_context:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir
            # Make get_context_data raise an exception
            mock_context.side_effect = Exception("Context data error")

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                with client.websocket_connect("/ws/test-session") as websocket:
                    # Should receive error message instead of init data
                    data = websocket.receive_json()
                    assert data["type"] == "error"
                    assert "Failed to load session data" in data["message"]
                    assert "Context data error" in data["message"]


# ============================================================================
# Tests for _handle_websocket_command()
# ============================================================================


class TestHandleWebsocketCommand:
    """Tests for _handle_websocket_command() helper function."""

    @pytest.mark.asyncio
    async def test_handle_command_routes_set_expansion(self):
        """_handle_websocket_command should route set_expansion command."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        with patch("activecontext.dashboard.routes._handle_set_expansion") as mock_handler:
            mock_handler.return_value = None

            from activecontext.dashboard.routes import _handle_websocket_command

            await _handle_websocket_command(
                mock_websocket,
                "session-1",
                {
                    "type": "set_expansion",
                    "node_id": "node-1",
                    "expansion": "collapsed",
                }
            )

            mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_command_sends_error_for_unknown(self):
        """_handle_websocket_command should send error for unknown command."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        from activecontext.dashboard.routes import _handle_websocket_command

        await _handle_websocket_command(
            mock_websocket,
            "session-1",
            {"type": "unknown_command"}
        )

        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "error"
        assert "Unknown command type" in call_args["message"]


# ============================================================================
# Tests for _handle_set_expansion()
# ============================================================================


class TestHandleSetExpansion:
    """Tests for _handle_set_expansion() helper function."""

    @pytest.mark.asyncio
    async def test_set_expansion_returns_error_when_no_manager(self):
        """_handle_set_expansion should error when manager not available."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager:
            mock_get_manager.return_value = None

            from activecontext.dashboard.routes import _handle_set_expansion

            await _handle_set_expansion(
                mock_websocket,
                "session-1",
                {"node_id": "node-1", "expansion": "collapsed"},
            )

            mock_websocket.send_json.assert_called_once()
            call_args = mock_websocket.send_json.call_args[0][0]
            assert call_args["type"] == "error"
            assert "not initialized" in call_args["message"]

    @pytest.mark.asyncio
    async def test_set_expansion_returns_error_when_session_not_found(self):
        """_handle_set_expansion should error when session not found."""
        mock_websocket = MagicMock()
        mock_websocket.send_json = AsyncMock()

        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager:
            mock_get_manager.return_value = mock_manager

            from activecontext.dashboard.routes import _handle_set_expansion

            await _handle_set_expansion(
                mock_websocket,
                "nonexistent",
                {"node_id": "node-1", "expansion": "collapsed"},
            )

            mock_websocket.send_json.assert_called_once()
            call_args = mock_websocket.send_json.call_args[0][0]
            assert call_args["type"] == "error"
            assert "Session not found" in call_args["message"]


# ============================================================================
# Tests for broadcast_update()
# ============================================================================


class TestBroadcastUpdate:
    """Tests for broadcast_update() function."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_connection_manager(self):
        """broadcast_update should send via connection_manager."""
        with patch("activecontext.dashboard.routes.connection_manager") as mock_conn_mgr, \
             patch("activecontext.dashboard.routes.format_session_update") as mock_format:
            mock_conn_mgr.broadcast = AsyncMock()
            mock_format.return_value = {"type": "update", "data": {}}

            from activecontext.dashboard.routes import broadcast_update

            await broadcast_update(
                "session-1",
                "node_changed",
                {"node_id": "node-1"},
                1234567890.0,
            )

            mock_format.assert_called_once_with(
                "node_changed",
                "session-1",
                {"node_id": "node-1"},
                1234567890.0,
            )
            mock_conn_mgr.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_includes_formatted_message(self):
        """broadcast_update should broadcast the formatted message."""
        with patch("activecontext.dashboard.routes.connection_manager") as mock_conn_mgr, \
             patch("activecontext.dashboard.routes.format_session_update") as mock_format:
            mock_conn_mgr.broadcast = AsyncMock()
            expected_message = {
                "type": "update",
                "kind": "node_changed",
                "session_id": "session-1",
                "timestamp": 1234567890.0,
                "payload": {"node_id": "node-1"},
            }
            mock_format.return_value = expected_message

            from activecontext.dashboard.routes import broadcast_update

            await broadcast_update(
                "session-1",
                "node_changed",
                {"node_id": "node-1"},
                1234567890.0,
            )

            mock_conn_mgr.broadcast.assert_called_once_with("session-1", expected_message)


# ============================================================================
# Tests for error handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across endpoints."""

    def test_all_session_endpoints_return_503_when_no_manager(self):
        """All session endpoints should return 503 when manager not initialized."""
        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = None
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            endpoints = [
                "/api/sessions/session-1/context",
                "/api/sessions/session-1/timeline",
                "/api/sessions/session-1/projection",
                "/api/sessions/session-1/message-history",
                "/api/sessions/session-1/rendered",
                "/api/sessions/session-1/features",
            ]

            with TestClient(app) as client:
                for endpoint in endpoints:
                    response = client.get(endpoint)
                    assert response.status_code == 503, f"Expected 503 for {endpoint}"
                    assert "Dashboard not initialized" in response.json()["detail"]

    def test_all_session_endpoints_return_404_for_unknown_session(self):
        """All session endpoints should return 404 for unknown session."""
        mock_manager = MagicMock()
        mock_manager.get_session = AsyncMock(return_value=None)

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            endpoints = [
                "/api/sessions/nonexistent/context",
                "/api/sessions/nonexistent/timeline",
                "/api/sessions/nonexistent/projection",
                "/api/sessions/nonexistent/message-history",
                "/api/sessions/nonexistent/rendered",
                "/api/sessions/nonexistent/features",
            ]

            with TestClient(app) as client:
                for endpoint in endpoints:
                    response = client.get(endpoint)
                    assert response.status_code == 404, f"Expected 404 for {endpoint}"
                    assert "not found" in response.json()["detail"]


# ============================================================================
# Tests for edge cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_session_list(self):
        """GET /api/sessions should handle empty session list."""
        mock_manager = MagicMock()
        mock_manager.list_sessions = AsyncMock(return_value=[])

        with patch("activecontext.dashboard.routes.get_manager") as mock_get_manager, \
             patch("activecontext.dashboard.routes.get_static_dir") as mock_static:
            mock_get_manager.return_value = mock_manager
            mock_dir = MagicMock(spec=Path)
            mock_dir.exists.return_value = False
            mock_static.return_value = mock_dir

            from activecontext.dashboard.routes import create_app
            app = create_app()

            with TestClient(app) as client:
                response = client.get("/api/sessions")
                assert response.status_code == 200
                assert response.json() == []

    def test_special_characters_in_session_id(self, client):
        """Endpoints should handle special characters in session IDs."""
        # Test with URL-safe special chars
        response = client.get("/api/sessions/session-with-dashes_and_underscores/context")
        # Should not crash, may return 200 or 404 depending on mock
        assert response.status_code in (200, 404)

    def test_concurrent_status_requests(self, client):
        """Server should handle concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/api/status")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        # All requests should succeed
        assert all(r.status_code == 200 for r in results)

    def test_websocket_handles_disconnect(self, app_with_mocks):
        """WebSocket should handle client disconnect gracefully."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            # Connect and immediately close
            with client.websocket_connect("/ws/session-1") as websocket:
                # Receive init
                websocket.receive_json()
                # Close connection (happens automatically on context exit)

        # Server should not crash


# ============================================================================
# Tests for connection_manager integration
# ============================================================================


class TestConnectionManagerIntegration:
    """Tests for WebSocket connection manager integration."""

    def test_connection_manager_receives_connections(self, app_with_mocks):
        """Connection manager should track active WebSocket connections."""
        app, mocks = app_with_mocks

        # Simply verify that the WebSocket connection flow works
        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-1") as websocket:
                # Connection should succeed and receive init data
                data = websocket.receive_json()
                assert data["type"] == "init"
                assert data["session_id"] == "session-1"
        # Server should not crash on disconnect

    def test_multiple_websocket_connections(self, app_with_mocks):
        """Multiple WebSocket connections should work simultaneously."""
        app, mocks = app_with_mocks

        with TestClient(app) as client:
            # Open two connections
            with client.websocket_connect("/ws/session-1") as ws1:
                data1 = ws1.receive_json()
                assert data1["type"] == "init"

                with client.websocket_connect("/ws/session-2") as ws2:
                    data2 = ws2.receive_json()
                    assert data2["type"] == "init"

                    # Both should respond to ping
                    ws1.send_text("ping")
                    assert ws1.receive_text() == "pong"

                    ws2.send_text("ping")
                    assert ws2.receive_text() == "pong"
