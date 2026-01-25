"""Session method compliance tests.

Tests verify compliance with ACP protocol spec (docs/acp-protocol.md).

session/new Request (per spec):
  cwd: string (required, absolute path)
  mcpServers?: array (OPTIONAL per spec)

session/new Response (per spec):
  sessionId: string (required)
  models?: {availableModels: array, currentModelId: string}
  modes?: {availableModes: array, currentModeId: string}

session/list Response (per spec):
  sessions: array[{sessionId, cwd, ...}]
  nextCursor?: string
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .helpers import ACPTestClient


@pytest.mark.asyncio(loop_scope="module")
class TestSessionNew:
    """Tests for session/new per ACP spec."""

    async def test_session_new_requires_only_cwd(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/new should only require cwd (mcpServers is optional per spec).

        The spec explicitly marks mcpServers as optional with '?'.
        This is achieved via runtime patching in activecontext/__main__.py.
        """
        response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd)},  # Only cwd, no mcpServers
        )

        # Per spec, this should succeed with just cwd
        if "error" in response:
            error = response["error"]
            # If error mentions mcpServers, that's a spec violation
            if "mcpServers" in str(error):
                pytest.fail(
                    f"SPEC VIOLATION: mcpServers should be optional per spec, "
                    f"but got error: {error}"
                )
            # Other errors might be valid (e.g., invalid cwd)
            pytest.fail(f"session/new failed: {response}")

        assert "result" in response

    async def test_session_new_returns_session_id(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/new must return a sessionId (spec required)."""
        response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},  # Include mcpServers for compatibility
        )
        assert "result" in response, f"session/new failed: {response}"

        result = response["result"]
        assert "sessionId" in result, f"Missing sessionId: {result}"
        assert isinstance(result["sessionId"], str), "sessionId must be string"
        assert len(result["sessionId"]) > 0, "sessionId must not be empty"

    async def test_session_new_modes_format(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/new modes (if present) must match spec format."""
        response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )
        result = response["result"]

        # modes is optional per spec
        if "modes" not in result:
            pytest.skip("modes not present in response (optional per spec)")

        modes_data = result["modes"]
        assert "availableModes" in modes_data, f"modes missing availableModes: {modes_data}"
        assert "currentModeId" in modes_data, f"modes missing currentModeId: {modes_data}"

        modes = modes_data["availableModes"]
        assert isinstance(modes, list), f"availableModes must be list: {modes}"

        # Each mode must have id, name, description per spec
        for mode in modes:
            assert "id" in mode, f"Mode missing 'id': {mode}"
            assert "name" in mode, f"Mode missing 'name': {mode}"
            # description is in spec examples

    async def test_session_new_models_format(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/new models (if present) must match spec format."""
        response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )
        result = response["result"]

        # models is optional per spec
        if "models" not in result:
            pytest.skip("models not present in response (optional per spec)")

        models_data = result["models"]
        assert "availableModels" in models_data, f"models missing availableModels: {models_data}"
        assert "currentModelId" in models_data, f"models missing currentModelId: {models_data}"

        models = models_data["availableModels"]
        assert isinstance(models, list), f"availableModels must be list: {models}"

        # Each model must have modelId, name per spec
        for model in models:
            assert "modelId" in model, f"Model missing 'modelId': {model}"
            assert "name" in model, f"Model missing 'name': {model}"


@pytest.mark.asyncio(loop_scope="module")
class TestSessionList:
    """Tests for session/list per ACP spec."""

    async def test_session_list_returns_sessions_array(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/list must return sessions array (spec required)."""
        # Create a session first
        await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )

        response = await initialized_client.send_request("session/list", {})
        assert "result" in response, f"session/list failed: {response}"

        result = response["result"]
        assert "sessions" in result, f"Missing sessions: {result}"
        assert isinstance(result["sessions"], list), "sessions must be list"

    async def test_session_list_contains_created_session(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """Created session should appear in session/list."""
        # Create a session
        create_response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )
        session_id = create_response["result"]["sessionId"]

        # List sessions
        list_response = await initialized_client.send_request("session/list", {})
        sessions = list_response["result"]["sessions"]

        # Find our session
        session_ids = [s.get("sessionId") for s in sessions]
        assert session_id in session_ids, f"Session {session_id} not in list: {session_ids}"

    async def test_session_list_entries_have_session_id(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """Each session in list must have sessionId (spec required)."""
        # Ensure at least one session exists
        await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )

        list_response = await initialized_client.send_request("session/list", {})
        sessions = list_response["result"]["sessions"]

        for session in sessions:
            assert "sessionId" in session, f"Session missing sessionId: {session}"
            assert isinstance(session["sessionId"], str), "sessionId must be string"


@pytest.mark.asyncio(loop_scope="module")
class TestSessionSetMode:
    """Tests for session/setMode.

    NOTE: session/setMode may not be implemented in all agents.
    Tests skip if method returns -32601 (not found).
    """

    async def test_session_set_mode_valid(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/setMode with valid mode should succeed."""
        # Create session and get available modes
        create_response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )
        result = create_response["result"]
        session_id = result["sessionId"]
        modes = result.get("modes", {}).get("availableModes", [])

        if len(modes) < 2:
            pytest.skip("Need at least 2 modes to test setMode")

        # Set to second mode
        target_mode = modes[1]["id"]
        response = await initialized_client.send_request(
            "session/setMode",
            {"sessionId": session_id, "modeId": target_mode},
        )

        # Skip if method not implemented
        if "error" in response and response["error"]["code"] == -32601:
            pytest.skip("session/setMode not implemented")

        assert "result" in response, f"setMode failed: {response}"

    async def test_session_set_mode_invalid_returns_error(self, initialized_client: ACPTestClient, test_cwd: Path) -> None:
        """session/setMode with invalid mode should return error (-32602)."""
        # Create session
        create_response = await initialized_client.send_request(
            "session/new",
            {"cwd": str(test_cwd), "mcpServers": []},
        )
        session_id = create_response["result"]["sessionId"]

        # Try invalid mode
        response = await initialized_client.send_request(
            "session/setMode",
            {"sessionId": session_id, "modeId": "nonexistent-mode-xyz"},
        )

        # Skip if method not implemented
        if "error" in response and response["error"]["code"] == -32601:
            pytest.skip("session/setMode not implemented")

        assert "error" in response, f"Expected error for invalid mode: {response}"
        # Per spec, invalid params should be -32602
        assert response["error"]["code"] == -32602, f"Expected -32602, got {response['error']}"
