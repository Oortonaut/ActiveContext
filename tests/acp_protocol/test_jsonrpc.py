"""JSON-RPC 2.0 format compliance tests.

Tests verify compliance with JSON-RPC 2.0 spec as used by ACP.

Per docs/acp-protocol.md:
- Transport: Newline-delimited JSON over stdio
- Format: JSON-RPC 2.0
- Every request gets exactly one response
- Responses have either 'result' or 'error', never both

JSON-RPC 2.0 Response Format:
  {
    "jsonrpc": "2.0",
    "id": <same-as-request>,
    "result": <result-value>  // OR
    "error": {"code": int, "message": string, "data"?: object}
  }
"""

from __future__ import annotations

import pytest

from .helpers import ACPTestClient, initialize_agent


@pytest.mark.asyncio(loop_scope="module")
class TestJsonRpcFormat:
    """Tests for JSON-RPC 2.0 message format compliance."""

    async def test_response_has_jsonrpc_field(self, client: ACPTestClient) -> None:
        """Every response must have jsonrpc: '2.0'."""
        response = await initialize_agent(client)
        assert "jsonrpc" in response, f"Missing jsonrpc field: {response}"
        assert response["jsonrpc"] == "2.0", f"Wrong jsonrpc version: {response['jsonrpc']}"

    async def test_response_id_matches_request(self, initialized_client: ACPTestClient) -> None:
        """Response ID must exactly match request ID."""
        custom_id = 99999
        response = await initialized_client.send_request(
            "session/list",
            {},
            request_id=custom_id,
        )
        assert response.get("id") == custom_id, f"ID mismatch: expected {custom_id}, got {response.get('id')}"

    async def test_response_string_id_matches(self, initialized_client: ACPTestClient) -> None:
        """String request IDs should be echoed exactly."""
        string_id = "test-request-abc-xyz"
        response = await initialized_client.send_request(
            "session/list",
            {},
            request_id=string_id,
        )
        assert response.get("id") == string_id, f"String ID not echoed: {response.get('id')}"

    async def test_success_response_has_result(self, initialized_client: ACPTestClient) -> None:
        """Successful response must have 'result' field, not 'error'."""
        response = await initialized_client.send_request("session/list", {})

        assert "result" in response, f"Success response missing 'result': {response}"
        assert "error" not in response, f"Success response should not have 'error': {response}"

    async def test_error_response_has_error(self, initialized_client: ACPTestClient) -> None:
        """Error response must have 'error' field, not 'result'."""
        response = await initialized_client.send_request("nonexistent/method", {})

        assert "error" in response, f"Error response missing 'error': {response}"
        assert "result" not in response, f"Error response should not have 'result': {response}"

    async def test_result_xor_error(self, initialized_client: ACPTestClient) -> None:
        """Response must have exactly one of 'result' or 'error', never both."""
        # Test with valid request
        response = await initialized_client.send_request("session/list", {})
        has_result = "result" in response
        has_error = "error" in response
        assert has_result != has_error, f"Must have result XOR error: {response}"

        # Test with invalid request
        response = await initialized_client.send_request("bad/method", {})
        has_result = "result" in response
        has_error = "error" in response
        assert has_result != has_error, f"Must have result XOR error: {response}"
