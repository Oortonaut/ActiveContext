"""Error handling compliance tests.

Tests verify compliance with ACP protocol spec (docs/acp-protocol.md).

Error Codes (per JSON-RPC 2.0 and ACP spec):
  -32700  Parse error      Invalid JSON received
  -32600  Invalid Request  Request invalid or session not found
  -32601  Method not found Method not implemented
  -32602  Invalid params   Invalid parameters passed
  -32603  Internal error   Internal agent error

Error Response Format:
  {
    "jsonrpc": "2.0",
    "id": <request-id>,
    "error": {
      "code": <error-code>,
      "message": <error-message>,
      "data": {"details": <optional>}
    }
  }
"""

from __future__ import annotations

import asyncio

import pytest

from .helpers import ACPTestClient

# Standard JSON-RPC 2.0 error codes per spec
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


@pytest.mark.asyncio(loop_scope="module")
class TestErrorCodes:
    """Tests for error code compliance per ACP spec."""

    async def test_method_not_found_code(self, initialized_client: ACPTestClient) -> None:
        """Unknown method must return -32601 (Method not found)."""
        response = await initialized_client.send_request("nonexistent/method", {})

        assert "error" in response, f"Expected error for unknown method: {response}"
        error = response["error"]
        assert error["code"] == METHOD_NOT_FOUND, f"Expected -32601, got {error['code']}"

    async def test_invalid_params_code(self, initialized_client: ACPTestClient) -> None:
        """Invalid parameters must return -32602 (Invalid params)."""
        # session/prompt with wrong type for sessionId
        response = await initialized_client.send_request(
            "session/prompt",
            {"sessionId": 12345, "prompt": []},  # sessionId should be string
        )

        assert "error" in response, f"Expected error for wrong param type: {response}"
        assert response["error"]["code"] == INVALID_PARAMS, f"Expected -32602: {response['error']}"

    async def test_session_not_found_code(self, initialized_client: ACPTestClient) -> None:
        """Unknown session ID must return -32600 (Invalid Request) per spec."""
        fake_session_id = "nonexistent-session-12345"

        response = await initialized_client.send_request(
            "session/prompt",
            {"sessionId": fake_session_id, "prompt": []},
        )

        assert "error" in response, f"Expected error for unknown session: {response}"
        # Per spec, session not found is -32600 (Invalid Request)
        assert response["error"]["code"] == INVALID_REQUEST, f"Expected -32600: {response['error']}"


@pytest.mark.asyncio(loop_scope="module")
class TestErrorFormat:
    """Tests for error response format per spec."""

    async def test_error_has_code(self, initialized_client: ACPTestClient) -> None:
        """Error response must have 'code' field (integer)."""
        response = await initialized_client.send_request("nonexistent/method", {})

        error = response["error"]
        assert "code" in error, f"Error missing 'code': {error}"
        assert isinstance(error["code"], int), f"Error code must be int: {error['code']}"

    async def test_error_has_message(self, initialized_client: ACPTestClient) -> None:
        """Error response must have 'message' field (string)."""
        response = await initialized_client.send_request("nonexistent/method", {})

        error = response["error"]
        assert "message" in error, f"Error missing 'message': {error}"
        assert isinstance(error["message"], str), f"Error message must be str: {error['message']}"
        assert len(error["message"]) > 0, "Error message must not be empty"

    async def test_error_data_is_optional(self, initialized_client: ACPTestClient) -> None:
        """Error 'data' field is optional but must be object if present."""
        response = await initialized_client.send_request("nonexistent/method", {})

        error = response["error"]
        if "data" in error:
            assert isinstance(error["data"], dict), f"Error data must be object: {error['data']}"


@pytest.mark.asyncio(loop_scope="module")
class TestParseErrorRecovery:
    """Tests for agent recovery from malformed input.

    Per spec, agent should handle malformed JSON gracefully and remain responsive.
    """

    async def test_invalid_json_recovery(self, initialized_client: ACPTestClient) -> None:
        """Agent should recover from invalid JSON."""
        # Send malformed JSON
        await initialized_client.send_raw("not valid json {{{")
        await asyncio.sleep(0.1)

        # Agent should still respond to valid requests
        response = await initialized_client.send_request("session/list", {})
        assert "result" in response, f"Agent unresponsive after bad JSON: {response}"

    async def test_empty_line_recovery(self, initialized_client: ACPTestClient) -> None:
        """Agent should recover from empty lines."""
        await initialized_client.send_raw("")
        await asyncio.sleep(0.1)

        response = await initialized_client.send_request("session/list", {})
        assert "result" in response, f"Agent unresponsive after empty line: {response}"

    async def test_partial_json_recovery(self, initialized_client: ACPTestClient) -> None:
        """Agent should recover from truncated JSON."""
        await initialized_client.send_raw('{"jsonrpc": "2.0", "id": 999, "method":')
        await asyncio.sleep(0.1)

        response = await initialized_client.send_request("session/list", {})
        assert "result" in response, f"Agent unresponsive after partial JSON: {response}"
