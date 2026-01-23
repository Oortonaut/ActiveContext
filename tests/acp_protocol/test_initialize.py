"""Initialize method compliance tests.

Tests verify compliance with ACP protocol spec (docs/acp-protocol.md).

Initialize Response (per spec):
  protocolVersion: int
  agentCapabilities: {
    loadSession: boolean,
    promptCapabilities: {image, audio, embeddedContext?},
    mcpCapabilities?: {http, sse},
    sessionCapabilities?: object
  }
  agentInfo: {name: string, version: string}
  authMethods?: array
"""

from __future__ import annotations

import pytest

from .helpers import ACPTestClient, initialize_agent


@pytest.mark.asyncio(loop_scope="module")
class TestInitialize:
    """Tests for the initialize method per ACP spec."""

    async def test_initialize_returns_agent_info(self, client: ACPTestClient) -> None:
        """Initialize response must have agentInfo with name and version (spec required)."""
        response = await initialize_agent(client)
        assert "result" in response, f"Initialize failed: {response}"

        result = response["result"]
        assert "agentInfo" in result, f"Missing agentInfo: {result}"

        agent_info = result["agentInfo"]
        assert "name" in agent_info, f"agentInfo missing 'name': {agent_info}"
        assert "version" in agent_info, f"agentInfo missing 'version': {agent_info}"
        assert isinstance(agent_info["name"], str), "agentInfo.name must be string"
        assert isinstance(agent_info["version"], str), "agentInfo.version must be string"

    async def test_initialize_returns_protocol_version(self, initialized_client: ACPTestClient) -> None:
        """Initialize response must include protocolVersion (spec required)."""
        response = await initialize_agent(initialized_client)
        result = response["result"]

        assert "protocolVersion" in result, f"Missing protocolVersion: {result}"
        assert isinstance(result["protocolVersion"], int), "protocolVersion must be int"

    async def test_initialize_returns_agent_capabilities(self, initialized_client: ACPTestClient) -> None:
        """Initialize response must have agentCapabilities (spec required)."""
        response = await initialize_agent(initialized_client)
        result = response["result"]

        assert "agentCapabilities" in result, f"Missing agentCapabilities: {result}"
        caps = result["agentCapabilities"]
        assert isinstance(caps, dict), "agentCapabilities must be dict"

    async def test_agent_capabilities_has_required_fields(self, initialized_client: ACPTestClient) -> None:
        """agentCapabilities must have loadSession and promptCapabilities (spec required)."""
        response = await initialize_agent(initialized_client)
        caps = response["result"]["agentCapabilities"]

        # loadSession is required per spec
        assert "loadSession" in caps, f"Missing loadSession in capabilities: {caps}"
        assert isinstance(caps["loadSession"], bool), "loadSession must be boolean"

        # promptCapabilities is required per spec
        assert "promptCapabilities" in caps, f"Missing promptCapabilities: {caps}"
        prompt_caps = caps["promptCapabilities"]
        assert isinstance(prompt_caps, dict), "promptCapabilities must be dict"

        # promptCapabilities must have image and audio
        assert "image" in prompt_caps, f"Missing image in promptCapabilities: {prompt_caps}"
        assert "audio" in prompt_caps, f"Missing audio in promptCapabilities: {prompt_caps}"

    async def test_initialize_idempotent(self, initialized_client: ACPTestClient) -> None:
        """Multiple initialize calls should succeed (protocol allows re-init)."""
        # Second initialize - should also succeed
        response = await initialized_client.send_request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {},
                "clientInfo": {"name": "test2", "title": "Test 2", "version": "2.0"},
            },
        )
        assert "result" in response, f"Second initialize failed: {response}"

        # Verify response still has required fields
        result = response["result"]
        assert "protocolVersion" in result
        assert "agentCapabilities" in result
        assert "agentInfo" in result
