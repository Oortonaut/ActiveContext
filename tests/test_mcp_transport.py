"""Tests for the MCP transport module."""

from unittest.mock import MagicMock, patch

import pytest

from activecontext.config.schema import MCPServerConfig


class TestExpandEnvVars:
    """Tests for _expand_env_vars function."""

    def test_expand_simple_var(self, monkeypatch):
        from activecontext.mcp.transport import _expand_env_vars

        monkeypatch.setenv("TEST_VAR", "test_value")
        env = {"KEY": "${TEST_VAR}"}

        result = _expand_env_vars(env)

        assert result["KEY"] == "test_value"

    def test_expand_missing_var(self, monkeypatch):
        from activecontext.mcp.transport import _expand_env_vars

        monkeypatch.delenv("MISSING_VAR", raising=False)
        env = {"KEY": "${MISSING_VAR}"}

        result = _expand_env_vars(env)

        assert result["KEY"] == ""

    def test_preserve_literal_value(self):
        from activecontext.mcp.transport import _expand_env_vars

        env = {"KEY": "literal_value"}

        result = _expand_env_vars(env)

        assert result["KEY"] == "literal_value"

    def test_expand_multiple_vars(self, monkeypatch):
        from activecontext.mcp.transport import _expand_env_vars

        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        env = {"K1": "${VAR1}", "K2": "${VAR2}", "K3": "literal"}

        result = _expand_env_vars(env)

        assert result["K1"] == "value1"
        assert result["K2"] == "value2"
        assert result["K3"] == "literal"

    def test_partial_match_not_expanded(self):
        from activecontext.mcp.transport import _expand_env_vars

        # Only exact ${VAR} pattern should be expanded
        env = {"KEY": "$VAR"}

        result = _expand_env_vars(env)

        assert result["KEY"] == "$VAR"

    def test_empty_dict(self):
        from activecontext.mcp.transport import _expand_env_vars

        result = _expand_env_vars({})

        assert result == {}


class TestCreateTransport:
    """Tests for create_transport function."""

    @pytest.mark.asyncio
    async def test_stdio_requires_command(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command=None,  # Missing command
        )

        with pytest.raises(ValueError, match="requires 'command'"):
            await create_transport(config)

    @pytest.mark.asyncio
    async def test_streamable_http_requires_url(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="streamable-http",
            url=None,  # Missing URL
        )

        with pytest.raises(ValueError, match="requires 'url'"):
            await create_transport(config)

    @pytest.mark.asyncio
    async def test_sse_requires_url(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="sse",
            url=None,  # Missing URL
        )

        with pytest.raises(ValueError, match="requires 'url'"):
            await create_transport(config)

    @pytest.mark.asyncio
    async def test_unknown_transport_raises(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="unknown",
        )

        with pytest.raises(ValueError, match="Unknown transport"):
            await create_transport(config)

    @pytest.mark.asyncio
    async def test_stdio_creates_client(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command=["python", "-m", "test_server"],
            extra_args=["--port", "8080"],
        )

        mock_client = MagicMock()
        with patch(
            "mcp.client.stdio.stdio_client", return_value=mock_client
        ) as mock_stdio:
            result = await create_transport(config)

            mock_stdio.assert_called_once()
            params = mock_stdio.call_args[0][0]
            assert params.command == "python"
            assert params.args == ["-m", "test_server", "--port", "8080"]
            assert result is mock_client

    @pytest.mark.asyncio
    async def test_stdio_merges_environment(self, monkeypatch):
        from activecontext.mcp.transport import create_transport

        monkeypatch.setenv("BASE_VAR", "base_value")
        monkeypatch.setenv("SECRET", "secret_value")

        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command=["test"],
            env={"API_KEY": "${SECRET}", "CUSTOM": "custom_value"},
        )

        with patch("mcp.client.stdio.stdio_client") as mock_stdio:
            await create_transport(config)

            params = mock_stdio.call_args[0][0]
            # Should have base environment
            assert "BASE_VAR" in params.env
            assert params.env["BASE_VAR"] == "base_value"
            # Should have expanded config env
            assert params.env["API_KEY"] == "secret_value"
            assert params.env["CUSTOM"] == "custom_value"

    @pytest.mark.asyncio
    async def test_streamable_http_creates_client(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="streamable-http",
            url="http://localhost:8080/mcp",
        )

        mock_client = MagicMock()
        with patch(
            "mcp.client.streamable_http.streamablehttp_client", return_value=mock_client
        ) as mock_http:
            result = await create_transport(config)

            mock_http.assert_called_once_with("http://localhost:8080/mcp")
            assert result is mock_client

    @pytest.mark.asyncio
    async def test_sse_creates_client(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
            timeout=60.0,
        )

        mock_client = MagicMock()
        with patch(
            "mcp.client.sse.sse_client", return_value=mock_client
        ) as mock_sse:
            result = await create_transport(config)

            mock_sse.assert_called_once()
            call_kwargs = mock_sse.call_args
            assert call_kwargs[0][0] == "http://localhost:8080/sse"
            assert call_kwargs[1]["timeout"] == 60.0
            assert result is mock_client

    @pytest.mark.asyncio
    async def test_sse_with_headers(self, monkeypatch):
        from activecontext.mcp.transport import create_transport

        monkeypatch.setenv("API_TOKEN", "secret123")

        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
            headers={"Authorization": "${API_TOKEN}", "X-Custom": "value"},
        )

        with patch("mcp.client.sse.sse_client") as mock_sse:
            await create_transport(config)

            call_kwargs = mock_sse.call_args[1]
            assert call_kwargs["headers"]["Authorization"] == "secret123"
            assert call_kwargs["headers"]["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_sse_filters_empty_headers(self, monkeypatch):
        from activecontext.mcp.transport import create_transport

        monkeypatch.delenv("MISSING_VAR", raising=False)

        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
            headers={"Empty": "${MISSING_VAR}", "Valid": "value"},
        )

        with patch("mcp.client.sse.sse_client") as mock_sse:
            await create_transport(config)

            call_kwargs = mock_sse.call_args[1]
            # Empty header should be filtered out
            assert "Empty" not in call_kwargs["headers"]
            assert call_kwargs["headers"]["Valid"] == "value"

    @pytest.mark.asyncio
    async def test_sse_no_headers_passes_none(self):
        from activecontext.mcp.transport import create_transport

        config = MCPServerConfig(
            name="test",
            transport="sse",
            url="http://localhost:8080/sse",
            headers={},  # Empty headers
        )

        with patch("mcp.client.sse.sse_client") as mock_sse:
            await create_transport(config)

            call_kwargs = mock_sse.call_args[1]
            assert call_kwargs["headers"] is None
