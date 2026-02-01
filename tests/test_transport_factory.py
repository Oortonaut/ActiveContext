"""Tests for the CAP transport factory."""

from __future__ import annotations

import pytest

from activecontext.plugins.tcp_transport import TcpTransport
from activecontext.plugins.transport import PluginTransportError, StdioTransport
from activecontext.plugins.transport_factory import (
    TransportConfig,
    create_transport,
    list_available_transports,
)
from activecontext.plugins.ws_transport import WebSocketTransport


class TestCreateTransport:
    """Tests for create_transport()."""

    def test_stdio_returns_stdio_transport(self) -> None:
        """Valid stdio config returns a StdioTransport instance."""
        config = TransportConfig(type="stdio", command=["python", "-m", "my_plugin"])
        transport = create_transport(config)
        assert isinstance(transport, StdioTransport)

    def test_stdio_without_command_raises_value_error(self) -> None:
        """stdio transport with empty command raises ValueError."""
        config = TransportConfig(type="stdio", command=[])
        with pytest.raises(ValueError, match="stdio transport requires 'command'"):
            create_transport(config)

    def test_grpc_returns_grpc_transport(self) -> None:
        """Valid gRPC config returns a GrpcTransport instance."""
        from activecontext.plugins.grpc_transport import GrpcTransport

        config = TransportConfig(type="grpc", address="localhost:50051")
        transport = create_transport(config)
        assert isinstance(transport, GrpcTransport)

    def test_grpc_without_address_raises_value_error(self) -> None:
        """gRPC transport without address raises ValueError before PluginTransportError."""
        config = TransportConfig(type="grpc", address="")
        with pytest.raises(ValueError, match="gRPC transport requires 'address'"):
            create_transport(config)

    def test_websocket_returns_websocket_transport(self) -> None:
        """Valid websocket config returns a WebSocketTransport instance."""
        config = TransportConfig(type="websocket", url="ws://localhost:8080")
        transport = create_transport(config)
        assert isinstance(transport, WebSocketTransport)

    def test_websocket_without_url_raises_value_error(self) -> None:
        """WebSocket transport without url raises ValueError before PluginTransportError."""
        config = TransportConfig(type="websocket", url="")
        with pytest.raises(ValueError, match="WebSocket transport requires 'url'"):
            create_transport(config)

    def test_tcp_returns_tcp_transport(self) -> None:
        """Valid tcp config returns a TcpTransport instance."""
        config = TransportConfig(type="tcp", address="localhost:9000")
        transport = create_transport(config)
        assert isinstance(transport, TcpTransport)

    def test_tcp_without_address_raises_value_error(self) -> None:
        """TCP transport without address raises ValueError before PluginTransportError."""
        config = TransportConfig(type="tcp", address="")
        with pytest.raises(ValueError, match="TCP transport requires 'address'"):
            create_transport(config)

    def test_tcp_parses_host_and_port(self) -> None:
        """TCP transport correctly parses host:port into separate fields."""
        config = TransportConfig(type="tcp", address="example.com:8080")
        transport = create_transport(config)
        assert isinstance(transport, TcpTransport)
        assert transport._host == "example.com"
        assert transport._port == 8080

    def test_tcp_invalid_address_format_raises_value_error(self) -> None:
        """TCP address without colon raises ValueError."""
        config = TransportConfig(type="tcp", address="localhost")
        with pytest.raises(ValueError, match="TCP address must be 'host:port'"):
            create_transport(config)

    def test_tcp_invalid_port_raises_value_error(self) -> None:
        """TCP address with non-numeric port raises ValueError."""
        config = TransportConfig(type="tcp", address="localhost:not_a_port")
        with pytest.raises(ValueError, match="Invalid TCP port"):
            create_transport(config)

    def test_unknown_type_raises_plugin_transport_error(self) -> None:
        """Unknown transport type raises PluginTransportError listing valid types."""
        config = TransportConfig(type="carrier_pigeon")
        with pytest.raises(PluginTransportError, match="Unknown transport type") as exc_info:
            create_transport(config)
        error_message = str(exc_info.value)
        assert "carrier_pigeon" in error_message
        assert "stdio" in error_message
        assert "grpc" in error_message
        assert "websocket" in error_message
        assert "tcp" in error_message

    def test_case_insensitive_type_uppercase(self) -> None:
        """Transport type matching is case-insensitive (STDIO)."""
        config = TransportConfig(type="STDIO", command=["python", "-m", "plugin"])
        transport = create_transport(config)
        assert isinstance(transport, StdioTransport)

    def test_case_insensitive_type_mixed_case(self) -> None:
        """Transport type matching is case-insensitive (Stdio)."""
        config = TransportConfig(type="Stdio", command=["python", "-m", "plugin"])
        transport = create_transport(config)
        assert isinstance(transport, StdioTransport)

    def test_stdio_passes_env(self) -> None:
        """stdio transport forwards env to StdioTransport."""
        config = TransportConfig(
            type="stdio",
            command=["python", "-m", "plugin"],
            env={"MY_VAR": "value"},
        )
        transport = create_transport(config)
        assert isinstance(transport, StdioTransport)
        # Verify env was passed through (StdioTransport stores it as _env)
        assert transport._env == {"MY_VAR": "value"}

    def test_stdio_passes_cwd(self) -> None:
        """stdio transport forwards cwd to StdioTransport."""
        config = TransportConfig(
            type="stdio",
            command=["python", "-m", "plugin"],
            cwd="/tmp/workdir",
        )
        transport = create_transport(config)
        assert isinstance(transport, StdioTransport)
        assert transport._cwd == "/tmp/workdir"

    def test_stdio_empty_env_passes_none(self) -> None:
        """stdio transport with empty env dict passes None to StdioTransport."""
        config = TransportConfig(
            type="stdio",
            command=["python", "-m", "plugin"],
            env={},
        )
        transport = create_transport(config)
        assert isinstance(transport, StdioTransport)
        # Empty dict becomes None via `env or None`
        assert transport._env is None


class TestTransportConfigFromDict:
    """Tests for TransportConfig.from_dict()."""

    def test_round_trip_stdio(self) -> None:
        """Round-trip a stdio config through from_dict."""
        data = {
            "transport": "stdio",
            "command": ["node", "server.js"],
            "env": {"NODE_ENV": "production"},
            "cwd": "/app",
            "timeout": 60.0,
        }
        config = TransportConfig.from_dict(data)
        assert config.type == "stdio"
        assert config.command == ["node", "server.js"]
        assert config.env == {"NODE_ENV": "production"}
        assert config.cwd == "/app"
        assert config.timeout == 60.0

    def test_round_trip_grpc(self) -> None:
        """Round-trip a gRPC config through from_dict."""
        data = {
            "transport": "grpc",
            "address": "localhost:50051",
            "timeout": 10.0,
        }
        config = TransportConfig.from_dict(data)
        assert config.type == "grpc"
        assert config.address == "localhost:50051"
        assert config.timeout == 10.0

    def test_round_trip_websocket(self) -> None:
        """Round-trip a WebSocket config through from_dict."""
        data = {
            "transport": "websocket",
            "url": "ws://localhost:8080/cap",
        }
        config = TransportConfig.from_dict(data)
        assert config.type == "websocket"
        assert config.url == "ws://localhost:8080/cap"

    def test_empty_dict_defaults(self) -> None:
        """Empty dict produces all defaults."""
        config = TransportConfig.from_dict({})
        assert config.type == "stdio"
        assert config.command == []
        assert config.env == {}
        assert config.cwd is None
        assert config.address == ""
        assert config.url == ""
        assert config.timeout == 30.0

    def test_extra_keys_ignored(self) -> None:
        """Extra keys in the dict are silently ignored."""
        data = {
            "transport": "stdio",
            "command": ["echo"],
            "unknown_field": "should be ignored",
        }
        config = TransportConfig.from_dict(data)
        assert config.type == "stdio"
        assert config.command == ["echo"]


class TestTransportConfigDefaults:
    """Tests for TransportConfig default values."""

    def test_default_type(self) -> None:
        """Default transport type is stdio."""
        config = TransportConfig()
        assert config.type == "stdio"

    def test_default_command_is_empty(self) -> None:
        """Default command is empty list."""
        config = TransportConfig()
        assert config.command == []

    def test_default_env_is_empty(self) -> None:
        """Default env is empty dict."""
        config = TransportConfig()
        assert config.env == {}

    def test_default_cwd_is_none(self) -> None:
        """Default cwd is None."""
        config = TransportConfig()
        assert config.cwd is None

    def test_default_address_is_empty(self) -> None:
        """Default address is empty string."""
        config = TransportConfig()
        assert config.address == ""

    def test_default_url_is_empty(self) -> None:
        """Default url is empty string."""
        config = TransportConfig()
        assert config.url == ""

    def test_default_timeout(self) -> None:
        """Default timeout is 30 seconds."""
        config = TransportConfig()
        assert config.timeout == 30.0


class TestListAvailableTransports:
    """Tests for list_available_transports()."""

    def test_stdio_always_available(self) -> None:
        """stdio is always in the available transports list."""
        available = list_available_transports()
        assert "stdio" in available

    def test_tcp_always_available(self) -> None:
        """tcp is always in the available transports list (stdlib asyncio)."""
        available = list_available_transports()
        assert "tcp" in available

    def test_returns_list(self) -> None:
        """Return type is a list of strings."""
        available = list_available_transports()
        assert isinstance(available, list)
        for item in available:
            assert isinstance(item, str)


class TestPluginsConfigIntegration:
    """Integration tests: plugins section in config loading."""

    def test_plugins_config_from_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """Plugin servers are parsed from config YAML."""
        from activecontext.config import load_config, reset_config

        reset_config()
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text(
            """
plugins:
  servers:
    - name: code-graph
      transport: stdio
      command: ["python", "-m", "code_graph"]
      connect: auto
    - name: remote-analyzer
      transport: grpc
      url: "localhost:50051"
      timeout: 15.0
"""
        )
        from activecontext.config.schema import PluginConnectMode

        config = load_config(session_root=str(tmp_path))
        assert len(config.plugins.servers) == 2

        cg = config.plugins.servers[0]
        assert cg.name == "code-graph"
        assert cg.transport == "stdio"
        assert cg.command == ["python", "-m", "code_graph"]
        assert cg.connect == PluginConnectMode.AUTO

        ra = config.plugins.servers[1]
        assert ra.name == "remote-analyzer"
        assert ra.transport == "grpc"
        assert ra.url == "localhost:50051"
        assert ra.timeout == 15.0
        assert ra.connect == PluginConnectMode.MANUAL

    def test_empty_plugins_config(self, tmp_path: pytest.TempPathFactory) -> None:
        """Missing plugins section gives empty defaults."""
        from activecontext.config import load_config, reset_config

        reset_config()
        config = load_config(session_root=str(tmp_path))
        assert config.plugins.servers == []

    def test_plugins_server_without_name_skipped(self, tmp_path: pytest.TempPathFactory) -> None:
        """Plugin server entries without a name are skipped."""
        from activecontext.config import load_config, reset_config

        reset_config()
        config_dir = tmp_path / ".ac"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text(
            """
plugins:
  servers:
    - transport: stdio
      command: ["no-name-server"]
    - name: valid
      command: ["valid-server"]
"""
        )
        config = load_config(session_root=str(tmp_path))
        assert len(config.plugins.servers) == 1
        assert config.plugins.servers[0].name == "valid"
