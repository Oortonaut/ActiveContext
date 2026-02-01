"""Tests for CAP Transport and Connection layers.

Tests the transport and connection without actual subprocesses by
mocking the asyncio subprocess. For integration tests with real
servers, see test_plugin_e2e.py (np-tests-e2e task).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.plugins.connection import PluginConnection
from activecontext.plugins.transport import (
    JsonRpcError,
    PluginTransport,
    PluginTransportError,
    _expand_env_vars,
)
from activecontext.plugins.wire import (
    ErrorCodes,
    PluginConnectionStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonrpc_response(result: Any, id: int | str) -> bytes:
    """Build a JSON-RPC response line."""
    msg = {"jsonrpc": "2.0", "result": result, "id": id}
    return json.dumps(msg).encode("utf-8") + b"\n"


def _jsonrpc_error(code: int, message: str, id: int | str) -> bytes:
    """Build a JSON-RPC error response line."""
    msg = {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": id,
    }
    return json.dumps(msg).encode("utf-8") + b"\n"


def _jsonrpc_notification(method: str, params: dict[str, Any]) -> bytes:
    """Build a JSON-RPC notification line."""
    msg = {"jsonrpc": "2.0", "method": method, "params": params}
    return json.dumps(msg).encode("utf-8") + b"\n"


class MockStdout:
    """Mock stdout stream that yields lines from an async queue.

    Lines can be pre-loaded or pushed dynamically. The reader blocks
    on readline() until a line is available or EOF is signalled.
    """

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        if lines:
            for line in lines:
                self._queue.put_nowait(line)

    def push(self, line: bytes) -> None:
        """Push a line to be read by readline()."""
        self._queue.put_nowait(line)

    def push_eof(self) -> None:
        """Signal EOF."""
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


class MockStdin:
    """Mock stdin stream that captures writes and can signal when data arrives."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.write_event = asyncio.Event()

    def write(self, data: bytes) -> None:
        self.written.append(data)
        self.write_event.set()

    async def drain(self) -> None:
        pass

    def get_messages(self) -> list[dict[str, Any]]:
        """Parse all written messages."""
        messages = []
        for data in self.written:
            for line in data.decode("utf-8").strip().split("\n"):
                if line:
                    messages.append(json.loads(line))
        return messages


# ---------------------------------------------------------------------------
# Transport tests
# ---------------------------------------------------------------------------


class TestPluginTransport:
    """Low-level transport tests."""

    @pytest.mark.asyncio
    async def test_not_started_raises(self) -> None:
        transport = PluginTransport(command=["echo"])
        with pytest.raises(PluginTransportError, match="not running"):
            await transport.send_request("test")

    @pytest.mark.asyncio
    async def test_double_start_raises(self) -> None:
        transport = PluginTransport(command=["echo"])
        stdout = MockStdout()
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = MockStdin()
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.start()
            with pytest.raises(PluginTransportError, match="already started"):
                await transport.start()
            stdout.push_eof()
            await transport.stop()

    @pytest.mark.asyncio
    async def test_send_request_and_receive_response(self) -> None:
        """Full request/response cycle."""
        stdin = MockStdin()
        stdout = MockStdout()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = stdin
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        transport = PluginTransport(command=["test"])

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.start()

            # Schedule response after request is sent
            async def respond() -> None:
                await stdin.write_event.wait()
                stdout.push(_jsonrpc_response({"status": "ok"}, 1))

            asyncio.create_task(respond())

            result = await transport.send_request("test/method", {"key": "val"})
            assert result == {"status": "ok"}

            # Verify the request was written
            messages = stdin.get_messages()
            assert len(messages) >= 1
            req = messages[0]
            assert req["method"] == "test/method"
            assert req["id"] == 1
            assert req["params"]["key"] == "val"

            stdout.push_eof()
            await transport.stop()

    @pytest.mark.asyncio
    async def test_error_response_raises_jsonrpc_error(self) -> None:
        stdin = MockStdin()
        stdout = MockStdout()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = stdin
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        transport = PluginTransport(command=["test"])

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.start()

            async def respond() -> None:
                await stdin.write_event.wait()
                stdout.push(_jsonrpc_error(ErrorCodes.NODE_NOT_FOUND, "Not found", 1))

            asyncio.create_task(respond())

            with pytest.raises(JsonRpcError) as exc_info:
                await transport.send_request("test")

            assert exc_info.value.code == ErrorCodes.NODE_NOT_FOUND
            stdout.push_eof()
            await transport.stop()

    @pytest.mark.asyncio
    async def test_notification_callback(self) -> None:
        """Server → host notifications reach the callback."""
        received: list[tuple[str, dict[str, Any]]] = []

        def on_notif(method: str, params: dict[str, Any]) -> None:
            received.append((method, params))

        stdin = MockStdin()
        stdout = MockStdout()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = stdin
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        transport = PluginTransport(command=["test"], on_notification=on_notif)

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.start()
            stdout.push(_jsonrpc_notification("node/dirty", {"node_id": "sh_1"}))
            # Give reader time to process
            await asyncio.sleep(0.05)
            stdout.push_eof()
            await transport.stop()

        assert len(received) == 1
        assert received[0][0] == "node/dirty"
        assert received[0][1]["node_id"] == "sh_1"

    @pytest.mark.asyncio
    async def test_send_notification(self) -> None:
        stdin = MockStdin()
        stdout = MockStdout()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = stdin
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        transport = PluginTransport(command=["test"])

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.start()
            await transport.send_notification("shutdown", {"reason": "done"})

            messages = stdin.get_messages()
            assert len(messages) == 1
            assert messages[0]["method"] == "shutdown"
            assert "id" not in messages[0]

            stdout.push_eof()
            await transport.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_pending(self) -> None:
        """Stopping transport fails pending requests."""
        stdin = MockStdin()
        stdout = MockStdout()  # Never push a response — just block

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = stdin
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        transport = PluginTransport(command=["test"])

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            await transport.start()

            # Start a request but don't await it yet
            req_task = asyncio.create_task(transport.send_request("test", timeout=10))
            await asyncio.sleep(0.05)

            # Stop should cancel the pending request
            await transport.stop()

            with pytest.raises(PluginTransportError):
                await req_task


class TestExpandEnvVars:
    """Environment variable expansion."""

    def test_simple_expansion(self) -> None:
        import os

        os.environ["_TEST_CAP_VAR"] = "hello"
        try:
            result = _expand_env_vars({"KEY": "${_TEST_CAP_VAR}"})
            assert result["KEY"] == "hello"
        finally:
            del os.environ["_TEST_CAP_VAR"]

    def test_missing_var_empty(self) -> None:
        result = _expand_env_vars({"KEY": "${_NONEXISTENT_CAP_VAR_XYZ}"})
        assert result["KEY"] == ""

    def test_no_expansion(self) -> None:
        result = _expand_env_vars({"KEY": "plain_value"})
        assert result["KEY"] == "plain_value"


# ---------------------------------------------------------------------------
# Connection tests
# ---------------------------------------------------------------------------


class TestPluginConnection:
    """High-level connection tests."""

    def test_initial_state(self) -> None:
        conn = PluginConnection(name="test", command=["echo"])
        assert conn.status == PluginConnectionStatus.DISCONNECTED
        assert conn.node_types == []
        assert conn.descriptors == []

    @pytest.mark.asyncio
    async def test_connect_handshake(self) -> None:
        """Full connect/disconnect cycle with mocked transport."""
        init_result = {
            "server_name": "test-server",
            "server_version": "1.0.0",
            "protocol_version": "1.0",
            "node_types": [
                {
                    "node_type": "lint",
                    "description": "Lint check",
                    "constructor": {
                        "positional": [{"name": "path", "type": "str"}],
                        "named": [
                            {"name": "fix", "type": "bool", "default": False},
                        ],
                    },
                    "properties": [
                        {
                            "name": "errors",
                            "type": "int",
                            "readable": True,
                        },
                    ],
                    "methods": [
                        {"name": "recheck", "description": "Run again"},
                    ],
                },
            ],
            "server_capabilities": {
                "sync": True,
                "immediate_call": True,
                "push_dirty": True,
                "push_notifications": True,
            },
        }

        stdin = MockStdin()
        stdout = MockStdout()

        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.pid = 12345
        mock_process.stdout = stdout
        mock_process.stdin = stdin
        mock_process.stderr = MagicMock()
        mock_process.terminate = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = AsyncMock()

        conn = PluginConnection(name="test-plugin", command=["test"])

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Push response after the initialize request is sent
            async def respond() -> None:
                await stdin.write_event.wait()
                stdout.push(_jsonrpc_response(init_result, 1))

            asyncio.create_task(respond())

            await conn.connect(session_id="sess_1", cwd="/project")

            assert conn.status == PluginConnectionStatus.CONNECTED
            assert conn.server_name == "test-server"
            assert conn.server_version == "1.0.0"
            assert conn.server_capabilities.immediate_call is True

            # Node types parsed correctly
            assert len(conn.node_types) == 1
            schema = conn.node_types[0]
            assert schema.node_type == "lint"
            assert len(schema.constructor.positional) == 1
            assert schema.constructor.positional[0].name == "path"
            assert len(schema.constructor.named) == 1
            assert schema.properties[0].name == "errors"
            assert schema.methods[0].name == "recheck"

            # Descriptors generated
            assert len(conn.descriptors) == 1
            desc = conn.descriptors[0]
            assert desc.node_type == "lint"
            assert desc.source.value == "remote_plugin"
            assert desc.server_name == "test-plugin"

            stdout.push_eof()
            await conn.disconnect()
            assert conn.status == PluginConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_failure_sets_error_status(self) -> None:
        conn = PluginConnection(name="bad", command=["nonexistent_command"])

        with pytest.raises(PluginTransportError):
            await conn.connect()

        assert conn.status == PluginConnectionStatus.ERROR

    @pytest.mark.asyncio
    async def test_disconnect_when_already_disconnected(self) -> None:
        conn = PluginConnection(name="test", command=["echo"])
        # Should not raise
        await conn.disconnect()
        assert conn.status == PluginConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send_request_when_not_connected(self) -> None:
        conn = PluginConnection(name="test", command=["echo"])
        with pytest.raises(PluginTransportError, match="Not connected"):
            await conn.send_request("test")

    @pytest.mark.asyncio
    async def test_send_notification_when_not_connected(self) -> None:
        conn = PluginConnection(name="test", command=["echo"])
        with pytest.raises(PluginTransportError, match="Not connected"):
            await conn.send_notification("test")
