"""Tests for the CAPTransport protocol extraction.

Verifies:
- CAPTransport is runtime_checkable
- StdioTransport (PluginTransport) satisfies CAPTransport
- PluginConnection accepts injected transports
- Backward compatibility of PluginTransport import
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activecontext.plugins.cap_transport import CAPTransport, NotificationCallback
from activecontext.plugins.connection import PluginConnection
from activecontext.plugins.transport import PluginTransport, StdioTransport
from activecontext.plugins.wire import PluginConnectionStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonrpc_response(result: Any, id: int | str) -> bytes:
    """Build a JSON-RPC response line."""
    msg = {"jsonrpc": "2.0", "result": result, "id": id}
    return json.dumps(msg).encode("utf-8") + b"\n"


class MockStdout:
    """Mock stdout stream that yields lines from an async queue."""

    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        if lines:
            for line in lines:
                self._queue.put_nowait(line)

    def push(self, line: bytes) -> None:
        self._queue.put_nowait(line)

    def push_eof(self) -> None:
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


class MockStdin:
    """Mock stdin stream that captures writes."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.write_event = asyncio.Event()

    def write(self, data: bytes) -> None:
        self.written.append(data)
        self.write_event.set()

    async def drain(self) -> None:
        pass


class FakeTransport:
    """Minimal CAPTransport implementation for injection tests."""

    def __init__(self) -> None:
        self._running = False
        self.started = False
        self.stopped = False
        self.requests: list[tuple[str, Any]] = []
        self.notifications: list[tuple[str, Any]] = []
        self._response: Any = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True
        self.started = True

    async def stop(self) -> None:
        self._running = False
        self.stopped = True

    async def send_request(
        self, method: str, params: Any = None, timeout: float = 30.0
    ) -> Any:
        self.requests.append((method, params))
        return self._response

    async def send_notification(self, method: str, params: Any = None) -> None:
        self.notifications.append((method, params))

    async def send_response(self, result: Any, request_id: int | str) -> None:
        pass

    async def send_error_response(
        self,
        code: int,
        message: str,
        request_id: int | str | None,
        data: Any = None,
    ) -> None:
        pass

    def set_response(self, response: Any) -> None:
        """Pre-set the response for the next send_request call."""
        self._response = response


# ---------------------------------------------------------------------------
# Protocol compliance tests
# ---------------------------------------------------------------------------


class TestCAPTransportProtocol:
    """Verify CAPTransport protocol properties."""

    def test_runtime_checkable(self) -> None:
        """CAPTransport is decorated with @runtime_checkable."""
        assert hasattr(CAPTransport, "__protocol_attrs__") or isinstance(
            CAPTransport, type
        )
        # The key check: isinstance works at runtime
        fake = FakeTransport()
        assert isinstance(fake, CAPTransport)

    def test_stdio_transport_satisfies_protocol(self) -> None:
        """StdioTransport (PluginTransport) is a CAPTransport."""
        transport = StdioTransport(command=["echo"])
        assert isinstance(transport, CAPTransport)

    def test_plugin_transport_is_stdio_transport(self) -> None:
        """PluginTransport and StdioTransport are the same class."""
        assert StdioTransport is PluginTransport

    def test_non_conforming_object_fails_isinstance(self) -> None:
        """An object missing protocol methods is not a CAPTransport."""

        class NotATransport:
            pass

        assert not isinstance(NotATransport(), CAPTransport)


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure existing imports and usage still work."""

    def test_plugin_transport_importable(self) -> None:
        """PluginTransport is still importable from transport module."""
        from activecontext.plugins.transport import PluginTransport as PT

        assert PT is not None
        assert PT is StdioTransport

    def test_stdio_transport_importable_from_package(self) -> None:
        """StdioTransport is importable from the plugins package."""
        from activecontext.plugins import StdioTransport as ST

        assert ST is PluginTransport

    def test_cap_transport_importable_from_package(self) -> None:
        """CAPTransport is importable from the plugins package."""
        from activecontext.plugins import CAPTransport as CT

        assert CT is CAPTransport

    def test_error_classes_still_importable(self) -> None:
        """Error classes remain importable from transport module."""
        from activecontext.plugins.transport import JsonRpcError as JE
        from activecontext.plugins.transport import PluginTransportError as PTE

        assert issubclass(JE, PTE)

    def test_notification_callback_re_exported(self) -> None:
        """NotificationCallback is available from cap_transport module."""
        from activecontext.plugins.cap_transport import NotificationCallback as NC

        assert NC is NotificationCallback


# ---------------------------------------------------------------------------
# Transport injection tests
# ---------------------------------------------------------------------------


class TestTransportInjection:
    """PluginConnection accepts an injected CAPTransport."""

    def test_connection_accepts_transport_parameter(self) -> None:
        """PluginConnection.__init__ accepts a transport keyword argument."""
        fake = FakeTransport()
        conn = PluginConnection(
            name="test",
            command=["echo"],
            transport=fake,
        )
        # The injected transport is stored
        assert conn._injected_transport is fake

    def test_connection_without_transport_has_none(self) -> None:
        """Without transport parameter, _injected_transport is None."""
        conn = PluginConnection(name="test", command=["echo"])
        assert conn._injected_transport is None

    @pytest.mark.asyncio
    async def test_connect_uses_injected_transport(self) -> None:
        """When transport is injected, connect() uses it instead of creating StdioTransport."""
        fake = FakeTransport()
        init_result = {
            "server_name": "fake-server",
            "server_version": "0.1.0",
            "protocol_version": "1.0",
            "node_types": [],
            "server_capabilities": {},
        }
        fake.set_response(init_result)

        conn = PluginConnection(
            name="test-injected",
            command=["should-not-be-spawned"],
            transport=fake,
        )

        await conn.connect(session_id="sess_1")

        # Transport was started
        assert fake.started
        # Connection is established
        assert conn.status == PluginConnectionStatus.CONNECTED
        assert conn.server_name == "fake-server"

        # Disconnect should stop the transport
        await conn.disconnect()
        assert fake.stopped
        assert conn.status == PluginConnectionStatus.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_creates_stdio_when_no_injection(self) -> None:
        """Without injection, connect() creates a StdioTransport as before."""
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

        init_result = {
            "server_name": "stdio-server",
            "server_version": "1.0.0",
            "protocol_version": "1.0",
            "node_types": [],
            "server_capabilities": {},
        }

        conn = PluginConnection(
            name="test-stdio",
            command=["test-cmd"],
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            async def respond() -> None:
                await stdin.write_event.wait()
                stdout.push(_jsonrpc_response(init_result, 1))

            asyncio.create_task(respond())

            await conn.connect(session_id="sess_2")

            assert conn.status == PluginConnectionStatus.CONNECTED
            assert conn.server_name == "stdio-server"
            # The transport created is a StdioTransport instance
            assert isinstance(conn._transport, StdioTransport)

            stdout.push_eof()
            await conn.disconnect()
