"""CAP Connection — handshake and session management.

Wraps PluginTransport with protocol-level concerns:
- Initialize handshake (exchange capabilities and schemas)
- Connection state machine
- Graceful shutdown
- Node type registration from server schemas
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from activecontext.plugins.cap_transport import CAPTransport
from activecontext.plugins.descriptor import NodePluginDescriptor, PluginSource
from activecontext.plugins.transport import (
    NotificationCallback,
    PluginTransportError,
    StdioTransport,
)
from activecontext.plugins.wire import (
    PROTOCOL_VERSION,
    ConstructorSchema,
    HostCapabilities,
    InitializeParams,
    InitializeResult,
    Methods,
    NodeTypeSchema,
    ParamSchema,
    PluginConnectionStatus,
    RootInfo,
    ServerCapabilities,
)

logger = logging.getLogger(__name__)

# Type for host API request handler: (method, params, request_id) -> result
HostAPIHandler = Callable[[str, dict[str, Any], int | str], Any]


class PluginConnection:
    """High-level connection to a CAP plugin server.

    Manages the full lifecycle:
    1. Spawn subprocess via PluginTransport
    2. Send initialize request, receive capabilities + schemas
    3. Register node types from schemas
    4. Route notifications and host API calls
    5. Graceful shutdown

    Usage:
        conn = PluginConnection(
            name="my-plugin",
            command=["python", "-m", "my_plugin"],
        )
        await conn.connect(session_id="sess_abc", cwd="/project")
        # conn.node_types contains schemas
        # conn.server_capabilities tells what the server supports
        result = await conn.send_request("node/create", {...})
        await conn.disconnect()
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        on_notification: NotificationCallback | None = None,
        on_host_api: HostAPIHandler | None = None,
        transport: CAPTransport | None = None,
    ) -> None:
        """Initialize connection configuration.

        Args:
            name: Human-readable server name for logging.
            command: Command to spawn the plugin server.
            env: Additional environment variables.
            cwd: Working directory for the subprocess.
            on_notification: Callback for server push notifications.
            on_host_api: Handler for server → host API requests.
            transport: Optional pre-built transport (bypasses StdioTransport creation).
        """
        self.name = name
        self._command = command
        self._env = env
        self._cwd = cwd
        self._user_notification_cb = on_notification
        self._host_api_handler = on_host_api

        self._injected_transport = transport
        self._transport: CAPTransport | None = None
        self._status = PluginConnectionStatus.DISCONNECTED
        self._server_name: str = ""
        self._server_version: str = ""
        self._server_capabilities = ServerCapabilities()
        self._node_types: list[NodeTypeSchema] = []
        self._descriptors: list[NodePluginDescriptor] = []

    @property
    def status(self) -> PluginConnectionStatus:
        """Current connection status."""
        return self._status

    @property
    def server_name(self) -> str:
        """Server's self-reported name (from initialize result)."""
        return self._server_name

    @property
    def server_version(self) -> str:
        """Server's self-reported version."""
        return self._server_version

    @property
    def server_capabilities(self) -> ServerCapabilities:
        """Server's declared capabilities."""
        return self._server_capabilities

    @property
    def node_types(self) -> list[NodeTypeSchema]:
        """Node type schemas advertised by the server."""
        return self._node_types

    @property
    def descriptors(self) -> list[NodePluginDescriptor]:
        """Plugin descriptors generated from server schemas."""
        return self._descriptors

    async def connect(
        self,
        session_id: str = "",
        cwd: str = "",
        roots: list[RootInfo] | None = None,
    ) -> InitializeResult:
        """Connect to the plugin server and perform handshake.

        Spawns the subprocess, sends initialize request, and parses
        the server's capabilities and node type schemas.

        Args:
            session_id: Session identifier for logging.
            cwd: Working directory for render methods.
            roots: Filesystem roots to expose to the server.

        Returns:
            The server's initialize result.

        Raises:
            PluginTransportError: If connection or handshake fails.
        """
        if self._status != PluginConnectionStatus.DISCONNECTED:
            raise PluginTransportError(f"Cannot connect: status is {self._status.value}")

        self._status = PluginConnectionStatus.CONNECTING

        try:
            # Create and start transport
            if self._injected_transport is not None:
                self._transport = self._injected_transport
            else:
                self._transport = StdioTransport(
                    command=self._command,
                    env=self._env,
                    cwd=self._cwd,
                    on_notification=self._on_message,
                )
            await self._transport.start()

            # Send initialize handshake
            params = InitializeParams(
                protocol_version=PROTOCOL_VERSION,
                host_capabilities=HostCapabilities(),
                roots=roots or [],
                session_id=session_id,
                cwd=cwd,
            )
            raw_result = await self._transport.send_request(
                Methods.INITIALIZE, params, timeout=30.0
            )

            # Parse result
            result = self._parse_initialize_result(raw_result)
            self._server_name = result.server_name
            self._server_version = result.server_version
            self._server_capabilities = result.server_capabilities
            self._node_types = result.node_types

            # Build descriptors for each node type
            self._descriptors = [
                NodePluginDescriptor(
                    node_type=schema.node_type,
                    source=PluginSource.REMOTE,
                    schema=schema,
                    server_name=self.name,
                )
                for schema in result.node_types
            ]

            self._status = PluginConnectionStatus.CONNECTED
            logger.info(
                "CAP connected to '%s' (%s v%s): %d node types",
                self.name,
                self._server_name,
                self._server_version,
                len(self._node_types),
            )
            return result

        except Exception as e:
            self._status = PluginConnectionStatus.ERROR
            logger.error("CAP connection failed for '%s': %s", self.name, e)
            # Clean up transport if it was started
            if self._transport:
                await self._transport.stop()
                self._transport = None
            raise

    async def disconnect(self) -> None:
        """Gracefully disconnect from the plugin server.

        Sends shutdown notification, then stops the transport.
        """
        if self._status == PluginConnectionStatus.DISCONNECTED:
            return

        if self._transport and self._transport.is_running:
            try:
                await self._transport.send_notification(Methods.SHUTDOWN)
            except Exception:
                pass  # Best-effort shutdown notification
            await self._transport.stop()

        self._transport = None
        self._status = PluginConnectionStatus.DISCONNECTED
        self._node_types = []
        self._descriptors = []
        logger.info("CAP disconnected from '%s'", self.name)

    async def send_request(self, method: str, params: Any = None, timeout: float = 30.0) -> Any:
        """Send a request to the plugin server.

        Args:
            method: RPC method name.
            params: Parameters (dataclass or dict).
            timeout: Seconds to wait for response.

        Returns:
            The result from the server.

        Raises:
            PluginTransportError: If not connected.
        """
        if self._status != PluginConnectionStatus.CONNECTED:
            raise PluginTransportError(f"Not connected (status: {self._status.value})")
        assert self._transport is not None
        return await self._transport.send_request(method, params, timeout)

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Send a notification to the plugin server.

        Args:
            method: RPC method name.
            params: Parameters (dataclass or dict).
        """
        if self._status != PluginConnectionStatus.CONNECTED:
            raise PluginTransportError(f"Not connected (status: {self._status.value})")
        assert self._transport is not None
        await self._transport.send_notification(method, params)

    def _on_message(self, method: str, params: dict[str, Any]) -> None:
        """Dispatch incoming server messages.

        Routes to either:
        - Host API handler (for server → host requests with _request_id)
        - User notification callback (for push notifications)
        """
        request_id = params.pop("_request_id", None)

        if request_id is not None and self._host_api_handler:
            # Server → host request (host API call)
            try:
                result = self._host_api_handler(method, params, request_id)
                # If handler returns a coroutine, we can't await it here
                # (we're in a sync callback). The connection layer or
                # manager should handle async host API calls.
                if result is not None and self._transport:
                    # Schedule response send
                    import asyncio

                    asyncio.get_event_loop().call_soon(
                        lambda: asyncio.ensure_future(
                            self._transport.send_response(result, request_id)  # type: ignore[union-attr]
                        )
                    )
            except Exception as e:
                logger.error("CAP host API error for %s: %s", method, e)

        if self._user_notification_cb:
            try:
                self._user_notification_cb(method, params)
            except Exception as e:
                logger.error("CAP notification callback error: %s", e)

    def _parse_initialize_result(self, raw: Any) -> InitializeResult:
        """Parse the raw initialize result dict into typed objects."""
        if not isinstance(raw, dict):
            return InitializeResult(server_name=self.name)

        # Parse server capabilities
        raw_caps = raw.get("server_capabilities", {})
        capabilities = ServerCapabilities(
            sync=raw_caps.get("sync", True),
            immediate_call=raw_caps.get("immediate_call", False),
            push_dirty=raw_caps.get("push_dirty", True),
            push_notifications=raw_caps.get("push_notifications", True),
        )

        # Parse node type schemas
        node_types: list[NodeTypeSchema] = []
        for raw_schema in raw.get("node_types", []):
            schema = self._parse_node_type_schema(raw_schema)
            node_types.append(schema)

        return InitializeResult(
            server_name=raw.get("server_name", self.name),
            server_version=raw.get("server_version", ""),
            protocol_version=raw.get("protocol_version", PROTOCOL_VERSION),
            node_types=node_types,
            server_capabilities=capabilities,
        )

    def _parse_node_type_schema(self, raw: dict[str, Any]) -> NodeTypeSchema:
        """Parse a single node type schema from wire format."""
        from activecontext.plugins.wire import (
            MethodSchema,
            PropertySchema,
        )

        # Parse constructor
        raw_ctor = raw.get("constructor", {})
        constructor = ConstructorSchema(
            positional=[self._parse_param(p) for p in raw_ctor.get("positional", [])],
            variadic=(
                self._parse_param(raw_ctor["variadic"]) if raw_ctor.get("variadic") else None
            ),
            named=[self._parse_param(p) for p in raw_ctor.get("named", [])],
        )

        # Parse properties
        properties = [
            PropertySchema(
                name=p["name"],
                type=p.get("type", "Any"),
                readable=p.get("readable", True),
                writable=p.get("writable", False),
                description=p.get("description", ""),
            )
            for p in raw.get("properties", [])
        ]

        # Parse methods
        methods = [
            MethodSchema(
                name=m["name"],
                params=[self._parse_param(p) for p in m.get("params", [])],
                returns=m.get("returns", "None"),
                description=m.get("description", ""),
                chainable=m.get("chainable", False),
            )
            for m in raw.get("methods", [])
        ]

        return NodeTypeSchema(
            node_type=raw["node_type"],
            description=raw.get("description", ""),
            constructor=constructor,
            properties=properties,
            methods=methods,
        )

    def _parse_param(self, raw: dict[str, Any]) -> ParamSchema:
        """Parse a parameter schema from wire format."""
        from activecontext.plugins.wire import MISSING

        return ParamSchema(
            name=raw["name"],
            type=raw.get("type", "str"),
            default=raw.get("default", MISSING),
            description=raw.get("description", ""),
        )
