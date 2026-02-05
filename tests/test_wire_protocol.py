"""Tests for the Node Plugin Wire Protocol.

Verifies that:
1. All wire protocol dataclasses are well-formed
2. JSON-RPC serialization helpers produce valid messages
3. Sentinel MISSING value works correctly
4. Schema types compose properly
"""

from __future__ import annotations

from dataclasses import asdict

from activecontext.plugins.wire import (
    MISSING,
    PROTOCOL_VERSION,
    ConstructorSchema,
    ErrorCodes,
    HostCapabilities,
    HostCreateNodeParams,
    HostCreateNodeResult,
    HostInvokeParams,
    HostInvokeResult,
    HostQueryRootsResult,
    HostResolveRootParams,
    HostResolveRootResult,
    InitializeParams,
    InitializeResult,
    Methods,
    MethodSchema,
    NodeCallParams,
    NodeCallResult,
    NodeCreateParams,
    NodeCreateResult,
    NodeDestroyParams,
    NodeDirtyParams,
    NodeNotificationParams,
    NodeSerializeParams,
    NodeSerializeResult,
    NodeSyncParams,
    NodeSyncResult,
    NodeTypeSchema,
    ParamSchema,
    PendingCall,
    PluginConnectionStatus,
    PropertySchema,
    RootInfo,
    ServerCapabilities,
    _MissingSentinel,
    to_jsonrpc_error,
    to_jsonrpc_notification,
    to_jsonrpc_request,
    to_jsonrpc_response,
)


class TestMissingSentinel:
    """The MISSING sentinel for required parameters."""

    def test_singleton(self) -> None:
        """MISSING is a singleton."""
        a = _MissingSentinel()
        b = _MissingSentinel()
        assert a is b
        assert a is MISSING

    def test_repr(self) -> None:
        assert repr(MISSING) == "MISSING"

    def test_falsy(self) -> None:
        assert not MISSING
        assert bool(MISSING) is False

    def test_is_not_none(self) -> None:
        """MISSING is distinct from None."""
        assert MISSING is not None


class TestParamSchema:
    """ParamSchema with MISSING default for required params."""

    def test_required_param(self) -> None:
        p = ParamSchema(name="command")
        assert p.name == "command"
        assert p.default is MISSING
        assert p.type == "str"

    def test_optional_param_with_none(self) -> None:
        p = ParamSchema(name="timeout", type="int", default=None)
        assert p.default is None
        assert p.default is not MISSING

    def test_optional_param_with_value(self) -> None:
        p = ParamSchema(name="verbose", type="bool", default=False)
        assert p.default is False

    def test_description(self) -> None:
        p = ParamSchema(name="x", description="The x coordinate")
        assert p.description == "The x coordinate"


class TestNodeTypeSchema:
    """Full node type schema composition."""

    def test_minimal_schema(self) -> None:
        schema = NodeTypeSchema(node_type="custom")
        assert schema.node_type == "custom"
        assert schema.properties == []
        assert schema.methods == []

    def test_full_schema(self) -> None:
        schema = NodeTypeSchema(
            node_type="ShellNode",
            description="Async shell command execution",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="command")],
                variadic=ParamSchema(name="args"),
                named=[
                    ParamSchema(name="timeout", type="int", default=120),
                    ParamSchema(name="cwd", type="str", default=""),
                ],
            ),
            properties=[
                PropertySchema(name="is_complete", type="bool", readable=True),
                PropertySchema(name="output", type="str", readable=True),
                PropertySchema(name="exit_code", type="int", readable=True),
            ],
            methods=[
                MethodSchema(
                    name="cancel",
                    description="Cancel the running command",
                ),
                MethodSchema(
                    name="set_timeout",
                    params=[ParamSchema(name="seconds", type="int")],
                    returns="ShellNode",
                    chainable=True,
                ),
            ],
        )
        assert schema.node_type == "ShellNode"
        assert len(schema.constructor.positional) == 1
        assert schema.constructor.variadic is not None
        assert schema.constructor.variadic.name == "args"
        assert len(schema.constructor.named) == 2
        assert len(schema.properties) == 3
        assert len(schema.methods) == 2
        assert schema.methods[1].chainable is True

    def test_asdict_roundtrip(self) -> None:
        """Schema can be serialized to dict (for JSON-RPC)."""
        schema = NodeTypeSchema(
            node_type="test",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="x", type="int")],
            ),
        )
        d = asdict(schema)
        assert d["node_type"] == "test"
        assert d["constructor"]["positional"][0]["name"] == "x"


class TestInitializeHandshake:
    """Initialize request/result messages."""

    def test_params_defaults(self) -> None:
        params = InitializeParams()
        assert params.protocol_version == PROTOCOL_VERSION
        assert params.host_capabilities.create_node is True
        assert params.roots == []
        assert params.session_id == ""

    def test_params_with_roots(self) -> None:
        params = InitializeParams(
            session_id="sess_abc",
            cwd="/project",
            roots=[
                RootInfo(uri="file:///project", name="cwd"),
                RootInfo(uri="file:///home/user", name="home"),
            ],
        )
        assert len(params.roots) == 2
        assert params.roots[0].name == "cwd"

    def test_result_with_node_types(self) -> None:
        result = InitializeResult(
            server_name="shell-plugin",
            server_version="1.0.0",
            node_types=[
                NodeTypeSchema(node_type="shell", description="Shell commands"),
            ],
        )
        assert result.server_name == "shell-plugin"
        assert len(result.node_types) == 1
        assert result.server_capabilities.sync is True

    def test_host_capabilities_defaults(self) -> None:
        caps = HostCapabilities()
        assert caps.create_node is True
        assert caps.invoke is True
        assert caps.roots is True
        assert caps.notifications is True

    def test_server_capabilities_defaults(self) -> None:
        caps = ServerCapabilities()
        assert caps.sync is True
        assert caps.immediate_call is False
        assert caps.push_dirty is True
        assert caps.push_notifications is True


class TestNodeManagement:
    """Node create/sync/call/destroy messages."""

    def test_create_params(self) -> None:
        params = NodeCreateParams(
            node_type="ShellNode",
            args=["pytest"],
            kwargs={"timeout": 120},
            node_id="sh_abc123",
        )
        assert params.node_type == "ShellNode"
        assert params.args == ["pytest"]
        assert params.node_id == "sh_abc123"

    def test_create_result(self) -> None:
        result = NodeCreateResult(
            node_id="sh_abc123",
            state={"command": "pytest", "status": "running"},
            renders={"header": "### Shell: pytest", "content": "", "detail": ""},
            tokens={"title": 10, "content": 0, "detail": 0},
            digest={"id": "sh_abc123", "type": "shell"},
        )
        assert result.node_id == "sh_abc123"
        assert result.state["status"] == "running"

    def test_sync_params_with_calls(self) -> None:
        params = NodeSyncParams(
            node_id="sh_abc123",
            calls=[
                PendingCall(method="set_timeout", args=[60]),
                PendingCall(method="cancel"),
            ],
            cwd="/project",
        )
        assert len(params.calls) == 2
        assert params.calls[0].method == "set_timeout"
        assert params.calls[1].method == "cancel"

    def test_sync_result(self) -> None:
        result = NodeSyncResult(
            state={"command": "pytest", "status": "completed"},
            renders={"header": "h", "content": "c", "detail": "d"},
            tokens={"title": 10, "content": 50, "detail": 200},
            digest={"id": "sh_1", "type": "shell"},
            notifications=[
                {"description": "Shell completed", "level": "wake"},
            ],
        )
        assert result.state["status"] == "completed"
        assert len(result.notifications) == 1

    def test_call_params(self) -> None:
        params = NodeCallParams(
            node_id="sh_abc",
            method="cancel",
        )
        assert params.method == "cancel"
        assert params.args == []

    def test_call_result_state_changed(self) -> None:
        result = NodeCallResult(result=True, state_changed=True)
        assert result.result is True
        assert result.state_changed is True

    def test_destroy_params(self) -> None:
        params = NodeDestroyParams(node_id="sh_abc")
        assert params.node_id == "sh_abc"

    def test_serialize_roundtrip(self) -> None:
        params = NodeSerializeParams(node_id="sh_abc")
        result = NodeSerializeResult(
            data={"node_type": "ShellNode", "command": "pytest", "node_id": "sh_abc"},
        )
        assert params.node_id == "sh_abc"
        assert result.data["node_type"] == "ShellNode"


class TestPushNotifications:
    """Server → host push messages."""

    def test_dirty_params(self) -> None:
        params = NodeDirtyParams(node_id="sh_abc")
        assert params.node_id == "sh_abc"

    def test_notification_params(self) -> None:
        params = NodeNotificationParams(
            node_id="sh_abc",
            description="Shell completed with exit code 0",
            level="wake",
        )
        assert params.level == "wake"
        assert "exit code" in params.description

    def test_notification_default_level(self) -> None:
        params = NodeNotificationParams(
            node_id="sh_abc",
            description="Minor update",
        )
        assert params.level == "hold"


class TestHostAPI:
    """Server → host API messages."""

    def test_create_node_params(self) -> None:
        params = HostCreateNodeParams(
            node_type="ArtifactNode",
            kwargs={"content": "output", "artifact_type": "output"},
            parent_id="sh_abc",
        )
        assert params.node_type == "ArtifactNode"
        assert params.parent_id == "sh_abc"

    def test_create_node_result(self) -> None:
        result = HostCreateNodeResult(node_id="art_xyz")
        assert result.node_id == "art_xyz"

    def test_invoke_params(self) -> None:
        params = HostInvokeParams(
            node_id="grp_abc",
            method="set_summary",
            args=["Updated summary"],
        )
        assert params.method == "set_summary"

    def test_invoke_result(self) -> None:
        result = HostInvokeResult(result="ok")
        assert result.result == "ok"

    def test_query_roots_result(self) -> None:
        result = HostQueryRootsResult(
            roots=[
                RootInfo(uri="file:///project", name="cwd", description="Working dir"),
                RootInfo(uri="file:///home/user", name="home"),
            ],
        )
        assert len(result.roots) == 2

    def test_resolve_root_params(self) -> None:
        params = HostResolveRootParams(
            root_uri="file:///project",
            normalized_path="src/main.py",
        )
        assert params.normalized_path == "src/main.py"

    def test_resolve_root_result(self) -> None:
        result = HostResolveRootResult(
            real_path="/project/src/main.py",
            exists=True,
        )
        assert result.exists is True


class TestJsonRpcHelpers:
    """JSON-RPC 2.0 message construction."""

    def test_request_with_dataclass(self) -> None:
        params = NodeCreateParams(node_type="ShellNode", args=["pytest"])
        msg = to_jsonrpc_request(Methods.NODE_CREATE, params, id=1)
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "node/create"
        assert msg["id"] == 1
        assert msg["params"]["node_type"] == "ShellNode"
        assert msg["params"]["args"] == ["pytest"]

    def test_request_with_dict(self) -> None:
        msg = to_jsonrpc_request("custom/method", {"key": "value"}, id="abc")
        assert msg["params"] == {"key": "value"}
        assert msg["id"] == "abc"

    def test_request_without_params(self) -> None:
        msg = to_jsonrpc_request(Methods.SHUTDOWN, None, id=2)
        assert "params" not in msg
        assert msg["method"] == "shutdown"

    def test_notification_no_id(self) -> None:
        params = NodeDirtyParams(node_id="sh_abc")
        msg = to_jsonrpc_notification(Methods.NODE_DIRTY, params)
        assert msg["jsonrpc"] == "2.0"
        assert "id" not in msg
        assert msg["params"]["node_id"] == "sh_abc"

    def test_notification_without_params(self) -> None:
        msg = to_jsonrpc_notification("ping", None)
        assert "params" not in msg
        assert "id" not in msg

    def test_response_with_dataclass(self) -> None:
        result = NodeCreateResult(
            node_id="sh_abc",
            state={"command": "pytest"},
        )
        msg = to_jsonrpc_response(result, id=1)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 1
        assert msg["result"]["node_id"] == "sh_abc"

    def test_response_with_dict(self) -> None:
        msg = to_jsonrpc_response({"status": "ok"}, id=5)
        assert msg["result"] == {"status": "ok"}

    def test_error_response(self) -> None:
        msg = to_jsonrpc_error(
            ErrorCodes.NODE_NOT_FOUND,
            "Node sh_abc not found",
            id=3,
        )
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 3
        assert msg["error"]["code"] == -32000
        assert "not found" in msg["error"]["message"]
        assert "data" not in msg["error"]

    def test_error_with_data(self) -> None:
        msg = to_jsonrpc_error(
            ErrorCodes.INVALID_PARAMS,
            "Missing required field",
            id=4,
            data={"field": "node_type"},
        )
        assert msg["error"]["data"] == {"field": "node_type"}

    def test_error_null_id(self) -> None:
        """Parse errors have null id."""
        msg = to_jsonrpc_error(ErrorCodes.PARSE_ERROR, "Invalid JSON", id=None)
        assert msg["id"] is None


class TestMethodConstants:
    """Method name constants."""

    def test_lifecycle_methods(self) -> None:
        assert Methods.INITIALIZE == "initialize"
        assert Methods.SHUTDOWN == "shutdown"

    def test_node_methods(self) -> None:
        assert Methods.NODE_CREATE == "node/create"
        assert Methods.NODE_SYNC == "node/sync"
        assert Methods.NODE_CALL == "node/call"
        assert Methods.NODE_DESTROY == "node/destroy"
        assert Methods.NODE_SERIALIZE == "node/serialize"

    def test_push_methods(self) -> None:
        assert Methods.NODE_DIRTY == "node/dirty"
        assert Methods.NODE_NOTIFICATION == "node/notification"

    def test_host_api_methods(self) -> None:
        assert Methods.HOST_CREATE_NODE == "host/create_node"
        assert Methods.HOST_INVOKE == "host/invoke"
        assert Methods.HOST_QUERY_ROOTS == "host/query_roots"
        assert Methods.HOST_RESOLVE_ROOT == "host/resolve_root"


class TestConnectionStatus:
    """Plugin connection status enum."""

    def test_values(self) -> None:
        assert PluginConnectionStatus.DISCONNECTED.value == "disconnected"
        assert PluginConnectionStatus.CONNECTING.value == "connecting"
        assert PluginConnectionStatus.CONNECTED.value == "connected"
        assert PluginConnectionStatus.ERROR.value == "error"


class TestErrorCodes:
    """Error code constants."""

    def test_standard_codes(self) -> None:
        assert ErrorCodes.PARSE_ERROR == -32700
        assert ErrorCodes.INVALID_REQUEST == -32600
        assert ErrorCodes.METHOD_NOT_FOUND == -32601
        assert ErrorCodes.INVALID_PARAMS == -32602
        assert ErrorCodes.INTERNAL_ERROR == -32603

    def test_plugin_codes(self) -> None:
        assert ErrorCodes.NODE_NOT_FOUND == -32000
        assert ErrorCodes.NODE_TYPE_UNKNOWN == -32001
        assert ErrorCodes.METHOD_NOT_AVAILABLE == -32002
        assert ErrorCodes.ROOT_NOT_FOUND == -32003
        assert ErrorCodes.PERMISSION_DENIED == -32004
        assert ErrorCodes.NODE_CREATE_FAILED == -32005
