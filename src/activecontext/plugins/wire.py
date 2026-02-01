"""Context Application Protocol (CAP) — JSON-RPC 2.0 over stdio.

This module defines the complete message catalog for the Context Application
Protocol, used for communication between the host (ActiveContext session)
and remote plugin servers. CAP uses JSON-RPC 2.0 with bidirectional
messaging over stdio pipes.

Transport: JSON-RPC 2.0, newline-delimited, over stdin/stdout.
Encoding: UTF-8.
Framing: Each message is a single JSON object followed by newline.

Connection lifecycle:
    1. Host spawns plugin server subprocess (or connects to shared server)
    2. Server sends `initialize` result with capabilities and node type schemas
    3. Host validates schemas, registers node types
    4. Normal operation: create/sync/call/destroy nodes
    5. Host sends `shutdown` notification, closes pipes

Sync-on-tick model:
    - DSL method calls are queued as MethodCall objects
    - On tick, host sends `node/sync` with all pending calls
    - Server processes calls, returns updated state + renders + notifications
    - One round trip per tick per dirty node

Push notifications (server → host):
    - `node/dirty`: one-bit signal, triggers sync on next tick
    - `node/notification`: immediate notification to parent chain

Host API (host → server, reverse direction):
    - `host/create_node`: server requests child node creation in host graph
    - `host/invoke`: server calls method on existing host node
    - `host/query_roots`: server queries filesystem roots
    - `host/resolve_root`: server maps normalized path to real path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

PROTOCOL_NAME = "CAP"
"""Context Application Protocol."""

PROTOCOL_VERSION = "1.0"
"""CAP version. Sent in initialize handshake."""


class Methods:
    """JSON-RPC method names organized by direction and category."""

    # --- Lifecycle (host → server) ---
    INITIALIZE = "initialize"
    SHUTDOWN = "shutdown"

    # --- Node management (host → server) ---
    NODE_CREATE = "node/create"
    NODE_SYNC = "node/sync"
    NODE_CALL = "node/call"
    NODE_DESTROY = "node/destroy"
    NODE_SERIALIZE = "node/serialize"

    # --- Push notifications (server → host, no response) ---
    NODE_DIRTY = "node/dirty"
    NODE_NOTIFICATION = "node/notification"

    # --- Host API (server → host) ---
    HOST_CREATE_NODE = "host/create_node"
    HOST_INVOKE = "host/invoke"
    HOST_QUERY_ROOTS = "host/query_roots"
    HOST_RESOLVE_ROOT = "host/resolve_root"


# ---------------------------------------------------------------------------
# Connection status
# ---------------------------------------------------------------------------


class PluginConnectionStatus(Enum):
    """Status of a plugin server connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Initialize handshake
# ---------------------------------------------------------------------------


@dataclass
class InitializeParams:
    """Parameters for the `initialize` request (host → server).

    Sent as the first message after connection. The server responds
    with its capabilities and node type schemas.
    """

    protocol_version: str = PROTOCOL_VERSION
    """Protocol version the host supports."""

    host_capabilities: HostCapabilities = field(default_factory=lambda: HostCapabilities())
    """Capabilities the host offers to the server."""

    roots: list[RootInfo] = field(default_factory=list)
    """Filesystem roots available to the server."""

    session_id: str = ""
    """Session identifier for logging and coordination."""

    cwd: str = ""
    """Working directory for the session."""


@dataclass
class HostCapabilities:
    """Capabilities the host offers to plugin servers."""

    create_node: bool = True
    """Host supports host/create_node for child creation."""

    invoke: bool = True
    """Host supports host/invoke for method calls on host nodes."""

    roots: bool = True
    """Host supports host/query_roots and host/resolve_root."""

    notifications: bool = True
    """Host accepts node/dirty and node/notification push."""


@dataclass
class InitializeResult:
    """Result of the `initialize` request (server → host).

    Contains the server's identity, capabilities, and the node types
    it provides with their full schemas.
    """

    server_name: str
    """Human-readable server name."""

    server_version: str = ""
    """Server version string."""

    protocol_version: str = PROTOCOL_VERSION
    """Protocol version the server supports."""

    node_types: list[NodeTypeSchema] = field(default_factory=list)
    """Node types this server provides, with full schemas."""

    server_capabilities: ServerCapabilities = field(default_factory=lambda: ServerCapabilities())
    """Capabilities the server supports."""


@dataclass
class ServerCapabilities:
    """Capabilities the server declares."""

    sync: bool = True
    """Server supports node/sync (tick-based synchronization)."""

    immediate_call: bool = False
    """Server supports node/call (immediate method execution outside tick)."""

    push_dirty: bool = True
    """Server pushes node/dirty notifications."""

    push_notifications: bool = True
    """Server pushes node/notification messages."""


# ---------------------------------------------------------------------------
# Node type schema (shared with MCP discovery — np-schema task)
# ---------------------------------------------------------------------------


@dataclass
class NodeTypeSchema:
    """Schema for a plugin node type.

    Advertised by the server during initialization. Used by the host for:
    - Generating DSL constructor functions
    - Type validation at call sites
    - LLM prompt documentation
    - Autocompletion

    The constructor schema uses Python calling convention:
    positional args, then keyword args with defaults.
    """

    node_type: str
    """Type identifier (e.g., "shell", "topic")."""

    description: str = ""
    """Human-readable description for LLM prompts."""

    constructor: ConstructorSchema = field(default_factory=lambda: ConstructorSchema())
    """Constructor parameter schema."""

    properties: list[PropertySchema] = field(default_factory=list)
    """Readable/writable properties exposed to the DSL."""

    methods: list[MethodSchema] = field(default_factory=list)
    """Callable methods exposed to the DSL."""


@dataclass
class ConstructorSchema:
    """Schema for the DSL constructor function.

    Maps to Python calling convention:
        node_type(pos1, pos2, *variadic, named1=default1, named2=default2)
    """

    positional: list[ParamSchema] = field(default_factory=list)
    """Positional parameters (in order)."""

    variadic: ParamSchema | None = None
    """Variadic parameter (*args), if any."""

    named: list[ParamSchema] = field(default_factory=list)
    """Keyword-only parameters (with defaults)."""


# Sentinel for "no default value" (distinct from None which is a valid default)
class _MissingSentinel:
    """Sentinel indicating no default value for a parameter."""

    _instance: _MissingSentinel | None = None

    def __new__(cls) -> _MissingSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


_MISSING = _MissingSentinel()
MISSING = _MISSING
"""Sentinel value indicating a required parameter with no default."""


@dataclass
class ParamSchema:
    """Schema for a single parameter."""

    name: str
    """Parameter name."""

    type: str = "str"
    """Type hint as string (e.g., "str", "int", "float", "bool", "list[str]")."""

    default: Any = _MISSING
    """Default value, or MISSING if required."""

    description: str = ""
    """Human-readable description."""


@dataclass
class PropertySchema:
    """Schema for a DSL-visible property."""

    name: str
    """Property name (e.g., "is_complete", "output")."""

    type: str = "Any"
    """Type hint as string."""

    readable: bool = True
    """Whether the DSL can read this property."""

    writable: bool = False
    """Whether the DSL can assign to this property."""

    description: str = ""
    """Human-readable description."""


@dataclass
class MethodSchema:
    """Schema for a DSL-callable method."""

    name: str
    """Method name (e.g., "cancel", "set_content")."""

    params: list[ParamSchema] = field(default_factory=list)
    """Method parameters (excluding self)."""

    returns: str = "None"
    """Return type hint as string."""

    description: str = ""
    """Human-readable description."""

    chainable: bool = False
    """If True, method returns self for chaining."""


# ---------------------------------------------------------------------------
# Filesystem roots
# ---------------------------------------------------------------------------


@dataclass
class RootInfo:
    """A filesystem root exposed to plugin servers.

    Akin to MCP roots. Plugins use roots to understand the filesystem
    layout and resolve normalized paths to real files.
    """

    uri: str
    """Root URI (e.g., "file:///project", "file:///home/user")."""

    name: str
    """Human-readable name (e.g., "cwd", "home")."""

    description: str = ""
    """What this root represents."""


# ---------------------------------------------------------------------------
# Node management messages (host → server)
# ---------------------------------------------------------------------------


@dataclass
class NodeCreateParams:
    """Parameters for `node/create` (host → server)."""

    node_type: str
    """Type of node to create."""

    args: list[Any] = field(default_factory=list)
    """Positional constructor arguments."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword constructor arguments."""

    node_id: str = ""
    """Host-assigned node ID (server must use this ID).
    If empty, server generates its own ID."""


@dataclass
class NodeCreateResult:
    """Result of `node/create` (server → host)."""

    node_id: str
    """Node instance ID."""

    state: dict[str, Any] = field(default_factory=dict)
    """Initial serialized state."""

    renders: dict[str, str] = field(default_factory=dict)
    """Initial renders: {"header": "...", "content": "...", "detail": "..."}."""

    tokens: dict[str, int] = field(default_factory=dict)
    """Initial token estimates: {"collapsed": N, "summary": N, "detail": N}."""

    digest: dict[str, Any] = field(default_factory=dict)
    """Initial digest metadata."""


@dataclass
class PendingCall:
    """A method call queued for batch execution during sync."""

    method: str
    """Method name (e.g., "set_content", "cancel")."""

    args: list[Any] = field(default_factory=list)
    """Positional arguments."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments."""


@dataclass
class NodeSyncParams:
    """Parameters for `node/sync` (host → server).

    The hot path: called once per tick per dirty node. Batches
    all pending method calls and returns the complete state.
    """

    node_id: str
    """Node to synchronize."""

    calls: list[PendingCall] = field(default_factory=list)
    """Queued method calls to execute before returning state."""

    cwd: str = ""
    """Working directory for render methods."""


@dataclass
class NodeSyncResult:
    """Result of `node/sync` (server → host).

    Contains everything the host needs to update its proxy:
    state, renders, tokens, digest, and any notifications.
    """

    state: dict[str, Any] = field(default_factory=dict)
    """Full serialized node state (all public fields)."""

    renders: dict[str, str] = field(default_factory=dict)
    """Rendered content: {"header": "...", "content": "...", "detail": "..."}."""

    tokens: dict[str, int] = field(default_factory=dict)
    """Token estimates: {"collapsed": N, "summary": N, "detail": N}."""

    digest: dict[str, Any] = field(default_factory=dict)
    """Compact metadata (same as get_digest() output)."""

    notifications: list[dict[str, Any]] = field(default_factory=list)
    """Notifications generated since last sync.
    Each: {"description": "...", "level": "ignore"|"hold"|"wake"}."""


@dataclass
class NodeCallParams:
    """Parameters for `node/call` (host → server).

    Immediate method execution outside tick. Used for urgent operations
    where the DSL needs a return value immediately.
    """

    node_id: str
    """Node to call method on."""

    method: str
    """Method name."""

    args: list[Any] = field(default_factory=list)
    """Positional arguments."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments."""


@dataclass
class NodeCallResult:
    """Result of `node/call` (server → host)."""

    result: Any = None
    """Return value of the method call."""

    state_changed: bool = False
    """Whether the call changed the node's state (triggers sync)."""


@dataclass
class NodeDestroyParams:
    """Parameters for `node/destroy` (host → server)."""

    node_id: str
    """Node to destroy."""


@dataclass
class NodeSerializeParams:
    """Parameters for `node/serialize` (host → server).

    Requests the full serialized state for checkpoint/persistence.
    """

    node_id: str
    """Node to serialize."""


@dataclass
class NodeSerializeResult:
    """Result of `node/serialize` (server → host)."""

    data: dict[str, Any] = field(default_factory=dict)
    """Full serialized state, suitable for from_dict() reconstruction."""


# ---------------------------------------------------------------------------
# Push notifications (server → host, no response expected)
# ---------------------------------------------------------------------------


@dataclass
class NodeDirtyParams:
    """Parameters for `node/dirty` notification (server → host).

    One-bit signal: the node's state has changed and needs sync.
    No payload beyond the node ID — the host will sync on next tick.
    """

    node_id: str
    """Node whose state changed."""


@dataclass
class NodeNotificationParams:
    """Parameters for `node/notification` (server → host).

    A notification to deliver to the parent chain. The host calls
    notify_parents() on the proxy node with the given description.
    """

    node_id: str
    """Source node."""

    description: str
    """Human-readable description of what changed."""

    level: str = "hold"
    """Notification level: "ignore", "hold", or "wake"."""


# ---------------------------------------------------------------------------
# Host API messages (server → host)
# ---------------------------------------------------------------------------


@dataclass
class HostCreateNodeParams:
    """Parameters for `host/create_node` (server → host).

    The plugin server requests creation of a child node in the host
    graph. Used for nested node creation (e.g., a shell plugin creating
    artifact children for output).
    """

    node_type: str
    """Type of node to create (must be registered in host)."""

    args: list[Any] = field(default_factory=list)
    """Positional constructor arguments."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword constructor arguments."""

    parent_id: str = ""
    """Parent node ID to link to (optional)."""


@dataclass
class HostCreateNodeResult:
    """Result of `host/create_node` (host → server)."""

    node_id: str
    """ID of the created node in the host graph."""


@dataclass
class HostInvokeParams:
    """Parameters for `host/invoke` (server → host).

    The plugin server calls a method on an existing node in the host
    graph. Used for cross-node interactions.
    """

    node_id: str
    """Target node in the host graph."""

    method: str
    """Method name to call."""

    args: list[Any] = field(default_factory=list)
    """Positional arguments."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments."""


@dataclass
class HostInvokeResult:
    """Result of `host/invoke` (host → server)."""

    result: Any = None
    """Return value of the method call."""


@dataclass
class HostQueryRootsResult:
    """Result of `host/query_roots` (host → server).

    Returns all filesystem roots available to the plugin.
    """

    roots: list[RootInfo] = field(default_factory=list)
    """Available filesystem roots."""


@dataclass
class HostResolveRootParams:
    """Parameters for `host/resolve_root` (server → host).

    Maps a normalized filename through a root to a real filesystem path.
    """

    root_uri: str
    """Root URI to resolve against."""

    normalized_path: str
    """Normalized path relative to the root."""


@dataclass
class HostResolveRootResult:
    """Result of `host/resolve_root` (host → server)."""

    real_path: str
    """Resolved absolute filesystem path."""

    exists: bool = False
    """Whether the file exists at the resolved path."""


# ---------------------------------------------------------------------------
# JSON-RPC serialization helpers
# ---------------------------------------------------------------------------


def to_jsonrpc_request(method: str, params: Any, id: int | str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request message.

    Args:
        method: The RPC method name (from Methods class).
        params: Parameters dataclass or dict.
        id: Request ID for matching response.

    Returns:
        JSON-serializable dict.
    """
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id,
    }
    if params is not None:
        if hasattr(params, "__dataclass_fields__"):
            from dataclasses import asdict

            msg["params"] = asdict(params)
        else:
            msg["params"] = params
    return msg


def to_jsonrpc_notification(method: str, params: Any) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 notification (no id, no response expected).

    Args:
        method: The RPC method name.
        params: Parameters dataclass or dict.

    Returns:
        JSON-serializable dict.
    """
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        if hasattr(params, "__dataclass_fields__"):
            from dataclasses import asdict

            msg["params"] = asdict(params)
        else:
            msg["params"] = params
    return msg


def to_jsonrpc_response(result: Any, id: int | str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response.

    Args:
        result: Result dataclass or dict.
        id: Request ID this responds to.

    Returns:
        JSON-serializable dict.
    """
    msg: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": id,
    }
    if hasattr(result, "__dataclass_fields__"):
        from dataclasses import asdict

        msg["result"] = asdict(result)
    else:
        msg["result"] = result
    return msg


def to_jsonrpc_error(
    code: int,
    message: str,
    id: int | str | None,
    data: Any = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response.

    Args:
        code: Error code (see ErrorCodes).
        message: Human-readable error message.
        id: Request ID this responds to (None for parse errors).
        data: Optional additional error data.

    Returns:
        JSON-serializable dict.
    """
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "error": error,
        "id": id,
    }


class ErrorCodes:
    """Standard JSON-RPC 2.0 error codes plus plugin-specific codes."""

    # Standard JSON-RPC
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Plugin-specific (-32000 to -32099)
    NODE_NOT_FOUND = -32000
    NODE_TYPE_UNKNOWN = -32001
    METHOD_NOT_AVAILABLE = -32002
    ROOT_NOT_FOUND = -32003
    PERMISSION_DENIED = -32004
    NODE_CREATE_FAILED = -32005
