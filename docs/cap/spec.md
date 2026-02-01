# Context Application Protocol (CAP) v1.0 — Formal Specification

> The canonical specification for the Context Application Protocol.

**Status:** Stable (1.0)
**Date:** 2025-01-30
**Authors:** ActiveContext Contributors

---

## Abstract

The **Context Application Protocol (CAP)** is a transport-independent, encoding-agnostic protocol enabling plugin servers to provide custom context node types to a host application's context graph. CAP uses a sync-on-tick model for efficient state synchronization, supports bidirectional messaging, and provides schema-driven discovery for dynamic code generation and validation.

This document defines:
1. Protocol semantics — message meanings, lifecycle, state model
2. Core message types and their interactions
3. Schema discovery mechanisms
4. Error handling and recovery

Companion documents define serialization bindings (JSON, Protobuf, MessagePack) and transport bindings (stdio, gRPC, WebSocket, TCP).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Terminology](#2-terminology)
3. [Protocol Model](#3-protocol-model)
4. [Connection Lifecycle](#4-connection-lifecycle)
5. [Message Categories](#5-message-categories)
6. [Sync-on-Tick Model](#6-sync-on-tick-model)
7. [Schema Discovery](#7-schema-discovery)
8. [Capabilities](#8-capabilities)
9. [Error Handling](#9-error-handling)
10. [Security Model](#10-security-model)
11. [Conformance](#11-conformance)

---

## 1. Introduction

### 1.1 Purpose

CAP enables language-independent plugin servers to extend a host application's context graph with custom node types. Plugins communicate via JSON-RPC 2.0 messages over a pluggable transport layer, with state synchronization batched to tick boundaries for efficiency.

### 1.2 Design Goals

1. **Language independence** — Plugin servers MAY be implemented in any language that supports the message format.
2. **Transport independence** — Abstract message semantics separate from concrete encoding and framing.
3. **Sync-on-tick** — Batched state synchronization minimizes round trips in the hot path.
4. **Schema-driven** — Machine-readable schemas enable code generation, validation, and LLM documentation.
5. **Minimal surface** — 13 message types organized into four categories.

### 1.3 Architecture

CAP follows a client-server model with bidirectional messaging:

- **Host**: Manages the context graph, drives the tick cycle, and owns the DAG of context nodes.
- **Plugin Server**: Provides one or more custom node types, manages node instances, and MAY push state change notifications.

Communication is bidirectional: the host sends lifecycle and node management messages; the server responds and MAY send push notifications or reverse requests (Host API).

### 1.4 Key Concepts

| Concept | Description |
|---------|-------------|
| **Node** | An instance of a node type in the context graph. Produces rendered content and has lifecycle methods. |
| **Node Type** | A class of nodes with a shared schema (e.g., `"shell"`, `"topic"`). |
| **Tick** | A discrete synchronization point. All state materialization occurs at tick boundaries. |
| **Dirty Flag** | Indicates a node's state changed and requires synchronization on the next tick. |
| **Render** | Text representation at three levels: header (collapsed), content (summary), detail (full). |
| **Root** | A filesystem path exposed to the plugin server, similar to a workspace root. |
| **Capability** | A feature flag enabling optional protocol features. |

---

## 2. Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

| Term | Definition |
|------|------------|
| **Host** | The application managing the context graph and driving the tick cycle. |
| **Plugin Server** | A separate process providing one or more node types via CAP. |
| **Node Instance** | A specific node in the context graph, identified by a unique node ID. |
| **Session** | A host-side execution context with its own graph, timeline, and message history. |
| **DSL** | Domain-specific language (Python-based) for agent manipulation of context nodes. |
| **Projection** | Token-budgeted composition of rendered nodes for LLM consumption. |
| **Method Call** | Invocation of a method on a node instance (e.g., `cancel`, `set_content`). |

---

## 3. Protocol Model

### 3.1 Request/Response Pattern

CAP uses JSON-RPC 2.0 message envelopes:

- **Request**: Has `method` and `id` fields; expects a response with matching `id`.
- **Notification**: Has `method` but no `id`; expects no response.
- **Response**: Has `id` and either `result` or `error`; matches a request.

### 3.2 Message Flow

```
Host                                    Plugin Server
  |                                             |
  |--- initialize (request) ------------------->|
  |<-- initialize (result) ---------------------|
  |                                             |
  |--- node/create (request) ------------------>|
  |<-- node/create (result) --------------------|
  |                                             |
  |<-- node/dirty (notification) --------------|
  |                                             |
  |--- node/sync (request) --------------------->|
  |<-- node/sync (result) -----------------------|
  |                                             |
  |--- shutdown (notification) ----------------->|
  |                                             |
```

### 3.3 Bidirectional Messaging

The server MAY send messages to the host:

- **Push notifications**: `node/dirty`, `node/notification`
- **Host API requests**: `host/create_node`, `host/invoke`, `host/query_roots`, `host/resolve_root`

---

## 4. Connection Lifecycle

### 4.1 Connection Models

CAP supports two connection models:

1. **Host-managed process**: Host spawns the server as a child process and owns its lifecycle.
2. **External process**: Host connects to a pre-existing server (shared service, remote server).

### 4.2 Connection States

```
DISCONNECTED  ── connect() ──►  CONNECTING
CONNECTING    ── handshake ──►  CONNECTED
CONNECTING    ── failure ────►  ERROR
CONNECTED     ── disconnect ──► DISCONNECTED
ERROR         ── disconnect ──► DISCONNECTED
```

### 4.3 Initialize Handshake

After establishing the transport connection, the host MUST send `initialize` as the first message. The server MUST respond with its capabilities and node type schemas.

**Handshake sequence:**

1. Host sends `initialize` request with protocol version, capabilities, and filesystem roots.
2. Server validates the protocol version and responds with server name, version, capabilities, and node types.
3. Host registers all advertised node types.
4. Connection transitions to `CONNECTED`.

### 4.4 Shutdown Sequence

**Graceful shutdown:**

1. Host sends `shutdown` notification.
2. Host waits for in-flight responses (timeout: 5 seconds recommended).
3. Host closes transport resources.
4. For host-managed processes: host waits for process exit, terminates forcefully if needed.
5. For external processes: host closes connection but does NOT terminate the server.

**Unclean shutdown:**

If the transport connection is lost (EOF, broken pipe, reset):
- Host fails all pending requests, closes resources, transitions to `DISCONNECTED`.
- For host-managed processes: host terminates the orphaned process.

---

## 5. Message Categories

CAP defines 13 message types in four categories:

| Category | Direction | Messages |
|----------|-----------|----------|
| **Lifecycle** | Host → Server | `initialize`, `shutdown` |
| **Node Management** | Host → Server | `node/create`, `node/sync`, `node/call`, `node/destroy`, `node/serialize` |
| **Push Notifications** | Server → Host | `node/dirty`, `node/notification` |
| **Host API** | Server → Host | `host/create_node`, `host/invoke`, `host/query_roots`, `host/resolve_root` |

### 5.1 Lifecycle Messages

#### `initialize` (request)

Establishes the connection and negotiates capabilities. MUST be the first message.

**Params (InitializeParams):**
- `protocol_version` (string, default `"1.0"`)
- `host_capabilities` (HostCapabilities)
- `roots` (RootInfo[])
- `session_id` (string)
- `cwd` (string)

**Result (InitializeResult):**
- `server_name` (string, REQUIRED)
- `server_version` (string)
- `protocol_version` (string)
- `node_types` (NodeTypeSchema[])
- `server_capabilities` (ServerCapabilities)

#### `shutdown` (notification)

Notifies the server to begin cleanup. No response expected.

### 5.2 Node Management Messages

#### `node/create` (request)

Creates a new node instance.

**Params:** `node_type`, `args`, `kwargs`, `node_id` (optional)
**Result:** `node_id`, `state`, `renders`, `tokens`, `digest`

#### `node/sync` (request)

**The hot path.** Synchronizes a dirty node by executing batched method calls and retrieving complete state.

**Params:** `node_id`, `calls` (PendingCall[]), `cwd`
**Result:** `state`, `renders`, `tokens`, `digest`, `notifications`

#### `node/call` (request)

Executes a method immediately, outside the tick cycle. Requires `immediate_call` capability.

**Params:** `node_id`, `method`, `args`, `kwargs`
**Result:** `result`, `state_changed`

#### `node/destroy` (request)

Destroys a node instance and releases resources.

**Params:** `node_id`
**Result:** Empty object `{}`

#### `node/serialize` (request)

Retrieves full serialized state for checkpoint/persistence.

**Params:** `node_id`
**Result:** `data` (includes `node_type` for deserialization dispatch)

### 5.3 Push Notifications

#### `node/dirty` (notification)

One-bit signal: node state changed, sync on next tick. Requires `push_dirty` capability.

**Params:** `node_id`

#### `node/notification` (notification)

General notification from a node. Requires `push_notifications` capability.

**Params:** `node_id`, `description`, `level` (`"ignore"` | `"hold"` | `"wake"`)

### 5.4 Host API (Server → Host Requests)

#### `host/create_node` (request)

Server requests child node creation in the host graph. Requires `create_node` capability.

**Params:** `node_type`, `args`, `kwargs`, `parent_id`
**Result:** `node_id`

#### `host/invoke` (request)

Server invokes a method on a host node. Requires `invoke` capability.

**Params:** `node_id`, `method`, `args`, `kwargs`
**Result:** `result`

#### `host/query_roots` (request)

Server queries available filesystem roots. Requires `roots` capability.

**Params:** None
**Result:** `roots` (RootInfo[])

#### `host/resolve_root` (request)

Server resolves a normalized path through a root. Requires `roots` capability.

**Params:** `root_uri`, `normalized_path`
**Result:** `real_path`, `exists`

---

## 6. Sync-on-Tick Model

### 6.1 Tick Cycle

The host drives synchronization at discrete tick boundaries. A tick occurs once per agent turn or on timer/event triggers.

**Tick sequence:**

1. Host collects all dirty nodes (from `node/dirty` notifications or DSL method calls).
2. For each dirty node, host sends one `node/sync` request with all pending calls.
3. Server executes calls in order, returns complete state.
4. Host updates proxy node with state, renders, tokens, digest, notifications.
5. Host delivers notifications to the graph's parent chain.

### 6.2 Dirty Flag Semantics

A node becomes dirty when:
- Server pushes `node/dirty` (asynchronous state change)
- DSL invokes a method (queued as PendingCall)
- Host explicitly marks for sync (e.g., after checkpoint restore)

The dirty flag is cleared after `node/sync` completes.

### 6.3 Method Call Batching

Between ticks, multiple DSL calls on the same node accumulate as PendingCall objects and are sent in a single `node/sync` request.

**Example DSL:**
```python
v.SetState(NodeState.ALL).SetTokens(500)
```

Translates to:
```json
{
  "calls": [
    {"method": "SetState", "args": ["ALL"], "kwargs": {}},
    {"method": "SetTokens", "args": [500], "kwargs": {}}
  ]
}
```

**Processing invariant:** The server MUST execute calls in array order.

### 6.4 Performance Guarantee

**Design invariant:** At most one `node/sync` request per node per tick, regardless of how many method calls were queued.

This bounds round trips to the number of dirty nodes, not the number of method calls.

### 6.5 Immediate Calls

For operations that cannot wait (e.g., DSL expressions needing return values), the host MAY use `node/call`. This bypasses batching and executes immediately.

Requires `immediate_call` capability. If `state_changed: true`, the host marks the node dirty for the next tick.

---

## 7. Schema Discovery

Node types are described by machine-readable schemas advertised during `initialize`.

### 7.1 NodeTypeSchema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_type` | string | Yes | Type identifier (e.g., `"shell"`). Must be unique per server. |
| `description` | string | No | Human-readable description for LLM prompts. |
| `constructor` | ConstructorSchema | No | Constructor parameter schema. |
| `properties` | PropertySchema[] | No | DSL-visible properties. |
| `methods` | MethodSchema[] | No | DSL-callable methods. |

### 7.2 ConstructorSchema

Models Python calling convention:
```
node_type(pos1, pos2, *variadic, named1=default1, named2=default2)
```

| Field | Type | Description |
|-------|------|-------------|
| `positional` | ParamSchema[] | Positional parameters, in order. |
| `variadic` | ParamSchema or null | Variadic parameter (`*args`), if any. |
| `named` | ParamSchema[] | Keyword-only parameters, with defaults. |

### 7.3 ParamSchema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Parameter name. |
| `type` | string | No (default `"str"`) | Type hint as string (e.g., `"int"`, `"list[str]"`). |
| `default` | Any | No (MISSING) | Default value. Absent = required parameter. |
| `description` | string | No | Human-readable description. |

**MISSING sentinel:** Absence of `default` field = required parameter. `default: null` = optional parameter with `None` default.

### 7.4 PropertySchema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Property name (e.g., `"is_complete"`). |
| `type` | string | `"Any"` | Type hint as string. |
| `readable` | boolean | `true` | DSL can read this property. |
| `writable` | boolean | `false` | DSL can assign to this property. |
| `description` | string | `""` | Human-readable description. |

### 7.5 MethodSchema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Method name (e.g., `"cancel"`). |
| `params` | ParamSchema[] | `[]` | Parameters (excluding `self`). |
| `returns` | string | `"None"` | Return type hint as string. |
| `description` | string | `""` | Human-readable description. |
| `chainable` | boolean | `false` | If true, returns `self` for fluent chaining. |

---

## 8. Capabilities

Capabilities are feature flags exchanged during `initialize`. They enable graceful degradation when a peer does not support an optional feature.

### 8.1 HostCapabilities

Declared by the host; tells the server which Host API features are available.

| Capability | Default | Description |
|------------|---------|-------------|
| `create_node` | `true` | Host supports `host/create_node`. |
| `invoke` | `true` | Host supports `host/invoke`. |
| `roots` | `true` | Host supports `host/query_roots` and `host/resolve_root`. |
| `notifications` | `true` | Host accepts `node/dirty` and `node/notification`. |

If a capability is `false`, the server MUST NOT send corresponding messages.

### 8.2 ServerCapabilities

Declared by the server; tells the host which synchronization features are supported.

| Capability | Default | Description |
|------------|---------|-------------|
| `sync` | `true` | Server supports `node/sync` (tick-based synchronization). |
| `immediate_call` | `false` | Server supports `node/call` (immediate method execution). |
| `push_dirty` | `true` | Server pushes `node/dirty` notifications. |
| `push_notifications` | `true` | Server pushes `node/notification` messages. |

If `immediate_call` is `false`, the host MUST NOT send `node/call`.

### 8.3 Negotiation

1. Host sends `initialize` with `host_capabilities`.
2. Server reads capabilities to determine which Host API methods it can use.
3. Server responds with `server_capabilities`.
4. Host reads capabilities to determine which node management methods it can use.

Capabilities are fixed for the connection lifetime.

---

## 9. Error Handling

### 9.1 Error Response Format

CAP uses JSON-RPC 2.0 error responses:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32000,
    "message": "Node not found: shell_a1b2c3d4",
    "data": {"node_id": "shell_a1b2c3d4"}
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | integer | Yes | Numeric error code. |
| `message` | string | Yes | Human-readable error description. |
| `data` | Any | No | Additional structured error data. |

### 9.2 Error Codes

#### Standard JSON-RPC 2.0

| Code | Name | Description |
|------|------|-------------|
| -32700 | `PARSE_ERROR` | Invalid JSON. |
| -32600 | `INVALID_REQUEST` | Invalid JSON-RPC request. |
| -32601 | `METHOD_NOT_FOUND` | Method does not exist. |
| -32602 | `INVALID_PARAMS` | Invalid method parameters. |
| -32603 | `INTERNAL_ERROR` | Internal server error. |

#### CAP-Specific (-32000 to -32099)

| Code | Name | Description |
|------|------|-------------|
| -32000 | `NODE_NOT_FOUND` | No node with the given ID. |
| -32001 | `NODE_TYPE_UNKNOWN` | Node type not registered. |
| -32002 | `METHOD_NOT_AVAILABLE` | Method not defined or requires undeclared capability. |
| -32003 | `ROOT_NOT_FOUND` | Filesystem root URI not found. |
| -32004 | `PERMISSION_DENIED` | Operation not permitted (capability not declared). |
| -32005 | `NODE_CREATE_FAILED` | Node creation failed. |

### 9.3 Recovery Semantics

| Error Type | Recovery |
|------------|----------|
| **Transient** (INTERNAL_ERROR, NODE_CREATE_FAILED) | Host MAY retry. No side effects. |
| **Permanent** (NODE_NOT_FOUND, NODE_TYPE_UNKNOWN) | Host SHOULD NOT retry with same params. Log and report to agent. |
| **Protocol** (PARSE_ERROR, INVALID_REQUEST) | Indicates sender bug. Log for debugging. Connection MAY continue. |
| **Permission** (PERMISSION_DENIED, ROOT_NOT_FOUND) | Sender SHOULD check capabilities before retrying. |
| **Connection loss** | All pending requests fail. Host transitions to ERROR. MAY reconnect. |

---

## 10. Security Model

### 10.1 Roots-Based Filesystem Access

The host exposes filesystem access through named roots (RootInfo). Plugin servers:
- MUST use `host/query_roots` and `host/resolve_root` to discover and resolve paths.
- MUST NOT access paths outside declared roots unless explicitly permitted.
- SHOULD use normalized paths relative to roots.

### 10.2 Permission Boundaries

| Boundary | Enforcement |
|----------|-------------|
| **File access** | Plugin operations go through host roots and permission checks. |
| **Node creation** | `host/create_node` capability and `PERMISSION_DENIED` error. |
| **Method invocation** | `host/invoke` capability. Host MAY apply additional method-level permissions. |
| **Capability gating** | All Host API operations gated on corresponding capability flags. |

### 10.3 Sandboxing Considerations

Hosts SHOULD:
- **Process isolation**: Run plugin servers with minimal OS privileges.
- **Resource limits**: Apply CPU, memory, I/O limits to plugin processes.
- **Network access**: Plugins SHOULD NOT have network access unless explicitly required.
- **Secrets**: Host MUST NOT pass secrets via `initialize` params or environment unless plugin is trusted.
- **Input validation**: Both sides MUST validate all incoming message parameters.

### 10.4 Trust Model

1. **Host** is trusted — owns the graph, controls process spawning.
2. **Plugin servers** are semi-trusted — separate processes with capability-gated access.
3. **Agent (LLM)** is untrusted — DSL calls go through host permission system.

---

## 11. Conformance

### 11.1 Required Features

A conforming CAP implementation MUST:
- Support the JSON serialization binding (see `docs/cap/serialization-bindings.md`)
- Support the stdio transport binding (see `docs/cap/transport-bindings.md`)
- Implement all lifecycle messages (`initialize`, `shutdown`)
- Implement all node management messages (`node/create`, `node/sync`, `node/destroy`, `node/serialize`)
- Support the sync-on-tick model (Section 6)
- Correctly handle all standard error codes (Section 9.2)
- Implement schema discovery (Section 7)
- Implement capability negotiation (Section 8)

### 11.2 Optional Features

A conforming implementation MAY:
- Support additional serialization bindings (Protobuf, MessagePack)
- Support additional transport bindings (gRPC, WebSocket, TCP)
- Implement `node/call` (requires `immediate_call` capability)
- Implement push notifications (`node/dirty`, `node/notification`)
- Implement Host API messages (`host/create_node`, `host/invoke`, etc.)

### 11.3 Conformance Levels

| Level | Requirements |
|-------|--------------|
| **Minimal** | JSON/stdio, lifecycle + node management (create/sync/destroy), no push, no Host API |
| **Standard** | Minimal + push notifications (`node/dirty`, `node/notification`) |
| **Full** | Standard + Host API + `node/call` + at least one additional serialization or transport |

### 11.4 Interoperability

Implementations at different conformance levels MUST interoperate at the intersection of their capabilities. Capability negotiation (Section 8) enables graceful degradation.

---

## References

- **CAP Message Catalog**: `docs/cap/messages.md`
- **CAP Serialization Bindings**: `docs/cap/serialization-bindings.md`
- **CAP Transport Bindings**: `docs/cap/transport-bindings.md`
- **CAP Conformance Tests**: `docs/cap/conformance.md`
- **Protobuf Schema**: `docs/cap.proto`
- **Reference Implementation**: `src/activecontext/plugins/`

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-30 | Initial stable release |
