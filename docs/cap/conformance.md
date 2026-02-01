# CAP Conformance and Test Requirements

> Conformance levels, interoperability requirements, and test specifications for Context Application Protocol v1.0.

**Specification Version:** 1.0
**Date:** 2025-01-30

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Conformance Levels](#2-conformance-levels)
3. [Required Features](#3-required-features)
4. [Optional Features](#4-optional-features)
5. [Interoperability Requirements](#5-interoperability-requirements)
6. [Test Suite](#6-test-suite)
7. [Compliance Checklist](#7-compliance-checklist)

---

## 1. Overview

This document defines conformance requirements for CAP implementations. Conformance ensures interoperability between hosts and plugin servers implemented in different languages and environments.

### 1.1 Conformance Classes

CAP defines three conformance levels:

| Level | Target Audience | Requirements |
|-------|----------------|--------------|
| **Minimal** | Simple plugins with no asynchronous state changes | Required features only |
| **Standard** | Most plugins with async state and notifications | Minimal + push notifications |
| **Full** | Advanced plugins with immediate calls, multiple transports | Standard + all optional features |

Implementations at different levels MUST interoperate at the intersection of their capabilities via capability negotiation.

### 1.2 Implementation Types

| Type | Role | Examples |
|------|------|----------|
| **Host Implementation** | Manages context graph, drives tick cycle | ActiveContext Session |
| **Server Implementation** | Provides custom node types | Shell plugin, file watcher plugin |
| **Library Implementation** | Provides both host and server APIs | ActiveContext Direct Transport |

---

## 2. Conformance Levels

### 2.1 Minimal Conformance

**Target:** Simple plugin servers with synchronous-only node types.

**Required capabilities:**
- JSON serialization binding
- stdio transport binding
- Lifecycle messages: `initialize`, `shutdown`
- Node management: `node/create`, `node/sync`, `node/destroy`, `node/serialize`
- Schema discovery: NodeTypeSchema, ConstructorSchema, ParamSchema, PropertySchema, MethodSchema
- Error handling: All standard and CAP-specific error codes
- Capability negotiation: HostCapabilities, ServerCapabilities

**Server capabilities declared:**
```json
{
  "sync": true,
  "immediate_call": false,
  "push_dirty": false,
  "push_notifications": false
}
```

**Host capabilities required:**
```json
{
  "create_node": false,
  "invoke": false,
  "roots": false,
  "notifications": false
}
```

**Use case:** A plugin that provides a read-only view of external data (e.g., database query results, static file content). State updates only occur in response to DSL method calls, batched via `node/sync`.

### 2.2 Standard Conformance

**Target:** Most production plugin servers.

**Required capabilities:**
- All Minimal requirements
- Push notifications: `node/dirty`, `node/notification`
- Server capabilities: `push_dirty: true`, `push_notifications: true`
- Host capabilities: `notifications: true`

**Server capabilities declared:**
```json
{
  "sync": true,
  "immediate_call": false,
  "push_dirty": true,
  "push_notifications": true
}
```

**Host capabilities required:**
```json
{
  "create_node": false,
  "invoke": false,
  "roots": false,
  "notifications": true
}
```

**Use case:** A shell plugin that pushes `node/dirty` when a subprocess completes, or a file watcher plugin that notifies when files change.

### 2.3 Full Conformance

**Target:** Advanced deployments with high-throughput, network-accessible servers, or immediate-mode DSL interactions.

**Required capabilities:**
- All Standard requirements
- `node/call` (immediate method execution)
- Host API: `host/create_node`, `host/invoke`, `host/query_roots`, `host/resolve_root`
- At least one additional serialization binding (Protobuf or MessagePack)
- At least one additional transport binding (gRPC, WebSocket, or TCP)
- Full capability matrix

**Server capabilities declared:**
```json
{
  "sync": true,
  "immediate_call": true,
  "push_dirty": true,
  "push_notifications": true
}
```

**Host capabilities required:**
```json
{
  "create_node": true,
  "invoke": true,
  "roots": true,
  "notifications": true
}
```

**Use case:** A complex plugin that creates child nodes dynamically (e.g., shell plugin creating artifact nodes for stdout/stderr), invokes methods on host nodes, resolves filesystem paths through roots, and supports immediate DSL queries.

---

## 3. Required Features

All conforming CAP implementations MUST support these features.

### 3.1 Serialization

- **JSON binding** (REQUIRED): JSON-RPC 2.0 with snake_case field names, UTF-8 encoding, NDJSON framing over stream transports.
- **MISSING sentinel**: Distinguish between absent field (MISSING) and `null` value.
- **Type mapping**: Correctly serialize all standard types (string, int, float, bool, null, list, dict, dataclass, enum).

See `serialization-bindings.md` Section 2 for complete JSON binding requirements.

### 3.2 Transport

- **stdio binding** (REQUIRED): Newline-delimited JSON over stdin/stdout, with stderr reserved for diagnostics.
- **Lifecycle operations**: `start()`, `stop()`, `is_running` property.
- **Error propagation**: Distinguish transport errors from protocol errors.
- **Connection state machine**: DISCONNECTED → CONNECTING → CONNECTED → DISCONNECTED/ERROR.

See [`transports.md`](transports.md) Section 3 for complete stdio binding requirements.

### 3.3 Lifecycle Messages

- **`initialize`**: MUST be first message. Server MUST respond with `server_name`. Host MUST register all advertised node types.
- **`shutdown`**: Host MUST send before closing connection. Server MUST complete in-flight responses and exit gracefully.

### 3.4 Node Management Messages

- **`node/create`**: MUST support host-assigned or server-generated node IDs. MUST return initial state, renders, tokens, digest.
- **`node/sync`**: MUST execute `calls` array in order. MUST return complete state, renders (header/content/detail), tokens (collapsed/summary/detail), digest, notifications.
- **`node/destroy`**: MUST release all resources. MUST NOT reuse node ID within same connection.
- **`node/serialize`**: MUST include `node_type` field in returned `data` object.

### 3.5 Schema Discovery

- **NodeTypeSchema**: MUST advertise all plugin-provided node types during `initialize`.
- **ConstructorSchema**: MUST describe positional, variadic, and named parameters.
- **ParamSchema**: MUST use MISSING sentinel for required parameters.
- **PropertySchema**: MUST declare readable/writable properties.
- **MethodSchema**: MUST declare chainable methods correctly.

### 3.6 Error Handling

- **Standard JSON-RPC errors**: PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR.
- **CAP-specific errors**: NODE_NOT_FOUND, NODE_TYPE_UNKNOWN, METHOD_NOT_AVAILABLE, ROOT_NOT_FOUND, PERMISSION_DENIED, NODE_CREATE_FAILED.
- **Error format**: MUST include `code` and `message` fields. MAY include `data` field.

### 3.7 Capability Negotiation

- **Host capabilities**: MUST declare during `initialize`.
- **Server capabilities**: MUST declare in `initialize` response.
- **Enforcement**: MUST NOT send messages requiring undeclared capabilities. MUST respond with PERMISSION_DENIED if peer violates capabilities.

---

## 4. Optional Features

Conforming implementations MAY support these features. If supported, they MUST be implemented correctly and declared via capabilities.

### 4.1 Push Notifications (Standard+)

- **`node/dirty`**: Server pushes one-bit signal when node state changes asynchronously. Host marks node dirty for next tick.
- **`node/notification`**: Server pushes description with level (ignore/hold/wake). Host propagates to notification system.
- **Capability**: `push_dirty` and `push_notifications` server capabilities.

### 4.2 Immediate Calls (Full)

- **`node/call`**: Execute method outside tick cycle, return result immediately.
- **State change tracking**: If `state_changed: true`, host marks node dirty for next tick.
- **Capability**: `immediate_call` server capability.

### 4.3 Host API (Full)

- **`host/create_node`**: Server creates child nodes in host graph.
- **`host/invoke`**: Server invokes methods on host nodes.
- **`host/query_roots`**: Server queries available filesystem roots.
- **`host/resolve_root`**: Server resolves normalized paths through roots.
- **Capabilities**: `create_node`, `invoke`, `roots` host capabilities.

### 4.4 Additional Serialization Bindings (Full)

- **Protobuf**: Binary serialization with typed message definitions. MUST support content negotiation via `initialize` extension.
- **MessagePack**: Compact binary encoding with same logical structure as JSON. MUST use string keys, MUST support MISSING semantics.

See `serialization-bindings.md` Sections 3-4 for complete requirements.

### 4.5 Additional Transport Bindings (Full)

- **gRPC**: HTTP/2 with protobuf or JSON. MUST support bidirectional stream or typed RPCs with callback service.
- **WebSocket**: WS text frames with JSON. MUST support `cap.v1` subprotocol, MUST handle Ping/Pong keepalive.
- **TCP**: Length-prefixed framing with JSON. MUST use 4-byte big-endian length prefix, SHOULD support TLS wrapper.

See [`transports.md`](transports.md) Sections 4-6 for complete requirements.

---

## 5. Interoperability Requirements

### 5.1 Graceful Degradation

Implementations at different conformance levels MUST interoperate at the intersection of their capabilities.

**Example:** A Full-conformant host connecting to a Minimal-conformant server:
- Host declares all capabilities (`create_node: true`, `invoke: true`, etc.)
- Server declares minimal capabilities (`push_dirty: false`, `immediate_call: false`, etc.)
- Host adapts: does NOT send `node/call`, does NOT expect `node/dirty` notifications, polls via `node/sync` instead.

### 5.2 Capability Matrix Compliance

| Host Capability | Server Capability | Interoperability Requirement |
|----------------|------------------|------------------------------|
| `notifications: false` | `push_dirty: true` | Server MUST NOT send `node/dirty` or `node/notification`. Host MAY still poll via `node/sync`. |
| `notifications: true` | `push_dirty: false` | Host MUST poll dirty nodes via `node/sync`. Server never pushes. |
| `create_node: false` | — | Server MUST NOT send `host/create_node`. Host responds with PERMISSION_DENIED if received. |
| — | `immediate_call: false` | Host MUST NOT send `node/call`. MUST queue all method calls for `node/sync`. |

### 5.3 Version Compatibility

- **Protocol version** is `"1.0"` for all implementations conforming to this specification.
- If a future version (`"2.0"`) is defined, implementations SHOULD support backward compatibility by inspecting `protocol_version` in `initialize` and adapting to the negotiated version.
- If versions are incompatible, the host SHOULD disconnect after `initialize` and log an error.

### 5.4 Cross-Language Testing

Implementations in different languages MUST pass cross-language interoperability tests:
- Python host ↔ Rust server
- Python host ↔ Go server
- TypeScript host ↔ Python server

Test vectors provided in Section 6.

---

## 6. Test Suite

### 6.1 Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Lifecycle** | Initialize, shutdown, reconnect | Handshake and teardown correctness |
| **Node Management** | Create, sync, call, destroy, serialize | Core operations |
| **Sync Batching** | Multiple calls in single sync | Batching and ordering |
| **Push Notifications** | Dirty flag, notifications | Async state change propagation |
| **Host API** | Create node, invoke, roots | Reverse direction correctness |
| **Schema Discovery** | Constructor, properties, methods | Schema parsing and validation |
| **Error Handling** | All error codes, recovery | Error propagation and recovery |
| **Capabilities** | Negotiation, enforcement | Capability matrix correctness |

### 6.2 Required Test Vectors

All implementations MUST pass these test vectors. Test vectors are JSON-RPC messages with expected responses.

#### 6.2.1 Initialize Handshake

**Test:** Minimal initialize handshake.

**Request (host → server):**
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1,
  "params": {
    "protocol_version": "1.0",
    "host_capabilities": {
      "create_node": false,
      "invoke": false,
      "roots": false,
      "notifications": false
    },
    "roots": [],
    "session_id": "test_session",
    "cwd": "/tmp"
  }
}
```

**Expected response (server → host):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "server_name": "<any non-empty string>",
    "server_version": "<any string>",
    "protocol_version": "1.0",
    "server_capabilities": {
      "sync": true,
      "immediate_call": false,
      "push_dirty": false,
      "push_notifications": false
    },
    "node_types": []
  }
}
```

**Assertion:** Response MUST have `server_name` field. Other fields use documented defaults if omitted.

#### 6.2.2 Node Create

**Test:** Create a node with host-assigned ID.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/create",
  "id": 2,
  "params": {
    "node_type": "test_node",
    "args": ["arg1"],
    "kwargs": {"key": "value"},
    "node_id": "test_node_001"
  }
}
```

**Expected response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "node_id": "test_node_001",
    "state": {},
    "renders": {
      "header": "",
      "content": "",
      "detail": ""
    },
    "tokens": {
      "collapsed": 0,
      "summary": 0,
      "detail": 0
    },
    "digest": {}
  }
}
```

**Assertion:** Response MUST echo back `node_id` exactly. Renders MUST have all three keys.

#### 6.2.3 Node Sync with Batched Calls

**Test:** Execute multiple method calls in single sync.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/sync",
  "id": 3,
  "params": {
    "node_id": "test_node_001",
    "calls": [
      {"method": "set_value", "args": [42], "kwargs": {}},
      {"method": "increment", "args": [], "kwargs": {}}
    ],
    "cwd": "/tmp"
  }
}
```

**Expected response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "state": {"value": 43},
    "renders": {
      "header": "Test Node [value=43]",
      "content": "Value: 43",
      "detail": "Value: 43\nLast operation: increment"
    },
    "tokens": {
      "collapsed": 10,
      "summary": 15,
      "detail": 25
    },
    "digest": {"value": 43},
    "notifications": []
  }
}
```

**Assertion:** Calls MUST execute in order. State reflects result of both calls.

#### 6.2.4 Error Handling: NODE_NOT_FOUND

**Test:** Sync non-existent node.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/sync",
  "id": 4,
  "params": {
    "node_id": "nonexistent",
    "calls": [],
    "cwd": "/tmp"
  }
}
```

**Expected response:**
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "error": {
    "code": -32000,
    "message": "Node not found: nonexistent",
    "data": {"node_id": "nonexistent"}
  }
}
```

**Assertion:** Error code MUST be -32000 (NODE_NOT_FOUND).

#### 6.2.5 Schema Discovery

**Test:** Advertise node type with constructor schema.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1,
  "params": {
    "protocol_version": "1.0",
    "host_capabilities": {
      "create_node": false,
      "invoke": false,
      "roots": false,
      "notifications": false
    },
    "roots": [],
    "session_id": "schema_test",
    "cwd": "/tmp"
  }
}
```

**Expected response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "server_name": "test_server",
    "protocol_version": "1.0",
    "node_types": [
      {
        "node_type": "test_node",
        "description": "A test node type",
        "constructor": {
          "positional": [
            {"name": "value", "type": "int", "description": "Initial value"}
          ],
          "variadic": null,
          "named": [
            {
              "name": "label",
              "type": "str",
              "default": "Test",
              "description": "Node label"
            }
          ]
        },
        "properties": [
          {
            "name": "value",
            "type": "int",
            "readable": true,
            "writable": false,
            "description": "Current value"
          }
        ],
        "methods": [
          {
            "name": "increment",
            "params": [],
            "returns": "None",
            "description": "Increment value by 1",
            "chainable": false
          }
        ]
      }
    ]
  }
}
```

**Assertion:** `node_types` array MUST contain valid NodeTypeSchema. Constructor MUST distinguish required (`value`) from optional (`label` has `default`).

### 6.3 Conformance Test Harness

A conformance test harness is provided in `tests/conformance/` (reference implementation):

```
tests/conformance/
├── test_lifecycle.py          # Initialize, shutdown tests
├── test_node_management.py    # Create, sync, call, destroy, serialize
├── test_push_notifications.py # Dirty flag, notifications
├── test_host_api.py           # Host API reverse requests
├── test_schema_discovery.py   # Schema parsing and validation
├── test_error_handling.py     # All error codes
├── test_capabilities.py       # Capability negotiation
└── test_interop.py            # Cross-language tests
```

**Usage:**
```bash
# Run conformance tests against a plugin server
pytest tests/conformance/ --server-command="python -m my_plugin"

# Test specific conformance level
pytest tests/conformance/ -m minimal
pytest tests/conformance/ -m standard
pytest tests/conformance/ -m full
```

---

## 7. Compliance Checklist

Use this checklist to verify CAP conformance.

### 7.1 Minimal Conformance Checklist

- [ ] JSON serialization binding implemented
- [ ] stdio transport binding implemented
- [ ] `initialize` handshake works correctly
- [ ] `shutdown` gracefully terminates connection
- [ ] `node/create` creates nodes and returns initial state
- [ ] `node/sync` executes batched calls in order
- [ ] `node/destroy` releases resources
- [ ] `node/serialize` returns reconstructable state
- [ ] NodeTypeSchema advertised during `initialize`
- [ ] All standard and CAP-specific error codes handled
- [ ] Capability negotiation works (minimal capabilities declared)
- [ ] MISSING sentinel correctly distinguishes required vs optional parameters
- [ ] All required test vectors pass

### 7.2 Standard Conformance Checklist

All Minimal requirements plus:

- [ ] `node/dirty` push notification implemented
- [ ] `node/notification` push notification implemented
- [ ] `push_dirty: true` declared in server capabilities
- [ ] `push_notifications: true` declared in server capabilities
- [ ] Host correctly marks nodes dirty upon receiving `node/dirty`
- [ ] Host correctly propagates `node/notification` to notification system
- [ ] Notification levels (ignore/hold/wake) handled correctly

### 7.3 Full Conformance Checklist

All Standard requirements plus:

- [ ] `node/call` immediate method execution implemented
- [ ] `immediate_call: true` declared if supported
- [ ] `state_changed` flag correctly set in `node/call` responses
- [ ] `host/create_node` creates child nodes in host graph
- [ ] `host/invoke` invokes methods on host nodes
- [ ] `host/query_roots` returns available roots
- [ ] `host/resolve_root` resolves normalized paths
- [ ] All Host API capabilities declared correctly
- [ ] At least one additional serialization binding (Protobuf or MessagePack)
- [ ] At least one additional transport binding (gRPC, WebSocket, or TCP)
- [ ] Content negotiation works for serialization formats
- [ ] Cross-transport interoperability verified (same session behavior)

### 7.4 Interoperability Checklist

- [ ] Graceful degradation works at capability intersections
- [ ] Protocol version negotiation works
- [ ] Cross-language tests pass (if applicable)
- [ ] Error recovery works correctly for all error types
- [ ] Connection loss handled gracefully (pending requests fail, resources released)

---

## Appendix A: Conformance Matrix

| Feature | Minimal | Standard | Full | Reference |
|---------|---------|----------|------|-----------|
| JSON serialization | ✓ | ✓ | ✓ | `serialization-bindings.md` §2 |
| stdio transport | ✓ | ✓ | ✓ | [`transports.md`](transports.md) §3 |
| `initialize` | ✓ | ✓ | ✓ | [`spec.md`](spec.md) §5.1 |
| `shutdown` | ✓ | ✓ | ✓ | [`spec.md`](spec.md) §5.1 |
| `node/create` | ✓ | ✓ | ✓ | [`spec.md`](spec.md) §5.2 |
| `node/sync` | ✓ | ✓ | ✓ | [`spec.md`](spec.md) §5.2 |
| `node/destroy` | ✓ | ✓ | ✓ | [`spec.md`](spec.md) §5.2 |
| `node/serialize` | ✓ | ✓ | ✓ | [`spec.md`](spec.md) §5.2 |
| `node/dirty` | — | ✓ | ✓ | [`spec.md`](spec.md) §5.3 |
| `node/notification` | — | ✓ | ✓ | [`spec.md`](spec.md) §5.3 |
| `node/call` | — | — | ✓ | [`spec.md`](spec.md) §5.2 |
| `host/create_node` | — | — | ✓ | [`spec.md`](spec.md) §5.4 |
| `host/invoke` | — | — | ✓ | [`spec.md`](spec.md) §5.4 |
| `host/query_roots` | — | — | ✓ | [`spec.md`](spec.md) §5.4 |
| `host/resolve_root` | — | — | ✓ | [`spec.md`](spec.md) §5.4 |
| Protobuf serialization | — | — | ✓ | `serialization-bindings.md` §3 |
| MessagePack serialization | — | — | ✓ | `serialization-bindings.md` §4 |
| gRPC transport | — | — | ✓ | [`transports.md`](transports.md) §4 |
| WebSocket transport | — | — | ✓ | [`transports.md`](transports.md) §5 |
| TCP transport | — | — | ✓ | [`transports.md`](transports.md) §6 |

---

## Appendix B: Reference Implementation

The canonical reference implementation is the ActiveContext Python codebase:

| Component | Path | Description |
|-----------|------|-------------|
| **Transport** | `src/activecontext/plugins/transport.py` | stdio transport (PluginTransport/StdioTransport) |
| **Serialization** | `src/activecontext/plugins/serialization.py` | JSON/MessagePack serializers |
| **Wire Types** | `src/activecontext/plugins/wire.py` | All message params/result dataclasses |
| **Protobuf Schema** | `docs/cap.proto` | Canonical protobuf definitions |
| **Conformance Tests** | `tests/test_mcp_transport.py` | Reference test suite |

---

## References

- **Protocol Specification**: [`spec.md`](spec.md)
- **Message Catalog**: [`messages.md`](messages.md)
- **Transport Bindings**: [`transports.md`](transports.md)
- **Serialization Bindings**: Moving `docs/cap-serialization-bindings.md` to `docs/cap/serialization.md`
- **Protobuf Schema**: `../cap.proto`
