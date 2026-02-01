# CAP Message Catalog

> Complete reference for all Context Application Protocol messages with wire examples.

**Specification Version:** 1.0
**Date:** 2025-01-30

---

## Overview

This document provides a comprehensive catalog of all 13 CAP message types, including:
- Message purpose and usage
- Parameter and result schemas
- Wire format examples (JSON-RPC 2.0)
- Error conditions
- Related capabilities

For protocol semantics, see [`spec.md`](spec.md). For serialization details, see `serialization-bindings.md`.

---

## Table of Contents

### Lifecycle Messages
- [`initialize`](#initialize) — Handshake
- [`shutdown`](#shutdown) — Graceful shutdown

### Node Management Messages
- [`node/create`](#nodecreate) — Create node instance
- [`node/sync`](#nodesync) — Synchronize node state (hot path)
- [`node/call`](#nodecall) — Immediate method call
- [`node/destroy`](#nodedestroy) — Destroy node instance
- [`node/serialize`](#nodeserialize) — Serialize for persistence

### Push Notifications
- [`node/dirty`](#nodedirty) — State changed (one-bit signal)
- [`node/notification`](#nodenotification) — General notification

### Host API (Server → Host)
- [`host/create_node`](#hostcreate_node) — Create child node
- [`host/invoke`](#hostinvoke) — Invoke host node method
- [`host/query_roots`](#hostquery_roots) — Query filesystem roots
- [`host/resolve_root`](#hostresolve_root) — Resolve root path

---

## Lifecycle Messages

### `initialize`

**Direction:** Host → Server (request)

**Purpose:** Establishes the connection and negotiates capabilities. MUST be the first message sent after transport connection is established.

**Method:** `"initialize"`

#### Parameters (InitializeParams)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `protocol_version` | string | No | `"1.0"` | Protocol version the host supports. |
| `host_capabilities` | HostCapabilities | No | all `true` | Capabilities the host offers. |
| `roots` | RootInfo[] | No | `[]` | Filesystem roots available to the server. |
| `session_id` | string | No | `""` | Session identifier for logging and coordination. |
| `cwd` | string | No | `""` | Working directory for the session. |

**HostCapabilities:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `create_node` | boolean | `true` | Host supports `host/create_node`. |
| `invoke` | boolean | `true` | Host supports `host/invoke`. |
| `roots` | boolean | `true` | Host supports `host/query_roots` and `host/resolve_root`. |
| `notifications` | boolean | `true` | Host accepts `node/dirty` and `node/notification`. |

**RootInfo:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `uri` | string | Yes | Root URI (e.g., `"file:///home/user/project"`). |
| `name` | string | Yes | Human-readable name (e.g., `"cwd"`, `"home"`). |
| `description` | string | No | What this root represents. |

#### Result (InitializeResult)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `server_name` | string | **Yes** | — | Human-readable server name. |
| `server_version` | string | No | `""` | Server version string. |
| `protocol_version` | string | No | `"1.0"` | Protocol version the server supports. |
| `node_types` | NodeTypeSchema[] | No | `[]` | Node types this server provides. |
| `server_capabilities` | ServerCapabilities | No | defaults | Capabilities the server declares. |

**ServerCapabilities:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sync` | boolean | `true` | Server supports `node/sync`. |
| `immediate_call` | boolean | `false` | Server supports `node/call`. |
| `push_dirty` | boolean | `true` | Server pushes `node/dirty` notifications. |
| `push_notifications` | boolean | `true` | Server pushes `node/notification` messages. |

#### Wire Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1,
  "params": {
    "protocol_version": "1.0",
    "host_capabilities": {
      "create_node": true,
      "invoke": true,
      "roots": true,
      "notifications": true
    },
    "roots": [
      {
        "uri": "file:///home/user/project",
        "name": "cwd",
        "description": "Current working directory"
      }
    ],
    "session_id": "sess_abc123",
    "cwd": "/home/user/project"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "server_name": "my-plugin",
    "server_version": "0.1.0",
    "protocol_version": "1.0",
    "server_capabilities": {
      "sync": true,
      "immediate_call": false,
      "push_dirty": true,
      "push_notifications": true
    },
    "node_types": [
      {
        "node_type": "custom_view",
        "description": "A custom file view with annotations",
        "constructor": {
          "positional": [],
          "variadic": null,
          "named": []
        },
        "properties": [],
        "methods": []
      }
    ]
  }
}
```

#### Error Conditions

- **INVALID_PARAMS**: Missing required fields or malformed params.
- **INTERNAL_ERROR**: Server initialization failed.

---

### `shutdown`

**Direction:** Host → Server (notification)

**Purpose:** Notifies the server to begin cleanup. No response expected.

**Method:** `"shutdown"`

#### Parameters

None.

#### Wire Example

```json
{
  "jsonrpc": "2.0",
  "method": "shutdown"
}
```

#### Processing

Upon receiving `shutdown`, the server MUST:
1. Stop accepting new work for this connection.
2. Complete any in-flight responses.
3. Close its end of the transport.
4. For host-managed processes: exit promptly.
5. For external processes: release connection resources but MAY continue serving other hosts.

---

## Node Management Messages

### `node/create`

**Direction:** Host → Server (request)

**Purpose:** Creates a new node instance of the specified type.

**Method:** `"node/create"`

#### Parameters (NodeCreateParams)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `node_type` | string | **Yes** | — | Type of node to create (must match a registered schema). |
| `args` | Any[] | No | `[]` | Positional constructor arguments. |
| `kwargs` | object | No | `{}` | Keyword constructor arguments. |
| `node_id` | string | No | `""` | Host-assigned node ID. If empty, server generates one. |

#### Result (NodeCreateResult)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `node_id` | string | **Yes** | — | Node instance ID (assigned or generated). |
| `state` | object | No | `{}` | Initial serialized node state. |
| `renders` | object | No | `{}` | Initial renders: `header`, `content`, `detail`. |
| `tokens` | object | No | `{}` | Initial token estimates: `collapsed`, `summary`, `detail`. |
| `digest` | object | No | `{}` | Initial digest metadata. |

#### Wire Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/create",
  "id": 2,
  "params": {
    "node_type": "shell",
    "args": ["pytest", "-v"],
    "kwargs": {"timeout": 120},
    "node_id": "shell_a1b2c3d4"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "node_id": "shell_a1b2c3d4",
    "state": {
      "command": "pytest -v",
      "exit_code": null,
      "is_complete": false
    },
    "renders": {
      "header": "[SHELL] pytest -v [RUNNING]",
      "content": "",
      "detail": ""
    },
    "tokens": {
      "collapsed": 12,
      "summary": 0,
      "detail": 0
    },
    "digest": {
      "type": "shell",
      "status": "running"
    }
  }
}
```

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_TYPE_UNKNOWN` (-32001) | The `node_type` is not registered on this server. |
| `NODE_CREATE_FAILED` (-32005) | Creation failed (invalid args, resource error, etc.). |
| `INVALID_PARAMS` (-32602) | Arguments do not match the constructor schema. |

---

### `node/sync`

**Direction:** Host → Server (request)

**Purpose:** The **hot path** of the protocol. Synchronizes a dirty node by executing batched method calls and retrieving complete state.

**Method:** `"node/sync"`

#### Parameters (NodeSyncParams)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `node_id` | string | **Yes** | — | Node to synchronize. |
| `calls` | PendingCall[] | No | `[]` | Method calls to execute before returning state. |
| `cwd` | string | No | `""` | Working directory for render methods. |

**PendingCall:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | Yes | Method name (e.g., `"set_content"`, `"cancel"`). |
| `args` | Any[] | No | Positional arguments. |
| `kwargs` | object | No | Keyword arguments. |

#### Result (NodeSyncResult)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `state` | object | No | `{}` | Full serialized node state. |
| `renders` | object | No | `{}` | Rendered content: `header`, `content`, `detail`. |
| `tokens` | object | No | `{}` | Token estimates: `collapsed`, `summary`, `detail`. |
| `digest` | object | No | `{}` | Compact digest metadata. |
| `notifications` | Notification[] | No | `[]` | Notifications generated since last sync. |

**Notification:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `description` | string | Yes | Human-readable description. |
| `level` | string | No (default `"hold"`) | `"ignore"`, `"hold"`, or `"wake"`. |

#### Wire Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/sync",
  "id": 5,
  "params": {
    "node_id": "shell_a1b2c3d4",
    "calls": [
      {"method": "cancel", "args": [], "kwargs": {}}
    ],
    "cwd": "/home/user/project"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "state": {
      "command": "pytest -v",
      "exit_code": -15,
      "is_complete": true
    },
    "renders": {
      "header": "[SHELL] pytest -v [CANCELLED]",
      "content": "Process cancelled by user.",
      "detail": "Exit code: -15\nRuntime: 3.2s"
    },
    "tokens": {
      "collapsed": 12,
      "summary": 8,
      "detail": 20
    },
    "digest": {
      "type": "shell",
      "status": "cancelled",
      "exit_code": -15
    },
    "notifications": [
      {
        "description": "Shell cancelled: pytest -v",
        "level": "wake"
      }
    ]
  }
}
```

#### Processing Semantics

1. Server MUST execute all `calls` in array order.
2. If a call fails, server SHOULD continue processing remaining calls and report errors via notifications or state.
3. After all calls are processed, server MUST return complete current state, renders, tokens, and digest.
4. The `renders` object MUST contain `header`, `content`, and `detail` keys (MAY be empty strings).

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_NOT_FOUND` (-32000) | No node with the given ID exists. |
| `METHOD_NOT_AVAILABLE` (-32002) | A method in `calls` is not defined on this node type. |

---

### `node/call`

**Direction:** Host → Server (request)

**Purpose:** Executes a method immediately, outside the tick cycle. Used for operations that need return values without waiting for next tick.

**Method:** `"node/call"`

**Capability Required:** `immediate_call` (server capability)

#### Parameters (NodeCallParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | string | Yes | Node to call method on. |
| `method` | string | Yes | Method name. |
| `args` | Any[] | No | Positional arguments. |
| `kwargs` | object | No | Keyword arguments. |

#### Result (NodeCallResult)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `result` | Any | No | `null` | Return value of the method call. |
| `state_changed` | boolean | No | `false` | Whether the call changed node state (triggers sync on next tick). |

#### Wire Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/call",
  "id": 8,
  "params": {
    "node_id": "custom_e5f6g7h8",
    "method": "get_summary",
    "args": [],
    "kwargs": {"max_tokens": 200}
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": {
    "result": "Authentication module: 3 endpoints, 2 middleware functions.",
    "state_changed": false
  }
}
```

#### Processing

If `state_changed` is `true`, the host SHOULD mark the node as dirty and include it in the next tick's sync cycle.

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_NOT_FOUND` (-32000) | No node with the given ID exists. |
| `METHOD_NOT_AVAILABLE` (-32002) | Method not defined or capability not declared. |
| `INVALID_PARAMS` (-32602) | Arguments do not match the method schema. |

---

### `node/destroy`

**Direction:** Host → Server (request)

**Purpose:** Destroys a node instance and releases all associated resources.

**Method:** `"node/destroy"`

#### Parameters (NodeDestroyParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | string | Yes | Node to destroy. |

#### Result

Empty object `{}` on success.

#### Wire Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/destroy",
  "id": 10,
  "params": {
    "node_id": "shell_a1b2c3d4"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": {}
}
```

#### Processing

After destruction, the node ID MUST NOT be reused for new nodes within the same connection.

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_NOT_FOUND` (-32000) | No node with the given ID exists. |

---

### `node/serialize`

**Direction:** Host → Server (request)

**Purpose:** Retrieves full serialized state for checkpoint or persistence. The returned data MUST be sufficient to reconstruct the node.

**Method:** `"node/serialize"`

#### Parameters (NodeSerializeParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | string | Yes | Node to serialize. |

#### Result (NodeSerializeResult)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `data` | object | No (default `{}`) | Full serialized state for reconstruction. |

The `data` object MUST include a `node_type` field for deserialization dispatch.

#### Wire Example

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "node/serialize",
  "id": 11,
  "params": {
    "node_id": "shell_a1b2c3d4"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 11,
  "result": {
    "data": {
      "node_type": "shell",
      "command": "pytest -v",
      "exit_code": 0,
      "is_complete": true,
      "output": "3 passed in 1.2s"
    }
  }
}
```

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_NOT_FOUND` (-32000) | No node with the given ID exists. |

---

## Push Notifications

### `node/dirty`

**Direction:** Server → Host (notification)

**Purpose:** One-bit signal indicating node state changed asynchronously. The host MUST mark the node dirty and include it in the next tick's sync cycle.

**Method:** `"node/dirty"`

**Capability Required:** `push_dirty` (server capability)

#### Parameters (NodeDirtyParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | string | Yes | Node whose state changed. |

#### Wire Example

```json
{
  "jsonrpc": "2.0",
  "method": "node/dirty",
  "params": {
    "node_id": "shell_a1b2c3d4"
  }
}
```

#### Processing

This notification carries no payload beyond the node ID. The full state is retrieved during the subsequent `node/sync`.

---

### `node/notification`

**Direction:** Server → Host (notification)

**Purpose:** Delivers a general notification from a node to the host. The host MUST propagate it to the node's parent chain via the graph's notification system.

**Method:** `"node/notification"`

**Capability Required:** `push_notifications` (server capability)

#### Parameters (NodeNotificationParams)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `node_id` | string | Yes | — | Source node. |
| `description` | string | Yes | — | Human-readable description. |
| `level` | string | No | `"hold"` | Notification level (see below). |

**Notification Levels:**

| Level | Semantics |
|-------|-----------|
| `"ignore"` | Informational only. Host MAY log but SHOULD NOT wake agent. |
| `"hold"` | Noteworthy change. Host SHOULD queue for agent's next turn. |
| `"wake"` | Urgent. Host SHOULD wake agent if idle or waiting. |

#### Wire Example

```json
{
  "jsonrpc": "2.0",
  "method": "node/notification",
  "params": {
    "node_id": "shell_a1b2c3d4",
    "description": "Shell completed: pytest -v (exit 0)",
    "level": "wake"
  }
}
```

---

## Host API (Server → Host Requests)

### `host/create_node`

**Direction:** Server → Host (request)

**Purpose:** Requests creation of a child node in the host's context graph. Used for nested node creation (e.g., a shell plugin creating artifact children for output segments).

**Method:** `"host/create_node"`

**Capability Required:** `create_node` (host capability)

#### Parameters (HostCreateNodeParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_type` | string | Yes | Type of node to create (must be registered in host). |
| `args` | Any[] | No | Positional constructor arguments. |
| `kwargs` | object | No | Keyword constructor arguments. |
| `parent_id` | string | No | Parent node ID to link the new node to. |

#### Result (HostCreateNodeResult)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | string | Yes | ID of the created node in the host graph. |

#### Wire Example

**Request (server sends):**
```json
{
  "jsonrpc": "2.0",
  "method": "host/create_node",
  "id": 100,
  "params": {
    "node_type": "artifact",
    "args": ["test output"],
    "kwargs": {
      "artifact_type": "output",
      "language": "text"
    },
    "parent_id": "shell_a1b2c3d4"
  }
}
```

**Response (host sends):**
```json
{
  "jsonrpc": "2.0",
  "id": 100,
  "result": {
    "node_id": "artifact_x9y8z7w6"
  }
}
```

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_TYPE_UNKNOWN` (-32001) | Node type not registered in host. |
| `NODE_NOT_FOUND` (-32000) | `parent_id` does not exist in host graph. |
| `NODE_CREATE_FAILED` (-32005) | Creation failed. |
| `PERMISSION_DENIED` (-32004) | Host does not support `create_node` capability. |

---

### `host/invoke`

**Direction:** Server → Host (request)

**Purpose:** Invokes a method on an existing node in the host graph. Used for cross-node interactions.

**Method:** `"host/invoke"`

**Capability Required:** `invoke` (host capability)

#### Parameters (HostInvokeParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `node_id` | string | Yes | Target node in the host graph. |
| `method` | string | Yes | Method name to call. |
| `args` | Any[] | No | Positional arguments. |
| `kwargs` | object | No | Keyword arguments. |

#### Result (HostInvokeResult)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `result` | Any | No | `null` | Return value of the method call. |

#### Wire Example

**Request (server sends):**
```json
{
  "jsonrpc": "2.0",
  "method": "host/invoke",
  "id": 101,
  "params": {
    "node_id": "text_m1n2o3p4",
    "method": "set_content",
    "args": ["Updated file content"],
    "kwargs": {}
  }
}
```

**Response (host sends):**
```json
{
  "jsonrpc": "2.0",
  "id": 101,
  "result": {
    "result": null
  }
}
```

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `NODE_NOT_FOUND` (-32000) | Target node does not exist in host graph. |
| `METHOD_NOT_AVAILABLE` (-32002) | Method not available on target node. |
| `PERMISSION_DENIED` (-32004) | Host does not support `invoke` capability. |

---

### `host/query_roots`

**Direction:** Server → Host (request)

**Purpose:** Queries the filesystem roots available to the plugin. Returns all roots provided during initialization plus any added since.

**Method:** `"host/query_roots"`

**Capability Required:** `roots` (host capability)

#### Parameters

None (empty object or omitted).

#### Result (HostQueryRootsResult)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `roots` | RootInfo[] | No (default `[]`) | Available filesystem roots. |

#### Wire Example

**Request (server sends):**
```json
{
  "jsonrpc": "2.0",
  "method": "host/query_roots",
  "id": 102,
  "params": {}
}
```

**Response (host sends):**
```json
{
  "jsonrpc": "2.0",
  "id": 102,
  "result": {
    "roots": [
      {
        "uri": "file:///home/user/project",
        "name": "cwd",
        "description": "Current working directory"
      },
      {
        "uri": "file:///home/user",
        "name": "home",
        "description": "User home directory"
      }
    ]
  }
}
```

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `PERMISSION_DENIED` (-32004) | Host does not support `roots` capability. |

---

### `host/resolve_root`

**Direction:** Server → Host (request)

**Purpose:** Maps a normalized path through a root to a real filesystem path. Used by plugins that work with file references but need absolute paths for I/O.

**Method:** `"host/resolve_root"`

**Capability Required:** `roots` (host capability)

#### Parameters (HostResolveRootParams)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `root_uri` | string | Yes | Root URI to resolve against. |
| `normalized_path` | string | Yes | Path relative to the root. |

#### Result (HostResolveRootResult)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `real_path` | string | Yes | — | Resolved absolute filesystem path. |
| `exists` | boolean | No | `false` | Whether the file exists at the resolved path. |

#### Wire Example

**Request (server sends):**
```json
{
  "jsonrpc": "2.0",
  "method": "host/resolve_root",
  "id": 103,
  "params": {
    "root_uri": "file:///home/user/project",
    "normalized_path": "src/main.py"
  }
}
```

**Response (host sends):**
```json
{
  "jsonrpc": "2.0",
  "id": 103,
  "result": {
    "real_path": "/home/user/project/src/main.py",
    "exists": true
  }
}
```

#### Error Conditions

| Error Code | Condition |
|------------|-----------|
| `ROOT_NOT_FOUND` (-32003) | `root_uri` does not match any known root. |
| `PERMISSION_DENIED` (-32004) | Host does not support `roots` capability. |

---

## Appendix: Message Summary Table

| Method | Direction | Type | Category | Capability |
|--------|-----------|------|----------|------------|
| `initialize` | Host → Server | Request | Lifecycle | — |
| `shutdown` | Host → Server | Notification | Lifecycle | — |
| `node/create` | Host → Server | Request | Node Management | — |
| `node/sync` | Host → Server | Request | Node Management | `sync` (server) |
| `node/call` | Host → Server | Request | Node Management | `immediate_call` (server) |
| `node/destroy` | Host → Server | Request | Node Management | — |
| `node/serialize` | Host → Server | Request | Node Management | — |
| `node/dirty` | Server → Host | Notification | Push | `push_dirty` (server) |
| `node/notification` | Server → Host | Notification | Push | `push_notifications` (server) |
| `host/create_node` | Server → Host | Request | Host API | `create_node` (host) |
| `host/invoke` | Server → Host | Request | Host API | `invoke` (host) |
| `host/query_roots` | Server → Host | Request | Host API | `roots` (host) |
| `host/resolve_root` | Server → Host | Request | Host API | `roots` (host) |

---

## References

- **Protocol Specification**: [`spec.md`](spec.md)
- **Serialization Bindings**: `serialization-bindings.md`
- **Transport Bindings**: `transport-bindings.md`
- **Protobuf Schema**: `../cap.proto`
