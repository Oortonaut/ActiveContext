# CAP Serialization Bindings

> Normative specification for concrete serialization formats of CAP messages, version 1.0.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## 1. Overview

This document defines **serialization bindings** for the Context Application
Protocol (CAP). A serialization binding maps the abstract CAP messages defined
in the [CAP Protocol Semantics](cap-protocol-semantics.md) specification to
concrete byte representations suitable for transmission over a transport.

The protocol semantics specification (Sections 1.2 and 1.3) establishes that
CAP is transport-independent and encoding-independent. This companion document
fulfills that promise by defining three serialization formats at distinct
points on the performance/debuggability spectrum:

| Binding | Status | Optimized For |
|---------|--------|---------------|
| **JSON** (Section 2) | REQUIRED | Debuggability, universality |
| **Protobuf** (Section 3) | OPTIONAL | Throughput, schema enforcement |
| **MessagePack** (Section 4) | OPTIONAL | Compact size, low overhead |

All implementations MUST support the JSON binding. Support for Protobuf and
MessagePack bindings is OPTIONAL.

### 1.1 Relationship to Protocol Semantics

This document does not define new message types or alter message semantics.
Every message type in this document corresponds to a message defined in
[CAP Protocol Semantics, Section 4](cap-protocol-semantics.md#4-message-categories).
The 12 message types (`initialize`, `shutdown`, `node/create`, `node/sync`,
`node/call`, `node/destroy`, `node/serialize`, `node/dirty`,
`node/notification`, `host/create_node`, `host/invoke`, `host/query_roots`,
`host/resolve_root`) are serialized identically by all bindings in terms of
logical content -- only the byte representation differs.

### 1.2 Reference Implementation

The reference serialization is implemented in `src/activecontext/plugins/wire.py`,
which defines Python dataclasses for all message params and results, plus
helper functions `to_jsonrpc_request()`, `to_jsonrpc_notification()`,
`to_jsonrpc_response()`, and `to_jsonrpc_error()` for the JSON binding.

---

## 2. JSON Binding (Required Baseline)

The JSON binding is the REQUIRED baseline serialization. All CAP
implementations MUST support it. It uses JSON-RPC 2.0 as the envelope format.

### 2.1 JSON-RPC 2.0 Envelope

All JSON-serialized CAP messages MUST conform to the
[JSON-RPC 2.0 specification](https://www.jsonrpc.org/specification).

#### 2.1.1 Request Format

A request expects a response. It carries an `id` field for correlation.

```json
{
  "jsonrpc": "2.0",
  "method": "<method-name>",
  "params": { ... },
  "id": <integer-or-string>
}
```

- The `jsonrpc` field MUST be exactly `"2.0"`.
- The `method` field MUST be one of the method names defined in
  [CAP Protocol Semantics, Appendix A](cap-protocol-semantics.md#appendix-a-message-summary)
  (e.g., `"initialize"`, `"node/create"`, `"host/invoke"`).
- The `params` field MUST be a JSON object (not an array). CAP does not use
  positional JSON-RPC parameters.
- The `id` field MUST be an integer or string. Implementations SHOULD use
  monotonically increasing integers.

**Example** -- `node/create` request (see Protocol Semantics, Section 4.2.1):

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

#### 2.1.2 Response Format (Success)

A successful response carries the `result` field.

```json
{
  "jsonrpc": "2.0",
  "result": { ... },
  "id": <matching-request-id>
}
```

- The `id` MUST match the request's `id`.
- The `result` field MUST be a JSON object whose shape matches the result
  type for the corresponding method.

**Example** -- `node/create` response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "node_id": "shell_a1b2c3d4",
    "state": {"command": "pytest -v", "exit_code": null, "is_complete": false},
    "renders": {
      "header": "[SHELL] pytest -v [RUNNING]",
      "content": "",
      "detail": ""
    },
    "tokens": {"collapsed": 12, "summary": 0, "detail": 0},
    "digest": {"type": "shell", "status": "running"}
  }
}
```

#### 2.1.3 Response Format (Error)

An error response carries the `error` field instead of `result`.

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": <integer>,
    "message": "<human-readable-description>",
    "data": <optional-structured-data>
  },
  "id": <matching-request-id-or-null>
}
```

- The `code` MUST be an integer from the error code table in
  [Protocol Semantics, Section 8.2](cap-protocol-semantics.md#82-error-codes).
- The `message` MUST be a non-empty string.
- The `data` field is OPTIONAL and MAY contain additional structured error
  context (e.g., `{"node_id": "shell_a1b2c3d4"}`).
- If the error occurred before the `id` could be parsed (e.g., `PARSE_ERROR`),
  `id` MUST be `null`.

**Example** -- `NODE_NOT_FOUND` error:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Node not found: shell_a1b2c3d4",
    "data": {"node_id": "shell_a1b2c3d4"}
  },
  "id": 5
}
```

#### 2.1.4 Notification Format

A notification is a one-way message with no `id` and no response expected.

```json
{
  "jsonrpc": "2.0",
  "method": "<method-name>",
  "params": { ... }
}
```

- Notifications MUST NOT include an `id` field.
- The receiver MUST NOT send a response to a notification.

CAP notification methods: `shutdown` (host to server), `node/dirty` and
`node/notification` (server to host). See Protocol Semantics, Sections 4.1.2
and 4.3.

**Example** -- `node/dirty` notification:

```json
{
  "jsonrpc": "2.0",
  "method": "node/dirty",
  "params": {"node_id": "shell_a1b2c3d4"}
}
```

### 2.2 Field Naming

All field names in JSON-serialized CAP messages MUST use `snake_case`.

This convention applies to:
- JSON-RPC envelope fields (`jsonrpc`, `method`, `params`, `result`, `error`, `id`)
  as mandated by JSON-RPC 2.0.
- All CAP params and result fields (`node_type`, `node_id`, `protocol_version`,
  `host_capabilities`, `server_capabilities`, `state_changed`, etc.).
- Fields within nested objects (`push_dirty`, `push_notifications`,
  `immediate_call`, `is_complete`, `exit_code`, etc.).

Implementations MUST reject messages with incorrectly cased field names
(e.g., `nodeType` instead of `node_type`) as `INVALID_PARAMS` (-32602).

### 2.3 Type Mapping

The following table defines the mapping from implementation types to JSON types.
The "Python Type" column uses the reference implementation's type annotations
from `wire.py`.

| Python Type | JSON Type | Notes |
|-------------|-----------|-------|
| `str` | string | UTF-8 encoded. |
| `int` | number | Integer values only. No fractional part. |
| `float` | number | IEEE 754 double-precision. `NaN` and `Infinity` MUST NOT appear; use `null` instead. |
| `bool` | boolean | `true` / `false`. |
| `None` | null | Explicit absence of value. Distinct from MISSING (Section 2.4). |
| `list[T]` | array | Homogeneous arrays. Elements follow their own type mapping. |
| `dict[str, T]` | object | Keys MUST be strings. Values follow their own type mapping. |
| `@dataclass` | object | Recursive serialization. Each field becomes a key-value pair. |
| `Enum` | string | Serialized as `enum_member.value`. E.g., `PluginConnectionStatus.CONNECTED` becomes `"connected"`. |
| `Any` | any JSON value | Used for dynamic fields like `state`, `result`, `data`. |

#### 2.3.1 Numeric Precision

Integers MUST be representable as signed 53-bit values (within the safe
integer range of IEEE 754: -2^53 + 1 to 2^53 - 1). Implementations that need
larger integers MUST encode them as strings.

Floating-point values MUST use the JSON number format. The special values
`NaN`, `+Infinity`, and `-Infinity` are not valid JSON. Implementations
encountering these values MUST serialize them as `null`.

#### 2.3.2 Dataclass Serialization

Dataclasses (the `@dataclass` types in `wire.py`) are serialized by converting
each field to a key-value pair in a JSON object. The field name becomes the
key (in `snake_case`), and the field value is recursively serialized according
to this type mapping.

**Example** -- `PendingCall` dataclass:

```python
PendingCall(method="cancel", args=[], kwargs={})
```

Serializes to:

```json
{"method": "cancel", "args": [], "kwargs": {}}
```

**Example** -- Nested dataclass (`InitializeParams` containing `HostCapabilities`):

```python
InitializeParams(
    protocol_version="1.0",
    host_capabilities=HostCapabilities(create_node=True, invoke=True, roots=True, notifications=True),
    roots=[RootInfo(uri="file:///project", name="cwd", description="Working dir")],
    session_id="sess_001",
    cwd="/project"
)
```

Serializes to:

```json
{
  "protocol_version": "1.0",
  "host_capabilities": {
    "create_node": true,
    "invoke": true,
    "roots": true,
    "notifications": true
  },
  "roots": [
    {"uri": "file:///project", "name": "cwd", "description": "Working dir"}
  ],
  "session_id": "sess_001",
  "cwd": "/project"
}
```

### 2.4 The MISSING Sentinel

The distinction between "not provided" and "explicitly null" is critical in
CAP, particularly for `ParamSchema.default` values (see Protocol Semantics,
Section 6.3.1).

#### 2.4.1 Semantics

| Wire Representation | Meaning | Example |
|---------------------|---------|---------|
| Field **absent** from JSON object | MISSING -- not provided, no value | `{"name": "timeout", "type": "int"}` (no `default` key) |
| Field **present** with value `null` | Explicitly `None` | `{"name": "cwd", "type": "str", "default": null}` |
| Field **present** with a value | Provided value | `{"name": "timeout", "type": "int", "default": 120}` |

#### 2.4.2 Serialization Rules

- When serializing a field whose value is the `MISSING` sentinel (as defined
  in `wire.py`), the sender MUST **omit** the field from the JSON object
  entirely. The sender MUST NOT include the field with a `null` value.

- When serializing a field whose value is `None`, the sender MUST include
  the field with the JSON value `null`.

- Receivers MUST distinguish between an absent field and a field with value
  `null`. An absent field means the sender did not provide a value; a `null`
  field means the sender explicitly provided `None`.

#### 2.4.3 Impact on ParamSchema

This distinction is most visible in constructor parameter schemas. A
`ParamSchema` with no `default` key describes a **required** parameter. A
`ParamSchema` with `"default": null` describes an **optional** parameter whose
default is `None`.

**Example** -- Required parameter (no default):

```json
{
  "name": "command",
  "type": "str",
  "description": "The command to execute."
}
```

**Example** -- Optional parameter with null default:

```json
{
  "name": "cwd",
  "type": "str",
  "default": null,
  "description": "Working directory override."
}
```

**Example** -- Optional parameter with non-null default:

```json
{
  "name": "timeout",
  "type": "int",
  "default": 120,
  "description": "Timeout in seconds."
}
```

#### 2.4.4 MISSING in Other Fields

Beyond `ParamSchema.default`, the MISSING convention applies to any field
marked as not required in the Protocol Semantics specification. When a params
or result field has a default value in its schema table (e.g., `state`
defaults to `{}` in `NodeCreateResult`), senders MAY omit the field if the
value equals the default. Receivers MUST apply the documented default when a
field is absent.

### 2.5 DateTime and Binary Encoding

#### 2.5.1 DateTime Values

Date and time values MUST be serialized as ISO 8601 strings in UTC with
the `Z` suffix.

```
Format: YYYY-MM-DDTHH:MM:SS.sssZ
```

**Example:**

```json
{"created_at": "2025-06-15T14:30:00.000Z"}
```

- Millisecond precision is RECOMMENDED but not required.
- Implementations MUST accept timestamps with or without fractional seconds.
- Implementations MUST accept timestamps with `Z` or `+00:00` UTC designator.
- Timestamps without a timezone designator SHOULD be interpreted as UTC.

> **Note:** CAP 1.0 does not currently define message fields with datetime
> types. This section is provided for future extensions and for dynamic
> `state` objects that may contain timestamps.

#### 2.5.2 Binary Data

Binary data MUST be encoded as base64 strings using the standard alphabet
(RFC 4648, Section 4) with `=` padding.

```json
{"data": "SGVsbG8gV29ybGQ="}
```

- Implementations MUST use standard base64 (not URL-safe variant) unless a
  future extension specifies otherwise.
- The base64 string MUST NOT contain line breaks.

> **Note:** Like datetime, CAP 1.0 does not define binary fields in its core
> messages. This section governs binary data that may appear in dynamic fields
> (`state`, `data`, `result`) or future protocol extensions.

### 2.6 Framing

When the JSON binding is used over a stream transport (e.g., stdio pipes),
messages MUST be framed as newline-delimited JSON (NDJSON):

- Each message is a single JSON object serialized on one line.
- Messages are separated by a single newline character (`\n`, U+000A).
- The JSON object MUST NOT contain unescaped newline characters.
- Implementations SHOULD send messages without trailing whitespace before the
  newline.

**Example** -- Two messages on the wire:

```
{"jsonrpc":"2.0","method":"node/dirty","params":{"node_id":"shell_1"}}\n
{"jsonrpc":"2.0","method":"node/sync","id":5,"params":{"node_id":"shell_1","calls":[],"cwd":"/project"}}\n
```

For transport bindings that provide their own framing (e.g., WebSocket message
boundaries), the newline delimiter is OPTIONAL but RECOMMENDED for consistency.

### 2.7 Character Encoding

All JSON messages MUST be encoded as UTF-8. Implementations MUST NOT use a
byte order mark (BOM). String values MAY contain any valid Unicode code point
except the control characters that JSON requires to be escaped (U+0000 through
U+001F).

---

## 3. Protobuf Binding (Performance Tier)

The Protocol Buffers (protobuf) binding provides a schema-enforced binary
serialization for performance-sensitive deployments. This binding is OPTIONAL.

### 3.1 Message Mapping

Each CAP message type maps to a pair of protobuf messages: one for
params/request and one for result/response. The method name is encoded in a
top-level `CapMessage` envelope.

#### 3.1.1 Envelope Message

```protobuf
syntax = "proto3";
package cap.v1;

message CapMessage {
  string jsonrpc = 1;       // Always "2.0" (for interop verification)
  oneof id_value {
    int64 id_int = 2;
    string id_str = 3;
  }
  string method = 4;

  oneof payload {
    // Lifecycle
    InitializeParams initialize_params = 10;
    InitializeResult initialize_result = 11;

    // Node Management
    NodeCreateParams node_create_params = 12;
    NodeCreateResult node_create_result = 13;
    NodeSyncParams node_sync_params = 14;
    NodeSyncResult node_sync_result = 15;
    NodeCallParams node_call_params = 16;
    NodeCallResult node_call_result = 17;
    NodeDestroyParams node_destroy_params = 18;
    NodeSerializeParams node_serialize_params = 19;
    NodeSerializeResult node_serialize_result = 20;

    // Push Notifications
    NodeDirtyParams node_dirty_params = 21;
    NodeNotificationParams node_notification_params = 22;

    // Host API
    HostCreateNodeParams host_create_node_params = 23;
    HostCreateNodeResult host_create_node_result = 24;
    HostInvokeParams host_invoke_params = 25;
    HostInvokeResult host_invoke_result = 26;
    HostQueryRootsResult host_query_roots_result = 27;
    HostResolveRootParams host_resolve_root_params = 28;
    HostResolveRootResult host_resolve_root_result = 29;

    // Error
    RpcError error = 30;
  }
}

message RpcError {
  int32 code = 1;
  string message = 2;
  google.protobuf.Value data = 3;
}
```

#### 3.1.2 Field Numbering Convention

Protobuf encodes field numbers 1-15 in a single byte, while higher numbers
require two or more bytes. Implementations SHOULD reserve field numbers 1-15
for the most frequently accessed fields (hot-path fields).

For each message type:

- **Fields 1-5**: Identifiers and keys (`node_id`, `node_type`, `method`).
  These appear on every message and benefit most from compact encoding.
- **Fields 6-15**: Primary payload fields (`state`, `renders`, `tokens`,
  `calls`, `args`, `kwargs`). These are read on every sync cycle.
- **Fields 16+**: Metadata, descriptions, and rarely-accessed fields
  (`description`, `digest`, `notifications`).

#### 3.1.3 Per-Message Definitions

Each CAP params/result dataclass maps to a protobuf message. Field names
use `snake_case` matching the JSON binding.

**Example** -- `NodeSyncParams` and `NodeSyncResult`:

```protobuf
message PendingCall {
  string method = 1;
  repeated google.protobuf.Value args = 2;
  google.protobuf.Struct kwargs = 3;
}

message NodeSyncParams {
  string node_id = 1;
  repeated PendingCall calls = 2;
  string cwd = 3;
}

message NodeSyncResult {
  google.protobuf.Struct state = 1;
  map<string, string> renders = 2;
  map<string, int32> tokens = 3;
  google.protobuf.Struct digest = 4;
  repeated Notification notifications = 15;
}

message Notification {
  string description = 1;
  string level = 2;    // "ignore", "hold", "wake"
}
```

**Example** -- `InitializeParams` and `InitializeResult`:

```protobuf
message HostCapabilities {
  bool create_node = 1;
  bool invoke = 2;
  bool roots = 3;
  bool notifications = 4;
}

message ServerCapabilities {
  bool sync = 1;
  bool immediate_call = 2;
  bool push_dirty = 3;
  bool push_notifications = 4;
}

message RootInfo {
  string uri = 1;
  string name = 2;
  string description = 3;
}

message InitializeParams {
  string protocol_version = 1;
  HostCapabilities host_capabilities = 2;
  repeated RootInfo roots = 3;
  string session_id = 4;
  string cwd = 5;
}

message InitializeResult {
  string server_name = 1;
  string server_version = 2;
  string protocol_version = 3;
  repeated NodeTypeSchema node_types = 4;
  ServerCapabilities server_capabilities = 5;
}
```

### 3.2 Dynamic State

CAP nodes expose dynamic `state` dictionaries with arbitrary structure (see
`NodeSyncResult.state` and `NodeCreateResult.state`). Protobuf's static
schema requires special handling for these dynamic fields.

#### 3.2.1 google.protobuf.Struct for Dictionaries

Dynamic dictionary fields (`state`, `digest`, `kwargs`) MUST use
`google.protobuf.Struct`. `Struct` maps to a JSON object and supports string
keys with `Value` payloads.

```protobuf
import "google/protobuf/struct.proto";

message NodeSyncResult {
  google.protobuf.Struct state = 1;   // Dynamic node state
  // ...
}
```

#### 3.2.2 google.protobuf.Value for Any-Typed Fields

Fields typed as `Any` in the Protocol Semantics spec (e.g.,
`NodeCallResult.result`, `HostInvokeResult.result`, `RpcError.data`) MUST use
`google.protobuf.Value`.

`Value` is a union type supporting null, number, string, bool, struct, and
list -- matching JSON's type system.

```protobuf
message NodeCallResult {
  google.protobuf.Value result = 1;   // Any return value
  bool state_changed = 2;
}
```

#### 3.2.3 When to Prefer Typed Fields

Implementations SHOULD use typed protobuf fields over `Struct`/`Value` when:

- The field has a fixed, known schema (e.g., `renders` as `map<string, string>`
  rather than `Struct`).
- The field appears in the hot path (`node/sync` results) where parsing
  overhead matters.
- The field is used for capability negotiation or handshake (where schema
  validation prevents subtle bugs).

Implementations SHOULD use `Struct`/`Value` when:

- The field carries plugin-defined dynamic data (`state`, `digest`).
- The field type is `Any` in the Protocol Semantics spec.
- The schema is not known at compile time.

### 3.3 MISSING in Protobuf

The MISSING sentinel (Section 2.4) maps to protobuf's field presence mechanism.

#### 3.3.1 Proto3 Field Presence

Proto3 supports two field presence models:

1. **Implicit presence** (default): Fields have default zero-values. There is
   no way to distinguish "not set" from "set to default".
2. **Explicit presence** (`optional` keyword): The `has_<field>()` method
   distinguishes "not set" from "set to default".

CAP fields that use the MISSING sentinel MUST be declared with the `optional`
keyword to preserve the absent-vs-null distinction.

#### 3.3.2 Application to ParamSchema

```protobuf
message ParamSchema {
  string name = 1;
  string type = 2;
  optional google.protobuf.Value default = 3;  // MISSING = not set
  string description = 4;
}
```

- If `has_default()` returns `false`: the parameter is **required** (MISSING).
- If `has_default()` returns `true` and value is `null_value`: default is `None`.
- If `has_default()` returns `true` and value is non-null: default is the value.

#### 3.3.3 Application to Optional Result Fields

Result fields with defaults (e.g., `NodeCreateResult.state` defaults to `{}`)
SHOULD use `optional` so that receivers can distinguish "server returned empty
state" from "server did not include state field".

```protobuf
message NodeCreateResult {
  string node_id = 1;
  optional google.protobuf.Struct state = 2;
  optional map<string, string> renders = 3;
  optional map<string, int32> tokens = 4;
  optional google.protobuf.Struct digest = 5;
}
```

### 3.4 Service Definition Pattern

When protobuf is used with gRPC or a similar RPC framework, the following
service definition patterns apply.

#### 3.4.1 Unary RPCs for Request/Response Messages

Standard CAP request/response pairs map directly to unary RPCs.

```protobuf
service CapNodeManagement {
  rpc Initialize(InitializeParams) returns (InitializeResult);
  rpc NodeCreate(NodeCreateParams) returns (NodeCreateResult);
  rpc NodeSync(NodeSyncParams) returns (NodeSyncResult);
  rpc NodeCall(NodeCallParams) returns (NodeCallResult);
  rpc NodeDestroy(NodeDestroyParams) returns (google.protobuf.Empty);
  rpc NodeSerialize(NodeSerializeParams) returns (NodeSerializeResult);
}

service CapHostApi {
  rpc HostCreateNode(HostCreateNodeParams) returns (HostCreateNodeResult);
  rpc HostInvoke(HostInvokeParams) returns (HostInvokeResult);
  rpc HostQueryRoots(google.protobuf.Empty) returns (HostQueryRootsResult);
  rpc HostResolveRoot(HostResolveRootParams) returns (HostResolveRootResult);
}
```

#### 3.4.2 Server Streaming for Push Notifications

Push notifications (`node/dirty`, `node/notification`) map to a server-side
stream from the plugin server to the host.

```protobuf
service CapPushNotifications {
  rpc SubscribeNotifications(google.protobuf.Empty) returns (stream PushNotification);
}

message PushNotification {
  oneof notification {
    NodeDirtyParams node_dirty = 1;
    NodeNotificationParams node_notification = 2;
  }
}
```

#### 3.4.3 Bidirectional Streaming Alternative

For deployments where both sides need to send messages asynchronously (the
common CAP case), a single bidirectional stream using the `CapMessage`
envelope provides a closer match to CAP's stdio model.

```protobuf
service CapBidirectional {
  rpc Channel(stream CapMessage) returns (stream CapMessage);
}
```

This pattern:
- Preserves the interleaved request/notification pattern of stdio-based CAP.
- Allows both host and server to send at any time without waiting.
- Requires application-level request/response correlation via `id` fields.

Implementations SHOULD prefer the bidirectional stream pattern when migrating
from stdio to gRPC, as it requires the fewest semantic changes.

### 3.5 Wire Example

**`node/sync` request as protobuf** (conceptual; actual bytes are binary):

```
CapMessage {
  jsonrpc: "2.0"
  id_int: 5
  method: "node/sync"
  node_sync_params: {
    node_id: "shell_a1b2c3d4"
    calls: [
      { method: "cancel", args: [], kwargs: {} }
    ]
    cwd: "/home/user/project"
  }
}
```

---

## 4. MessagePack Binding (Compact Tier)

The MessagePack binding provides a compact binary encoding that preserves the
logical structure of the JSON binding while reducing message size and parse
overhead. This binding is OPTIONAL.

### 4.1 Structure

MessagePack-serialized CAP messages MUST follow the same logical structure as
the JSON binding (Section 2). The MessagePack encoding replaces JSON's text
representation with a binary format but does NOT change the structure.

#### 4.1.1 Mapping from JSON

| JSON Construct | MessagePack Equivalent |
|----------------|------------------------|
| JSON object `{...}` | MessagePack map |
| JSON array `[...]` | MessagePack array |
| JSON string | MessagePack str |
| JSON integer | MessagePack int (signed/unsigned, variable width) |
| JSON float | MessagePack float64 |
| JSON boolean | MessagePack bool |
| JSON null | MessagePack nil |

#### 4.1.2 Field Names as Strings

MessagePack-serialized CAP messages MUST use string keys (not integer keys)
in map objects. This preserves debuggability -- a MessagePack message can be
decoded and inspected without a schema.

```
MessagePack equivalent of {"jsonrpc": "2.0", "method": "node/dirty", "params": {"node_id": "shell_1"}}:

fixmap(3):
  fixstr("jsonrpc")  -> fixstr("2.0")
  fixstr("method")   -> fixstr("node/dirty")
  fixstr("params")   -> fixmap(1):
                           fixstr("node_id") -> fixstr("shell_1")
```

> **Rationale:** Integer keys would save a few bytes per field but make
> debugging significantly harder. Since MessagePack already achieves good
> compression over JSON through binary encoding of values, the additional
> saving from integer keys is marginal.

#### 4.1.3 MISSING Semantics

The MISSING sentinel follows the same convention as the JSON binding:

- A field whose value is MISSING MUST be **omitted** from the MessagePack map.
- A field whose value is `None` MUST be included as MessagePack `nil`.
- Receivers MUST distinguish between an absent key and a key with `nil` value.

### 4.2 Envelope Structure

MessagePack messages use the same envelope structure as JSON-RPC 2.0, encoded
in MessagePack binary format.

**Request:**
```
map {
  "jsonrpc": "2.0",
  "method": "<method-name>",
  "params": { ... },
  "id": <integer>
}
```

**Notification:**
```
map {
  "jsonrpc": "2.0",
  "method": "<method-name>",
  "params": { ... }
}
```

**Response (success):**
```
map {
  "jsonrpc": "2.0",
  "result": { ... },
  "id": <integer>
}
```

**Response (error):**
```
map {
  "jsonrpc": "2.0",
  "error": {"code": <int>, "message": "<str>"},
  "id": <integer-or-nil>
}
```

### 4.3 Type Mapping Specifics

| CAP Type | MessagePack Format |
|----------|-------------------|
| `str` | str family (fixstr, str 8/16/32). UTF-8 encoded. |
| `int` | int family. Implementations SHOULD use the smallest encoding that fits the value. |
| `float` | float 64. Implementations MUST use float64, not float32, to match JSON numeric precision. |
| `bool` | true (0xc3) or false (0xc2). |
| `None` | nil (0xc0). |
| `list` | array family. |
| `dict` | map family. Keys MUST be str. |
| `bytes` | bin family (bin 8/16/32). Unlike JSON binding, binary data MAY use native binary encoding instead of base64. |
| `datetime` | ext type -1 (timestamp extension, MessagePack spec). Alternatively, str in ISO 8601 format. |

#### 4.3.1 Binary Data Advantage

Unlike the JSON binding, the MessagePack binding MAY encode binary data using
MessagePack's native binary format (`bin` family) instead of base64-encoded
strings. This avoids the ~33% size overhead of base64 encoding.

Implementations MUST support both formats for interoperability:
- When **sending**, implementations SHOULD use native binary encoding.
- When **receiving**, implementations MUST accept both native binary and
  base64-encoded strings.

### 4.4 Framing

When MessagePack is used over a stream transport, messages MUST be
length-prefixed:

```
[4 bytes: big-endian uint32 message length][message bytes]
```

- The length prefix is the byte count of the MessagePack-encoded message,
  NOT including the 4-byte prefix itself.
- Implementations MUST NOT use newline-delimited framing for MessagePack
  (unlike the JSON binding) because MessagePack messages may contain 0x0A
  bytes in their payload.

### 4.5 When to Prefer MessagePack

The MessagePack binding is RECOMMENDED for:

- **Bandwidth-constrained transports**: Mobile devices, embedded systems,
  metered network connections.
- **High-frequency sync messages**: When `node/sync` is called at high tick
  rates (sub-second), the reduced parse and serialization overhead is
  beneficial.
- **Large state objects**: Nodes with substantial `state` dictionaries benefit
  from compact encoding.

The MessagePack binding is NOT RECOMMENDED when:

- **Human readability matters**: Development, debugging, and logging benefit
  from JSON's text format.
- **Interoperability is paramount**: JSON has broader library support across
  languages and environments.
- **Message size is not a concern**: For stdio-based local connections, the
  JSON binding's overhead is negligible.

---

## 5. Content Negotiation

When a host and server support multiple serialization formats, they MUST
negotiate the format during the initialize handshake.

### 5.1 Initialize Handshake Extension

Content negotiation extends the `initialize` message (Protocol Semantics,
Section 4.1.1) with an additional field in `InitializeParams`.

#### 5.1.1 Extended InitializeParams

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `serialization_format` | string | No | `"json"` | Preferred serialization format. |

> **Note:** The `initialize` message itself MUST always be serialized using
> the JSON binding, regardless of the negotiated format. The negotiated format
> applies to all **subsequent** messages after the initialize handshake
> completes.

#### 5.1.2 Extended InitializeResult

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `serialization_format` | string | No | `"json"` | Accepted serialization format for this connection. |

### 5.2 Format Identifiers

| Identifier | Binding | Section |
|------------|---------|---------|
| `"json"` | JSON-RPC 2.0 (default, always supported) | Section 2 |
| `"protobuf"` | Protocol Buffers | Section 3 |
| `"msgpack"` | MessagePack | Section 4 |

Format identifiers are case-sensitive strings. Implementations MUST NOT
accept variants (e.g., `"JSON"`, `"Protobuf"`, `"MessagePack"`).

### 5.3 Negotiation Rules

The negotiation follows a propose-accept model:

1. **Host proposes**: The host includes `serialization_format` in the
   `initialize` request, indicating its preferred format.

2. **Server accepts or counter-proposes**: The server responds with
   `serialization_format` in the `initialize` result:
   - If the server supports the requested format, it SHOULD echo the same
     identifier back.
   - If the server does not support the requested format, it MUST respond
     with `"json"` (the universal fallback).

3. **Format takes effect**: After the initialize response is received and
   validated, both sides MUST switch to the negotiated format for all
   subsequent messages on this connection.

4. **JSON fallback**: Both sides MUST support `"json"`. If the host does not
   include `serialization_format` (or includes an unrecognized identifier),
   the connection MUST use `"json"`.

5. **No renegotiation**: The serialization format is fixed for the lifetime
   of the connection. To change formats, the host MUST disconnect and
   reconnect.

#### 5.3.1 Negotiation Examples

**Successful protobuf negotiation:**

```json
// Host proposes protobuf
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1,
  "params": {
    "protocol_version": "1.0",
    "serialization_format": "protobuf",
    "host_capabilities": { ... }
  }
}

// Server accepts
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "server_name": "my-plugin",
    "serialization_format": "protobuf",
    "server_capabilities": { ... }
  }
}

// All subsequent messages use protobuf encoding
```

**Fallback to JSON:**

```json
// Host proposes msgpack
{
  "jsonrpc": "2.0",
  "method": "initialize",
  "id": 1,
  "params": {
    "protocol_version": "1.0",
    "serialization_format": "msgpack",
    "host_capabilities": { ... }
  }
}

// Server does not support msgpack, falls back to JSON
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "server_name": "my-plugin",
    "serialization_format": "json",
    "server_capabilities": { ... }
  }
}

// All subsequent messages use JSON encoding
```

---

## 6. Conformance Requirements

### 6.1 JSON Binding (REQUIRED)

All CAP implementations MUST support the JSON binding as defined in Section 2.
This includes:

- JSON-RPC 2.0 envelope format (Section 2.1).
- `snake_case` field naming (Section 2.2).
- Type mapping (Section 2.3).
- MISSING sentinel semantics (Section 2.4).
- DateTime as ISO 8601, binary as base64 (Section 2.5).
- NDJSON framing over stream transports (Section 2.6).
- UTF-8 encoding (Section 2.7).

A conformant JSON implementation MUST correctly handle:
- All 12 CAP message types (Protocol Semantics, Appendix A).
- All CAP error codes (Protocol Semantics, Section 8.2).
- The absent-vs-null distinction for MISSING fields (Section 2.4).
- Nested dataclass serialization (Section 2.3.2).

### 6.2 Protobuf Binding (OPTIONAL)

Implementations MAY support the protobuf binding as defined in Section 3.
A conformant protobuf implementation MUST:

- Map all 12 CAP message types to protobuf messages (Section 3.1).
- Use `google.protobuf.Struct` for dynamic dictionaries (Section 3.2.1).
- Use `google.protobuf.Value` for `Any`-typed fields (Section 3.2.2).
- Preserve MISSING semantics via `optional` field presence (Section 3.3).
- Support the JSON binding as a fallback.

### 6.3 MessagePack Binding (OPTIONAL)

Implementations MAY support the MessagePack binding as defined in Section 4.
A conformant MessagePack implementation MUST:

- Preserve the JSON binding's logical structure (Section 4.1).
- Use string keys in all maps (Section 4.1.2).
- Preserve MISSING semantics via absent keys (Section 4.1.3).
- Use length-prefixed framing over stream transports (Section 4.4).
- Support the JSON binding as a fallback.

### 6.4 Content Negotiation (CONDITIONAL)

Implementations that support more than one serialization format MUST
implement the content negotiation extension defined in Section 5.
Implementations that support only the JSON binding MAY omit content
negotiation entirely (the JSON binding is the implicit default).

### 6.5 Conformance Matrix

| Requirement | JSON-Only Implementation | Multi-Format Implementation |
|-------------|--------------------------|----------------------------|
| JSON binding (Section 2) | MUST | MUST |
| Protobuf binding (Section 3) | -- | MAY |
| MessagePack binding (Section 4) | -- | MAY |
| Content negotiation (Section 5) | MAY | MUST |
| MISSING semantics (Section 2.4) | MUST | MUST (all bindings) |
| Error codes (Protocol Semantics, Section 8) | MUST | MUST |

---

## Appendix A: Complete JSON Wire Examples

This appendix provides complete JSON wire examples for all 12 CAP message
types, suitable for use as test vectors.

### A.1 initialize

```json
// Request (host -> server)
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocol_version":"1.0","host_capabilities":{"create_node":true,"invoke":true,"roots":true,"notifications":true},"roots":[{"uri":"file:///home/user/project","name":"cwd","description":"Current working directory"}],"session_id":"sess_abc123","cwd":"/home/user/project"}}

// Response (server -> host)
{"jsonrpc":"2.0","id":1,"result":{"server_name":"my-plugin","server_version":"0.1.0","protocol_version":"1.0","server_capabilities":{"sync":true,"immediate_call":false,"push_dirty":true,"push_notifications":true},"node_types":[{"node_type":"custom_view","description":"A custom file view with annotations","constructor":{"positional":[],"variadic":null,"named":[]},"properties":[],"methods":[]}]}}
```

### A.2 shutdown

```json
// Notification (host -> server)
{"jsonrpc":"2.0","method":"shutdown"}
```

### A.3 node/create

```json
// Request
{"jsonrpc":"2.0","method":"node/create","id":2,"params":{"node_type":"shell","args":["pytest","-v"],"kwargs":{"timeout":120},"node_id":"shell_a1b2c3d4"}}

// Response
{"jsonrpc":"2.0","id":2,"result":{"node_id":"shell_a1b2c3d4","state":{"command":"pytest -v","exit_code":null,"is_complete":false},"renders":{"header":"[SHELL] pytest -v [RUNNING]","content":"","detail":""},"tokens":{"collapsed":12,"summary":0,"detail":0},"digest":{"type":"shell","status":"running"}}}
```

### A.4 node/sync

```json
// Request
{"jsonrpc":"2.0","method":"node/sync","id":5,"params":{"node_id":"shell_a1b2c3d4","calls":[{"method":"cancel","args":[],"kwargs":{}}],"cwd":"/home/user/project"}}

// Response
{"jsonrpc":"2.0","id":5,"result":{"state":{"command":"pytest -v","exit_code":-15,"is_complete":true},"renders":{"header":"[SHELL] pytest -v [CANCELLED]","content":"Process cancelled by user.","detail":"Exit code: -15\nRuntime: 3.2s"},"tokens":{"collapsed":12,"summary":8,"detail":20},"digest":{"type":"shell","status":"cancelled","exit_code":-15},"notifications":[{"description":"Shell cancelled: pytest -v","level":"wake"}]}}
```

### A.5 node/call

```json
// Request
{"jsonrpc":"2.0","method":"node/call","id":8,"params":{"node_id":"custom_e5f6g7h8","method":"get_summary","args":[],"kwargs":{"max_tokens":200}}}

// Response
{"jsonrpc":"2.0","id":8,"result":{"result":"Authentication module: 3 endpoints, 2 middleware functions.","state_changed":false}}
```

### A.6 node/destroy

```json
// Request
{"jsonrpc":"2.0","method":"node/destroy","id":10,"params":{"node_id":"shell_a1b2c3d4"}}

// Response
{"jsonrpc":"2.0","id":10,"result":{}}
```

### A.7 node/serialize

```json
// Request
{"jsonrpc":"2.0","method":"node/serialize","id":11,"params":{"node_id":"shell_a1b2c3d4"}}

// Response
{"jsonrpc":"2.0","id":11,"result":{"data":{"node_type":"shell","command":"pytest -v","exit_code":0,"is_complete":true,"output":"3 passed in 1.2s"}}}
```

### A.8 node/dirty

```json
// Notification (server -> host)
{"jsonrpc":"2.0","method":"node/dirty","params":{"node_id":"shell_a1b2c3d4"}}
```

### A.9 node/notification

```json
// Notification (server -> host)
{"jsonrpc":"2.0","method":"node/notification","params":{"node_id":"shell_a1b2c3d4","description":"Shell completed: pytest -v (exit 0)","level":"wake"}}
```

### A.10 host/create_node

```json
// Request (server -> host)
{"jsonrpc":"2.0","method":"host/create_node","id":100,"params":{"node_type":"artifact","args":["test output"],"kwargs":{"artifact_type":"output","language":"text"},"parent_id":"shell_a1b2c3d4"}}

// Response (host -> server)
{"jsonrpc":"2.0","id":100,"result":{"node_id":"artifact_x9y8z7w6"}}
```

### A.11 host/invoke

```json
// Request (server -> host)
{"jsonrpc":"2.0","method":"host/invoke","id":101,"params":{"node_id":"text_m1n2o3p4","method":"set_content","args":["Updated file content"],"kwargs":{}}}

// Response (host -> server)
{"jsonrpc":"2.0","id":101,"result":{"result":null}}
```

### A.12 host/query_roots

```json
// Request (server -> host)
{"jsonrpc":"2.0","method":"host/query_roots","id":102,"params":{}}

// Response (host -> server)
{"jsonrpc":"2.0","id":102,"result":{"roots":[{"uri":"file:///home/user/project","name":"cwd","description":"Current working directory"}]}}
```

### A.13 host/resolve_root

```json
// Request (server -> host)
{"jsonrpc":"2.0","method":"host/resolve_root","id":103,"params":{"root_uri":"file:///home/user/project","normalized_path":"src/main.py"}}

// Response (host -> server)
{"jsonrpc":"2.0","id":103,"result":{"real_path":"/home/user/project/src/main.py","exists":true}}
```

## Appendix B: Error Response Examples

### B.1 Standard JSON-RPC Errors

```json
// PARSE_ERROR (-32700)
{"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error: unexpected token at position 42"},"id":null}

// INVALID_REQUEST (-32600)
{"jsonrpc":"2.0","error":{"code":-32600,"message":"Invalid request: missing 'method' field"},"id":3}

// METHOD_NOT_FOUND (-32601)
{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found: node/unknown"},"id":4}

// INVALID_PARAMS (-32602)
{"jsonrpc":"2.0","error":{"code":-32602,"message":"Invalid params: 'node_type' is required"},"id":5}

// INTERNAL_ERROR (-32603)
{"jsonrpc":"2.0","error":{"code":-32603,"message":"Internal error: out of memory"},"id":6}
```

### B.2 CAP-Specific Errors

```json
// NODE_NOT_FOUND (-32000)
{"jsonrpc":"2.0","error":{"code":-32000,"message":"Node not found: shell_a1b2c3d4","data":{"node_id":"shell_a1b2c3d4"}},"id":7}

// NODE_TYPE_UNKNOWN (-32001)
{"jsonrpc":"2.0","error":{"code":-32001,"message":"Unknown node type: foobar","data":{"node_type":"foobar"}},"id":8}

// METHOD_NOT_AVAILABLE (-32002)
{"jsonrpc":"2.0","error":{"code":-32002,"message":"Method 'fly' is not available on node type 'shell'","data":{"method":"fly","node_type":"shell"}},"id":9}

// ROOT_NOT_FOUND (-32003)
{"jsonrpc":"2.0","error":{"code":-32003,"message":"Root not found: file:///nonexistent","data":{"root_uri":"file:///nonexistent"}},"id":10}

// PERMISSION_DENIED (-32004)
{"jsonrpc":"2.0","error":{"code":-32004,"message":"Permission denied: host does not support 'create_node' capability"},"id":11}

// NODE_CREATE_FAILED (-32005)
{"jsonrpc":"2.0","error":{"code":-32005,"message":"Failed to create node: invalid timeout value","data":{"node_type":"shell","reason":"timeout must be positive"}},"id":12}
```
