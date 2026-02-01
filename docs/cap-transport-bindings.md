# CAP Transport Bindings

> Transport layer requirements and concrete transport bindings for the
> Context Application Protocol, version 1.0.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

This document is a companion to the
[CAP Protocol Semantics](cap-protocol-semantics.md) specification. It
defines the abstract requirements that every transport implementation must
satisfy and specifies four concrete transport bindings: stdio, gRPC,
WebSocket, and TCP.

---

## 1. Overview

A transport binding defines how CAP messages are framed and delivered over a
specific communication channel. The protocol semantics specification
(Section 1.3) declares CAP to be transport-independent: message types,
sequencing rules, capabilities, and error codes are defined without
reference to any particular framing or encoding.

This document bridges the gap between abstract semantics and concrete
communication by:

1. Establishing the invariants every transport MUST uphold (Section 2).
2. Defining the reference stdio binding in full detail (Section 3).
3. Specifying three additional bindings -- gRPC (Section 4), WebSocket
   (Section 5), and TCP (Section 6) -- for deployments that require
   network access, browser compatibility, or higher throughput.

Each binding maps CAP's bidirectional request/response/notification model
onto the primitives of the underlying channel. Implementations MAY support
one or more bindings; the stdio binding is REQUIRED.

---

## 2. Abstract Transport Requirements

Every conforming transport binding MUST satisfy the requirements in this
section. These requirements are transport-independent and apply equally to
all bindings defined in this document and to any future bindings.

### 2.1 Bidirectional Messaging

A CAP transport carries messages in two directions:

- **Host to Server**: requests (with `id`) and notifications (no `id`).
- **Server to Host**: responses (matching `id`), push notifications
  (`node/dirty`, `node/notification`), and reverse requests (Host API:
  `host/create_node`, `host/invoke`, `host/query_roots`,
  `host/resolve_root`).

The transport MUST support both directions simultaneously. Neither side may
be restricted to send-only or receive-only operation after the connection is
established.

### 2.2 Request/Response Correlation

- The transport MUST preserve the `id` field of every request message and
  deliver it unmodified to the receiver. The receiver's response MUST carry
  the same `id`.
- The transport MUST NOT reorder a response relative to its corresponding
  request on a single logical connection. That is, a response MUST NOT
  arrive before its request was sent.
- The transport MAY deliver independent messages (requests with different
  `id` values, or notifications) concurrently or out of order relative to
  each other, provided that per-sender ordering (Section 2.3) is preserved.

### 2.3 Ordered Delivery

- Messages from a single sender MUST be delivered to the receiver in the
  order they were written.
- Cross-sender ordering is NOT guaranteed. The host and server are
  independent senders; their messages may interleave arbitrarily at the
  transport level.
- Within a `node/sync` response, the `calls` array ordering MUST be
  preserved from request to execution. The transport MUST NOT reorder
  elements within a single message payload.

### 2.4 Connection Lifecycle

Every transport MUST expose the following lifecycle operations:

| Operation | Semantics |
|-----------|-----------|
| `start()` | Establish the transport connection and make it ready for messaging. After `start()` returns successfully, the transport MUST accept `send_request()` and `send_notification()` calls. |
| `stop()` | Close the transport connection and release all resources (pipes, sockets, file descriptors, threads). After `stop()` returns, the transport MUST reject all further send operations. `stop()` MUST be idempotent. |
| `is_running` | Observable boolean property. `true` after a successful `start()` and before `stop()` completes or the connection is lost. |

The transport MUST transition through the following states, consistent with
the connection state machine defined in the protocol semantics (Section 3.1):

```
DISCONNECTED  --- start() --->  CONNECTING
CONNECTING    --- success --->  CONNECTED
CONNECTING    --- failure --->  ERROR
CONNECTED     --- stop() ---->  DISCONNECTED
CONNECTED     --- lost ------>  ERROR
ERROR         --- stop() ---->  DISCONNECTED
```

The transport MUST NOT allow message sending in any state other than
`CONNECTED`.

### 2.5 Error Propagation

- Transport-level errors (connection lost, write failure, read timeout)
  MUST be surfaced to the caller. For pending requests, the transport MUST
  resolve their futures with an error.
- The transport MUST distinguish between **transport errors** (the channel
  failed) and **protocol errors** (a valid JSON-RPC error response from the
  peer). Transport errors indicate infrastructure failure; protocol errors
  indicate application-level rejection.
- When a transport error occurs, the transport SHOULD transition to the
  `ERROR` state and fail all outstanding pending requests.

### 2.6 Backpressure

- The transport SHOULD apply backpressure when the receiver cannot consume
  messages fast enough. The specific mechanism is binding-dependent (e.g.,
  TCP flow control, async write buffering with limits).
- The transport MUST NOT silently drop messages. If a message cannot be
  delivered due to buffer exhaustion, the transport MUST surface an error
  to the sender.

### 2.7 Serialization

- The default serialization format for all bindings is JSON, consistent
  with the JSON-RPC 2.0 wire format used in the protocol semantics
  specification.
- Alternative serialization formats (e.g., MessagePack, Protocol Buffers)
  MAY be used by specific bindings where noted.
- When JSON is used, all messages MUST be encoded as UTF-8.
- Numeric values MUST NOT lose precision during serialization. In
  particular, JSON integers outside the safe integer range (beyond 2^53)
  SHOULD be avoided in message `id` fields and numeric parameters.

### 2.8 Message Framing

Each binding defines its own framing mechanism. Regardless of the framing
method, the transport MUST guarantee:

- **Atomicity**: A complete CAP message is delivered as a single unit. The
  receiver MUST NOT observe a partial message.
- **Boundary detection**: The receiver MUST be able to determine where one
  message ends and the next begins without ambiguity.

---

## 3. Stdio Binding (Reference)

The stdio binding is the reference transport for CAP. It communicates via
the standard input and output streams of a subprocess. This binding is
REQUIRED for all CAP implementations.

The reference implementation is `PluginTransport` (aliased as
`StdioTransport`) in `src/activecontext/plugins/transport.py`.

### 3.1 Framing

- Each message is a single JSON object serialized as a compact (no
  extraneous whitespace) UTF-8 string, followed by a single newline
  character (`\n`, U+000A). This is newline-delimited JSON (NDJSON).
- There is no `Content-Length` header or other envelope. This differs from
  the Language Server Protocol (LSP), which uses HTTP-style headers.
- Each line MUST contain exactly one complete JSON object. Multi-line
  pretty-printed JSON is not permitted.
- Empty lines (bare `\n`) MUST be silently ignored by the receiver.
- The JSON object MUST conform to the JSON-RPC 2.0 message format as
  defined in the protocol semantics specification.

**Wire example (host sends `initialize` request):**

```
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocol_version":"1.0","host_capabilities":{"create_node":true,"invoke":true,"roots":true,"notifications":true},"roots":[],"session_id":"sess_1","cwd":"/home/user/project"}}\n
```

**Wire example (server responds):**

```
{"jsonrpc":"2.0","id":1,"result":{"server_name":"my-plugin","server_version":"0.1.0","protocol_version":"1.0","node_types":[],"server_capabilities":{"sync":true,"immediate_call":false,"push_dirty":true,"push_notifications":true}}}\n
```

### 3.2 Channels

The stdio binding uses three file descriptors:

| Channel | Direction | Purpose |
|---------|-----------|---------|
| `stdin` (fd 0) | Host writes, Server reads | Protocol messages (host to server). |
| `stdout` (fd 1) | Server writes, Host reads | Protocol messages (server to host). |
| `stderr` (fd 2) | Server writes, Host reads (optional) | Diagnostic logging. MUST NOT contain protocol messages. |

- The host writes JSON-RPC messages to the server's `stdin`.
- The host reads JSON-RPC messages from the server's `stdout`.
- The server's `stderr` is reserved for human-readable diagnostic output.
  The host MAY capture stderr for logging but MUST NOT parse it as protocol
  traffic.

### 3.3 Connection Establishment

1. The host spawns the plugin server as a child subprocess with pipes
   attached to `stdin`, `stdout`, and `stderr`.
2. The connection is considered ready when the process has started and all
   pipes are open. There is no transport-level handshake; the CAP
   `initialize` message (defined in the protocol semantics, Section 3.2)
   serves as the application-level handshake.
3. The host MUST send the `initialize` request as the first message after
   the pipes are open.
4. The server MUST begin reading from `stdin` immediately upon startup. The
   server MUST NOT write any output to `stdout` before receiving the
   `initialize` request (except that it MAY write diagnostic output to
   `stderr` at any time).

For the stdio binding, only the **host-managed process** connection model
(protocol semantics, Section 3.1) is applicable. The host always spawns
and owns the server process.

### 3.4 Connection Termination

Graceful shutdown follows the protocol semantics (Section 3.4):

1. The host sends the `shutdown` notification via `stdin`.
2. The host closes the `stdin` pipe. This signals EOF to the server's read
   loop.
3. The host waits for the server process to exit. The RECOMMENDED timeout
   is 5 seconds.
4. If the server does not exit within the timeout, the host SHOULD
   terminate the process (SIGTERM or platform equivalent), wait briefly,
   then kill forcefully (SIGKILL or platform equivalent) if still running.
5. The host closes the `stdout` and `stderr` pipes after the process exits.

The server, upon receiving the `shutdown` notification or detecting EOF on
`stdin`, MUST:

1. Complete any in-flight responses.
2. Stop reading from `stdin`.
3. Close `stdout` (triggering EOF on the host's read side).
4. Exit with code 0 for clean shutdown.

### 3.5 Error Detection

| Condition | Detection | Semantics |
|-----------|-----------|-----------|
| EOF on `stdout` | `readline()` returns empty bytes | Server disconnected or exited. Connection lost. |
| Write to closed `stdin` | `BrokenPipeError` or equivalent | Server's `stdin` is closed. Connection lost. |
| Process exit (non-zero) | `returncode != 0` after wait | Abnormal termination. Treat as unclean shutdown. |
| Process exit (zero) | `returncode == 0` after wait | Clean exit. Expected during shutdown. |
| Invalid JSON on `stdout` | `json.JSONDecodeError` | Malformed message. Log and skip. SHOULD NOT terminate the connection for a single parse error. |

When the connection is lost (EOF or broken pipe), the transport MUST:

1. Fail all pending request futures with a `PluginTransportError`.
2. Clear the pending request map.
3. Transition `is_running` to `false`.

### 3.6 Threading Model

The reference implementation uses the following concurrency model:

- **Reader task**: A single `asyncio.Task` continuously reads lines from
  `stdout` and dispatches incoming messages (responses to pending futures,
  notifications to callbacks, server requests to the host API handler).
- **Writer**: Writes to `stdin` are serialized under an `asyncio.Lock` to
  prevent interleaved partial writes from concurrent coroutines.
- **Dispatch**: Incoming messages are classified by the presence of `id`
  and/or `method` fields:
  - Response: has `id`, no `method`. Resolves the matching pending future.
  - Notification: has `method`, no `id`. Invokes the notification callback.
  - Server request: has both `method` and `id`. Routes to the host API
    handler via the notification callback with the `_request_id` attached.

### 3.7 Limitations

- **Single host**: One host per server subprocess (1:1 relationship). No
  multiplexing of multiple hosts over a single stdio pair.
- **Local only**: Requires the ability to spawn local subprocesses. Not
  suitable for remote or network-accessible servers.
- **No reconnection**: If the connection is lost, a new subprocess must be
  spawned. There is no reconnection mechanism.
- **Buffering**: Large messages may cause buffering delays. The server
  SHOULD flush `stdout` after each message to minimize latency.

---

## 4. gRPC Binding

The gRPC binding uses HTTP/2 with Protocol Buffers (or JSON) serialization
for high-performance communication. This binding is OPTIONAL.

### 4.1 Framing

- Standard gRPC framing: length-prefixed Protocol Buffers messages over
  HTTP/2 streams.
- Alternatively, gRPC-JSON (using `application/grpc+json` content type)
  MAY be used for interoperability with languages lacking protobuf support.
- The gRPC framing layer handles message boundaries, so no additional
  newline or length prefix is needed at the CAP level.

### 4.2 Service Definition

Two service design approaches are defined. Implementations MAY support
either or both.

#### Approach A: Bidirectional Stream

A single bidirectional streaming RPC carries all CAP messages:

```protobuf
syntax = "proto3";

package cap.v1;

// A generic CAP message envelope.
message CAPMessage {
  // JSON-RPC 2.0 message as a JSON string.
  // This preserves full compatibility with the JSON-RPC wire format.
  string json_payload = 1;
}

// Single bidirectional stream for all message types.
service CAPPlugin {
  rpc Session(stream CAPMessage) returns (stream CAPMessage);
}
```

This approach most closely mirrors the stdio model: a single bidirectional
channel carrying interleaved requests, responses, and notifications. It is
simpler to implement and maintains the same ordering guarantees as stdio.

#### Approach B: Typed RPCs

Individual RPCs per message type, with a notification stream for
server-to-host push:

```protobuf
syntax = "proto3";

package cap.v1;

// Typed request/response messages (one per CAP message type).
// Fields correspond to the JSON-RPC params/result schemas
// defined in the protocol semantics specification.

message InitializeRequest { ... }
message InitializeResponse { ... }
message NodeCreateRequest { ... }
message NodeCreateResponse { ... }
message NodeSyncRequest { ... }
message NodeSyncResponse { ... }
message NodeCallRequest { ... }
message NodeCallResponse { ... }
message NodeDestroyRequest { ... }
message NodeDestroyResponse { ... }
message NodeSerializeRequest { ... }
message NodeSerializeResponse { ... }
message ShutdownNotification { ... }

// Server push messages
message ServerPush {
  oneof push {
    NodeDirtyNotification dirty = 1;
    NodeNotification notification = 2;
  }
}

// Host API reverse requests
message HostCreateNodeRequest { ... }
message HostCreateNodeResponse { ... }
message HostInvokeRequest { ... }
message HostInvokeResponse { ... }
message HostQueryRootsRequest { ... }
message HostQueryRootsResponse { ... }
message HostResolveRootRequest { ... }
message HostResolveRootResponse { ... }

service CAPPlugin {
  // Lifecycle
  rpc Initialize(InitializeRequest) returns (InitializeResponse);
  rpc Shutdown(ShutdownNotification) returns (google.protobuf.Empty);

  // Node management
  rpc NodeCreate(NodeCreateRequest) returns (NodeCreateResponse);
  rpc NodeSync(NodeSyncRequest) returns (NodeSyncResponse);
  rpc NodeCall(NodeCallRequest) returns (NodeCallResponse);
  rpc NodeDestroy(NodeDestroyRequest) returns (NodeDestroyResponse);
  rpc NodeSerialize(NodeSerializeRequest) returns (NodeSerializeResponse);

  // Server push (server-streaming)
  rpc Notifications(google.protobuf.Empty) returns (stream ServerPush);

  // Host API (reverse direction -- requires bidirectional streaming
  // or a separate callback channel)
  rpc HostCreateNode(HostCreateNodeRequest) returns (HostCreateNodeResponse);
  rpc HostInvoke(HostInvokeRequest) returns (HostInvokeResponse);
  rpc HostQueryRoots(HostQueryRootsRequest) returns (HostQueryRootsResponse);
  rpc HostResolveRoot(HostResolveRootRequest) returns (HostResolveRootResponse);
}
```

This approach provides better observability (per-method metrics), load
balancing (individual RPCs can be routed independently), and type safety
(compile-time schema validation). However, it requires handling the Host
API reverse direction separately, either through a callback channel or a
second gRPC service on the host side.

### 4.3 Connection Establishment

1. The host creates a gRPC channel to the server's address and port.
2. For host-managed processes, the host spawns the server and waits for the
   gRPC port to become available (e.g., via a readiness file, health check
   endpoint, or a known port convention).
3. For external processes, the host connects to a pre-configured address.
4. TLS is RECOMMENDED for non-localhost connections. For localhost, plaintext
   gRPC is acceptable.
5. After the gRPC channel is established, the host sends the `Initialize`
   RPC (or the `initialize` message over the bidirectional stream).

### 4.4 Connection Termination

1. The host sends the `Shutdown` RPC (or `shutdown` message over the
   stream).
2. The gRPC channel is closed gracefully (HTTP/2 GOAWAY frame).
3. For host-managed processes, the host waits for the process to exit, with
   the same timeout and force-kill behavior as the stdio binding.
4. For external processes, the host closes the channel but does not
   terminate the server.

### 4.5 Multiplexing

- gRPC supports multiple concurrent RPCs on a single HTTP/2 connection.
- Multiple independent hosts MAY connect to a single external server, each
  with its own gRPC channel and CAP session.
- With Approach B (typed RPCs), multiple `node/sync` requests for different
  nodes MAY be in flight simultaneously on the same connection. The
  transport MUST still preserve per-sender ordering for messages from a
  single host.

### 4.6 Error Mapping

gRPC status codes map to CAP transport errors as follows:

| gRPC Status | CAP Interpretation |
|-------------|--------------------|
| `OK` | Success. |
| `UNAVAILABLE` | Transport error: server unreachable. |
| `DEADLINE_EXCEEDED` | Timeout: equivalent to `asyncio.TimeoutError`. |
| `CANCELLED` | Request cancelled (e.g., during shutdown). |
| `INTERNAL` | Server internal error. Map to JSON-RPC `INTERNAL_ERROR`. |
| `UNIMPLEMENTED` | Method not found. Map to JSON-RPC `METHOD_NOT_FOUND`. |
| `INVALID_ARGUMENT` | Invalid params. Map to JSON-RPC `INVALID_PARAMS`. |
| `PERMISSION_DENIED` | Capability not declared. Map to `PERMISSION_DENIED`. |

Application-level CAP errors (the JSON-RPC error object) are carried in the
response payload, not in gRPC status codes. The gRPC status SHOULD be `OK`
even when the CAP response contains a JSON-RPC error. gRPC status codes are
reserved for transport-level failures.

### 4.7 Host API (Reverse Direction)

The Host API (server-to-host requests) requires the server to send requests
that the host responds to. In gRPC, this can be handled by:

1. **Bidirectional stream** (Approach A): Both directions share the same
   stream. Server requests and host responses are interleaved naturally.
2. **Callback service**: The host runs a gRPC server of its own that the
   plugin server connects to for Host API calls. The host's address is
   communicated during initialization (e.g., via an extension field in
   `InitializeParams`).
3. **Embedded in notification stream**: Host API requests are sent over the
   `Notifications` server-streaming RPC, and responses are sent via a
   dedicated client-streaming RPC.

Approach A is RECOMMENDED for simplicity. Approach B (callback service) is
RECOMMENDED when the host and server are on separate machines.

### 4.8 Use Cases

- High-throughput local deployments requiring native protobuf performance.
- Polyglot environments where gRPC provides cross-language interoperability.
- Microservice architectures where plugin servers run as independent services.
- Deployments requiring mutual TLS authentication.

---

## 5. WebSocket Binding

The WebSocket binding uses the WebSocket protocol ([RFC 6455](https://www.rfc-editor.org/rfc/rfc6455))
for communication. This binding is OPTIONAL.

### 5.1 Framing

- Each CAP message is one WebSocket **text frame** containing a single JSON
  object (the JSON-RPC 2.0 message).
- No additional framing (newlines, length prefixes) is needed. The
  WebSocket protocol handles message boundaries natively.
- Binary frames MAY be used for alternative serialization formats (e.g.,
  MessagePack or Protocol Buffers). If binary frames are used, the
  subprotocol negotiation (Section 5.2) MUST indicate the serialization
  format.
- The receiver MUST treat each WebSocket message as exactly one CAP
  message. Concatenating multiple CAP messages into a single WebSocket
  frame is NOT permitted.

**Wire example (host sends `node/create` over WebSocket text frame):**

```json
{"jsonrpc":"2.0","method":"node/create","id":2,"params":{"node_type":"shell","args":["pytest","-v"],"kwargs":{"timeout":120},"node_id":"shell_a1b2c3d4"}}
```

### 5.2 Connection Establishment

1. The host initiates a WebSocket handshake (HTTP Upgrade) to the server's
   endpoint.
2. The default path is `/cap`. Implementations MAY use a configurable path.
3. The host SHOULD request the `cap.v1` subprotocol during the handshake:

   ```
   GET /cap HTTP/1.1
   Host: localhost:8080
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Protocol: cap.v1
   Sec-WebSocket-Version: 13
   Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
   ```

4. The server MUST accept the `cap.v1` subprotocol if it supports this
   binding. If the server does not recognize the subprotocol, it MAY accept
   the connection without a subprotocol, but the host SHOULD log a warning.
5. After the WebSocket connection is established, the host sends the
   `initialize` message as the first WebSocket text frame.
6. TLS (`wss://`) is RECOMMENDED for non-localhost connections.

For the WebSocket binding, both connection models (host-managed and external
process) are applicable:

- **Host-managed**: The host spawns the server, which opens a WebSocket
  listener on a port. The host connects after the port is available.
- **External**: The host connects to a pre-configured WebSocket URL.

### 5.3 Connection Termination

1. The host sends the `shutdown` notification as a WebSocket text frame.
2. The host sends a WebSocket Close frame with status code `1000` (Normal
   Closure).
3. The server completes any in-flight responses, then sends its own Close
   frame.
4. Both sides close the underlying TCP connection.

For abnormal termination:

| Condition | WebSocket Close Code | Semantics |
|-----------|---------------------|-----------|
| Normal shutdown | `1000` (Normal Closure) | Clean shutdown after `shutdown` notification. |
| Host going away | `1001` (Going Away) | Host is shutting down; no `shutdown` was sent. |
| Protocol error | `1002` (Protocol Error) | Received invalid WebSocket data. |
| Unexpected condition | `1011` (Unexpected Condition) | Server encountered an error. |

### 5.4 Multiplexing

- One CAP session per WebSocket connection. There is no session
  multiplexing within a single WebSocket connection.
- Multiple hosts connect to the same server via separate WebSocket
  connections. The server distinguishes sessions by connection identity.
- If multiplexing is needed, use separate WebSocket connections (one per
  session) or upgrade to the gRPC binding.

### 5.5 Keepalive

- WebSocket Ping/Pong frames SHOULD be used for connection liveness
  detection.
- The host SHOULD send Ping frames at a regular interval (RECOMMENDED: 30
  seconds). If no Pong is received within 10 seconds, the host SHOULD
  treat the connection as lost.
- The server MUST respond to Ping frames with Pong frames (this is
  required by RFC 6455).

### 5.6 Host API (Reverse Direction)

WebSocket is naturally bidirectional: both sides can send messages at any
time. The Host API works identically to the stdio binding -- the server
sends a JSON-RPC request over the WebSocket, and the host sends a JSON-RPC
response back over the same connection.

### 5.7 Use Cases

- Browser-based plugin servers (e.g., WASM plugins running in a browser
  tab).
- Network-accessible servers in environments where HTTP/WebSocket is the
  only allowed protocol (corporate firewalls, cloud functions).
- Lightweight remote deployments where gRPC is too heavy.
- Development and debugging tools that benefit from browser DevTools
  WebSocket inspection.

---

## 6. TCP Binding

The TCP binding uses raw TCP sockets with length-prefixed framing. This
binding is OPTIONAL.

### 6.1 Framing

- Each message is preceded by a **4-byte big-endian unsigned integer**
  indicating the length of the message payload in bytes.
- The payload is a complete JSON object encoded as UTF-8.
- No trailing delimiter (no newline after the payload).

**Wire format:**

```
+-------------------+-----------------------------+
| Length (4 bytes)   | JSON payload (Length bytes)  |
| big-endian uint32 | UTF-8 encoded               |
+-------------------+-----------------------------+
```

**Wire example (host sends `initialize` request):**

```
Bytes 0-3:   00 00 01 2A   (298 bytes payload)
Bytes 4-301: {"jsonrpc":"2.0","method":"initialize","id":1,"params":{...}}
```

- The maximum message size SHOULD be 16 MiB (16,777,216 bytes). Messages
  exceeding this limit SHOULD be rejected by the receiver with a transport
  error.
- A length of 0 is invalid. The receiver MUST treat a zero-length message
  as a framing error.

### 6.2 Connection Establishment

1. The server listens on a TCP port (configurable; no default port is
   assigned).
2. The host connects via standard TCP `connect()` to the server's address
   and port.
3. After the TCP connection is established, the host sends the `initialize`
   message as the first length-prefixed frame.

For the TCP binding, both connection models apply:

- **Host-managed**: The host spawns the server, which opens a TCP listener.
  The host connects after the port is available.
- **External**: The host connects to a pre-configured host:port.

TLS is NOT applied by default. For secure communication, the implementation
SHOULD wrap the TCP socket in a TLS layer (e.g., `ssl.wrap_socket` in
Python, `rustls` in Rust). When TLS is used, the connection is established
after the TLS handshake completes, before the `initialize` message is sent.

### 6.3 Connection Termination

1. The host sends the `shutdown` notification as a length-prefixed frame.
2. The host sends a TCP FIN (half-close) by shutting down the write side of
   the socket.
3. The server completes any in-flight responses, sends them, then closes
   its end of the connection.
4. Both sides close the socket.

For abnormal termination:

| Condition | TCP Behavior | Semantics |
|-----------|-------------|-----------|
| Normal shutdown | FIN after `shutdown` notification | Clean disconnect. |
| Abnormal termination | RST (connection reset) | Unclean disconnect; treat as connection lost. |
| Read timeout | No data within timeout | Connection may be dead; close and reconnect. |

### 6.4 Keepalive

- TCP keepalive (`SO_KEEPALIVE`) is RECOMMENDED to detect dead connections.
  Implementations SHOULD set `TCP_KEEPIDLE` (or platform equivalent) to a
  reasonable value (RECOMMENDED: 60 seconds).
- Application-level keepalive via periodic ping messages is OPTIONAL. If
  implemented, it SHOULD use a JSON-RPC notification with a reserved method
  name (e.g., `"ping"`) that the receiver silently ignores.

### 6.5 Host API (Reverse Direction)

TCP is bidirectional: both sides can send length-prefixed frames at any
time. The Host API works identically to the stdio binding, with the server
sending JSON-RPC requests and the host sending JSON-RPC responses over the
same TCP connection.

### 6.6 Use Cases

- Simple network deployments without HTTP overhead.
- Embedded systems and IoT devices with minimal protocol stacks.
- High-throughput local communication where stdio pipe buffering is a
  bottleneck.
- Environments where neither gRPC nor WebSocket libraries are available.

---

## 7. Transport Selection

### 7.1 Decision Matrix

| Factor | stdio | gRPC | WebSocket | TCP |
|--------|-------|------|-----------|-----|
| **Setup complexity** | Low | Medium | Medium | Low |
| **Language support** | Any | Broad | Broad | Any |
| **Multiplexing** | No | Yes | No | No |
| **Browser support** | No | No (gRPC-Web required) | Yes | No |
| **Performance** | Good | Best | Good | Good |
| **Debugging ease** | Easy (line-based) | Hard (binary) | Medium (browser DevTools) | Medium (Wireshark) |
| **Network capable** | No | Yes | Yes | Yes |
| **TLS support** | N/A | Native | Native (WSS) | Manual |
| **Reconnection** | No (new process) | Configurable | Manual | Manual |
| **Host API support** | Native | Requires callback channel or bidi stream | Native | Native |
| **External process model** | No | Yes | Yes | Yes |
| **Process management** | Required | Optional | Optional | Optional |

### 7.2 Recommendations

- **Default**: stdio. Simplest setup, works everywhere, no network
  configuration. Best for local single-host deployments and IDE
  integrations.
- **Performance-critical**: gRPC. Native protobuf serialization, HTTP/2
  multiplexing, built-in load balancing. Best for high-throughput
  deployments with many concurrent nodes.
- **Web/browser**: WebSocket. The only binding that works in browsers
  natively. Best for web-based plugin servers and WASM plugins.
- **Embedded/minimal**: TCP. Minimal dependencies, no HTTP overhead. Best
  for constrained environments and custom network topologies.

### 7.3 Mixed Transport Deployments

A single host MAY use different transport bindings for different plugin
servers simultaneously. For example:

- Local plugins connected via stdio.
- A shared analytics plugin connected via gRPC.
- A browser-based visualization plugin connected via WebSocket.

The host's `PluginConnection` layer programs against the `CAPTransport`
protocol interface (defined in `src/activecontext/plugins/cap_transport.py`),
so the transport binding is transparent to the protocol logic.

---

## 8. Conformance

### 8.1 Required Binding

All CAP implementations MUST support the **stdio binding** (Section 3).
This ensures baseline interoperability across all environments and
languages.

### 8.2 Optional Bindings

The gRPC (Section 4), WebSocket (Section 5), and TCP (Section 6) bindings
are OPTIONAL. Implementations SHOULD document which bindings they support.

### 8.3 Transport Protocol Interface

All transport implementations MUST satisfy the `CAPTransport` protocol
interface:

```python
@runtime_checkable
class CAPTransport(Protocol):
    @property
    def is_running(self) -> bool: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send_request(self, method: str, params: Any = None, timeout: float = 30.0) -> Any: ...
    async def send_notification(self, method: str, params: Any = None) -> None: ...
    async def send_response(self, result: Any, request_id: int | str) -> None: ...
    async def send_error_response(self, code: int, message: str, request_id: int | str | None, data: Any = None) -> None: ...
```

This interface is defined in `src/activecontext/plugins/cap_transport.py`.
New transport bindings MUST implement all methods. The `PluginConnection`
layer depends only on this interface, enabling any conforming transport to
be used without changes to the protocol logic.

### 8.4 Interoperability Testing

Implementations supporting multiple bindings SHOULD verify that the same
CAP session (initialize, create, sync, destroy, shutdown) produces
identical application-level behavior regardless of which binding is used.
Transport-level details (framing, error surfaces) will differ, but the
protocol semantics MUST be preserved.

---

## Appendix A: Binding Summary

| Binding | Section | Framing | Connection Model | Multiplexing | Status |
|---------|---------|---------|------------------|--------------|--------|
| stdio | 3 | Newline-delimited JSON | Host-managed only | No | **Required** |
| gRPC | 4 | HTTP/2 + protobuf | Both | Yes | Optional |
| WebSocket | 5 | WS text frames | Both | No | Optional |
| TCP | 6 | 4-byte length prefix + JSON | Both | No | Optional |

## Appendix B: Wire Format Comparison

The following table shows how the same `node/create` message is framed
across all four bindings:

| Binding | Framing |
|---------|---------|
| **stdio** | `{"jsonrpc":"2.0","method":"node/create","id":2,"params":{...}}\n` |
| **gRPC (Approach A)** | gRPC frame containing `CAPMessage { json_payload: "{\"jsonrpc\":\"2.0\",...}" }` |
| **gRPC (Approach B)** | gRPC frame containing `NodeCreateRequest { node_type: "shell", ... }` |
| **WebSocket** | Text frame: `{"jsonrpc":"2.0","method":"node/create","id":2,"params":{...}}` |
| **TCP** | `[4-byte length]{"jsonrpc":"2.0","method":"node/create","id":2,"params":{...}}` |

## Appendix C: Security Considerations by Binding

| Binding | Encryption | Authentication | Notes |
|---------|-----------|----------------|-------|
| **stdio** | N/A (local process) | Process-level (parent owns child) | Inherits OS process isolation. |
| **gRPC** | TLS (RECOMMENDED) | mTLS, token-based, or custom | Standard gRPC auth mechanisms. |
| **WebSocket** | WSS (RECOMMENDED) | Origin checking, token headers | Vulnerable to CSWSH without origin validation. |
| **TCP** | TLS wrapper (manual) | Application-level | No built-in auth; must be layered on. |

For all network-accessible bindings (gRPC, WebSocket, TCP), implementations
SHOULD:

- Use TLS for all non-localhost connections.
- Authenticate connections before accepting the `initialize` handshake.
- Validate the `Origin` header for WebSocket connections to prevent
  cross-site WebSocket hijacking (CSWSH).
- Apply rate limiting to prevent denial-of-service attacks.
