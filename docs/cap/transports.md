# CAP Transport Bindings

> Transport layer requirements and concrete transport bindings for the Context Application Protocol v1.0.

**Specification Version:** 1.0
**Date:** 2025-01-30

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

---

## Table of Contents

1. [Overview](#1-overview)
2. [Abstract Transport Requirements](#2-abstract-transport-requirements)
3. [Stdio Binding (Required)](#3-stdio-binding-required)
4. [gRPC Binding (Optional)](#4-grpc-binding-optional)
5. [WebSocket Binding (Optional)](#5-websocket-binding-optional)
6. [TCP Binding (Optional)](#6-tcp-binding-optional)
7. [Transport Selection](#7-transport-selection)
8. [Conformance](#8-conformance)

---

## 1. Overview

A transport binding defines how CAP messages are framed and delivered over a specific communication channel. The [protocol specification](spec.md) is transport-independent: message types, sequencing rules, capabilities, and error codes are defined without reference to any particular framing or encoding.

This document:
1. Establishes invariants every transport MUST uphold (Section 2)
2. Defines the reference stdio binding in full detail (Section 3)
3. Specifies three optional bindings — gRPC, WebSocket, TCP — for network access and higher throughput

**Implementations MUST support the stdio binding.** Support for other bindings is OPTIONAL.

---

## 2. Abstract Transport Requirements

Every conforming transport binding MUST satisfy these requirements. They are transport-independent and apply to all bindings defined in this document and to any future bindings.

### 2.1 Bidirectional Messaging

A CAP transport carries messages in two directions:
- **Host to Server**: requests (with `id`) and notifications (no `id`)
- **Server to Host**: responses (matching `id`), push notifications, and reverse requests (Host API)

The transport MUST support both directions simultaneously. Neither side may be restricted to send-only or receive-only operation after connection establishment.

### 2.2 Request/Response Correlation

- The transport MUST preserve the `id` field of every request message unmodified. The receiver's response MUST carry the same `id`.
- The transport MUST NOT reorder a response relative to its request on a single logical connection.
- The transport MAY deliver independent messages (different `id` values, or notifications) concurrently or out of order, provided per-sender ordering (Section 2.3) is preserved.

### 2.3 Ordered Delivery

- Messages from a single sender MUST be delivered to the receiver in the order they were written.
- Cross-sender ordering is NOT guaranteed. Host and server are independent senders; their messages may interleave arbitrarily.
- Within a `node/sync` response, the `calls` array ordering MUST be preserved from request to execution.

### 2.4 Connection Lifecycle

Every transport MUST expose these lifecycle operations:

| Operation | Semantics |
|-----------|-----------|
| `start()` | Establish transport connection. After success, MUST accept `send_request()` and `send_notification()` calls. |
| `stop()` | Close transport connection and release all resources. After completion, MUST reject all send operations. MUST be idempotent. |
| `is_running` | Observable boolean property. `true` after successful `start()` and before `stop()` completes or connection is lost. |

**State machine:**
```
DISCONNECTED  --- start() --->  CONNECTING
CONNECTING    --- success --->  CONNECTED
CONNECTING    --- failure --->  ERROR
CONNECTED     --- stop() ---->  DISCONNECTED
CONNECTED     --- lost ------>  ERROR
ERROR         --- stop() ---->  DISCONNECTED
```

The transport MUST NOT allow message sending in any state other than `CONNECTED`.

### 2.5 Error Propagation

- Transport-level errors (connection lost, write failure, read timeout) MUST be surfaced to the caller.
- For pending requests, the transport MUST resolve their futures with an error.
- The transport MUST distinguish between **transport errors** (channel failed) and **protocol errors** (valid JSON-RPC error response from peer).
- When a transport error occurs, the transport SHOULD transition to `ERROR` state and fail all outstanding pending requests.

### 2.6 Backpressure

- The transport SHOULD apply backpressure when the receiver cannot consume messages fast enough (mechanism is binding-dependent).
- The transport MUST NOT silently drop messages. If a message cannot be delivered due to buffer exhaustion, the transport MUST surface an error.

### 2.7 Serialization

- The default serialization format for all bindings is JSON, consistent with JSON-RPC 2.0 wire format.
- Alternative serialization formats (MessagePack, Protocol Buffers) MAY be used by specific bindings.
- When JSON is used, all messages MUST be encoded as UTF-8.
- Numeric values MUST NOT lose precision. JSON integers outside the safe integer range (beyond 2^53) SHOULD be avoided in `id` fields and numeric parameters.

### 2.8 Message Framing

Each binding defines its own framing mechanism. Regardless of the framing method, the transport MUST guarantee:
- **Atomicity**: A complete CAP message is delivered as a single unit. The receiver MUST NOT observe a partial message.
- **Boundary detection**: The receiver MUST be able to determine where one message ends and the next begins without ambiguity.

---

## 3. Stdio Binding (Required)

The stdio binding is the reference transport for CAP. It communicates via standard input and output streams of a subprocess. **This binding is REQUIRED for all CAP implementations.**

**Reference implementation:** `PluginTransport` (aliased as `StdioTransport`) in `src/activecontext/plugins/transport.py`.

### 3.1 Framing

- Each message is a single JSON object serialized as compact UTF-8, followed by a newline (`\n`, U+000A). This is newline-delimited JSON (NDJSON).
- No `Content-Length` header or other envelope (differs from LSP).
- Each line MUST contain exactly one complete JSON object. Multi-line pretty-printed JSON is not permitted.
- Empty lines (bare `\n`) MUST be silently ignored by the receiver.
- The JSON object MUST conform to JSON-RPC 2.0 message format.

**Wire example (host sends `initialize`):**
```
{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocol_version":"1.0",...}}\n
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
- The server's `stderr` is reserved for human-readable diagnostics. The host MAY capture stderr for logging but MUST NOT parse it as protocol traffic.

### 3.3 Connection Establishment

1. Host spawns plugin server as child subprocess with pipes attached to `stdin`, `stdout`, `stderr`.
2. Connection is ready when process has started and all pipes are open. No transport-level handshake; CAP `initialize` message serves as application-level handshake.
3. Host MUST send `initialize` request as the first message after pipes are open.
4. Server MUST begin reading from `stdin` immediately upon startup. Server MUST NOT write to `stdout` before receiving `initialize` (except diagnostic output to `stderr` is allowed).

For stdio binding, only **host-managed process** connection model is applicable. The host always spawns and owns the server process.

### 3.4 Connection Termination

**Graceful shutdown:**

1. Host sends `shutdown` notification via `stdin`.
2. Host closes `stdin` pipe (signals EOF to server's read loop).
3. Host waits for server process to exit (RECOMMENDED timeout: 5 seconds).
4. If server does not exit within timeout, host SHOULD terminate the process (SIGTERM or platform equivalent), wait briefly, then kill forcefully (SIGKILL) if still running.
5. Host closes `stdout` and `stderr` pipes after process exits.

**Server shutdown behavior:**

Upon receiving `shutdown` notification or detecting EOF on `stdin`, server MUST:
1. Complete any in-flight responses.
2. Stop reading from `stdin`.
3. Close `stdout` (triggers EOF on host's read side).
4. Exit with code 0 for clean shutdown.

### 3.5 Error Detection

| Condition | Detection | Semantics |
|-----------|-----------|-----------|
| EOF on `stdout` | `readline()` returns empty bytes | Server disconnected or exited. Connection lost. |
| Write to closed `stdin` | `BrokenPipeError` or equivalent | Server's `stdin` closed. Connection lost. |
| Process exit (non-zero) | `returncode != 0` after wait | Abnormal termination. Treat as unclean shutdown. |
| Process exit (zero) | `returncode == 0` after wait | Clean exit. Expected during shutdown. |
| Invalid JSON on `stdout` | `json.JSONDecodeError` | Malformed message. Log and skip. SHOULD NOT terminate connection for a single parse error. |

When connection is lost (EOF or broken pipe), the transport MUST:
1. Fail all pending request futures with error.
2. Clear pending request map.
3. Transition `is_running` to `false`.

### 3.6 Threading Model

The reference implementation uses this concurrency model:
- **Reader task**: Single `asyncio.Task` continuously reads lines from `stdout` and dispatches incoming messages.
- **Writer**: Writes to `stdin` are serialized under `asyncio.Lock` to prevent interleaved partial writes.
- **Dispatch**: Incoming messages are classified by `id` and `method` fields:
  - Response: has `id`, no `method`. Resolves matching pending future.
  - Notification: has `method`, no `id`. Invokes notification callback.
  - Server request: has both `method` and `id`. Routes to host API handler via notification callback with `_request_id` attached.

### 3.7 Limitations

- **Single host**: One host per server subprocess (1:1 relationship). No multiplexing.
- **Local only**: Requires ability to spawn local subprocesses. Not suitable for remote servers.
- **No reconnection**: If connection is lost, new subprocess must be spawned.
- **Buffering**: Large messages may cause buffering delays. Server SHOULD flush `stdout` after each message.

---

## 4. gRPC Binding (Optional)

The gRPC binding uses HTTP/2 with Protocol Buffers (or JSON) serialization for high-performance communication. **This binding is OPTIONAL.**

### 4.1 Framing

- Standard gRPC framing: length-prefixed Protocol Buffers messages over HTTP/2 streams.
- Alternatively, gRPC-JSON (`application/grpc+json` content type) MAY be used for interoperability with languages lacking protobuf support.
- The gRPC framing layer handles message boundaries; no additional newline or length prefix is needed.

### 4.2 Service Definition

Two service design approaches are defined. Implementations MAY support either or both.

#### Approach A: Bidirectional Stream

A single bidirectional streaming RPC carries all CAP messages:

```protobuf
syntax = "proto3";
package cap.v1;

message CAPMessage {
  string json_payload = 1;  // JSON-RPC 2.0 message as JSON string
}

service CAPPlugin {
  rpc Session(stream CAPMessage) returns (stream CAPMessage);
}
```

This approach most closely mirrors the stdio model: a single bidirectional channel carrying interleaved requests, responses, and notifications. Simpler to implement and maintains same ordering guarantees as stdio.

#### Approach B: Typed RPCs

Individual RPCs per message type, with notification stream for server-to-host push:

```protobuf
syntax = "proto3";
package cap.v1;

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

  // Server push
  rpc Notifications(google.protobuf.Empty) returns (stream ServerPush);
}
```

This approach provides better observability (per-method metrics), load balancing (individual RPCs route independently), and type safety (compile-time schema validation). Requires handling Host API reverse direction separately.

### 4.3 Connection Establishment

1. Host creates gRPC channel to server's address and port.
2. For host-managed processes: host spawns server and waits for gRPC port to become available (via readiness file, health check, or known port convention).
3. For external processes: host connects to pre-configured address.
4. TLS is RECOMMENDED for non-localhost connections. For localhost, plaintext gRPC is acceptable.
5. After gRPC channel is established, host sends `Initialize` RPC (or `initialize` message over bidirectional stream).

### 4.4 Connection Termination

1. Host sends `Shutdown` RPC (or `shutdown` message over stream).
2. gRPC channel is closed gracefully (HTTP/2 GOAWAY frame).
3. For host-managed processes: host waits for process exit, with same timeout and force-kill behavior as stdio binding.
4. For external processes: host closes channel but does NOT terminate server.

### 4.5 Multiplexing

- gRPC supports multiple concurrent RPCs on a single HTTP/2 connection.
- Multiple independent hosts MAY connect to a single external server, each with its own gRPC channel and CAP session.
- With Approach B (typed RPCs), multiple `node/sync` requests for different nodes MAY be in flight simultaneously on same connection. Transport MUST still preserve per-sender ordering.

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

Application-level CAP errors (JSON-RPC error object) are carried in response payload, not in gRPC status codes. gRPC status SHOULD be `OK` even when CAP response contains JSON-RPC error.

### 4.7 Host API (Reverse Direction)

The Host API (server-to-host requests) requires server to send requests that host responds to. In gRPC, this can be handled by:

1. **Bidirectional stream** (Approach A): Both directions share same stream. Server requests and host responses interleave naturally.
2. **Callback service**: Host runs gRPC server that plugin server connects to for Host API calls. Host's address communicated during initialization.
3. **Embedded in notification stream**: Host API requests sent over `Notifications` server-streaming RPC, responses via dedicated client-streaming RPC.

Approach A is RECOMMENDED for simplicity. Approach B (callback service) is RECOMMENDED when host and server are on separate machines.

### 4.8 Use Cases

- High-throughput local deployments requiring native protobuf performance.
- Polyglot environments where gRPC provides cross-language interoperability.
- Microservice architectures where plugin servers run as independent services.
- Deployments requiring mutual TLS authentication.

---

## 5. WebSocket Binding (Optional)

The WebSocket binding uses the WebSocket protocol ([RFC 6455](https://www.rfc-editor.org/rfc/rfc6455)) for communication. **This binding is OPTIONAL.**

### 5.1 Framing

- Each CAP message is one WebSocket **text frame** containing a single JSON object (JSON-RPC 2.0 message).
- No additional framing (newlines, length prefixes) needed. WebSocket protocol handles message boundaries natively.
- Binary frames MAY be used for alternative serialization formats (e.g., MessagePack, Protocol Buffers). If binary frames used, subprotocol negotiation (Section 5.2) MUST indicate serialization format.
- Receiver MUST treat each WebSocket message as exactly one CAP message. Concatenating multiple CAP messages into single WebSocket frame is NOT permitted.

**Wire example (host sends `node/create`):**
```json
{"jsonrpc":"2.0","method":"node/create","id":2,"params":{"node_type":"shell",...}}
```

### 5.2 Connection Establishment

1. Host initiates WebSocket handshake (HTTP Upgrade) to server's endpoint.
2. Default path is `/cap`. Implementations MAY use configurable path.
3. Host SHOULD request `cap.v1` subprotocol during handshake:
   ```
   GET /cap HTTP/1.1
   Host: localhost:8080
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Protocol: cap.v1
   Sec-WebSocket-Version: 13
   Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
   ```
4. Server MUST accept `cap.v1` subprotocol if it supports this binding. If server does not recognize subprotocol, it MAY accept connection without subprotocol, but host SHOULD log warning.
5. After WebSocket connection established, host sends `initialize` message as first WebSocket text frame.
6. TLS (`wss://`) is RECOMMENDED for non-localhost connections.

Both connection models (host-managed and external process) are applicable:
- **Host-managed**: Host spawns server, which opens WebSocket listener on a port. Host connects after port available.
- **External**: Host connects to pre-configured WebSocket URL.

### 5.3 Connection Termination

1. Host sends `shutdown` notification as WebSocket text frame.
2. Host sends WebSocket Close frame with status code `1000` (Normal Closure).
3. Server completes any in-flight responses, then sends its own Close frame.
4. Both sides close underlying TCP connection.

**Abnormal termination:**

| Condition | WebSocket Close Code | Semantics |
|-----------|---------------------|-----------|
| Normal shutdown | `1000` (Normal Closure) | Clean shutdown after `shutdown` notification. |
| Host going away | `1001` (Going Away) | Host shutting down; no `shutdown` sent. |
| Protocol error | `1002` (Protocol Error) | Received invalid WebSocket data. |
| Unexpected condition | `1011` (Unexpected Condition) | Server encountered error. |

### 5.4 Multiplexing

- One CAP session per WebSocket connection. No session multiplexing within single WebSocket connection.
- Multiple hosts connect to same server via separate WebSocket connections. Server distinguishes sessions by connection identity.
- If multiplexing needed, use separate WebSocket connections (one per session) or upgrade to gRPC binding.

### 5.5 Keepalive

- WebSocket Ping/Pong frames SHOULD be used for connection liveness detection.
- Host SHOULD send Ping frames at regular interval (RECOMMENDED: 30 seconds). If no Pong received within 10 seconds, host SHOULD treat connection as lost.
- Server MUST respond to Ping frames with Pong frames (required by RFC 6455).

### 5.6 Host API (Reverse Direction)

WebSocket is naturally bidirectional: both sides can send messages at any time. Host API works identically to stdio binding — server sends JSON-RPC request over WebSocket, host sends JSON-RPC response back over same connection.

### 5.7 Use Cases

- Browser-based plugin servers (e.g., WASM plugins running in browser tab).
- Network-accessible servers where HTTP/WebSocket is only allowed protocol (corporate firewalls, cloud functions).
- Lightweight remote deployments where gRPC is too heavy.
- Development and debugging tools that benefit from browser DevTools WebSocket inspection.

---

## 6. TCP Binding (Optional)

The TCP binding uses raw TCP sockets with length-prefixed framing. **This binding is OPTIONAL.**

### 6.1 Framing

- Each message preceded by **4-byte big-endian unsigned integer** indicating message payload length in bytes.
- Payload is complete JSON object encoded as UTF-8.
- No trailing delimiter (no newline after payload).

**Wire format:**
```
+-------------------+-----------------------------+
| Length (4 bytes)   | JSON payload (Length bytes) |
| big-endian uint32 | UTF-8 encoded               |
+-------------------+-----------------------------+
```

**Wire example (host sends `initialize`):**
```
Bytes 0-3:   00 00 01 2A   (298 bytes payload)
Bytes 4-301: {"jsonrpc":"2.0","method":"initialize","id":1,"params":{...}}
```

- Maximum message size SHOULD be 16 MiB (16,777,216 bytes). Messages exceeding this SHOULD be rejected with transport error.
- A length of 0 is invalid. Receiver MUST treat zero-length message as framing error.

### 6.2 Connection Establishment

1. Server listens on TCP port (configurable; no default port assigned).
2. Host connects via standard TCP `connect()` to server's address and port.
3. After TCP connection established, host sends `initialize` message as first length-prefixed frame.

Both connection models apply:
- **Host-managed**: Host spawns server, which opens TCP listener. Host connects after port available.
- **External**: Host connects to pre-configured host:port.

TLS is NOT applied by default. For secure communication, implementation SHOULD wrap TCP socket in TLS layer (e.g., `ssl.wrap_socket` in Python, `rustls` in Rust). When TLS used, connection established after TLS handshake completes, before `initialize` message sent.

### 6.3 Connection Termination

1. Host sends `shutdown` notification as length-prefixed frame.
2. Host sends TCP FIN (half-close) by shutting down write side of socket.
3. Server completes any in-flight responses, sends them, then closes its end.
4. Both sides close socket.

**Abnormal termination:**

| Condition | TCP Behavior | Semantics |
|-----------|-------------|-----------|
| Normal shutdown | FIN after `shutdown` notification | Clean disconnect. |
| Abnormal termination | RST (connection reset) | Unclean disconnect; treat as connection lost. |
| Read timeout | No data within timeout | Connection may be dead; close and reconnect. |

### 6.4 Keepalive

- TCP keepalive (`SO_KEEPALIVE`) is RECOMMENDED to detect dead connections. Implementations SHOULD set `TCP_KEEPIDLE` (or platform equivalent) to reasonable value (RECOMMENDED: 60 seconds).
- Application-level keepalive via periodic ping messages is OPTIONAL. If implemented, SHOULD use JSON-RPC notification with reserved method name (e.g., `"ping"`) that receiver silently ignores.

### 6.5 Host API (Reverse Direction)

TCP is bidirectional: both sides can send length-prefixed frames at any time. Host API works identically to stdio binding, with server sending JSON-RPC requests and host sending JSON-RPC responses over same TCP connection.

### 6.6 Use Cases

- Simple network deployments without HTTP overhead.
- Embedded systems and IoT devices with minimal protocol stacks.
- High-throughput local communication where stdio pipe buffering is bottleneck.
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

- **Default**: stdio. Simplest setup, works everywhere, no network configuration. Best for local single-host deployments and IDE integrations.
- **Performance-critical**: gRPC. Native protobuf serialization, HTTP/2 multiplexing, built-in load balancing. Best for high-throughput deployments with many concurrent nodes.
- **Web/browser**: WebSocket. Only binding that works in browsers natively. Best for web-based plugin servers and WASM plugins.
- **Embedded/minimal**: TCP. Minimal dependencies, no HTTP overhead. Best for constrained environments and custom network topologies.

### 7.3 Mixed Transport Deployments

A single host MAY use different transport bindings for different plugin servers simultaneously. For example:
- Local plugins connected via stdio
- Shared analytics plugin connected via gRPC
- Browser-based visualization plugin connected via WebSocket

The host's `PluginConnection` layer programs against the `CAPTransport` protocol interface (defined in `src/activecontext/plugins/cap_transport.py`), so transport binding is transparent to protocol logic.

---

## 8. Conformance

### 8.1 Required Binding

All CAP implementations MUST support the **stdio binding** (Section 3). This ensures baseline interoperability across all environments and languages.

### 8.2 Optional Bindings

The gRPC (Section 4), WebSocket (Section 5), and TCP (Section 6) bindings are OPTIONAL. Implementations SHOULD document which bindings they support.

### 8.3 Transport Protocol Interface

All transport implementations MUST satisfy the `CAPTransport` protocol interface:

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

This interface is defined in `src/activecontext/plugins/cap_transport.py`. New transport bindings MUST implement all methods.

### 8.4 Interoperability Testing

Implementations supporting multiple bindings SHOULD verify that the same CAP session (initialize, create, sync, destroy, shutdown) produces identical application-level behavior regardless of which binding is used. Transport-level details (framing, error surfaces) will differ, but protocol semantics MUST be preserved.

---

## Appendix A: Binding Summary

| Binding | Section | Framing | Connection Model | Multiplexing | Status |
|---------|---------|---------|------------------|--------------|--------|
| stdio | 3 | Newline-delimited JSON | Host-managed only | No | **Required** |
| gRPC | 4 | HTTP/2 + protobuf | Both | Yes | Optional |
| WebSocket | 5 | WS text frames | Both | No | Optional |
| TCP | 6 | 4-byte length prefix + JSON | Both | No | Optional |

## Appendix B: Wire Format Comparison

The same `node/create` message framed across all four bindings:

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

For all network-accessible bindings (gRPC, WebSocket, TCP), implementations SHOULD:
- Use TLS for all non-localhost connections.
- Authenticate connections before accepting `initialize` handshake.
- Validate `Origin` header for WebSocket connections to prevent cross-site WebSocket hijacking (CSWSH).
- Apply rate limiting to prevent denial-of-service attacks.

---

## References

- **Protocol Specification**: [`spec.md`](spec.md)
- **Message Catalog**: [`messages.md`](messages.md)
- **Serialization Bindings**: Moving existing `docs/cap-serialization-bindings.md` to `docs/cap/serialization.md` (companion document)
- **Protobuf Schema**: `../cap.proto`
- **Reference Implementation**: `src/activecontext/plugins/`
