/**
 * CAP (Context Application Protocol) type definitions.
 *
 * Mirrors the wire types from the Python reference implementation
 * (src/activecontext/plugins/wire.py). All types here correspond
 * directly to JSON-RPC message params and results.
 *
 * @module types
 */

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/** Protocol identifier. */
export const PROTOCOL_NAME = "CAP";

/** Current protocol version, sent during initialize handshake. */
export const PROTOCOL_VERSION = "1.0";

// ---------------------------------------------------------------------------
// JSON-RPC method names
// ---------------------------------------------------------------------------

/** JSON-RPC method names organized by direction and category. */
export const Methods = {
  // --- Lifecycle (host -> server) ---
  INITIALIZE: "initialize",
  SHUTDOWN: "shutdown",

  // --- Node management (host -> server) ---
  NODE_CREATE: "node/create",
  NODE_SYNC: "node/sync",
  NODE_CALL: "node/call",
  NODE_DESTROY: "node/destroy",
  NODE_SERIALIZE: "node/serialize",

  // --- Push notifications (server -> host, no response) ---
  NODE_DIRTY: "node/dirty",
  NODE_NOTIFICATION: "node/notification",

  // --- Host API (server -> host) ---
  HOST_CREATE_NODE: "host/create_node",
  HOST_INVOKE: "host/invoke",
  HOST_QUERY_ROOTS: "host/query_roots",
  HOST_RESOLVE_ROOT: "host/resolve_root",
} as const;

/** Union type of all method name strings. */
export type MethodName = (typeof Methods)[keyof typeof Methods];

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

/** Standard JSON-RPC 2.0 error codes plus CAP-specific codes. */
export const ErrorCodes = {
  // Standard JSON-RPC 2.0
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,

  // CAP-specific (-32000 to -32099)
  NODE_NOT_FOUND: -32000,
  NODE_TYPE_UNKNOWN: -32001,
  METHOD_NOT_AVAILABLE: -32002,
  ROOT_NOT_FOUND: -32003,
  PERMISSION_DENIED: -32004,
  NODE_CREATE_FAILED: -32005,
} as const;

/** Union type of all error code values. */
export type ErrorCode = (typeof ErrorCodes)[keyof typeof ErrorCodes];

// ---------------------------------------------------------------------------
// Connection status
// ---------------------------------------------------------------------------

/** Status of a plugin server connection. */
export type PluginConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

// ---------------------------------------------------------------------------
// Filesystem roots
// ---------------------------------------------------------------------------

/** A filesystem root exposed to the plugin server. */
export interface RootInfo {
  /** Root URI (e.g., "file:///home/user/project"). */
  uri: string;
  /** Human-readable name (e.g., "cwd", "home"). */
  name: string;
  /** What this root represents. */
  description?: string;
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/** Capabilities the host offers to plugin servers. */
export interface HostCapabilities {
  /** Host supports host/create_node for child creation. */
  create_node?: boolean;
  /** Host supports host/invoke for method calls on host nodes. */
  invoke?: boolean;
  /** Host supports host/query_roots and host/resolve_root. */
  roots?: boolean;
  /** Host accepts node/dirty and node/notification push. */
  notifications?: boolean;
}

/** Capabilities the server declares. */
export interface ServerCapabilities {
  /** Server supports node/sync (tick-based synchronization). */
  sync?: boolean;
  /** Server supports node/call (immediate method execution outside tick). */
  immediate_call?: boolean;
  /** Server pushes node/dirty notifications. */
  push_dirty?: boolean;
  /** Server pushes node/notification messages. */
  push_notifications?: boolean;
}

// ---------------------------------------------------------------------------
// Schema types
// ---------------------------------------------------------------------------

/**
 * Sentinel value representing a required parameter with no default.
 * In JSON wire format, a required parameter omits the `default` field.
 */
export const MISSING = Symbol("MISSING");
export type Missing = typeof MISSING;

/** Schema for a single parameter. */
export interface ParamSchema {
  /** Parameter name. */
  name: string;
  /** Type hint as string (e.g., "str", "int", "bool", "list[str]"). */
  type?: string;
  /**
   * Default value, or omitted if the parameter is required.
   * A value of `null` means the parameter is optional with default null.
   * Omitting this field entirely means the parameter is required (MISSING).
   */
  default?: unknown;
  /** Human-readable description. */
  description?: string;
}

/** Schema for the DSL constructor function. */
export interface ConstructorSchema {
  /** Positional parameters, in order. */
  positional?: ParamSchema[];
  /** Variadic parameter (*args), if any. */
  variadic?: ParamSchema | null;
  /** Keyword-only parameters, with defaults. */
  named?: ParamSchema[];
}

/** Schema for a DSL-visible property. */
export interface PropertySchema {
  /** Property name (e.g., "is_complete", "output"). */
  name: string;
  /** Type hint as string. */
  type?: string;
  /** Whether the DSL can read this property. */
  readable?: boolean;
  /** Whether the DSL can assign to this property. */
  writable?: boolean;
  /** Human-readable description. */
  description?: string;
}

/** Schema for a DSL-callable method. */
export interface MethodSchema {
  /** Method name (e.g., "cancel", "set_content"). */
  name: string;
  /** Method parameters (excluding self). */
  params?: ParamSchema[];
  /** Return type hint as string. */
  returns?: string;
  /** Human-readable description. */
  description?: string;
  /** If true, method returns self for fluent chaining. */
  chainable?: boolean;
}

/** Full schema for a plugin node type. */
export interface NodeTypeSchema {
  /** Type identifier (e.g., "shell", "topic"). Must be unique per server. */
  node_type: string;
  /** Human-readable description for documentation and LLM prompts. */
  description?: string;
  /** Constructor parameter schema. */
  constructor?: ConstructorSchema;
  /** DSL-visible properties. */
  properties?: PropertySchema[];
  /** DSL-callable methods. */
  methods?: MethodSchema[];
}

// ---------------------------------------------------------------------------
// Initialize handshake
// ---------------------------------------------------------------------------

/** Parameters for the `initialize` request (host -> server). */
export interface InitializeParams {
  /** Protocol version the host supports. */
  protocol_version?: string;
  /** Capabilities the host offers. */
  host_capabilities?: HostCapabilities;
  /** Filesystem roots available to the server. */
  roots?: RootInfo[];
  /** Session identifier for logging and coordination. */
  session_id?: string;
  /** Working directory for the session. */
  cwd?: string;
}

/** Result of the `initialize` request (server -> host). */
export interface InitializeResult {
  /** Human-readable server name. */
  server_name: string;
  /** Server version string. */
  server_version?: string;
  /** Protocol version the server supports. */
  protocol_version?: string;
  /** Node types this server provides, with full schemas. */
  node_types?: NodeTypeSchema[];
  /** Capabilities the server declares. */
  server_capabilities?: ServerCapabilities;
}

// ---------------------------------------------------------------------------
// Node management messages (host -> server)
// ---------------------------------------------------------------------------

/** Parameters for `node/create` (host -> server). */
export interface NodeCreateParams {
  /** Type of node to create. */
  node_type: string;
  /** Positional constructor arguments. */
  args?: unknown[];
  /** Keyword constructor arguments. */
  kwargs?: Record<string, unknown>;
  /** Host-assigned node ID. If empty, server generates one. */
  node_id?: string;
}

/** Result of `node/create` (server -> host). */
export interface NodeCreateResult {
  /** Node instance ID. */
  node_id: string;
  /** Initial serialized state. */
  state?: Record<string, unknown>;
  /** Initial renders: {"header": "...", "content": "...", "detail": "..."}. */
  renders?: RenderSnapshot;
  /** Initial token estimates: {"collapsed": N, "summary": N, "detail": N}. */
  tokens?: TokenEstimate;
  /** Initial digest metadata. */
  digest?: Record<string, unknown>;
}

/** A method call queued for batch execution during sync. */
export interface PendingCall {
  /** Method name (e.g., "set_content", "cancel"). */
  method: string;
  /** Positional arguments. */
  args?: unknown[];
  /** Keyword arguments. */
  kwargs?: Record<string, unknown>;
}

/** Parameters for `node/sync` (host -> server). */
export interface NodeSyncParams {
  /** Node to synchronize. */
  node_id: string;
  /** Queued method calls to execute before returning state. */
  calls?: PendingCall[];
  /** Working directory for render methods. */
  cwd?: string;
}

/** Result of `node/sync` (server -> host). */
export interface NodeSyncResult {
  /** Full serialized node state. */
  state?: Record<string, unknown>;
  /** Rendered content at all three levels. */
  renders?: RenderSnapshot;
  /** Token estimates for each rendering level. */
  tokens?: TokenEstimate;
  /** Compact digest metadata. */
  digest?: Record<string, unknown>;
  /** Notifications generated since last sync. */
  notifications?: Notification[];
}

/** Parameters for `node/call` (host -> server). */
export interface NodeCallParams {
  /** Node to call method on. */
  node_id: string;
  /** Method name. */
  method: string;
  /** Positional arguments. */
  args?: unknown[];
  /** Keyword arguments. */
  kwargs?: Record<string, unknown>;
}

/** Result of `node/call` (server -> host). */
export interface NodeCallResult {
  /** Return value of the method call. */
  result?: unknown;
  /** Whether the call changed the node's state (triggers sync). */
  state_changed?: boolean;
}

/** Parameters for `node/destroy` (host -> server). */
export interface NodeDestroyParams {
  /** Node to destroy. */
  node_id: string;
}

/** Parameters for `node/serialize` (host -> server). */
export interface NodeSerializeParams {
  /** Node to serialize. */
  node_id: string;
}

/** Result of `node/serialize` (server -> host). */
export interface NodeSerializeResult {
  /** Full serialized state for reconstruction. Must include node_type. */
  data?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Push notifications (server -> host)
// ---------------------------------------------------------------------------

/** Parameters for `node/dirty` notification (server -> host). */
export interface NodeDirtyParams {
  /** Node whose state changed. */
  node_id: string;
}

/** Parameters for `node/notification` (server -> host). */
export interface NodeNotificationParams {
  /** Source node. */
  node_id: string;
  /** Human-readable description of what changed. */
  description: string;
  /** Notification level: "ignore", "hold", or "wake". */
  level?: NotificationLevel;
}

// ---------------------------------------------------------------------------
// Host API messages (server -> host)
// ---------------------------------------------------------------------------

/** Parameters for `host/create_node` (server -> host). */
export interface HostCreateNodeParams {
  /** Type of node to create. */
  node_type: string;
  /** Positional constructor arguments. */
  args?: unknown[];
  /** Keyword constructor arguments. */
  kwargs?: Record<string, unknown>;
  /** Parent node ID to link to. */
  parent_id?: string;
}

/** Result of `host/create_node` (host -> server). */
export interface HostCreateNodeResult {
  /** ID of the created node in the host graph. */
  node_id: string;
}

/** Parameters for `host/invoke` (server -> host). */
export interface HostInvokeParams {
  /** Target node in the host graph. */
  node_id: string;
  /** Method name to call. */
  method: string;
  /** Positional arguments. */
  args?: unknown[];
  /** Keyword arguments. */
  kwargs?: Record<string, unknown>;
}

/** Result of `host/invoke` (host -> server). */
export interface HostInvokeResult {
  /** Return value of the method call. */
  result?: unknown;
}

/** Result of `host/query_roots` (host -> server). */
export interface HostQueryRootsResult {
  /** Available filesystem roots. */
  roots?: RootInfo[];
}

/** Parameters for `host/resolve_root` (server -> host). */
export interface HostResolveRootParams {
  /** Root URI to resolve against. */
  root_uri: string;
  /** Normalized path relative to the root. */
  normalized_path: string;
}

/** Result of `host/resolve_root` (host -> server). */
export interface HostResolveRootResult {
  /** Resolved absolute filesystem path. */
  real_path: string;
  /** Whether the file exists at the resolved path. */
  exists?: boolean;
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/** Pre-rendered content at all three levels. */
export interface RenderSnapshot {
  /** Header section -- always shown. */
  header?: string;
  /** Content section -- shown at CONTENT expansion and above. */
  content?: string;
  /** Detail section -- shown at ALL expansion only. */
  detail?: string;
}

/** Token estimates for each rendering level. */
export interface TokenEstimate {
  /** Tokens for header rendering. */
  collapsed?: number;
  /** Additional tokens for content rendering. */
  summary?: number;
  /** Additional tokens for detail rendering. */
  detail?: number;
}

/** A notification from a node to its parent chain. */
export interface Notification {
  /** Human-readable description of what changed. */
  description: string;
  /** Notification level. */
  level?: NotificationLevel;
}

/** Notification severity levels. */
export type NotificationLevel = "ignore" | "hold" | "wake";

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 message types
// ---------------------------------------------------------------------------

/** A JSON-RPC 2.0 request message. */
export interface JsonRpcRequest {
  jsonrpc: "2.0";
  method: string;
  id: number | string;
  params?: unknown;
}

/** A JSON-RPC 2.0 notification message (no id, no response expected). */
export interface JsonRpcNotification {
  jsonrpc: "2.0";
  method: string;
  params?: unknown;
}

/** A JSON-RPC 2.0 success response. */
export interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number | string;
  result: unknown;
}

/** A JSON-RPC 2.0 error object. */
export interface JsonRpcErrorData {
  code: number;
  message: string;
  data?: unknown;
}

/** A JSON-RPC 2.0 error response. */
export interface JsonRpcErrorResponse {
  jsonrpc: "2.0";
  id: number | string | null;
  error: JsonRpcErrorData;
}

/** Any JSON-RPC 2.0 message. */
export type JsonRpcMessage =
  | JsonRpcRequest
  | JsonRpcNotification
  | JsonRpcResponse
  | JsonRpcErrorResponse;
