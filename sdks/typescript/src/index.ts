/**
 * @activecontext/cap-plugin -- CAP Plugin SDK for TypeScript.
 *
 * Build custom context node types for ActiveContext using the
 * Context Application Protocol (CAP).
 *
 * @example
 * ```ts
 * import {
 *   CAPServer,
 *   BaseNode,
 *   nodeTypeSchema,
 *   required,
 *   property,
 *   method,
 *   constructor_,
 * } from "@activecontext/cap-plugin";
 *
 * class EchoNode extends BaseNode {
 *   readonly nodeType = "echo";
 *   private text: string;
 *
 *   constructor(text: string, nodeId?: string) {
 *     super(nodeId);
 *     this.text = text;
 *   }
 *
 *   renderHeader(cwd: string): string {
 *     return `[ECHO] ${this.text.substring(0, 40)}`;
 *   }
 *
 *   renderContent(cwd: string): string {
 *     return this.text;
 *   }
 * }
 *
 * const server = new CAPServer({ name: "echo-plugin", version: "0.1.0" });
 *
 * server.register(
 *   "echo",
 *   (args) => new EchoNode(String(args[0] ?? "")),
 *   nodeTypeSchema("echo", {
 *     description: "Echoes input text.",
 *     constructor: constructor_({
 *       positional: [required("text", "str", "Text to echo.")],
 *     }),
 *   })
 * );
 *
 * server.serve();
 * ```
 *
 * @module index
 */

// --- Types ---
export {
  PROTOCOL_NAME,
  PROTOCOL_VERSION,
  Methods,
  ErrorCodes,
} from "./types";

export type {
  MethodName,
  ErrorCode,
  PluginConnectionStatus,
  RootInfo,
  HostCapabilities,
  ServerCapabilities,
  ParamSchema,
  ConstructorSchema,
  PropertySchema,
  MethodSchema,
  NodeTypeSchema,
  InitializeParams,
  InitializeResult,
  NodeCreateParams,
  NodeCreateResult,
  PendingCall,
  NodeSyncParams,
  NodeSyncResult,
  NodeCallParams,
  NodeCallResult,
  NodeDestroyParams,
  NodeSerializeParams,
  NodeSerializeResult,
  NodeDirtyParams,
  NodeNotificationParams,
  HostCreateNodeParams,
  HostCreateNodeResult,
  HostInvokeParams,
  HostInvokeResult,
  HostQueryRootsResult,
  HostResolveRootParams,
  HostResolveRootResult,
  RenderSnapshot,
  TokenEstimate,
  Notification,
  NotificationLevel,
  JsonRpcRequest,
  JsonRpcNotification,
  JsonRpcResponse,
  JsonRpcErrorData,
  JsonRpcErrorResponse,
  JsonRpcMessage,
} from "./types";

export { MISSING } from "./types";
export type { Missing } from "./types";

// --- Node ---
export type { NodePlugin } from "./node";
export { BaseNode } from "./node";

// --- Schema helpers ---
export {
  param,
  required,
  optional,
  property,
  method,
  constructor_,
  nodeTypeSchema,
} from "./schema";

// --- Transport ---
export { StdioTransport } from "./transport";
export type { RequestHandler, NotificationHandler } from "./transport";

// --- Server ---
export { CAPServer } from "./server";
export type { CAPServerOptions, NodeFactory } from "./server";

// --- Host API ---
export { HostAPI, CAPError } from "./host-api";
