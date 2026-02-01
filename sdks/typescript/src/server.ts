/**
 * CAPServer -- the main entry point for writing CAP plugin servers.
 *
 * Manages node type registration, instance lifecycle, and the JSON-RPC
 * dispatch loop. A plugin server creates a CAPServer, registers its
 * node types with schemas and factories, then calls serve() to start
 * the main loop.
 *
 * @module server
 */

import {
  Methods,
  ErrorCodes,
  PROTOCOL_VERSION,
} from "./types";
import type {
  ServerCapabilities,
  HostCapabilities,
  NodeTypeSchema,
  InitializeParams,
  InitializeResult,
  NodeCreateParams,
  NodeCreateResult,
  NodeSyncParams,
  NodeSyncResult,
  NodeCallParams,
  NodeCallResult,
  NodeDestroyParams,
  NodeSerializeParams,
  NodeSerializeResult,
  PendingCall,
  Notification,
  RootInfo,
} from "./types";
import { StdioTransport } from "./transport";
import { HostAPI, CAPError } from "./host-api";
import type { NodePlugin } from "./node";

// ---------------------------------------------------------------------------
// Node type registration
// ---------------------------------------------------------------------------

/**
 * Factory function that creates a node instance.
 *
 * Called when the host sends `node/create`. Receives the positional args,
 * keyword args, and (optionally) a host-assigned node ID.
 */
export type NodeFactory<T extends NodePlugin = NodePlugin> = (
  args: unknown[],
  kwargs: Record<string, unknown>,
  nodeId?: string
) => T | Promise<T>;

interface NodeTypeRegistration {
  schema: NodeTypeSchema;
  factory: NodeFactory;
}

// ---------------------------------------------------------------------------
// CAPServer
// ---------------------------------------------------------------------------

/**
 * Options for creating a CAPServer.
 */
export interface CAPServerOptions {
  /** Human-readable server name (required). */
  name: string;
  /** Server version string. */
  version?: string;
  /** Server capabilities to advertise. */
  capabilities?: ServerCapabilities;
  /** Default request timeout in milliseconds. */
  timeout?: number;
}

/**
 * A CAP plugin server.
 *
 * Usage:
 * 1. Create a CAPServer with a name.
 * 2. Register node types with schemas and factories.
 * 3. Call serve() to start the stdio main loop.
 *
 * @example
 * ```ts
 * const server = new CAPServer({ name: "my-plugin", version: "0.1.0" });
 *
 * server.register("echo", echoFactory, echoSchema);
 *
 * server.serve().catch(console.error);
 * ```
 */
export class CAPServer {
  private readonly serverName: string;
  private readonly serverVersion: string;
  private readonly serverCapabilities: ServerCapabilities;
  private readonly transport: StdioTransport;
  private readonly nodeTypes = new Map<string, NodeTypeRegistration>();
  private readonly instances = new Map<string, NodePlugin>();

  private hostCapabilities: HostCapabilities = {};
  private roots: RootInfo[] = [];
  private cwd = "";
  private sessionId = "";
  private hostApi: HostAPI | null = null;
  private initialized = false;
  private shuttingDown = false;

  constructor(options: CAPServerOptions) {
    this.serverName = options.name;
    this.serverVersion = options.version ?? "";
    this.serverCapabilities = {
      sync: true,
      immediate_call: false,
      push_dirty: true,
      push_notifications: true,
      ...options.capabilities,
    };
    this.transport = new StdioTransport(options.timeout ?? 30000);
  }

  /**
   * Register a node type.
   *
   * Must be called before serve(). Each node type has a unique string
   * identifier, a factory function for creating instances, and a schema
   * describing its constructor, properties, and methods.
   *
   * @param nodeType - Type identifier (must match schema.node_type).
   * @param factory - Factory function to create node instances.
   * @param schema - Full node type schema for the initialize handshake.
   */
  register<T extends NodePlugin>(
    nodeType: string,
    factory: NodeFactory<T>,
    schema: NodeTypeSchema
  ): void {
    if (this.initialized) {
      throw new Error("Cannot register node types after initialization");
    }
    if (schema.node_type !== nodeType) {
      throw new Error(
        `Schema node_type "${schema.node_type}" does not match registration key "${nodeType}"`
      );
    }
    this.nodeTypes.set(nodeType, {
      schema,
      factory: factory as NodeFactory,
    });
  }

  /**
   * Get the Host API client.
   *
   * Available after initialization. Throws if called before serve()
   * completes the handshake.
   *
   * @returns The HostAPI instance for making server-to-host requests.
   */
  getHostApi(): HostAPI {
    if (!this.hostApi) {
      throw new Error("Host API not available until after initialization");
    }
    return this.hostApi;
  }

  /**
   * Get the current working directory from the host.
   *
   * @returns The cwd string from the initialize handshake.
   */
  getCwd(): string {
    return this.cwd;
  }

  /**
   * Get the session ID from the host.
   *
   * @returns The session_id string from the initialize handshake.
   */
  getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Get the roots from the host.
   *
   * @returns The roots array from the initialize handshake.
   */
  getRoots(): RootInfo[] {
    return [...this.roots];
  }

  /**
   * Push a dirty notification for a node.
   *
   * Tells the host that this node's state has changed and needs sync.
   * Only available if push_dirty capability is declared.
   *
   * @param nodeId - The node that changed.
   */
  notifyDirty(nodeId: string): void {
    if (!this.serverCapabilities.push_dirty) return;
    this.transport.sendNotification(Methods.NODE_DIRTY, { node_id: nodeId });
  }

  /**
   * Push a notification for a node.
   *
   * Delivers a notification to the node's parent chain in the host graph.
   * Only available if push_notifications capability is declared.
   *
   * @param nodeId - Source node.
   * @param description - What changed.
   * @param level - Notification level: "ignore", "hold", or "wake".
   */
  notifyNode(
    nodeId: string,
    description: string,
    level: "ignore" | "hold" | "wake" = "hold"
  ): void {
    if (!this.serverCapabilities.push_notifications) return;
    this.transport.sendNotification(Methods.NODE_NOTIFICATION, {
      node_id: nodeId,
      description,
      level,
    });
  }

  /**
   * Start the server main loop.
   *
   * Begins reading from stdin and dispatching messages. The returned
   * promise resolves when the server shuts down (stdin closes or
   * shutdown notification received).
   */
  async serve(): Promise<void> {
    return new Promise<void>((resolve) => {
      // Wire up request handler
      this.transport.onRequest(
        async (method: string, params: unknown, _id: number | string) => {
          return this.handleRequest(method, params);
        }
      );

      // Wire up notification handler (only shutdown for now)
      this.transport.onNotification((method: string, _params: unknown) => {
        if (method === Methods.SHUTDOWN) {
          this.shuttingDown = true;
          process.stderr.write(`[${this.serverName}] Shutdown received\n`);
          // Allow in-flight responses to complete, then stop
          setImmediate(() => {
            this.transport.stop();
            resolve();
          });
        }
      });

      // Start the read loop
      this.transport.start();

      // Also resolve if stdin closes (EOF)
      process.stdin.on("end", () => {
        if (!this.shuttingDown) {
          this.transport.stop();
          resolve();
        }
      });
    });
  }

  // -----------------------------------------------------------------------
  // Request dispatch
  // -----------------------------------------------------------------------

  private async handleRequest(
    method: string,
    params: unknown
  ): Promise<unknown> {
    switch (method) {
      case Methods.INITIALIZE:
        return this.handleInitialize(params as InitializeParams);
      case Methods.NODE_CREATE:
        return this.handleNodeCreate(params as NodeCreateParams);
      case Methods.NODE_SYNC:
        return this.handleNodeSync(params as NodeSyncParams);
      case Methods.NODE_CALL:
        return this.handleNodeCall(params as NodeCallParams);
      case Methods.NODE_DESTROY:
        return this.handleNodeDestroy(params as NodeDestroyParams);
      case Methods.NODE_SERIALIZE:
        return this.handleNodeSerialize(params as NodeSerializeParams);
      default:
        throw new CAPError(
          ErrorCodes.METHOD_NOT_FOUND,
          `Unknown method: ${method}`
        );
    }
  }

  // -----------------------------------------------------------------------
  // Lifecycle handlers
  // -----------------------------------------------------------------------

  private handleInitialize(params: InitializeParams): InitializeResult {
    this.hostCapabilities = params.host_capabilities ?? {};
    this.roots = params.roots ?? [];
    this.cwd = params.cwd ?? "";
    this.sessionId = params.session_id ?? "";
    this.initialized = true;

    // Create the Host API client
    this.hostApi = new HostAPI(this.transport, this.hostCapabilities);

    // Collect all registered schemas
    const nodeTypes: NodeTypeSchema[] = [];
    for (const reg of this.nodeTypes.values()) {
      nodeTypes.push(reg.schema);
    }

    return {
      server_name: this.serverName,
      server_version: this.serverVersion,
      protocol_version: PROTOCOL_VERSION,
      node_types: nodeTypes,
      server_capabilities: this.serverCapabilities,
    };
  }

  // -----------------------------------------------------------------------
  // Node management handlers
  // -----------------------------------------------------------------------

  private async handleNodeCreate(
    params: NodeCreateParams
  ): Promise<NodeCreateResult> {
    const registration = this.nodeTypes.get(params.node_type);
    if (!registration) {
      throw new CAPError(
        ErrorCodes.NODE_TYPE_UNKNOWN,
        `Unknown node type: ${params.node_type}`
      );
    }

    let node: NodePlugin;
    try {
      node = await registration.factory(
        params.args ?? [],
        params.kwargs ?? {},
        params.node_id || undefined
      );
    } catch (err) {
      throw new CAPError(
        ErrorCodes.NODE_CREATE_FAILED,
        `Failed to create ${params.node_type}: ${(err as Error).message}`
      );
    }

    this.instances.set(node.nodeId, node);

    const cwd = this.cwd || ".";
    return {
      node_id: node.nodeId,
      state: node.toDict(),
      renders: {
        header: node.renderHeader(cwd),
        content: node.renderContent(cwd),
        detail: node.renderDetail(cwd),
      },
      tokens: node.getTokenBreakdown(),
      digest: node.getDigest(),
    };
  }

  private async handleNodeSync(
    params: NodeSyncParams
  ): Promise<NodeSyncResult> {
    const node = this.requireNode(params.node_id);
    const cwd = params.cwd || this.cwd || ".";
    const allNotifications: Notification[] = [];

    // Execute batched method calls in order
    if (params.calls && params.calls.length > 0) {
      for (const call of params.calls) {
        await this.executeCall(node, call, allNotifications);
      }
    }

    // Tick the node
    const tickResult = await Promise.resolve(node.tick());
    if (Array.isArray(tickResult)) {
      allNotifications.push(...tickResult);
    }

    // Clear dirty flag if present
    if ("clearDirty" in node && typeof (node as { clearDirty: () => void }).clearDirty === "function") {
      (node as { clearDirty: () => void }).clearDirty();
    }

    return {
      state: node.toDict(),
      renders: {
        header: node.renderHeader(cwd),
        content: node.renderContent(cwd),
        detail: node.renderDetail(cwd),
      },
      tokens: node.getTokenBreakdown(),
      digest: node.getDigest(),
      notifications: allNotifications,
    };
  }

  private async handleNodeCall(
    params: NodeCallParams
  ): Promise<NodeCallResult> {
    if (!this.serverCapabilities.immediate_call) {
      throw new CAPError(
        ErrorCodes.METHOD_NOT_AVAILABLE,
        "Server does not support immediate_call"
      );
    }

    const node = this.requireNode(params.node_id);

    if (!node.handleCall) {
      throw new CAPError(
        ErrorCodes.METHOD_NOT_AVAILABLE,
        `Node ${params.node_id} does not support method calls`
      );
    }

    const result = await Promise.resolve(
      node.handleCall(
        params.method,
        params.args ?? [],
        params.kwargs ?? {}
      )
    );

    // Check if state changed
    const stateChanged =
      "isDirty" in node &&
      typeof (node as { isDirty: () => boolean }).isDirty === "function" &&
      (node as { isDirty: () => boolean }).isDirty();

    return {
      result,
      state_changed: stateChanged || false,
    };
  }

  private async handleNodeDestroy(
    params: NodeDestroyParams
  ): Promise<Record<string, never>> {
    this.requireNode(params.node_id);
    this.instances.delete(params.node_id);
    return {};
  }

  private handleNodeSerialize(
    params: NodeSerializeParams
  ): NodeSerializeResult {
    const node = this.requireNode(params.node_id);
    return {
      data: {
        ...node.toDict(),
        node_type: node.nodeType,
      },
    };
  }

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  private requireNode(nodeId: string): NodePlugin {
    const node = this.instances.get(nodeId);
    if (!node) {
      throw new CAPError(
        ErrorCodes.NODE_NOT_FOUND,
        `Node not found: ${nodeId}`
      );
    }
    return node;
  }

  private async executeCall(
    node: NodePlugin,
    call: PendingCall,
    notifications: Notification[]
  ): Promise<void> {
    if (!node.handleCall) {
      throw new CAPError(
        ErrorCodes.METHOD_NOT_AVAILABLE,
        `Method "${call.method}" not available on node ${node.nodeId}`
      );
    }

    try {
      await Promise.resolve(
        node.handleCall(
          call.method,
          call.args ?? [],
          call.kwargs ?? {}
        )
      );
    } catch (err) {
      // Per spec: if a call fails, continue processing remaining calls
      // and report the error via notifications
      notifications.push({
        description: `Method "${call.method}" failed: ${(err as Error).message}`,
        level: "hold",
      });
    }
  }
}
