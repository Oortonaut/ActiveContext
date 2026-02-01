/**
 * Host API client.
 *
 * Provides typed methods for server-to-host requests. Plugin nodes
 * use the HostAPI to interact with the host's context graph:
 * creating child nodes, invoking methods on host nodes, and querying
 * filesystem roots.
 *
 * The HostAPI checks host capabilities before sending requests and
 * throws if a capability is not available.
 *
 * @module host-api
 */

import { Methods, ErrorCodes } from "./types";
import type {
  HostCapabilities,
  HostCreateNodeParams,
  HostCreateNodeResult,
  HostInvokeParams,
  HostInvokeResult,
  HostQueryRootsResult,
  HostResolveRootParams,
  HostResolveRootResult,
  RootInfo,
} from "./types";
import type { StdioTransport } from "./transport";

// ---------------------------------------------------------------------------
// CAPError
// ---------------------------------------------------------------------------

/**
 * Error class for CAP protocol errors with error code.
 */
export class CAPError extends Error {
  readonly code: number;
  readonly data?: unknown;

  constructor(code: number, message: string, data?: unknown) {
    super(message);
    this.name = "CAPError";
    this.code = code;
    this.data = data;
  }
}

// ---------------------------------------------------------------------------
// HostAPI
// ---------------------------------------------------------------------------

/**
 * Typed client for the CAP Host API.
 *
 * Provides methods matching the four host API endpoints:
 * - createNode: Create a child node in the host graph.
 * - invoke: Call a method on an existing host node.
 * - queryRoots: Get available filesystem roots.
 * - resolveRoot: Resolve a normalized path through a root.
 *
 * Each method checks host capabilities before sending the request.
 * If the required capability is not declared, the method throws a
 * CAPError with PERMISSION_DENIED.
 *
 * @example
 * ```ts
 * // Inside a node's tick() or handleCall():
 * const nodeId = await hostApi.createNode("artifact", ["output text"], {
 *   artifact_type: "output",
 * });
 *
 * const roots = await hostApi.queryRoots();
 * const resolved = await hostApi.resolveRoot(roots[0].uri, "src/main.ts");
 * ```
 */
export class HostAPI {
  private readonly transport: StdioTransport;
  private readonly capabilities: HostCapabilities;

  constructor(transport: StdioTransport, capabilities: HostCapabilities) {
    this.transport = transport;
    this.capabilities = capabilities;
  }

  /**
   * Create a child node in the host's context graph.
   *
   * Requires the `create_node` host capability.
   *
   * @param nodeType - Type of node to create.
   * @param args - Positional constructor arguments.
   * @param kwargs - Keyword constructor arguments.
   * @param parentId - Parent node ID to link to (optional).
   * @returns The ID of the created node.
   */
  async createNode(
    nodeType: string,
    args: unknown[] = [],
    kwargs: Record<string, unknown> = {},
    parentId?: string
  ): Promise<string> {
    this.requireCapability("create_node", "host/create_node");

    const params: HostCreateNodeParams = {
      node_type: nodeType,
      args,
      kwargs,
      ...(parentId ? { parent_id: parentId } : {}),
    };

    const result = (await this.transport.sendRequest(
      Methods.HOST_CREATE_NODE,
      params
    )) as HostCreateNodeResult;

    return result.node_id;
  }

  /**
   * Invoke a method on an existing node in the host graph.
   *
   * Requires the `invoke` host capability.
   *
   * @param nodeId - Target node ID.
   * @param method - Method name.
   * @param args - Positional arguments.
   * @param kwargs - Keyword arguments.
   * @returns The return value of the method call.
   */
  async invoke(
    nodeId: string,
    method: string,
    args: unknown[] = [],
    kwargs: Record<string, unknown> = {}
  ): Promise<unknown> {
    this.requireCapability("invoke", "host/invoke");

    const params: HostInvokeParams = {
      node_id: nodeId,
      method,
      args,
      kwargs,
    };

    const result = (await this.transport.sendRequest(
      Methods.HOST_INVOKE,
      params
    )) as HostInvokeResult;

    return result.result;
  }

  /**
   * Query available filesystem roots.
   *
   * Requires the `roots` host capability.
   *
   * @returns Array of available roots.
   */
  async queryRoots(): Promise<RootInfo[]> {
    this.requireCapability("roots", "host/query_roots");

    const result = (await this.transport.sendRequest(
      Methods.HOST_QUERY_ROOTS,
      {}
    )) as HostQueryRootsResult;

    return result.roots ?? [];
  }

  /**
   * Resolve a normalized path through a root to a real filesystem path.
   *
   * Requires the `roots` host capability.
   *
   * @param rootUri - Root URI to resolve against.
   * @param normalizedPath - Path relative to the root.
   * @returns The resolved path and existence flag.
   */
  async resolveRoot(
    rootUri: string,
    normalizedPath: string
  ): Promise<HostResolveRootResult> {
    this.requireCapability("roots", "host/resolve_root");

    const params: HostResolveRootParams = {
      root_uri: rootUri,
      normalized_path: normalizedPath,
    };

    const result = (await this.transport.sendRequest(
      Methods.HOST_RESOLVE_ROOT,
      params
    )) as HostResolveRootResult;

    return result;
  }

  // -----------------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------------

  private requireCapability(
    capability: keyof HostCapabilities,
    methodName: string
  ): void {
    const value = this.capabilities[capability];
    if (value === false) {
      throw new CAPError(
        ErrorCodes.PERMISSION_DENIED,
        `Host does not support ${methodName} (capability '${capability}' is false)`
      );
    }
  }
}
