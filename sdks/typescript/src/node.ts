/**
 * NodePlugin interface and BaseNode abstract class.
 *
 * Mirrors the protocol from src/activecontext/plugins/protocol.py.
 * Every node type in a CAP plugin server must satisfy the NodePlugin
 * interface. BaseNode provides default implementations for convenience.
 *
 * @module node
 */

import type {
  RenderSnapshot,
  TokenEstimate,
  Notification,
} from "./types";

// ---------------------------------------------------------------------------
// NodePlugin interface
// ---------------------------------------------------------------------------

/**
 * The contract that all CAP node types must satisfy.
 *
 * This mirrors the Python NodePlugin protocol. Any class that implements
 * these methods can be registered as a node type in a CAPServer.
 *
 * Key design principles:
 * - Render methods return composable sections (header, content, detail).
 *   The projection engine composes them based on the node's expansion state.
 * - tick() is the synchronous state materialization point. Async work
 *   happens internally; tick() makes it visible.
 * - toDict() must include all data needed to reconstruct the node.
 */
export interface NodePlugin {
  /** Node type identifier (e.g., "shell", "echo"). Stable after creation. */
  readonly nodeType: string;

  /** Unique node instance identifier. Stable for the lifetime of the node. */
  readonly nodeId: string;

  /**
   * Render the header section -- always shown regardless of expansion.
   * Should aim for ~50 tokens or fewer.
   *
   * @param cwd - Working directory for path resolution.
   * @returns The header string.
   */
  renderHeader(cwd: string): string;

  /**
   * Render the content section -- the main body of the node.
   * Shown at CONTENT expansion and above.
   *
   * @param cwd - Working directory for path resolution.
   * @returns The content string, or empty string if none.
   */
  renderContent(cwd: string): string;

  /**
   * Render the detail section -- full content including extras.
   * Shown at ALL expansion only.
   *
   * @param cwd - Working directory for path resolution.
   * @returns The detail string, or empty string if identical to content.
   */
  renderDetail(cwd: string): string;

  /**
   * Estimate token counts for each rendering level.
   *
   * @returns Object with collapsed, summary, and detail token estimates.
   */
  getTokenBreakdown(): TokenEstimate;

  /**
   * Return compact metadata for the handles dict.
   *
   * The projection engine includes digests as quick references for each
   * node. Typical fields: id, type, status, key metrics.
   *
   * @returns A plain object with digest metadata.
   */
  getDigest(): Record<string, unknown>;

  /**
   * Synchronous state materialization point.
   *
   * Called at tick boundaries for dirty nodes. This is where async work
   * becomes visible: check if internal state has changed, update public
   * fields, and optionally return notifications.
   *
   * May be synchronous or return a Promise.
   *
   * @returns Void or a list of notifications generated during tick.
   */
  tick(): void | Promise<void> | Notification[] | Promise<Notification[]>;

  /**
   * Serialize node state to a dictionary.
   *
   * Must include all fields needed to reconstruct the node. The dict
   * must include "node_type" for deserialization dispatch.
   *
   * @returns Serialized node state.
   */
  toDict(): Record<string, unknown>;

  /**
   * Handle a method call from the DSL or host.
   *
   * Called during node/sync (batched) or node/call (immediate). The node
   * should execute the method and return a result.
   *
   * @param method - Method name.
   * @param args - Positional arguments.
   * @param kwargs - Keyword arguments.
   * @returns The method's return value.
   */
  handleCall?(
    method: string,
    args: unknown[],
    kwargs: Record<string, unknown>
  ): unknown | Promise<unknown>;
}

// ---------------------------------------------------------------------------
// BaseNode abstract class
// ---------------------------------------------------------------------------

let _nodeIdCounter = 0;

/** Generate a unique node ID with a type prefix. */
function generateNodeId(nodeType: string): string {
  _nodeIdCounter += 1;
  const suffix = Math.random().toString(36).substring(2, 10);
  return `${nodeType}_${suffix}${_nodeIdCounter}`;
}

/**
 * Abstract base class implementing NodePlugin with sensible defaults.
 *
 * Subclasses must:
 * - Set `nodeType` (the type identifier string).
 * - Override render methods as needed.
 *
 * BaseNode provides:
 * - Automatic node ID generation.
 * - Default empty renders and zero-token estimates.
 * - A no-op tick() implementation.
 * - A basic toDict() that includes nodeType and nodeId.
 * - A dirty flag for tracking state changes.
 */
export abstract class BaseNode implements NodePlugin {
  abstract readonly nodeType: string;

  readonly nodeId: string;

  /** Whether the node's internal state has changed since last sync. */
  protected dirty = false;

  constructor(nodeId?: string) {
    this.nodeId = nodeId || generateNodeId(this.getNodeType());
  }

  /**
   * Get the node type. Override this if your nodeType is not available
   * during constructor time (for the ID generation).
   */
  protected getNodeType(): string {
    return "node";
  }

  renderHeader(_cwd: string): string {
    return `[${this.nodeType.toUpperCase()}] ${this.nodeId}`;
  }

  renderContent(_cwd: string): string {
    return "";
  }

  renderDetail(_cwd: string): string {
    return "";
  }

  getTokenBreakdown(): TokenEstimate {
    return { collapsed: 0, summary: 0, detail: 0 };
  }

  getDigest(): Record<string, unknown> {
    return {
      type: this.nodeType,
      id: this.nodeId,
    };
  }

  tick(): void | Notification[] {
    // Default: no-op. Subclasses override for async state materialization.
    return;
  }

  toDict(): Record<string, unknown> {
    return {
      node_type: this.nodeType,
      node_id: this.nodeId,
    };
  }

  handleCall?(
    method: string,
    args: unknown[],
    kwargs: Record<string, unknown>
  ): unknown | Promise<unknown>;

  /** Mark this node as dirty, triggering sync on the next tick. */
  protected markDirty(): void {
    this.dirty = true;
  }

  /** Clear the dirty flag after sync. */
  clearDirty(): void {
    this.dirty = false;
  }

  /** Check whether the node has unsynchronized changes. */
  isDirty(): boolean {
    return this.dirty;
  }

  /**
   * Build the full render snapshot for this node.
   *
   * @param cwd - Working directory.
   * @returns All three render levels.
   */
  getRenderSnapshot(cwd: string): RenderSnapshot {
    return {
      header: this.renderHeader(cwd),
      content: this.renderContent(cwd),
      detail: this.renderDetail(cwd),
    };
  }
}
