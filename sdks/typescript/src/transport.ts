/**
 * Stdio transport for CAP JSON-RPC communication.
 *
 * Reads newline-delimited JSON from stdin, writes to stdout.
 * Implements the reference stdio binding from the CAP transport spec.
 *
 * @module transport
 */

import * as readline from "readline";
import type {
  JsonRpcRequest,
  JsonRpcNotification,
  JsonRpcResponse,
  JsonRpcErrorResponse,
  JsonRpcMessage,
} from "./types";

// ---------------------------------------------------------------------------
// Message classification
// ---------------------------------------------------------------------------

/** Classify an incoming JSON-RPC message by its shape. */
export type IncomingMessage =
  | { kind: "request"; message: JsonRpcRequest }
  | { kind: "notification"; message: JsonRpcNotification }
  | { kind: "response"; message: JsonRpcResponse }
  | { kind: "error"; message: JsonRpcErrorResponse };

function classifyMessage(raw: Record<string, unknown>): IncomingMessage | null {
  const hasId = "id" in raw && raw.id !== undefined;
  const hasMethod = "method" in raw && typeof raw.method === "string";
  const hasResult = "result" in raw;
  const hasError = "error" in raw;

  if (hasMethod && hasId) {
    return { kind: "request", message: raw as unknown as JsonRpcRequest };
  }
  if (hasMethod && !hasId) {
    return { kind: "notification", message: raw as unknown as JsonRpcNotification };
  }
  if (hasResult && hasId) {
    return { kind: "response", message: raw as unknown as JsonRpcResponse };
  }
  if (hasError && hasId) {
    return { kind: "error", message: raw as unknown as JsonRpcErrorResponse };
  }

  return null;
}

// ---------------------------------------------------------------------------
// Pending request tracking
// ---------------------------------------------------------------------------

interface PendingRequest {
  resolve: (result: unknown) => void;
  reject: (error: Error) => void;
  timer?: ReturnType<typeof setTimeout>;
}

// ---------------------------------------------------------------------------
// StdioTransport
// ---------------------------------------------------------------------------

/** Handler for incoming JSON-RPC requests from the host. */
export type RequestHandler = (
  method: string,
  params: unknown,
  id: number | string
) => Promise<unknown>;

/** Handler for incoming JSON-RPC notifications from the host. */
export type NotificationHandler = (method: string, params: unknown) => void;

/**
 * StdioTransport reads JSON-RPC messages from stdin and writes to stdout.
 *
 * Supports:
 * - Sending requests with response tracking (via pending futures).
 * - Sending notifications (no response expected).
 * - Sending responses to incoming requests.
 * - Dispatching incoming messages to registered handlers.
 *
 * Framing: newline-delimited JSON (NDJSON). Each message is a single
 * compact JSON object followed by `\n`.
 */
export class StdioTransport {
  private rl: readline.Interface | null = null;
  private nextId = 1;
  private pending = new Map<number | string, PendingRequest>();
  private running = false;
  private requestHandler: RequestHandler | null = null;
  private notificationHandler: NotificationHandler | null = null;
  private readonly defaultTimeout: number;

  /**
   * @param defaultTimeout - Default timeout in milliseconds for requests (default: 30000).
   */
  constructor(defaultTimeout = 30000) {
    this.defaultTimeout = defaultTimeout;
  }

  /** Whether the transport is actively reading from stdin. */
  get isRunning(): boolean {
    return this.running;
  }

  /** Register a handler for incoming requests from the host. */
  onRequest(handler: RequestHandler): void {
    this.requestHandler = handler;
  }

  /** Register a handler for incoming notifications from the host. */
  onNotification(handler: NotificationHandler): void {
    this.notificationHandler = handler;
  }

  /**
   * Start reading from stdin.
   *
   * This begins the read loop. Messages are dispatched to registered
   * handlers as they arrive.
   */
  start(): void {
    if (this.running) return;
    this.running = true;

    this.rl = readline.createInterface({
      input: process.stdin,
      crlfDelay: Infinity,
    });

    this.rl.on("line", (line: string) => {
      this.handleLine(line);
    });

    this.rl.on("close", () => {
      this.handleClose();
    });
  }

  /**
   * Stop the transport and release resources.
   *
   * Fails all pending requests with a transport error. Idempotent.
   */
  stop(): void {
    if (!this.running) return;
    this.running = false;

    if (this.rl) {
      this.rl.close();
      this.rl = null;
    }

    // Fail all pending requests
    for (const [id, pending] of this.pending) {
      if (pending.timer) clearTimeout(pending.timer);
      pending.reject(new Error(`Transport stopped, request ${id} abandoned`));
    }
    this.pending.clear();
  }

  /**
   * Send a JSON-RPC request and wait for the response.
   *
   * @param method - The RPC method name.
   * @param params - Parameters object.
   * @param timeout - Timeout in milliseconds (default: transport default).
   * @returns The result from the response.
   * @throws Error if the request times out or the transport is closed.
   */
  sendRequest(
    method: string,
    params?: unknown,
    timeout?: number
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (!this.running) {
        reject(new Error("Transport is not running"));
        return;
      }

      const id = this.nextId++;
      const timeoutMs = timeout ?? this.defaultTimeout;

      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Request ${method} (id=${id}) timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timer });

      const message: JsonRpcRequest = {
        jsonrpc: "2.0",
        method,
        id,
        ...(params !== undefined ? { params } : {}),
      };

      this.writeLine(message);
    });
  }

  /**
   * Send a JSON-RPC notification (no response expected).
   *
   * @param method - The RPC method name.
   * @param params - Parameters object.
   */
  sendNotification(method: string, params?: unknown): void {
    if (!this.running) return;

    const message: JsonRpcNotification = {
      jsonrpc: "2.0",
      method,
      ...(params !== undefined ? { params } : {}),
    };

    this.writeLine(message);
  }

  /**
   * Send a JSON-RPC success response.
   *
   * @param result - The result value.
   * @param id - The request ID this responds to.
   */
  sendResponse(result: unknown, id: number | string): void {
    if (!this.running) return;

    const message: JsonRpcResponse = {
      jsonrpc: "2.0",
      id,
      result,
    };

    this.writeLine(message);
  }

  /**
   * Send a JSON-RPC error response.
   *
   * @param code - Error code (from ErrorCodes).
   * @param message - Human-readable error description.
   * @param id - The request ID this responds to (null for parse errors).
   * @param data - Optional additional error data.
   */
  sendErrorResponse(
    code: number,
    message: string,
    id: number | string | null,
    data?: unknown
  ): void {
    if (!this.running) return;

    const errorResp: JsonRpcErrorResponse = {
      jsonrpc: "2.0",
      id: id as number | string,
      error: {
        code,
        message,
        ...(data !== undefined ? { data } : {}),
      },
    };

    this.writeLine(errorResp);
  }

  // -----------------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------------

  private writeLine(message: JsonRpcMessage): void {
    const json = JSON.stringify(message);
    process.stdout.write(json + "\n");
  }

  private handleLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed) return; // Ignore empty lines per spec

    let raw: Record<string, unknown>;
    try {
      raw = JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
      // Malformed JSON -- log to stderr, do not crash
      process.stderr.write(`[CAP] Invalid JSON on stdin: ${trimmed.substring(0, 100)}\n`);
      return;
    }

    const classified = classifyMessage(raw);
    if (!classified) {
      process.stderr.write(`[CAP] Unclassifiable message: ${trimmed.substring(0, 100)}\n`);
      return;
    }

    switch (classified.kind) {
      case "request":
        this.handleRequest(classified.message);
        break;
      case "notification":
        this.handleNotification(classified.message);
        break;
      case "response":
        this.handleResponse(classified.message);
        break;
      case "error":
        this.handleErrorResponse(classified.message);
        break;
    }
  }

  private handleRequest(msg: JsonRpcRequest): void {
    if (!this.requestHandler) {
      this.sendErrorResponse(
        -32601,
        `No handler for method: ${msg.method}`,
        msg.id
      );
      return;
    }

    this.requestHandler(msg.method, msg.params, msg.id)
      .then((result) => {
        this.sendResponse(result, msg.id);
      })
      .catch((err: Error) => {
        this.sendErrorResponse(
          -32603,
          err.message || "Internal error",
          msg.id
        );
      });
  }

  private handleNotification(msg: JsonRpcNotification): void {
    if (this.notificationHandler) {
      this.notificationHandler(msg.method, msg.params);
    }
  }

  private handleResponse(msg: JsonRpcResponse): void {
    const pending = this.pending.get(msg.id);
    if (!pending) return;

    this.pending.delete(msg.id);
    if (pending.timer) clearTimeout(pending.timer);
    pending.resolve(msg.result);
  }

  private handleErrorResponse(msg: JsonRpcErrorResponse): void {
    const pending = this.pending.get(msg.id);
    if (!pending) return;

    this.pending.delete(msg.id);
    if (pending.timer) clearTimeout(pending.timer);

    const err = new Error(
      `JSON-RPC error ${msg.error.code}: ${msg.error.message}`
    );
    (err as unknown as Record<string, unknown>).code = msg.error.code;
    (err as unknown as Record<string, unknown>).data = msg.error.data;
    pending.reject(err);
  }

  private handleClose(): void {
    // stdin closed (EOF) -- treat as shutdown
    this.stop();
  }
}
