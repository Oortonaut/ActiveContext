/**
 * Echo Plugin -- a minimal CAP plugin server example.
 *
 * Demonstrates the full CAP plugin API:
 * - Defining a node type (EchoNode) with BaseNode.
 * - Building a schema with the schema helpers.
 * - Registering the node type with a CAPServer.
 * - Handling method calls (set_text).
 * - Running the server main loop.
 *
 * Usage:
 *   npx ts-node examples/echo-plugin/index.ts
 *
 * Or configure in your ActiveContext config:
 *   plugins:
 *     - name: echo
 *       command: ["npx", "ts-node", "examples/echo-plugin/index.ts"]
 */

import {
  CAPServer,
  BaseNode,
  nodeTypeSchema,
  constructor_,
  required,
  optional,
  property,
  method,
} from "../../src/index";

import type { TokenEstimate, Notification } from "../../src/index";

// ---------------------------------------------------------------------------
// EchoNode
// ---------------------------------------------------------------------------

/**
 * A simple node that echoes input text.
 *
 * Demonstrates:
 * - Constructor with positional and keyword arguments.
 * - Rendering at three granularity levels.
 * - Method handling (set_text).
 * - Token estimation.
 * - State serialization.
 */
class EchoNode extends BaseNode {
  readonly nodeType = "echo";
  private text: string;
  private echoCount: number;
  private createdAt: string;

  constructor(text: string, echoCount = 1, nodeId?: string) {
    super(nodeId);
    this.text = text;
    this.echoCount = echoCount;
    this.createdAt = new Date().toISOString();
  }

  protected getNodeType(): string {
    return "echo";
  }

  renderHeader(_cwd: string): string {
    const preview = this.text.length > 30
      ? this.text.substring(0, 30) + "..."
      : this.text;
    return `[ECHO] "${preview}" (x${this.echoCount})`;
  }

  renderContent(_cwd: string): string {
    const lines: string[] = [];
    for (let i = 0; i < this.echoCount; i++) {
      lines.push(this.text);
    }
    return lines.join("\n");
  }

  renderDetail(_cwd: string): string {
    return [
      `Echo count: ${this.echoCount}`,
      `Text length: ${this.text.length} characters`,
      `Created: ${this.createdAt}`,
      `Node ID: ${this.nodeId}`,
    ].join("\n");
  }

  getTokenBreakdown(): TokenEstimate {
    // Rough estimate: ~4 chars per token
    const headerTokens = 15;
    const contentTokens = Math.ceil((this.text.length * this.echoCount) / 4);
    const detailTokens = 20;
    return {
      collapsed: headerTokens,
      summary: contentTokens,
      detail: detailTokens,
    };
  }

  getDigest(): Record<string, unknown> {
    return {
      type: "echo",
      id: this.nodeId,
      text_length: this.text.length,
      echo_count: this.echoCount,
    };
  }

  toDict(): Record<string, unknown> {
    return {
      node_type: this.nodeType,
      node_id: this.nodeId,
      text: this.text,
      echo_count: this.echoCount,
      created_at: this.createdAt,
    };
  }

  handleCall(
    methodName: string,
    args: unknown[],
    kwargs: Record<string, unknown>
  ): unknown {
    switch (methodName) {
      case "set_text": {
        const newText = (args[0] as string) ?? (kwargs.text as string) ?? "";
        this.text = newText;
        this.markDirty();
        return null;
      }
      case "set_count": {
        const newCount = (args[0] as number) ?? (kwargs.count as number) ?? 1;
        this.echoCount = Math.max(1, Math.floor(newCount));
        this.markDirty();
        return null;
      }
      case "get_stats": {
        return {
          text_length: this.text.length,
          echo_count: this.echoCount,
          total_chars: this.text.length * this.echoCount,
        };
      }
      default:
        throw new Error(`Unknown method: ${methodName}`);
    }
  }

  tick(): Notification[] {
    // No async work in this simple node.
    // If we had background tasks, we would check their status here.
    return [];
  }
}

// ---------------------------------------------------------------------------
// Schema definition
// ---------------------------------------------------------------------------

const echoSchema = nodeTypeSchema("echo", {
  description: "Echoes input text back, optionally repeated multiple times.",
  constructor: constructor_({
    positional: [
      required("text", "str", "The text to echo."),
    ],
    named: [
      optional("echo_count", 1, "int", "Number of times to repeat the text."),
    ],
  }),
  properties: [
    property("text", {
      type: "str",
      readable: true,
      writable: false,
      description: "The current echo text.",
    }),
    property("echo_count", {
      type: "int",
      readable: true,
      writable: false,
      description: "Number of times the text is repeated.",
    }),
  ],
  methods: [
    method("set_text", {
      params: [required("text", "str", "New text to echo.")],
      description: "Update the echoed text.",
    }),
    method("set_count", {
      params: [required("count", "int", "New echo count.")],
      description: "Update the number of repetitions.",
    }),
    method("get_stats", {
      returns: "dict",
      description: "Get statistics about the echo node.",
    }),
  ],
});

// ---------------------------------------------------------------------------
// Server setup
// ---------------------------------------------------------------------------

const server = new CAPServer({
  name: "echo-plugin",
  version: "0.1.0",
});

server.register(
  "echo",
  (args, kwargs, nodeId) => {
    const text = String(args[0] ?? "");
    const echoCount = (kwargs.echo_count as number) ?? 1;
    return new EchoNode(text, echoCount, nodeId);
  },
  echoSchema
);

// Start serving
server.serve().then(() => {
  process.stderr.write("[echo-plugin] Server stopped\n");
  process.exit(0);
});
