# @activecontext/cap-plugin

TypeScript SDK for building CAP (Context Application Protocol) plugin servers for ActiveContext.

## Quick Start

```bash
npm install @activecontext/cap-plugin
```

```typescript
import {
  CAPServer,
  BaseNode,
  nodeTypeSchema,
  constructor_,
  required,
  property,
} from "@activecontext/cap-plugin";

// 1. Define your node class
class GreetNode extends BaseNode {
  readonly nodeType = "greet";
  private name: string;

  constructor(name: string, nodeId?: string) {
    super(nodeId);
    this.name = name;
  }

  renderHeader(_cwd: string): string {
    return `[GREET] Hello, ${this.name}!`;
  }

  renderContent(_cwd: string): string {
    return `Greeting for: ${this.name}`;
  }

  toDict() {
    return { node_type: "greet", node_id: this.nodeId, name: this.name };
  }
}

// 2. Define the schema
const greetSchema = nodeTypeSchema("greet", {
  description: "A greeting node.",
  constructor: constructor_({
    positional: [required("name", "str", "Name to greet.")],
  }),
  properties: [
    property("name", { type: "str", description: "The greeted name." }),
  ],
});

// 3. Create the server, register, and serve
const server = new CAPServer({ name: "greet-plugin", version: "0.1.0" });
server.register("greet", (args) => new GreetNode(String(args[0])), greetSchema);
server.serve();
```

Configure in `.ac/config.yaml`:

```yaml
plugins:
  - name: greet
    command: ["node", "dist/index.js"]
```

Then in the ActiveContext DSL:

```python
g = greet("World")
# Renders: [GREET] Hello, World!
```

## API Overview

### CAPServer

The main entry point. Manages node type registration and the stdio JSON-RPC loop.

```typescript
const server = new CAPServer({
  name: "my-plugin",      // Required: server name
  version: "1.0.0",       // Optional: version string
  capabilities: {         // Optional: override defaults
    sync: true,
    immediate_call: false,
    push_dirty: true,
    push_notifications: true,
  },
});

// Register node types
server.register("my_node", factory, schema);

// Start the server (blocks until shutdown)
await server.serve();
```

### NodePlugin Interface

The contract every node type must satisfy:

```typescript
interface NodePlugin {
  readonly nodeType: string;
  readonly nodeId: string;

  renderHeader(cwd: string): string;
  renderContent(cwd: string): string;
  renderDetail(cwd: string): string;

  getTokenBreakdown(): TokenEstimate;
  getDigest(): Record<string, unknown>;
  tick(): void | Promise<void> | Notification[] | Promise<Notification[]>;
  toDict(): Record<string, unknown>;

  handleCall?(method: string, args: unknown[], kwargs: Record<string, unknown>): unknown;
}
```

### BaseNode

Abstract base class with default implementations:

```typescript
class MyNode extends BaseNode {
  readonly nodeType = "my_type";

  constructor(nodeId?: string) {
    super(nodeId); // Auto-generates ID if not provided
  }

  // Override render methods as needed
  renderHeader(cwd: string): string { ... }
  renderContent(cwd: string): string { ... }
  renderDetail(cwd: string): string { ... }

  // Override for method call support
  handleCall(method: string, args: unknown[], kwargs: Record<string, unknown>) { ... }
}
```

### Schema Helpers

Declarative builders for node type schemas:

```typescript
import {
  nodeTypeSchema, constructor_, param, required, optional, property, method
} from "@activecontext/cap-plugin";

const schema = nodeTypeSchema("my_node", {
  description: "What this node does.",
  constructor: constructor_({
    positional: [required("path", "str", "File path.")],
    variadic: param("args", { type: "str" }),
    named: [
      optional("timeout", 30, "int", "Timeout in seconds."),
      optional("verbose", false, "bool", "Enable verbose output."),
    ],
  }),
  properties: [
    property("status", { type: "str", readable: true, description: "Current status." }),
    property("output", { type: "str", readable: true, writable: false }),
  ],
  methods: [
    method("cancel", { description: "Cancel the operation." }),
    method("SetTokens", {
      params: [required("tokens", "int")],
      returns: "self",
      chainable: true,
    }),
  ],
});
```

### Host API

Server-to-host requests for interacting with the host context graph:

```typescript
// Available after initialization via server.getHostApi()
const api = server.getHostApi();

// Create a child node in the host graph
const childId = await api.createNode("artifact", ["content"], {
  artifact_type: "output",
}, parentNodeId);

// Invoke a method on a host node
await api.invoke(targetNodeId, "set_content", ["new content"]);

// Query filesystem roots
const roots = await api.queryRoots();

// Resolve a path through a root
const resolved = await api.resolveRoot(roots[0].uri, "src/main.ts");
console.log(resolved.real_path, resolved.exists);
```

### Push Notifications

Notify the host of state changes outside the tick cycle:

```typescript
// Mark a node as dirty (triggers sync on next tick)
server.notifyDirty(node.nodeId);

// Send a notification to the node's parent chain
server.notifyNode(node.nodeId, "Processing complete", "wake");
```

## Node Lifecycle

1. **Registration**: Call `server.register()` with a node type, factory, and schema before `serve()`.

2. **Initialize**: The host sends `initialize` with its capabilities and roots. The server responds with its capabilities and all registered node type schemas.

3. **Create**: The host sends `node/create` with constructor arguments. The server's factory creates a node instance and returns initial state, renders, tokens, and digest.

4. **Sync (tick cycle)**: Each tick, the host sends `node/sync` for dirty nodes. The server processes batched method calls, ticks the node, and returns updated state. This is the hot path -- at most one round trip per dirty node per tick.

5. **Call (immediate)**: If the server declares `immediate_call: true`, the host can send `node/call` for urgent operations outside the tick cycle.

6. **Destroy**: The host sends `node/destroy` when a node is removed from the graph. The server releases all resources.

7. **Shutdown**: The host sends a `shutdown` notification. The server completes in-flight work and exits.

## Schema Definition Guide

Schemas describe your node types to the host. They enable:

- **DSL constructor generation**: The host creates DSL functions from your schema.
- **Type validation**: Arguments are checked against the schema.
- **LLM documentation**: The schema is included in the agent's prompt.
- **Autocompletion**: IDEs can use the schema for suggestions.

### Constructor Schema

Mirrors Python calling convention:

```typescript
constructor_({
  // Required positional args (in order)
  positional: [
    required("command", "str", "The command to run."),
    required("path", "str", "Target file path."),
  ],

  // Variadic (*args) -- at most one
  variadic: param("extra_args", { type: "str", description: "Additional arguments." }),

  // Keyword-only args (with defaults)
  named: [
    optional("timeout", 120, "int", "Timeout in seconds."),
    optional("cwd", null, "str", "Working directory."),
  ],
});
```

This generates a DSL constructor like:
```python
my_node("git", "src/", "status", timeout=60, cwd="/project")
```

### Parameter Types

Type hints use Python-style strings: `"str"`, `"int"`, `"float"`, `"bool"`, `"list[str]"`, `"dict"`, `"Any"`.

### Required vs Optional

- **Required**: No `default` field in the schema. The parameter must be provided.
- **Optional**: Has a `default` field. Can be `null` for nullable parameters.

```typescript
required("path", "str")           // Must be provided
optional("timeout", 120, "int")   // Defaults to 120
optional("label", null, "str")    // Defaults to null
```

### Properties

Expose node state to the DSL:

```typescript
property("is_complete", {
  type: "bool",
  readable: true,   // node.is_complete works
  writable: false,  // node.is_complete = x raises error
  description: "Whether the operation finished.",
})
```

### Methods

Expose callable operations:

```typescript
method("cancel", {
  description: "Cancel the running operation.",
})

method("SetTokens", {
  params: [required("tokens", "int", "Token budget.")],
  returns: "self",
  chainable: true,  // Enables: node.SetTokens(500).SetState(...)
  description: "Set the token budget.",
})
```

## Rendering Levels

Nodes render at three granularity levels. The projection engine composes them based on the node's expansion state:

| Level | Method | When Shown | Purpose |
|-------|--------|------------|---------|
| **Header** | `renderHeader(cwd)` | Always | Type badge, name, status. Aim for ~50 tokens. |
| **Content** | `renderContent(cwd)` | CONTENT, ALL | Main body: summaries, output preview, data. |
| **Detail** | `renderDetail(cwd)` | ALL only | Full content: complete output, timing, traces. |

The host never calls render methods directly over the wire. Instead, renders are returned as part of `node/create` and `node/sync` responses, cached on the host side.

## Token Estimation

Return approximate token counts for each level:

```typescript
getTokenBreakdown(): TokenEstimate {
  return {
    collapsed: 15,   // Tokens for header
    summary: 200,    // Additional tokens for content
    detail: 500,     // Additional tokens for detail
  };
}
```

These estimates drive the projection engine's token budgeting. They do not need to be exact -- rough estimates from content length (approximately 4 characters per token) are sufficient.

## Error Handling

The SDK uses standard JSON-RPC 2.0 error codes plus CAP-specific codes:

| Code | Name | When |
|------|------|------|
| -32000 | NODE_NOT_FOUND | Node ID does not exist |
| -32001 | NODE_TYPE_UNKNOWN | Unregistered node type |
| -32002 | METHOD_NOT_AVAILABLE | Method not defined or capability missing |
| -32003 | ROOT_NOT_FOUND | Unknown root URI |
| -32004 | PERMISSION_DENIED | Capability not declared |
| -32005 | NODE_CREATE_FAILED | Factory threw an error |

Throw `CAPError` from factories or `handleCall` for typed error responses:

```typescript
import { CAPError, ErrorCodes } from "@activecontext/cap-plugin";

handleCall(method: string, args: unknown[]) {
  if (method === "unknown") {
    throw new CAPError(ErrorCodes.METHOD_NOT_AVAILABLE, `Unknown method: ${method}`);
  }
}
```

## Examples

See `examples/echo-plugin/` for a complete working example with:

- Node class with all render methods.
- Schema with constructor, properties, and methods.
- Method call handling.
- Token estimation.
- State serialization.
