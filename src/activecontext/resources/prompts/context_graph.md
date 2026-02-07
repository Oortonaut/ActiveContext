# Context Graph (DAG)

The context graph is a Directed Acyclic Graph (DAG) that organizes context nodes.

## Structure

```
         ┌─────────┐
         │ GroupA  │  (root)
         └────┬────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐
│ View1 │ │ View2 │ │ View3 │
└───────┘ └───────┘ └───┬───┘
                        │
                        ▼
                   ┌─────────┐
                   │ GroupB  │  (shared child)
                   └─────────┘
```

## Key Properties

### Multiple Parents
A node can have multiple parents. This allows the same content to appear in different organizational structures.

```python
v = text("src/auth.py")
g1 = group(v)  # v is child of g1
g2 = group(v)  # v is also child of g2
```

### Change Notification Cascade
When a child node changes, it notifies all parents:

1. Child's `version` increments
2. Child calls `notify_parents(description)`
3. Each parent's `on_child_changed()` is called
4. Parents propagate upward to their parents

```
View changes → GroupA notified → Root notified
            → GroupB notified → ...
```

### Root Nodes
Nodes with no parents are "root nodes". The projection engine starts rendering from roots.

## Node Types

| Type | Purpose | Default Expansion |
|------|---------|-------------------|
| `TextNode` | File content | ALL |
| `GroupNode` | Summary of children | CONTENT |
| `TopicNode` | Conversation segment | ALL |
| `ArtifactNode` | Code/output/error | ALL |
| `ShellNode` | Command execution | ALL |
| `PtyNode` | Interactive PTY session | CONTENT |
| `SessionNode` | Session metadata | HEADER |
| `WorkNode` | Work coordination | ALL |
| `MessageNode` | Conversation message | ALL |
| `LockNode` | File lock for coordination | HEADER |
| `MCPServerNode` | MCP server connection | ALL |
| `MCPToolNode` | MCP tool definition | HEADER |
| `MarkdownNode` | Parsed markdown structure | ALL |
| `PluginManagerNode` | Plugin server tracking | HEADER |
| `TraceNode` | Statement execution trace | CONTENT |
| `StatementNode` | Executed statement | CONTENT |
| `HelpNode` | Help documentation | CONTENT |

## Manipulation

### Adding Nodes
```python
v = text("file.py")           # Creates and adds to graph
g = group(v)                  # Creates group with v as child
```

### Linking/Unlinking
```python
link(child, parent)           # Add edge (child first, parent second)
unlink(child, parent)         # Remove edge
```

Note: Argument order is `(child, parent)` - the child comes first.

### Checkpointing
Save and restore DAG structure (edges), not content:

```python
checkpoint("exploration")     # Save current structure
# ... make changes ...
restore("exploration")        # Restore structure
```

## Tick Phase

Each turn, the session "ticks":

1. Process pending shell results
2. For each running node:
   - Call `Recompute()` if frequency says so
   - Generate traces for changes
   - Notify parents
3. Build projection from roots

## Projection Rendering

The projection engine walks the graph from roots:

1. Collect visible nodes (not HIDDEN)
2. Allocate token budget
3. Render each node according to its state
4. Concatenate into final context

Groups render differently based on expansion:
- `HEADER`: Metadata only
- `CONTENT`: Show cached summary (or children if stale)
- `INDEX`: Summary plus child headers
- `ALL`: Render all children recursively
