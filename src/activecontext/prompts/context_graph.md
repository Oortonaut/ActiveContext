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

| Type | Purpose | Typical State |
|------|---------|---------------|
| `TextNode` | File content | DETAILS |
| `GroupNode` | Summary of children | SUMMARY |
| `TopicNode` | Conversation segment | DETAILS |
| `ArtifactNode` | Code/output/error | DETAILS |
| `ShellNode` | Command execution | DETAILS |
| `SessionNode` | Session metadata | COLLAPSED |
| `WorkNode` | Work coordination | DETAILS |
| `MessageNode` | Conversation message | DETAILS |
| `LockNode` | File lock for coordination | COLLAPSED |
| `MCPServerNode` | MCP server connection | DETAILS |
| `MCPToolNode` | MCP tool definition | COLLAPSED |
| `MarkdownNode` | Parsed markdown structure | DETAILS |

## Manipulation

### Adding Nodes
```python
v = text("file.py")           # Creates and adds to graph
g = group(v)                  # Creates group with v as child
```

### Linking/Unlinking
```python
link(parent, child)           # Add edge
unlink(parent, child)         # Remove edge
```

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

Groups render differently based on state:
- `SUMMARY`: Show cached summary (or children if stale)
- `DETAILS`/`ALL`: Render all children
