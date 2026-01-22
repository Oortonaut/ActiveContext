# ActiveContext Internal Architecture Diagrams

PlantUML diagrams documenting the internal architecture of ActiveContext - the context rendering and projection system.

## Diagrams

| File | Description |
|------|-------------|
| `01-projection-engine.puml` | How ProjectionEngine builds projections from the context graph |
| `02-context-graph.puml` | ContextGraph DAG structure and node types |
| `03-node-states.puml` | NodeState visibility model and rendering rules |

## Key Concepts

### Projection Engine

The `ProjectionEngine` transforms the context graph into a `Projection` that gets sent to the LLM:

1. **Collect Render Path** - Traverse graph respecting visibility rules
2. **Render Path** - Convert nodes to `ProjectionSection` strings
3. **Assemble Projection** - Bundle sections with handles for incremental updates

### Context Graph

A directed acyclic graph (DAG) of `ContextNode` objects:

- **TextNode** - File views with position tracking
- **GroupNode** - Summary facades over children
- **MessageNode** - Conversation messages
- **ShellNode** - Async shell commands
- **MCPServerNode** - MCP tool connections
- **TopicNode**, **ArtifactNode**, **WorkNode**, etc.

### NodeState Visibility

Controls how nodes appear in projections:

| State | Renders Self | Renders Children |
|-------|--------------|------------------|
| HIDDEN | No | No |
| COLLAPSED | Yes (~50 tokens) | No |
| SUMMARY | Yes (summary) | No |
| DETAILS | Yes (full) | Yes |
| ALL | Yes (summary + full) | Yes |

## Rendering

See [`../acp/README.md`](../acp/README.md) for PlantUML rendering instructions.

Quick start:
```bash
# VS Code: Install "PlantUML" extension, open .puml, press Alt+D
# CLI: plantuml *.puml
# Online: http://www.plantuml.com/plantuml/uml
```

## Related Code

- `src/activecontext/core/projection_engine.py` - ProjectionEngine
- `src/activecontext/context/graph.py` - ContextGraph
- `src/activecontext/context/nodes.py` - All node types
- `src/activecontext/context/state.py` - NodeState enum
- `src/activecontext/context/view.py` - AgentView rendering
- `src/activecontext/session/protocols.py` - Projection dataclass
