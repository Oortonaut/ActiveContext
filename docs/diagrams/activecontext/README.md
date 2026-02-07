# ActiveContext Internal Architecture Diagrams

PlantUML diagrams documenting the internal architecture of ActiveContext - the context rendering and projection system.

## Diagrams

| File | Description |
|------|-------------|
| `01-projection-engine.puml` | How ProjectionEngine builds projections from the context graph |
| `02-context-graph.puml` | ContextGraph DAG structure and node types |
| `03-node-states.puml` | Expansion and visibility model |
| `04-ownership-structure.puml` | Session/Timeline/Manager ownership structure |
| `05-node-view-architecture.puml` | NodeView layer and progression views |

## Key Concepts

### Projection Engine

The `ProjectionEngine` transforms the context graph into a `Projection` that gets sent to the LLM:

1. **Collect Render Path** - Traverse graph respecting visibility rules
2. **Render Path** - Convert nodes to `ProjectionSection` strings via NodeView
3. **Assemble Projection** - Bundle sections with handles for incremental updates

### Context Graph

A directed acyclic graph (DAG) of `ContextNode` objects in `context/nodes/`:

- **TextNode** - File views with position tracking
- **GroupNode** - Summary facades over children
- **MessageNode** - Conversation messages
- **ShellNode** - Async shell commands
- **PtyNode** - Interactive PTY sessions
- **MCPServerNode** - MCP tool connections
- **LockNode** - File lock coordination
- **PluginManagerNode** - Plugin server tracking
- **TraceNode** - Statement execution traces
- **TopicNode**, **ArtifactNode**, **WorkNode**, etc.

### NodeView Layer

DSL functions return `NodeView` wrappers, not raw `ContextNode`:

- **NodeView owns**: `hidden`, `expansion`, `notifications`
- **ContextNode owns**: `default_expansion`, `content`, `mode`, `tick_frequency`

Progression views provide structured workflow patterns:
- **ChoiceView** - Dropdown-like selection
- **SequenceView** - Ordered steps
- **LoopView** - Iterative refinement
- **StateView** - State machine navigation

### Expansion and Visibility

**Expansion** (on NodeView) controls rendering detail:

| Expansion | Renders Self | Renders Children |
|-----------|--------------|------------------|
| HEADER | Yes (~50 tokens) | No |
| CONTENT | Yes (content) | No |
| INDEX | Yes (content) | Headers only |
| ALL | Yes (full) | Yes |

**Visibility** (on NodeView) controls projection inclusion:
- `view.hidden = True` - Node excluded from projection (still ticks)
- `view.hidden = False` - Node included in projection

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
- `src/activecontext/context/nodes/` - Node type package
- `src/activecontext/context/state.py` - Expansion, TickFrequency enums
- `src/activecontext/context/view.py` - NodeView and progression views
- `src/activecontext/session/protocols.py` - Projection dataclass
