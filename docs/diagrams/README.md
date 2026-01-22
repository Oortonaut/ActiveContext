# ACP Protocol PlantUML Diagrams

This directory contains PlantUML diagrams documenting the Agent Client Protocol (ACP).

## Diagrams

| File | Description |
|------|-------------|
| `01-architecture.puml` | High-level architecture overview |
| `02-message-flow.puml` | Complete message flow sequence |
| `03-initialization.puml` | Initialization handshake with JSON examples |
| `04-session-lifecycle.puml` | Session state machine |
| `05-prompt-turn.puml` | Detailed prompt turn lifecycle |
| `06-permissions.puml` | Permission request flow |
| `07-terminal.puml` | Terminal operations |
| `08-filesystem.puml` | File system operations |
| `09-session-modes.puml` | Session mode switching |
| `10-tool-call-states.puml` | Tool call status lifecycle |
| `11-content-blocks.puml` | Content block types |
| `12-plan-updates.puml` | Agent plan communication |
| `13-message-types.puml` | Complete message type reference |

## Rendering Diagrams

### Using the Render Script (Recommended)

The project includes a Python script that uses the public PlantUML server:

```bash
# Render all diagrams to PNG and SVG
uv run python docs/diagrams/render.py

# PNG only
uv run python docs/diagrams/render.py --png

# SVG only  
uv run python docs/diagrams/render.py --svg

# Render specific directory
uv run python docs/diagrams/render.py acp/
uv run python docs/diagrams/render.py activecontext/

# Render single file
uv run python docs/diagrams/render.py activecontext/01-projection-engine.puml

# Clean generated images
uv run python docs/diagrams/render.py --clean
```

### VS Code (Preview Only)

1. Install "PlantUML" extension by jebbs
2. Open any `.puml` file
3. Press `Alt+D` to preview

### Online

- http://www.plantuml.com/plantuml/uml
- https://kroki.io/