# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ActiveContext** is an agent loop architecture where an LLM controls a structured, reversible working context through a Python statement timeline. The agent executes Python statements that manipulate "context objects" (views, groups) with tick-driven updates.

## Development Commands

```bash
uv sync --all-extras     # Install dependencies
uv run pytest            # Run tests
uv run pytest -xvs       # Run tests with verbose output, stop on first failure
uv run ruff check src/   # Lint
uv run ruff format src/  # Format
uv run mypy src/         # Type check
python -m activecontext  # Run ACP agent (stdio)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TRANSPORT LAYER                      │
│  ┌──────────────────┐    ┌────────────────────────┐    │
│  │  ACP Transport   │    │   Direct Transport     │    │
│  │  (JSON-RPC/stdio)│    │   (async Python API)   │    │
│  └────────┬─────────┘    └───────────┬────────────┘    │
└───────────┼──────────────────────────┼─────────────────┘
            │      SessionUpdate       │
            ▼                          ▼
┌─────────────────────────────────────────────────────────┐
│                    SESSION LAYER                        │
│  SessionManager → Session (1:1) → Timeline              │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                      CORE LAYER                         │
│  Timeline (StatementLog + PythonExec + ContextObjects)  │
└─────────────────────────────────────────────────────────┘
```

## Package Structure

```
src/activecontext/
├── __init__.py           # Public API exports
├── __main__.py           # ACP agent entry point
├── logging.py            # Logging configuration
├── config/               # Configuration management
│   ├── schema.py         # Config dataclasses (Config, LLMConfig, etc.)
│   ├── paths.py          # Platform-aware config path resolution
│   ├── loader.py         # YAML loading, env overrides, caching
│   ├── merge.py          # Deep merge algorithm
│   └── watcher.py        # Config file change monitoring
├── session/              # Session management
│   ├── protocols.py      # Core protocols (SessionProtocol, TimelineProtocol, etc.)
│   ├── timeline.py       # Statement timeline with Python execution
│   └── session_manager.py# Session lifecycle management
├── transport/
│   ├── direct/           # Python library API (ActiveContext, AsyncSession)
│   └── acp/              # ACP adapter for Rider/Zed
├── core/                 # AgentLoop, ProjectionEngine, LLM providers
├── context/              # (Future) Real ViewHandle, GroupHandle
└── events/               # (Future) EventBus, DeltaFormatter
```

## Usage

### Direct Transport (Python API)

```python
from activecontext import ActiveContext

async with ActiveContext() as ctx:
    session = await ctx.create_session(cwd=".")

    # Execute Python directly
    await session.execute('v = view("main.py", tokens=2000, state=NodeState.ALL)')

    # Or stream updates from a prompt
    async for update in session.prompt("v.SetState(NodeState.SUMMARY)"):
        print(update)

    # Access namespace and context objects
    print(session.get_namespace())
    print(session.get_context_objects())
```

### ACP Transport (IDE integration)

Configure in your IDE's `acp.json`:
```json
{
  "agent_servers": {
    "activecontext": {
      "command": "python",
      "args": ["-m", "activecontext"]
    }
  }
}
```

## Environment Variables

### LLM Configuration

At least one API key must be set for LLM functionality:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude models) |
| `OPENAI_API_KEY` | OpenAI API key (GPT models) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |

Models are auto-discovered from available API keys. The first available model becomes the default.

### Debugging

| Variable | Description |
|----------|-------------|
| `ACTIVECONTEXT_LOG` | File path for diagnostic logs. Logs all ACP messages with timestamps. |
| `ACTIVECONTEXT_DEBUG` | Set to any value to print projection contents to stderr. |

Example ACP config with logging:
```json
{
  "agent_servers": {
    "activecontext": {
      "command": "python",
      "args": ["-m", "activecontext"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-...",
        "ACTIVECONTEXT_LOG": "C:\\Users\\You\\activecontext.log"
      }
    }
  }
}
```

## Configuration Files

ActiveContext supports hierarchical YAML configuration files that merge together:

| Level | Windows | Unix |
|-------|---------|------|
| System | `%PROGRAMDATA%\activecontext\config.yaml` | `/etc/activecontext/config.yaml` |
| User | `%APPDATA%\activecontext\config.yaml` | `~/.ac/config.yaml` |
| Project | `$PROJECT_ROOT/.ac/config.yaml` | `$PROJECT_ROOT/.ac/config.yaml` |

Later levels override earlier ones. Environment variables override all config files.

### Example config.yaml

```yaml
llm:
  model: claude-sonnet-4-20250514
  temperature: 0.0
  max_tokens: 8192

session:
  default_mode: plan
  modes:
    - id: normal
      name: Normal
      description: Standard mode
    - id: plan
      name: Plan
      description: Plan before acting

projection:
  total_budget: 16000
  conversation_ratio: 0.25
  views_ratio: 0.55
  groups_ratio: 0.20

logging:
  level: INFO
  file: ~/activecontext.log
```

### Configuration API

```python
from activecontext import Config, load_config, get_config

# Load config with project-specific settings
config = load_config(session_root="/path/to/project")

# Access typed configuration
print(config.llm.model)
print(config.projection.total_budget)

# Get cached global config
config = get_config()
```

### Config File Watching

Config files are monitored for changes and automatically reloaded:

```python
from activecontext.config import start_watching, on_config_reload

# Start watching for changes
start_watching(session_root="/path/to/project")

# Register callback for config changes
def on_change(new_config):
    print("Config reloaded:", new_config.llm.model)

unregister = on_config_reload(on_change)
```

## Key Protocols

- `SessionProtocol` - session lifecycle: prompt(), tick(), cancel()
- `TimelineProtocol` - statement execution: execute_statement(), replay_from()
- `SessionManagerProtocol` - multi-session management

## DSL for the LLM

```python
from activecontext import NodeState, TickFrequency

v = view("main.py", pos="1:0", tokens=2000, state=NodeState.ALL)  # create view
v.SetState(NodeState.SUMMARY).SetTokens(500)                      # adjust rendering
v.Run(TickFrequency.turn())                                       # enable auto-updates
g = group(v, tokens=300, state=NodeState.SUMMARY)                 # create summary group
g = group(v.node_id, "other_id", summary="Summary text")          # accept node IDs and summary
```

## Key Design Invariants

1. **Tick boundary**: All automatic mutations occur at tick phases, never in background threads
2. **1:1 session-timeline**: Each session has exactly one statement timeline
3. **Groups are summaries**: A group is a summarized facade over its members
4. **State-based rendering**: NodeState controls visibility and detail level
   - HIDDEN: Not shown in projection (but still ticked)
   - COLLAPSED: Metadata only
   - SUMMARY: LLM-generated summary
   - DETAILS: Full view with child settings
   - ALL: Everything including diffs
