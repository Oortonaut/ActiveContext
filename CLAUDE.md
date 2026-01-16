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
├── session/              # Session management
│   ├── protocols.py      # Core protocols (SessionProtocol, TimelineProtocol, etc.)
│   ├── timeline.py       # Statement timeline with Python execution
│   └── session_manager.py# Session lifecycle management
├── transport/
│   ├── direct/           # Python library API (ActiveContext, AsyncSession)
│   └── acp/              # ACP adapter for Rider/Zed
├── core/                 # (Future) AgentLoop, ProjectionEngine
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
    await session.execute('v = view("main.py", tokens=2000)')

    # Or stream updates from a prompt
    async for update in session.prompt("v.SetLod(1)"):
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

## Key Protocols

- `SessionProtocol` - session lifecycle: prompt(), tick(), cancel()
- `TimelineProtocol` - statement execution: execute_statement(), replay_from()
- `SessionManagerProtocol` - multi-session management

## DSL for the LLM

```python
v = view("main.py", pos="0:0", tokens=2000)  # create view
v.SetLod(1).SetTokens(500)                    # adjust rendering
v.Run(freq="Sync")                            # enable auto-updates
g = group(v, tokens=300, lod=2)               # create summary group
```

## Key Design Invariants

1. **Tick boundary**: All automatic mutations occur at tick phases, never in background threads
2. **1:1 session-timeline**: Each session has exactly one statement timeline
3. **Groups are summaries**: A group is a summarized facade over its members
4. **LOD ladder**: lod=0 (raw), lod=1 (structured), lod=2 (summary), lod=3 (diff-only)
