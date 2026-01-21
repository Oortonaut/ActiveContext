# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ActiveContext** is an agent loop architecture where an LLM controls a structured, reversible working context through a Python statement timeline. The agent executes Python statements that manipulate "context objects" (views, groups, shells, MCP connections) with tick-driven updates.

## Working Directory

The cwd is `C:\ActiveContext`. Do not use path prefixes in commands:

```bash
# ✅ Correct
git status
uv run pytest

# ❌ Wrong - unnecessary path
git -C C:\ActiveContext status
cd C:\ActiveContext && uv run pytest
pushd C:\ActiveContext && uv run pytest
```

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
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│                      CORE LAYER                         │
│  ContextGraph │ ProjectionEngine │ PermissionManagers   │
└─────────────────────────────────────────────────────────┘
```

## Package Structure

```
src/activecontext/
├── __init__.py           # Public API exports
├── __main__.py           # ACP agent entry point
├── logging.py            # Logging configuration
├── clean.py              # Build artifact cleanup
├── config/               # Configuration management
│   ├── schema.py         # Config dataclasses (Config, LLMConfig, SandboxConfig, etc.)
│   ├── paths.py          # Platform-aware config path resolution
│   ├── loader.py         # YAML loading, env overrides, caching
│   ├── merge.py          # Deep merge algorithm
│   ├── secrets.py        # Secret management (fetch_secret)
│   └── watcher.py        # Config file change monitoring
├── session/              # Session management
│   ├── protocols.py      # Core protocols (SessionProtocol, TimelineProtocol, etc.)
│   ├── timeline.py       # Statement timeline with Python execution
│   ├── session_manager.py# Session lifecycle management
│   ├── permissions.py    # File/shell/import/web permission system
│   ├── storage.py        # Session persistence
│   └── xml_parser.py     # XML tag parsing for LLM output
├── context/              # Context graph and node types
│   ├── graph.py          # ContextGraph DAG with checkpointing
│   ├── nodes.py          # All node types (View, Group, Shell, MCP, etc.)
│   ├── checkpoint.py     # Checkpoint/GroupState for DAG snapshots
│   └── state.py          # NodeState enum
├── coordination/         # Multi-agent work coordination
│   ├── schema.py         # WorkEntry, FileAccess, Conflict dataclasses
│   └── scratchpad.py     # ScratchpadManager for file-based coordination
├── terminal/             # Shell execution subsystem
│   ├── protocol.py       # TerminalExecutor protocol
│   ├── subprocess_executor.py  # Local subprocess implementation
│   ├── acp_executor.py   # ACP-delegated execution
│   └── result.py         # ShellResult dataclass
├── mcp/                  # MCP client subsystem
│   ├── client.py         # MCPClientManager
│   ├── transport.py      # stdio/SSE transports
│   ├── types.py          # MCPToolInfo, MCPToolResult, etc.
│   └── permissions.py    # MCP permission integration
├── prompts/              # LLM reference prompts
│   ├── system.md         # System prompt template
│   ├── dsl_reference.md  # DSL function documentation
│   ├── node_states.md    # NodeState documentation
│   ├── context_graph.md  # DAG manipulation guide
│   ├── context_guide.md  # Context management guide
│   ├── work_coordination.md  # Multi-agent coordination guide
│   └── mcp.md            # MCP usage guide
├── dashboard/            # Web monitoring interface
│   ├── server.py         # Dashboard server (start/stop)
│   ├── routes.py         # HTTP API routes
│   ├── websocket.py      # Real-time updates via WebSocket
│   └── static/           # HTML/CSS/JS frontend
├── agents/               # Child agent management
│   ├── manager.py        # AgentManager (spawn, pause, terminate)
│   ├── schema.py         # AgentEntry, AgentMessage, AgentType
│   ├── registry.py       # Agent type registry
│   └── handle.py         # Agent handle for parent interaction
├── transport/
│   ├── direct/           # Python library API (ActiveContext, AsyncSession)
│   └── acp/              # ACP adapter for Rider/Zed
├── core/                 # ProjectionEngine, prompts
│   └── llm/              # LLM provider subsystem
│       ├── provider.py   # LLMProvider protocol
│       ├── litellm_provider.py  # LiteLLM implementation
│       ├── discovery.py  # Model auto-discovery from env
│       └── providers.py  # Provider registry
└── events/               # EventBus, DeltaFormatter
```

## Usage

### Direct Transport (Python API)

```python
from activecontext import ActiveContext

async with ActiveContext() as ctx:
    session = await ctx.create_session(cwd=".")

    # Execute Python directly
    await session.execute('v = text("main.py", tokens=2000, state=NodeState.ALL)')

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
  role: coding              # Last selected role
  provider: anthropic       # Last selected provider
  role_providers:           # Per-role provider preferences (optional)
    - role: fast
      provider: openai
      model: gpt-5-mini     # Optional model override
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

logging:
  level: INFO
  file: ~/activecontext.log

user:
  display_name: Ace         # Shown in message history

sandbox:
  allow_cwd: true           # Auto-grant read access to cwd
  allow_cwd_write: false    # Auto-grant write access to cwd
  deny_by_default: true     # Deny unlisted paths
  file_permissions:
    - pattern: "src/**/*.py"
      access: read_write
    - pattern: "*.md"
      access: read
  shell_deny_by_default: true
  shell_permissions:
    - pattern: "git *"
      allow: true
    - pattern: "pytest *"
      allow: true
  imports:
    allowed: [json, pathlib, re]

mcp:
  allow_dynamic_servers: true
  servers:
    - name: filesystem
      command: ["npx", "-y", "@modelcontextprotocol/server-filesystem"]
      args: ["/home/user/allowed"]
      connect: auto
```

### Configuration API

```python
from activecontext import Config, load_config, get_config

# Load config with project-specific settings
config = load_config(session_root="/path/to/project")

# Access typed configuration
print(config.llm.role)
print(config.projection.total_budget)
print(config.user.display_name)

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
    print("Config reloaded:", new_config.llm.role)

unregister = on_config_reload(on_change)
```

## Key Protocols

- `SessionProtocol` - session lifecycle: prompt(), tick(), cancel()
- `TimelineProtocol` - statement execution: execute_statement(), replay_from()
- `SessionManagerProtocol` - multi-session management
- `TerminalExecutor` - shell command execution

## Context Node Types

The context graph supports multiple node types:

| Node Type | Purpose |
|-----------|---------|
| `TextNode` | File or region view with position tracking |
| `GroupNode` | Summary facade over multiple nodes |
| `TopicNode` | Conversation topic/thread marker |
| `ArtifactNode` | Code snippets, outputs, errors |
| `ShellNode` | Async shell command with status tracking |
| `LockNode` | File lock for coordination |
| `SessionNode` | Session metadata and statistics |
| `MessageNode` | Conversation message with role/content |
| `WorkNode` | Multi-agent work registration |
| `MCPServerNode` | MCP server connection with tools |
| `MarkdownNode` | Structured markdown with heading hierarchy |

## Python DSL Reference

See `src/activecontext/prompts/dsl_reference.md` for complete documentation.

### Core Functions

```python
from activecontext import NodeState, TickFrequency

# File views
v = text("main.py", pos="1:0", tokens=2000, state=NodeState.ALL)
v.SetState(NodeState.SUMMARY).SetTokens(500)
v.Run(TickFrequency.turn())

# Groups
g = group(v1, v2, state=NodeState.SUMMARY)
g = group("node_id_1", "node_id_2", summary="Auth module overview")

# Topics and artifacts
t = topic("Authentication Implementation")
a = artifact("def foo(): pass", artifact_type="code", language="python")

# DAG manipulation
link(group_node, view_node)
unlink(group_node, view_node)

# Checkpointing
checkpoint("before_refactor")
restore("before_refactor")
branch("before_refactor", "attempt_2")
```

### Shell Execution

```python
s = shell("pytest", "-v", "tests/", timeout=120)
# s.is_complete, s.is_success, s.exit_code, s.output
wait(s)  # Wait for completion
```

### MCP Integration

```python
fs = mcp_connect("filesystem")                    # From config
gh = mcp_connect("gh", command=["npx", "-y", "@mcp/server-github"])
result = fs.read_file(path="/home/user/data.txt")
mcp_disconnect("filesystem")
```

### Work Coordination

```python
work_on("Implementing OAuth", "src/auth/oauth.py", "src/auth/config.py")
conflicts = work_check("src/shared.py")
work_update(intent="OAuth: Adding token refresh")
work_done()
```

### Agent Control

```python
done("Refactoring complete")
wait(shell_node1, shell_node2)      # Wait for all
wait_any(s1, s2, s3)                # Wait for first
```

## Key Design Invariants

1. **Tick boundary**: All automatic mutations occur at tick phases, never in background threads
2. **1:1 session-timeline**: Each session has exactly one statement timeline
3. **DAG structure**: Context nodes form a directed acyclic graph with parent-child relationships
4. **Groups are summaries**: A group is a summarized facade over its members
5. **State-based rendering**: NodeState controls visibility and detail level
   - HIDDEN: Not shown in projection (but still ticked)
   - COLLAPSED: Metadata only, aim for 50 or fewer tokens of content
   - SUMMARY: Agent-generated summary
   - DETAILS: Full view with child settings
   - ALL: Everything including traces, SUMMARY union DETAILS
6. **Permission boundaries**: File, shell, import, and web access require explicit grants

## Claude Code Permission Matching

The `.claude/settings.local.json` file uses prefix matching with the `:*` pattern. Commands must be phrased consistently to match the allowlist.

### Command Phrasing Guidelines

**✅ DO** - Commands that match the allowlist:

```bash
# uv commands - use space after command
uv sync --all-extras
uv run pytest -xvs
uv run ruff check src/

# git commands - use space after subcommand
git status
git diff --stat
git log --oneline -5

# Python module execution - use -m flag consistently
python -m activecontext
.venv/Scripts/python.exe -m activecontext --help

# Utilities - use command as prefix
ls -la
dir /b
grep -r "pattern"
cat file.txt
```

**❌ DON'T** - Commands that will be blocked:

```bash
# Missing space after command (won't match "git status:*")
git-status

# Wrong module invocation format (not pre-authorized)
python activecontext.py
python3 -m activecontext

# Unapproved git operations
git push --force
git reset --hard
git rebase -i

# Direct python without uv run (unless using -m flag)
python script.py
python3 test.py
```

### Permission Pattern Reference

| Pattern | Matches | Example |
|---------|---------|---------|
| `Bash(git status)` | Exact match only | `git status` |
| `Bash(git status:*)` | With arguments | `git status --short` |
| `Bash(git diff:*)` | With arguments | `git diff --stat src/` |
| `Bash(uv run pytest:*)` | Full prefix | `uv run pytest -xvs tests/` |
| `Bash(timeout:*)` | All timeout commands | `timeout 30 uv run pytest` |

### Why This Matters

Claude Code's permission system uses **prefix matching** on the full command string. When you write:
- `"Bash(git:*)"` - Matches any command starting with `git` (overly permissive)
- `"Bash(git diff:*)"` - Matches only `git diff` commands (scoped permission)

Always phrase commands to match your allowlist patterns. If a command fails with a permission error, check:
1. Does the command start with an allowed prefix?
2. Is there a space vs. colon mismatch?
3. Are you using the exact tool path specified (e.g., `uv run` vs. bare `pytest`)?

## ACP Protocol Notes

See `docs/acp-protocol.md` for the full protocol specification.
See `docs/diagrams/*.puml` for PlantUML sequence diagrams.

### Session Lifecycle

- **`session/cancel`** is ONLY for cancelling in-progress prompts (e.g., user clicks stop during generation). It is NOT for session termination.
- There is NO ACP message for session/process termination. Shutdown happens via process termination.
- Every `session/prompt` request MUST receive exactly one `PromptResponse`. If cancelled mid-stream, return `stop_reason="cancelled"`.

### Rider Shutdown Behavior (Windows)

**Known Rider Bug**: When deleting the *active* chat (the one you're currently in), Rider fails to close stdio pipes or terminate the subprocess, causing the agent to hang. This is a Rider bug that has been reported.

**Workaround**: Delete chats from the chat list sidebar instead of while the chat is active. Deleting from the chat list works correctly.

When properly closing a chat (from the list or closing Rider):
1. Rider closes the stdio pipes
2. Agent detects EOF and exits
3. Rider cleans up resources

### Debugging ACP

Rider logs are at:
- `%LOCALAPPDATA%\JetBrains\Rider2025.3\log\acp\acp.log` - High-level events
- `%LOCALAPPDATA%\JetBrains\Rider2025.3\log\acp\acp-transport.log` - Raw JSON-RPC messages

## Guidance

Ask a lot of questions during planning.

## TODO Management

The project tracks work in `ACE-TODO.md` with prioritized, parallelizable streams.

**Task status markers:**
- `[In progress]` - Top-level item currently under development
- `[/]` - Subtask in progress
- `[x]` - Completed task/subtask
- `[ ]` - Not started

**When completing work:**
- Commit ACE-TODO.md when starting work (records task in git history)
- Mark subtasks as `[/]` when starting, `[x]` when done
- Remove completely finished tasks (all subtasks done) in the final commit
- Update the parallelization matrix if dependencies change

**When discovering new work:**
- Add to appropriate priority tier (P0-P3)
- Assign to a stream (A/B/C) if parallelizable
- Include enough detail for another agent to execute

**Priority tiers:**
- **P0**: Critical path, blocks other work
- **P1**: Core features, can parallelize
- **P2**: Important but deferrable
- **P3**: Tech debt, run alongside anything
