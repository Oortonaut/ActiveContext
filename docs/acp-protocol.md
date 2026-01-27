# Agent Client Protocol (ACP) Documentation

> Comprehensive protocol reference with PlantUML diagrams

## Overview

The **Agent Client Protocol (ACP)** standardizes communication between code editors/IDEs and AI coding agents. It uses JSON-RPC 2.0 over stdio for local agents, enabling any ACP-compatible agent to work with any ACP-compatible editor.

**Similar to LSP**: Just as the Language Server Protocol standardized language tool integration, ACP decouples agents from editors, allowing developers to choose best-of-breed tools without vendor lock-in.

### Key Resources

- **Official Docs**: https://agentclientprotocol.com/
- **GitHub**: https://github.com/agentclientprotocol/agent-client-protocol
- **JetBrains**: https://www.jetbrains.com/help/ai-assistant/acp.html
- **Zed**: https://zed.dev/acp

### Supported Editors

- JetBrains IDEs (Rider, IntelliJ, PyCharm, etc.)
- Zed
- Neovim (via plugins)
- Emacs, Obsidian, marimo (community plugins)

### SDK Libraries

- **Python**: `agent-client-protocol` on PyPI
- **Rust**: `agent-client-protocol` on crates.io
- **TypeScript**: `@agentclientprotocol/sdk` on npm
- **Kotlin**: `acp-kotlin`

---

## Protocol Fundamentals

### Transport

- **Local agents**: JSON-RPC 2.0 over stdio (newline-delimited, UTF-8)
- **Remote agents**: HTTP/WebSocket (in progress)
- Messages must not contain embedded newlines
- Agents may write logs to stderr

### Key Rules

1. **All file paths must be absolute**
2. **Line numbers are 1-based**
3. **Capabilities determine available features** - check before calling
4. **Every prompt gets exactly one PromptResponse** (even if cancelled)
5. **session/cancel is for in-progress prompts only**, not session termination
6. **Custom extensions use underscore prefix** (`_vendor/method`)

---

## PlantUML Diagrams

### 1. Architecture Overview

```plantuml
@startuml ACP_Architecture
!theme plain
skinparam backgroundColor #FEFEFE

title Agent Client Protocol - Architecture Overview

package "Code Editor / IDE" as IDE {
  [Chat UI] as chatui
  [File Editor] as editor
  [Terminal Emulator] as term_ui
  [Permission Dialog] as perms

  package "ACP Client" as client {
    [JSON-RPC Handler] as rpc_handler
    [Session Manager] as client_session
    [Capability Manager] as caps
  }
}

package "Agent Process" as Agent {
  [JSON-RPC Processor] as agent_rpc
  [Session Controller] as session_ctrl
  [LLM Interface] as llm
  [Tool Executor] as tools
  [MCP Client] as mcp
}

cloud "LLM Provider" as provider {
  [Claude / GPT / etc.] as model
}

database "Session Storage" as storage

' User interactions
chatui --> client_session : user message
editor --> client_session : file context
term_ui <-- client_session : terminal output
perms <-- client_session : permission request

' Client-Agent communication
rpc_handler <--> agent_rpc : JSON-RPC\nover stdio

' Agent internals
agent_rpc --> session_ctrl
session_ctrl --> llm
session_ctrl --> tools
tools --> mcp : MCP calls
llm --> model : API calls

' Persistence
session_ctrl --> storage : persist sessions

note right of rpc_handler
  Messages:
  - initialize
  - session/new
  - session/prompt
  - session/cancel
  - fs/read_text_file
  - terminal/create
end note

@enduml
```

### 2. Protocol Message Flow

```plantuml
@startuml ACP_Message_Flow
!theme plain

title ACP Protocol - Complete Message Flow

participant "Client (IDE)" as C
participant "Agent" as A
participant "LLM" as L

== Initialization Phase ==
C -> A : initialize(protocolVersion, clientCapabilities, clientInfo)
A --> C : InitializeResponse(protocolVersion, agentCapabilities, agentInfo)

== Session Creation ==
C -> A : session/new(cwd, mcpServers)
A --> C : NewSessionResponse(sessionId, models, modes)
A -> C : session/update(available_commands_update)

== Prompt Turn ==
C -> A : session/prompt(sessionId, prompt[])
activate A

A -> L : Submit to LLM
activate L
L --> A : Response stream
deactivate L

loop For each chunk
  A -> C : session/update(agent_message_text_update)
end

opt Tool Execution
  A -> C : session/update(tool_call_update: pending)

  alt Requires Permission
    A -> C : session/request_permission(toolCall, options)
    C --> A : PermissionResponse(optionId)
  end

  A -> C : session/update(tool_call_update: in_progress)
  A -> C : session/update(tool_call_update: completed)
end

A --> C : PromptResponse(stop_reason: end_turn)
deactivate A

== Cancellation ==
C -> A : session/cancel(sessionId)
note right: Agent returns stop_reason: cancelled

@enduml
```

### 3. Initialization Sequence

```plantuml
@startuml ACP_Initialization
!theme plain

title ACP Protocol - Initialization Handshake

participant "Client" as C
participant "Agent Process" as A

C -> C : spawn agent subprocess
activate A

C -> A : initialize
note right
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "protocolVersion": 1,
    "clientCapabilities": {
      "fs": {
        "readTextFile": true,
        "writeTextFile": true
      },
      "terminal": true
    },
    "clientInfo": {
      "name": "rider",
      "title": "JetBrains Rider",
      "version": "2025.3"
    }
  }
}
end note

A --> C : InitializeResponse
note left
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "protocolVersion": 1,
    "agentCapabilities": {
      "loadSession": true,
      "promptCapabilities": {
        "image": true,
        "audio": false,
        "embeddedContext": true
      },
      "mcpCapabilities": {
        "http": false,
        "sse": false
      }
    },
    "agentInfo": {
      "name": "activecontext",
      "title": "ActiveContext",
      "version": "0.1.0"
    },
    "authMethods": []
  }
}
end note

note over C, A
  Protocol version negotiation:
  - If agent supports requested version, respond with same
  - Otherwise, respond with latest supported version
end note

@enduml
```

### 4. Session Lifecycle

```plantuml
@startuml ACP_Session_Lifecycle
!theme plain

title ACP Protocol - Session Lifecycle

[*] --> Initializing : Agent started

state Initializing {
  [*] --> AwaitingInit
  AwaitingInit --> Initialized : initialize()
}

Initialized --> SessionActive : session/new()

state SessionActive {
  [*] --> Idle

  Idle --> Processing : session/prompt()
  Processing --> Streaming : LLM responding
  Streaming --> Executing : Tool calls
  Executing --> Streaming : More LLM output
  Streaming --> Idle : PromptResponse

  Processing --> Cancelled : session/cancel()
  Streaming --> Cancelled : session/cancel()
  Executing --> Cancelled : session/cancel()
  Cancelled --> Idle : stop_reason: cancelled
}

SessionActive --> SessionActive : session/set_mode()
SessionActive --> SessionActive : session/set_model()

note right of SessionActive
  Session persists between prompts
  Can be resumed via session/load
end note

SessionActive --> [*] : Process terminated

@enduml
```

### 5. Prompt Turn Detail

```plantuml
@startuml ACP_Prompt_Turn
!theme plain

title ACP Protocol - Prompt Turn Lifecycle

participant "Client" as C
participant "Agent" as A
participant "LLM" as L

C -> A : session/prompt
note right
{
  "sessionId": "sess_abc123",
  "prompt": [
    {"type": "text", "text": "Fix the bug in auth.py"}
  ]
}
end note
activate A

A -> L : Submit prompt + context
activate L

== Response Streaming ==

loop Response tokens
  L --> A : Token chunk
  A -> C : session/update
  note right
  {
    "sessionId": "sess_abc123",
    "sessionUpdate": "agent_message_text_update",
    "text": "I'll analyze..."
  }
  end note
end

L --> A : Tool call request
deactivate L

== Tool Execution ==

A -> C : session/update (tool_call_update: pending)
note right
{
  "sessionUpdate": "tool_call_update",
  "toolCallId": "tc_001",
  "title": "Reading auth.py",
  "kind": "read",
  "status": "pending"
}
end note

A -> C : session/request_permission
C --> A : PermissionResponse(allow_once)

A -> C : session/update (status: in_progress)
A -> A : Execute tool
A -> C : session/update (status: completed)

== Continue with LLM ==

A -> L : Tool result
L --> A : Continue response...

== Completion ==

A --> C : PromptResponse
note left
{
  "stopReason": "end_turn"
}
end note
deactivate A

@enduml
```

### 6. Permission Request Flow

```plantuml
@startuml ACP_Permissions
!theme plain

title ACP Protocol - Permission Request Flow

participant "Client" as C
participant "Agent" as A
participant "User" as U

A -> C : session/request_permission
note right
{
  "sessionId": "sess_abc123",
  "toolCall": {
    "toolCallId": "tc_001",
    "title": "Write to config.json",
    "kind": "edit"
  },
  "options": [
    {"id": "opt1", "kind": "allow_once", "label": "Allow once"},
    {"id": "opt2", "kind": "allow_always", "label": "Always allow"},
    {"id": "opt3", "kind": "reject_once", "label": "Deny"},
    {"id": "opt4", "kind": "reject_always", "label": "Never allow"}
  ]
}
end note

C -> U : Show permission dialog
U --> C : User clicks "Allow once"

C --> A : PermissionResponse
note left
{
  "optionId": "opt1"
}
end note

alt User cancels prompt during permission
  C --> A : PermissionResponse
  note left
  {
    "outcome": "cancelled"
  }
  end note
end

@enduml
```

### 7. Terminal Operations

```plantuml
@startuml ACP_Terminal
!theme plain

title ACP Protocol - Terminal Operations

participant "Agent" as A
participant "Client" as C
participant "Shell" as S

== Create Terminal ==

A -> C : terminal/create
note right
{
  "sessionId": "sess_abc123",
  "command": "pytest",
  "args": ["-v", "tests/"],
  "cwd": "/home/user/project",
  "outputByteLimit": 1048576
}
end note

C -> S : spawn process
activate S

C --> A : CreateTerminalResponse
note left
{
  "terminalId": "term_001"
}
end note

note over A, C
  terminal/create returns immediately
  Command runs asynchronously
end note

== Poll Output ==

A -> C : terminal/output
note right: {"terminalId": "term_001"}

C --> A : TerminalOutputResponse
note left
{
  "output": "test_auth.py::test_login PASSED\n...",
  "truncated": false
}
end note

== Wait for Exit ==

A -> C : terminal/wait_for_exit
note right: {"terminalId": "term_001"}

S --> C : Process exits
deactivate S

C --> A : WaitForExitResponse
note left
{
  "exitCode": 0,
  "signal": null
}
end note

== Or Kill ==

A -> C : terminal/kill
note right: Terminates running command

== Release ==

A -> C : terminal/release
note right: Free all resources

@enduml
```

### 8. File System Operations

```plantuml
@startuml ACP_FileSystem
!theme plain

title ACP Protocol - File System Operations

participant "Agent" as A
participant "Client" as C
participant "Editor Buffer" as E

== Read File ==

A -> C : fs/read_text_file
note right
{
  "sessionId": "sess_abc123",
  "path": "/home/user/project/src/main.py",
  "line": 10,
  "limit": 50
}
end note

C -> E : Get content (may be unsaved)
E --> C : Current buffer state

C --> A : ReadTextFileResponse
note left
{
  "content": "def main():\n    ..."
}
end note

== Write File ==

A -> C : fs/write_text_file
note right
{
  "sessionId": "sess_abc123",
  "path": "/home/user/project/config.json",
  "content": "{ \"debug\": true }"
}
end note

C -> E : Update buffer / create file

C --> A : WriteTextFileResponse
note left: null (success)

note over A, C
  Key features:
  - Access to unsaved editor state
  - Client tracks modifications
  - Creates files if missing
  - All paths must be absolute
end note

@enduml
```

### 9. Session Modes

```plantuml
@startuml ACP_Session_Modes
!theme plain

title ACP Protocol - Session Mode Switching

participant "Client" as C
participant "Agent" as A
participant "LLM" as L

== Mode Advertisement ==

C -> A : session/new(cwd, mcpServers)
A --> C : NewSessionResponse
note left
{
  "sessionId": "sess_abc123",
  "modeState": {
    "currentModeId": "normal",
    "availableModes": [
      {"id": "normal", "name": "Normal", "description": "Standard mode"},
      {"id": "plan", "name": "Plan", "description": "Plan before acting"},
      {"id": "architect", "name": "Architect", "description": "Design only"}
    ]
  }
}
end note

== Client-Initiated Switch ==

C -> A : session/set_mode
note right
{
  "sessionId": "sess_abc123",
  "modeId": "plan"
}
end note

A --> C : SetModeResponse
note left: Mode switched to "plan"

== Agent-Initiated Switch ==

A -> L : Planning complete, request implementation
L --> A : Call "exit_plan_mode" tool

A -> C : session/update
note right
{
  "sessionUpdate": "current_mode_update",
  "modeId": "normal"
}
end note

note over A, C
  Common modes:
  - Ask: Request permission before changes
  - Plan/Architect: Design without implementing
  - Code/Normal: Full tool access
end note

@enduml
```

### 10. Tool Call Status Lifecycle

```plantuml
@startuml ACP_Tool_Call_States
!theme plain

title ACP Protocol - Tool Call Status Lifecycle

state "pending" as pending : Tool created\nAwaiting permission\nor streaming input

state "in_progress" as progress : Tool executing\nMay send output updates

state "completed" as completed : Execution successful\nResult available

state "failed" as failed : Execution error\nError details provided

[*] --> pending : tool_call_update created

pending --> progress : Permission granted\nor auto-approved
pending --> failed : Permission denied
pending --> completed : No execution needed

progress --> completed : Success
progress --> failed : Error

completed --> [*]
failed --> [*]

note right of pending
  Tool call fields:
  - toolCallId
  - title
  - kind (read/edit/delete/execute/...)
  - status
  - content (optional)
  - locations (optional)
end note

@enduml
```

### 11. Content Block Types

```plantuml
@startuml ACP_Content_Blocks
!theme plain

title ACP Protocol - Content Block Types

package "User → Agent (Prompt)" {
  class TextBlock {
    type: "text"
    text: string
    annotations?: object
  }

  class ImageBlock {
    type: "image"
    mimeType: string
    data: base64
    uri?: string
  }

  class AudioBlock {
    type: "audio"
    mimeType: string
    data: base64
  }

  class ResourceBlock {
    type: "resource"
    resource: EmbeddedResource
  }

  class ResourceLinkBlock {
    type: "resource_link"
    uri: string
    name: string
    mimeType?: string
  }
}

package "Agent → User (Response)" {
  class AgentTextBlock {
    type: "text"
    text: string
  }

  class DiffBlock {
    type: "diff"
    path: string
    oldText: string
    newText: string
  }

  class TerminalBlock {
    type: "terminal"
    terminalId: string
  }
}

note bottom of ImageBlock
  Requires: promptCapabilities.image
end note

note bottom of AudioBlock
  Requires: promptCapabilities.audio
end note

note bottom of ResourceBlock
  Requires: promptCapabilities.embeddedContext
  Preferred for @-mentions
end note

@enduml
```

### 12. Plan Updates

```plantuml
@startuml ACP_Plan_Updates
!theme plain

title ACP Protocol - Agent Plan Communication

participant "Agent" as A
participant "Client" as C

A -> C : session/update (plan)
note right
{
  "sessionId": "sess_abc123",
  "sessionUpdate": "plan",
  "entries": [
    {
      "content": "Read existing auth implementation",
      "priority": "high",
      "status": "completed"
    },
    {
      "content": "Identify security vulnerabilities",
      "priority": "high",
      "status": "in_progress"
    },
    {
      "content": "Implement OAuth2 flow",
      "priority": "medium",
      "status": "pending"
    },
    {
      "content": "Add unit tests",
      "priority": "low",
      "status": "pending"
    }
  ]
}
end note

note over A, C
  Plan entries have:
  - content: Human-readable description
  - priority: high | medium | low
  - status: pending | in_progress | completed

  Each update replaces entire plan.
  Plans can evolve during execution.
end note

@enduml
```

### 13. Complete Message Type Reference

```plantuml
@startuml ACP_Message_Types
!theme plain
skinparam classAttributeIconSize 0

title ACP Protocol - Message Types Reference

package "Agent Methods (required)" {
  class "initialize" as init {
    protocolVersion: int
    clientCapabilities: object
    clientInfo: Implementation
    --
    → InitializeResponse
  }

  class "session/new" as new {
    cwd: string
    mcpServers?: array
    --
    → NewSessionResponse
  }

  class "session/prompt" as prompt {
    sessionId: string
    prompt: ContentBlock[]
    --
    → PromptResponse
  }
}

package "Agent Methods (optional)" {
  class "session/load" as load {
    sessionId: string
    cwd: string
    mcpServers?: array
    --
    → LoadSessionResponse
  }

  class "session/list" as list {
    cwd: string
    cursor?: string
    --
    → ListSessionsResponse
  }

  class "session/set_mode" as mode {
    sessionId: string
    modeId: string
    --
    → SetModeResponse
  }

  class "session/set_model" as model {
    sessionId: string
    modelId: string
    --
    → SetModelResponse
  }
}

package "Agent Notifications" {
  class "session/cancel" as cancel {
    sessionId: string
    --
    (no response)
  }
}

package "Client Methods" {
  class "session/request_permission" as perm {
    sessionId: string
    toolCall: object
    options: PermissionOption[]
    --
    → PermissionResponse
  }

  class "fs/read_text_file" as read {
    sessionId: string
    path: string
    line?: int
    limit?: int
    --
    → {content: string}
  }

  class "fs/write_text_file" as write {
    sessionId: string
    path: string
    content: string
    --
    → null
  }

  class "terminal/create" as tcreate {
    sessionId: string
    command: string
    args?: string[]
    cwd?: string
    --
    → {terminalId: string}
  }

  class "terminal/output" as toutput {
    terminalId: string
    --
    → {output, truncated, exitStatus?}
  }

  class "terminal/wait_for_exit" as twait {
    terminalId: string
    --
    → {exitCode, signal}
  }

  class "terminal/kill" as tkill {
    terminalId: string
    --
    → null
  }

  class "terminal/release" as trelease {
    terminalId: string
    --
    → null
  }
}

package "Session Updates (Notifications)" {
  class agent_message_text_update {
    text: string
  }

  class agent_thought_text_update {
    text: string
  }

  class tool_call_update {
    toolCallId: string
    title?: string
    kind?: string
    status?: string
    content?: ContentBlock[]
  }

  class plan {
    entries: PlanEntry[]
  }

  class current_mode_update {
    modeId: string
  }

  class available_commands_update {
    commands: AvailableCommand[]
  }
}

@enduml
```

---

## Message Reference

### Agent Methods (Required)

| Method | Parameters | Response | Description |
|--------|------------|----------|-------------|
| `initialize` | `protocolVersion`, `clientCapabilities`, `clientInfo` | `InitializeResponse` | Handshake and capability exchange |
| `session/new` | `cwd`, `mcpServers?` | `NewSessionResponse` | Create new session |
| `session/prompt` | `sessionId`, `prompt[]` | `PromptResponse` | Send user message |

### Agent Methods (Optional)

| Method | Parameters | Response | Description |
|--------|------------|----------|-------------|
| `session/load` | `sessionId`, `cwd`, `mcpServers?` | `LoadSessionResponse` | Resume session |
| `session/list` | `cwd`, `cursor?` | `ListSessionsResponse` | List available sessions |
| `session/set_mode` | `sessionId`, `modeId` | `SetModeResponse` | Switch operating mode |
| `session/set_model` | `sessionId`, `modelId` | `SetModelResponse` | Switch LLM model |

### Agent Notifications

| Notification | Parameters | Description |
|--------------|------------|-------------|
| `session/cancel` | `sessionId` | Cancel in-progress prompt |

### Client Methods

| Method | Parameters | Response | Description |
|--------|------------|----------|-------------|
| `session/request_permission` | `sessionId`, `toolCall`, `options[]` | `PermissionResponse` | Request user permission |
| `fs/read_text_file` | `sessionId`, `path`, `line?`, `limit?` | `{content}` | Read file content |
| `fs/write_text_file` | `sessionId`, `path`, `content` | `null` | Write file content |
| `terminal/create` | `sessionId`, `command`, `args?`, `cwd?`, `outputByteLimit?` | `{terminalId}` | Start terminal |
| `terminal/output` | `terminalId` | `{output, truncated, exitStatus?}` | Get terminal output |
| `terminal/wait_for_exit` | `terminalId` | `{exitCode, signal}` | Wait for completion |
| `terminal/kill` | `terminalId` | `null` | Terminate command |
| `terminal/release` | `terminalId` | `null` | Release resources |

### Session Update Types

| Update Type | Fields | Description |
|-------------|--------|-------------|
| `agent_message_text_update` | `text` | Streamed response text |
| `agent_thought_text_update` | `text` | Intermediate thinking |
| `tool_call_update` | `toolCallId`, `title?`, `kind?`, `status?`, `content?` | Tool execution progress |
| `plan` | `entries[]` | Execution plan |
| `current_mode_update` | `modeId` | Mode changed |
| `available_commands_update` | `commands[]` | Slash commands |

---

## Stop Reasons

| Reason | Description |
|--------|-------------|
| `end_turn` | Model completed normally |
| `max_tokens` | Token limit reached |
| `max_turn_requests` | Too many LLM calls in turn |
| `refusal` | Agent refused to proceed |
| `cancelled` | Client cancelled prompt |

---

## Capabilities Reference

### Client Capabilities

```json
{
  "fs": {
    "readTextFile": true,
    "writeTextFile": true
  },
  "terminal": true
}
```

### Agent Capabilities

```json
{
  "loadSession": true,
  "promptCapabilities": {
    "image": true,
    "audio": false,
    "embeddedContext": true
  },
  "mcpCapabilities": {
    "http": false,
    "sse": false
  }
}
```

---

## Configuration Example

### JetBrains `~/.jetbrains/acp.json`

```json
{
  "agent_servers": {
    "activecontext": {
      "command": "python",
      "args": ["-m", "activecontext"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-...",
        "AC_LOG": "C:\\Users\\You\\activecontext.log"
      },
      "use_idea_mcp": false,
      "use_custom_mcp": true
    }
  }
}
```

---

## Extensibility

### Custom Methods

Custom methods use underscore prefix: `_vendor.domain/feature/action`

### Meta Fields

All protocol types include `_meta` field for custom data:

```json
{
  "agentCapabilities": {
    "loadSession": true,
    "_meta": {
      "vendor": {
        "customFeature": true
      }
    }
  }
}
```

Reserved root keys: `traceparent`, `tracestate`, `baggage` (W3C trace context)

---

## Implementation Notes & Quirks

### JetBrains Session Resumption Workaround

**Issue**: JetBrains IDEs (Rider, IntelliJ, PyCharm, etc.) don't pass a session ID in `session/new` or call `session/load` for chat resumption. Each time you return to a chat, the IDE creates a new session instead of resuming.

**Workaround**: Read the chat UUID directly from the JetBrains task history filesystem:

```
%LOCALAPPDATA%\JetBrains\{IDE}\aia-task-history\*.events
```

The most recently modified `.events` file contains the current chat's UUID. This allows the agent to resume sessions even when the IDE doesn't explicitly request it.

**Activation**: Set `AC_CLIENT_JETBRAINS=1` in the `env` block of your `acp.json`:

```json
{
  "agent_servers": {
    "activecontext": {
      "command": "python",
      "args": ["-m", "activecontext"],
      "env": {
        "AC_CLIENT_JETBRAINS": "1"
      }
    }
  }
}
```

### Rider Shutdown Bug (Windows)

**Known Issue**: When deleting the *active* chat (the one you're currently viewing) in Rider, the IDE fails to close stdio pipes or terminate the agent subprocess.

**Symptom**: The agent process hangs indefinitely, waiting for input that never arrives.

**Workaround**: Delete chats from the chat list sidebar instead of while the chat is active. Deleting from the sidebar works correctly.

**Status**: Reported to JetBrains.

### Response Batching (Nagle-style)

To reduce overhead from per-token LLM streaming, the agent buffers `agent_message_text_update` notifications:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `flush_threshold` | 100 chars | Flush immediately when buffer exceeds this |
| `flush_interval` | 50ms | Flush after this delay if threshold not reached |

Non-RESPONSE_CHUNK updates (e.g., tool calls) trigger an immediate flush to ensure proper ordering.

**Configuration** (in `config.yaml`):

```yaml
acp:
  batching:
    enabled: true
    flush_interval: 0.05
    flush_threshold: 100
```

### Update Modes

The agent supports two update delivery modes:

| Mode | Config | Behavior |
|------|--------|----------|
| **Async** | `out_of_band_update: true` | Send updates immediately as notifications. May arrive while `prompt()` is blocking on the client. |
| **Sync** | `out_of_band_update: false` | Queue updates during idle periods. Flush at the start of the next `prompt()`. |

Sync mode is safer for clients that don't handle out-of-band notifications well.

### Cancel Notification Handling

**Key invariant**: `session/cancel` is a notification (no response), but the agent must still return `PromptResponse(stop_reason="cancelled")` for any in-progress prompt.

**Implementation order is critical**:

1. Mark session as closed (prevents new updates from being sent)
2. Cancel the active prompt task with 1s timeout
3. Clean up chunk buffers (with timeout to avoid blocking)
4. Cancel the session in the session manager
5. Return `stop_reason: cancelled` in the PromptResponse

### Session Lifecycle Invariants

1. Every `session/prompt` MUST receive exactly one `PromptResponse`
2. `session/cancel` is ONLY for cancelling in-progress prompts, not session termination
3. There is NO ACP message for session/process termination
4. Shutdown happens via process termination (EOF on stdin or SIGTERM)
5. The agent must handle EOF on stdin gracefully

### Post-Session Setup Timing

After returning `NewSessionResponse`, the agent performs deferred setup:

1. Run startup scripts (emitting `tool_call_update` for each)
2. Advertise available slash commands via `available_commands_update`
3. Start the background agent loop for async processing

This is deferred using `asyncio.create_task()` so the session creation response returns quickly.

### Permission Request Patterns

| Permission Type | `kind` Value | Notes |
|-----------------|--------------|-------|
| File read | `read` | Absolute path required |
| File write | `edit` | Creates file if missing |
| Shell command | `execute` | Full command shown |
| Import | `execute` | Includes submodule option |
| Website GET | `read` | Domain shown |
| Website POST | `edit` | Domain shown |

All permission dialogs include at minimum:
- `allow_once` - Grant for this request only
- `allow_always` - Persist the grant
- `reject_once` / `deny` - Deny this request

### Debug Logging

Rider logs are located at:
- **High-level events**: `%LOCALAPPDATA%\JetBrains\{IDE}\log\acp\acp.log`
- **Raw JSON-RPC**: `%LOCALAPPDATA%\JetBrains\{IDE}\log\acp\acp-transport.log`

Enable extended logging via Rider's Registry key: `llm.agent.extended.logging`

---

## ActiveContext Implementation

This project implements ACP in:

- **Entry point**: `src/activecontext/__main__.py`
- **Agent implementation**: `src/activecontext/transport/acp/agent.py`
- **Session protocols**: `src/activecontext/session/protocols.py`
- **Terminal executor**: `src/activecontext/terminal/acp_executor.py`

### Additional Features

- Nagle-style response batching (50ms / 100 char threshold)
- JetBrains chat UUID detection for session resumption
- Configurable sync/async update modes
- Built-in slash commands: `/help`, `/clear`, `/context`, `/title`, `/dashboard`
