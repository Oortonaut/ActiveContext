# MCACP Supervised Test Fixture

Structured test plan for exercising ActiveContext as a child agent via MCACP.
Each test is a self-contained section with steps, prompts, and verification criteria.

Tests are ordered by dependency — later tests assume earlier ones passed.

## Conventions

- **Operator**: The agent running these tests (Claude Code, or eventually ActiveContext itself)
- **Subject**: The ActiveContext instance under test
- **Session refs**: Use `$SESSION` as placeholder for the active session ID
- **Log**: `C:\ActiveContext\debug.log` (set via `AC_LOG` in agent env)

---

# Phase 1: Lifecycle

## T01 — Initialize agent

**Goal**: Spawn the ActiveContext process and complete the ACP handshake.

**Steps**:
1. Call `discover_agents` — confirm "Active Context" appears with source `jetbrains`
2. Call `initialize(agentId="Active Context")`
3. Record `agentInfo.name`, `agentInfo.version`, `agentCapabilities`

**Verify**:
- `agentInfo.name` == `"activecontext"`
- `agentCapabilities.loadSession` == `true`
- Debug log contains `"Ready to accept ACP requests"`

**Teardown**: None (agent stays running for subsequent tests)

---

## T02 — Create session

**Goal**: Create a new session and confirm it is usable.

**Steps**:
1. Call `new_session(agentId="Active Context", cwd="C:\\ActiveContext", permissionPolicy="allow_all")`
2. Record `sessionId` as `$SESSION`
3. Record `modes.availableModes` and `modes.currentModeId`

**Verify**:
- `sessionId` is a valid UUID
- `modes.currentModeId` == `"normal"`
- At least `normal`, `plan`, `brave` in `availableModes`
- Debug log contains `"Created session $SESSION"`

---

## T03 — Agent status after session

**Goal**: Confirm the agent reports the active session.

**Steps**:
1. Call `get_agent_status(agentId="Active Context")`

**Verify**:
- `activeSessions` contains `$SESSION`
- `status` is not empty
- `agentInfo.version` matches T01

---

## T04 — Close and reload session

**Goal**: Test session persistence via close + load.

**Steps**:
1. Send a prompt: `"Remember the codeword: TANGERINE"`
2. Wait for completion
3. Call `close_session(sessionId=$SESSION)`
4. Call `load_session(agentId="Active Context", sessionId=$SESSION, cwd="C:\\ActiveContext")`
5. Send a prompt: `"What was the codeword I told you?"`
6. Collect response

**Verify**:
- Step 3 succeeds without error
- Step 4 returns the same `sessionId`
- Step 6 response contains `"TANGERINE"`
- Debug log shows agent loop stop then restart for `$SESSION`

---

# Phase 2: DSL Basics

## T05 — Text node creation

**Goal**: Have the subject create a text node and confirm it renders.

**Prompt**:
```
Execute this statement and show me the result:
v = text("pyproject.toml", expansion=Expansion.CONTENT)
show(v)
```

**Verify**:
- Response mentions creating a text node or shows file content
- No error in response or debug log
- If subject returns namespace, `v` is present

---

## T06 — Shell execution

**Goal**: Run a shell command through the DSL and collect output.

**Prompt**:
```
Run this and tell me the output:
s = shell("git", args=["log", "--oneline", "-3"])
wait(s)
```

**Verify**:
- Response contains 3 git log lines from the ActiveContext repo
- Debug log shows shell execution events
- No permission errors (we used `allow_all`)

---

## T07 — Group creation

**Goal**: Create multiple nodes and group them.

**Prompt**:
```
Execute these statements:
v1 = text("src/activecontext/__init__.py", expansion=Expansion.CONTENT)
v2 = text("src/activecontext/session/protocols.py", expansion=Expansion.CONTENT)
g = group(v1, v2)
g.expansion = Expansion.CONTENT
```

**Verify**:
- Response acknowledges the group creation
- No errors about missing files or invalid syntax

---

## T08 — Checkpoint and restore

**Goal**: Test DAG snapshotting.

**Prompt** (send as two sequential prompts):

Prompt 1:
```
Execute:
checkpoint("before_experiment")
a = artifact("code", content="print('hello world')", language="python")
```

Prompt 2:
```
Execute:
restore("before_experiment")
```

**Verify**:
- After prompt 1: artifact node exists
- After prompt 2: response confirms restore, artifact is gone

---

# Phase 3: Session Features

## T09 — Mode switch to Plan

**Goal**: Switch to plan mode and observe behavioral change.

**Steps**:
1. Call `set_mode` (via ACP) or send prompt: `"Switch to plan mode"`
2. Send prompt: `"I want to add a new node type called DiagramNode"`

**Verify**:
- Subject responds with a plan/outline rather than immediately writing code
- Response style is more deliberate and asks clarifying questions
- Debug log or status reflects mode change

---

## T10 — MCP server passthrough

**Goal**: Create a session with a custom MCP server and verify the subject can use it.

**Steps**:
1. Call `new_session` with:
   ```json
   {
     "agentId": "Active Context",
     "cwd": "C:\\ActiveContext",
     "permissionPolicy": "allow_all",
     "mcpServers": [
       {
         "name": "test-fs",
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\ActiveContext\\tests"]
       }
     ]
   }
   ```
2. Record new `$SESSION2`
3. Send prompt: `"List your available MCP servers and their tools"`

**Verify**:
- Response mentions `test-fs` (or `filesystem`) server
- Server has tools like `read_file`, `list_directory`
- Debug log shows `"Connected to MCP server 'test-fs'"`

**Teardown**: Close `$SESSION2`

---

## T11 — Permission policy: operator

**Goal**: Test the `operator` permission flow where the operator must grant permissions.

**Steps**:
1. Call `new_session(agentId="Active Context", cwd="C:\\ActiveContext", permissionPolicy="operator")`
2. Record `$SESSION3`
3. Send prompt: `"Read the file pyproject.toml and tell me the project name"`
4. Poll with `prompt` — expect a `permission_request` event
5. Call `grant_permission(sessionId=$SESSION3, toolCallId=<from event>, optionId=<approve>)`
6. Continue polling for completion

**Verify**:
- Step 4 returns a permission request (not immediate completion)
- After granting, subject completes the task
- Response contains the project name from pyproject.toml

**Teardown**: Close `$SESSION3`

---

# Phase 4: Multi-Session

## T12 — Concurrent sessions on same agent

**Goal**: Two sessions coexist on the same agent process.

**Steps**:
1. Ensure `$SESSION` is still active from T02
2. Call `new_session` to create `$SESSION_B`
3. Call `get_agent_status` — confirm both in `activeSessions`
4. Send prompt to `$SESSION`: `"What is your session ID?"`
5. Send prompt to `$SESSION_B`: `"What is your session ID?"`
6. Collect both responses

**Verify**:
- Agent status lists both sessions
- Each session reports a different ID
- Prompts to one don't leak into the other

**Teardown**: Close `$SESSION_B`

---

## T13 — Work coordination across sessions

**Goal**: Test `work_on` / `work_check` conflict detection.

**Steps**:
1. Create `$SESSION_X` and `$SESSION_Y`
2. Prompt `$SESSION_X`:
   ```
   Execute: work_on("Refactoring nodes", "src/activecontext/context/nodes.py")
   ```
3. Prompt `$SESSION_Y`:
   ```
   Execute: conflicts = work_check("src/activecontext/context/nodes.py")
   ```
4. Collect `$SESSION_Y` response

**Verify**:
- `$SESSION_Y` reports a conflict or active work on `nodes.py`
- No crash or unhandled error

**Teardown**: Prompt both to `work_done()`, then close both sessions

---

# Phase 5: Resilience

## T14 — Cancel mid-stream

**Goal**: Cancel a prompt while the subject is generating.

**Steps**:
1. Send a long prompt: `"Write a detailed 2000-word essay about the history of programming languages"`
2. Immediately call `cancel(sessionId=$SESSION)`
3. Poll for events

**Verify**:
- Receive a `complete` event with `stopReason` == `"cancelled"`
- Session remains usable — send a follow-up prompt and get a normal response
- Debug log shows `"Agent loop cancelled"` followed by resumed activity

---

## T15 — Prompt after error

**Goal**: Confirm the session recovers from a bad DSL statement.

**Prompt**:
```
Execute: this_function_does_not_exist()
```

Then follow up:
```
Execute: v = text("pyproject.toml", expansion=Expansion.CONTENT)
```

**Verify**:
- First prompt returns an error (NameError or similar)
- Second prompt succeeds — session is not poisoned

---

## T16 — Agent shutdown and reinitialize

**Goal**: Full lifecycle teardown and restart.

**Steps**:
1. Call `shutdown(agentId="Active Context")`
2. Confirm all sessions are gone
3. Call `initialize(agentId="Active Context")` again
4. Call `new_session` and send a test prompt

**Verify**:
- Shutdown succeeds
- Re-initialize succeeds with same capabilities
- New session is functional

---

# Phase 6: Observability

## T17 — Debug log correlation

**Goal**: Verify that log entries can be correlated with MCACP operations.

**Steps**:
1. Note the current time
2. Send prompt: `"Execute: topic('Log Correlation Test')"`
3. Read debug.log, filter entries after the noted time

**Verify**:
- Log contains `"Queued message"` with a message ID for `$SESSION`
- Log contains `"Message ... completed"` for the same message ID
- Timestamps are monotonically increasing

---

## T18 — Dashboard availability

**Goal**: Check if the web dashboard starts and is reachable.

**Steps**:
1. Send prompt: `"Open the dashboard"` (or check debug log for dashboard URL)
2. If URL found (e.g., `http://127.0.0.1:31993`), fetch it

**Verify**:
- Debug log contains `"Dashboard started on http://..."`
- If reachable, response contains HTML with session/graph information

---

# Self-Test Bootstrapping

When ActiveContext runs this fixture against itself:

1. Parse this markdown to extract test sections (heading level 2 = test)
2. For each test, extract **Steps** and **Verify** blocks
3. Execute steps using MCACP tools (or internal session API)
4. Evaluate verify conditions — report pass/fail per test
5. Aggregate results into a summary

The self-test agent should:
- Skip tests that require a second agent process (T10, T11, T12, T13)
- Adapt `$SESSION` placeholders to actual IDs
- Respect test ordering (Phase 1 before Phase 2, etc.)
- Halt on Phase 1 failures (can't proceed without a working session)
