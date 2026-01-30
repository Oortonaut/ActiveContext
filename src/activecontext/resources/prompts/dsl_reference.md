# DSL Reference

Functions and syntax available in the ActiveContext Timeline namespace.

## Code Execution

Execute statements in `python/acrepl` fenced code blocks:

~~~markdown
```python/acrepl
v = text("src/main.py")
v.expansion = Expansion.CONTENT
```
~~~

Regular `python` blocks show examples without execution. Only `python/acrepl` blocks run in the REPL.

## Node Access

Nodes are accessible by their display ID directly in the namespace:

```python
v = text("src/main.py")   # Creates text_1
text_1.expansion = Expansion.CONTENT  # Direct access works

g = group(text_1, text_2)  # Creates group_1
group_1.expansion = Expansion.CONTENT  # Direct access
```

Display IDs follow the format `{type}_{sequence}`: `text_1`, `group_2`, `shell_3`, etc.

User-defined variables take precedence:
```python
v = text("main.py")   # v shadows text_1
v.expansion = ...         # Use v
text_1.expansion = ...    # Or use display ID directly
```

## Enums and Constants

### Expansion

Controls rendering detail level for context nodes. Used with node constructors via the `expansion` parameter.

```python
from activecontext import Expansion

Expansion.HEADER   # Title and metadata only (~50 tokens)
Expansion.CONTENT  # Main content/summary (default for groups)
Expansion.INDEX    # Content plus section headings without recursing
Expansion.ALL      # Full view with all details (default for views)
```

See `node_states.md` for detailed documentation on each expansion level.

### Visibility (hide/unhide)

Control whether a node appears in the projection.

```python
hide(text_1)                           # Hide from projection (but still ticked)
hide(text_1, text_2, group_1)          # Hide multiple nodes
unhide(text_1)                         # Restore to previous expansion
unhide(text_1, expand=Expansion.CONTENT)  # Restore with specific expansion
```

### TickFrequency

Controls when running nodes update their content.

```python
from activecontext import TickFrequency

TickFrequency.turn()        # Update every turn
TickFrequency.period(5)     # Update every 5 seconds
TickFrequency.async_()      # Async execution
TickFrequency.never()       # No automatic updates
```

## Context Node Constructors

### `text(path, *, pos="1:0", expansion=Expansion.ALL, mode="paused", parent=None)`
Create a text view of a file or file region.

```python
v = text("src/main.py")                          # Entire file
v = text("src/main.py", pos="50:0")               # Start at line 50
v = text("src/main.py", expansion=Expansion.CONTENT)
v = text("src/main.py", parent=group_node)       # Link to parent at creation
```

### `markdown(path, *, content=None, expansion=Expansion.ALL, parent=None)`
Parse a markdown file into a tree of TextNodes, where each heading section is a separate node.

```python
m = markdown("README.md")                        # Parse file
m = markdown("inline.md", content="# Title\n\nContent")  # Inline content
m = markdown("docs/guide.md", parent=docs_group) # Link to parent
```

Returns the root TextNode. Child sections are accessible via `children_ids`.

### `view(media_type, path, expansion=Expansion.ALL, **kwargs)`
Dispatcher that routes to `text()` or `markdown()` based on media type.

```python
v = view("text", "src/main.py")                  # Same as text()
m = view("markdown", "docs/README.md")           # Same as markdown()
```

### `group(*members, expansion=Expansion.CONTENT, summary=None, parent=None)`
Create a summary group over multiple nodes.

```python
g = group(v1, v2, v3)                    # Group from node objects
g = group("node_id_1", "node_id_2")      # Group from node IDs
g = group(v1, v2, summary="Auth module overview")
```

### `choice(*children, selected=None, expansion=Expansion.ALL, parent=None)`
Create a dropdown-like selection view. Only the selected child is visible.

```python
c = choice(option1, option2, option3)          # First child selected by default
c = choice(opt1, opt2, selected="opt2_id")     # Specify initial selection
c.select("opt3_id")                             # Switch selection
c.get_options()                                 # Get list of option titles
```

## Progression Views

Progression views provide structured iteration patterns for agent workflows.

### `sequence(*children, expansion=Expansion.ALL, parent=None)`
Create a sequential workflow. Agent works through steps in order.

```python
# Create sequence of review steps
seq = sequence(step1, step2, step3)

# Progress through steps
seq.advance()           # Mark current complete, move to next
seq.back()              # Go back one step
seq.mark_complete()     # Mark current complete without advancing
seq.skip()              # Skip current step without completing
seq.goto(2)             # Jump to specific step index

# Check status
seq.current_index       # 0, 1, 2, ... (0-based)
seq.progress            # "2/3"
seq.is_complete         # True when all steps done
seq.completed_steps     # Set of completed step indices

# Rendering
seq.render_progress()   # Markdown progress list
```

### `loop_view(child, max_iterations=None, expansion=Expansion.ALL, parent=None)`
Create an iterative refinement loop. Agent iterates on a single prompt.

```python
# Create loop with optional iteration limit
loop = loop_view(review_prompt, max_iterations=5)

# Iterate with state updates
loop.iterate(feedback="Add error handling")
loop.iterate(feedback="Looks good!", approved=True)
loop.done()             # Exit loop early
loop.reset()            # Start over

# Update state without incrementing iteration
loop.update_state(note="Important finding")

# Check status
loop.iteration          # 1, 2, 3, ... (1-based)
loop.state              # Accumulated state dict
loop.is_done            # True when done() called or max reached
loop.max_iterations     # Limit or None
loop.iterations_remaining  # Remaining or None

# Rendering
loop.render_header()    # "## Review Loop [iteration 2/5]"
loop.render_state()     # Formatted state display
```

### `state_machine(states, transitions, initial=None, expansion=Expansion.ALL, parent=None)`
Create a state machine for branching workflows.

```python
# Define states and allowed transitions
fsm = state_machine(
    states={
        "idle": idle_node.node_id,
        "working": working_node.node_id,
        "done": done_node.node_id
    },
    transitions={
        "idle": ["working"],
        "working": ["done", "idle"],  # Can go back
        "done": []                     # Terminal state
    },
    initial="idle"
)

# Navigate states
fsm.transition("working")           # Valid transition
fsm.force_transition("done")        # Bypass rules (use carefully)
fsm.reset()                         # Back to initial state

# Check status
fsm.current_state                   # "working"
fsm.valid_transitions               # ["done", "idle"]
fsm.can_transition("done")          # True
fsm.state_history                   # ["idle"]
fsm.all_states                      # ["idle", "working", "done"]

# Rendering
fsm.render_header()                 # "## Task: working → [done, idle]"
```

Progression views can be combined:
- Sequence of loops (each step is iterative)
- State machine with loops at each state

State is automatically persisted to node tags for session save/restore.

### `topic(title, *, status="active", parent=None)`
Create a conversation topic/thread marker. Note: Topics do not have an expansion parameter.

```python
t = topic("Authentication Implementation")
t = topic("Bug Fix", status="resolved")   # Status: active, resolved, deferred
```

### `artifact(artifact_type="code", *, content="", language=None, parent=None)`
Create an artifact (code snippet, output, error). Note: Artifacts do not have an expansion parameter.

```python
a = artifact("code", content="def foo(): pass", language="python")
a = artifact("error", content=error_text)
a = artifact("output", content=command_output)
```

## Path Prefixes

Path arguments in node constructors support special prefixes:

### `@prompts/` - Bundled Reference Prompts

Access bundled reference documentation from the ActiveContext package:

```python
m = markdown("@prompts/dsl_reference.md")      # DSL function reference
m = markdown("@prompts/node_states.md")        # Expansion documentation
m = markdown("@prompts/context_graph.md")      # DAG manipulation guide
m = markdown("@prompts/work_coordination.md")  # Multi-agent coordination
m = markdown("@prompts/mcp.md")                # MCP usage guide
```

Available prompts:
- `dsl_reference.md` - This reference document
- `node_states.md` - Expansion enum documentation
- `context_graph.md` - DAG manipulation guide
- `work_coordination.md` - Multi-agent coordination guide
- `mcp.md` - MCP server integration guide
- `context_guide.md` - Context management guide

These paths are automatically resolved to the bundled package content, regardless of the session's working directory.

## Node Methods

All context nodes support chainable configuration methods.

### Common Methods

```python
node.expansion = Expansion.CONTENT     # Change rendering state
node.Run(TickFrequency.turn())     # Start running with frequency
node.Pause()                       # Stop automatic updates
```

### TextNode Methods

```python
v.SetPos("50:0")      # Jump to line 50
```

### Method Chaining

Use direct assignment for configuration:

```python
v = text("src/main.py")
v.expansion = Expansion.ALL
v.Run(TickFrequency.turn())
```

## DAG Manipulation

### `link(child, parent)`
Add a parent-child relationship in the context graph.

```python
link(view_node, group_node)   # Make view_node a child of group_node
link(v1, g)                   # Add v1 to group g
```

Note: Argument order is `(child, parent)` - the child comes first.

### `unlink(child, parent)`
Remove a parent-child relationship.

```python
unlink(view_node, group_node)
```

## Visibility Control

### `hide(*nodes)`
Hide nodes from projection rendering. Nodes continue to tick but don't appear in the projection.

```python
hide(text_1)              # Hide single node
hide(text_1, text_2)      # Hide multiple nodes
hide("text_1", group_2)   # Mix of IDs and objects
```

Returns the count of nodes hidden. The previous expansion state is saved for later restoration via `unhide()`.

### `unhide(*nodes, expand=None)`
Restore hidden nodes to projection rendering.

```python
unhide(text_1)                         # Restore to previous expand state
unhide(text_1, text_2)                 # Restore multiple
unhide(text_1, expand=Expansion.CONTENT)  # Force specific expansion
```

Returns the count of nodes restored. If `expand` is not specified, restores to the state before `hide()` was called. If the node was never hidden, defaults to `Expansion.ALL`.

## Checkpointing

### `checkpoint(name)`
Save current DAG structure for later restoration.

```python
checkpoint("before_refactor")
```

### `restore(name)`
Restore DAG structure from a checkpoint.

```python
restore("before_refactor")
```

### `checkpoints()`
List available checkpoints. Returns list of checkpoint metadata dicts.

```python
cps = checkpoints()  # [{"name": "before_refactor", "created": ..., ...}]
```

### `branch(name)`
Save current DAG structure as a checkpoint and continue (alias for checkpoint).

```python
branch("refactor_attempt_2")  # Same as checkpoint("refactor_attempt_2")
```

## Shell Execution

### `shell(command, args=None, cwd=None, env=None, timeout=30.0, *, expansion=Expansion.ALL)`
Execute a shell command asynchronously. Returns a ShellNode.

```python
s = shell("pytest", args=["-v", "tests/"])
s = shell("npm", args=["run", "build"], timeout=120)
s = shell("git", args=["status"], cwd="/path/to/repo")
s = shell("make", env={"CC": "clang"})

# Check status
s.is_complete    # True when done
s.is_success     # True if exit_code == 0
s.exit_code      # Exit code
s.output         # stdout/stderr
s.full_command   # Command with args as string
```

## HTTP Requests

### `fetch(url, *, method="GET", headers=None, data=None, json=None, timeout=30.0)`
Make an HTTP/HTTPS request (requires permission).

```python
response = fetch("https://api.example.com/data")
response = fetch("https://api.example.com/post", method="POST", data=payload)
response = fetch("https://api.example.com/api", method="POST", json={"key": "value"})

# You can also use await explicitly if preferred:
response = await fetch("https://api.example.com/data")
```

> **Note**: The DSL supports top-level `await`. You can write `result = await func()` or just `result = func()` - both work. When you omit `await`, the Timeline automatically awaits coroutines stored in the namespace.

## File Locking

### `lock_file(lockfile, timeout=30.0, *, expansion=Expansion.HEADER)`
Acquire an exclusive file lock asynchronously for coordination. Returns a LockNode.

The lock uses OS-level file locking (fcntl on Unix, msvcrt on Windows).
The lockfile is created if it doesn't exist.

```python
lock = lock_file(".mylock", timeout=60)
wait(lock, wake_prompt="Lock acquired, proceeding...")
# Lock is acquired when node shows is_held = True
# Do work with the locked resource
lock_release(lock)

# Check status
lock.is_complete  # True when acquisition attempt finished
lock.is_held      # True if lock is currently held
```

### `lock_release(lock)`
Release a previously acquired file lock.

```python
lock_release(lock)          # By LockNode reference
lock_release("lock_1")      # By node ID
```

## Notification Control

### `notify(node, level=NotificationLevel.WAKE)`
Set notification level for a node. Controls how the agent is alerted to changes.

```python
from activecontext import NotificationLevel

notify(shell_node, NotificationLevel.WAKE)    # Wake agent when done
notify(text_node, NotificationLevel.HOLD)     # Buffer notifications
notify(group_node, "ignore")                  # String level also works
```

Notification levels:
- `WAKE` - Immediately wake the agent when node changes
- `HOLD` - Buffer notifications until next tick
- `IGNORE` - Don't notify on changes

## Work Coordination

### `work_on(intent, *files, mode="write", dependencies=None)`
Register work area for multi-agent coordination.

```python
work_on("Implementing OAuth", "src/auth/oauth.py", "src/auth/config.py")
```

### `work_check(*files, mode="write")`
Check for conflicts with other agents.

```python
conflicts = work_check("src/shared.py")
# Returns: [{"agent_id": "...", "file": "...", "their_mode": "...", "their_intent": "..."}]
```

### `work_update(intent=None, files=None, status=None)`
Update work registration.

```python
work_update(intent="OAuth: Adding token refresh")
work_update(status="paused")
```

### `work_done()`
Unregister work area.

```python
work_done()
```

### `work_list()`
List all active work entries from all agents.

```python
entries = work_list()
```

## MCP (Model Context Protocol)

### `mcp_connect(name, *, command=None, url=None, env=None, expansion=Expansion.ALL)`
Connect to an MCP server. Returns an MCPServerNode.

```python
fs = mcp_connect("filesystem")                                    # From config
gh = mcp_connect("gh", command=["npx", "-y", "@mcp/server-github"])  # Dynamic
result = fs.read_file(path="/home/user/data.txt")                 # Call tool
```

### `mcp_disconnect(name)`
Disconnect from an MCP server.

```python
mcp_disconnect("filesystem")
```

### `mcp_list()`
List connected servers.

```python
servers = mcp_list()  # [{"name": "...", "status": "connected", "tools": 5}]
```

### `mcp_tools(server=None)`
List available tools.

```python
tools = mcp_tools()              # All tools
tools = mcp_tools("filesystem")  # Specific server
```

## Agent Control

### `done(message="")`
Signal task completion.

```python
done("Refactoring complete")
```

### `wait(node, *, wake_prompt="Node completed.", timeout=None, timeout_prompt=None, failure_prompt=None)`
Wait for a single node to complete.

Ends the current turn and waits for the specified node (typically a ShellNode or LockNode) to complete. Ticks continue while waiting. When the node completes, the wake_prompt is injected and the agent turn resumes.

```python
wait(shell_node, wake_prompt="Build complete, checking results...")
wait(lock_node, timeout=30, timeout_prompt="Lock timed out!")
wait(s1, failure_prompt="Command failed: {node}")
```

The wake_prompt can use `{node}` placeholder for the completed node.

### `wait_any(*nodes, wake_prompt="...", timeout=None)`
Wait for any of the specified nodes to complete.

```python
wait_any(s1, s2, s3)  # Resume when first completes
```

### `wait_all(*nodes, wake_prompt="...", timeout=None)`
Wait for all specified nodes to complete. Alias for multiple-node wait.

```python
wait_all(shell_node1, shell_node2)  # Resume when all complete
```

## Multi-Agent Functions

These functions are available when an agent manager is configured (typically in multi-agent scenarios).

### `spawn(agent_type, *, prompt=None, context=None, **kwargs)`
Spawn a child agent of the specified type.

```python
child = spawn("worker", prompt="Implement the auth module")
child = spawn("researcher", context={"files": ["src/api/"]})
```

### `send(agent_id, message)`
Send a message to another agent.

```python
send(child.agent_id, "Please also add tests")
```

### `send_update(agent_id, **updates)`
Send a status update to another agent.

```python
send_update(child.agent_id, status="paused", reason="Waiting for review")
```

### `recv()`
Receive messages from other agents.

```python
messages = recv()  # List of AgentMessage
for msg in messages:
    print(f"From {msg.sender}: {msg.content}")
```

### `agents()`
List all agents in the session.

```python
all_agents = agents()  # List of agent info dicts
```

### `agent_status(agent_id)`
Get the status of a specific agent.

```python
status = agent_status(child.agent_id)
```

### `pause_agent(agent_id)` / `resume_agent(agent_id)` / `terminate_agent(agent_id)`
Control agent lifecycle.

```python
pause_agent(child.agent_id)
resume_agent(child.agent_id)
terminate_agent(child.agent_id)
```

### `get_shared_node(node_id)`
Get a node that's shared between agents.

```python
shared = get_shared_node("shared_config")
```

### `wait_message(*, from_agent=None, timeout=None)`
Wait for a message from another agent.

```python
wait_message(from_agent=child.agent_id, timeout=60)
```

## Event System

These functions are available in agent namespace for event-driven programming.

### `event_response(event_name, handler)`
Register a handler for an event.

```python
event_response("mcp_result", lambda data: print(f"MCP result: {data}"))
```

### `wait_file_change(path, *, timeout=None)`
Wait for a file to change.

```python
wait_file_change("src/config.yaml", timeout=300)
```

### `on_file_change(path, handler)`
Register a handler for file changes.

```python
on_file_change("src/config.yaml", lambda: print("Config changed!"))
```

## Introspection

### `ls()`
List all context objects in the namespace.

```python
handles = ls()  # {"v": TextNode(...), "g": GroupNode(...)}
```

### `show(node_or_id)`
Show detailed info about a node.

```python
show(v)
show("node_id_123")
```

### `get(name)`
Look up a node by name with fuzzy matching.

```python
# Exact lookup by variable name
v = get("my_view")

# Lookup by node ID
v = get("text_1")

# Lookup by file path (for TextNodes)
v = get("main.py")

# Fuzzy match on partial name
v = get("main")  # Matches "main.py", "main_module", etc.

# Returns None if no match found
result = get("nonexistent")  # None
```

Lookup order:
1. Exact variable name in namespace
2. Exact node ID
3. Fuzzy match on variable names, node IDs, file paths, and titles

### `nodes`
Dict-like accessor for node lookup. Supports indexing and attribute access.

```python
# Indexing (raises KeyError if not found)
v = nodes["my_view"]
v = nodes["text_1"]

# Attribute access (raises AttributeError if not found)
v = nodes.my_view
v = nodes.text_1

# Check if node exists
if "main.py" in nodes:
    v = nodes["main.py"]

# Iteration
for node_id in nodes:
    print(node_id)

# List all node IDs
print(nodes.keys())

# Get all nodes as views
for view in nodes.values():
    print(view.node_id)
```

### `ls_permissions()`
List file permission rules.

### `ls_imports()`
List allowed import modules.

### `ls_shell_permissions()`
List allowed shell commands.

### `ls_website_permissions()`
List allowed URLs/domains.

### `set_title(title)`
Set the session title.

```python
set_title("Auth Module Refactor")
```

## XML Syntax Alternative

You can use XML-style tags instead of Python syntax. Tags are converted to Python before execution.

### Object Creation

```xml
<!-- name becomes the variable name -->
<view name="v" path="src/main.py" expansion="all"/>
<group name="g" expansion="content">
    <member ref="v"/>
</group>
<topic name="t" title="Feature X"/>
<artifact name="a" artifact_type="code" content="def foo(): pass" language="python"/>
```

Note: Use `<view>` tag for both text and markdown files. The `<text>` and `<markdown>` tags are not supported.

### Method Calls and Assignments

```xml
<!-- Direct field assignment using assign tag -->
<assign target="v.expansion" value="content"/>
<SetPos self="v" pos="50:0"/>
<Run self="v" freq="turn"/>
<Pause self="v"/>
```

### Shell Execution

```xml
<shell command="pytest" args="tests/,-v" timeout="60"/>
<shell command="git" args="status,--short"/>
```

### DAG Manipulation

```xml
<link child="v" parent="g"/>   <!-- Note: child first, parent second -->
<unlink child="v" parent="g"/>
<checkpoint name="before_refactor"/>
<restore name="before_refactor"/>
<branch name="attempt_2"/>
```

### Utility Functions

```xml
<ls/>
<show self="v"/>
<done message="Task complete"/>
<wait nodes="s1,s2"/>
```

### MCP Operations

```xml
<mcp_connect name="filesystem"/>
<mcp_disconnect name="filesystem"/>
```

Use whichever syntax you prefer - Python or XML. They are functionally equivalent.
