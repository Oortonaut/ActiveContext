# DSL Reference

Functions and syntax available in the ActiveContext Timeline namespace.

## Code Execution

Execute statements in `python/acrepl` fenced code blocks:

~~~markdown
```python/acrepl
v = text("src/main.py", tokens=2000)
v.expansion = Expansion.SUMMARY
```
~~~

Regular `python` blocks show examples without execution. Only `python/acrepl` blocks run in the REPL.

## Node Access

Nodes are accessible by their display ID directly in the namespace:

```python
v = text("src/main.py")   # Creates text_1
text_1.expansion = Expansion.SUMMARY  # Direct access works

g = group(text_1, text_2)  # Creates group_1
group_1.tokens = 500     # Direct access
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

Controls rendering detail level for context nodes.

```python
from activecontext import Expansion

Expansion.HIDDEN     # Not shown in projection (but still ticked if running)
Expansion.COLLAPSED  # Title and metadata only (~50 tokens)
Expansion.SUMMARY    # LLM-generated summary
Expansion.DETAILS    # Full view with child settings

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

### `text(path, *, pos="1:0", tokens=2000, expansion=Expansion.DETAILS, mode="paused", parent=None)`
Create a text view of a file or file region.

```python
v = text("src/main.py")                          # Entire file
v = text("src/main.py", pos="50:0", tokens=500)  # Start at line 50
v = text("src/main.py", tokens=500, expansion=Expansion.SUMMARY)
v = text("src/main.py", parent=group_node)       # Link to parent at creation
```

### `markdown(path, *, content=None, tokens=2000, expansion=Expansion.DETAILS, parent=None)`
Parse a markdown file into a tree of TextNodes, where each heading section is a separate node.

```python
m = markdown("README.md")                        # Parse file
m = markdown("inline.md", content="# Title\n\nContent")  # Inline content
m = markdown("docs/guide.md", parent=docs_group) # Link to parent
```

Returns the root TextNode. Child sections are accessible via `children_ids`.

### `view(media_type, path, tokens=2000, expansion=Expansion.DETAILS, **kwargs)`
Dispatcher that routes to `text()` or `markdown()` based on media type.

```python
v = view("text", "src/main.py")                  # Same as text()
m = view("markdown", "docs/README.md")           # Same as markdown()
```

### `group(*members, tokens=500, expansion=Expansion.SUMMARY, summary=None, parent=None)`
Create a summary group over multiple nodes.

```python
g = group(v1, v2, v3)                    # Group from node objects
g = group("node_id_1", "node_id_2")      # Group from node IDs
g = group(v1, v2, summary="Auth module overview")
```

### `topic(title, *, tokens=1000, status="active", parent=None)`
Create a conversation topic/thread marker.

```python
t = topic("Authentication Implementation")
t = topic("Bug Fix", status="resolved")   # Status: active, resolved, deferred
```

### `artifact(artifact_type="code", *, content="", language=None, tokens=500, parent=None)`
Create an artifact (code snippet, output, error).

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
node.expansion = Expansion.SUMMARY     # Change rendering state
node.tokens = 500                      # Change token budget
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
v.tokens = 2000
v.expansion = Expansion.DETAILS
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

### `shell(command, args=None, cwd=None, env=None, timeout=30.0, *, tokens=2000, expansion=Expansion.DETAILS)`
Execute a shell command asynchronously. Returns a ShellNode.

```python
s = shell("pytest", args=["-v", "tests/"])
s = shell("npm", args=["run", "build"], timeout=120)
s = shell("git", args=["status"])

# Check status
s.is_complete    # True when done
s.is_success     # True if exit_code == 0
s.exit_code      # Exit code
s.output         # stdout/stderr
```

## HTTP Requests

### `fetch(url, *, method="GET", headers=None, data=None, json=None, timeout=30.0)`
Make an HTTP/HTTPS request (requires permission).

```python
response = await fetch("https://api.example.com/data")
response = await fetch("https://api.example.com/post", method="POST", data=payload)
response = await fetch("https://api.example.com/api", method="POST", json={"key": "value"})
```

## File Locking

### `lock_file(lockfile, timeout=30.0, *, tokens=200, expansion=Expansion.COLLAPSED)`
Acquire a file lock for coordination. Returns a LockNode.

```python
lock = lock_file("src/shared.py.lock", timeout=60)
# Lock is acquired when node shows success
# Do work with the locked resource
lock_release(lock)
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

### `mcp_connect(name, *, command=None, url=None, env=None, tokens=1000, expansion=Expansion.DETAILS)`
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

### `wait(*nodes, wake_prompt="...", timeout=None)`
Wait for all specified nodes to complete.

```python
wait(shell_node1, shell_node2)
```

### `wait_any(*nodes, wake_prompt="...", timeout=None)`
Wait for any of the specified nodes to complete.

```python
wait_any(s1, s2, s3)  # Resume when first completes
```

### `wait_all(*nodes, wake_prompt="...", timeout=None)`
Alias for `wait()`.

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
<view name="v" path="src/main.py" tokens="3000" state="all"/>
<group name="g" tokens="500" state="summary">
    <member ref="v"/>
</group>
<topic name="t" title="Feature X" tokens="1000"/>
<artifact name="a" artifact_type="code" content="def foo(): pass" language="python"/>
```

Note: Use `<view>` tag for both text and markdown files. The `<text>` and `<markdown>` tags are not supported.

### Method Calls and Assignments

```xml
<!-- Direct field assignment using assign tag -->
<assign target="v.expansion" value="collapsed"/>
<assign target="v.tokens" value="500"/>
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
