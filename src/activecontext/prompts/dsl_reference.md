# DSL Reference

Functions and syntax available in the ActiveContext Timeline namespace.

## Code Execution

Execute statements in `python/acrepl` fenced code blocks:

~~~markdown
```python/acrepl
v = text("src/main.py", tokens=2000)
v.SetState(NodeState.SUMMARY)
```
~~~

Regular `python` blocks show examples without execution. Only `python/acrepl` blocks run in the REPL.

## Node Access

Nodes are accessible by their display ID directly in the namespace:

```python
v = text("src/main.py")   # Creates text_1
text_1.SetState(NodeState.SUMMARY)  # Direct access works

g = group(text_1, text_2)  # Creates group_1
group_1.SetTokens(500)     # Direct access
```

Display IDs follow the format `{type}_{sequence}`: `text_1`, `group_2`, `shell_3`, etc.

User-defined variables take precedence:
```python
v = text("main.py")   # v shadows text_1
v.SetState(...)       # Use v
text_1.SetState(...)  # Or use display ID directly
```

## Enums and Constants

### NodeState

Controls rendering detail level for context nodes.

```python
from activecontext import NodeState

NodeState.HIDDEN     # Not shown in projection (but still ticked if running)
NodeState.COLLAPSED  # Title and metadata only (~50 tokens)
NodeState.SUMMARY    # LLM-generated summary
NodeState.DETAILS    # Full view with child settings
NodeState.ALL        # Everything including full traces
```

### TickFrequency

Controls when running nodes update their content.

```python
from activecontext import TickFrequency

TickFrequency.turn()        # Update every turn
TickFrequency.seconds(5)    # Update every 5 seconds
TickFrequency.async_()      # Async execution
TickFrequency.never()       # No automatic updates
TickFrequency.idle()        # Only when explicitly triggered
```

## Context Node Constructors

### `text(path, pos="1:0", end_pos=None, tokens=1000, state=NodeState.DETAILS)`
Create a text view of a file or file region.

```python
v = text("src/main.py")                          # Entire file
v = text("src/main.py", pos="50:0", end_pos="100:0")  # Lines 50-100
v = text("src/main.py", tokens=500, state=NodeState.SUMMARY)
```

### `markdown(path, content=None, tokens=2000, state=NodeState.DETAILS)`
Parse a markdown file into a tree of TextNodes, where each heading section is a separate node.

```python
m = markdown("README.md")                        # Parse file
m = markdown("inline.md", content="# Title\n\nContent")  # Inline content
```

Returns the root TextNode. Child sections are accessible via `children_ids`.

### `view(media_type, path, tokens=2000, state=NodeState.ALL, **kwargs)`
Dispatcher that routes to `text()` or `markdown()` based on media type.

```python
v = view("text", "src/main.py")                  # Same as text()
m = view("markdown", "docs/README.md")           # Same as markdown()
```

### `group(*members, tokens=500, state=NodeState.SUMMARY, summary=None)`
Create a summary group over multiple nodes.

```python
g = group(v1, v2, v3)                    # Group from node objects
g = group("node_id_1", "node_id_2")      # Group from node IDs
g = group(v1, v2, summary="Auth module overview")
```

### `topic(title, tokens=1000, state=NodeState.DETAILS)`
Create a conversation topic/thread marker.

```python
t = topic("Authentication Implementation")
```

### `artifact(content, artifact_type="code", language=None, tokens=500)`
Create an artifact (code snippet, output, error).

```python
a = artifact("def foo(): pass", artifact_type="code", language="python")
a = artifact(error_text, artifact_type="error")
```

## Path Prefixes

Path arguments in node constructors support special prefixes:

### `@prompts/` - Bundled Reference Prompts

Access bundled reference documentation from the ActiveContext package:

```python
m = markdown("@prompts/dsl_reference.md")      # DSL function reference
m = markdown("@prompts/node_states.md")        # NodeState documentation
m = markdown("@prompts/context_graph.md")      # DAG manipulation guide
m = markdown("@prompts/work_coordination.md")  # Multi-agent coordination
m = markdown("@prompts/mcp.md")                # MCP usage guide
```

Available prompts:
- `dsl_reference.md` - This reference document
- `node_states.md` - NodeState enum documentation
- `context_graph.md` - DAG manipulation guide
- `work_coordination.md` - Multi-agent coordination guide
- `mcp.md` - MCP server integration guide
- `context_guide.md` - Context management guide

These paths are automatically resolved to the bundled package content, regardless of the session's working directory.

## Node Methods

All context nodes support chainable configuration methods.

### Common Methods

```python
node.SetState(NodeState.SUMMARY)   # Change rendering state
node.SetTokens(500)                # Change token budget
node.Run(TickFrequency.turn())     # Start running with frequency
node.Pause()                       # Stop automatic updates
node.Refresh()                     # Force immediate update
```

### TextNode Methods

```python
v.SetPos("50:0")      # Jump to line 50
v.SetEndPos("100:0")  # Set end of region
v.Scroll(10)          # Scroll down 10 lines
v.Scroll(-5)          # Scroll up 5 lines
```

### Method Chaining

Methods return `self` for chaining:

```python
v = text("src/main.py").SetTokens(2000).SetState(NodeState.ALL).Run(TickFrequency.turn())
```

## DAG Manipulation

### `link(parent, child)` / `link(parent, child1, child2, ...)`
Add parent-child relationships in the context graph.

```python
link(group_node, view_node)
link(g, v1, v2, v3)  # Link multiple children
```

### `unlink(parent, child)`
Remove a parent-child relationship.

```python
unlink(group_node, view_node)
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
List available checkpoints.

```python
names = checkpoints()  # ["before_refactor", "exploration"]
```

### `branch(from_checkpoint, new_name)`
Create a new checkpoint branching from an existing one.

```python
branch("before_refactor", "refactor_attempt_2")
```

## Shell Execution

### `shell(command, *args, timeout=60, max_output=50000)`
Execute a shell command asynchronously. Returns a ShellNode.

```python
s = shell("pytest", "-v", "tests/")
s = shell("npm", "run", "build", timeout=120)

# Check status
s.is_complete    # True when done
s.is_success     # True if exit_code == 0
s.exit_code      # Exit code
s.output         # stdout/stderr
```

## HTTP Requests

### `fetch(url, method="GET", headers=None, body=None)`
Make an HTTP/HTTPS request (requires permission).

```python
response = await fetch("https://api.example.com/data")
response = await fetch("https://api.example.com/post", method="POST", body=data)
```

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

### `mcp_connect(name, *, command=None, url=None, env=None, tokens=1000, state=NodeState.DETAILS)`
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
<text name="v" path="src/main.py" tokens="3000" state="all"/>
<markdown name="m" path="README.md" tokens="2000"/>
<group name="g" tokens="500" state="summary">
    <member ref="v"/>
</group>
<topic name="t" title="Feature X" tokens="1000"/>
<artifact name="a" content="def foo(): pass" artifact_type="code" language="python"/>
```

### Method Calls

```xml
<!-- self refers to the variable to call the method on -->
<SetState self="v" s="collapsed"/>
<SetTokens self="v" n="500"/>
<SetPos self="v" pos="50:0"/>
<Scroll self="v" delta="10"/>
<Run self="v" freq="turn"/>
<Pause self="v"/>
<Refresh self="v"/>
```

### Shell Execution

```xml
<shell command="pytest" args="tests/,-v" timeout="60"/>
<shell command="git" args="status,--short"/>
```

### DAG Manipulation

```xml
<link child="v" parent="g"/>
<unlink child="v" parent="g"/>
<checkpoint name="before_refactor"/>
<restore name="before_refactor"/>
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
