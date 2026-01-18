# DSL Reference

Functions available in the ActiveContext Timeline namespace.

## Context Node Constructors

### `view(path, pos="1:0", end_pos=None, tokens=1000, state=NodeState.DETAILS)`
Create a view of a file or file region.

```python
v = view("src/main.py")                          # Entire file
v = view("src/main.py", pos="50:0", end_pos="100:0")  # Lines 50-100
v = view("src/main.py", tokens=500, state=NodeState.SUMMARY)
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
handles = ls()  # {"v": ViewNode(...), "g": GroupNode(...)}
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
