# MCP (Model Context Protocol) Reference

Connect to external MCP servers to access their tools and resources.

## Connecting to Servers

### `mcp_connect(name, *, command=None, url=None, env=None, tokens=1000, state=NodeState.DETAILS)`
Connect to an MCP server. Returns an MCPServerNode.

```python
# Connect to configured server (from config.yaml)
fs = mcp_connect("filesystem")

# Dynamic stdio connection (spawns subprocess)
gh = mcp_connect("github", command=["npx", "-y", "@modelcontextprotocol/server-github"])

# Dynamic HTTP connection
tools = mcp_connect("remote", url="http://localhost:8000/mcp")

# With environment variables
gh = mcp_connect("github",
    command=["npx", "-y", "@mcp/server-github"],
    env={"GITHUB_TOKEN": "ghp_..."})
```

### `mcp_disconnect(name)`
Disconnect from an MCP server.

```python
mcp_disconnect("filesystem")
```

## Calling Tools

Once connected, tools are available as methods on the server proxy:

```python
fs = mcp_connect("filesystem")

# Call tools directly
result = fs.read_file(path="/home/user/data.txt")
result = fs.write_file(path="/home/user/out.txt", content="Hello")

# Results contain content and metadata
print(result.text())              # Extract text content
print(result.content)             # Raw content blocks
print(result.is_error)            # True if tool call failed
print(result.error_message)       # Error details if failed
print(result.structured_content)  # Structured JSON if available
```

## Introspection

### `mcp_list()`
List all connected MCP servers.

```python
servers = mcp_list()
# [{"name": "filesystem", "status": "connected", "tools": 5, "resources": 2}]
```

### `mcp_tools(server=None)`
List available tools, optionally filtered by server.

```python
# All tools from all servers
all_tools = mcp_tools()

# Tools from specific server
fs_tools = mcp_tools("filesystem")
# [{"server": "filesystem", "name": "read_file", "description": "..."}]
```

## MCPServerNode Rendering

When connected, the MCPServerNode appears in your context with tool documentation:

- **COLLAPSED**: `MCP: filesystem [OK] (5 tools)`
- **SUMMARY**: Server + tool names list
- **DETAILS**: Tool names + descriptions + usage hint
- **ALL**: Full documentation with parameter schemas

```python
fs = mcp_connect("filesystem")
fs.SetState(NodeState.ALL)      # Full tool documentation
fs.SetState(NodeState.SUMMARY)  # Just tool names
```

## Configuration

Define servers in `config.yaml`:

```yaml
mcp:
  allow_dynamic_servers: true
  servers:
    - name: filesystem
      command: ["npx", "-y", "@modelcontextprotocol/server-filesystem"]
      args: ["/home/user/allowed"]
      connect: auto

    - name: github
      command: ["npx", "-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_TOKEN: "${GITHUB_TOKEN}"
      connect: manual

    - name: remote-api
      url: "http://localhost:8000/mcp"
      transport: streamable-http
```

- **connect**: Connection mode
  - `critical`: Must connect on startup, fail session if cannot
  - `auto`: Auto-connect on startup, warn if fails
  - `manual`: Only connect via `mcp_connect()` (default)
  - `never`: Disabled, cannot be connected
- **env**: Environment variables (supports `${VAR}` expansion)
- **transport**: `stdio` (default), `streamable-http`, or `sse`

## Common MCP Servers

```python
# Filesystem access
fs = mcp_connect("fs", command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path"])

# GitHub
gh = mcp_connect("gh", command=["npx", "-y", "@modelcontextprotocol/server-github"])

# Fetch URLs
fetch = mcp_connect("fetch", command=["npx", "-y", "@modelcontextprotocol/server-fetch"])

# Brave Search
brave = mcp_connect("brave", command=["npx", "-y", "@anthropic/server-brave-search"])
```

## Error Handling

```python
result = fs.read_file(path="/nonexistent")

if result.is_error:
    print(f"Error: {result.error_message}")
else:
    print(result.text())
```
