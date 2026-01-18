# Work Coordination (Scratchpad)

Multi-agent coordination system for communicating which areas of the codebase agents are working on.

## Overview

The scratchpad is a shared YAML file (`.ac/scratchpad.yaml`) where agents register their work areas. It is **advisory only** - warns about conflicts but doesn't block.

## Basic Usage

### Register Work Area
```python
work_on("Implementing OAuth2", "src/auth/oauth.py", "src/auth/config.py")
```

This:
- Creates an entry in the scratchpad
- Creates a WorkNode showing your work status
- Checks for conflicts with other agents

### Check for Conflicts
Before modifying a shared file:

```python
conflicts = work_check("src/utils.py")
if conflicts:
    print(f"Warning: Agent {conflicts[0]['agent_id']} is also working here")
    print(f"Their intent: {conflicts[0]['their_intent']}")
```

### Update Status
```python
work_update(intent="OAuth2: Adding token refresh")
work_update(status="paused")  # Pausing work
```

### Complete Work
```python
work_done()  # Removes your entry from scratchpad
```

### List All Work
```python
entries = work_list()
for e in entries:
    print(f"{e['agent_id']}: {e['intent']} ({e['status']})")
```

## Conflict Rules

| You Want | They Have | Conflict? |
|----------|-----------|-----------|
| write | write | Yes |
| write | read | Yes |
| read | write | Yes |
| read | read | No |

Glob patterns are supported:
```python
work_on("Auth refactor", "src/auth/*.py")  # Claims all .py files in auth/
```

## File Format

`.ac/scratchpad.yaml`:
```yaml
version: 1
entries:
  - id: "a1b2c3d4"
    session_id: "session-uuid"
    intent: "Implementing OAuth2"
    status: active
    files:
      - path: "src/auth/oauth.py"
        mode: write
    dependencies: []
    started_at: "2026-01-17T10:30:00Z"
    updated_at: "2026-01-17T10:35:00Z"
    heartbeat_at: "2026-01-17T10:35:00Z"
```

## Stale Entry Cleanup

- Entries without heartbeat for 5 minutes are automatically removed
- Heartbeat updates on every operation
- Cleanup runs opportunistically during register/update

## WorkNode

When you call `work_on()`, a WorkNode is created in the context graph showing:
- Your current intent
- Files being worked on
- Any detected conflicts

The node renders in the projection so you can see coordination status.

## Best Practices

1. **Register early**: Call `work_on()` before starting significant changes
2. **Be specific**: List actual files, not entire directories
3. **Check before writing**: Use `work_check()` before modifying shared code
4. **Update intent**: Keep your intent current with `work_update()`
5. **Clean up**: Call `work_done()` when finished

## Advisory Nature

The system is advisory only:
- Conflicts are warnings, not blocks
- Agents can still modify files even with conflicts
- Communication is the goal, not enforcement

This allows flexibility while providing visibility into what other agents are doing.
