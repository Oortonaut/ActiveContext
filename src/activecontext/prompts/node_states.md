# Node States and Tick Frequency

## NodeState Enum

Controls how a node renders in the projection.

| State | Description | Use Case |
|-------|-------------|----------|
| `HIDDEN` | Not rendered at all | Temporary exclusion, completed work |
| `COLLAPSED` | Metadata only (title, trace count) | Background context, low priority |
| `SUMMARY` | LLM-generated summary | Groups, large files |
| `DETAILS` | Full content with child settings | Active work files |
| `ALL` | Everything including pending traces | Debugging, detailed review |

```python
from activecontext import NodeState

v = text("src/main.py", state=NodeState.DETAILS)
v.SetState(NodeState.SUMMARY)  # Reduce detail
v.SetState(NodeState.HIDDEN)   # Hide completely
```

## TickFrequency

Controls when a running node recomputes.

| Frequency | Description |
|-----------|-------------|
| `TickFrequency.turn()` | Recompute every turn (default) |
| `TickFrequency.seconds(n)` | Recompute every n seconds |
| `TickFrequency.async_()` | Async execution |
| `TickFrequency.never()` | No automatic updates |
| `TickFrequency.idle()` | Only recompute when explicitly triggered |

```python
from activecontext import TickFrequency

v.Run(TickFrequency.turn())       # Update every turn
v.Run(TickFrequency.seconds(30))  # Update every 30 seconds
v.Run(TickFrequency.async_())     # Async execution
v.Run()                           # Default: turn()
```

## Fluent API

All context nodes support method chaining:

```python
v = text("src/main.py") \
    .SetState(NodeState.DETAILS) \
    .SetTokens(2000) \
    .Run(TickFrequency.turn())

v.Pause()  # Stop automatic updates
```

### Available Methods

| Method | Description |
|--------|-------------|
| `.SetState(state)` | Set rendering state |
| `.SetTokens(n)` | Set token budget |
| `.Run(freq=None)` | Enable tick updates |
| `.Pause()` | Disable tick updates |

### TextNode Additional Methods

| Method | Description |
|--------|-------------|
| `.SetPos(pos)` | Set start position ("line:col") |
| `.SetEndPos(end_pos)` | Set end position |

## Mode: Running vs Paused

Nodes have a `mode` that controls tick participation:

- **`"paused"`** (default): Node does not recompute automatically
- **`"running"`**: Node recomputes on tick according to its `TickFrequency`

```python
v.Run()    # mode = "running"
v.Pause()  # mode = "paused"
```

Running nodes:
- Refresh their content each tick
- Notify parent nodes of changes
- Consume processing time

Use `Run()` for files that change frequently during a task. Use `Pause()` for reference material.
