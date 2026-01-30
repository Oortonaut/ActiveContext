# Node States and Tick Frequency

## Expansion Enum

Controls how a node renders in the projection.

| State | Description | Use Case |
|-------|-------------|----------|
| `HEADER` | Metadata only (title, trace count) | Background context, low priority |
| `CONTENT` | Main content/summary | Groups, large files |
| `INDEX` | Content plus section headings | Navigation, quick reference |
| `ALL` | Full content with all details | Active work files |

```python
from activecontext import Expansion

v = text("src/main.py", expansion=Expansion.ALL)
v.expansion = Expansion.CONTENT  # Reduce detail
v.expansion = Expansion.HEADER   # Show minimal info
```

## Visibility (hide/unhide)

Controls whether a node appears in the projection at all.

```python
hide(text_1)              # Remove from projection (node still ticks)
unhide(text_1)            # Restore to previous expansion
unhide(text_1, expand=Expansion.CONTENT)  # Restore with specific expansion
```

Hidden nodes:
- Do not appear in the projection
- Continue to tick if mode="running"
- Retain all state for restoration

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
v = text("src/main.py")
v.expansion = Expansion.ALL
v.Run(TickFrequency.turn())

v.Pause()  # Stop automatic updates
```

### Available Fields and Methods

| Field/Method | Description |
|--------------|-------------|
| `.expansion = ...` | Set rendering expansion |
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
