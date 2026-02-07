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

v = text("src/main.py", default_expansion=Expansion.ALL)
v.expansion = Expansion.CONTENT  # Reduce detail (on the view)
v.expansion = Expansion.HEADER   # Show minimal info
```

## Visibility (NodeView.hidden)

Controls whether a node appears in the projection at all.

```python
v = text("main.py")      # Returns NodeView
v.hidden = True          # Hide from projection (node still ticks)
v.hidden = False         # Restore to projection

# Check visibility
if not v.hidden:
    print("Node is visible")
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
| `TickFrequency.period(n)` | Recompute every n seconds |
| `TickFrequency.async_()` | Async execution |
| `TickFrequency.never()` | No automatic updates |

```python
from activecontext import TickFrequency

v.Run(TickFrequency.turn())       # Update every turn
v.Run(TickFrequency.period(30))   # Update every 30 seconds
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
