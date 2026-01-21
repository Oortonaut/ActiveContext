# ActiveContext - Context Guide

This guide explains how to use the context system to work with files and code.

## Code Execution

Use ` ```python/acrepl ` blocks for code that will be executed:

```python/acrepl
v = text("src/__main__.py", tokens=2000)
```

Regular ` ```python ` blocks are for showing examples (not executed).

## Node Access

Every node has a display ID (e.g., `text_1`, `group_2`) that can be used directly:

```python/acrepl
v = text("src/main.py")    # Creates text_1
text_1.SetState(NodeState.SUMMARY)  # Both work
v.SetState(NodeState.SUMMARY)       # Same effect
```

Display IDs follow the pattern `{type}_{sequence}`. Useful when you need to reference a node without storing it in a variable.

## Text Views

A **text view** is a window into a file. The view appears in your context on the next turn, showing the file content with line numbers.

### Text Parameters

| Parameter | Default    | Description                                      |
|-----------|------------|--------------------------------------------------|
| `path`    | required   | File path (relative to session cwd)              |
| `pos`     | `"1:0"`    | Start position as `"line:col"`                   |
| `end_pos` | `None`     | End position as `"line:col"` (limits view range) |
| `tokens`  | `2000`     | Token budget for content                         |
| `state`   | `DETAILS`  | Rendering state (HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL) |
| `mode`    | `"paused"` | `"paused"` or `"running"`                        |

### Text Methods

```python/acrepl
v.SetPos("50:0")      # Jump to line 50
v.SetEndPos("100:0")  # Limit view to lines 50-100
v.SetTokens(500)      # Reduce token budget
v.SetState(NodeState.SUMMARY)  # Switch to summary view
v.Run()               # Enable auto-updates each turn
v.Pause()             # Disable auto-updates
```

Methods are chainable:
```python/acrepl
v.SetPos("100:0").SetTokens(1000).Run()
```

## Markdown Files

For markdown files, use `markdown()` to parse heading structure into a tree of TextNodes:

```python/acrepl
m = markdown("README.md")  # Parse into heading tree
```

Each heading section becomes a separate TextNode with its line range.

## Groups

A **group** summarizes multiple views:

```python/acrepl
g = group(v1, v2, v3, tokens=500, state=NodeState.SUMMARY)
```

Groups are useful for maintaining awareness of related files without consuming too many tokens.

## Utilities

```python/acrepl
ls()        # List all context handles
show(v)     # Force-render a specific view
```

## Completing Tasks

When you have finished the user's request, call `done()` with a summary:

```python/acrepl
done("I've analyzed the code and found the bug on line 42. The issue is...")
```

This signals that you've completed the task. The message is your final response to the user.

## Example Session

```python/acrepl
# First, create a view of the main file
main = text("src/main.py", tokens=3000)

# On the next turn, you'll see the file content
# Then you can adjust the view:
main.SetPos("50:0").SetTokens(1000)

# Create views of related files
utils = text("src/utils.py", tokens=1000)
config = text("config.yaml", tokens=500)

# Group them for a summary
overview = group(main, utils, config, state=NodeState.SUMMARY)
```

## How Context Works

1. You execute `text("file.py")` - this creates a TextNode
2. On the next prompt, the Projection includes the rendered file content
3. The LLM sees the file with line numbers in its context
4. You can adjust views, and changes appear in the next projection

The projection is token-managed: views share the available budget based on their `tokens` parameter.
