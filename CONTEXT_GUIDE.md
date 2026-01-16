# ActiveContext - Context Guide

This guide explains how to use the context system to work with files and code.

## Code Execution

Use ` ```python/acrepl ` blocks for code that will be executed:

```python/acrepl
v = view("src/__main__.py", tokens=2000)
```

Regular ` ```python ` blocks are for showing examples (not executed).

## Views

A **view** is a window into a file. The view appears in your context on the next turn, showing the file content with line numbers.

### View Parameters

| Parameter | Default    | Description                                      |
|-----------|------------|--------------------------------------------------|
| `path`    | required   | File path (relative to session cwd)              |
| `pos`     | `"0:0"`    | Start position as `"line:col"`                   |
| `tokens`  | `2000`     | Token budget for content                         |
| `lod`     | `0`        | Level of detail (0=raw, 1=structured, 2=summary) |
| `mode`    | `"paused"` | `"paused"` or `"running"`                        |

### View Methods

```python/acrepl
v.SetPos("50:0")      # Jump to line 50
v.SetTokens(500)      # Reduce token budget
v.SetLod(1)           # Switch to structured view
v.Run()               # Enable auto-updates each turn
v.Pause()             # Disable auto-updates
```

Methods are chainable:
```python/acrepl
v.SetPos("100:0").SetTokens(1000).Run()
```

## Groups

A **group** summarizes multiple views:

```python/acrepl
g = group(v1, v2, v3, tokens=500, lod=2)
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
main = view("src/main.py", tokens=3000)

# On the next turn, you'll see the file content
# Then you can adjust the view:
main.SetPos("50:0").SetTokens(1000)

# Create views of related files
utils = view("src/utils.py", tokens=1000)
config = view("config.yaml", tokens=500)

# Group them for a summary
overview = group(main, utils, config, lod=2)
```

## How Context Works

1. You execute `view("file.py")` - this creates a ViewHandle
2. On the next prompt, the Projection includes the rendered file content
3. The LLM sees the file with line numbers in its context
4. You can adjust views, and changes appear in the next projection

The projection is token-managed: views share the available budget based on their `tokens` parameter.
