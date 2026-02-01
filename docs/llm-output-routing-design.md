# LLM Output Routing Design

**Task:** enchanted-goldfinch
**Author:** worker-16
**Date:** 2026-01-30
**Status:** Design Proposal

## Executive Summary

This document proposes a mechanism for routing certain LLM outputs directly to files or nodes, bypassing the context window and projection system. This enables efficient handling of large generated artifacts (code, data, documentation) without consuming token budget.

## Problem Statement

Currently, all LLM output flows through the projection system:

```
LLM → Session → MessageNode → ContextGraph → ProjectionEngine → Context Window
```

**Issues:**
1. Large generated code or data consumes valuable context tokens
2. No way to stream output directly to destination
3. Generated content may never need to be in context (just referenced)
4. Token budget limits size of generated artifacts

**Example Use Case:**
```
User: "Generate a complete Django app with 10 models"
LLM: Generates 2000+ lines of code

Current: All 2000 lines enter context (≈20k tokens)
Desired: Lightweight reference in context (≈100 tokens)
         Full code written directly to files
```

## Requirements

### Functional Requirements

1. **FR1**: LLM can tag output sections for direct routing
2. **FR2**: Routed content bypasses context/projection token counting
3. **FR3**: Context shows lightweight confirmation/reference instead of full content
4. **FR4**: Works with streaming (routing decision made early)
5. **FR5**: Supports routing to:
   - File paths (with permission checks)
   - Existing node IDs (TextNode, ArtifactNode)
   - New nodes (created on-demand)
6. **FR6**: Routed operations are reversible via checkpoint/restore
7. **FR7**: Error handling for permission denied, path not found, etc.

### Non-Functional Requirements

1. **NFR1**: Minimal latency increase for streaming
2. **NFR2**: No breaking changes to existing DSL/API
3. **NFR3**: Clear error messages for malformed routing tags
4. **NFR4**: Auditable (routing events logged in timeline)

## Design Overview

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                         LLM Output                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Session (Agent Loop)                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │    OutputRouter (NEW)                              │     │
│  │    - Parse routing tags from stream                │     │
│  │    - Route content to destinations                 │     │
│  │    - Generate lightweight references               │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│  MessageNode    │            │  Routed Output  │
│  (reference)    │            │  (file/node)    │
└─────────────────┘            └─────────────────┘
```

### Key Components

1. **OutputRouter**: New class in `src/activecontext/session/output_router.py`
2. **Routing Tags**: XML-based syntax in LLM output
3. **RouteRecord**: Audit trail in Timeline statements
4. **Checkpoint Integration**: Extend checkpoint system to track routed writes

## Output Tagging Format

### Syntax

```xml
<route target="path/to/file.py" mode="write">
    def hello():
        return "world"
</route>

<route target="node:artifact_1" mode="append">
    Additional content for existing node
</route>

<route target="new:TextNode" path="src/generated.py">
    Content for new node and file
</route>
```

### Tag Attributes

| Attribute | Required | Values | Description |
|-----------|----------|--------|-------------|
| `target` | Yes | `path`, `node:ID`, `new:NodeType` | Routing destination |
| `mode` | No | `write`, `append`, `replace` | Write mode (default: `write`) |
| `path` | Conditional | file path | Required for `new:` targets |
| `permissions` | No | `auto`, `prompt` | Permission handling (default: `auto`) |
| `checkpoint` | No | `true`, `false` | Enable checkpointing (default: `true`) |

### Examples

#### Direct File Write
```xml
<route target="src/models.py" mode="write">
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
</route>
```

Result in context:
```
✓ Wrote 5 lines to src/models.py (142 tokens bypassed)
```

#### Append to Existing Node
```xml
<route target="node:artifact_1" mode="append">
def new_function():
    pass
</route>
```

Result:
```
✓ Appended 3 lines to artifact_1
```

#### Create New TextNode
```xml
<route target="new:TextNode" path="tests/test_user.py">
import pytest

def test_user_creation():
    assert True
</route>
```

Result:
```
✓ Created text_42 → tests/test_user.py (8 lines)
```

## Implementation Plan

### Phase 1: Core Routing Infrastructure

**File:** `src/activecontext/session/output_router.py`

```python
from dataclasses import dataclass
from typing import AsyncIterator, Literal

@dataclass
class RouteTarget:
    """Parsed route destination."""
    type: Literal["file", "node", "new_node"]
    target: str  # path, node_id, or NodeType
    mode: Literal["write", "append", "replace"]
    path: str | None  # For new nodes
    permissions: Literal["auto", "prompt"]
    checkpoint_enabled: bool

@dataclass
class RouteResult:
    """Result of a routing operation."""
    success: bool
    target: RouteTarget
    bytes_written: int
    lines_written: int
    reference_text: str  # Lightweight text for context
    error: str | None

class OutputRouter:
    """Routes LLM output to files/nodes, bypassing context."""

    def __init__(
        self,
        session_id: str,
        context_graph: ContextGraph,
        permission_manager: PermissionManager,
        cwd: str,
    ):
        self._session_id = session_id
        self._context_graph = context_graph
        self._permission_manager = permission_manager
        self._cwd = cwd
        self._active_routes: dict[str, RouteTarget] = {}

    async def process_stream(
        self,
        stream: AsyncIterator[StreamChunk],
    ) -> AsyncIterator[StreamChunk | RouteResult]:
        """Process LLM stream, detecting and routing tagged content.

        Yields:
            - StreamChunk: Regular content for context
            - RouteResult: Confirmation of routed content
        """
        buffer = ""
        in_route = False
        route_target: RouteTarget | None = None
        route_content: list[str] = []

        async for chunk in stream:
            buffer += chunk.text

            # Detect route start
            if "<route" in buffer and not in_route:
                # Parse route tag
                route_target = self._parse_route_tag(buffer)
                if route_target:
                    in_route = True
                    # Yield any content before route tag
                    prefix = buffer.split("<route")[0]
                    if prefix:
                        yield StreamChunk(text=prefix)
                    buffer = ""
                    continue

            # Collect route content
            if in_route:
                if "</route>" in buffer:
                    # End of routed content
                    content = buffer.split("</route>")[0]
                    route_content.append(content)

                    # Execute routing
                    result = await self._route_content(
                        route_target,
                        "".join(route_content)
                    )
                    yield result

                    # Reset state
                    in_route = False
                    route_content = []
                    buffer = buffer.split("</route>", 1)[1]
                else:
                    route_content.append(buffer)
                    buffer = ""
            else:
                # Regular content - yield chunks
                if len(buffer) > 100:  # Buffer threshold
                    yield StreamChunk(text=buffer)
                    buffer = ""

        # Flush remaining buffer
        if buffer:
            yield StreamChunk(text=buffer)

    def _parse_route_tag(self, text: str) -> RouteTarget | None:
        """Parse <route> tag attributes."""
        import re

        match = re.search(
            r'<route\s+target="([^"]+)"(?:\s+mode="([^"]+)")?(?:\s+path="([^"]+)")?',
            text
        )
        if not match:
            return None

        target = match.group(1)
        mode = match.group(2) or "write"
        path = match.group(3)

        # Determine target type
        if target.startswith("node:"):
            target_type = "node"
            target = target[5:]  # Strip "node:" prefix
        elif target.startswith("new:"):
            target_type = "new_node"
            target = target[4:]  # Strip "new:" prefix
        else:
            target_type = "file"

        return RouteTarget(
            type=target_type,
            target=target,
            mode=mode,
            path=path,
            permissions="auto",
            checkpoint_enabled=True,
        )

    async def _route_content(
        self,
        target: RouteTarget,
        content: str,
    ) -> RouteResult:
        """Execute routing to file or node."""
        try:
            if target.type == "file":
                return await self._route_to_file(target, content)
            elif target.type == "node":
                return await self._route_to_node(target, content)
            elif target.type == "new_node":
                return await self._route_to_new_node(target, content)
        except Exception as e:
            return RouteResult(
                success=False,
                target=target,
                bytes_written=0,
                lines_written=0,
                reference_text=f"✗ Failed to route to {target.target}: {e}",
                error=str(e),
            )

    async def _route_to_file(
        self,
        target: RouteTarget,
        content: str,
    ) -> RouteResult:
        """Write content directly to a file."""
        import os

        path = os.path.join(self._cwd, target.target)

        # Permission check
        permission_granted = await self._permission_manager.check_permission(
            path,
            mode="write"
        )
        if not permission_granted:
            raise PermissionError(f"Write permission denied: {path}")

        # Write file
        mode = "w" if target.mode == "write" else "a"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)

        lines = len(content.splitlines())
        bytes_written = len(content.encode())

        return RouteResult(
            success=True,
            target=target,
            bytes_written=bytes_written,
            lines_written=lines,
            reference_text=f"✓ Wrote {lines} lines to {target.target}",
            error=None,
        )

    async def _route_to_node(
        self,
        target: RouteTarget,
        content: str,
    ) -> RouteResult:
        """Route content to an existing node."""
        node = self._context_graph.get_node(target.target)
        if node is None:
            raise ValueError(f"Node not found: {target.target}")

        if isinstance(node, ArtifactNode):
            if target.mode == "append":
                node.content += "\n" + content
            else:
                node.content = content
            node._mark_changed(description="Content routed from LLM")
        elif isinstance(node, TextNode):
            # For TextNode, write to the file it references
            return await self._route_to_file(
                RouteTarget(
                    type="file",
                    target=node.path,
                    mode=target.mode,
                    path=None,
                    permissions=target.permissions,
                    checkpoint_enabled=target.checkpoint_enabled,
                ),
                content
            )
        else:
            raise ValueError(f"Cannot route to node type: {type(node).__name__}")

        lines = len(content.splitlines())
        return RouteResult(
            success=True,
            target=target,
            bytes_written=len(content.encode()),
            lines_written=lines,
            reference_text=f"✓ Routed {lines} lines to {target.target}",
            error=None,
        )

    async def _route_to_new_node(
        self,
        target: RouteTarget,
        content: str,
    ) -> RouteResult:
        """Create a new node and route content to it."""
        if target.target == "TextNode":
            if not target.path:
                raise ValueError("TextNode requires path attribute")

            # Create TextNode
            node = TextNode(path=target.path)
            self._context_graph.add_node(node)

            # Write file
            file_result = await self._route_to_file(
                RouteTarget(
                    type="file",
                    target=target.path,
                    mode="write",
                    path=None,
                    permissions=target.permissions,
                    checkpoint_enabled=target.checkpoint_enabled,
                ),
                content
            )

            return RouteResult(
                success=file_result.success,
                target=target,
                bytes_written=file_result.bytes_written,
                lines_written=file_result.lines_written,
                reference_text=f"✓ Created {node.node_id} → {target.path} ({file_result.lines_written} lines)",
                error=file_result.error,
            )
        elif target.target == "ArtifactNode":
            # Create ArtifactNode
            node = ArtifactNode(content=content, artifact_type="code")
            self._context_graph.add_node(node)

            lines = len(content.splitlines())
            return RouteResult(
                success=True,
                target=target,
                bytes_written=len(content.encode()),
                lines_written=lines,
                reference_text=f"✓ Created {node.node_id} ({lines} lines)",
                error=None,
            )
        else:
            raise ValueError(f"Unknown node type: {target.target}")
```

### Phase 2: Session Integration

**File:** `src/activecontext/session/agent.py`

Modifications to `run_agent_loop()`:

```python
async def run_agent_loop(self, has_pending_work: Any = None) -> AsyncIterator[SessionUpdate]:
    # ... existing code ...

    # Create output router
    router = OutputRouter(
        session_id=self._agent_id,
        context_graph=self._context_graph,
        permission_manager=self._permission_manager,
        cwd=self._cwd,
    )

    # Stream LLM output through router
    stream = self._llm.stream(messages, max_tokens=max_tokens)
    routed_stream = router.process_stream(stream)

    collected_text = []

    async for item in routed_stream:
        if isinstance(item, StreamChunk):
            # Regular content - add to context
            collected_text.append(item.text)
            yield SessionUpdate(
                type="content",
                content=item.text,
            )
        elif isinstance(item, RouteResult):
            # Routed content - add reference to context
            collected_text.append(item.reference_text)
            yield SessionUpdate(
                type="route_result",
                content=item.reference_text,
                metadata={
                    "target": item.target.target,
                    "lines": item.lines_written,
                    "bytes": item.bytes_written,
                    "success": item.success,
                    "error": item.error,
                }
            )

    # Store agent response with routed references
    response_text = "".join(collected_text)
    self._add_message(Role.ASSISTANT, response_text, originator="agent")
```

### Phase 3: Timeline Audit Trail

**File:** `src/activecontext/session/timeline.py`

```python
@dataclass
class RouteRecord:
    """Record of a routing operation for audit trail."""
    timestamp: float
    target_type: str
    target: str
    mode: str
    lines_written: int
    bytes_written: int
    success: bool
    error: str | None
    checkpoint_ref: str | None

class Timeline:
    def __init__(self, ...):
        # ... existing code ...
        self._route_records: list[RouteRecord] = []

    def record_route(self, result: RouteResult) -> None:
        """Record a routing operation in the timeline."""
        record = RouteRecord(
            timestamp=time.time(),
            target_type=result.target.type,
            target=result.target.target,
            mode=result.target.mode,
            lines_written=result.lines_written,
            bytes_written=result.bytes_written,
            success=result.success,
            error=result.error,
            checkpoint_ref=None,  # Set during checkpoint
        )
        self._route_records.append(record)
```

### Phase 4: Checkpoint Integration

**File:** `src/activecontext/context/graph.py`

Extend checkpoint system to track file writes:

```python
@dataclass
class RouteCheckpoint:
    """Checkpoint entry for routed file writes."""
    target: str
    backup_path: str  # Temporary backup for rollback
    original_exists: bool
    original_content: str | None

class ContextGraph:
    def checkpoint(self, label: str) -> bool:
        # ... existing code ...

        # Backup files written via routing
        route_backups: list[RouteCheckpoint] = []
        for record in self._timeline._route_records:
            if record.checkpoint_ref is None:
                backup = self._backup_routed_file(record.target)
                route_backups.append(backup)

        checkpoint = Checkpoint(
            label=label,
            timestamp=time.time(),
            graph_state=self._export_state(),
            route_backups=route_backups,
        )
        # ... store checkpoint ...

    def restore(self, label: str) -> bool:
        checkpoint = self._checkpoints.get(label)
        if not checkpoint:
            return False

        # Restore graph state
        self._import_state(checkpoint.graph_state)

        # Restore routed files
        for backup in checkpoint.route_backups:
            self._restore_routed_file(backup)

        return True
```

## Permission Handling

### Permission Checks

1. **File Writes**: Use existing `PermissionManager.check_permission(path, mode="write")`
2. **Node Writes**: Check if node exists and is writable
3. **Auto Permission**: `permissions="auto"` uses existing grants
4. **Prompt Permission**: `permissions="prompt"` triggers ACP permission request

### Permission Flow

```
Router detects file write
    ↓
Check PermissionManager cache
    ↓
Permission granted? → Write file → Record audit
    ↓
Permission denied? → Yield RouteResult(error="Permission denied")
```

## Error Handling

### Error Types

1. **Parse Error**: Malformed `<route>` tag
   - Result: Treat as regular text, emit warning in context

2. **Permission Error**: Write denied
   - Result: `RouteResult(success=False, error="Permission denied")`

3. **Target Not Found**: Node ID doesn't exist
   - Result: `RouteResult(success=False, error="Node not found: {id}")`

4. **File System Error**: Disk full, path invalid
   - Result: `RouteResult(success=False, error="{filesystem_error}")`

### Error Recovery

```python
try:
    result = await router._route_content(target, content)
    yield result
except PermissionError as e:
    yield RouteResult(
        success=False,
        error=str(e),
        reference_text=f"✗ Permission denied: {target.target}",
    )
except Exception as e:
    yield RouteResult(
        success=False,
        error=str(e),
        reference_text=f"✗ Routing failed: {e}",
    )
    # Continue processing stream
```

## Token Budget Handling

### Current Behavior

```
MessageNode → ProjectionEngine → counts all tokens
```

### With Routing

```
MessageNode (reference text only) → ProjectionEngine → counts ~100 tokens
Routed content (not in MessageNode) → bypasses projection
```

### Implementation

```python
class MessageNode:
    def get_token_breakdown(self) -> TokenInfo:
        # Only count the reference text, not routed content
        content = self.content  # Contains "✓ Wrote 50 lines to..." not the actual lines
        return TokenInfo(
            summary=count_tokens(content),
            detail=0,
        )
```

## LLM Prompt Integration

### System Prompt Addition

Add to `src/activecontext/resources/prompts/system.md`:

```markdown
### Output Routing

You can route large outputs directly to files or nodes using `<route>` tags:

**File Write:**
```xml
<route target="path/to/file.py" mode="write">
    your code here
</route>
```

**Append to Node:**
```xml
<route target="node:artifact_1" mode="append">
    additional content
</route>
```

**Create New Node:**
```xml
<route target="new:TextNode" path="tests/test.py">
    test code
</route>
```

Routed content bypasses the context window and is written directly to the destination.
A lightweight reference appears in the context instead.
```

## Testing Strategy

### Unit Tests

**File:** `tests/test_output_router.py`

```python
class TestOutputRouter:
    async def test_parse_route_tag(self):
        """Test route tag parsing."""

    async def test_route_to_file(self):
        """Test direct file write."""

    async def test_route_to_node(self):
        """Test routing to existing node."""

    async def test_permission_denied(self):
        """Test permission error handling."""

    async def test_streaming_detection(self):
        """Test route detection in stream."""

    async def test_checkpoint_rollback(self):
        """Test checkpoint/restore with routed files."""
```

### Integration Tests

**File:** `tests/test_routing_integration.py`

```python
class TestRoutingIntegration:
    async def test_end_to_end_file_write(self):
        """Test complete flow from LLM to file."""

    async def test_mixed_content_stream(self):
        """Test stream with both regular and routed content."""

    async def test_multiple_routes_in_response(self):
        """Test response with multiple route tags."""
```

## Migration Path

### Phase 1: Infrastructure (Week 1)
- Implement `OutputRouter` class
- Add unit tests
- No breaking changes

### Phase 2: Session Integration (Week 1-2)
- Integrate router into `Agent.run_agent_loop()`
- Add timeline audit trail
- Update system prompts

### Phase 3: Checkpoint Integration (Week 2)
- Extend checkpoint system for file backups
- Add restore logic
- Integration tests

### Phase 4: Documentation & Refinement (Week 3)
- Update DSL reference
- Add examples to guides
- Performance profiling

## Performance Considerations

### Latency

- **Route Tag Parsing**: Regex match on buffered chunks (~1ms)
- **File Write**: OS-dependent (10-100ms for small files)
- **Node Lookup**: O(1) hash lookup (<1ms)

**Total overhead per route:** ~10-100ms (minimal for large content)

### Memory

- **Buffer Size**: Max 1MB per route tag (configurable)
- **Checkpoint Backups**: One backup file per routed write

### Throughput

- **Streaming**: Route detection doesn't block stream
- **Concurrent Routes**: Process routes sequentially (simplicity > parallelism)

## Security Considerations

1. **Path Traversal**: Validate file paths are within cwd or allowed paths
2. **Permission Bypass**: All routes go through PermissionManager
3. **Arbitrary Code**: Routed content is data, not executed
4. **Quota Limits**: Track total bytes written per session

## Open Questions

1. **Should routes support wildcards?** (e.g., `target="*.py"`)
   - Recommendation: No, too complex for v1

2. **How to handle binary content?**
   - Recommendation: Base64 encoding in `<route>` tag

3. **Should routes be cancellable mid-stream?**
   - Recommendation: Yes, via session cancel propagation

## Alternative Designs Considered

### 1. JSON-Based Routing

**Format:**
```json
{
  "type": "route",
  "target": "path/to/file.py",
  "content": "..."
}
```

**Pros:** Machine-readable, structured
**Cons:** Harder for LLM to emit mid-response, requires JSON escaping

### 2. Markdown Code Blocks with Metadata

**Format:**
```
```python path="src/foo.py"
def hello():
    pass
```

**Pros:** Natural for LLMs, human-readable
**Cons:** Ambiguous parsing, conflicts with documentation

### 3. Dedicated DSL Commands

**Format:**
```
@write src/foo.py
def hello():
    pass
@end
```

**Pros:** Simple parser
**Cons:** Less structured, harder to extend

**Decision:** XML tags provide the best balance of structure, extensibility, and LLM compatibility.

## Success Metrics

1. **Token Savings**: Measure context token reduction for large generations
2. **Latency**: Router overhead <100ms per route
3. **Adoption**: Track usage in agent loops
4. **Error Rate**: <1% routing failures in production

## Conclusion

This design provides a clean, extensible mechanism for LLM output routing that:
- ✅ Bypasses context window for large outputs
- ✅ Integrates with existing permission system
- ✅ Supports checkpointing and rollback
- ✅ Maintains audit trail
- ✅ Works with streaming
- ✅ Has clear error handling

The implementation is backward compatible and can be deployed incrementally.

## References

- ActiveContext Architecture: `CLAUDE.md`
- DSL Reference: `src/activecontext/prompts/dsl_reference.md`
- Session Protocol: `src/activecontext/session/protocols.py`
- Permission System: `src/activecontext/session/permissions.py`
