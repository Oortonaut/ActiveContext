# Session Class Refactoring Design

## Executive Summary

The Session class in `session_manager.py` is 1,897 lines (lines 78-1975) with 50+ methods. This document proposes extracting two focused components to improve maintainability while preserving Session's core role as orchestrator.

**Recommendation**: Extract projection and agent loop logic into helper classes while keeping Session as the primary coordinator. This is lower priority than Timeline refactoring as Session is more cohesive than initially assessed.

## Current State Analysis

### Session Class Overview

**Size**: 1,897 lines, 50+ methods
**Primary Responsibilities**:
1. Session lifecycle and persistence
2. Context graph coordination (owns root, manages stack)
3. Message queue and history management
4. LLM integration and prompt processing
5. Projection building and rendering
6. Event-driven agent loop
7. Multi-task coordination (Phase 5)
8. Text buffer management
9. Path resolution (@prompts/, roots)
10. Mode management (ChoiceView-based)

### Method Categorization

**Lifecycle & Metadata** (~15 methods, ~200 lines):
- `__init__`, `save`, `from_file`
- `_add_system_prompt_node`, `_create_metadata_nodes`
- `_restore_system_prompt_buffer`
- Properties: `session_id`, `cwd`, `title`, `created_at`, `mode`
- Setters: `set_title`, `set_mode`, `set_llm`

**Projection & Rendering** (~5 methods, ~100 lines):
- `_build_projection_engine` (4 lines - trivial factory)
- `get_projection` (20 lines - delegates to ProjectionEngine)
- `_update_alerts_group` (30 lines)
- Related to context presentation

**Agent Loop** (~6 methods, ~200 lines):
- `run_agent_loop` (40 lines - main loop)
- `_process_next_message` (65 lines - LLM processing)
- `_has_pending_work` (6 lines - work detection)
- `stop_agent_loop` (4 lines - shutdown)
- `wake` (3 lines - wake signal)
- `queue_user_message` (40 lines - message queueing)

**Message Management** (~8 methods, ~150 lines):
- `queue_user_message`, `has_pending_messages`, `get_pending_messages`
- `mark_message_processed`, `_add_message`
- `clear_message_history`

**Context Graph Coordination** (~8 methods, ~150 lines):
- `current_group`, `push_group`, `pop_group`
- `add_node`, `begin_tool_use`, `end_tool_use`
- `get_context_objects`, `get_context_graph`

**Path Resolution** (~4 methods, ~120 lines):
- `resolve_path` (50 lines - complex @prompts/ and root logic)
- `_expand_path_roots` (30 lines)
- Helper functions: `strip_leading_seps`, `is_path_boundary`

**Text Buffers** (~5 methods, ~50 lines):
- `get_text_buffer`, `add_text_buffer`, `get_or_create_text_buffer`
- Property: `text_buffers`

**Multi-Task Support** (~5 methods, ~100 lines):
- `get_task`, `list_tasks`, `create_task`
- `create_conversation_handle`, `delegate_conversation`

**LLM Processing** (~10 methods, ~800 lines):
- `prompt`, `_prompt_with_llm`, `_prompt_direct`
- `startup`, `tick`
- Large complex methods with streaming logic

## Proposed Refactoring

### Option A: Extract SessionRenderer + AgentLoopRunner (Original Proposal)

#### 1. SessionRenderer

**Purpose**: Encapsulate projection building and alerts management

**Methods to Extract**:
- `_build_projection_engine()` (4 lines)
- `get_projection()` (20 lines)
- `_update_alerts_group()` (30 lines)

**Estimated Size**: ~100 lines (including infrastructure)

**Interface**:
```python
class SessionRenderer:
    """Handles projection building and alerts management."""

    def __init__(
        self,
        context_graph: ContextGraph,
        config: Config | None = None,
        alerts_group: GroupNode | None = None
    ):
        self._context_graph = context_graph
        self._projection_engine = self._build_projection_engine(config)
        self._alerts_group = alerts_group

    def get_projection(self) -> Projection:
        """Build current context projection."""
        ...

    def update_alerts(self, notifications: list[Notification]) -> None:
        """Update alerts group with new notifications."""
        ...

    def _build_projection_engine(self, config: Config | None) -> ProjectionEngine:
        """Factory for ProjectionEngine."""
        ...
```

**Pros**:
- Isolates projection logic
- Clear single responsibility
- Easy to test independently

**Cons**:
- Very small extraction (~100 lines)
- Minimal complexity reduction
- `get_projection()` is only 20 lines and mostly delegates
- Weak cohesion - alerts and projection aren't strongly related

#### 2. AgentLoopRunner

**Purpose**: Encapsulate event-driven agent loop mechanics

**Methods to Extract**:
- `run_agent_loop()` (40 lines)
- `_process_next_message()` (65 lines)
- `_has_pending_work()` (6 lines)
- `stop_agent_loop()` (4 lines)
- `wake()` (3 lines)

**Estimated Size**: ~200 lines (including infrastructure)

**Interface**:
```python
class AgentLoopRunner:
    """Event-driven agent loop for async message processing."""

    def __init__(
        self,
        session_id: str,
        timeline: Timeline,
        llm: LLMProvider | None,
        get_pending_messages: Callable[[], list[MessageNode]],
        mark_processed: Callable[[str], None],
        tick_callback: Callable[[], Awaitable[list[SessionUpdate]]],
        prompt_with_llm_callback: Callable[[str], AsyncIterator[SessionUpdate]],
        prompt_direct_callback: Callable[[str], AsyncIterator[SessionUpdate]],
        get_projection_callback: Callable[[], Projection],
    ):
        self._session_id = session_id
        self._timeline = timeline
        self._llm = llm
        self._running = False
        self._wake_event = asyncio.Event()
        # Callbacks to Session
        self._get_pending_messages = get_pending_messages
        self._mark_processed = mark_processed
        self._tick = tick_callback
        self._prompt_with_llm = prompt_with_llm_callback
        self._prompt_direct = prompt_direct_callback
        self._get_projection = get_projection_callback

    async def run(self) -> AsyncIterator[SessionUpdate]:
        """Main agent loop - wait for wake, process messages, tick."""
        ...

    def stop(self) -> None:
        """Stop the loop gracefully."""
        ...

    def wake(self) -> None:
        """Wake the loop to process pending work."""
        ...

    async def _process_next_message(self) -> AsyncIterator[SessionUpdate]:
        """Process oldest pending message."""
        ...

    def _has_pending_work(self) -> bool:
        """Check for pending work."""
        ...
```

**Pros**:
- Separates control flow from business logic
- Makes agent loop testable in isolation
- Reasonable size (~200 lines)

**Cons**:
- Heavy callback coupling to Session
- Requires 5+ callback parameters
- Business logic still in Session (_prompt_with_llm is 400+ lines)
- Doesn't reduce Session complexity much

**Dependencies**:
- Needs access to: message queue, timeline, LLM, tick logic
- All delegated via callbacks → high coupling

### Option B: Extract PathResolver (Alternative)

**Purpose**: Isolate complex path resolution logic

**Methods to Extract**:
- `resolve_path()` (50 lines)
- `_expand_path_roots()` (30 lines)
- Helper functions

**Estimated Size**: ~120 lines

**Rationale**: Path resolution is self-contained, has clear inputs/outputs, and is independently testable

**Interface**:
```python
class PathResolver:
    """Resolves @prompts/, {home}, {cwd} and other path roots."""

    def __init__(self, cwd: str, config: Config | None = None):
        self._cwd = cwd
        self._config = config

    def resolve(self, path: str) -> tuple[str, str | None]:
        """Resolve path with roots, return (resolved_path, root_name)."""
        ...

    def expand_roots(self, path: str) -> str:
        """Expand {home}, {cwd} placeholders."""
        ...
```

### Option C: No Extraction (Recommended)

**Rationale**:
1. **Session is cohesive**: Despite 50+ methods, they work together to orchestrate a session
2. **Small extractions don't help**: SessionRenderer would be ~100 lines, minimal value
3. **Agent loop is tightly coupled**: Requires 5+ callbacks, doesn't reduce complexity
4. **Clear structure already**: Methods are well-organized into logical groups
5. **Lower priority**: Timeline refactoring (statement execution, DSL namespace) is more impactful

**Alternative Improvements**:
- Add docstring sections to group related methods
- Extract large methods like `_prompt_with_llm` (400+ lines) into smaller helpers
- Move path resolution to a utility module if reused elsewhere
- Consider extracting multi-task support when Phase 5 is fully implemented

## Recommendation

**Do NOT extract SessionRenderer or AgentLoopRunner at this time.**

**Reasons**:
1. Session's 1,897 lines are primarily due to LLM processing methods (`_prompt_with_llm`, `tick`, etc.), not the proposed extraction targets
2. The proposed extractions are too small (~100-200 lines each) to justify the added complexity
3. Both extractions have weak cohesion or heavy coupling
4. Current organization is already good - methods are well-named and logically grouped

**If refactoring is needed**, prioritize:
1. **Timeline refactoring** (1,285 lines, 50+ methods) - higher impact
2. **Break up large methods**: `_prompt_with_llm` (~400 lines) could be split into streaming handlers
3. **Extract PathResolver**: If path logic is reused elsewhere, move to a utility module
4. **Wait for Phase 5**: Multi-task support may justify a TaskCoordinator extraction later

## Impact Analysis

### If we proceed with Option A:

**SessionRenderer**:
- **Lines saved**: ~100 (minimal)
- **Complexity reduction**: Low - mostly delegation
- **Test value**: Low - simple wrappers
- **Risk**: Low
- **Effort**: 1-2 days

**AgentLoopRunner**:
- **Lines saved**: ~200 (moderate)
- **Complexity reduction**: Medium - but moves to callbacks
- **Test value**: Medium - event loop testable
- **Risk**: Medium - event handling is tricky
- **Effort**: 3-5 days (including callback setup and testing)

**Total**: 5-7 days effort for ~300 lines extracted, minimal complexity reduction

### If we defer:

**Cost**: None - current code works fine
**Benefit**: Team can focus on Timeline or feature work
**Maintenance**: Manageable - Session is well-organized

## Conclusion

Session class is large but cohesive. The proposed extractions provide minimal value for moderate effort. Recommend deferring this refactoring in favor of higher-impact work.

**Priority Ranking**:
1. Timeline refactoring (statement execution, DSL complexity)
2. Large method decomposition (_prompt_with_llm, tick)
3. PathResolver extraction (if reused)
4. Session extraction (low priority)
