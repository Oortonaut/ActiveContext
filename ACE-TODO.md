Today's Goals: Develop ActiveContext with itself
=================================================

## P0 - Critical Path (Sequential)

### Agent Notification System
**Dependency: Required for effective agent coordination**

- [x] Add `NotificationLevel` enum to context nodes: `IGNORE`, `HOLD`, `WAKE`
  - `IGNORE`: Propagate changes upward, no effect on agent
  - `HOLD`: Collect notifications, deliver at next tick boundary
  - `WAKE`: Hold and signal waiting agent to resume its event loop
- [x] Implement notification message queue on nodes
- [x] Notifications propagate to parent nodes (agent typically subscribes at topmost node)
- [x] Add `notify()` DSL function and `on_notify` callback mechanism
- [x] Add Alerts group for displaying notifications in projection
- [x] Auto-subscribe root context as notification collection point

---

## P1 - Core Features (Parallelizable)

### Stream A: Node Enhancements

#### Originator Field Migration [In progress]
- [ ] Move `actor` field from MessageNode to base ContextNode class
- [ ] Rename field to `originator` (arbitrary string reference)
- [ ] Support node ID, filename, or arbitrary text as originator value
- [ ] Update MessageNode to use inherited `originator` instead of `actor`
- [ ] Update `display_label` and `effective_role` to use new field name
- [ ] Add originator to change notification payloads

#### Text Node Insertion
- [ ] Add `insert(pos, content)` method to TextNode
- [ ] Support insertion at line:column positions
- [ ] Validate positions against current content boundaries
- [ ] Emit appropriate notifications on content change

#### New Node Types
- [ ] **FileSystemNode**: Directory tree view with filtering, collapse/expand
- [ ] **ClockNode**: Timer/countdown with tick-driven updates (useful for timeouts)
- [ ] **FunctionDocNode**: Extract and display function signatures + docstrings

### Stream B: Testing & Quality

#### Notification System Testing
- [ ] Unit tests for NotificationLevel enum and Notification dataclass
- [ ] Test ContextNode._emit_notification() called when level != IGNORE
- [ ] Test ContextNode._format_notification_header() formatting
- [ ] Test TextNode header override with line change info
- [ ] Test ContextGraph.emit_notification() adds to queue
- [ ] Test deduplication via trace_id (same trace not added twice)
- [ ] Test ContextGraph.flush_notifications() clears state
- [ ] Test ContextGraph.has_wake_notification() flag
- [ ] Test Session Alerts group created and linked to root
- [ ] Test tick() processes notifications into Alerts group
- [ ] Test WAKE level triggers _wake_event.set()
- [ ] Test HOLD level queues until tick boundary
- [ ] Test notify() DSL function sets level correctly
- [ ] Test string level conversion ("wake" -> NotificationLevel.WAKE)

#### MCP Integration Testing
- [ ] Test server connection lifecycle (connect/reconnect/disconnect)
- [ ] Test tool invocation with various parameter types
- [ ] Test error handling and timeout scenarios
- [ ] Test permission boundary enforcement

#### Node Manipulation Testing
- [ ] Test DAG link/unlink operations
- [ ] Test checkpoint/restore/branch cycles
- [ ] Test state transitions (HIDDEN->COLLAPSED->SUMMARY->DETAILS->ALL)
- [ ] Test group summarization triggers

#### Test Coverage Update
- [ ] Run coverage analysis on current test suite
- [ ] Identify untested code paths
- [ ] Add tests for critical gaps

### Stream C: Documentation & Prompts

#### ACP Protocol Documentation
- [ ] Diagram Agent Client Protocol from primary sources (JetBrains spec, acp-sdk)
- [ ] Document message flow: initialize -> prompt -> response -> cancel
- [ ] Document session lifecycle and state transitions
- [ ] Capture known quirks and implementation notes

#### Prompt Cleanup
- [ ] Review `dsl_reference.md` for accuracy against current implementation
- [ ] Update examples to use current API patterns
- [ ] Remove deprecated function references
- [ ] Add missing DSL functions to reference

#### Markdown Test Fixture
- [ ] Add test cases for collapsed TextNodes
- [ ] Add test cases for summarized TextNodes
- [ ] Verify render_tags output for all node states

---

## P2 - Important but Deferrable

### Security Review
- [ ] Comprehensive security audit of permission system (file, shell, import, web)
- [ ] Review path traversal prevention in file operations
- [ ] Audit shell command injection vectors
- [ ] Review MCP server permission boundaries
- [ ] Validate sandbox isolation guarantees
- [ ] Check for sensitive data exposure in logs/projections
- [ ] Review session storage for credential leakage

### Summarization Pipeline
- [ ] Implement LLM-based summarization for TextNodes
- [ ] Add summary caching to avoid redundant LLM calls
- [ ] Support incremental summarization for large files
- [ ] Add `summarize()` DSL function

### Filesystem API Testing
- [ ] Test file permission boundaries
- [ ] Test path traversal prevention
- [ ] Test read/write/watch operations

### Multi-user Dashboard
- [ ] Verify WebSocket multiplexing works
- [ ] Test concurrent session updates
- [ ] Check for race conditions in shared state
- [ ] Test dashboard behavior with multiple agents running simultaneously

### Subagent Testing
- [ ] Test spawn/pause/terminate lifecycle
- [ ] Test message passing between parent and child
- [ ] Test resource cleanup on termination

---

## P3 - Tech Debt & Cleanup

### Code Quality (Parallelizable)

#### Code Simplification
- [ ] Identify overly complex functions (cyclomatic complexity > 10)
- [ ] Extract helper functions where appropriate
- [ ] Simplify nested conditionals

#### Dead Code Removal
- [ ] Run static analysis to find unreachable code
- [ ] Remove unused imports and variables
- [ ] Archive deprecated modules

### Infrastructure

#### Terminal Executor Capabilities
- [ ] Add capability detection for terminal features (color, unicode, size)
- [ ] Graceful degradation for limited terminals
- [ ] PTY support for interactive commands

#### Hooking System
- [ ] Design hook points (pre/post command, state change, tick boundary)
- [ ] Implement hook registration API
- [ ] Add built-in hooks for logging/metrics

#### Rider Integration Hacks
- [ ] Document known Rider quirks and workarounds
- [ ] Add Rider-specific ACP message handling if needed
- [ ] Test with latest Rider builds

---

## Parallelization Matrix

| Stream | Tasks | Can Run With |
|--------|-------|--------------|
| **A** | Text insertion, New nodes | B, C |
| **B** | MCP tests, Node tests, Coverage, Notification tests | A, C |
| **C** | Prompt cleanup, Markdown fixtures | A, B |
| **Debt** | Code cleanup, Dead code removal | Any (low risk) |

### Suggested Parallel Execution

**Agent 1**: Stream A (Node Enhancements)
**Agent 2**: Stream B (Testing)
**Agent 3**: Stream C (Documentation)

P0 (Notifications) is now complete. Notification testing should be prioritized in Stream B.
