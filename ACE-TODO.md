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

#### Text Node Manipulation
- [ ] Add `replace_lines(line_no, num_removed, lines)` method to TextNode
  - 1-based indexing (line 0 also accepted as first line)
  - EOF marker: -1 (replacing at EOF appends)
  - `num_removed < 0` removes from line_no to end
  - `num_removed = 0` inserts without removing
- [ ] Register file change callback (separate from context notifications)
- [ ] Track which nodes reference each file path
- [ ] On file change: filter nodes whose displayed range overlaps the change
- [ ] Emit context notifications on change-relevant nodes only

#### MarkdownNode Enhancements
- [ ] Parse markdown list items into their own child nodes
  - [ ] Support ordered and unordered lists
  - [ ] Nested list items become nested child nodes
  - [ ] Individual list items can have independent NodeState
  - [ ] Rendered output should preserve original markdown appearance (bullets, numbering, indentation)

#### New Node Types
- [ ] **FileSystemNode**: Directory tree view with filtering, collapse/expand
- [ ] **ClockNode**: Timer/countdown with tick-driven updates (useful for timeouts)
- [ ] **FunctionDocNode**: Extract and display function signatures + docstrings

#### Node Help System
- [ ] Add `.help()` method to base ContextNode class
- [ ] Creates or unhides a help node/panel attached to the calling node
- [ ] SUMMARY state: brief description and common methods
- [ ] DETAILS state: full method list with signatures and examples
- [ ] Generate help content dynamically from class metadata/docstrings

### Stream B: Testing & Quality

#### Notification System Testing
- [x] Unit tests for NotificationLevel enum and Notification dataclass
- [x] Test ContextNode._emit_notification() called when level != IGNORE
- [x] Test ContextNode._format_notification_header() formatting
- [x] Test TextNode header override with line change info
- [x] Test ContextGraph.emit_notification() adds to queue
- [x] Test deduplication via trace_id (same trace not added twice)
- [x] Test ContextGraph.flush_notifications() clears state
- [x] Test ContextGraph.has_wake_notification() flag
- [x] Test Session Alerts group created and linked to root
- [x] Test tick() processes notifications into Alerts group
- [x] Test WAKE level triggers _wake_event.set()
- [x] Test HOLD level queues until tick boundary
- [x] Test notify() DSL function sets level correctly
- [x] Test string level conversion ("wake" -> NotificationLevel.WAKE)

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

#### Test Coverage Improvement (Audit: 2026-01-20)
**Current**: 699 tests, 61% coverage (3,156 statements missing)

##### P0 - Critical Gaps (Zero Coverage)
- [ ] `dashboard/*` (447 stmts) - Decide: test or remove
- [ ] `clean.py` (31 stmts) - Simple file cleanup utility

##### P0 - Core Engine Coverage
- [ ] `session/timeline.py` (49%, 540 missing) - Core execution engine
  - [ ] Statement replay/rollback (`replay_from()`)
  - [ ] Error handling paths
  - [ ] Shell execution flows
  - [ ] Lock management
- [ ] `transport/acp/agent.py` (21%, 512 missing) - ACP protocol handler
  - [ ] Session lifecycle (create/prompt/cancel)
  - [ ] JSON-RPC message handling
  - [ ] Error recovery paths

##### P1 - Module Coverage
- [ ] `mcp/client.py` (27%, 131 missing) - MCP client manager
  - [ ] Server connection lifecycle
  - [ ] Tool invocation
  - [ ] Error handling
- [ ] `mcp/transport.py` (16%, 27 missing) - Transport abstraction
- [ ] `context/view.py` (31%, 97 missing) - File view node
- [ ] `agents/handle.py` (30%, 49 missing) - Agent interaction
- [ ] `agents/manager.py` (31%, 55 missing) - Agent lifecycle
- [ ] `agents/registry.py` (33%, 28 missing) - Agent type registry
- [ ] `watching/watcher.py` (49%, 70 missing) - File watching
- [ ] `context/nodes.py` (65%, 444 missing) - Node types
  - [ ] ShellNode lifecycle
  - [ ] MCPServerNode operations
  - [ ] WorkNode coordination
  - [ ] AgentNode management

##### Quick Wins (Easy to Add)
- [ ] `clean.py` - 31 stmts, simple deletion logic
- [ ] `mcp/transport.py` - 27 stmts, transport abstraction
- [ ] `agents/registry.py` - 28 stmts, simple registry

**Target**: 80% coverage

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
- [ ] Add test cases for MarkdownNode list item child nodes
  - [ ] Verify ordered/unordered lists render with original appearance
  - [ ] Test nested list indentation preservation
  - [ ] Test individual list item state changes
- [ ] Demonstrate rendering of all TextNode visibility states
  - [ ] Use nested lists to show high-density state combinations
  - [ ] HIDDEN, COLLAPSED, SUMMARY, DETAILS, ALL side-by-side
  - [ ] Show same content at each state for comparison

### Stream D: Session/REPL

#### Node Lookup
- [ ] Add node lookup by name (e.g., `get("my_view")` or `nodes["my_view"]`)
- [ ] Support fuzzy/partial name matching

#### Traversal Control
- [ ] Add `remove(node)` DSL function to exclude node from traversal
- [ ] Removed nodes skip rendering but retain state for potential restoration

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
| **A** | Text insertion, New nodes, Help system | B, C, D |
| **B** | MCP tests, Node tests, Coverage, Notification tests | A, C, D |
| **C** | Prompt cleanup, Markdown fixtures, ACP docs | A, B, D |
| **D** | Node lookup, Traversal control | A, B, C |
| **Debt** | Code cleanup, Dead code removal | Any (low risk) |

### Suggested Parallel Execution

**Agent 1**: Stream A (Node Enhancements)
**Agent 2**: Stream B (Testing)
**Agent 3**: Stream C (Documentation)
**Agent 4**: Stream D (Session/REPL)

P0 (Notifications) is now complete. Notification testing should be prioritized in Stream B.
