ActiveContext Development Roadmap
=================================

## Bugs

_(none)_

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
  - [ ] Rendered output should preserve original markdown appearance

#### New Node Types
- [ ] **FileSystemNode**: Directory tree view with filtering, collapse/expand
- [ ] **ClockNode**: Timer/countdown with tick-driven updates
- [ ] **FunctionDocNode**: Extract and display function signatures + docstrings

#### Node Help System
- [ ] Add `.help()` method to base ContextNode class
- [ ] Creates or unhides a help node/panel attached to the calling node
- [ ] SUMMARY: brief description and common methods
- [ ] DETAILS: full method list with signatures and examples

#### MCP Node Enhancements
- [ ] Show individual tool details as child nodes of MCPServerNode
- [ ] Each tool node displays name, description, and parameter schema

### Stream B: Testing & Quality

#### MCP Integration Testing
- [ ] Test server connection lifecycle (connect/reconnect/disconnect)
- [ ] Test tool invocation with various parameter types
- [ ] Test error handling and timeout scenarios
- [ ] Test permission boundary enforcement

#### Node Manipulation Testing
- [ ] Test DAG link/unlink operations
- [ ] Test checkpoint/restore/branch cycles
- [ ] Test state transitions (HIDDEN→COLLAPSED→SUMMARY→DETAILS→ALL)
- [ ] Test group summarization triggers

#### Test Coverage Improvement
**Current**: 699 tests, 61% coverage (3,156 statements missing) | **Target**: 80%

| Priority | File | Coverage | Notes |
|----------|------|----------|-------|
| P0 | `dashboard/*` | 0% (447) | Decide: test or remove |
| P0 | `session/timeline.py` | 49% (540) | Core execution engine |
| P0 | `transport/acp/agent.py` | 21% (512) | ACP protocol handler |
| P1 | `mcp/client.py` | 27% (131) | MCP client manager |
| P1 | `context/nodes.py` | 65% (444) | Node types |
| P1 | `watching/watcher.py` | 49% (70) | File watching |
| Quick | `clean.py` | 0% (31) | Simple deletion logic |
| Quick | `mcp/transport.py` | 16% (27) | Transport abstraction |
| Quick | `agents/registry.py` | 33% (28) | Simple registry |

### Stream C: Documentation & Prompts

#### ACP Protocol Documentation
- [ ] Diagram Agent Client Protocol from primary sources (JetBrains spec, acp-sdk)
- [ ] Document message flow: initialize → prompt → response → cancel
- [ ] Document session lifecycle and state transitions
- [ ] Capture known quirks and implementation notes

#### Prompt Cleanup
- [ ] Review `dsl_reference.md` for accuracy against current implementation
- [ ] Update examples to use current API patterns
- [ ] Remove deprecated function references
- [ ] Add missing DSL functions to reference

#### Markdown Test Fixtures
- [ ] Add test cases for collapsed/summarized TextNodes
- [ ] Verify render_tags output for all node states
- [ ] Add test cases for MarkdownNode list item child nodes
- [ ] Demonstrate all TextNode visibility states side-by-side

### Stream D: Session/REPL

#### Node Lookup
- [ ] Add node lookup by name (e.g., `get("my_view")` or `nodes["my_view"]`)
- [ ] Support fuzzy/partial name matching

#### Traversal Control
- [ ] Add `remove(node)` DSL function to exclude node from traversal
- [ ] Removed nodes skip rendering but retain state for potential restoration

#### Cross-Platform Path Roots
- [ ] Register path roots automatically (`~`, `$HOME`, `%USERPROFILE%`, `{home}`, etc.)
- [ ] Normalize paths so agent doesn't need to know which OS it's on

---

## P2 - Important but Deferrable

### Security Review
- [ ] Comprehensive security audit of permission system (file, shell, import, web)
- [ ] Review path traversal prevention in file operations
- [ ] Audit shell command injection vectors
- [ ] Review MCP server permission boundaries
- [ ] Validate sandbox isolation guarantees
- [ ] Check for sensitive data exposure in logs/projections

### Summarization Pipeline
- [ ] Implement LLM-based summarization for TextNodes
- [ ] Add summary caching to avoid redundant LLM calls
- [ ] Support incremental summarization for large files
- [ ] Add `summarize()` DSL function

### Dashboard Testing
- [ ] Verify WebSocket multiplexing works
- [ ] Test concurrent session updates
- [ ] Check for race conditions in shared state
- [ ] Test dashboard behavior with multiple agents running

### Agent Extensions
- [ ] Expose async agent loop extension API (message queueing, out-of-band status)
- [ ] Add `promptCapabilities`: text, image, audio, resource, resource_link
- [ ] Add mime type support in dashboard and agent consumption
- [ ] Implement ACP plan mode protocol support

### Subagent Testing
- [ ] Test spawn/pause/terminate lifecycle
- [ ] Test message passing between parent and child
- [ ] Test resource cleanup on termination

---

## P3 - Tech Debt & Cleanup

### Code Quality
- [ ] Identify overly complex functions (cyclomatic complexity > 10)
- [ ] Extract helper functions where appropriate
- [ ] Run static analysis to find unreachable code
- [ ] Remove unused imports and variables

### Infrastructure
- [ ] Terminal capability detection (color, unicode, size) with graceful degradation
- [ ] PTY support for interactive commands
- [ ] Design hook points (pre/post command, state change, tick boundary)
- [ ] Document known Rider quirks and workarounds

---

## Epics

- [ ] Claude skills support

---

## Parallelization Matrix

| Stream | Focus | Can Run With |
|--------|-------|--------------|
| **A** | Node enhancements, MCP nodes | B, C, D |
| **B** | Testing, Coverage | A, C, D |
| **C** | Documentation, Prompts | A, B, D |
| **D** | Session/REPL features | A, B, C |
| **Debt** | Code cleanup | Any |

---

## Closed

- [x] 2026-01-20: Agent Notification System (P0)
  - NotificationLevel enum, message queue, propagation, Alerts group
- [x] 2026-01-20: Notification System Testing (Stream B)
  - Full test coverage for notification infrastructure
