# Agent Loop Design: Executable Context as a Python Statement Timeline

Version: 0.2
Scope: Agent loop architecture only. Persistence/serialization is pluggable and out of scope (JSON/YAML/SQLite/etc.).
Primary goal: A Claude Code–style CLI agent where the LLM has first-class control over a structured, reversible working context. The context presented to the LLM is a timeline of executed Python statements operating over live “context objects” (views, groups, etc.). The agent can re-execute from any point.

---

## 0. Executive summary

The agent loop is built around:

1. **A real Python execution environment** with a controlled namespace.
2. **Stateful wrapper objects** created in that namespace (e.g., `view("~/Foo.cs", ...)`) whose methods mutate local Python state.
3. **A statement log**: the canonical timeline is “what Python statements were executed.”
4. **Groups are summaries**: grouping produces a summary node; a group can contain 1 node; group summaries update automatically.
5. **Running/paused nodes** with a tick-driven update model:

   * **Sync** tick: per turn
   * **Periodic(seconds)**: reserved
   * **Async**: reserved
     Async preparation can be performed in the background, but **mutations are applied only at tick boundaries**, and **async completions are applied before periodic ticks**.

The projection to the LLM is not a transcript. It is a compact representation of:

* handles (variables) in the Python environment,
* group summaries (primary surface),
* and deltas since first version seen (especially for running nodes). This nearby history provides context for recent changes to the viewed object.

---

## 1. Concepts and invariants

### 1.1 Timeline semantics

* The “true” interaction history is an ordered list of **Statement** records.
* Each statement has one or more **Execution** results (re-runs create new executions).
* Re-execution from statement N resets the Python environment to a checkpoint (or empty) and replays statements from N onward.
* Determinism is “best effort”; divergence is expected and handled via diffs and provenance.

### 1.2 Context objects

The LLM works with actual Python objects using its own repl, not remote handles:
         
**First Turn**  
```
--- agent message ---
<script type="text/x-python">
foo_cs_file = open("~/Foo.cs")
foo_cs = view(foo_cs_file, pos=0, tokens=2000)
</script>
--- framework message ---
# functionally, view(...) outputs this message to the context when first executed and then reevaluated during context generation.
# // This is foo.cs. It is a long, long c# file with a lot of code. 
# // This comment itself goes on for over 15000 tokens.
# // ... token 1999
---many turns of conversation and data ---
--- agent message ---
<script type="text/x-python">
foo_cs.SetLod(1).SetTokens(All) # When view is reevaluated next, it will render only the summary
</script>
```
**Next Turn**  
```
--- agent message ---
<script type="text/x-python">
foo_cs_file = open("~/Foo.cs")
# foo_cs = view(foo_cs_file, pos=0, tokens=2000) # msg < 12
foo_cs = view(foo_cs_file, pos=0, tokens=All, lod=1)
</script>
--- framework message ---
# functionally, view(...) outputs this message to the context when first executed and then reevaluated during context generation.
# // This is foo.cs. It is a long, long c# file with a lot of code. 
# // This comment itself goes on for over 15000 tokens.
# // ... token 1999
---many turns of conversation and data ---
--- agent message ---
<script type="text/x-python">
foo_cs.SetLod(1).SetTokens(All) # When view is reevaluated next, it will render only the summary
</script> 
```

These methods mutate the foo_cs object state; this mutation may touch related objects. But the primary effect is to adjust the mapping of the view onto the referenced object for the visible updates. 

### 1.3 Groups are summaries

* A **Group** is itself a summary node over its members.
* Creating a group is equivalent to creating a summarized façade over its members.
* Group summarization is automatic; the group always represents a current summary given its policy (LOD/budget).
* Grouping one node is valid and useful (it creates a summarized façade with stable identity and policy knobs).

### 1.4 Node states: paused vs running

* **Paused**: updates only via explicit method calls. Shows last known state.
* **Running**: eligible for automatic updates via tick scheduling.

### 1.5 Tick-driven update boundary (critical invariant)

All automatic node mutations occur at deterministic boundaries:

* during statement execution, or
* during the tick phase(s).

No background thread/task may directly mutate node state.

Async work may run in the background only as “prepare” steps; applying prepared results is tick-only.

---

## 2. High-level architecture

### 2.1 Modules

**Core runtime**

* `AgentLoop`: orchestrates turns, projection, execution, ticking.
* `StatementLog`: append-only timeline of statements + executions.
* `PythonExec`: controlled Python environment with tracked namespace and output capture.
* `ProjectionEngine`: builds minimal LLM prompt pack from current graph of objects/groups/summaries/deltas.
* `Scheduler`: tick scheduling and async payload application ordering.

**Context objects (Python wrappers)**

* `ViewHandle`: file/URI views with cursor, token budget, LOD, running mode.
* `GroupHandle`: summary façade over member nodes (including 1 node).
* (Optional) `ToolHandle`, `NoteHandle`, etc.

**Eventing**

* `EventBus` (in-process): statement events, node change events, tick events.
* `DeltaFormatter`: produces concise “change feed” items for projection.

**Transports (optional, out of scope but supported)**

* CLI loop: user input → AgentLoop turn.
* ACP adapter: requests map to turns + file events.
* MCP tool exposure: optional, not required for core.

Persistence adapters are not in this file. The system emits structured state that any backend can serialize.

---

## 3. Data model (in-memory canonical)

### 3.1 Statement and Execution

**Statement**

* `statement_id: str`
* `index: int` (monotonic)
* `timestamp`
* `source: str` (exact Python text)
* `kind: enum {python, meta}` (meta for internal control statements if needed)
* `tags: dict` (optional)
* `fingerprint: hash(source + environment_meta)` (optional)

**Execution**

* `execution_id: str`
* `statement_id`
* `started_at`, `ended_at`
* `status: enum {ok, error, timeout, cancelled}`
* `stdout: str` (bounded)
* `stderr: str` (bounded)
* `exception: {type, message, trace_summary}?`
* `events: [Event]` (method calls, node mutations, created vars)
* `state_diff: NamespaceDiff` (vars added/changed/deleted)
* `artifacts: [Artifact]` (optional; file outputs, plots, etc.)
* `cost: {time_ms, tokens?, tool_cost?}`

### 3.2 Namespace diff

Compute after each statement (and optionally after tick phases).

`NamespaceDiff`

* `added: {name: TypeInfo}`
* `changed: {name: ChangeInfo}` (only for tracked objects or primitive changes)
* `deleted: [name]`

TypeInfo includes:

* python type name
* short repr (capped)
* size hints (best-effort)

### 3.3 Events

Emit structured events for every meaningful action.

Key event types:

* `StatementExecuted(statement_id, execution_id, status)`
* `MethodCalled(obj_id, obj_type, method, args, kwargs, before_digest, after_digest)`
* `NodeChanged(obj_id, delta)`
* `TickApplied(obj_id, tick_kind, delta)`
* `AsyncPayloadPrepared(obj_id, payload_meta)`
* `AsyncPayloadApplied(obj_id, payload_meta, delta)`
* `GroupRecomputed(group_id, delta)`

---

## 4. Python execution environment

### 4.1 Controlled namespace

Use a dedicated `globals` and `locals` mapping. The locals mapping must support:

* enumerating variables
* tracking assignments/deletions
* optionally disallowing dangerous builtins/imports later (TBD)

Minimal requirement now:

* maintain a list of “context objects” (instances of known wrapper types)
* compute diffs of variables between steps

### 4.2 Output capture

Capture stdout/stderr per statement execution. Enforce:

* max bytes per stream per statement
* truncated marker if exceeded

### 4.3 Injected functions/objects

Expose these names by default:

* `view(path, pos="line:col" | "line" | opaque, tokens=int, lod=int=0, mode="paused|running"=paused, freq="Sync|Periodic|Async"=paused)`
* `group(*members, tokens=int?, lod=int?, mode?, freq?)`
* `tick()` (internal; should not be used by model directly unless you want)
* Optional helpers:

  * `ls()` list handles and brief digests
  * `show(obj, lod=?, tokens=?)` force render for a handle
  * `pin(obj, reason=...)` mark for projection priority

The LLM mainly uses `view`, `group`, and object methods.

---

## 5. Wrapper objects (stateful, mutable)

### 5.1 Common interface (NodeBase)

All node objects implement:

* Identity:

  * `obj_id: str` stable within the agent session
  * `obj_type: str` (`view`, `group`, etc.)

* State:

  * `mode: paused|running`
  * `freq: Sync | Periodic(seconds) | Async`
  * `min_turn_interval: int = 1` (throttle for Sync)
  * `last_tick_turn: int`
  * `dirty: bool`

* Policy:

  * `tokens: int` desired budget for projection
  * `lod: int` representation level

* Methods:

  * `GetDigest() -> dict` small structured digest for projection
  * `Render(lod:int|None=None, tokens:int|None=None) -> RenderedView`
  * `Pause()`
  * `Run(freq=...)`
  * `Tick(now, turn_id, tick_kind) -> TickResult`
  * `OnMemberChanged(...)` for groups (if needed)

The base class also handles event emission around mutations.

### 5.2 ViewHandle (file/URI view)

Represents a controllable view onto an external document.

State:

* `path: str`
* `pos: str` (e.g., `"0:0"`)
* `window_spec: {range_lines, range_tokens, anchors}` (simple at first)
* `lod: int`
* `tokens: int`
* `rendered: RenderedView` cached
* `source_fingerprint: hash?` (optional: last seen file hash)
* `dirty_flags: {pos_changed, policy_changed, source_changed}`

Methods:

* `SetPos(pos)`
* `Scroll(delta_lines|delta_tokens)`
* `SetTokens(n)`
* `SetLod(k)`
* `Refresh()` (explicit fetch/recompute; sets rendered)
* `Run(freq="Sync"|("Periodic", seconds)|"Async")`
* `Pause()`

LOD ladder (recommended):

* `lod=0`: raw excerpt around `pos`
* `lod=1`: structured excerpt (headers/symbols + nearby lines)
* `lod=2`: semantic summary of region (bullets: responsibilities/invariants)
* `lod=3`: diff-only view vs last rendered revision

Tick behavior (Sync):

* If `dirty` or `source_changed`, refresh according to policy:

  * default: produce `lod=3` diff summary unless forced
  * full refresh only if requested or if needed to satisfy pinned constraints
* Emit delta event summarizing changes and update `rendered`.

### 5.3 GroupHandle (group == summary)

Group contains members and maintains an automatically updated summary.

State:

* `members: list[NodeBase]`
* `summary_rendered: RenderedView`
* `summary_policy: {tokens, lod, salience_rules}`
* `dirty: bool` set when any member changes

Methods:

* `Add(member)` / `Remove(member)` (optional; could be static groups only)
* `SetTokens(n)` / `SetLod(k)`
* `Refresh()` recompute summary (explicit)
* Tick recomputes summary if dirty or scheduled

Semantics:

* Group is the default projection surface; members are rarely projected directly unless requested or pinned.

---

## 6. Tick scheduling and async handoff

### 6.1 Tick kinds and ordering per turn

Within each agent turn, the runtime performs:

1. **Apply-ready async payloads** (prepared by background work; tick-only mutation)
2. **Sync ticks** for running nodes with `freq=Sync` and throttle satisfied
3. **Periodic ticks** for nodes due (TBD implementation)
4. **Group recompute ticks** for dirty groups (may be included as part of steps above)
5. **Projection** for the next LLM call

This enforces:

* finished async passes are applied before periodic ticks
* groups see the newest member state before summarizing

### 6.2 Async: two-phase update (TBD but spec’d now)

Async does not mutate nodes directly. It produces `UpdatePayload`s.

**UpdatePayload**

* `payload_id`
* `obj_id`
* `prepared_at`
* `basis: {obj_digest_hash, policy_hash, deps_hash}` to validate staleness
* `kind: enum {diff, render, parse, index, other}`
* `data: any` (bounded)
* `cost: {time_ms, bytes}`
* `payload_hash`

Async completion enqueues `(obj_id, payload_id)` into the `ready_queue`.

Tick apply step:

* validate `basis` matches current object state; otherwise drop payload
* apply payload to object state (mutate locally)
* emit `AsyncPayloadApplied` and `NodeChanged`

Coalescing rule:

* keep only the latest compatible payload per `(obj_id, kind)` before applying.

Non-starvation:

* cap number of async payload applications per turn.

### 6.3 Periodic (reserved)

Periodic tick eligibility:

* due if `now - last_tick_time >= interval_seconds`

Actual implementation is TBD.

---

## 7. Agent loop control flow

### 7.1 Turn lifecycle

A “turn” is one iteration of the agent loop prompted by user input or external event.

1. **Ingest input**

   * user message, file change notifications, external tool results (transport dependent)
2. **Decide**

   * build prompt projection from current state
3. **Model call**

   * get next Python statement batch (the agent “program step”) and/or final user response
4. **Execute statements**

   * run Python code in controlled namespace
   * log statement and execution results
5. **Tick**

   * apply async ready payloads
   * sync tick
   * periodic tick (TBD)
   * group recompute tick
6. **Project + respond**

   * present final response to user, plus keep internal projection state for next call

### 7.2 Output channels

The model can emit:

* a batch of Python statements to execute
* a user-facing response
* or both (e.g., “do some work, then answer”)

Implementation detail: treat “python batch” as one Statement (cell) or multiple statements; recommended: one cell per model turn for coherence and replay.

---

## 8. Projection to the LLM

### 8.1 Projection goals

* Provide enough structured context for the LLM to plan and act.
* Prefer group summaries over raw content.
* Include deltas for running nodes (change feed).
* Preserve stable references: handle names and statement indices.

### 8.2 Projection contents

A projected prompt pack should include:

1. **Handle table** (names → digests)

   * for each known wrapper object (ViewHandle, GroupHandle)
   * include: type, mode, freq, lod, tokens, changed flag

2. **Primary group summaries**

   * include summaries for selected groups (pinned, recent, relevant)
   * each summary references member handles by name/id

3. **Delta feed**

   * since last model call:

     * node changed events
     * tick deltas
     * async applied summaries
     * errors/timeouts

4. **On-demand expansions**

   * only when requested or required by policy
   * expansions are usually new groups or increased LOD/tokens on existing nodes

### 8.3 Token budgeting

Each node has `tokens` and `lod` preferences; the projection engine chooses:

* which groups to include
* at what LOD
* how to compress deltas

A simple policy is sufficient initially:

* include pinned group summaries
* include recent delta feed
* include digests for all known handles (very compact)
* include full view content only on explicit request

---

## 9. DSL contract for the LLM

The LLM writes Python that manipulates live context objects.

### 9.1 Core constructors

```python
foo_cs = view("~/Foo.cs", pos="0:0", tokens=2000)           # paused by default
foo_cs.Run(freq="Sync")                                     # running, per turn
foo_sum = group(foo_cs, tokens=500, lod=2).Run(freq="Sync") # group summary node
```

### 9.2 View controls

```python
foo_cs.SetLod(1)
foo_cs.SetTokens(10000)
foo_cs.SetPos("120:0")
foo_cs.Scroll(50)
foo_cs.Refresh()             # explicit refresh
foo_cs.Pause()
```

### 9.3 “Group == summary” usage

To “expand,” you typically create a new group with higher budget/LOD:

```python
foo_detail = group(foo_cs, tokens=4000, lod=0)  # a more detailed façade
```

To “compress,” lower budget/LOD:

```python
foo_sum.SetTokens(300)
foo_sum.SetLod(3)  # diff-only façade if supported
```

### 9.4 Running updates

* Sync nodes update once per turn (throttled by `min_turn_interval`).
* Async/Periodic are reserved but the API accepts them.

---

## 10. Safety, determinism, and replay (constraints)

### 10.1 Replay boundary

Replay is defined at statement granularity:

* replay from statement index k
* restore checkpoint ≤ k
* execute statements k..end

### 10.2 Side-effect control (TBD)

Initially, allow normal Python but strongly recommend:

* wrapper-mediated file access
* wrapper-mediated tool access
  Later, add:
* import restrictions
* restricted builtins
* sandboxing policy

### 10.3 Divergence handling

Re-execution may diverge due to:

* file changes
* nondeterminism
* tool outputs changing

The system’s contract is:

* detect and summarize divergences (diffs in outputs/state/artifacts)
* surface them in delta feed and group summaries

---

## 11. Implementation plan (milestones)

### M0: Skeleton loop

* Controlled Python exec, statement logging, stdout/stderr capture
* `view` returns a stateful ViewHandle with SetTokens/SetLod/SetPos/Refresh
* Handle digest listing

### M1: Groups as summaries

* `group` returns GroupHandle summary façade
* Group auto-refresh on explicit calls
* Projection prefers groups

### M2: Sync tick

* Node `Run(freq="Sync")` and `Pause()`
* Tick phase per turn
* Node dirty flags and throttling
* Delta feed in projection

### M3: Async framework stubs

* UpdatePayload structure
* ready_queue and apply-before-periodic ordering
* No actual background workers yet (TBD)

### M4: Re-execution

* statement replay from index
* checkpoint interface (pluggable; can start with “replay from start”)
* diff execution results (stdout/state digests)

---

## 12. Acceptance criteria

1. LLM can create a view object, adjust LOD/tokens, and see updates reflected in subsequent statements (local Python state mutation).
2. Groups summarize members automatically and are the main projected context.
3. Running nodes update deterministically at tick boundaries (Sync).
4. The projection contains:

   * handle digests
   * group summaries
   * deltas since last turn
5. Re-execution from statement N reproduces the environment up to best effort and surfaces diffs.

---

## 13. Notes for Claude Code handoff

When implementing, focus on these decisions first:

* Statement granularity: one “cell” per model turn is simplest.
* Controlled namespace: track variables and wrapper objects explicitly.
* Event emission: wrap all wrapper method calls to capture before/after digests.
* Tick ordering: async apply (stub) → sync tick → periodic tick (stub) → groups → projection.

Avoid building persistence early; emit structured state so any serializer can be added later.

If you want, I can also produce a minimal set of Python class skeletons and function signatures (no persistence) aligned exactly with this design, so Claude Code can start coding immediately.
