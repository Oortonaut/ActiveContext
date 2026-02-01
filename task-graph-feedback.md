# Task-Graph Feedback

Feedback from `claude-deliver-1` session on 2026-01-29.

## 1. Workflow Discovery Is Unavailable Pre-Connection

**Problem:** The `connect` tool accepts a `workflow` parameter (e.g., `"swarm"` for `workflow-swarm.yaml`), but there is no way to discover available workflows *before* connecting. An agent asked to "connect to the hierarchical workflow" has no path to verify the workflow exists or list alternatives without first connecting (which requires choosing a workflow or defaulting to none).

**What I tried:**
- `connect(workflow="hierarchical")` — would be a blind guess
- Globbing `task-graph/*.yaml` — no workflow files found
- Full-text search for "workflow" — searches task content, not server config
- `list_skills` — returns skill metadata, not workflow metadata
- `get_current_config` (Serena) — wrong server entirely

**What's missing:**
- A `list_workflows` tool callable before or after `connect`
- Or: workflow names included in the `connect` response under a `available_workflows` field
- Or: a `get_server_info` tool that returns available workflows, config paths, and version info without requiring connection

**Impact:** An agent given the instruction "connect to workflow X" must either guess and hope, or connect without a workflow and then try to figure out what's available — at which point it's too late since the workflow is set at connection time.

**Suggested fix (pick one):**
1. Add `list_workflows` tool (no `worker_id` required) — returns names + descriptions of available `workflow-*.yaml` files
2. Include `available_workflows: ["swarm", "hierarchical", ...]` in the `connect` response so agents can disconnect and reconnect if they picked wrong
3. Allow `connect(workflow="...")` to be called again to switch workflows without full disconnect/reconnect

---

## 2. Task Titles Are Overloaded with Full Descriptions

**Observation:** Many tasks in this graph use multi-line titles that duplicate the description field. For example, task `lsp-003` has a title that is 5 lines long including bullet points. This makes `list_tasks` output very noisy — every task listing reads like a wall of text rather than a scannable list.

**Suggestion:** Enforce or encourage short titles (single line, <80 chars) and keep detail in the description. The `list_tasks` output could truncate titles to a max length with `...` suffix.

---

## 3. `list_tasks(ready=true)` Returns Category/Triage Tasks

**Observation:** Organizational tasks like `stream-a` ("Stream A: Core Features"), `priority-p2` ("Priority P2: Deferred"), and `activecontext-migration` show up as "ready" even though they're category containers, not actionable work items. An agent scanning for work has to mentally filter these out.

**Suggestion:** Either:
- Allow marking tasks as `type: "category"` or `type: "epic"` so `ready=true` can exclude them
- Or use a convention where container tasks are in a non-claimable status

---

## 4. ~~No Way to See Task Tree Structure from `list_tasks`~~ (Resolved)

**Update:** `scan(task="stream-a", below=-1, format="markdown")` works well for tree views. The output is usable — shows all 16 children with status, points, and IDs. The verbosity is mainly a side-effect of #2 (overloaded titles), not a structural problem.

**Remaining minor issue:** `list_tasks` is still flat-only, so an agent must already know a parent ID to scan from. A `list_tasks(parent="null")` to get root tasks would help bootstrap tree exploration. *(Note: `parent="null"` may already work — untested.)*

---

## 5. Workflow Config Not Reflected in Connect Response

**Observation:** After `connect(workflow="hierarchical")`, the response includes `"workflow": "hierarchical"` but nothing from the workflow file itself — no roles, no states, no prompts, no gates. The agent gets the same generic config shape regardless of workflow.

**What I expected:** The connect response to include workflow-specific info like:
- Available roles and which one I matched based on my tags
- State machine (valid transitions) so the agent knows what states exist
- Any entry prompts for initial state

**What actually happened:** The response has `"config"` with generic states/phases, `"tag_warnings": ["Unknown tag 'worker'. Known tags: []"]` (the workflow defines `worker` as a role tag, but `known_tags` is empty), and no indication of what the workflow provides.

**Suggestion:** The connect response should include a `workflow_config` section with at minimum:
- Role matched (or "no role matched")
- Role-specific constraints (max_claims, can_assign, etc.)
- A summary of workflow-specific states/phases/gates

---

## 6. `known_tags` Is Empty Despite Workflow Defining Tags

**Observation:** The hierarchical workflow defines `roles.lead.tags: [lead, coordinator]` and `roles.worker.tags: [worker]`, but after connecting with `tags: ["worker"]` I got `"tag_warnings": ["Unknown tag 'worker'. Known tags: []"]`. The workflow's role tags aren't being registered as known tags.

**Impact:** This warning is misleading — the tag IS meaningful in this workflow, it's just not populated in the server's known_tags list. Agents may doubt whether their tags are correct.

**Suggestion:** Workflow role tags should automatically be added to `known_tags` when a workflow is active.

---

## 7. Workflow Files Live in Server Source, Not Project Directory

**Observation:** Workflow YAML files are at `c:\projects\task-graph-mcp\config\`, not in the project's `task-graph/` directory. The project directory only has `tasks.db`, `logs/`, and `media/`. An agent operating within the project has no filesystem path to discover workflows — they'd need to know the server installation path.

**Impact:** Reinforces the need for a `list_workflows` tool (feedback #1). The workflow files are a server-level concept, not a project-level artifact, so filesystem discovery is the wrong approach.

---
