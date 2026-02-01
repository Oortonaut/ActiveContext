# MCP Apps Design

**Version:** 1.0
**Status:** Design Specification
**Author:** ActiveContext Team
**Date:** 2026-01-31

---

## Executive Summary

**MCP Apps** is an LLM-driven UI framework where agents create interactive user interfaces through MCP tools. Instead of static dashboards or fixed forms, the LLM dynamically constructs UIs based on task requirements, user context, and data availability. The agent holds canonical state and the client renders from state snapshots, enabling rich, interactive experiences without hardcoded UI logic.

**Core Principles:**
- **Server-authoritative state**: LLM owns the canonical UI state
- **Declarative UI**: LLM declares component trees, client renders them
- **Event-driven interaction**: User actions flow back to LLM as structured events
- **Context integration**: Apps are first-class context nodes in ActiveContext DAG
- **Session persistence**: Apps survive session reloads and agent restarts

---

## 1. Component Taxonomy

### 1.1 Primitive Components

Components are the building blocks of MCP Apps. Each component has:
- **Type identifier** (e.g., `text`, `button`, `input`)
- **Props** (configuration object)
- **Children** (array of child components, for containers)
- **State** (component-specific reactive state)

#### Text & Content

| Component | Props | Description |
|-----------|-------|-------------|
| `text` | `content: str`, `style?: TextStyle` | Plain text with optional styling |
| `heading` | `level: 1-6`, `content: str` | Heading (h1-h6) |
| `markdown` | `content: str`, `syntax_highlight?: bool` | Rendered markdown with optional code highlighting |
| `code` | `content: str`, `language?: str`, `line_numbers?: bool` | Code block with syntax highlighting |
| `image` | `src: str`, `alt?: str`, `width?: int`, `height?: int` | Image from URL or data URI |

#### Input & Forms

| Component | Props | Description |
|-----------|-------|-------------|
| `button` | `label: str`, `variant?: 'primary'\|'secondary'\|'danger'`, `disabled?: bool`, `on_click: str` | Clickable button with event handler ID |
| `input` | `name: str`, `type?: 'text'\|'number'\|'email'\|'password'`, `value?: str`, `placeholder?: str`, `on_change: str` | Text input field |
| `textarea` | `name: str`, `value?: str`, `rows?: int`, `placeholder?: str`, `on_change: str` | Multi-line text input |
| `select` | `name: str`, `options: list[{label: str, value: str}]`, `value?: str`, `on_change: str` | Dropdown selection |
| `checkbox` | `name: str`, `label: str`, `checked?: bool`, `on_change: str` | Checkbox toggle |
| `radio_group` | `name: str`, `options: list[{label: str, value: str}]`, `value?: str`, `on_change: str` | Radio button group |

#### Data Display

| Component | Props | Description |
|-----------|-------|-------------|
| `table` | `columns: list[{key: str, label: str, width?: str}]`, `rows: list[dict]`, `sortable?: bool` | Data table with optional sorting |
| `chart` | `type: 'line'\|'bar'\|'pie'\|'scatter'`, `data: ChartData`, `options?: ChartOptions` | Chart visualization (delegates to charting library) |
| `progress` | `value: float`, `max?: float`, `label?: str`, `variant?: 'bar'\|'circular'` | Progress indicator |
| `alert` | `level: 'info'\|'success'\|'warning'\|'error'`, `title?: str`, `message: str`, `dismissible?: bool` | Alert/notification banner |

#### Layout Containers

| Component | Props | Description |
|-----------|-------|-------------|
| `card` | `title?: str`, `children: list`, `collapsible?: bool`, `collapsed?: bool` | Card container with optional header |
| `grid` | `columns: int`, `gap?: int`, `children: list` | CSS grid layout |
| `flex` | `direction: 'row'\|'column'`, `gap?: int`, `align?: str`, `justify?: str`, `children: list` | Flexbox layout |
| `stack` | `spacing?: int`, `children: list` | Vertical stack (alias for `flex` column) |
| `tabs` | `tabs: list[{id: str, label: str, content: Component}]`, `active?: str` | Tabbed interface |
| `modal` | `title: str`, `content: Component`, `open: bool`, `on_close: str` | Modal overlay dialog |
| `divider` | `orientation?: 'horizontal'\|'vertical'`, `thickness?: int` | Visual separator |

#### Special Components

| Component | Props | Description |
|-----------|-------|-------------|
| `iframe` | `src: str`, `width?: int`, `height?: int`, `sandbox?: list[str]` | Embedded iframe (with CSP restrictions) |
| `custom` | `type: str`, `props: dict` | Extension point for custom components |

### 1.2 Component Tree Example

```json
{
  "type": "card",
  "props": {
    "title": "User Profile Editor"
  },
  "children": [
    {
      "type": "flex",
      "props": {
        "direction": "column",
        "gap": 16
      },
      "children": [
        {
          "type": "input",
          "props": {
            "name": "username",
            "placeholder": "Username",
            "value": "alice",
            "on_change": "evt_username_change"
          }
        },
        {
          "type": "input",
          "props": {
            "name": "email",
            "type": "email",
            "placeholder": "Email",
            "value": "alice@example.com",
            "on_change": "evt_email_change"
          }
        },
        {
          "type": "flex",
          "props": {
            "direction": "row",
            "gap": 8,
            "justify": "flex-end"
          },
          "children": [
            {
              "type": "button",
              "props": {
                "label": "Cancel",
                "variant": "secondary",
                "on_click": "evt_cancel"
              }
            },
            {
              "type": "button",
              "props": {
                "label": "Save",
                "variant": "primary",
                "on_click": "evt_save"
              }
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 2. State Model

### 2.1 Server-Authoritative State

The LLM holds the **canonical state** for each app. The client renders from **state snapshots** delivered via MCP tools.

**State Schema:**
```json
{
  "app_id": "app_12345",
  "version": 42,
  "title": "User Profile Editor",
  "root": {
    "type": "card",
    "props": {...},
    "children": [...]
  },
  "event_handlers": {
    "evt_username_change": {
      "component_path": ["root", 0, 0],
      "event_type": "change",
      "handler_id": "evt_username_change"
    },
    "evt_save": {
      "component_path": ["root", 0, 2, 1],
      "event_type": "click",
      "handler_id": "evt_save"
    }
  },
  "metadata": {
    "created_at": 1706745600,
    "updated_at": 1706745612,
    "session_id": "sess_abc",
    "node_id": "app_node_xyz"
  }
}
```

### 2.2 State Updates

**Update Semantics:**

1. **Full Replace**: Replace entire component tree
   ```python
   app_update(app_id="app_12345", root={...})
   ```

2. **Patch Update**: Partial update via JSON Patch (RFC 6902)
   ```python
   app_patch(app_id="app_12345", patches=[
       {"op": "replace", "path": "/root/children/0/props/value", "value": "new_value"},
       {"op": "add", "path": "/root/children/-", "value": {...}}
   ])
   ```

3. **Component Update**: Update specific component by path
   ```python
   app_update_component(
       app_id="app_12345",
       path=["root", 0, 2],  # Navigate to button
       props={"disabled": True}
   )
   ```

### 2.3 Versioning

- Each state change increments `version` counter
- Client tracks last known version
- Client sends version with events → server detects stale clients
- Server can reject events from outdated versions

**Version Conflict Handling:**
```json
{
  "error": "version_conflict",
  "client_version": 40,
  "server_version": 42,
  "resolution": "refresh"  // Client should re-fetch full state
}
```

---

## 3. Interaction Protocol

### 3.1 Event Flow

```
┌─────────┐                  ┌─────────┐                 ┌─────────┐
│ Client  │                  │   MCP   │                 │   LLM   │
│ (UI)    │                  │  Server │                 │  Agent  │
└────┬────┘                  └────┬────┘                 └────┬────┘
     │                            │                           │
     │  User clicks button        │                           │
     ├────────────────────────────>                           │
     │  app.on_event(             │                           │
     │    app_id="app_12345",     │                           │
     │    handler="evt_save",     │                           │
     │    event={type: "click"},  │                           │
     │    version=42              │                           │
     │  )                         │                           │
     │                            ├──────────────────────────>│
     │                            │  Notification or await    │
     │                            │                           │
     │                            │     LLM processes event   │
     │                            │     Updates state         │
     │                            │                           │
     │                            │<──────────────────────────┤
     │                            │  Updated state (v43)      │
     │<────────────────────────────                           │
     │  State snapshot pushed     │                           │
     │  via WebSocket             │                           │
     │                            │                           │
```

### 3.2 Event Types

**Click Events:**
```json
{
  "type": "click",
  "handler_id": "evt_save",
  "app_id": "app_12345",
  "version": 42,
  "timestamp": 1706745615
}
```

**Input Change Events:**
```json
{
  "type": "change",
  "handler_id": "evt_username_change",
  "app_id": "app_12345",
  "version": 42,
  "value": "alice_updated",
  "name": "username",
  "timestamp": 1706745610
}
```

**Form Submit Events:**
```json
{
  "type": "submit",
  "handler_id": "evt_form_submit",
  "app_id": "app_12345",
  "version": 42,
  "form_data": {
    "username": "alice",
    "email": "alice@example.com"
  },
  "timestamp": 1706745620
}
```

**Navigation Events:**
```json
{
  "type": "navigate",
  "handler_id": "evt_tab_switch",
  "app_id": "app_12345",
  "version": 42,
  "tab_id": "profile_tab",
  "timestamp": 1706745625
}
```

### 3.3 MCP Transport Pattern

**Option A: Notification-Based (Recommended)**

- User events sent as MCP notifications (`notifications/app/event`)
- No blocking wait for response
- State updates pushed via WebSocket or polling
- Better for async, multi-user scenarios

**Option B: Tool Call Response**

- User events trigger `app.on_event` tool call
- LLM responds synchronously with updated state
- Simpler for single-user, turn-based interactions
- Higher latency for complex updates

**Hybrid Approach:** Use notifications for rapid events (input changes), tool calls for critical actions (form submit, save).

---

## 4. Layout System

### 4.1 Composition Model

Apps compose into pages/views using **layout containers** (`card`, `grid`, `flex`, `stack`, `tabs`, `modal`).

**Layout Hierarchy:**
```
App
└─ Page (root container)
   ├─ Header (flex row)
   │  ├─ Logo (image)
   │  └─ Navigation (flex row)
   │     ├─ Tab 1 (button)
   │     └─ Tab 2 (button)
   ├─ Body (grid)
   │  ├─ Sidebar (stack)
   │  │  ├─ Card 1
   │  │  └─ Card 2
   │  └─ Main (stack)
   │     ├─ Content Area
   │     └─ Action Bar
   └─ Footer (flex row)
```

### 4.2 Grid System

```json
{
  "type": "grid",
  "props": {
    "columns": 12,
    "gap": 16,
    "template_areas": [
      "header header header",
      "sidebar main main",
      "footer footer footer"
    ]
  },
  "children": [
    {"type": "card", "props": {"grid_area": "header"}, "children": [...]},
    {"type": "card", "props": {"grid_area": "sidebar"}, "children": [...]},
    {"type": "card", "props": {"grid_area": "main"}, "children": [...]}
  ]
}
```

### 4.3 Responsive Behavior

Layout containers support **breakpoint-aware props**:
```json
{
  "type": "grid",
  "props": {
    "columns": {"mobile": 1, "tablet": 2, "desktop": 3},
    "gap": {"mobile": 8, "tablet": 12, "desktop": 16}
  }
}
```

Client selects props based on viewport width.

### 4.4 Modal Overlays

```json
{
  "type": "modal",
  "props": {
    "title": "Confirm Delete",
    "open": true,
    "on_close": "evt_modal_close"
  },
  "children": [
    {
      "type": "text",
      "props": {"content": "Are you sure you want to delete this item?"}
    },
    {
      "type": "flex",
      "props": {"direction": "row", "gap": 8, "justify": "flex-end"},
      "children": [
        {"type": "button", "props": {"label": "Cancel", "on_click": "evt_cancel"}},
        {"type": "button", "props": {"label": "Delete", "variant": "danger", "on_click": "evt_confirm_delete"}}
      ]
    }
  ]
}
```

---

## 5. App Lifecycle

### 5.1 Lifecycle States

```
┌──────────┐
│  DRAFT   │  App created but not visible
└────┬─────┘
     │ app.show()
     ▼
┌──────────┐
│  VISIBLE │  App rendered to user
└────┬─────┘
     │ app.hide()
     ▼
┌──────────┐
│  HIDDEN  │  App exists but not shown (can be re-shown)
└────┬─────┘
     │ app.destroy()
     ▼
┌──────────┐
│ DESTROYED│  App removed, state discarded
└──────────┘
```

### 5.2 Lifecycle Operations

**Create:**
```python
app = app_create(
    title="User Dashboard",
    root={...},
    visibility="visible"  # or "hidden" for draft
)
# Returns: app_id
```

**Configure (update props, layout):**
```python
app_update(app_id="app_12345", root={...})
app_patch(app_id="app_12345", patches=[...])
```

**Show/Hide:**
```python
app_show(app_id="app_12345")
app_hide(app_id="app_12345")
```

**Destroy:**
```python
app_destroy(app_id="app_12345")
```

### 5.3 Session Persistence

Apps are persisted as **AppNode** context nodes:

```python
@dataclass
class AppNode(ContextNode):
    """MCP App component tree node."""

    app_id: str
    title: str
    root: dict[str, Any]  # Component tree
    event_handlers: dict[str, dict]
    version: int
    visibility: str  # "visible" | "hidden" | "destroyed"

    def GetDigest(self) -> dict:
        return {
            "type": "app",
            "app_id": self.app_id,
            "title": self.title,
            "version": self.version,
            "visibility": self.visibility,
            "component_count": self._count_components(self.root)
        }

    def Render(self, exp: Expansion) -> str:
        if exp == Expansion.COLLAPSED:
            return f"App: {self.title} (v{self.version}, {self.visibility})"
        elif exp == Expansion.SUMMARY:
            return f"App: {self.title}\nComponents: {self._count_components(self.root)}\nState: {self.visibility}"
        else:  # DETAILS/ALL
            return json.dumps({
                "app_id": self.app_id,
                "title": self.title,
                "version": self.version,
                "root": self.root
            }, indent=2)
```

**Session Restore:**
- Apps with `visibility="visible"` are re-rendered on session reload
- Apps with `visibility="hidden"` are loaded but not shown
- Apps with `visibility="destroyed"` are pruned from graph

---

## 6. MCP Tool Surface

### 6.1 Core Tools

#### `app.create`

**Description:** Create a new MCP App.

**Input Schema:**
```json
{
  "title": {"type": "string", "description": "App title"},
  "root": {"type": "object", "description": "Root component tree"},
  "visibility": {"type": "string", "enum": ["visible", "hidden"], "default": "visible"},
  "metadata": {"type": "object", "description": "Optional metadata"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "node_id": "app_node_xyz",
  "version": 1
}
```

#### `app.update`

**Description:** Replace entire component tree.

**Input Schema:**
```json
{
  "app_id": {"type": "string"},
  "root": {"type": "object", "description": "New root component tree"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "version": 2
}
```

#### `app.patch`

**Description:** Apply JSON Patch to state.

**Input Schema:**
```json
{
  "app_id": {"type": "string"},
  "patches": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "op": {"enum": ["add", "remove", "replace", "move", "copy", "test"]},
        "path": {"type": "string"},
        "value": {}
      }
    }
  }
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "version": 3
}
```

#### `app.update_component`

**Description:** Update specific component by path.

**Input Schema:**
```json
{
  "app_id": {"type": "string"},
  "path": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
  "props": {"type": "object", "description": "Props to merge/replace"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "version": 4
}
```

#### `app.get_state`

**Description:** Fetch current app state snapshot.

**Input Schema:**
```json
{
  "app_id": {"type": "string"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "version": 4,
  "title": "User Dashboard",
  "root": {...},
  "event_handlers": {...},
  "metadata": {...}
}
```

#### `app.on_event`

**Description:** Register or handle a user event (tool call variant).

**Input Schema:**
```json
{
  "app_id": {"type": "string"},
  "handler_id": {"type": "string"},
  "event": {"type": "object", "description": "Event payload"},
  "version": {"type": "integer", "description": "Client's last known version"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "version": 5,
  "updates": [...]  // Optional: describe what changed
}
```

#### `app.show` / `app.hide`

**Description:** Toggle app visibility.

**Input Schema:**
```json
{
  "app_id": {"type": "string"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "visibility": "visible"
}
```

#### `app.destroy`

**Description:** Destroy app and free resources.

**Input Schema:**
```json
{
  "app_id": {"type": "string"}
}
```

**Returns:**
```json
{
  "app_id": "app_12345",
  "destroyed": true
}
```

### 6.2 MCP Notifications

**`notifications/app/event`** (Notification-based event delivery)

**Payload:**
```json
{
  "app_id": "app_12345",
  "handler_id": "evt_save",
  "event": {
    "type": "click",
    "timestamp": 1706745615
  },
  "version": 42
}
```

**`notifications/app/state_changed`** (Server pushes state updates)

**Payload:**
```json
{
  "app_id": "app_12345",
  "version": 43,
  "patches": [...]  // Optional: incremental changes for optimization
}
```

---

## 7. Rendering Contract

### 7.1 Client Responsibilities

The client (ActiveContext dashboard or standalone viewer) receives JSON component trees and:

1. **Parse** component tree into virtual DOM
2. **Render** components to HTML/React/Vue
3. **Bind** event handlers to DOM elements
4. **Serialize** user events and send to MCP server
5. **Apply** state updates (full replace or patches)
6. **Track** version numbers for conflict detection

### 7.2 Component Rendering Mapping

| Component | HTML Rendering |
|-----------|----------------|
| `text` | `<span>` or `<p>` with styling |
| `heading` | `<h1>` to `<h6>` |
| `markdown` | Parsed markdown → HTML (via marked.js or similar) |
| `code` | `<pre><code class="language-{lang}">` with Prism.js highlighting |
| `image` | `<img src="{src}" alt="{alt}">` |
| `button` | `<button class="{variant}" onclick="{handler}">` |
| `input` | `<input type="{type}" name="{name}" value="{value}" onchange="{handler}">` |
| `select` | `<select name="{name}" onchange="{handler}"><option>...</option></select>` |
| `table` | `<table><thead><tr><th>...</th></tr></thead><tbody><tr><td>...</td></tr></tbody></table>` |
| `chart` | Delegates to Chart.js / Recharts / D3 |
| `card` | `<div class="card"><div class="card-header">{title}</div><div class="card-body">{children}</div></div>` |
| `grid` | `<div style="display: grid; grid-template-columns: repeat({columns}, 1fr); gap: {gap}px;">{children}</div>` |
| `flex` | `<div style="display: flex; flex-direction: {direction}; gap: {gap}px;">{children}</div>` |
| `modal` | `<div class="modal-overlay"><div class="modal-content">{children}</div></div>` (with portal rendering) |

### 7.3 Event Handler Registration

```javascript
// Pseudocode for client-side rendering
function renderComponent(component) {
  const element = createElementFromType(component.type);

  // Apply props
  Object.entries(component.props).forEach(([key, value]) => {
    if (key.startsWith('on_')) {
      // Bind event handler
      const eventType = key.substring(3); // "on_click" -> "click"
      element.addEventListener(eventType, (e) => {
        sendEventToMCP({
          app_id: currentAppId,
          handler_id: value,  // e.g., "evt_save"
          event: serializeEvent(e),
          version: currentVersion
        });
      });
    } else {
      element.setAttribute(key, value);
    }
  });

  // Render children
  if (component.children) {
    component.children.forEach(child => {
      element.appendChild(renderComponent(child));
    });
  }

  return element;
}
```

### 7.4 State Update Handling

**Full Replace:**
```javascript
function handleStateUpdate(newState) {
  currentVersion = newState.version;
  const newTree = renderComponent(newState.root);
  container.replaceChildren(newTree);
}
```

**Patch Update:**
```javascript
function handleStatePatch(patches) {
  patches.forEach(patch => {
    applyJsonPatch(currentState, patch);
  });
  currentVersion++;
  // Re-render affected components (virtual DOM diffing)
  updateDOM();
}
```

---

## 8. Integration with ActiveContext

### 8.1 AppNode in Context Graph

Apps are first-class context nodes:

```
context (root)
├─ system_prompt
├─ session (SessionNode)
├─ mcp_manager (MCPManagerNode)
├─ app_dashboard (AppNode) ← MCP App
│  ├─ app_settings (AppNode)
│  └─ app_debugger (AppNode)
├─ v1 (TextNode)
└─ g1 (GroupNode)
```

**DSL Integration:**
```python
# Create app from timeline
app1 = app_create(
    title="User Profile",
    root={
        "type": "card",
        "props": {"title": "Profile"},
        "children": [...]
    }
)

# Link to group
link(profile_group, app1)

# Update via DSL
app1.SetVisibility("hidden")
app1.UpdateRoot({...})
```

### 8.2 Dashboard Integration

The existing ActiveContext dashboard (`src/activecontext/dashboard/`) extends to support MCP Apps:

**New Routes:**
- `GET /api/apps` → List all apps
- `GET /api/apps/{app_id}` → Get app state
- `POST /api/apps` → Create app (delegates to MCP tool)
- `PATCH /api/apps/{app_id}` → Update app
- `WS /ws/apps/{app_id}` → WebSocket for real-time updates

**Frontend:**
- App viewer component (renders component trees)
- Event dispatcher (sends events to backend)
- State synchronization (tracks versions, applies patches)

### 8.3 Projection Engine

Apps appear in projections based on NodeState:

| NodeState | Rendering |
|-----------|-----------|
| HIDDEN | Not included |
| COLLAPSED | `App: {title} (v{version}, {visibility})` |
| SUMMARY | `App: {title}\nComponents: {count}\nState: {visibility}` |
| DETAILS | Full JSON component tree |
| ALL | Component tree + event handler metadata |

**Token Budget:**
- Apps with `visibility="visible"` consume tokens proportional to tree depth
- Apps with `visibility="hidden"` only consume COLLAPSED tokens
- LLM can adjust app NodeState to manage token budget

---

## 9. Security Considerations

### 9.1 Sandboxing

**iframe Sandboxing:**
- `iframe` components enforce CSP restrictions
- Default sandbox: `allow-scripts allow-same-origin`
- Configurable via `sandbox` prop

**Script Injection:**
- Component props are sanitized (escape HTML entities)
- Event handlers are ID references, not inline JS
- No `eval()` or `Function()` constructor in rendering

### 9.2 Permission Model

Apps inherit ActiveContext's permission system:

```yaml
sandbox:
  app_permissions:
    - app_id: "app_*"
      allowed_tools: ["app.create", "app.update", "app.on_event"]
    - app_id: "admin_app"
      allowed_tools: ["*"]
      require_confirmation: true
```

### 9.3 Rate Limiting

- Event notifications rate-limited (e.g., 100/sec per app)
- Prevents DOS from malicious clients
- Configurable per app or globally

### 9.4 Data Validation

- Component props validated against schema
- Invalid components rejected with error response
- Client-side validation mirrors server-side rules

---

## 10. Use Cases & Examples

### 10.1 Form Builder

**Scenario:** LLM creates a dynamic form based on user request.

```python
app = app_create(
    title="User Registration",
    root={
        "type": "card",
        "props": {"title": "Register"},
        "children": [
            {
                "type": "stack",
                "props": {"spacing": 12},
                "children": [
                    {
                        "type": "input",
                        "props": {
                            "name": "username",
                            "placeholder": "Username",
                            "on_change": "evt_username"
                        }
                    },
                    {
                        "type": "input",
                        "props": {
                            "name": "password",
                            "type": "password",
                            "placeholder": "Password",
                            "on_change": "evt_password"
                        }
                    },
                    {
                        "type": "button",
                        "props": {
                            "label": "Register",
                            "variant": "primary",
                            "on_click": "evt_submit"
                        }
                    }
                ]
            }
        ]
    }
)

# Handle submit event
def on_evt_submit(event):
    form_data = event["form_data"]
    # Validate, create user, update app state
    app_update(
        app_id=app["app_id"],
        root={
            "type": "alert",
            "props": {
                "level": "success",
                "message": "Registration successful!"
            }
        }
    )
```

### 10.2 Data Dashboard

**Scenario:** LLM generates a dashboard with charts and tables.

```python
app = app_create(
    title="Analytics Dashboard",
    root={
        "type": "grid",
        "props": {"columns": 2, "gap": 16},
        "children": [
            {
                "type": "card",
                "props": {"title": "User Growth"},
                "children": [
                    {
                        "type": "chart",
                        "props": {
                            "type": "line",
                            "data": {
                                "labels": ["Jan", "Feb", "Mar"],
                                "datasets": [
                                    {
                                        "label": "Users",
                                        "data": [100, 200, 350]
                                    }
                                ]
                            }
                        }
                    }
                ]
            },
            {
                "type": "card",
                "props": {"title": "Top Users"},
                "children": [
                    {
                        "type": "table",
                        "props": {
                            "columns": [
                                {"key": "name", "label": "Name"},
                                {"key": "score", "label": "Score"}
                            ],
                            "rows": [
                                {"name": "Alice", "score": 95},
                                {"name": "Bob", "score": 87}
                            ]
                        }
                    }
                ]
            }
        ]
    }
)
```

### 10.3 Interactive Wizard

**Scenario:** Multi-step wizard with tab navigation.

```python
app = app_create(
    title="Setup Wizard",
    root={
        "type": "tabs",
        "props": {
            "active": "step1",
            "tabs": [
                {
                    "id": "step1",
                    "label": "Step 1: Details",
                    "content": {
                        "type": "input",
                        "props": {"name": "project_name", "placeholder": "Project Name"}
                    }
                },
                {
                    "id": "step2",
                    "label": "Step 2: Settings",
                    "content": {
                        "type": "checkbox",
                        "props": {"name": "enable_logging", "label": "Enable Logging"}
                    }
                },
                {
                    "id": "step3",
                    "label": "Step 3: Review",
                    "content": {
                        "type": "text",
                        "props": {"content": "Review your settings and click Finish."}
                    }
                }
            ]
        }
    }
)

# Navigate to next step
def on_next_step(event):
    current_step = get_active_tab(app)
    next_step = get_next_tab(current_step)
    app_patch(
        app_id=app["app_id"],
        patches=[
            {"op": "replace", "path": "/root/props/active", "value": next_step}
        ]
    )
```

---

## 11. Future Extensions

### 11.1 Component Library Registry

- Allow custom components via registry
- LLM queries available components
- Components bundled as JS/WASM modules

### 11.2 Multi-User Collaboration

- Shared app state across sessions
- Operational Transform (OT) or CRDT for conflict resolution
- Presence indicators (who's viewing/editing)

### 11.3 App Templates

- Pre-built app templates (dashboard, form, wizard)
- LLM instantiates from template with custom data
- Template marketplace

### 11.4 Advanced Charting

- Time-series streaming
- Real-time updates via WebSocket
- Zoom, pan, export controls

### 11.5 Accessibility (a11y)

- ARIA attributes in component schema
- Keyboard navigation support
- Screen reader compatibility

---

## 12. Implementation Roadmap

### Phase 1: Core Infrastructure (2-3 weeks)

- [ ] Define `AppNode` class in `src/activecontext/context/nodes.py`
- [ ] Implement MCP tools (`app.create`, `app.update`, `app.get_state`)
- [ ] Add app lifecycle management (create, show, hide, destroy)
- [ ] Session persistence (save/load AppNode state)

### Phase 2: Event System (1-2 weeks)

- [ ] Event handler registration in component tree
- [ ] MCP notification transport for events
- [ ] Version tracking and conflict detection
- [ ] Client-side event serialization

### Phase 3: Dashboard Integration (2-3 weeks)

- [ ] Extend dashboard routes (`/api/apps`, `/ws/apps`)
- [ ] Component renderer (React or vanilla JS)
- [ ] State synchronization (WebSocket updates)
- [ ] Event dispatcher (click, change, submit)

### Phase 4: Component Library (2-3 weeks)

- [ ] Implement primitive components (text, button, input, table, chart)
- [ ] Layout containers (card, grid, flex, tabs, modal)
- [ ] Styling system (CSS classes, inline styles)
- [ ] Chart integration (Chart.js or Recharts)

### Phase 5: Testing & Polish (1-2 weeks)

- [ ] Unit tests for AppNode, MCP tools
- [ ] Integration tests (event flow, state updates)
- [ ] Example apps (form, dashboard, wizard)
- [ ] Documentation and demos

---

## 13. Open Questions

1. **Chart Library Choice:** Chart.js (simpler) vs. D3 (more flexible)?
2. **React vs. Vanilla JS:** Use React for dashboard or keep lightweight?
3. **JSON Patch Library:** Use `jsonpatch` (Python) / `fast-json-patch` (JS)?
4. **Multi-App Coordination:** How do multiple apps share state or communicate?
5. **Performance:** How to optimize rendering for large component trees (1000+ nodes)?

---

## Appendix A: JSON Schema for Component Tree

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MCP App Component Tree",
  "type": "object",
  "properties": {
    "type": {
      "type": "string",
      "enum": ["text", "heading", "markdown", "code", "image", "button", "input", "textarea", "select", "checkbox", "radio_group", "table", "chart", "progress", "alert", "card", "grid", "flex", "stack", "tabs", "modal", "divider", "iframe", "custom"]
    },
    "props": {
      "type": "object",
      "additionalProperties": true
    },
    "children": {
      "type": "array",
      "items": {
        "$ref": "#"
      }
    }
  },
  "required": ["type", "props"]
}
```

---

## Appendix B: Complete Tool Definitions

See [MCP Tool Schema Documentation](https://modelcontextprotocol.io/docs/tools) for detailed specifications.

**Example: `app.create` Tool Definition**

```json
{
  "name": "app.create",
  "description": "Create a new MCP App with a component tree",
  "inputSchema": {
    "type": "object",
    "properties": {
      "title": {
        "type": "string",
        "description": "App title for display"
      },
      "root": {
        "type": "object",
        "description": "Root component tree (see Component Tree schema)"
      },
      "visibility": {
        "type": "string",
        "enum": ["visible", "hidden"],
        "default": "visible",
        "description": "Initial visibility state"
      },
      "metadata": {
        "type": "object",
        "description": "Optional app metadata (tags, owner, etc.)"
      }
    },
    "required": ["title", "root"]
  }
}
```

---

## Appendix C: Diagram - Event Flow Sequence

```
User Action → Client Event Handler → MCP Notification → LLM Agent
                                                              ↓
                                                        Process Event
                                                              ↓
                                                        Update State
                                                              ↓
Client State Update ← WebSocket Push ← MCP Server ← State Changed
        ↓
    Re-render UI
```

---

## Appendix D: Reference Implementation Sketch

**Timeline DSL:**
```python
# Create app
app1 = app_create(
    title="My Dashboard",
    root={
        "type": "card",
        "props": {"title": "Overview"},
        "children": [
            {"type": "text", "props": {"content": "Welcome!"}}
        ]
    }
)

# Update component
app1.UpdateComponent(
    path=["root", 0],
    props={"content": "Updated message"}
)

# Handle event (async)
@app1.on_event("evt_refresh")
async def handle_refresh(event):
    data = await fetch_data()
    app1.Patch([
        {"op": "replace", "path": "/root/children/0/props/content", "value": data}
    ])
```

**Backend Implementation (pseudo):**
```python
# src/activecontext/mcp/app_tools.py
async def app_create(title: str, root: dict, visibility: str = "visible") -> dict:
    """Create MCP App."""
    app_id = f"app_{uuid.uuid4().hex[:8]}"
    node = AppNode(
        node_id=f"app_node_{uuid.uuid4().hex[:8]}",
        app_id=app_id,
        title=title,
        root=root,
        event_handlers={},
        version=1,
        visibility=visibility
    )

    # Add to context graph
    session.graph.add_node(node)
    link(session.graph.get_node("context"), node)

    return {
        "app_id": app_id,
        "node_id": node.node_id,
        "version": node.version
    }
```

---

**End of Design Document**
