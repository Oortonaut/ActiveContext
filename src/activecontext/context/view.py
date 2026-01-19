"""Per-agent view of shared content.

AgentView represents how a specific agent sees a piece of content.
Each agent can have different visibility settings (hidden, state)
for the same underlying ContentData.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.state import NodeState

if TYPE_CHECKING:
    from activecontext.context.content import ContentData, ContentRegistry


@dataclass
class AgentView:
    """Per-agent view of content.

    Represents how a specific agent sees a piece of shared content.
    Multiple AgentViews can reference the same ContentData but have
    different visibility settings.

    Attributes:
        view_id: Unique identifier for this view
        agent_id: Which agent owns this view
        content_id: Reference to ContentData
        node_id: Original node ID (for backward compatibility with DSL)
        hidden: Whether the view is hidden (orthogonal to state)
        state: Expansion state (COLLAPSED, SUMMARY, DETAILS, ALL)
        tokens: Token budget for this view
        parent_ids: Parent view IDs in the agent's DAG
        children_ids: Child view IDs in the agent's DAG
        mode: "running" or "paused" for tick processing
        created_at: Unix timestamp of creation
        updated_at: Unix timestamp of last update
        tags: Arbitrary metadata
    """

    view_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_id: str = ""
    content_id: str = ""
    node_id: str = ""  # For DSL compatibility

    # Visibility settings (per-agent)
    hidden: bool = False
    state: NodeState = NodeState.DETAILS
    tokens: int = 1000

    # DAG structure (per-agent)
    parent_ids: set[str] = field(default_factory=set)
    children_ids: set[str] = field(default_factory=set)

    # Tick control
    mode: str = "paused"

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Metadata
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set node_id to view_id if not provided."""
        if not self.node_id:
            self.node_id = self.view_id

    def render(
        self,
        content: "ContentData",
        budget: int | None = None,
    ) -> str:
        """Render this view of the content.

        Args:
            content: The ContentData to render
            budget: Token budget override (uses self.tokens if None)

        Returns:
            Rendered string based on state and hidden flag
        """
        effective_budget = budget if budget is not None else self.tokens

        # Hidden content shows placeholder
        if self.hidden:
            return f"[{content.content_type}: {content.token_count} tokens]"

        # Render based on state
        if self.state == NodeState.COLLAPSED:
            return self._render_collapsed(content)
        elif self.state == NodeState.SUMMARY:
            return self._render_summary(content, effective_budget)
        elif self.state == NodeState.DETAILS:
            return self._render_details(content, effective_budget)
        elif self.state == NodeState.ALL:
            return self._render_all(content, effective_budget)
        else:
            # HIDDEN state (legacy) - treat as hidden flag
            return f"[{content.content_type}: {content.token_count} tokens]"

    def _render_collapsed(self, content: "ContentData") -> str:
        """Render collapsed view - metadata only."""
        info = content.source_info
        type_label = content.content_type.title()

        if content.content_type == "file":
            path = info.get("path", "unknown")
            return f"[{type_label}: {path} ({content.token_count} tokens)]"
        elif content.content_type == "shell":
            cmd = info.get("command", "")[:30]
            return f"[{type_label}: {cmd}... ({content.token_count} tokens)]"
        else:
            return f"[{type_label}: {content.token_count} tokens]"

    def _render_summary(self, content: "ContentData", budget: int) -> str:
        """Render summary view."""
        if content.summary:
            # Truncate if over budget
            if content.summary_tokens > budget:
                ratio = budget / content.summary_tokens
                chars = int(len(content.summary) * ratio)
                return content.summary[:chars] + f"... [{content.token_count} tokens total]"
            return content.summary + f"\n[{content.token_count} tokens total]"
        else:
            # No summary yet - show collapsed
            return self._render_collapsed(content) + " [no summary]"

    def _render_details(self, content: "ContentData", budget: int) -> str:
        """Render details view - full content."""
        if content.token_count <= budget:
            return content.raw_content

        # Truncate if over budget
        ratio = budget / content.token_count
        chars = int(len(content.raw_content) * ratio)
        truncated = content.token_count - budget
        return content.raw_content[:chars] + f"\n... [{truncated} tokens truncated]"

    def _render_all(self, content: "ContentData", budget: int) -> str:
        """Render ALL view - summary + details."""
        parts = []

        # Summary first if available
        if content.summary:
            summary_budget = min(budget // 4, content.summary_tokens)
            parts.append("## Summary")
            parts.append(self._render_summary(content, summary_budget))
            budget -= summary_budget

        # Then details
        parts.append("## Details")
        parts.append(self._render_details(content, budget))

        return "\n\n".join(parts)

    # Fluent API
    def SetHidden(self, hidden: bool) -> "AgentView":
        """Set hidden flag."""
        self.hidden = hidden
        self.updated_at = time.time()
        return self

    def SetState(self, state: NodeState) -> "AgentView":
        """Set expansion state."""
        self.state = state
        self.updated_at = time.time()
        return self

    def SetTokens(self, tokens: int) -> "AgentView":
        """Set token budget."""
        self.tokens = tokens
        self.updated_at = time.time()
        return self

    def Run(self) -> "AgentView":
        """Enable tick processing."""
        self.mode = "running"
        self.updated_at = time.time()
        return self

    def Pause(self) -> "AgentView":
        """Disable tick processing."""
        self.mode = "paused"
        self.updated_at = time.time()
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "view_id": self.view_id,
            "agent_id": self.agent_id,
            "content_id": self.content_id,
            "node_id": self.node_id,
            "hidden": self.hidden,
            "state": self.state.value,
            "tokens": self.tokens,
            "parent_ids": list(self.parent_ids),
            "children_ids": list(self.children_ids),
            "mode": self.mode,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentView":
        """Deserialize from dictionary."""
        return cls(
            view_id=data["view_id"],
            agent_id=data.get("agent_id", ""),
            content_id=data.get("content_id", ""),
            node_id=data.get("node_id", data["view_id"]),
            hidden=data.get("hidden", False),
            state=NodeState(data.get("state", "details")),
            tokens=data.get("tokens", 1000),
            parent_ids=set(data.get("parent_ids", [])),
            children_ids=set(data.get("children_ids", [])),
            mode=data.get("mode", "paused"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tags=data.get("tags", {}),
        )


class ViewRegistry:
    """Registry for agent views.

    Manages AgentView instances, indexed by (agent_id, view_id) or
    (agent_id, node_id) for backward compatibility.
    """

    def __init__(self, content_registry: "ContentRegistry") -> None:
        """Initialize with a content registry reference.

        Args:
            content_registry: ContentRegistry for resolving content_ids
        """
        self._content_registry = content_registry
        self._views: dict[str, AgentView] = {}  # view_id -> AgentView
        self._by_agent: dict[str, set[str]] = {}  # agent_id -> set of view_ids
        self._by_node: dict[str, str] = {}  # node_id -> view_id (for DSL compat)

    def register(self, view: AgentView) -> str:
        """Register a view and return its ID.

        Args:
            view: AgentView to register

        Returns:
            The view_id
        """
        self._views[view.view_id] = view

        # Index by agent
        if view.agent_id not in self._by_agent:
            self._by_agent[view.agent_id] = set()
        self._by_agent[view.agent_id].add(view.view_id)

        # Index by node_id for DSL compatibility
        if view.node_id:
            self._by_node[view.node_id] = view.view_id

        return view.view_id

    def get(self, view_id: str) -> AgentView | None:
        """Get view by ID."""
        return self._views.get(view_id)

    def get_by_node(self, node_id: str) -> AgentView | None:
        """Get view by node_id (for DSL compatibility)."""
        view_id = self._by_node.get(node_id)
        if view_id:
            return self._views.get(view_id)
        return None

    def get_agent_views(self, agent_id: str) -> list[AgentView]:
        """Get all views for an agent."""
        view_ids = self._by_agent.get(agent_id, set())
        return [self._views[vid] for vid in view_ids if vid in self._views]

    def remove(self, view_id: str) -> bool:
        """Remove a view from registry."""
        view = self._views.get(view_id)
        if not view:
            return False

        del self._views[view_id]

        # Remove from agent index
        if view.agent_id in self._by_agent:
            self._by_agent[view.agent_id].discard(view_id)

        # Remove from node index
        if view.node_id in self._by_node:
            del self._by_node[view.node_id]

        return True

    def render_view(self, view_id: str, budget: int | None = None) -> str | None:
        """Render a view, resolving its content.

        Args:
            view_id: View ID to render
            budget: Optional token budget override

        Returns:
            Rendered string, or None if view/content not found
        """
        view = self._views.get(view_id)
        if not view:
            return None

        content = self._content_registry.get(view.content_id)
        if not content:
            return f"[Content {view.content_id} not found]"

        return view.render(content, budget)

    def __len__(self) -> int:
        """Return number of registered views."""
        return len(self._views)
