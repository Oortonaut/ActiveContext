"""Projection engine for building token-aware LLM context.

The ProjectionEngine transforms session state (conversation, views, groups)
into a single Projection that becomes the LLM's context after the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from activecontext.session.protocols import Projection, ProjectionSection

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import ContextNode


@dataclass
class ProjectionConfig:
    """Configuration for projection building."""

    total_budget: int = 8000
    conversation_ratio: float = 0.3  # 30% for conversation
    views_ratio: float = 0.5  # 50% for views
    groups_ratio: float = 0.2  # 20% for groups


class ProjectionEngine:
    """Builds token-aware projections from session state.

    The projection engine is responsible for:
    1. Allocating token budget across components
    2. Rendering views at appropriate LOD levels
    3. Summarizing conversation history if needed
    4. Assembling the final projection
    """

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        if config:
            self.config = config
        else:
            # Try to load from app config
            self.config = self._config_from_app_config()

    def _config_from_app_config(self) -> ProjectionConfig:
        """Build ProjectionConfig from app config or defaults."""
        try:
            from activecontext.config import get_config

            app_config = get_config()
            proj = app_config.projection
            return ProjectionConfig(
                total_budget=proj.total_budget if proj.total_budget is not None else 8000,
                conversation_ratio=(
                    proj.conversation_ratio if proj.conversation_ratio is not None else 0.3
                ),
                views_ratio=proj.views_ratio if proj.views_ratio is not None else 0.5,
                groups_ratio=proj.groups_ratio if proj.groups_ratio is not None else 0.2,
            )
        except ImportError:
            return ProjectionConfig()

    def build(
        self,
        *,
        context_graph: ContextGraph | None = None,
        context_objects: dict[str, Any] | None = None,
        conversation: list[Any],  # list[Message]
        cwd: str = ".",
        token_budget: int | None = None,
    ) -> Projection:
        """Build a projection from current session state.

        Args:
            context_graph: ContextGraph (DAG of nodes) - preferred
            context_objects: ViewHandle/GroupHandle instances - legacy fallback
            conversation: Message history
            cwd: Working directory for file access
            token_budget: Override total token budget

        Returns:
            Complete Projection ready for LLM
        """
        budget = token_budget or self.config.total_budget
        sections: list[ProjectionSection] = []

        # Compute budget allocation
        conv_budget = int(budget * self.config.conversation_ratio)
        tree_budget = int(budget * (self.config.views_ratio + self.config.groups_ratio))

        # 1. Render conversation history
        conv_section = self._render_conversation(conversation, conv_budget)
        if conv_section:
            sections.append(conv_section)

        # 2. Render context (prefer graph, fall back to legacy objects)
        if context_graph and len(context_graph) > 0:
            graph_sections = self._render_graph(context_graph, tree_budget, cwd)
            sections.extend(graph_sections)

            # Build handles dict from graph
            handles = {
                node.node_id: node.GetDigest()
                for node in context_graph
            }
        elif context_objects:
            # Legacy path: render views and groups separately
            views_budget = int(budget * self.config.views_ratio)
            groups_budget = int(budget * self.config.groups_ratio)

            views = {k: v for k, v in context_objects.items() if self._is_view(v)}
            view_sections = self._render_views(views, views_budget, cwd)
            sections.extend(view_sections)

            groups = {k: v for k, v in context_objects.items() if self._is_group(v)}
            group_sections = self._render_groups(groups, groups_budget)
            sections.extend(group_sections)

            handles = {
                obj_id: obj.GetDigest()
                for obj_id, obj in context_objects.items()
                if hasattr(obj, "GetDigest")
            }
        else:
            handles = {}

        return Projection(
            sections=sections,
            token_budget=budget,
            handles=handles,
        )

    def _render_graph(
        self,
        graph: ContextGraph,
        budget: int,
        cwd: str,
    ) -> list[ProjectionSection]:
        """Render visible nodes from the context graph.

        Visible nodes are:
        - Running nodes (mode="running")
        - Root nodes that are paused (explicitly included)

        Args:
            graph: The context graph
            budget: Token budget for all graph content
            cwd: Working directory for file access

        Returns:
            List of ProjectionSections for visible nodes
        """
        sections: list[ProjectionSection] = []

        # Collect visible nodes
        visible_nodes = self._collect_visible_nodes(graph)

        if not visible_nodes:
            return sections

        # Allocate budget proportionally
        per_node_budget = budget // len(visible_nodes)

        for node in visible_nodes:
            content = node.Render(tokens=per_node_budget, cwd=cwd)

            sections.append(
                ProjectionSection(
                    section_type=node.node_type,
                    source_id=node.node_id,
                    content=content,
                    tokens_used=len(content) // 4,
                    lod=node.lod,
                    metadata=node.GetDigest(),
                )
            )

            # Clear pending diffs after rendering
            node.clear_pending_diffs()

        return sections

    def _collect_visible_nodes(self, graph: ContextGraph) -> list[ContextNode]:
        """Collect nodes that should be rendered in projection.

        Visibility rules:
        - Running nodes are always visible
        - Paused root nodes are visible (explicit includes)
        - Non-root paused nodes are not visible (summarized by parent)

        Args:
            graph: The context graph

        Returns:
            List of nodes to render
        """
        visible: list[ContextNode] = []

        # Running nodes are always visible
        visible.extend(graph.get_running_nodes())

        # Add paused root nodes (nodes with no parents)
        for node in graph.get_roots():
            if node.mode == "paused" and node not in visible:
                visible.append(node)

        return visible

    def _render_conversation(
        self,
        conversation: list[Any],
        budget: int,
    ) -> ProjectionSection | None:
        """Render conversation history within token budget."""
        if not conversation:
            return None

        parts = ["## Conversation History\n"]
        tokens_used = 50  # header overhead
        char_budget = budget * 4

        # Process messages (most recent first for summarization)
        messages_to_include: list[str] = []
        for msg in reversed(conversation):
            role = msg.role.value.upper()
            actor = getattr(msg, "actor", None)
            content = msg.content

            # Truncate very long messages
            if len(content) > 2000:
                content = content[:2000] + "..."

            # Format with actor if present: "USER (user):" or "ASSISTANT (agent):"
            if actor:
                entry = f"**{role}** ({actor}): {content}\n\n"
            else:
                entry = f"**{role}**: {content}\n\n"
            entry_chars = len(entry)

            if tokens_used * 4 + entry_chars > char_budget:
                # Summarize remaining messages
                remaining = len(conversation) - len(messages_to_include)
                if remaining > 0:
                    parts.append(f"[{remaining} earlier messages omitted]\n\n")
                break

            messages_to_include.insert(0, entry)  # Prepend to maintain order
            tokens_used += entry_chars // 4

        parts.extend(messages_to_include)

        return ProjectionSection(
            section_type="conversation",
            source_id="conversation",
            content="".join(parts),
            tokens_used=tokens_used,
            lod=0,
        )

    def _render_views(
        self,
        views: dict[str, Any],
        budget: int,
        cwd: str,
    ) -> list[ProjectionSection]:
        """Render views with fair token allocation."""
        if not views:
            return []

        sections = []
        per_view_budget = budget // len(views) if views else budget

        for view_id, view in views.items():
            # Ensure view has cwd set
            if hasattr(view, "_cwd") and not view._cwd:
                view._cwd = cwd

            if hasattr(view, "Render"):
                content = view.Render(tokens=per_view_budget)
            else:
                # Fallback for views without Render
                digest = view.GetDigest() if hasattr(view, "GetDigest") else {}
                content = f"[View {view_id}: {digest.get('path', '?')}]"

            tokens_used = len(content) // 4

            sections.append(
                ProjectionSection(
                    section_type="view",
                    source_id=view_id,
                    content=content,
                    tokens_used=tokens_used,
                    lod=view.lod if hasattr(view, "lod") else 0,
                    metadata=view.GetDigest() if hasattr(view, "GetDigest") else {},
                )
            )

        return sections

    def _render_groups(
        self,
        groups: dict[str, Any],
        budget: int,
    ) -> list[ProjectionSection]:
        """Render groups (summarized member content)."""
        if not groups:
            return []

        sections = []
        per_group_budget = budget // len(groups) if groups else budget

        for group_id, group in groups.items():
            if hasattr(group, "Render"):
                content = group.Render(tokens=per_group_budget)
            else:
                member_count = len(group.members) if hasattr(group, "members") else 0
                content = f"[Group {group_id}: {member_count} members]"

            sections.append(
                ProjectionSection(
                    section_type="group",
                    source_id=group_id,
                    content=content,
                    tokens_used=len(content) // 4,
                    lod=group.lod if hasattr(group, "lod") else 1,
                    metadata=group.GetDigest() if hasattr(group, "GetDigest") else {},
                )
            )

        return sections

    def _is_view(self, obj: Any) -> bool:
        """Check if object is a view."""
        if hasattr(obj, "GetDigest"):
            digest = obj.GetDigest()
            return bool(digest.get("type") == "view")
        return False

    def _is_group(self, obj: Any) -> bool:
        """Check if object is a group."""
        if hasattr(obj, "GetDigest"):
            digest = obj.GetDigest()
            return bool(digest.get("type") == "group")
        return False
