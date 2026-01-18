"""Projection engine for building token-aware LLM context.

The ProjectionEngine transforms session state (conversation, views, groups)
into a single Projection that becomes the LLM's context after the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from activecontext.context.state import NodeState
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


    def _get_user_display_name(self) -> str:
        """Get the user's display name from config or environment.

        Resolution order:
        1. Config file: user.display_name
        2. Environment: USER (Unix) or USERNAME (Windows)
        3. Default: "User"
        """
        import os

        try:
            from activecontext.config import get_config

            app_config = get_config()
            if (
                hasattr(app_config, "user")
                and app_config.user
                and app_config.user.display_name
            ):
                return app_config.user.display_name
        except ImportError:
            pass

        # Fall back to environment
        return os.environ.get("USER") or os.environ.get("USERNAME") or "User"

    def build(
        self,
        *,
        context_graph: ContextGraph | None = None,
        context_objects: dict[str, Any] | None = None,
        conversation: list[Any],  # list[Message]
        cwd: str = ".",
        token_budget: int | None = None,
        show_message_actors: bool = True,
    ) -> Projection:
        """Build a projection from current session state.

        Args:
            context_graph: ContextGraph (DAG of nodes) - preferred
            context_objects: ViewHandle/GroupHandle instances - legacy fallback
            conversation: Message history
            cwd: Working directory for file access
            token_budget: Override total token budget
            show_message_actors: Whether to show message actors in conversation

        Returns:
            Complete Projection ready for LLM
        """
        budget = token_budget or self.config.total_budget
        sections: list[ProjectionSection] = []

        # Compute budget allocation
        conv_budget = int(budget * self.config.conversation_ratio)
        tree_budget = int(budget * (self.config.views_ratio + self.config.groups_ratio))

        # 1. Render conversation history
        # Prefer MessageNodes from graph if available, fall back to legacy conversation list
        conv_section = None
        if context_graph:
            conv_section = self._render_messages(
                context_graph,
                conv_budget,
                user_display_name=self._get_user_display_name(),
            )
        if not conv_section and conversation:
            # Fall back to legacy conversation rendering
            conv_section = self._render_conversation(
                conversation, conv_budget, show_actors=show_message_actors
            )
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
            # Skip hidden nodes (state=HIDDEN means not in projection)
            if node.state == NodeState.HIDDEN:
                continue

            content = node.Render(tokens=per_node_budget, cwd=cwd)

            sections.append(
                ProjectionSection(
                    section_type=node.node_type,
                    source_id=node.node_id,
                    content=content,
                    tokens_used=len(content) // 4,
                    state=node.state,
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
        show_actors: bool = True,
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

            # Format with actor if present and show_actors is True
            if show_actors and actor:
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
            state=NodeState.DETAILS,
        )

    def _render_messages(
        self,
        graph: ContextGraph,
        budget: int,
        user_display_name: str = "User",
    ) -> ProjectionSection | None:
        """Render conversation from MessageNodes with role alternation.

        Groups adjacent same-role messages into blocks for proper LLM
        pretraining compatibility. Each block shows the first message's ID.

        Args:
            graph: Context graph containing MessageNodes
            budget: Token budget for conversation
            user_display_name: Display name for user messages (e.g., "Ace")

        Returns:
            ProjectionSection with rendered conversation, or None if no messages
        """
        from activecontext.context.nodes import MessageNode

        # Get all message nodes sorted by creation time
        msg_nodes: list[MessageNode] = []
        for node in graph:
            if isinstance(node, MessageNode):
                msg_nodes.append(node)

        if not msg_nodes:
            return None

        # Sort by creation time
        msg_nodes.sort(key=lambda n: n.created_at)

        # Group into blocks by effective role
        blocks: list[tuple[str, list[MessageNode]]] = []  # (role, nodes)
        for node in msg_nodes:
            effective_role = node.effective_role  # "USER" or "ASSISTANT"
            if blocks and blocks[-1][0] == effective_role:
                blocks[-1][1].append(node)  # Merge into current block
            else:
                blocks.append((effective_role, [node]))  # New block

        # Render blocks
        parts = ["## Conversation\n\n"]
        tokens_used = 50  # header overhead
        char_budget = budget * 4

        for role, nodes in blocks:
            # Get block ID from first node
            block_id = nodes[0].node_id

            # Get display label
            label = self._get_block_label(nodes, user_display_name)

            # Render block content
            block_content = self._render_block_content(nodes, char_budget // len(blocks))

            # Format block with ID and label
            block_header = f"[msg:{block_id}] **{label}**:\n"
            entry = block_header + block_content + "\n"

            entry_chars = len(entry)
            if tokens_used * 4 + entry_chars > char_budget:
                # Over budget, truncate and note
                current_idx = blocks.index((role, nodes))
                remaining_blocks = len(blocks) - current_idx - 1
                if remaining_blocks > 0:
                    parts.append(f"[{remaining_blocks} earlier message blocks omitted]\n\n")
                break

            parts.append(entry)
            tokens_used += entry_chars // 4

        return ProjectionSection(
            section_type="conversation",
            source_id="messages",
            content="".join(parts),
            tokens_used=tokens_used,
            state=NodeState.DETAILS,
        )

    def _get_block_label(
        self,
        nodes: list[ContextNode],
        user_display_name: str,
    ) -> str:
        """Get the display label for a message block.

        Args:
            nodes: MessageNodes in this block
            user_display_name: Configured user display name

        Returns:
            Label like "Ace", "Agent", "Tool Call: grep", etc.
        """
        from activecontext.context.nodes import MessageNode

        first_node = nodes[0]
        if not isinstance(first_node, MessageNode):
            return "Unknown"

        # User messages use configured display name
        if first_node.actor == "user":
            return user_display_name

        # Check for agent content in block (may be mixed with tool calls)
        has_agent = any(
            isinstance(n, MessageNode) and n.actor == "agent"
            for n in nodes
        )

        # If block has agent prose, label as "Agent"
        if has_agent:
            return "Agent"

        # Use first node's display_label
        return first_node.display_label

    def _render_block_content(
        self,
        nodes: list[ContextNode],
        budget: int,
    ) -> str:
        """Render the content of a message block.

        Combines multiple messages (agent prose + tool calls/results)
        into a single block.

        Args:
            nodes: MessageNodes in this block
            budget: Character budget for this block

        Returns:
            Rendered block content
        """
        from activecontext.context.nodes import MessageNode

        parts: list[str] = []
        chars_used = 0

        for node in nodes:
            if not isinstance(node, MessageNode):
                continue

            # Different formatting based on message type
            if node.role == "tool_call":
                # Format tool call inline
                tool_name = node.tool_name or "unknown"
                if node.tool_args:
                    args_str = ", ".join(f'{k}="{v}"' for k, v in node.tool_args.items())
                    content = f"[Tool: {tool_name}] {args_str}\n"
                else:
                    content = f"[Tool: {tool_name}]\n"
            elif node.role == "tool_result":
                # Format tool result
                result_content = node.content
                if len(result_content) > budget // len(nodes):
                    result_content = result_content[:budget // len(nodes) - 20] + "..."
                content = f"[Result] {result_content}\n"
            else:
                # Regular message content
                content = node.content
                if len(content) > budget // len(nodes):
                    content = content[:budget // len(nodes) - 20] + "..."
                content = content + "\n"

            if chars_used + len(content) > budget:
                remaining = len(nodes) - nodes.index(node)
                if remaining > 1:
                    parts.append(f"... [{remaining} more items]\n")
                break

            parts.append(content)
            chars_used += len(content)

        return "".join(parts)

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
                    state=view.state if hasattr(view, "state") else NodeState.DETAILS,
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
                    state=group.state if hasattr(group, "state") else NodeState.SUMMARY,
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
