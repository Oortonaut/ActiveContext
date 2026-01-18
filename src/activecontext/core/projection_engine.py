"""Projection engine for building token-aware LLM context.

The ProjectionEngine transforms session state (conversation, views, groups)
into a single Projection that becomes the LLM's context after the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from activecontext.context.state import NodeState
from activecontext.core.tokens import MediaType, count_tokens
from activecontext.session.protocols import Projection, ProjectionSection

if TYPE_CHECKING:
    from activecontext.context.content import ContentRegistry
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import ContextNode
    from activecontext.context.view import ViewRegistry


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
        # Per-agent view support (split architecture)
        agent_id: str | None = None,
        view_registry: "ViewRegistry | None" = None,
        content_registry: "ContentRegistry | None" = None,
    ) -> Projection:
        """Build a projection from current session state.

        Args:
            context_graph: ContextGraph (DAG of nodes) - preferred
            context_objects: ViewHandle/GroupHandle instances - legacy fallback
            conversation: Message history
            cwd: Working directory for file access
            token_budget: Override total token budget
            show_message_actors: Whether to show message actors in conversation
            agent_id: Optional agent ID for per-agent view resolution
            view_registry: Optional ViewRegistry for per-agent visibility
            content_registry: Optional ContentRegistry for shared content

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
            graph_sections = self._render_graph(
                context_graph,
                tree_budget,
                cwd,
                agent_id=agent_id,
                view_registry=view_registry,
                content_registry=content_registry,
            )
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
        *,
        agent_id: str | None = None,
        view_registry: "ViewRegistry | None" = None,
        content_registry: "ContentRegistry | None" = None,
    ) -> list[ProjectionSection]:
        """Render visible nodes from the context graph.

        Visible nodes are:
        - Running nodes (mode="running")
        - Root nodes that are paused (explicitly included)

        When view_registry and agent_id are provided, per-agent visibility
        settings (hidden, state, tokens) are used instead of node defaults.

        Args:
            graph: The context graph
            budget: Token budget for all graph content
            cwd: Working directory for file access
            agent_id: Optional agent ID for per-agent view resolution
            view_registry: Optional ViewRegistry for per-agent visibility
            content_registry: Optional ContentRegistry for shared content

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
            # Check for per-agent view if available
            agent_view = None
            if view_registry and agent_id:
                agent_view = view_registry.get_by_node(node.node_id)
                if agent_view and agent_view.agent_id != agent_id:
                    agent_view = None  # Not this agent's view

            # Determine visibility settings (per-agent or node default)
            if agent_view:
                # Use agent-specific visibility
                if agent_view.hidden:
                    # Hidden content shows token placeholder
                    content = self._render_hidden_placeholder(node, content_registry)
                    tokens_used = 10  # Minimal tokens for placeholder
                    state = agent_view.state
                elif content_registry and node.content_id:
                    # Render via AgentView + ContentData
                    content_data = content_registry.get(node.content_id)
                    if content_data:
                        content = agent_view.render(
                            content_data, budget=agent_view.tokens
                        )
                        tokens_used = count_tokens(content, MediaType.TEXT)
                        state = agent_view.state
                    else:
                        # Content not found, render node normally
                        content = node.Render(tokens=agent_view.tokens, cwd=cwd)
                        media_type = getattr(node, "media_type", MediaType.TEXT)
                        tokens_used = count_tokens(content, media_type)
                        state = agent_view.state
                else:
                    # AgentView without ContentData - use node's Render
                    content = node.Render(tokens=agent_view.tokens, cwd=cwd)
                    media_type = getattr(node, "media_type", MediaType.TEXT)
                    tokens_used = count_tokens(content, media_type)
                    state = agent_view.state
            else:
                # Default path: use node's built-in settings
                if node.state == NodeState.HIDDEN:
                    continue

                content = node.Render(tokens=per_node_budget, cwd=cwd)
                media_type = getattr(node, "media_type", MediaType.TEXT)
                tokens_used = count_tokens(content, media_type)
                state = node.state

            sections.append(
                ProjectionSection(
                    section_type=node.node_type,
                    source_id=node.node_id,
                    content=content,
                    tokens_used=tokens_used,
                    state=state,
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

    def _render_hidden_placeholder(
        self,
        node: "ContextNode",
        content_registry: "ContentRegistry | None",
    ) -> str:
        """Render a placeholder for hidden content.

        Shows the node type and token count without revealing content.

        Args:
            node: The context node
            content_registry: Optional ContentRegistry for token info

        Returns:
            Placeholder string like "[file: 500 tokens]"
        """
        # Try to get token count from ContentData if available
        if content_registry and node.content_id:
            content_data = content_registry.get(node.content_id)
            if content_data:
                return f"[{content_data.content_type}: {content_data.token_count} tokens]"

        # Fall back to node info
        node_type = node.node_type
        tokens = node.tokens
        return f"[{node_type}: {tokens} tokens]"

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

            entry_tokens = count_tokens(entry, MediaType.TEXT)
            if tokens_used + entry_tokens > budget:
                # Summarize remaining messages
                remaining = len(conversation) - len(messages_to_include)
                if remaining > 0:
                    parts.append(f"[{remaining} earlier messages omitted]\n\n")
                break

            messages_to_include.insert(0, entry)  # Prepend to maintain order
            tokens_used += entry_tokens

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
        """Render conversation from MessageNodes with structural containment.

        Renders root-level items (messages and groups) in temporal order.
        Adjacent same-role messages are merged into blocks.
        Groups are rendered with their children indented inside.

        Args:
            graph: Context graph containing MessageNodes and GroupNodes
            budget: Token budget for conversation
            user_display_name: Display name for user messages (e.g., "Ace")

        Returns:
            ProjectionSection with rendered conversation, or None if no messages
        """
        from activecontext.context.nodes import GroupNode, MessageNode

        # Collect root-level conversation items in temporal order
        # Each item is (created_at, type, data) where:
        # - type="msg": data is MessageNode
        # - type="group": data is GroupNode
        items: list[tuple[float, str, object]] = []

        for node in graph:
            if isinstance(node, MessageNode):
                # Only add root-level messages (not inside groups)
                if not node.parent_ids:
                    items.append((node.created_at, "msg", node))
            elif isinstance(node, GroupNode):
                # Add root-level groups that have message children (tool use groups)
                if not node.parent_ids:
                    has_message_child = any(
                        isinstance(graph.get_node(cid), MessageNode)
                        for cid in node.children_ids
                    )
                    if has_message_child:
                        items.append((node.created_at, "group", node))

        if not items:
            return None

        # Sort by creation time
        items.sort(key=lambda x: x[0])

        # Group adjacent messages by effective role, but keep groups separate
        blocks: list[tuple[str, list[object]]] = []  # (block_type, items)

        for _, item_type, item in items:
            if item_type == "group":
                # Groups are always their own block
                blocks.append(("group", [item]))
            elif item_type == "msg":
                assert isinstance(item, MessageNode)
                role = item.effective_role  # "USER" or "ASSISTANT"
                # Try to merge with previous block if same role and previous was msg
                if blocks and blocks[-1][0] == role:
                    # Check that previous block is all messages (not a group)
                    prev_items = blocks[-1][1]
                    if all(isinstance(x, MessageNode) for x in prev_items):
                        blocks[-1][1].append(item)
                        continue
                # Start new block
                blocks.append((role, [item]))

        # Render blocks
        parts = ["## Conversation\n\n"]
        tokens_used = 50  # header overhead

        for block_type, block_items in blocks:
            if block_type == "group":
                # Render group with children
                group = block_items[0]
                assert isinstance(group, GroupNode)
                entry = self._render_group_entry(group, graph, user_display_name)
            else:
                # Render message block (may contain multiple merged messages)
                msg_nodes = [n for n in block_items if isinstance(n, MessageNode)]
                per_block_budget = budget // len(blocks) if blocks else budget
                entry = self._render_message_block(msg_nodes, user_display_name, per_block_budget)

            entry_tokens = count_tokens(entry, MediaType.TEXT)
            if tokens_used + entry_tokens > budget:
                break

            parts.append(entry)
            tokens_used += entry_tokens

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

    def _render_message_entry(
        self,
        node: ContextNode,
        user_display_name: str,
    ) -> str:
        """Render a single message node entry.

        Args:
            node: MessageNode to render
            user_display_name: Display name for user messages

        Returns:
            Formatted entry string with ID and label
        """
        from activecontext.context.nodes import MessageNode

        if not isinstance(node, MessageNode):
            return ""

        # Get display label
        label = user_display_name if node.actor == "user" else node.display_label

        # Render content based on role
        if node.role == "tool_call":
            content = self._render_tool_call_content(node)
        elif node.role == "tool_result":
            result_label = self._get_tool_result_label(node)
            content = f"[{result_label}] {node.content[:200]}..." if len(node.content) > 200 else f"[{result_label}] {node.content}"
        else:
            content = node.content

        return f"[msg:{node.node_id}] **{label}**:\n{content}\n\n"

    def _render_message_block(
        self,
        nodes: list[ContextNode],
        user_display_name: str,
        char_budget: int,
    ) -> str:
        """Render a block of merged same-role messages.

        Args:
            nodes: MessageNodes in this block (all same effective role)
            user_display_name: Display name for user messages
            char_budget: Character budget for this block

        Returns:
            Formatted block string with ID and label
        """
        from activecontext.context.nodes import MessageNode

        if not nodes:
            return ""

        # Get block ID from first node
        first_node = nodes[0]
        if not isinstance(first_node, MessageNode):
            return ""

        block_id = first_node.node_id

        # Get display label
        label = self._get_block_label(nodes, user_display_name)

        # Render block content
        content = self._render_block_content(nodes, char_budget)

        return f"[msg:{block_id}] **{label}**:\n{content}\n\n"

    def _render_group_entry(
        self,
        group: ContextNode,
        graph: ContextGraph,
        user_display_name: str,
    ) -> str:
        """Render a group with its children.

        Args:
            group: GroupNode to render
            graph: Context graph for looking up children
            user_display_name: Display name for user messages

        Returns:
            Formatted group entry with indented children
        """
        from activecontext.context.nodes import GroupNode, MessageNode, ViewNode

        if not isinstance(group, GroupNode):
            return ""

        parts = []

        # Group header
        summary_text = group.summary_prompt or "Group"
        parts.append(f"[group:{group.node_id}] {summary_text}\n")

        # Get children sorted by creation time
        children = []
        for child_id in group.children_ids:
            child = graph.get_node(child_id)
            if child:
                children.append((child.created_at, child))
        children.sort(key=lambda x: x[0])

        # Render children with indentation
        for _, child in children:
            if isinstance(child, MessageNode):
                if child.role == "tool_call":
                    content = self._render_tool_call_content(child)
                    parts.append(f"  [call] {content}\n")
                elif child.role == "tool_result":
                    label = self._get_tool_result_label(child)
                    truncated = child.content[:100] + "..." if len(child.content) > 100 else child.content
                    parts.append(f"  [{label}] {truncated}\n")
                else:
                    parts.append(f"  [msg] {child.content[:100]}\n")
            elif isinstance(child, ViewNode):
                parts.append(f"  [view:{child.node_id}] {child.path}\n")
            elif isinstance(child, GroupNode):
                # Nested group (rare, but possible)
                parts.append(f"  [group:{child.node_id}] {child.summary_prompt or 'Nested'}\n")

        # Group summary (if available)
        if group.cached_summary:
            parts.append(f"  Summary: {group.cached_summary}\n")

        parts.append("\n")
        return "".join(parts)

    def _render_tool_call_content(self, node: ContextNode) -> str:
        """Render tool call content showing tool name and args.

        Args:
            node: MessageNode with role=tool_call

        Returns:
            Formatted tool call string like 'view("main.py", tokens=2000)'
        """
        from activecontext.context.nodes import MessageNode

        if not isinstance(node, MessageNode):
            return ""

        tool_name = node.tool_name or "unknown"
        args = node.tool_args or {}

        if not args:
            return f"{tool_name}()"

        # Format args as key=value pairs
        arg_parts = []
        for k, v in args.items():
            if isinstance(v, str):
                arg_parts.append(f'{k}="{v}"')
            else:
                arg_parts.append(f"{k}={v}")

        return f"{tool_name}({', '.join(arg_parts)})"

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

        for idx, node in enumerate(nodes):
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
                # Format tool result with dynamic label
                result_label = self._get_tool_result_label(node)
                result_content = node.content
                if len(result_content) > budget // len(nodes):
                    result_content = result_content[:budget // len(nodes) - 20] + "..."
                content = f"[{result_label}] {result_content}\n"
            else:
                # Regular message content
                content = node.content
                if len(content) > budget // len(nodes):
                    content = content[:budget // len(nodes) - 20] + "..."
                content = content + "\n"

            if chars_used + len(content) > budget:
                remaining = len(nodes) - idx
                if remaining > 1:
                    parts.append(f"... [{remaining} more items]\n")
                break

            parts.append(content)
            chars_used += len(content)

        return "".join(parts)

    def _get_tool_result_label(self, node: ContextNode) -> str:
        """Get a dynamic label for a tool result.

        Uses tool_args to extract meaningful context like filenames.

        Args:
            node: A MessageNode with role="tool_result"

        Returns:
            Dynamic label like "main.py" or "grep: pattern" or "Result"
        """
        from activecontext.context.nodes import MessageNode

        if not isinstance(node, MessageNode):
            return "Result"

        tool_name = node.tool_name or ""
        args = node.tool_args or {}

        # File-based tools: show the filename
        if tool_name in ("view", "read", "read_file", "Read"):
            path = args.get("path") or args.get("file_path") or args.get("file")
            if path:
                # Extract just the filename from path
                import os
                return os.path.basename(str(path))

        # Search tools: show the pattern
        if tool_name in ("grep", "Grep", "search", "rg"):
            pattern = args.get("pattern") or args.get("query")
            if pattern:
                # Truncate long patterns
                pattern_str = str(pattern)
                if len(pattern_str) > 20:
                    pattern_str = pattern_str[:17] + "..."
                return f"grep: {pattern_str}"

        # Glob/find tools: show the pattern
        if tool_name in ("glob", "Glob", "find", "find_file"):
            pattern = args.get("pattern") or args.get("file_mask")
            if pattern:
                return f"glob: {pattern}"

        # Shell/bash tools: show the command
        if tool_name in ("shell", "bash", "Bash", "execute"):
            cmd = args.get("command") or args.get("cmd")
            if cmd:
                cmd_str = str(cmd)
                if len(cmd_str) > 25:
                    cmd_str = cmd_str[:22] + "..."
                return cmd_str

        # Default: use tool name if available
        if tool_name:
            return f"{tool_name} result"

        return "Result"

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

            # Use view's media_type if available
            media_type = getattr(view, "media_type", MediaType.TEXT)
            tokens_used = count_tokens(content, media_type)

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
                    tokens_used=count_tokens(content, MediaType.TEXT),
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
