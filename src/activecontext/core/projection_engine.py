"""Projection engine for building token-aware LLM context.

The ProjectionEngine transforms session state (context graph nodes)
into a single Projection that becomes the LLM's entire context.

The agent manipulates the render path by showing, hiding, expanding,
and collapsing nodes. All nodes are ticked regardless of visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.core.tokens import MediaType, count_tokens
from activecontext.session.protocols import Projection, ProjectionSection

if TYPE_CHECKING:
    from activecontext.context.content import ContentRegistry
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import ContextNode
    from activecontext.context.view import NodeView


@dataclass
class ProjectionConfig:
    """Configuration for projection building."""

    pass  # Budget removed - agent manages via node visibility and line ranges


@dataclass
class RenderPath:
    """Path through the context graph for rendering.

    Captures which nodes to render and their relationships,
    similar to Checkpoint's edge structure. This allows:
    - Hierarchical rendering (parents before children)
    - Budget allocation per subtree
    - Group summarization of children

    Attributes:
        node_ids: Ordered list of node IDs to render
        edges: List of (child_id, parent_id) tuples for structure
        root_ids: Node IDs that are roots in this path (no parents in path)
        total_tokens: Sum of all root nodes' total_tokens (complete context size)
    """

    node_ids: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    root_ids: set[str] = field(default_factory=set)
    total_tokens: int = 0

    def __len__(self) -> int:
        """Return number of nodes in path."""
        return len(self.node_ids)

    def __bool__(self) -> bool:
        """Return True if path has nodes."""
        return len(self.node_ids) > 0


class ProjectionEngine:
    """Builds token-aware projections from the context graph.

    The projection engine is responsible for:
    1. Collecting the render path (visible nodes and their structure)
    2. Allocating token budget across the path
    3. Rendering nodes at appropriate LOD levels based on state
    4. Assembling the final projection

    The agent controls what appears in the projection by manipulating
    node visibility (show/hide/expand/collapse). All nodes tick regardless
    of whether they appear in the rendered projection.
    """

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        if config:
            self.config = config
        else:
            # Try to load from app config
            self.config = self._config_from_app_config()

        # View graph (node_id -> NodeView) for rendering
        self._views: dict[str, NodeView] = {}

    def _config_from_app_config(self) -> ProjectionConfig:
        """Build ProjectionConfig from app config or defaults."""
        return ProjectionConfig()

    @property
    def views(self) -> dict[str, NodeView]:
        """Get the view graph (node_id -> NodeView) for rendering."""
        return self._views

    def build(
        self,
        *,
        context_graph: ContextGraph | None = None,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
        content_registry: ContentRegistry | None = None,
    ) -> Projection:
        """Build a projection from current session state.

        The projection renders visible nodes from the context graph. Visibility
        is controlled by NodeView.hidden. Expansion is controlled by NodeView.expansion.
        The agent manipulates the path by showing, hiding, expanding, and
        collapsing nodes. All nodes are ticked regardless of visibility.

        Args:
            context_graph: ContextGraph (DAG of nodes)
            cwd: Working directory for file access
            text_buffers: Dict of buffer_id -> TextBuffer for markdown nodes
            content_registry: Optional ContentRegistry for shared content

        Returns:
            Complete Projection ready for LLM
        """
        if context_graph and len(context_graph) > 0:
            # Collect the render path (creates views on-demand)
            render_path = self._collect_render_path(context_graph, self._views)

            # Apply ChoiceView selection filters
            from activecontext.context.view import ChoiceView

            for view in self._views.values():
                if isinstance(view, ChoiceView):
                    view.apply_selection(self._views)

            # Render the path
            sections = self._render_path(
                context_graph,
                render_path,
                cwd,
                text_buffers=text_buffers,
                views=self._views,
                content_registry=content_registry,
            )

            # Build handles dict from graph
            handles = {node.node_id: node.GetDigest() for node in context_graph}
        else:
            sections = []
            handles = {}

        return Projection(
            sections=sections,
            handles=handles,
        )

    def _collect_render_path(
        self,
        graph: ContextGraph,
        views: dict[str, NodeView] | None = None,
    ) -> RenderPath:
        """Collect the render path through the graph in document order.

        Visibility rules:
        - Hidden views (view.hidden=True) are excluded
        - COLLAPSED/SUMMARY nodes render themselves (not their children)
        - DETAILS nodes render children according to child_order

        Args:
            graph: The context graph
            views: Optional dict mapping node_id -> NodeView for visibility

        Returns:
            RenderPath capturing nodes in document order with token totals
        """
        path = RenderPath()
        seen: set[str] = set()

        # Start from root context if set, otherwise collect all root nodes
        root = graph.get_root()
        if root is not None:
            path.total_tokens = self._collect_from_node(graph, root, path, seen, views)
        else:
            # Collect all root nodes (nodes with no parents)
            for node in graph.get_roots():
                path.total_tokens += self._collect_from_node(graph, node, path, seen, views)

        return path

    def _collect_from_node(
        self,
        graph: ContextGraph,
        node: ContextNode,
        path: RenderPath,
        seen: set[str],
        views: dict[str, NodeView] | None = None,
    ) -> int:
        """Recursively collect nodes in document order, computing token totals.

        Always recurses into children regardless of parent state - each node's
        own state controls its rendering. This ensures complete token information
        is available for the agent to understand expansion costs.

        Args:
            graph: The context graph
            node: Current node to process
            path: RenderPath to append to
            seen: Set of already-seen node IDs
            views: Optional dict mapping node_id -> NodeView for visibility

        Returns:
            Total tokens for this subtree (used for parent's children_tokens)
        """
        # Check if hidden via view
        if node.node_id in seen:
            return 0

        # Ensure view exists for this node (create on-demand)
        if views is not None and node.node_id not in views:
            from activecontext.context.view import NodeView
            views[node.node_id] = NodeView(node, expansion=node.default_expansion)

        if views is not None:
            view = views.get(node.node_id)
            if view is not None and view.hidden:
                return 0

        seen.add(node.node_id)
        path.node_ids.append(node.node_id)

        # Track root status
        if not node.parent_ids:
            path.root_ids.add(node.node_id)

        # Recurse into children first (post-order) to compute their totals
        children_total = 0
        child_order = getattr(node, "child_order", None)
        if child_order:
            for child_id in child_order:
                child = graph.get_node(child_id)
                if child:
                    path.edges.append((child_id, node.node_id))
                    child_tokens = self._collect_from_node(graph, child, path, seen, views)
                    if isinstance(child_tokens, int):
                        children_total += child_tokens

        # Cache children tokens on this node (guard for Mock objects)
        if hasattr(node, "_cached_children_tokens"):
            node._cached_children_tokens = children_total

        # Return total for this subtree (guard for Mock objects)
        total = getattr(node, "total_tokens", 0)
        if isinstance(total, int):
            return total
        # Fallback for Mock objects: use tokens property
        tokens = getattr(node, "tokens", 0)
        return tokens if isinstance(tokens, int) else 0

    def _render_path(
        self,
        graph: ContextGraph,
        path: RenderPath,
        cwd: str,
        *,
        text_buffers: dict[str, Any] | None = None,
        views: dict[str, NodeView] | None = None,
        content_registry: ContentRegistry | None = None,
    ) -> list[ProjectionSection]:
        """Render the collected path into projection sections.

        Args:
            graph: The context graph (for node lookup)
            path: The render path to render
            cwd: Working directory for file access
            text_buffers: Dict of buffer_id -> TextBuffer for markdown nodes
            views: Dict mapping node_id -> NodeView for visibility/expansion
            content_registry: Optional ContentRegistry for shared content

        Returns:
            List of ProjectionSections for the path
        """
        if not path:
            return []

        sections: list[ProjectionSection] = []

        for node_id in path.node_ids:
            node = graph.get_node(node_id)
            if node is None:
                continue

            # Get view for this node (if available)
            view = views.get(node_id) if views else None

            # Skip if hidden via view
            if view is not None and view.hidden:
                continue

            section = self._render_node(
                node,
                cwd,
                view=view,
                text_buffers=text_buffers,
            )

            if section:
                sections.append(section)

        return sections

    def _render_node(
        self,
        node: ContextNode,
        cwd: str,
        *,
        view: NodeView | None = None,
        text_buffers: dict[str, Any] | None = None,
    ) -> ProjectionSection | None:
        """Render a single node.

        Args:
            node: The context node to render
            cwd: Working directory for file access
            view: Optional NodeView for expansion-aware rendering
            text_buffers: Dict of buffer_id -> TextBuffer for markdown nodes

        Returns:
            ProjectionSection or None if node should be skipped
        """
        if view is not None:
            content = view.render(cwd=cwd, text_buffers=text_buffers)
            effective_expand = view.expansion
        else:
            content = node.render_content(cwd=cwd, text_buffers=text_buffers)
            effective_expand = node.default_expansion

        media_type = getattr(node, "media_type", MediaType.TEXT)
        tokens_used = count_tokens(content, media_type)

        return ProjectionSection(
            section_type=node.node_type,
            source_id=node.node_id,
            content=content,
            tokens_used=tokens_used,
            expansion=effective_expand,
            metadata=node.GetDigest(),
        )
