"""Projection engine for building token-aware LLM context.

The ProjectionEngine transforms session state (context graph nodes)
into a single Projection that becomes the LLM's entire context.

The agent manipulates the render path by showing, hiding, expanding,
and collapsing nodes. All nodes are ticked regardless of visibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.view import NodeView
from activecontext.core.tokens import MediaType, count_tokens
from activecontext.session.protocols import Projection, ProjectionSection

if TYPE_CHECKING:
    from activecontext.context.content import ContentRegistry
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import ContextNode


@dataclass
class ProjectionConfig:
    """Configuration for projection building.

    Tree character set (always enabled) controls ASCII tree-drawing prefixes:
    - tree_detail: Vertical continuation for non-last ancestors (e.g., "| ")
    - tree_content: Content line marker (e.g., "|.")
    - tree_child: Branch prefix for non-last children (e.g., "+-")
    - tree_last_child: Branch prefix for last child (e.g., "\\-")
    """

    # Tree character set (always enabled)
    tree_detail: str = "| "  # Vertical continuation for non-last ancestors
    tree_content: str = "|:"  # Content line marker
    tree_child: str = "+-"  # Branch prefix for non-last children
    tree_last_child: str = "\\-"  # Branch prefix for last child


@dataclass
class RenderPath:
    """Path through the context graph for rendering.

    Captures which views to render and their relationships,
    similar to Checkpoint's edge structure. This allows:
    - Hierarchical rendering (parents before children)
    - Token usage collection per subtree
    - Group summarization of children

    Attributes:
        views: Ordered list of NodeViews to render (document order)
        edges: List of (child_id, parent_id) tuples for structure
        root_ids: Node IDs that are roots in this path (no parents in path)
        total_tokens: Sum of all root nodes' total_tokens
    """

    views: list[NodeView] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    root_ids: set[str] = field(default_factory=set)
    total_tokens: int = 0

    def __len__(self) -> int:
        """Return number of views in path."""
        return len(self.views)

    def __bool__(self) -> bool:
        """Return True if path has views."""
        return len(self.views) > 0


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
        context_graph: ContextGraph | None = None,  # Optional: returns empty projection when None
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
            text_buffers: Dict of buffer_id -> TextBuffer for markdown nodes
            content_registry: Optional ContentRegistry for shared content

        Returns:
            Complete Projection ready for LLM
        """
        if context_graph and len(context_graph) > 0:
            # Collect the render path (creates views on-demand)
            render_path: RenderPath = self._collect_render_path(context_graph)

            # TODO: move this somewhere good.
            #  # Apply ChoiceView selection filters
            #  from activecontext.context.view import ChoiceView
            #
            #  for view in self._views.values():
            #      if isinstance(view, ChoiceView):
            #          view.apply_selection(self._views)

            # Render the path
            sections: list[ProjectionSection] = self._render_path(
                render_path,
                text_buffers=text_buffers,
                content_registry=content_registry,
            )

            # Build handles dict from graph
            handles: dict[str, Any] = {node.node_id: node.GetDigest() for node in context_graph}
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
    ) -> RenderPath:
        """Collect the render path through the graph in document order.

        Visibility rules:
        - Hidden views (view.hidden=True) are excluded
        - COLLAPSED/SUMMARY nodes render themselves (not their children)
        - DETAILS nodes render children according to child_order

        Args:
            graph: The context graph

        Returns:
            RenderPath capturing nodes in document order with token totals
        """
        path: RenderPath = RenderPath()
        seen: set[str] = set()

        # Start from root context if set, otherwise collect all root nodes
        root: ContextNode | None = graph.get_root()
        if root is not None:
            path.total_tokens = self._collect_from_node(graph, root, path, seen)
        else:
            # Collect all root nodes (nodes with no parents)
            for node in graph.get_roots():
                path.total_tokens += self._collect_from_node(graph, node, path, seen)

        return path

    def _compute_tree_prefix(self, ancestor_is_last: list[bool]) -> str:
        """Compute tree prefix from ancestor last-sibling stack.

        Args:
            ancestor_is_last: List of booleans indicating if each ancestor
                              is the last sibling at that level. Empty for roots.

        Returns:
            Tree prefix string (e.g., "| +-" or "  \\-")
        """
        if not ancestor_is_last:
            return ""

        cfg = self.config
        blank = " " * len(cfg.tree_detail)

        parts: list[str] = []
        # All ancestors except the last one: show continuation or blank
        for is_last in ancestor_is_last[:-1]:
            parts.append(blank if is_last else cfg.tree_detail)
        # Last entry: show branch character
        parts.append(cfg.tree_last_child if ancestor_is_last[-1] else cfg.tree_child)
        return "".join(parts)

    def _collect_from_node(
        self,
        graph: ContextGraph,
        node: ContextNode,
        path: RenderPath,
        seen: set[str],
        depth: int = 0,
        ancestor_is_last: list[bool] | None = None,
    ) -> int:
        """Recursively collect nodes in document order, computing token totals.

        Always recurses into children regardless of parent state - each node's
        own state controls its rendering. This ensures complete token information
        is available for the agent to understand expansion costs.

        Args:
            graph: The context graph
            node: Current node to process
            path: RenderPath to append views to
            seen: Set of already-seen node IDs
            depth: Current traversal depth (0 for roots)
            ancestor_is_last: Stack of booleans tracking last-sibling status
                              for tree prefix computation

        Returns:
            Total tokens for this subtree (used for parent's children_tokens)
        """
        if ancestor_is_last is None:
            ancestor_is_last = []

        # Check if hidden via view
        if node.node_id in seen:
            return 0

        # Compute tree prefix for this node
        tree_prefix = self._compute_tree_prefix(ancestor_is_last)

        view: NodeView
        # Ensure view exists for this node (create on-demand with correct indent)
        if node.node_id not in self.views:
            view = self.views[node.node_id] = NodeView(
                node,
                indent=depth,
                expansion=node.default_expansion,
                hidden=node.default_hidden,
                tree_prefix=tree_prefix,
            )
        else:
            view = self.views[node.node_id]
            # Update tree_prefix even for existing views
            view.tree_prefix = tree_prefix

        seen.add(node.node_id)
        path.views.append(view)

        # Track root status
        if not node.parent_ids:
            path.root_ids.add(node.node_id)

        # Recurse into children first (post-order) to compute their totals
        children_total = 0
        child_order = getattr(node, "child_order", None)
        if child_order:
            child_ids = list(child_order)
            child_count = len(child_ids)
            for i, child_id in enumerate(child_ids):
                child = graph.get_node(child_id)
                if child:
                    path.edges.append((child_id, node.node_id))
                    # Build ancestor_is_last for child: append whether this child is last
                    child_ancestor_is_last = ancestor_is_last + [i == child_count - 1]
                    child_tokens = self._collect_from_node(
                        graph, child, path, seen, depth + 1, child_ancestor_is_last
                    )
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
        path: RenderPath,
        *,
        text_buffers: dict[str, Any] | None = None,
        content_registry: ContentRegistry | None = None,
    ) -> list[ProjectionSection]:
        """Render the collected path into projection sections.

        Args:
            path: The render path (list of NodeViews in document order)
            text_buffers: Dict of buffer_id -> TextBuffer for markdown nodes
            content_registry: Optional ContentRegistry for shared content

        Returns:
            List of ProjectionSections for the path
        """
        if not path:
            return []

        sections: list[ProjectionSection] = []

        for view in path.views:
            # Skip if hidden via view
            if view.hidden:
                continue

            section = self._render_node(
                view.node,
                view=view,
                text_buffers=text_buffers,
            )

            if section:
                sections.append(section)

        return sections

    def _render_node(
        self,
        node: ContextNode,
        *,
        view: NodeView,
        text_buffers: dict[str, Any] | None = None,
    ) -> ProjectionSection | None:
        """Render a single node.

        Args:
            node: The context node to render
            view: NodeView for expansion-aware rendering (required)
            text_buffers: Dict of buffer_id -> TextBuffer for markdown nodes

        Returns:
            ProjectionSection or None if node should be skipped
        """
        content: str = view.render(text_buffers=text_buffers)

        media_type: MediaType = getattr(node, "media_type", MediaType.TEXT)
        tokens_used: int = count_tokens(content, media_type)

        return ProjectionSection(
            section_type=node.node_type,
            source_id=node.node_id,
            content=content,
            indent=view.indent,
            tokens_used=tokens_used,
            expansion=view.expansion,
            metadata=node.GetDigest(),
            tree_prefix=view.tree_prefix,
        )
