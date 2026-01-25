"""View layer for context nodes.

NodeView provides view-specific state (hide, expand) while delegating
content operations to the underlying ContextNode.

Architecture:
- Content graph: DAG of ContextNodes for ticking and token rollup
- View graph: Flat list of NodeViews for rendering order

NodeView owns visibility state, ContextNode owns content data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from activecontext.context.state import Expansion

if TYPE_CHECKING:
    from activecontext.context.nodes import ContextNode


class NodeView:
    """View wrapper for ContextNode with visibility state.

    Separates view concerns (hide, expand) from content concerns (data, ticking).
    The DSL binds variables to NodeViews, allowing view-specific state
    while forwarding content operations to the underlying node.

    Attributes:
        _node: The underlying ContextNode (content)
        _hide: Whether this view is hidden from projection
        _expand: Expansion state for rendering (HEADER, CONTENT, INDEX, ALL)
        _tags: View-specific metadata (e.g., _hidden_expansion for unhide restoration)
    """

    __slots__ = ("_node", "_hide", "_expand", "_tags")

    _node: ContextNode
    _hide: bool
    _expand: Expansion
    _tags: dict[str, Any]

    def __init__(
        self,
        node: ContextNode,
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a view wrapping a node.

        Args:
            node: The ContextNode to wrap
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: uses node.expansion for migration)
        """
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_hide", hide)
        # Migration: use node.expansion as default if not specified
        object.__setattr__(self, "_expand", expand if expand is not None else node.expansion)
        object.__setattr__(self, "_tags", {})

    def node(self) -> ContextNode:
        """Return the underlying ContextNode."""
        return self._node

    # --- View State Properties ---

    @property
    def hide(self) -> bool:
        """Whether this view is hidden from projection."""
        return self._hide

    @hide.setter
    def hide(self, value: bool) -> None:
        """Set hide state."""
        object.__setattr__(self, "_hide", value)

    @property
    def expand(self) -> Expansion:
        """Expansion state for rendering."""
        return self._expand

    @expand.setter
    def expand(self, value: Expansion) -> None:
        """Set expansion state."""
        object.__setattr__(self, "_expand", value)

    @property
    def tags(self) -> dict[str, Any]:
        """View-specific metadata."""
        return self._tags

    # --- Backward Compatibility: expansion property ---

    @property
    def expansion(self) -> Expansion:
        """Alias for expand (backward compatibility)."""
        return self._expand

    @expansion.setter
    def expansion(self, value: Expansion) -> None:
        """Alias for expand (backward compatibility)."""
        object.__setattr__(self, "_expand", value)

    # --- Token Calculations ---

    @property
    def visible_tokens(self) -> int:
        """Tokens visible at current expand level.

        Returns:
            0 if hidden, otherwise tokens based on expand state.
        """
        if self._hide:
            return 0

        node = self._node
        if self._expand == Expansion.HEADER:
            return node.header_tokens
        elif self._expand == Expansion.CONTENT:
            return node.header_tokens + getattr(node, "summary_tokens", 0)
        else:  # DETAILS
            return node.header_tokens + getattr(node, "content_tokens", node.tokens)

    # --- Fluent API ---

    def SetHide(self, hide: bool) -> NodeView:
        """Set hide state (fluent API).

        Args:
            hide: Whether to hide this view
        """
        self.hide = hide
        return self

    def SetExpand(self, expand: Expansion) -> NodeView:
        """Set expansion state (fluent API).

        Args:
            expand: New expansion state (HEADER, CONTENT, INDEX, ALL)
        """
        self.expand = expand
        return self

    def SetExpansion(self, s: Expansion) -> NodeView:
        """Set expansion state (backward compatibility).

        Args:
            s: New expansion state
        """
        self.expand = s
        # Also update node for backward compatibility during migration
        self._node.expansion = s
        return self

    def SetTokens(self, n: int) -> NodeView:
        """Set token budget (fluent API)."""
        self._node.tokens = n
        return self

    def Run(self, freq: Any = None) -> NodeView:
        """Enable tick recomputation with given frequency."""
        self._node.Run(freq)
        return self

    def Pause(self) -> NodeView:
        """Disable tick recomputation."""
        self._node.Pause()
        return self

    def SetNotify(self, level: Any) -> NodeView:
        """Set notification level."""
        self._node.notification_level = level
        return self

    # --- Attribute Forwarding ---

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying node."""
        return getattr(self._node, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignment to the underlying node.

        Properties (hide, expand, expansion, tags) go through their setters.
        Slot attributes (_node, _hide, _expand, _tags) use object.__setattr__.
        Other attributes are forwarded to the underlying node.
        """
        # Slot attributes - use object.__setattr__
        if name in ("_node", "_hide", "_expand", "_tags"):
            object.__setattr__(self, name, value)
        # Property setters - let descriptor protocol handle it
        elif name in ("hide", "expand", "expansion"):
            # Get the property descriptor from the class and call its setter
            prop = type(self).__dict__.get(name)
            if prop is not None and hasattr(prop, "__set__"):
                prop.__set__(self, value)
            else:
                object.__setattr__(self, name, value)
        else:
            setattr(self._node, name, value)

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        return f"NodeView({self._node!r}, expand={self._expand.value}{hide_str})"

    def __eq__(self, other: Any) -> bool:
        """Compare views by their underlying node."""
        if isinstance(other, NodeView):
            return self._node is other._node
        if isinstance(other, type(self._node)):
            return self._node is other
        return False

    def __hash__(self) -> int:
        """Hash by underlying node ID."""
        return hash(self._node.node_id)


class ChoiceView(NodeView):
    """View that provides dropdown-like selection behavior for nodes with children.

    ChoiceView is a view-layer concern - any node can become a "choice" by wrapping
    it in ChoiceView. Selection controls which children are visible.

    Behavior by expansion mode:
    - HEADER/CONTENT/INDEX: No changes to children via apply_selection
    - INDEX: Use render_index() to get header lines for all children
    - ALL: Only selected child visible (with its own expand mode),
           or all children hidden if no selection

    Rendering helpers:
    - render_index(): Header line for each child (for INDEX mode)
    - render_brief(): "selected [A | B | C]" format

    Example:
        # Wrap any node in ChoiceView
        choice = ChoiceView(group_node, selected_id="child-b")

        # Select different option (fluent API)
        choice.select("option-2")

        # Get INDEX data (children headers)
        print(choice.render_index())

    Attributes:
        _selected_id: ID of the currently selected child (or None)
    """

    __slots__ = ("_selected_id",)

    _selected_id: str | None

    def __init__(
        self,
        node: ContextNode,
        selected_id: str | None = None,
        hide: bool = False,
        expand: Expansion | None = None,
    ) -> None:
        """Create a choice view wrapping a node.

        Args:
            node: The ContextNode to wrap
            selected_id: ID of the initially selected child (or None)
            hide: Whether the view is hidden (default False)
            expand: Expansion state (default: uses node.expansion)
        """
        super().__init__(node, hide=hide, expand=expand)
        object.__setattr__(self, "_selected_id", selected_id)

    @property
    def selected_id(self) -> str | None:
        """ID of the currently selected child."""
        return self._selected_id

    @selected_id.setter
    def selected_id(self, value: str | None) -> None:
        """Set the selected child ID."""
        object.__setattr__(self, "_selected_id", value)

    def select(self, child_id: str) -> ChoiceView:
        """Select a child by ID (fluent API).

        Args:
            child_id: ID of the child to select

        Returns:
            Self for method chaining
        """
        object.__setattr__(self, "_selected_id", child_id)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute assignment, including _selected_id slot."""
        if name == "_selected_id":
            object.__setattr__(self, name, value)
        elif name == "selected_id":
            # Property setter - use descriptor protocol
            prop = type(self).__dict__.get(name)
            if prop is not None and hasattr(prop, "__set__"):
                prop.__set__(self, value)
            else:
                object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def _get_child_ids(self) -> list[str]:
        """Get ordered list of child IDs from the node."""
        node = self._node
        child_order = getattr(node, "child_order", None)
        if child_order is not None:
            # child_order might be a LinkedChildOrder with to_list() or a regular list
            if hasattr(child_order, "to_list"):
                return child_order.to_list()
            else:
                return list(child_order)
        else:
            return list(getattr(node, "children_ids", set()))

    def apply_selection(self, views: dict[str, NodeView]) -> None:
        """Apply selection filtering to child views based on expansion mode.

        Behavior by expansion mode:
        - HEADER/CONTENT/INDEX: No changes to children (INDEX renders headers itself)
        - ALL: Only selected child visible (with its own expand mode),
               or all children hidden if no selection

        Args:
            views: Dict mapping node_id -> NodeView for all views
        """
        if self._expand != Expansion.ALL:
            return  # Only filter in ALL mode; INDEX renders headers itself

        child_ids = self._get_child_ids()

        # ALL mode: show only selected child, hide all if no selection
        for child_id in child_ids:
            if child_id in views:
                if self._selected_id is None:
                    views[child_id].hide = True
                else:
                    views[child_id].hide = (child_id != self._selected_id)
                # Selected child keeps its own expand mode

    def get_options(self) -> list[str]:
        """Get titles of all child options.

        Returns:
            List of child titles in child_order order
        """
        node = self._node
        graph = getattr(node, "_graph", None)
        if graph is None:
            return []

        # Get child IDs in order
        child_order = getattr(node, "child_order", None)
        if child_order is not None:
            if hasattr(child_order, "to_list"):
                child_ids = child_order.to_list()
            else:
                child_ids = list(child_order)
        else:
            child_ids = list(getattr(node, "children_ids", set()))

        # Get titles for each child
        titles = []
        for child_id in child_ids:
            child = graph.get_node(child_id)
            if child:
                title = getattr(child, "title", None) or child_id
                titles.append(title)
        return titles

    def render_index(self) -> str:
        """Render INDEX data: header line for each child option.

        Returns:
            Newline-separated headers of all children
        """
        node = self._node
        graph = getattr(node, "_graph", None)
        if graph is None:
            return ""

        child_ids = self._get_child_ids()
        lines = []
        for child_id in child_ids:
            child = graph.get_node(child_id)
            if child:
                # Get header line from child
                header = getattr(child, "render_header", None)
                if callable(header):
                    lines.append(header())
                else:
                    # Fallback to title or ID
                    title = getattr(child, "title", None) or child_id
                    lines.append(f"- {title}")
        return "\n".join(lines)

    def render_brief(self) -> str:
        """Render as 'selected [A | B | C]'.

        Returns:
            Brief string showing selected option and all choices
        """
        options = self.get_options()
        if not options:
            return "[No options]"

        # Get the selected child's title
        selected_title = None
        if self._selected_id:
            node = self._node
            graph = getattr(node, "_graph", None)
            if graph:
                selected_child = graph.get_node(self._selected_id)
                if selected_child:
                    selected_title = getattr(selected_child, "title", None) or self._selected_id

        if selected_title is None:
            selected_title = options[0] if options else "?"

        return f"{selected_title} [{' | '.join(options)}]"

    def __repr__(self) -> str:
        """Return string representation."""
        hide_str = ", hide=True" if self._hide else ""
        selected_str = f", selected={self._selected_id!r}" if self._selected_id else ""
        return f"ChoiceView({self._node!r}, expand={self._expand.value}{hide_str}{selected_str})"
