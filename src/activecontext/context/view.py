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
