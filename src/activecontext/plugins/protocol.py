"""NodePlugin protocol — the contract all node types satisfy.

This module defines the structural interface for the Context Application
Protocol (CAP). Every node type (builtin, local plugin, or remote RPC
plugin) satisfies this protocol. The existing ContextNode base class is
compatible via its abstract methods and rendering interface.

Design principles:
- Nodes do NOT handle view information. Expansion and visibility live
  on NodeView; nodes produce content and expose read/write to the DSL.
- Render methods return composable sections, not cumulative strings.
  The projection engine composes them based on the view's expansion:
    HEADER  → render_header()
    CONTENT → render_header() + render_content()
    INDEX   → render_header() + render_content() + (children headers)
    ALL     → render_header() + render_content() + (children content)
- Token estimation is approximate — nodes estimate from their data,
  not by enumerating the full tree.
- tick() is the synchronous state materialization point. Async work
  happens internally; a dirty flag triggers state update on tick.
- The DSL uses Python positional + named argument syntax. Nodes expose
  public properties (no _ prefix) as DSL-readable, and methods as
  DSL-callable. The trace_all_fields decorator auto-traces assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from activecontext.context.headers import TokenInfo


# ---------------------------------------------------------------------------
# NodePlugin protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NodePlugin(Protocol):
    """Protocol that all context node types must satisfy.

    This is a structural protocol — any class that implements these
    methods is a valid node plugin, whether it subclasses ContextNode
    or not. In practice:

    - Builtin nodes: subclass ContextNode (which satisfies this protocol)
    - Local plugins: subclass ContextNode, registered via plugin descriptor
    - Remote plugins: RemoteNode proxy satisfies this protocol by caching
      state from a JSON-RPC connection to an external process

    Properties vs Methods:
    - Properties (node_type, node_id) are identity — stable after creation.
    - Render methods produce content sections for the projection engine.
    - Lifecycle methods (tick, notify_parents) participate in the agent loop.
    - Serialization methods (to_dict, from_dict) enable checkpoint/restore.
    """

    # --- Identity ---

    @property
    def node_type(self) -> str:
        """Node type identifier (e.g., "shell", "topic", "artifact").

        Used for registry lookup, serialization dispatch, and DSL
        constructor naming. Must be unique across all registered types.
        """
        ...

    @property
    def node_id(self) -> str:
        """Unique node instance identifier.

        Generated at creation time (typically 8-char UUID suffix).
        Stable for the lifetime of the node.
        """
        ...

    # --- Rendering (composable sections, no view concepts) ---
    #
    # Nodes produce content via render_digest() and render_content().
    # Header rendering (with token counts and expansion state) is handled
    # by NodeView.render_header(). The projection engine composes them
    # via NodeView.render() based on the view's Expansion state.
    #
    # Nodes never see Expansion — that's a view concept.

    def render_digest(self) -> str:
        """Render node metadata — the framework prepends the title line.

        Returns a short metadata string (e.g., display name, status).
        Used by the framework for compact representations.
        """
        ...

    def render_content(self, cwd: str = ".") -> str:
        """Render the content section — the actual content of this node.

        Shown at CONTENT expansion and above. Contains the primary
        information: summary text, output preview, message content, etc.

        Return empty string if the node has no content beyond its header
        (e.g., TopicNode has minimal content).
        """
        ...

    # --- Token estimation ---

    def get_token_breakdown(self) -> TokenInfo:
        """Estimate token counts for each rendering level.

        Returns a TokenInfo with:
        - title: tokens for the metadata/header line
        - content: tokens for render_content() output
        - index: tokens for children headers (INDEX mode)
        - detail: tokens for rendered children (ALL mode)
        - total: total recursive tokens (for groups)

        These are estimates, not exact counts. Nodes should estimate
        from their data (line count, content length) rather than
        rendering and counting.
        """
        ...

    # --- Metadata ---

    def get_digest(self) -> dict[str, Any]:
        """Return compact metadata for the handles dict.

        The projection engine includes digests in the Projection.handles
        dict, giving the LLM a quick reference for each node without
        rendering. Typical fields: id, type, status, key metrics.
        """
        ...

    # --- Lifecycle ---

    def tick(self) -> None:
        """Synchronous state materialization point.

        Called by the session at tick boundaries for nodes with
        mode="running". This is where async work becomes visible:

        1. Check if internal async state has changed (dirty flag)
        2. If dirty, update public fields from internal state
        3. Call _mark_changed() to generate traces and notifications
        4. Notifications propagate upward via notify_parents()

        For remote plugins (RemoteNode), tick() sends one JSON-RPC
        sync request that batches all pending method calls and returns
        the updated state snapshot.

        Default implementation (ContextNode.Recompute) does nothing —
        subclasses override when they have async or periodic work.
        """
        ...

    def notify_parents(self, description: str = "") -> None:
        """Notify parent nodes that this node changed.

        Propagates upward through the DAG. Parents may invalidate
        cached summaries (GroupNode), aggregate notifications
        (subscription points), or cascade further up.

        Args:
            description: Human-readable description of what changed.
        """
        ...

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize node state to a dictionary.

        Must include all fields needed to reconstruct the node via
        from_dict(). The base ContextNode implementation serializes
        common fields; subclasses call super().to_dict() and add
        their specific fields.

        The dict must include "node_type" for deserialization dispatch.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodePlugin:
        """Deserialize a node from a dictionary.

        The base ContextNode.from_dict() uses the node type registry
        to dispatch to the correct subclass. Plugin nodes must implement
        _from_dict() as a classmethod for the registry to call.
        """
        ...


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MethodCall:
    """A queued method call for batched RPC sync.

    Used by RemoteNode to queue DSL method calls (e.g., set_content,
    cancel) and send them in a single sync request on the next tick.
    """

    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SyncResult:
    """Result of a node/sync RPC call.

    Returned by the remote plugin server when the host sends a sync
    request. Contains the complete state snapshot, pre-rendered content
    at all three levels, token estimates, and any pending notifications.
    """

    state: dict[str, Any]
    """Full serialized node state (all public fields)."""

    renders: RenderSnapshot
    """Pre-rendered content at all three levels."""

    tokens: TokenEstimate
    """Token estimates for each rendering level."""

    digest: dict[str, Any]
    """Compact metadata (same as get_digest() output)."""

    notifications: list[NodeNotification] = field(default_factory=list)
    """Notifications generated since last sync."""


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    """Pre-rendered content at all three levels.

    Cached by RemoteNode after each sync. The projection engine reads
    from the cache instead of calling render methods over RPC.
    """

    header: str = ""
    content: str = ""
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Token estimates for each rendering level.

    Maps to TokenInfo but as a simple data transfer object for RPC.
    RemoteNode converts this to TokenInfo for the projection engine.
    """

    title: int = 0
    content: int = 0
    detail: int = 0


@dataclass(frozen=True, slots=True)
class NodeNotification:
    """A notification from a remote node to its parent chain.

    Generated by the remote plugin when internal state changes.
    The host delivers these by calling notify_parents() on the proxy.
    """

    description: str
    """Human-readable description of what changed."""

    level: str = "ignore"
    """Notification level: "ignore", "hold", or "wake"."""
