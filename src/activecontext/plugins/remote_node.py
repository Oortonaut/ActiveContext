"""RemoteNode -- proxy for remote CAP plugin nodes.

Represents a node running in a remote plugin server process.
Caches state locally, queues method calls, syncs on tick.
The rest of the system (Timeline, Graph, ProjectionEngine) treats
RemoteNode identically to local nodes.

Sync-on-tick model:
    1. Between ticks: reads from cache, writes queue as MethodCall
    2. On tick: if dirty or pending calls, send node/sync RPC
    3. RPC response updates all caches
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from activecontext.context.headers import TokenInfo
from activecontext.context.nodes import ContextNode
from activecontext.context.state import Expansion
from activecontext.plugins.protocol import (
    MethodCall,
    RenderSnapshot,
    TokenEstimate,
)
from activecontext.plugins.wire import (
    Methods,
    NodeSyncParams,
    PendingCall,
)

if TYPE_CHECKING:
    from activecontext.plugins.connection import PluginConnection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

# Rough chars-per-token ratio for fallback estimation
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# RemoteNode
# ---------------------------------------------------------------------------


@dataclass
class RemoteNode(ContextNode):
    """Proxy node for a remote CAP plugin server node.

    Behavior:
    - Property reads return from _cached_state
    - Method calls queue in _pending_calls
    - render_digest/content return from _cached_renders
    - get_token_breakdown returns cached token info
    - tick() syncs with remote: sends pending calls, receives updated state
    - to_dict/from_dict handle serialization for checkpointing

    The sync-on-tick model means:
    1. Between ticks: reads from cache, writes queue as MethodCall
    2. On tick: if dirty or pending calls, send node/sync RPC
    3. RPC response updates all caches
    """

    # The real remote type (e.g., "lint", "shell")
    _actual_node_type: str = field(default="remote")

    # Connection info
    _connection: PluginConnection | None = field(default=None, repr=False)
    _remote_id: str = ""  # ID in the remote process
    _server_name: str = ""  # For reconnection

    # Cached state from last sync
    _cached_state: dict[str, Any] = field(default_factory=dict)
    _cached_renders: RenderSnapshot = field(default_factory=RenderSnapshot)
    _cached_tokens: TokenEstimate = field(default_factory=TokenEstimate)
    _cached_digest: dict[str, Any] = field(default_factory=dict)

    # Pending operations
    _pending_calls: list[MethodCall] = field(default_factory=list)
    _dirty: bool = False  # Remote signaled change via push notification

    # Stale flag -- set when sync fails so the UI can indicate degraded state
    _stale: bool = False

    # Async sync result (populated by tick, consumed by callers)
    _last_sync_error: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Set node_type from _actual_node_type if provided."""
        # _actual_node_type stores the real type for the property.
        # If it was left as default "remote", keep it.
        # The remote_id defaults to our local node_id.
        if not self._remote_id:
            self._remote_id = self.node_id

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def node_type(self) -> str:
        """Return the actual remote node type, not 'remote'."""
        return self._actual_node_type

    @property
    def is_remote(self) -> bool:
        """This node proxies a remote plugin server node."""
        return True

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Rendering (composable sections from cache)
    # ------------------------------------------------------------------

    def render_digest(self) -> str:
        """Return cached digest summary, title, or type fallback."""
        if self._cached_digest:
            summary = self._cached_digest.get("summary", "")
            if summary:
                return str(summary)
        if self.title:
            return self.title
        return f"{self._actual_node_type} (remote)"

    def render_content(
        self,
        cwd: str = ".",
        text_buffers: dict[str, Any] | None = None,
    ) -> str:
        """Return merged cached content + detail sections."""
        content = self._cached_renders.content
        detail = self._cached_renders.detail
        if detail:
            return content + detail
        return content

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def get_token_breakdown(self) -> TokenInfo:
        """Return cached token info, with fallback estimation."""
        ct = self._cached_tokens
        if ct.title > 0 or ct.content > 0 or ct.detail > 0:
            return TokenInfo(
                title=ct.title,
                content=ct.content,
                detail=ct.detail,
            )
        # Fallback: estimate from cached render text
        return TokenInfo(
            title=_estimate_tokens(self._cached_renders.header),
            content=_estimate_tokens(self._cached_renders.content),
            detail=_estimate_tokens(self._cached_renders.detail),
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def GetDigest(self) -> dict[str, Any]:
        """Return cached digest or build a minimal one."""
        if self._cached_digest:
            return dict(self._cached_digest)
        return {
            "id": self.node_id,
            "type": self._actual_node_type,
            "remote": True,
            "server": self._server_name,
            "stale": self._stale,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def get_digest(self) -> dict[str, Any]:
        """NodePlugin protocol adapter."""
        return self.GetDigest()

    # ------------------------------------------------------------------
    # Method queuing
    # ------------------------------------------------------------------

    def queue_method_call(
        self,
        name: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Queue a method call for the next sync.

        Args:
            name: Method name (e.g., "set_content", "cancel").
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        call = MethodCall(
            method=name,
            args=tuple(args) if args else (),
            kwargs=kwargs or {},
        )
        self._pending_calls.append(call)
        # Queued calls mean we need a sync on next tick
        self._dirty = True

    def mark_dirty(self) -> None:
        """Called when server pushes a dirty notification.

        Sets the dirty flag so the next tick will sync.
        """
        self._dirty = True

    # ------------------------------------------------------------------
    # Lifecycle -- tick (sync with remote)
    # ------------------------------------------------------------------

    def Recompute(self) -> None:
        """Synchronous tick entry point.

        For RemoteNode, tick/Recompute cannot do async I/O directly.
        The actual sync is performed by tick_async(), which the session
        or plugin manager calls during the async tick phase.

        This method is a no-op; the async path handles everything.
        """
        pass

    async def tick_async(self, cwd: str = ".") -> None:
        """Async sync with the remote plugin server.

        This is the core sync method:
        1. If not dirty and no pending calls, return early
        2. Build sync request from pending calls + current state
        3. Send node/sync RPC via connection
        4. Update caches from SyncResult response
        5. Clear pending calls, reset dirty flag
        6. Process any notifications from the sync result
        7. Call notify_parents() if state changed

        Args:
            cwd: Working directory for render methods.
        """
        if not self._dirty and not self._pending_calls:
            return

        if self._connection is None:
            logger.warning(
                "RemoteNode %s: no connection, skipping sync (stale)",
                self.node_id,
            )
            self._stale = True
            return

        # Build pending calls for the wire format
        wire_calls = [
            PendingCall(
                method=call.method,
                args=list(call.args),
                kwargs=dict(call.kwargs),
            )
            for call in self._pending_calls
        ]

        params = NodeSyncParams(
            node_id=self._remote_id,
            calls=wire_calls,
            cwd=cwd,
        )

        old_version = self.version

        try:
            raw_result = await self._connection.send_request(Methods.NODE_SYNC, params)
            self._apply_sync_result(raw_result)
            self._pending_calls.clear()
            self._dirty = False
            self._stale = False
            self._last_sync_error = None

            # Notify parents if state changed
            if self.version != old_version:
                self.notify_parents(f"Remote node {self._actual_node_type} synced")

        except Exception as e:
            logger.error(
                "RemoteNode %s: sync failed: %s",
                self.node_id,
                e,
            )
            self._stale = True
            self._last_sync_error = str(e)
            # Do NOT clear pending calls on failure -- retry next tick

    def _apply_sync_result(self, raw: Any) -> None:
        """Update all caches from a sync result.

        Args:
            raw: Raw dict from the JSON-RPC response.
        """
        if not isinstance(raw, dict):
            logger.warning(
                "RemoteNode %s: unexpected sync result type: %s",
                self.node_id,
                type(raw).__name__,
            )
            return

        # Update cached state
        state = raw.get("state", {})
        if state:
            self._cached_state = dict(state)

        # Update cached renders
        renders = raw.get("renders", {})
        if renders:
            self._cached_renders = RenderSnapshot(
                header=renders.get("header", ""),
                content=renders.get("content", ""),
                detail=renders.get("detail", ""),
            )

        # Update cached tokens
        tokens = raw.get("tokens", {})
        if tokens:
            self._cached_tokens = TokenEstimate(
                title=tokens.get("title", 0),
                content=tokens.get("content", 0),
                detail=tokens.get("detail", 0),
            )

        # Update cached digest
        digest = raw.get("digest", {})
        if digest:
            self._cached_digest = dict(digest)

        # Process notifications
        notifications = raw.get("notifications", [])
        for notif_data in notifications:
            desc = notif_data.get("description", "")
            if desc:
                self.mark_changed(desc)

        # Bump version to signal change
        self.version += 1
        import time

        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Attribute access -- proxy to cached state
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Fall through to cached state for unknown attributes.

        This lets code like ``node.errors`` work by reading from the
        remote node's cached property values.  Only fires for attributes
        not found via normal lookup (i.e., not dataclass fields or
        methods).
        """
        # Guard against infinite recursion during init: _cached_state
        # itself may not exist yet.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            cached = object.__getattribute__(self, "_cached_state")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None
        if name in cached:
            return cached[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpointing.

        Includes remote identity and all cached data.
        Does NOT include _connection (transient).
        """
        base = super().to_dict()
        base.update(
            {
                "node_type": "remote",
                "_actual_node_type": self._actual_node_type,
                "_remote_id": self._remote_id,
                "_server_name": self._server_name,
                "_cached_state": dict(self._cached_state),
                "_cached_renders": {
                    "header": self._cached_renders.header,
                    "content": self._cached_renders.content,
                    "detail": self._cached_renders.detail,
                },
                "_cached_tokens": {
                    "title": self._cached_tokens.title,
                    "content": self._cached_tokens.content,
                    "detail": self._cached_tokens.detail,
                },
                "_cached_digest": dict(self._cached_digest),
            },
        )
        return base

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemoteNode:
        """Restore from checkpoint.

        Rebuilds cached state. Connection will be re-established
        when the plugin manager reconnects.
        """
        # Parse cached renders
        renders_data = data.get("_cached_renders", {})
        renders = RenderSnapshot(
            header=renders_data.get("header", ""),
            content=renders_data.get("content", ""),
            detail=renders_data.get("detail", ""),
        )

        # Parse cached tokens
        tokens_data = data.get("_cached_tokens", {})
        tokens = TokenEstimate(
            title=tokens_data.get("title", 0),
            content=tokens_data.get("content", 0),
            detail=tokens_data.get("detail", 0),
        )

        node = cls(
            node_id=data.get("node_id", ""),
            _actual_node_type=data.get("_actual_node_type", "remote"),
            _remote_id=data.get("_remote_id", ""),
            _server_name=data.get("_server_name", ""),
            _cached_state=data.get("_cached_state", {}),
            _cached_renders=renders,
            _cached_tokens=tokens,
            _cached_digest=data.get("_cached_digest", {}),
        )

        # Restore common ContextNode fields
        node.parent_ids = set(data.get("parent_ids", []))
        node.children_ids = set(data.get("children_ids", []))
        if data.get("expansion"):
            node.default_expansion = Expansion(data["expansion"])
        node.mode = data.get("mode", "paused")
        node.version = data.get("version", 0)
        node.created_at = data.get("created_at", 0.0)
        node.updated_at = data.get("updated_at", 0.0)
        node.originator = data.get("originator")
        node.title = data.get("title", "")
        node.content_id = data.get("content_id")
        node.display_sequence = data.get("display_sequence")

        return node

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def set_connection(self, connection: PluginConnection) -> None:
        """Set or update the connection for this remote node.

        Called by the plugin manager when (re)connecting.

        Args:
            connection: The active PluginConnection.
        """
        self._connection = connection
        self._stale = False

    def clear_connection(self) -> None:
        """Clear the connection (server disconnected).

        The node enters degraded state, returning stale cached data.
        """
        self._connection = None
        self._stale = True

    @property
    def has_connection(self) -> bool:
        """Whether this node has an active connection."""
        return self._connection is not None

    @property
    def is_stale(self) -> bool:
        """Whether cached data may be outdated."""
        return self._stale

    def __repr__(self) -> str:
        conn_status = "connected" if self._connection else "disconnected"
        stale = " STALE" if self._stale else ""
        pending = f" +{len(self._pending_calls)}q" if self._pending_calls else ""
        return (
            f"RemoteNode(id={self.node_id!r}, "
            f"type={self._actual_node_type!r}, "
            f"server={self._server_name!r}, "
            f"{conn_status}{stale}{pending})"
        )
