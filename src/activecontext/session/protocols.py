"""Core protocols for the session layer.

These protocols define the contract between:
- Transport layer (ACP, Direct) and session management
- Session management and core runtime (AgentLoop, StatementLog, etc.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from activecontext.context.state import Expansion

if TYPE_CHECKING:
    pass

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class UpdateKind(Enum):
    """Types of session updates emitted during execution."""

    STATEMENT_PARSED = "statement_parsed"
    STATEMENT_EXECUTING = "statement_executing"
    STATEMENT_EXECUTED = "statement_executed"
    NODE_CHANGED = "node_changed"
    TICK_APPLIED = "tick_applied"
    PROJECTION_READY = "projection_ready"
    RESPONSE_CHUNK = "response_chunk"
    ERROR = "error"
    # Conversation delegation updates (progress bars - no good ACP equivalent)
    CONVERSATION_PROGRESS = "conversation_progress"  # Progress update


class ExecutionStatus(Enum):
    """Status of a statement execution."""

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class WaitMode(Enum):
    """Mode for wait conditions."""

    SINGLE = "single"  # Wait for a single node
    ALL = "all"        # Wait for all nodes
    ANY = "any"        # Wait for any node (first to complete)
    MESSAGE = "message"  # Wait for incoming message
    AGENT = "agent"    # Wait for agent to complete


class EventResponse(Enum):
    """Response type for event handlers."""

    WAKE = "wake"    # Wake agent immediately when event occurs
    QUEUE = "queue"  # Queue event for processing on next wake


@dataclass(slots=True)
class EventHandler:
    """Handler configuration for an event type.

    Built-in event names:
    - "message": incoming message from another agent
    - "agent_done": child agent completed (DONE/TERMINATED)
    - "tick": tick occurred

    Attributes:
        event_name: Name of the event (e.g., "message", "agent_done")
        response: How to respond (WAKE or QUEUE)
        prompt_template: Template for wake prompt, can use {event_data} placeholders
        once: If True, handler is removed after firing once
        target_id: Optional target ID (e.g., specific agent ID for agent_done)
    """

    event_name: str
    response: EventResponse
    prompt_template: str
    once: bool = False
    target_id: str | None = None


@dataclass(slots=True)
class QueuedEvent:
    """An event that was queued for later processing."""

    event_name: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=lambda: __import__("time").time())


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionUpdate:
    """Transport-agnostic update emitted during session operations.

    Both ACP and Direct transports consume these and transform them
    appropriately (JSON-RPC notifications vs async yields).
    """

    kind: UpdateKind
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class Statement:
    """A recorded Python statement in the timeline."""

    statement_id: str
    index: int
    source: str
    timestamp: float
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NamespaceTrace:
    """Changes to the Python namespace after execution."""

    added: dict[str, str] = field(default_factory=dict)  # name -> type repr
    changed: dict[str, str] = field(default_factory=dict)  # name -> change info
    deleted: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExecutionResult:
    """Result of executing a statement."""

    execution_id: str
    statement_id: str
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exception: dict[str, Any] | None = None
    state_trace: NamespaceTrace = field(default_factory=NamespaceTrace)
    duration_ms: float = 0.0


@dataclass(slots=True)
class WaitCondition:
    """Condition that blocks turn progression until satisfied.

    When a wait condition is active, the session enters WAITING_FOR_NODE state.
    Ticks continue (for time-based updates), but turn-based ticks are paused.
    The condition is checked each tick, and when satisfied (or timed out),
    the appropriate prompt is injected and execution resumes.

    Attributes:
        node_ids: Node IDs to watch for completion
        mode: SINGLE, ALL, ANY, MESSAGE, or AGENT
        wake_prompt: Prompt injected when condition is satisfied
        timeout: Timeout in seconds (None = no timeout)
        timeout_prompt: Prompt injected on timeout
        failure_prompt: Prompt injected if any watched node fails
        started_at: When the wait condition was created (for timeout tracking)
        cancel_others: For ANY mode, whether to cancel other nodes on first completion
        agent_id: For MESSAGE/AGENT modes, the agent ID to wait for
    """

    node_ids: list[str]
    mode: WaitMode
    wake_prompt: str
    timeout: float | None = None
    timeout_prompt: str | None = None
    failure_prompt: str | None = None
    started_at: float = field(default_factory=lambda: __import__("time").time())
    cancel_others: bool = False
    agent_id: str | None = None  # For MESSAGE/AGENT modes

    def is_timed_out(self) -> bool:
        """Check if the wait condition has timed out."""
        if self.timeout is None:
            return False
        import time
        return (time.time() - self.started_at) >= self.timeout

    def elapsed_seconds(self) -> float:
        """Get elapsed time since wait started."""
        import time
        return time.time() - self.started_at


@dataclass(slots=True)
class ProjectionSection:
    """A rendered section in the projection."""

    section_type: str  # "conversation", "view", "group"
    source_id: str
    content: str
    tokens_used: int
    expansion: Expansion = Expansion.ALL
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Projection:
    """LLM context - the single source of context after system prompt.

    Contains conversation history, view contents, and group summaries,
    all rendered. Agent manages budget via node visibility and line ranges.
    """

    sections: list[ProjectionSection] = field(default_factory=list)

    handles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def render(self) -> str:
        """Render full projection as string for LLM."""
        return "\n\n".join(s.content for s in self.sections if s.content)


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


@runtime_checkable
class TimelineProtocol(Protocol):
    """Protocol for statement timeline operations.

    A timeline is the canonical history of executed Python statements
    for a session. It wraps StatementLog and PythonExec.
    """

    @property
    def session_id(self) -> str:
        """Unique identifier for this timeline's session."""
        ...

    async def execute_statement(self, source: str) -> ExecutionResult:
        """Execute a Python statement and record it in the timeline.

        Args:
            source: Python source code to execute

        Returns:
            ExecutionResult with status, output, and namespace changes
        """
        ...

    async def replay_from(self, statement_index: int) -> AsyncIterator[ExecutionResult]:
        """Re-execute statements from a given index.

        Resets the namespace and replays statements from index to end.
        Yields ExecutionResult for each statement.

        Args:
            statement_index: 0-based index to start replay from
        """
        ...

    def get_statements(self) -> list[Statement]:
        """Get all statements in the timeline."""
        ...

    def get_namespace(self) -> dict[str, Any]:
        """Get current Python namespace snapshot."""
        ...

    def get_context_objects(self) -> dict[str, Any]:
        """Get all tracked context objects (ViewHandle, GroupHandle, etc.)."""
        ...


@runtime_checkable
class SessionProtocol(Protocol):
    """Protocol for session lifecycle and interaction.

    A session wraps a timeline and provides the high-level interface
    for prompts, ticks, and projections.
    """

    @property
    def session_id(self) -> str:
        """Unique session identifier."""
        ...

    @property
    def timeline(self) -> TimelineProtocol:
        """The underlying statement timeline."""
        ...

    @property
    def cwd(self) -> str:
        """Working directory for this session."""
        ...

    async def prompt(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process a user prompt through the agent loop.

        This is the main interaction method. It:
        1. Sends the prompt to the LLM
        2. Executes any Python statements the LLM generates
        3. Runs the tick phase
        4. Yields updates throughout

        Args:
            content: User prompt text

        Yields:
            SessionUpdate objects as processing progresses
        """
        ...

    async def tick(self) -> list[SessionUpdate]:
        """Manually trigger the tick phase.

        Runs all scheduled ticks (async apply, sync, periodic, groups).

        Returns:
            List of updates from tick processing
        """
        ...

    async def cancel(self) -> None:
        """Cancel the current operation."""
        ...

    def get_projection(self) -> Projection:
        """Build current LLM projection from session state."""
        ...


@runtime_checkable
class SessionManagerProtocol(Protocol):
    """Protocol for managing multiple sessions.

    The session manager is the top-level entry point for both
    ACP and Direct transports.
    """

    async def create_session(
        self,
        cwd: str,
        session_id: str | None = None,
    ) -> SessionProtocol:
        """Create a new session with its own timeline.

        Args:
            cwd: Working directory for the session
            session_id: Optional specific ID; generated if not provided

        Returns:
            New session instance
        """
        ...

    async def get_session(self, session_id: str) -> SessionProtocol | None:
        """Get an existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        ...

    async def load_session(
        self,
        session_id: str,
        cwd: str,
    ) -> SessionProtocol | None:
        """Load a previously persisted session.

        The actual persistence mechanism is pluggable.

        Args:
            session_id: Session identifier to load
            cwd: Working directory

        Returns:
            Loaded session if found, None otherwise
        """
        ...

    async def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        ...

    async def close_session(self, session_id: str) -> None:
        """Close and clean up a session.

        Args:
            session_id: Session to close
        """
        ...


# -----------------------------------------------------------------------------
# Transport Protocols
# -----------------------------------------------------------------------------


class UpdateSink(Protocol):
    """Protocol for receiving session updates (push model).

    Used by the core to push updates to the transport layer.
    """

    async def emit(self, update: SessionUpdate) -> None:
        """Emit an update to the sink."""
        ...


class UpdateCallback(Protocol):
    """Callback signature for update notifications."""

    def __call__(self, update: SessionUpdate) -> None:
        """Handle an update."""
        ...


@runtime_checkable
class TransportAdapter(Protocol):
    """Protocol for transport-layer adapters.

    Both ACP and Direct transports implement this to provide
    a uniform interface to the session layer.
    """

    async def handle_prompt(
        self,
        session_id: str,
        content: str,
        sink: UpdateSink,
    ) -> None:
        """Handle a prompt request, streaming updates to sink.

        Args:
            session_id: Target session
            content: Prompt content
            sink: Where to send updates
        """
        ...

    async def handle_cancel(self, session_id: str) -> None:
        """Handle cancellation request."""
        ...
