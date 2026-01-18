"""AsyncSession: Direct async interface to a session.

No serialization overhead - direct async function calls.
This is the "fast path" for CLI/TUI clients.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from activecontext.session.protocols import (
    ExecutionResult,
    Projection,
    SessionUpdate,
)

if TYPE_CHECKING:
    from activecontext.session.session_manager import Session
    from activecontext.session.timeline import Timeline


class AsyncSession:
    """Direct async interface to a session.

    Provides the same functionality as ACP sessions but without
    JSON-RPC serialization overhead. Use this for:
    - CLI clients
    - TUI clients
    - Programmatic library usage
    - Testing
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._update_callbacks: list[Callable[[SessionUpdate], None]] = []

    @property
    def session_id(self) -> str:
        """Unique session identifier."""
        return self._session.session_id

    @property
    def timeline(self) -> Timeline:
        """Direct access to the statement timeline."""
        return self._session.timeline

    @property
    def cwd(self) -> str:
        """Working directory for this session."""
        return self._session.cwd

    async def prompt(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Send a prompt and stream updates.

        This is the main interaction method. Yields SessionUpdate
        objects directly (no JSON-RPC).

        Args:
            content: Prompt text (or Python code)

        Yields:
            SessionUpdate objects as processing progresses
        """
        async for update in self._session.prompt(content):
            # Notify callbacks
            for callback in self._update_callbacks:
                callback(update)
            yield update

    async def execute(self, source: str) -> ExecutionResult:
        """Execute Python code in the session namespace.

        Lower-level than prompt() - just runs code without any
        LLM processing or prompt handling.

        Args:
            source: Python source code

        Returns:
            ExecutionResult with status, output, and namespace changes
        """
        return await self._session.timeline.execute_statement(source)

    async def tick(self) -> list[SessionUpdate]:
        """Manually trigger the tick phase.

        Runs all scheduled ticks for running context objects.

        Returns:
            List of updates from tick processing
        """
        return await self._session.tick()

    async def cancel(self) -> None:
        """Cancel the current operation."""
        await self._session.cancel()

    def get_namespace(self) -> dict[str, Any]:
        """Get current Python namespace snapshot.

        Returns user-defined variables (excludes builtins and
        injected functions).
        """
        return self._session.timeline.get_namespace()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all tracked context objects.

        Returns ViewHandle, GroupHandle instances created in
        this session.
        """
        return self._session.timeline.get_context_objects()

    def get_projection(self) -> Projection:
        """Get current LLM projection.

        Returns the structured context that would be sent to
        the LLM, including handles, summaries, and deltas.
        """
        return self._session.get_projection()

    def clear_message_history(self) -> None:
        """Clear the message history."""
        self._session.clear_message_history()

    def on_update(self, callback: Callable[[SessionUpdate], None]) -> Callable[[], None]:
        """Register a callback for all updates.

        The callback is invoked for every SessionUpdate emitted
        during prompt processing.

        Args:
            callback: Function to call with each update

        Returns:
            Unsubscribe function - call it to remove the callback
        """
        self._update_callbacks.append(callback)

        def unsubscribe() -> None:
            if callback in self._update_callbacks:
                self._update_callbacks.remove(callback)

        return unsubscribe

    async def replay_from(self, statement_index: int) -> AsyncIterator[ExecutionResult]:
        """Re-execute statements from a given index.

        Resets the namespace and replays statements from index to end.

        Args:
            statement_index: 0-based index to start replay from

        Yields:
            ExecutionResult for each replayed statement
        """
        async for result in self._session.timeline.replay_from(statement_index):
            yield result
