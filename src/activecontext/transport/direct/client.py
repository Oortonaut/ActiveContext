"""ActiveContext: Main entry point for direct transport.

Usage:
    from activecontext import ActiveContext
    from activecontext.core import LiteLLMProvider

    # With LLM
    llm = LiteLLMProvider("claude-sonnet-4-20250514")
    async with ActiveContext(llm=llm) as ctx:
        session = await ctx.create_session(cwd=".")
        async for update in session.prompt("Show me the main file"):
            print(update)

    # Without LLM (direct Python execution)
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd=".")
        await session.execute('v = view("main.py")')
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from activecontext.session.session_manager import SessionManager
from activecontext.transport.direct.async_session import AsyncSession

if TYPE_CHECKING:
    from activecontext.core.llm.provider import LLMProvider


class ActiveContext:
    """Main entry point for programmatic use of ActiveContext.

    Provides a clean async context manager interface for creating
    and managing sessions. This is the "direct transport" that
    bypasses JSON-RPC for maximum performance.

    Args:
        llm: Optional LLM provider for AI-powered prompts.
             If not provided, prompts execute Python directly.

    Usage:
        # With LLM (AI-powered)
        from activecontext.core import LiteLLMProvider
        llm = LiteLLMProvider("claude-sonnet-4-20250514")

        async with ActiveContext(llm=llm) as ctx:
            session = await ctx.create_session(cwd="/path/to/project")
            async for update in session.prompt("Show me main.py"):
                print(update)

        # Without LLM (direct execution)
        async with ActiveContext() as ctx:
            session = await ctx.create_session(cwd=".")
            result = await session.execute('v = view("main.py")')
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        """Initialize ActiveContext.

        Args:
            llm: Optional LLM provider for AI-powered prompts
        """
        self._manager: SessionManager | None = None
        self._llm = llm

    async def __aenter__(self) -> ActiveContext:
        """Enter the async context and initialize the session manager."""
        self._manager = SessionManager(default_llm=self._llm)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context and clean up all sessions."""
        if self._manager:
            for session_id in await self._manager.list_sessions():
                await self._manager.close_session(session_id)
            self._manager = None

    def _ensure_manager(self) -> SessionManager:
        """Ensure the manager is initialized."""
        if self._manager is None:
            raise RuntimeError(
                "ActiveContext must be used as an async context manager. "
                "Use: async with ActiveContext() as ctx: ..."
            )
        return self._manager

    def set_llm(self, llm: LLMProvider | None) -> None:
        """Set or update the default LLM provider.

        New sessions will use this provider unless overridden.
        """
        self._llm = llm
        if self._manager:
            self._manager.set_default_llm(llm)

    async def create_session(
        self,
        cwd: str,
        session_id: str | None = None,
        llm: LLMProvider | None = None,
    ) -> AsyncSession:
        """Create a new session with its own timeline.

        Args:
            cwd: Working directory for the session
            session_id: Optional specific ID; generated if not provided
            llm: Optional LLM provider; uses default if not provided

        Returns:
            AsyncSession wrapper for direct interaction
        """
        manager = self._ensure_manager()
        session = await manager.create_session(cwd=cwd, session_id=session_id, llm=llm)
        return AsyncSession(session)

    async def get_session(self, session_id: str) -> AsyncSession | None:
        """Get an existing session by ID.

        Args:
            session_id: Session identifier

        Returns:
            AsyncSession if found, None otherwise
        """
        manager = self._ensure_manager()
        session = await manager.get_session(session_id)
        if session:
            return AsyncSession(session)
        return None

    async def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        manager = self._ensure_manager()
        return await manager.list_sessions()

    async def close_session(self, session_id: str) -> None:
        """Close and clean up a session.

        Args:
            session_id: Session to close
        """
        manager = self._ensure_manager()
        await manager.close_session(session_id)
