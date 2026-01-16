"""ACP Agent implementation for ActiveContext.

This module provides the ACP-compatible agent that can be used with
Rider, Zed, and other ACP-supporting editors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import acp
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    ModelInfo,
    PromptCapabilities,
    SessionCapabilities,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SetSessionModelResponse,
    SetSessionModeResponse,
    TextContentBlock,
)

from activecontext.core.llm import (
    LiteLLMProvider,
    get_available_models,
    get_default_model,
)
from activecontext.session.protocols import UpdateKind
from activecontext.session.session_manager import SessionManager

if TYPE_CHECKING:
    from acp.interfaces import Client

# Hardcoded session modes
SESSION_MODES = [
    SessionMode(id="normal", name="Normal", description="Standard mode"),
    SessionMode(id="plan", name="Plan", description="Plan before acting"),
    SessionMode(id="brave", name="Brave", description="Autonomous with fewer confirmations"),
]
DEFAULT_MODE_ID = "normal"


class ActiveContextAgent:
    """ACP Agent adapter for ActiveContext.

    This implements the ACP Agent protocol by delegating to
    SessionManager for actual session management.
    """

    def __init__(self) -> None:
        # Set up LLM provider based on available API keys
        default_model = get_default_model()
        default_llm: LiteLLMProvider | None = None
        if default_model:
            default_llm = LiteLLMProvider(default_model)

        # Note: type ignore due to Protocol/async-generator limitation in mypy
        self._manager = SessionManager(default_llm=default_llm)  # type: ignore[arg-type]
        self._conn: Client | None = None
        self._sessions_cwd: dict[str, str] = {}  # session_id -> cwd
        self._sessions_model: dict[str, str] = {}  # session_id -> model_id
        self._sessions_mode: dict[str, str] = {}  # session_id -> mode_id
        self._current_model_id = default_model

    def on_connect(self, conn: Client) -> None:
        """Called when a client connects."""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> acp.InitializeResponse:
        """Handle initialization request from client."""
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(
                name="activecontext",
                version="0.1.0",
            ),
            agent_capabilities=AgentCapabilities(
                load_session=False,  # Persistence not yet implemented
                prompt_capabilities=PromptCapabilities(
                    image=False,
                    audio=False,
                ),
                session_capabilities=SessionCapabilities(),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.NewSessionResponse:
        """Create a new session."""
        session = await self._manager.create_session(cwd=cwd)
        self._sessions_cwd[session.session_id] = cwd
        self._sessions_mode[session.session_id] = DEFAULT_MODE_ID

        # Build available models from environment
        available = get_available_models()
        models_state = None
        if available:
            current_model = self._current_model_id or available[0].model_id
            self._sessions_model[session.session_id] = current_model
            models_state = SessionModelState(
                available_models=[
                    ModelInfo(
                        model_id=m.model_id,
                        name=m.name,
                        description=m.description,
                    )
                    for m in available
                ],
                current_model_id=current_model,
            )

        # Build session modes
        modes_state = SessionModeState(
            available_modes=SESSION_MODES,
            current_mode_id=DEFAULT_MODE_ID,
        )

        return acp.NewSessionResponse(
            session_id=session.session_id,
            models=models_state,
            modes=modes_state,
        )

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        session_id: str = "",
        **kwargs: Any,
    ) -> acp.LoadSessionResponse | None:
        """Load a previously persisted session."""
        # Persistence not yet implemented
        return None

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> acp.schema.ListSessionsResponse:
        """List all active sessions."""
        session_ids = await self._manager.list_sessions()

        # Filter by cwd if provided
        if cwd:
            session_ids = [
                sid for sid in session_ids if self._sessions_cwd.get(sid) == cwd
            ]

        return acp.schema.ListSessionsResponse(
            sessions=[
                acp.schema.SessionInfo(
                    session_id=sid,
                    cwd=self._sessions_cwd.get(sid, ""),
                )
                for sid in session_ids
            ]
        )

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse | None:
        """Set session mode."""
        # Validate mode exists
        valid_modes = {m.id for m in SESSION_MODES}
        if mode_id not in valid_modes:
            raise acp.RequestError(
                code=-32602,
                message=f"Invalid mode: {mode_id}. Valid modes: {valid_modes}",
            )

        self._sessions_mode[session_id] = mode_id
        return SetSessionModeResponse()

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        """Set session model - switches the LLM provider for this session."""
        session = await self._manager.get_session(session_id)
        if not session:
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )

        # Create new LLM provider with the requested model
        new_llm = LiteLLMProvider(model_id)
        session.set_llm(new_llm)  # type: ignore[arg-type]
        self._sessions_model[session_id] = model_id

        return SetSessionModelResponse()

    async def authenticate(
        self,
        method_id: str,
        **kwargs: Any,
    ) -> acp.AuthenticateResponse | None:
        """Handle authentication (not implemented)."""
        return None

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        **kwargs: Any,
    ) -> acp.PromptResponse:
        """Handle a prompt request.

        This is the main interaction method. It:
        1. Gets the session
        2. Processes the prompt through the session
        3. Streams updates back via session_update
        4. Returns the final response
        """
        import sys
        import traceback

        session = await self._manager.get_session(session_id)
        if not session:
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )

        # Extract text from prompt blocks
        content = ""
        for block in prompt:
            if isinstance(block, TextContentBlock) or hasattr(block, "text"):
                content += block.text

        # Handle slash commands before LLM sees them
        if content.strip().startswith("/"):
            handled, response = await self._handle_slash_command(
                content.strip(), session_id
            )
            if handled:
                if self._conn and response:
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_message_text(response),
                    )
                return acp.PromptResponse(stop_reason="end_turn")

        # Process prompt and stream updates
        try:
            response_text = ""
            async for update in session.prompt(content):
                if self._conn:
                    await self._emit_update(session_id, update)

                # Collect response chunks
                if update.kind == UpdateKind.RESPONSE_CHUNK:
                    response_text += update.payload.get("text", "")
                elif update.kind == UpdateKind.STATEMENT_EXECUTED:
                    # Include execution output in response
                    stdout = update.payload.get("stdout", "")
                    if stdout:
                        response_text += f"\n{stdout}"
        except Exception as e:
            # Log the error but don't crash the agent
            from activecontext.__main__ import _log

            _log(f"ERROR in prompt: {e}")
            _log(traceback.format_exc())
            # Send error message to user
            if self._conn:
                await self._conn.session_update(
                    session_id,
                    acp.update_agent_message_text(f"\n\nError: {e}"),
                )

        return acp.PromptResponse(stop_reason="end_turn")

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.schema.ForkSessionResponse:
        """Fork a session (not implemented)."""
        raise acp.RequestError(
            code=-32601,
            message="Fork session not implemented",
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.schema.ResumeSessionResponse:
        """Resume a session."""
        session = await self._manager.get_session(session_id)
        if not session:
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )
        return acp.schema.ResumeSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel the current operation in a session."""
        session = await self._manager.get_session(session_id)
        if session:
            await session.cancel()

    async def ext_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle extension methods."""
        return {}

    async def ext_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Handle extension notifications."""
        pass

    async def _handle_slash_command(
        self, content: str, session_id: str
    ) -> tuple[bool, str]:
        """Handle slash commands before they reach the LLM.

        Returns:
            (handled, response) - handled=True if command was processed
        """
        import os
        import sys

        parts = content.split(maxsplit=1)
        command = parts[0].lower()
        # args = parts[1] if len(parts) > 1 else ""

        if command == "/exit":
            print("[activecontext] /exit command received, shutting down", file=sys.stderr)
            # Clean shutdown
            await self._manager.close_session(session_id)
            # Give time for response to be sent, then exit
            import asyncio
            asyncio.get_event_loop().call_later(0.5, lambda: os._exit(0))
            return True, "Goodbye!"

        elif command == "/help":
            return True, (
                "Available commands:\n"
                "  /exit - Shutdown the agent\n"
                "  /help - Show this help\n"
                "  /clear - Clear conversation history\n"
                "  /context - Show current context objects"
            )

        elif command == "/clear":
            session = await self._manager.get_session(session_id)
            if session:
                session.clear_conversation()
            return True, "Conversation history cleared."

        elif command == "/context":
            session = await self._manager.get_session(session_id)
            if session:
                objects = session.get_context_objects()
                if objects:
                    lines = ["Current context objects:"]
                    for obj_id, obj in objects.items():
                        digest = obj.GetDigest() if hasattr(obj, "GetDigest") else {}
                        obj_type = digest.get("type", "unknown")
                        path = digest.get("path", "")
                        lines.append(f"  {obj_id}: {obj_type} {path}")
                    return True, "\n".join(lines)
                else:
                    return True, "No context objects."
            return True, "Session not found."

        # Unknown command - let it pass through to LLM
        return False, ""

    async def _emit_update(self, session_id: str, update: Any) -> None:
        """Convert and emit a SessionUpdate as an ACP notification."""
        if not self._conn:
            return

        match update.kind:
            case UpdateKind.STATEMENT_EXECUTING:
                # Emit as tool call start
                await self._conn.session_update(
                    session_id,
                    acp.start_tool_call(
                        tool_call_id=update.payload.get("source", "")[:20],
                        title="python_exec",
                        status="in_progress",
                    ),
                )

            case UpdateKind.STATEMENT_EXECUTED:
                status = update.payload.get("status", "ok")
                stdout = update.payload.get("stdout", "")

                # Emit tool call completion
                await self._conn.session_update(
                    session_id,
                    acp.update_tool_call(
                        tool_call_id=update.payload.get("statement_id", "")[:20],
                        status="completed" if status == "ok" else "failed",
                    ),
                )

                # Emit output as message if any
                if stdout:
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_message_text(stdout),
                    )

            case UpdateKind.RESPONSE_CHUNK:
                text = update.payload.get("text", "")
                if text:
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_message_text(text),
                    )

            case UpdateKind.PROJECTION_READY:
                # Could emit as agent thought
                handles = update.payload.get("handles", {})
                if handles:
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_thought_text(
                            f"Context: {len(handles)} handles"
                        ),
                    )


def create_agent() -> ActiveContextAgent:
    """Create a new ActiveContext ACP agent."""
    return ActiveContextAgent()
