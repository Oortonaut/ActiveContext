"""Base extension class with overridable forwarding methods."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from acp_debug.types.notifications import CancelNotification, SessionUpdate
from acp_debug.types.requests import (
    CreateTerminalRequest,
    InitializeRequest,
    KillTerminalRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PermissionRequest,
    PromptRequest,
    ReadFileRequest,
    ReleaseTerminalRequest,
    SetModelRequest,
    SetModeRequest,
    TerminalOutputRequest,
    WaitForExitRequest,
    WriteFileRequest,
)
from acp_debug.types.responses import (
    CreateTerminalResponse,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionResponse,
    PromptResponse,
    ReadFileResponse,
    SetModelResponse,
    SetModeResponse,
    TerminalOutputResponse,
    WaitForExitResponse,
)

if TYPE_CHECKING:
    from acp_debug.state.session import SessionState


T = TypeVar("T")
R = TypeVar("R")


class ACPExtension:
    """Base extension class - override methods you care about, others pass through.

    Usage:
        class MyHooks(ACPExtension):
            async def session_prompt(self, req, call):
                print(f"Prompt: {req.prompt}")
                response = await call(req)
                print(f"Response: {response.stop_reason}")
                return response
    """

    # Framework-maintained session state
    sessions: dict[str, SessionState]

    def __init__(self) -> None:
        """Initialize extension."""
        from acp_debug.state.session import SessionState

        self.sessions: dict[str, SessionState] = {}
        self._SessionState = SessionState

    # === Lifecycle Hooks ===

    def on_initialize(self) -> None:
        """Called at debugger startup, before any connections."""
        pass

    def on_session_bind(self, session_id: str) -> None:
        """Called when a session ID becomes known."""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._SessionState(session_id=session_id)

    def on_shutdown(self) -> None:
        """Called when debugger is shutting down."""
        pass

    # === Agent Methods (Client → Agent) ===

    async def initialize(
        self,
        req: InitializeRequest,
        call: Callable[[InitializeRequest], Awaitable[InitializeResponse]],
    ) -> InitializeResponse:
        """Handle initialize request."""
        return await call(req)

    async def session_new(
        self,
        req: NewSessionRequest,
        call: Callable[[NewSessionRequest], Awaitable[NewSessionResponse]],
    ) -> NewSessionResponse:
        """Handle session/new request."""
        response = await call(req)
        self.on_session_bind(response.session_id)
        return response

    async def session_load(
        self,
        req: LoadSessionRequest,
        call: Callable[[LoadSessionRequest], Awaitable[LoadSessionResponse]],
    ) -> LoadSessionResponse:
        """Handle session/load request."""
        self.on_session_bind(req.session_id)
        return await call(req)

    async def session_list(
        self,
        req: ListSessionsRequest,
        call: Callable[[ListSessionsRequest], Awaitable[ListSessionsResponse]],
    ) -> ListSessionsResponse:
        """Handle session/list request."""
        return await call(req)

    async def session_prompt(
        self,
        req: PromptRequest,
        call: Callable[[PromptRequest], Awaitable[PromptResponse]],
    ) -> PromptResponse:
        """Handle session/prompt request."""
        return await call(req)

    async def session_set_mode(
        self,
        req: SetModeRequest,
        call: Callable[[SetModeRequest], Awaitable[SetModeResponse]],
    ) -> SetModeResponse:
        """Handle session/set_mode request."""
        response = await call(req)
        if req.session_id in self.sessions:
            self.sessions[req.session_id].mode = response.mode_id
        return response

    async def session_set_model(
        self,
        req: SetModelRequest,
        call: Callable[[SetModelRequest], Awaitable[SetModelResponse]],
    ) -> SetModelResponse:
        """Handle session/set_model request."""
        response = await call(req)
        if req.session_id in self.sessions:
            self.sessions[req.session_id].model = response.model_id
        return response

    async def session_cancel(
        self,
        notif: CancelNotification,
        forward: Callable[[CancelNotification], Awaitable[None]],
    ) -> None:
        """Handle session/cancel notification."""
        await forward(notif)

    # === Client Methods (Agent → Client) ===

    async def request_permission(
        self,
        req: PermissionRequest,
        call: Callable[[PermissionRequest], Awaitable[PermissionResponse]],
    ) -> PermissionResponse:
        """Handle session/request_permission."""
        return await call(req)

    async def fs_read_text_file(
        self,
        req: ReadFileRequest,
        call: Callable[[ReadFileRequest], Awaitable[ReadFileResponse]],
    ) -> ReadFileResponse:
        """Handle fs/read_text_file."""
        return await call(req)

    async def fs_write_text_file(
        self,
        req: WriteFileRequest,
        call: Callable[[WriteFileRequest], Awaitable[None]],
    ) -> None:
        """Handle fs/write_text_file."""
        return await call(req)

    async def terminal_create(
        self,
        req: CreateTerminalRequest,
        call: Callable[[CreateTerminalRequest], Awaitable[CreateTerminalResponse]],
    ) -> CreateTerminalResponse:
        """Handle terminal/create."""
        return await call(req)

    async def terminal_output(
        self,
        req: TerminalOutputRequest,
        call: Callable[[TerminalOutputRequest], Awaitable[TerminalOutputResponse]],
    ) -> TerminalOutputResponse:
        """Handle terminal/output."""
        return await call(req)

    async def terminal_wait_for_exit(
        self,
        req: WaitForExitRequest,
        call: Callable[[WaitForExitRequest], Awaitable[WaitForExitResponse]],
    ) -> WaitForExitResponse:
        """Handle terminal/wait_for_exit."""
        return await call(req)

    async def terminal_kill(
        self,
        req: KillTerminalRequest,
        call: Callable[[KillTerminalRequest], Awaitable[None]],
    ) -> None:
        """Handle terminal/kill."""
        return await call(req)

    async def terminal_release(
        self,
        req: ReleaseTerminalRequest,
        call: Callable[[ReleaseTerminalRequest], Awaitable[None]],
    ) -> None:
        """Handle terminal/release."""
        return await call(req)

    # === Notifications (Agent → Client) ===

    async def session_update(
        self,
        update: SessionUpdate,
        emit: Callable[[SessionUpdate], Awaitable[None]],
    ) -> None:
        """Handle session/update notification."""
        await emit(update)
