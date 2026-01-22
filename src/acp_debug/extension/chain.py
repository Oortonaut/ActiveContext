"""Extension chain composition."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from acp_debug.extension.base import ACPExtension
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

T = TypeVar("T")
R = TypeVar("R")


class ExtensionChain:
    """Composes multiple extensions into a processing chain.

    Each extension's methods wrap the next extension in the chain,
    with the final extension calling the actual transport.
    """

    def __init__(self, extensions: list[ACPExtension]) -> None:
        """Initialize chain with extensions."""
        self.extensions = extensions

    def initialize_all(self) -> None:
        """Call on_initialize() on all extensions."""
        for ext in self.extensions:
            ext.on_initialize()

    def shutdown_all(self) -> None:
        """Call on_shutdown() on all extensions."""
        for ext in self.extensions:
            try:
                ext.on_shutdown()
            except Exception:
                pass

    def _build_chain(
        self,
        method_name: str,
        final: Callable[[T], Awaitable[R]],
    ) -> Callable[[T], Awaitable[R]]:
        """Build a chain of method calls through all extensions.

        The chain is built in reverse order so that the first extension
        in the list is the outermost wrapper.
        """
        chain = final

        # Build chain in reverse order
        for ext in reversed(self.extensions):
            method = getattr(ext, method_name)
            # Capture current chain and method in closure
            chain = self._wrap_method(method, chain)

        return chain

    def _wrap_method(
        self,
        method: Callable[[T, Callable[[T], Awaitable[R]]], Awaitable[R]],
        next_call: Callable[[T], Awaitable[R]],
    ) -> Callable[[T], Awaitable[R]]:
        """Wrap a method to inject the next call."""

        async def wrapper(req: T) -> R:
            return await method(req, next_call)

        return wrapper

    # === Agent Methods (Client → Agent) ===

    async def initialize(
        self,
        req: InitializeRequest,
        final: Callable[[InitializeRequest], Awaitable[InitializeResponse]],
    ) -> InitializeResponse:
        """Process initialize through chain."""
        chain = self._build_chain("initialize", final)
        return await chain(req)

    async def session_new(
        self,
        req: NewSessionRequest,
        final: Callable[[NewSessionRequest], Awaitable[NewSessionResponse]],
    ) -> NewSessionResponse:
        """Process session/new through chain."""
        chain = self._build_chain("session_new", final)
        return await chain(req)

    async def session_load(
        self,
        req: LoadSessionRequest,
        final: Callable[[LoadSessionRequest], Awaitable[LoadSessionResponse]],
    ) -> LoadSessionResponse:
        """Process session/load through chain."""
        chain = self._build_chain("session_load", final)
        return await chain(req)

    async def session_list(
        self,
        req: ListSessionsRequest,
        final: Callable[[ListSessionsRequest], Awaitable[ListSessionsResponse]],
    ) -> ListSessionsResponse:
        """Process session/list through chain."""
        chain = self._build_chain("session_list", final)
        return await chain(req)

    async def session_prompt(
        self,
        req: PromptRequest,
        final: Callable[[PromptRequest], Awaitable[PromptResponse]],
    ) -> PromptResponse:
        """Process session/prompt through chain."""
        chain = self._build_chain("session_prompt", final)
        return await chain(req)

    async def session_set_mode(
        self,
        req: SetModeRequest,
        final: Callable[[SetModeRequest], Awaitable[SetModeResponse]],
    ) -> SetModeResponse:
        """Process session/set_mode through chain."""
        chain = self._build_chain("session_set_mode", final)
        return await chain(req)

    async def session_set_model(
        self,
        req: SetModelRequest,
        final: Callable[[SetModelRequest], Awaitable[SetModelResponse]],
    ) -> SetModelResponse:
        """Process session/set_model through chain."""
        chain = self._build_chain("session_set_model", final)
        return await chain(req)

    async def session_cancel(
        self,
        notif: CancelNotification,
        final: Callable[[CancelNotification], Awaitable[None]],
    ) -> None:
        """Process session/cancel through chain."""
        chain = self._build_chain("session_cancel", final)
        await chain(notif)

    # === Client Methods (Agent → Client) ===

    async def request_permission(
        self,
        req: PermissionRequest,
        final: Callable[[PermissionRequest], Awaitable[PermissionResponse]],
    ) -> PermissionResponse:
        """Process request_permission through chain."""
        chain = self._build_chain("request_permission", final)
        return await chain(req)

    async def fs_read_text_file(
        self,
        req: ReadFileRequest,
        final: Callable[[ReadFileRequest], Awaitable[ReadFileResponse]],
    ) -> ReadFileResponse:
        """Process fs/read_text_file through chain."""
        chain = self._build_chain("fs_read_text_file", final)
        return await chain(req)

    async def fs_write_text_file(
        self,
        req: WriteFileRequest,
        final: Callable[[WriteFileRequest], Awaitable[None]],
    ) -> None:
        """Process fs/write_text_file through chain."""
        chain = self._build_chain("fs_write_text_file", final)
        await chain(req)

    async def terminal_create(
        self,
        req: CreateTerminalRequest,
        final: Callable[[CreateTerminalRequest], Awaitable[CreateTerminalResponse]],
    ) -> CreateTerminalResponse:
        """Process terminal/create through chain."""
        chain = self._build_chain("terminal_create", final)
        return await chain(req)

    async def terminal_output(
        self,
        req: TerminalOutputRequest,
        final: Callable[[TerminalOutputRequest], Awaitable[TerminalOutputResponse]],
    ) -> TerminalOutputResponse:
        """Process terminal/output through chain."""
        chain = self._build_chain("terminal_output", final)
        return await chain(req)

    async def terminal_wait_for_exit(
        self,
        req: WaitForExitRequest,
        final: Callable[[WaitForExitRequest], Awaitable[WaitForExitResponse]],
    ) -> WaitForExitResponse:
        """Process terminal/wait_for_exit through chain."""
        chain = self._build_chain("terminal_wait_for_exit", final)
        return await chain(req)

    async def terminal_kill(
        self,
        req: KillTerminalRequest,
        final: Callable[[KillTerminalRequest], Awaitable[None]],
    ) -> None:
        """Process terminal/kill through chain."""
        chain = self._build_chain("terminal_kill", final)
        await chain(req)

    async def terminal_release(
        self,
        req: ReleaseTerminalRequest,
        final: Callable[[ReleaseTerminalRequest], Awaitable[None]],
    ) -> None:
        """Process terminal/release through chain."""
        chain = self._build_chain("terminal_release", final)
        await chain(req)

    # === Notifications ===

    async def session_update(
        self,
        update: SessionUpdate,
        final: Callable[[SessionUpdate], Awaitable[None]],
    ) -> None:
        """Process session/update through chain."""
        chain = self._build_chain("session_update", final)
        await chain(update)
