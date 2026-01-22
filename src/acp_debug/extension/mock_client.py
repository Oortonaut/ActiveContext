"""Mock client base class for testing agent implementations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from acp_debug.extension.base import ACPExtension
from acp_debug.types.common import PermissionOptionKind
from acp_debug.types.requests import (
    CreateTerminalRequest,
    KillTerminalRequest,
    PermissionRequest,
    ReadFileRequest,
    ReleaseTerminalRequest,
    TerminalOutputRequest,
    WaitForExitRequest,
    WriteFileRequest,
)
from acp_debug.types.responses import (
    CreateTerminalResponse,
    PermissionOutcome,
    PermissionResponse,
    ReadFileResponse,
    TerminalOutputResponse,
    WaitForExitResponse,
)


class MockClientBase(ACPExtension):
    """Base class for mock clients - override methods to customize requests.

    Default behavior provides sensible implementations:
    - Permissions: allow_once by default
    - File reads: read from actual filesystem
    - File writes: write to actual filesystem
    - Terminal: not implemented (raises NotImplementedError)

    Usage:
        class MyMockClient(MockClientBase):
            async def request_permission(self, req, call):
                # Auto-deny certain operations
                if "dangerous" in req.tool_call.title:
                    return PermissionResponse(
                        outcome=PermissionOutcome(outcome="selected", option_id="reject_once")
                    )
                return await super().request_permission(req, call)
    """

    def __init__(self) -> None:
        super().__init__()
        self.auto_approve_permissions = True

    async def request_permission(
        self,
        req: PermissionRequest,
        call: Callable[[PermissionRequest], Awaitable[PermissionResponse]],
    ) -> PermissionResponse:
        """Handle permission request - default: allow_once."""
        if self.auto_approve_permissions:
            # Find the allow_once option
            for option in req.options:
                if option.kind == PermissionOptionKind.ALLOW_ONCE:
                    return PermissionResponse(
                        outcome=PermissionOutcome(
                            outcome="selected",
                            option_id=option.option_id,
                        )
                    )

        # Fallback to first option
        return PermissionResponse(
            outcome=PermissionOutcome(
                outcome="selected",
                option_id=req.options[0].option_id if req.options else None,
            )
        )

    async def fs_read_text_file(
        self,
        req: ReadFileRequest,
        call: Callable[[ReadFileRequest], Awaitable[ReadFileResponse]],
    ) -> ReadFileResponse:
        """Read file from filesystem."""
        path = Path(req.path)
        content = path.read_text(encoding="utf-8")

        # Apply line/limit if specified
        if req.line is not None or req.limit is not None:
            lines = content.splitlines(keepends=True)
            start = (req.line or 1) - 1  # Convert to 0-based
            end = start + (req.limit or len(lines))
            content = "".join(lines[start:end])

        return ReadFileResponse(content=content)

    async def fs_write_text_file(
        self,
        req: WriteFileRequest,
        call: Callable[[WriteFileRequest], Awaitable[None]],
    ) -> None:
        """Write file to filesystem."""
        path = Path(req.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(req.content, encoding="utf-8")

    async def terminal_create(
        self,
        req: CreateTerminalRequest,
        call: Callable[[CreateTerminalRequest], Awaitable[CreateTerminalResponse]],
    ) -> CreateTerminalResponse:
        """Create terminal - not implemented by default."""
        raise NotImplementedError(
            "Terminal operations not implemented in MockClientBase. "
            "Override terminal_* methods to add support."
        )

    async def terminal_output(
        self,
        req: TerminalOutputRequest,
        call: Callable[[TerminalOutputRequest], Awaitable[TerminalOutputResponse]],
    ) -> TerminalOutputResponse:
        """Get terminal output - not implemented by default."""
        raise NotImplementedError("Terminal operations not implemented in MockClientBase.")

    async def terminal_wait_for_exit(
        self,
        req: WaitForExitRequest,
        call: Callable[[WaitForExitRequest], Awaitable[WaitForExitResponse]],
    ) -> WaitForExitResponse:
        """Wait for terminal exit - not implemented by default."""
        raise NotImplementedError("Terminal operations not implemented in MockClientBase.")

    async def terminal_kill(
        self,
        req: KillTerminalRequest,
        call: Callable[[KillTerminalRequest], Awaitable[None]],
    ) -> None:
        """Kill terminal - not implemented by default."""
        raise NotImplementedError("Terminal operations not implemented in MockClientBase.")

    async def terminal_release(
        self,
        req: ReleaseTerminalRequest,
        call: Callable[[ReleaseTerminalRequest], Awaitable[None]],
    ) -> None:
        """Release terminal - not implemented by default."""
        raise NotImplementedError("Terminal operations not implemented in MockClientBase.")
