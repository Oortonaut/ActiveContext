"""Conversation delegation protocols for interactive handlers.

This module provides the protocol definitions for delegating bidirectional
conversations to external handlers (shells, menus, debuggers, etc.).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol


class InputType(Enum):
    """Type of input request for conversation handlers."""

    TEXT = "text"  # Free-form text input
    CHOICE = "choice"  # Select from options
    CONFIRM = "confirm"  # Yes/No confirmation
    PASSWORD = "password"  # Hidden input


class ConversationTransport(Protocol):
    """Protocol for bidirectional conversation control.

    All messages flow through Session → MessageNode → ContextGraph.
    The originator field distinguishes sources (user, shell, subagent, etc.).

    Example usage:
        >>> transport = SessionConversationTransport(session, originator="shell:bash")
        >>> await transport.send_output("Hello from shell")
        >>> response = await transport.request_input("Enter command:")
        >>> print(f"User entered: {response}")
    """

    async def send_output(
        self,
        text: str,
        *,
        append: bool = True,
        originator: str | None = None,
    ) -> None:
        """Send output to user (creates MessageNode with handler's originator).

        Args:
            text: Output text to display
            append: If True, append to existing output. If False, replace.
            originator: Override originator for this message (default: handler's originator)
        """
        ...

    async def send_progress(
        self,
        current: int,
        total: int,
        *,
        status: str = "",
        show_percentage: bool = True,
    ) -> None:
        """Send progress update (non-blocking).

        Args:
            current: Current progress value
            total: Total progress value
            status: Optional status message
            show_percentage: Show percentage calculation
        """
        ...

    async def request_input(
        self,
        prompt: str,
        *,
        input_type: InputType = InputType.TEXT,
        choices: list[str] | None = None,
        default: str | None = None,
    ) -> str:
        """Request input from user (blocking until response).

        Flow:
        1. Creates MessageNode(role="user", originator=handler_originator, content=prompt)
        2. Waits for response
        3. Response arrives as MessageNode(role="user", originator="user", content=response)
        4. Returns response string

        Args:
            prompt: Prompt text to display
            input_type: Type of input to request
            choices: Available choices (for CHOICE type)
            default: Default value

        Returns:
            User's input response

        Raises:
            asyncio.CancelledError: If conversation is cancelled
        """
        ...

    def check_cancelled(self) -> bool:
        """Check if user cancelled (regular code path, not exception).

        Returns:
            True if conversation was cancelled
        """
        ...

    def check_input(self) -> bool:
        """Check if input is available without blocking (for sync polling).

        Returns:
            True if input response is queued and available
        """
        ...

    def get_session_id(self) -> str:
        """Get session ID for this conversation.

        Returns:
            Session identifier
        """
        ...

    def get_originator(self) -> str:
        """Get the originator identifier for this conversation.

        Returns:
            Originator string (e.g., "shell:bash", "mcp:menu")
        """
        ...


class ConversationHandler(Protocol):
    """Protocol for handlers that manage delegated conversations.

    Handlers implement the conversation logic for specific interaction patterns:
    - InteractiveShellHandler: Connects user to bash shell
    - MCPMenuHandler: Runs MCP configuration menu
    - DebuggerHandler: Attaches to gdb/lldb session

    Example:
        >>> class EchoHandler:
        ...     async def handle(self, transport: ConversationTransport) -> str:
        ...         await transport.send_output("Echo handler started")
        ...         user_input = await transport.request_input("Say something:")
        ...         await transport.send_output(f"You said: {user_input}")
        ...         return user_input
    """

    async def handle(self, transport: ConversationTransport) -> Any:
        """Handle the delegated conversation.

        Args:
            transport: Communication channel to user

        Returns:
            Result of the conversation (becomes MessageNode with handler's originator)

        Raises:
            asyncio.CancelledError: If conversation is cancelled
        """
        ...
