"""Capability protocols for context nodes.

This module defines runtime-checkable protocols that nodes can implement
to declare capabilities. Using protocols allows adding new node types
without modifying capability-checking code.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = ["Waitable"]


@runtime_checkable
class Waitable(Protocol):
    """Node that can be waited on for completion.

    Nodes implementing this protocol have an async operation that
    completes at some point. The `is_complete` property indicates
    whether that operation has finished.

    Built-in waitable nodes:
    - ShellNode: Shell command execution
    - LockNode: File lock acquisition
    - PtyNode: PTY session (complete when closed)
    """

    @property
    def node_id(self) -> str:
        """Unique identifier for this node."""
        ...

    @property
    def is_complete(self) -> bool:
        """True when the async operation has finished."""
        ...

    @property
    def error(self) -> str | None:
        """Error message if failed, None if successful."""
        ...

    def get_wake_data(self) -> dict[str, Any]:
        """Get data for formatting wake prompt templates."""
        ...
