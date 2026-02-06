"""Shared enums for context nodes."""

from enum import Enum


class ShellStatus(Enum):
    """Status of a shell command execution."""

    PENDING = "pending"  # Created, not yet started
    RUNNING = "running"  # Subprocess is executing
    COMPLETED = "completed"  # Finished successfully (exit_code == 0)
    FAILED = "failed"  # Finished with error (exit_code != 0)
    TIMEOUT = "timeout"  # Killed due to timeout
    CANCELLED = "cancelled"  # Cancelled by user


class LockStatus(Enum):
    """Status of a file lock."""

    PENDING = "pending"  # Waiting to acquire lock
    ACQUIRED = "acquired"  # Lock held
    TIMEOUT = "timeout"  # Failed to acquire within timeout
    RELEASED = "released"  # Lock released
    ERROR = "error"  # Error during lock operation


class MessageRole(Enum):
    """Role of a message in the conversation history."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class AgentRelation(Enum):
    """Relationship of an agent to the viewing agent."""

    SELF = "self"
    PARENT = "parent"
    CHILD = "child"
    PEER = "peer"
