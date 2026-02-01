"""Agent work coordination system.

A file-based scratchpad for agents to communicate which areas of the codebase
they're working on. Advisory only - warns about conflicts but doesn't block.

Also provides a TaskGraphBridge for integrating with the task-graph MCP
server for structured task tracking and time metrics.
"""

from activecontext.coordination.schema import (
    Conflict,
    FileAccess,
    Scratchpad,
    WorkEntry,
)
from activecontext.coordination.scratchpad import ScratchpadManager
from activecontext.coordination.task_bridge import (
    MCPCaller,
    TaskGraphBridge,
    TaskInfo,
    WorkerInfo,
)

__all__ = [
    "Conflict",
    "FileAccess",
    "MCPCaller",
    "Scratchpad",
    "ScratchpadManager",
    "TaskGraphBridge",
    "TaskInfo",
    "WorkEntry",
    "WorkerInfo",
]
