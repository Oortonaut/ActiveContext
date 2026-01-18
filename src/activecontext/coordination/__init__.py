"""Agent work coordination system.

A file-based scratchpad for agents to communicate which areas of the codebase
they're working on. Advisory only - warns about conflicts but doesn't block.
"""

from activecontext.coordination.schema import (
    Conflict,
    FileAccess,
    Scratchpad,
    WorkEntry,
)
from activecontext.coordination.scratchpad import ScratchpadManager

__all__ = [
    "Conflict",
    "FileAccess",
    "Scratchpad",
    "ScratchpadManager",
    "WorkEntry",
]
