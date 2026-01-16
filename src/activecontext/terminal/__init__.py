"""Terminal execution support for shell commands.

Provides the shell() DSL function for executing shell commands,
with implementations for both ACP (IDE terminal) and subprocess (local).
"""

from activecontext.terminal.protocol import TerminalExecutor
from activecontext.terminal.result import ShellResult
from activecontext.terminal.subprocess_executor import SubprocessTerminalExecutor

__all__ = [
    "ShellResult",
    "TerminalExecutor",
    "SubprocessTerminalExecutor",
]

# ACPTerminalExecutor is imported separately to avoid requiring 'acp' package
# when using direct transport
