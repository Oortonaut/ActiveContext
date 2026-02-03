"""Terminal execution support for shell commands.

Provides the shell() DSL function for executing shell commands,
with implementations for both ACP (IDE terminal) and subprocess (local).

Also includes:
- Terminal capability detection (color, unicode, size)
- PTY support for interactive commands
- Hook system for command lifecycle events
- Rider-specific quirks and workarounds
"""

from activecontext.terminal.capabilities import (
    ColorSupport,
    TerminalCapabilities,
    detect_capabilities,
    format_with_fallback,
    get_capabilities,
    get_unicode_char,
)
from activecontext.terminal.hooks import (
    HookManager,
    HookPhase,
    PostCommandContext,
    PostCommandHook,
    PreCommandContext,
    PreCommandHook,
    StateChangeContext,
    StateChangeHook,
    TickBoundaryContext,
    TickBoundaryHook,
    get_hook_manager,
    log_command_hook,
    log_result_hook,
)
from activecontext.terminal.protocol import TerminalExecutor
from activecontext.terminal.pty_backend import (
    PtyBackend,
    UnixPtyBackend,
    WinPtyBackend,
    create_pty_backend,
)
from activecontext.terminal.pty_support import (
    PTYConfig,
    execute_with_pty,
    is_pty_supported,
)
from activecontext.terminal.result import ShellResult
from activecontext.terminal.subprocess_executor import SubprocessTerminalExecutor

__all__ = [
    # Core execution
    "ShellResult",
    "TerminalExecutor",
    "SubprocessTerminalExecutor",
    # Capabilities
    "ColorSupport",
    "TerminalCapabilities",
    "detect_capabilities",
    "get_capabilities",
    "format_with_fallback",
    "get_unicode_char",
    # Hooks
    "HookManager",
    "HookPhase",
    "PreCommandContext",
    "PreCommandHook",
    "PostCommandContext",
    "PostCommandHook",
    "StateChangeContext",
    "StateChangeHook",
    "TickBoundaryContext",
    "TickBoundaryHook",
    "get_hook_manager",
    "log_command_hook",
    "log_result_hook",
    # PTY support (legacy one-shot)
    "PTYConfig",
    "is_pty_supported",
    "execute_with_pty",
    # PTY backend (long-lived interactive)
    "PtyBackend",
    "UnixPtyBackend",
    "WinPtyBackend",
    "create_pty_backend",
]

# ACPTerminalExecutor is imported separately to avoid requiring 'acp' package
# when using direct transport
