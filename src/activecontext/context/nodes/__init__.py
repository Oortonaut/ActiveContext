"""Context node types for the context DAG.

This package defines the typed node hierarchy:
- ContextNode: Base class with common fields and notification
- TextNode: File content view (text)
- GroupNode: Summary facade over children
- TopicNode: Conversation segment
- ArtifactNode: Code/output artifact
- ShellNode: Async shell command execution
- PtyNode: Interactive PTY session
- LockNode: File lock acquisition
- SessionNode: Session metadata
- MessageNode: Conversation message with ID for referencing
- MessageSegmentNode: Parsed segment of LLM response
- WorkNode: Multi-agent work coordination
- MCPServerNode: MCP server connection
- MCPToolNode: Individual MCP tool
- MCPManagerNode: MCP server manager
- PluginManagerNode: Plugin server manager
- AgentNode: Child agent representation
- TraceNode: Change trace history
- TaskNode: Task tracking
- HelpNode: Documentation node
- MarkdownNode: Structured markdown with list parsing
- MarkdownListItemNode: Individual list items with nesting support
- FileSystemNode: Directory tree view with filtering
- ClockNode: Timer/countdown with tick-driven updates
- FunctionDocNode: Function signature and docstring extraction
- StatementNode: Python statement in REPL timeline
- StatementResultNode: Statement execution result
"""

# Base classes and core types
# Node types
from activecontext.context.nodes.agent import AgentNode
from activecontext.context.nodes.artifact import ArtifactNode
from activecontext.context.nodes.base import (
    TOKEN_COUNTS_OVERHEAD,
    ContextNode,
    OnChildChangedHook,
    SimpleNode,
)
from activecontext.context.nodes.clock import ClockNode

# Enums
from activecontext.context.nodes.enums import (
    AgentRelation,
    LockStatus,
    MessageRole,
    ShellStatus,
)

# File watcher
from activecontext.context.nodes.file_watcher import (
    LineChange,
    _file_watchers,
    get_watchers,
    on_file_change,
    register_file_watcher,
    unregister_file_watcher,
)
from activecontext.context.nodes.filesystem import FileSystemNode
from activecontext.context.nodes.function_doc import FunctionDocNode
from activecontext.context.nodes.group import GroupNode
from activecontext.context.nodes.help import HelpNode, _extract_help_content
from activecontext.context.nodes.lock import LockNode
from activecontext.context.nodes.markdown import MarkdownListItemNode, MarkdownNode
from activecontext.context.nodes.mcp import MCPManagerNode, MCPServerNode, MCPToolNode
from activecontext.context.nodes.message import MessageNode, MessageSegmentNode
from activecontext.context.nodes.plugin_manager import PluginManagerNode

# PtyStatus is in pty.py (not enums.py as it's PtyNode-specific)
from activecontext.context.nodes.pty import (
    _PTY_MAX_BYTES,
    _PTY_MAX_LINES,
    PtyNode,
    PtyStatus,
    _strip_ansi,
)
from activecontext.context.nodes.session import SessionNode
from activecontext.context.nodes.shell import ShellNode
from activecontext.context.nodes.statement import StatementNode, StatementResultNode
from activecontext.context.nodes.task import TaskNode
from activecontext.context.nodes.text import TextNode
from activecontext.context.nodes.topic import TopicNode
from activecontext.context.nodes.trace import TraceNode
from activecontext.context.nodes.work import WorkNode

__all__ = [
    # Base classes
    "ContextNode",
    "SimpleNode",
    "OnChildChangedHook",
    "TOKEN_COUNTS_OVERHEAD",
    # Enums
    "ShellStatus",
    "LockStatus",
    "MessageRole",
    "AgentRelation",
    "PtyStatus",
    # File watcher
    "LineChange",
    "_file_watchers",
    "register_file_watcher",
    "unregister_file_watcher",
    "get_watchers",
    "on_file_change",
    # PTY constants
    "_PTY_MAX_LINES",
    "_PTY_MAX_BYTES",
    "_strip_ansi",
    # Node types
    "TextNode",
    "GroupNode",
    "TopicNode",
    "ArtifactNode",
    "ShellNode",
    "PtyNode",
    "LockNode",
    "SessionNode",
    "MessageNode",
    "MessageSegmentNode",
    "WorkNode",
    "MCPServerNode",
    "MCPToolNode",
    "MCPManagerNode",
    "PluginManagerNode",
    "AgentNode",
    "TraceNode",
    "TaskNode",
    "HelpNode",
    "_extract_help_content",
    "MarkdownListItemNode",
    "MarkdownNode",
    "FileSystemNode",
    "ClockNode",
    "FunctionDocNode",
    "StatementNode",
    "StatementResultNode",
]
