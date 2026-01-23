"""Context objects: ViewHandle, GroupHandle, and base classes.

This module provides the context graph (DAG) and typed context nodes
for managing the agent's working context.

Split architecture for multi-agent support:
- ContentData: shared content storage (file content, artifacts, etc.)
- AgentView: per-agent visibility settings (hidden, state, tokens)
- ContentRegistry: registry for shared ContentData
- ViewRegistry: registry for per-agent AgentViews
"""

from activecontext.context.buffer import TextBuffer
from activecontext.context.checkpoint import Checkpoint, GroupState
from activecontext.context.content import (
    ContentData,
    ContentRegistry,
    content_from_artifact,
    content_from_file,
    content_from_markdown,
    content_from_shell,
)
from activecontext.context.graph import ContextGraph
from activecontext.context.headers import TokenInfo, format_token_info, render_header
from activecontext.context.markdown_parser import (
    HeadingSection,
    MarkdownParser,
    ParseResult,
    parse_markdown,
)
from activecontext.context.nodes import (
    AgentNode,
    ArtifactNode,
    ContextNode,
    GroupNode,
    LockNode,
    LockStatus,
    MCPServerNode,
    MessageNode,
    OnChildChangedHook,
    SessionNode,
    ShellNode,
    ShellStatus,
    TextNode,
    TopicNode,
    TraceNode,
    WorkNode,
)
from activecontext.context.traceable import (
    trace_all_fields,
    traceable,
    format_value,
    register_formatter,
    is_traceable,
    get_traceable_fields,
)
from activecontext.context.view import AgentView, ViewRegistry

__all__ = [
    # Traceable
    "trace_all_fields",
    "traceable",
    "format_value",
    "register_formatter",
    "is_traceable",
    "get_traceable_fields",
    # Buffer
    "TextBuffer",
    # Checkpoint
    "Checkpoint",
    "GroupState",
    # Markdown Parser
    "HeadingSection",
    "MarkdownParser",
    "ParseResult",
    "parse_markdown",
    # Headers
    "TokenInfo",
    "format_token_info",
    "render_header",
    # Graph
    "ContextGraph",
    # Content (split architecture)
    "ContentData",
    "ContentRegistry",
    "content_from_file",
    "content_from_artifact",
    "content_from_shell",
    "content_from_markdown",
    # View (split architecture)
    "AgentView",
    "ViewRegistry",
    # Nodes
    "ContextNode",
    "TraceNode",
    "OnChildChangedHook",
    "TextNode",
    "GroupNode",
    "TopicNode",
    "ArtifactNode",
    "ShellNode",
    "ShellStatus",
    "LockNode",
    "LockStatus",
    "MCPServerNode",
    "MessageNode",
    "SessionNode",
    "WorkNode",
    "AgentNode",
]
