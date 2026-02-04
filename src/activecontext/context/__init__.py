"""Context objects: ViewHandle, GroupHandle, and base classes.

This module provides the context graph (DAG) and typed context nodes
for managing the agent's working context.

Architecture (view/content separation):
- Content graph: DAG of ContextNodes for ticking and token rollup
- View graph: Flat list of NodeViews for rendering order
- ContentData: shared content storage (file content, artifacts, etc.)
- ContentRegistry: registry for shared ContentData
- NodeView: view-specific state (hide, expand) over ContextNode
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
from activecontext.context.exposed import exposed, get_exposed, is_exposed
from activecontext.context.graph import ContextGraph
from activecontext.context.headers import TokenInfo
from activecontext.context.markdown_parser import (
    HeadingSection,
    MarkdownParser,
    ParseResult,
    parse_markdown,
)
from activecontext.context.nodes import (
    AgentNode,
    AgentRelation,
    ArtifactNode,
    ContextNode,
    GroupNode,
    HelpNode,
    LineChange,
    LockNode,
    LockStatus,
    MCPServerNode,
    MessageNode,
    MessageRole,
    OnChildChangedHook,
    PluginManagerNode,
    PtyNode,
    PtyStatus,
    SessionNode,
    ShellNode,
    ShellStatus,
    TaskNode,
    TextNode,
    TopicNode,
    TraceNode,
    WorkNode,
    get_watchers,
    on_file_change,
    register_file_watcher,
    unregister_file_watcher,
)
from activecontext.context.registry import NodeTypeRegistry, get_node_registry
from activecontext.context.traceable import (
    format_value,
    get_traceable_fields,
    is_traceable,
    register_formatter,
    trace_all_fields,
    traceable,
)
from activecontext.context.view import ChoiceView, NodeView

__all__ = [
    # Registry
    "NodeTypeRegistry",
    "get_node_registry",
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
    # Graph
    "ContextGraph",
    # Content
    "ContentData",
    "ContentRegistry",
    "content_from_file",
    "content_from_artifact",
    "content_from_shell",
    "content_from_markdown",
    # View
    "ChoiceView",
    "NodeView",
    # Exposed
    "exposed",
    "is_exposed",
    "get_exposed",
    # Nodes
    "ContextNode",
    "TraceNode",
    "HelpNode",
    "OnChildChangedHook",
    "TextNode",
    "GroupNode",
    "TopicNode",
    "ArtifactNode",
    "ShellNode",
    "ShellStatus",
    "PtyNode",
    "PtyStatus",
    "LockNode",
    "LockStatus",
    "MCPServerNode",
    "PluginManagerNode",
    "MessageNode",
    "SessionNode",
    "WorkNode",
    "AgentNode",
    "AgentRelation",
    "MessageRole",
    "TaskNode",
    # File change tracking
    "LineChange",
    "register_file_watcher",
    "unregister_file_watcher",
    "get_watchers",
    "on_file_change",
]
