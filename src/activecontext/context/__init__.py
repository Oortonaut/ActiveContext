"""Context objects: ViewHandle, GroupHandle, and base classes.

This module provides the context graph (DAG) and typed context nodes
for managing the agent's working context.

Split architecture for multi-agent support:
- ContentData: shared content storage (file content, artifacts, etc.)
- AgentView: per-agent visibility settings (hidden, state, tokens)
- ContentRegistry: registry for shared ContentData
- ViewRegistry: registry for per-agent AgentViews
"""

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
from activecontext.context.nodes import (
    AgentNode,
    ArtifactNode,
    ContextNode,
    Trace,
    GroupNode,
    LockNode,
    LockStatus,
    MarkdownNode,
    MCPServerNode,
    MessageNode,
    OnChildChangedHook,
    SessionNode,
    ShellNode,
    ShellStatus,
    TopicNode,
    ViewNode,
    WorkNode,
)
from activecontext.context.view import AgentView, ViewRegistry

__all__ = [
    # Checkpoint
    "Checkpoint",
    "GroupState",
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
    "Trace",
    "OnChildChangedHook",
    "ViewNode",
    "GroupNode",
    "TopicNode",
    "ArtifactNode",
    "ShellNode",
    "ShellStatus",
    "LockNode",
    "LockStatus",
    "MarkdownNode",
    "MCPServerNode",
    "MessageNode",
    "SessionNode",
    "WorkNode",
    "AgentNode",
]
