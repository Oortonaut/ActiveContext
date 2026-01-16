"""Context objects: ViewHandle, GroupHandle, and base classes.

This module provides the context graph (DAG) and typed context nodes
for managing the agent's working context.
"""

from activecontext.context.checkpoint import Checkpoint, GroupState
from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    ArtifactNode,
    ContextNode,
    Diff,
    GroupNode,
    OnChildChangedHook,
    TopicNode,
    ViewNode,
)

__all__ = [
    "Checkpoint",
    "ContextGraph",
    "ContextNode",
    "Diff",
    "GroupState",
    "ViewNode",
    "GroupNode",
    "TopicNode",
    "ArtifactNode",
    "OnChildChangedHook",
]
