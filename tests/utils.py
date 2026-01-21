"""Shared test utilities and fixtures for ActiveContext tests."""

from __future__ import annotations

import asyncio
from typing import Any

from activecontext.core.llm.provider import Message, Role


def create_mock_message(role: str, content: str, actor: str | None = None) -> Message:
    """Create a mock Message object.

    Args:
        role: Message role ("system", "user", "assistant")
        content: Message content
        actor: Optional actor identifier

    Returns:
        Message instance
    """
    role_enum = Role(role)
    return Message(role=role_enum, content=content, originator=actor)


def create_mock_context_node(
    node_id: str,
    node_type: str,
    mode: str = "idle",
    parent_ids: set[str] | None = None,
    children_ids: set[str] | None = None,
) -> Any:
    """Create a mock ContextNode for testing.

    Args:
        node_id: Node identifier
        node_type: Type of node (e.g., "view", "group")
        mode: Node mode ("idle", "running")
        parent_ids: Set of parent node IDs
        children_ids: Set of child node IDs

    Returns:
        Mock node object with required attributes
    """
    from unittest.mock import Mock

    node = Mock()
    node.node_id = node_id
    node.node_type = node_type
    node.mode = mode
    node.parent_ids = parent_ids or set()
    node.children_ids = children_ids or set()
    node.child_order = None  # Lazily initialized like real nodes
    node._graph = None

    # Add add_child method that mimics real ContextNode behavior
    def add_child(child, *, after=None):
        if not node._graph:
            raise RuntimeError(f"Cannot add_child: node {node.node_id} is not in a graph")
        return node._graph.link(child.node_id, node.node_id, after=after)

    node.add_child = add_child

    # Add GetDigest method
    node.GetDigest = Mock(
        return_value={
            "node_id": node_id,
            "node_type": node_type,
            "mode": mode,
        }
    )

    return node


async def wait_for_async(coro, timeout: float = 1.0):
    """Wait for an async coroutine with a timeout.

    Args:
        coro: Coroutine to await
        timeout: Timeout in seconds

    Returns:
        Result of the coroutine

    Raises:
        asyncio.TimeoutError: If timeout is exceeded
    """
    return await asyncio.wait_for(coro, timeout=timeout)


def create_mock_llm_response(content: str = "Test response") -> dict[str, Any]:
    """Create a mock LiteLLM response object.

    Args:
        content: Response content

    Returns:
        Dict mimicking litellm response structure
    """
    from unittest.mock import Mock

    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = content
    response.choices[0].finish_reason = "stop"

    response.usage = Mock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30

    return response


def create_mock_llm_stream_chunk(
    text: str = "chunk", is_final: bool = False
) -> dict[str, Any]:
    """Create a mock streaming chunk from LiteLLM.

    Args:
        text: Chunk text content
        is_final: Whether this is the final chunk

    Returns:
        Dict mimicking litellm streaming chunk structure
    """
    from unittest.mock import Mock

    chunk = Mock()
    chunk.choices = [Mock()]
    chunk.choices[0].delta = Mock()
    chunk.choices[0].delta.content = text
    chunk.choices[0].finish_reason = "stop" if is_final else None

    return chunk
