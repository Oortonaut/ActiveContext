"""Multi-agent support for ActiveContext.

This module provides multi-agent coordination capabilities:
- AgentManager: Manages agent lifecycle, messaging, and shared nodes
- AgentHandle: DSL handle for interacting with agents from timeline
- AgentTypeRegistry: Registry for agent type definitions
- Schema classes: AgentState, AgentEntry, AgentMessage, AgentType

Example usage:

    # From Python API
    from activecontext.agents import AgentManager, AgentTypeRegistry
    from activecontext.coordination import ScratchpadManager

    registry = AgentTypeRegistry()
    scratchpad = ScratchpadManager(cwd="/project")
    agent_manager = AgentManager(session_manager, scratchpad, registry)

    handle = await agent_manager.spawn_agent(
        agent_type="explorer",
        task="Find authentication code",
    )

    # From DSL (within timeline)
    agent = spawn("explorer", task="find auth code")
    send(agent, "look for OAuth patterns", my_view)
    messages = recv()
    wait_message()
"""

from activecontext.agents.handle import AgentHandle
from activecontext.agents.manager import AgentManager
from activecontext.agents.registry import AgentTypeRegistry
from activecontext.agents.schema import (
    AgentEntry,
    AgentMessage,
    AgentState,
    AgentType,
)

__all__ = [
    "AgentHandle",
    "AgentManager",
    "AgentTypeRegistry",
    "AgentEntry",
    "AgentMessage",
    "AgentState",
    "AgentType",
]
