"""Agent handle for DSL interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from activecontext.agents.schema import AgentState

if TYPE_CHECKING:
    from activecontext.agents.manager import AgentManager
    from activecontext.context.nodes import ContextNode


class AgentHandle:
    """DSL handle for an agent, exposed to LLM namespace.

    Provides a fluent interface for agent interaction from the DSL:

    ```python
    # Spawn an agent
    agent = spawn("explorer", task="find auth code")

    # Check status
    print(agent.state)  # AgentState.RUNNING
    print(agent.task)   # "find auth code"

    # Send a message with node references
    agent.Send("summarize this", some_group_node)

    # Pause/resume
    agent.Pause()
    agent.Resume()

    # Terminate
    agent.Terminate()
    ```
    """

    def __init__(
        self,
        agent_id: str,
        agent_manager: AgentManager,
    ) -> None:
        """Initialize the handle.

        Args:
            agent_id: The agent's unique ID
            agent_manager: Reference to the agent manager
        """
        self._agent_id = agent_id
        self._manager = agent_manager

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self._agent_id

    @property
    def state(self) -> AgentState:
        """Get the agent's current state."""
        entry = self._manager.get_agent(self._agent_id)
        return entry.state if entry else AgentState.TERMINATED

    @property
    def task(self) -> str:
        """Get the agent's task description."""
        entry = self._manager.get_agent(self._agent_id)
        return entry.task if entry else ""

    @property
    def agent_type(self) -> str:
        """Get the agent's type."""
        entry = self._manager.get_agent(self._agent_id)
        return entry.agent_type if entry else ""

    @property
    def parent_id(self) -> str | None:
        """Get the parent agent's ID."""
        entry = self._manager.get_agent(self._agent_id)
        return entry.parent_id if entry else None

    def Send(self, content: str, *node_refs: ContextNode) -> str:
        """Send a message to this agent.

        Args:
            content: Message content
            *node_refs: Nodes to reference in the message

        Returns:
            Message ID
        """
        # Share the nodes so recipient can access them
        ref_ids: list[str] = []
        for node in node_refs:
            self._manager.share_node(node)
            ref_ids.append(node.node_id)

        # Get sender from current agent context if available
        # This will be set when called from DSL
        sender = getattr(self, "_sender_id", "unknown")

        # Create message synchronously (scratchpad operations are sync)
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create a task
            future = asyncio.ensure_future(
                self._manager.send_message(
                    sender=sender,
                    recipient=self._agent_id,
                    content=content,
                    node_refs=ref_ids,
                )
            )
            # Wait for the result
            return loop.run_until_complete(future)
        except RuntimeError:
            # No running loop, we can use asyncio.run
            return asyncio.get_event_loop().run_until_complete(
                self._manager.send_message(
                    sender=sender,
                    recipient=self._agent_id,
                    content=content,
                    node_refs=ref_ids,
                )
            )

    def Pause(self) -> AgentHandle:
        """Pause this agent.

        Returns:
            Self for chaining
        """
        import asyncio

        try:
            asyncio.get_running_loop()  # Raises RuntimeError if no loop
            asyncio.ensure_future(self._manager.pause_agent(self._agent_id))
        except RuntimeError:
            asyncio.get_event_loop().run_until_complete(self._manager.pause_agent(self._agent_id))
        return self

    def Resume(self) -> AgentHandle:
        """Resume this agent.

        Returns:
            Self for chaining
        """
        import asyncio

        try:
            asyncio.get_running_loop()  # Raises RuntimeError if no loop
            asyncio.ensure_future(self._manager.resume_agent(self._agent_id))
        except RuntimeError:
            asyncio.get_event_loop().run_until_complete(self._manager.resume_agent(self._agent_id))
        return self

    def Terminate(self) -> AgentHandle:
        """Terminate this agent.

        Returns:
            Self for chaining
        """
        import asyncio

        try:
            asyncio.get_running_loop()  # Raises RuntimeError if no loop
            asyncio.ensure_future(self._manager.terminate_agent(self._agent_id))
        except RuntimeError:
            asyncio.get_event_loop().run_until_complete(
                self._manager.terminate_agent(self._agent_id)
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        entry = self._manager.get_agent(self._agent_id)
        if entry:
            return entry.to_dict()
        return {
            "agent_id": self._agent_id,
            "state": AgentState.TERMINATED.value,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"AgentHandle({self._agent_id!r}, state={self.state.value})"
