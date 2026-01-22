"""Agent spawning and communication manager.

Manages multi-agent operations including spawning child agents, inter-agent
messaging, and agent lifecycle control.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from activecontext.agents.handle import AgentHandle
    from activecontext.agents.manager import AgentManager
    from activecontext.context.graph import ContextGraph
    from activecontext.context.nodes import ContextNode


class AgentSpawner:
    """Manages agent spawning and inter-agent communication.

    Responsibilities:
    - Spawn child agents from DSL
    - Send/receive messages between agents
    - Control agent lifecycle (pause/resume/terminate)
    - Share context nodes between agents
    - Query agent status
    """

    def __init__(
        self,
        *,
        agent_manager: AgentManager | None,
        agent_id: str | None,
        context_graph: ContextGraph,
        cwd: str,
    ):
        """Initialize agent spawner.

        Args:
            agent_manager: The AgentManager for multi-agent coordination
            agent_id: This agent's ID (None for root timeline)
            context_graph: The session's context graph
            cwd: Working directory for spawned agents
        """
        self._agent_manager = agent_manager
        self._agent_id = agent_id
        self._context_graph = context_graph
        self._cwd = cwd

    def set_agent_manager(self, manager: AgentManager) -> None:
        """Set the agent manager (called after Timeline creation)."""
        self._agent_manager = manager

    def set_agent_id(self, agent_id: str) -> None:
        """Set this agent's ID (called after spawn)."""
        self._agent_id = agent_id

    def spawn(
        self,
        agent_type: str,
        task: str,
        **kwargs: Any,
    ) -> AgentHandle:
        """Spawn a new child agent.

        DSL function: spawn(agent_type, task, **kwargs)

        Args:
            agent_type: Type ID (explorer, summarizer, etc.)
            task: Task description
            **kwargs: Additional arguments for session creation

        Returns:
            AgentHandle for interacting with the spawned agent

        Example:
            agent = spawn("explorer", task="find auth code")
            agent.Send("look for OAuth")
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        # Spawn synchronously using existing loop
        try:
            loop = asyncio.get_running_loop()
            # Create a coroutine and schedule it
            coro = self._agent_manager.spawn_agent(
                agent_type=agent_type,
                task=task,
                parent_id=self._agent_id,
                cwd=self._cwd,
                **kwargs,
            )
            # Use ensure_future to schedule the coroutine
            future = asyncio.ensure_future(coro)
            return future  # type: ignore  # Will be awaited by exec
        except RuntimeError:
            # No running loop - shouldn't happen in normal DSL execution
            raise RuntimeError("spawn() must be called in async context")

    async def spawn_async(
        self,
        agent_type: str,
        task: str,
        **kwargs: Any,
    ) -> AgentHandle:
        """Async version of spawn for internal use."""
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        return await self._agent_manager.spawn_agent(
            agent_type=agent_type,
            task=task,
            parent_id=self._agent_id,
            cwd=self._cwd,
            **kwargs,
        )

    def send_message(
        self,
        target: Any,  # AgentHandle or str (agent_id)
        content: str,
        *node_refs: ContextNode,
    ) -> str:
        """Send a message to another agent.

        DSL function: send(target, content, *node_refs)

        Args:
            target: AgentHandle or agent ID string
            content: Message content
            *node_refs: Nodes to share with the recipient

        Returns:
            Message ID

        Example:
            send(agent, "summarize this group", my_group)
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        # Resolve target to agent_id
        if isinstance(target, AgentHandle):
            recipient = target.agent_id
        elif isinstance(target, str):
            recipient = target
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(target)}")

        # Share nodes and collect IDs
        ref_ids: list[str] = []
        for node in node_refs:
            self._agent_manager.share_node(node)
            ref_ids.append(node.node_id)

        # Get sender ID
        sender = self._agent_id or "unknown"

        # Send message (sync operation via scratchpad)
        try:
            loop = asyncio.get_running_loop()
            coro = self._agent_manager.send_message(
                sender=sender,
                recipient=recipient,
                content=content,
                node_refs=ref_ids,
            )
            return asyncio.ensure_future(coro)  # type: ignore
        except RuntimeError:
            raise RuntimeError("send() must be called in async context")

    def send_update(
        self,
        content: str,
        *node_refs: ContextNode,
    ) -> str:
        """Send an update message to parent agent and continue running.

        DSL function: send_update(content, *node_refs)

        Unlike send(), this sends to the parent agent specifically and
        does NOT end the turn - the agent continues executing.

        Args:
            content: Message content
            *node_refs: Nodes to share with the parent

        Returns:
            Message ID

        Example:
            send_update("Found 3 auth files so far", partial_results)
            # Agent continues working...
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        # Get parent ID
        entry = self._agent_manager.get_agent(self._agent_id) if self._agent_id else None
        if not entry or not entry.parent_id:
            raise RuntimeError("No parent agent to send update to")

        parent_id = entry.parent_id

        # Share nodes and collect IDs
        ref_ids: list[str] = []
        for node in node_refs:
            self._agent_manager.share_node(node)
            ref_ids.append(node.node_id)

        # Get sender ID
        sender = self._agent_id or "unknown"

        # Send message (sync operation via scratchpad)
        try:
            loop = asyncio.get_running_loop()
            coro = self._agent_manager.send_message(
                sender=sender,
                recipient=parent_id,
                content=content,
                node_refs=ref_ids,
            )
            future = asyncio.ensure_future(coro)
            # Don't set _done_called - agent continues running
            return future  # type: ignore
        except RuntimeError:
            raise RuntimeError("send_update() must be called in async context")

    def recv_messages(self) -> list[dict[str, Any]]:
        """Receive pending messages for this agent.

        DSL function: recv()

        Returns:
            List of message dictionaries with sender, content, node_refs, etc.

        Example:
            messages = recv()
            for msg in messages:
                print(f"From {msg['sender']}: {msg['content']}")
        """
        if not self._agent_manager:
            return []

        if not self._agent_id:
            return []

        messages = self._agent_manager.get_messages(self._agent_id, status="pending")

        # Mark as read and convert to dicts
        result: list[dict[str, Any]] = []
        for msg in messages:
            self._agent_manager.mark_message_read(msg.id)
            result.append({
                "id": msg.id,
                "sender": msg.sender,
                "content": msg.content,
                "node_refs": msg.node_refs,
                "created_at": msg.created_at.isoformat(),
                "reply_to": msg.reply_to,
                "metadata": msg.metadata,
            })

        return result

    def list_agents(self) -> list[dict[str, Any]]:
        """List all active agents.

        DSL function: agents()

        Returns:
            List of agent info dictionaries
        """
        if not self._agent_manager:
            return []

        entries = self._agent_manager.list_agents()
        return [
            {
                "id": e.id,
                "type": e.agent_type,
                "task": e.task,
                "state": e.state.value,
                "parent_id": e.parent_id,
                "session_id": e.session_id,
            }
            for e in entries
        ]

    def get_agent_status(self, agent: Any) -> dict[str, Any]:
        """Get status of a specific agent.

        DSL function: agent_status(agent)

        Args:
            agent: AgentHandle or agent ID string

        Returns:
            Agent status dictionary
        """
        if not self._agent_manager:
            return {"error": "Agent manager not available"}

        from activecontext.agents.handle import AgentHandle

        # Resolve agent_id
        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            return {"error": f"Expected AgentHandle or str, got {type(agent)}"}

        entry = self._agent_manager.get_agent(agent_id)
        if not entry:
            return {"error": f"Agent {agent_id} not found"}

        return {
            "id": entry.id,
            "type": entry.agent_type,
            "task": entry.task,
            "state": entry.state.value,
            "parent_id": entry.parent_id,
            "session_id": entry.session_id,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
        }

    def pause_agent(self, agent: Any) -> None:
        """Pause an agent.

        DSL function: pause_agent(agent)

        Args:
            agent: AgentHandle or agent ID string
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(agent)}")

        asyncio.ensure_future(self._agent_manager.pause_agent(agent_id))

    def resume_agent(self, agent: Any) -> None:
        """Resume a paused agent.

        DSL function: resume_agent(agent)

        Args:
            agent: AgentHandle or agent ID string
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(agent)}")

        asyncio.ensure_future(self._agent_manager.resume_agent(agent_id))

    def terminate_agent(self, agent: Any) -> None:
        """Terminate an agent.

        DSL function: terminate_agent(agent)

        Args:
            agent: AgentHandle or agent ID string
        """
        if not self._agent_manager:
            raise RuntimeError("Agent manager not available")

        from activecontext.agents.handle import AgentHandle

        if isinstance(agent, AgentHandle):
            agent_id = agent.agent_id
        elif isinstance(agent, str):
            agent_id = agent
        else:
            raise TypeError(f"Expected AgentHandle or str, got {type(agent)}")

        asyncio.ensure_future(self._agent_manager.terminate_agent(agent_id))

    def get_shared_node(self, node_id: str) -> ContextNode | None:
        """Get a shared node by ID.

        DSL function: get_shared_node(node_id)

        Args:
            node_id: Node ID from another agent's message

        Returns:
            The shared node, or None if not found
        """
        if not self._agent_manager:
            return None

        return self._agent_manager.get_shared_node(node_id)

    def setup_namespace(self, namespace: dict[str, Any]) -> None:
        """Add agent functions to namespace.

        Called by Timeline._setup_agent_namespace() when agent manager is set.

        Args:
            namespace: The Timeline namespace to update
        """
        if self._agent_manager is None:
            return

        namespace.update({
            "spawn": self.spawn,
            "send": self.send_message,
            "send_update": self.send_update,
            "recv": self.recv_messages,
            "agents": self.list_agents,
            "agent_status": self.get_agent_status,
            "pause_agent": self.pause_agent,
            "resume_agent": self.resume_agent,
            "terminate_agent": self.terminate_agent,
            "get_shared_node": self.get_shared_node,
        })
