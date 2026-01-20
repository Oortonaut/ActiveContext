"""Agent manager for multi-agent coordination."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from activecontext.agents.registry import AgentTypeRegistry
from activecontext.agents.schema import AgentEntry, AgentMessage, AgentState

if TYPE_CHECKING:
    from activecontext.agents.handle import AgentHandle
    from activecontext.context.nodes import ContextNode
    from activecontext.coordination import ScratchpadManager
    from activecontext.session.session_manager import Session, SessionManager


class AgentManager:
    """Manages agent lifecycle within a process.

    Works alongside SessionManager to add agent-specific functionality:
    - Agent spawning with parent/child relationships
    - Message routing between agents
    - Agent type registry
    - Shared node registry for cross-agent references

    Example:
        ```python
        # Create via SessionManager
        manager = SessionManager()
        agent_manager = AgentManager(manager, scratchpad, type_registry)

        # Spawn agents
        handle = await agent_manager.spawn_agent(
            agent_type="explorer",
            task="Find authentication code",
            cwd="/project",
        )

        # Send messages
        await agent_manager.send_message(
            sender="agent_a",
            recipient=handle.agent_id,
            content="Search for OAuth",
        )
        ```
    """

    def __init__(
        self,
        session_manager: SessionManager,
        scratchpad_manager: ScratchpadManager,
        type_registry: AgentTypeRegistry | None = None,
    ) -> None:
        """Initialize the agent manager.

        Args:
            session_manager: Session manager for creating underlying sessions
            scratchpad_manager: Scratchpad manager for coordination
            type_registry: Optional agent type registry; creates default if not provided
        """
        self._session_manager = session_manager
        self._scratchpad_manager = scratchpad_manager
        self._type_registry = type_registry or AgentTypeRegistry()

        # Local caches (authoritative source is scratchpad)
        self._agents: dict[str, AgentEntry] = {}
        self._agent_sessions: dict[str, Session] = {}

        # Shared node registry for cross-agent references
        self._shared_nodes: dict[str, ContextNode] = {}

    @property
    def type_registry(self) -> AgentTypeRegistry:
        """Get the agent type registry."""
        return self._type_registry

    async def spawn_agent(
        self,
        agent_type: str,
        task: str,
        cwd: str | None = None,
        parent_id: str | None = None,
        **kwargs: Any,
    ) -> AgentHandle:
        """Spawn a new agent.

        Creates an underlying Session with the agent type's system prompt.
        Registers the agent in the scratchpad for coordination.

        Args:
            agent_type: Type ID (explorer, summarizer, etc.)
            task: Task description
            cwd: Working directory; uses scratchpad's cwd if not provided
            parent_id: Parent agent ID if spawned by another agent
            **kwargs: Additional arguments passed to create_session

        Returns:
            AgentHandle for interacting with the agent

        Raises:
            ValueError: If agent_type is not registered
        """
        from activecontext.agents.handle import AgentHandle

        # Validate agent type
        type_def = self._type_registry.get(agent_type)
        if type_def is None:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Generate agent ID
        agent_id = uuid.uuid4().hex[:8]

        # Use scratchpad's cwd as default
        if cwd is None:
            cwd = str(self._scratchpad_manager._cwd)

        # Create underlying session
        session = await self._session_manager.create_session(
            cwd=cwd,
            **kwargs,
        )

        # Register in scratchpad
        entry = self._scratchpad_manager.register_agent(
            agent_id=agent_id,
            session_id=session.session_id,
            agent_type=agent_type,
            task=task,
            parent_id=parent_id,
        )

        # Cache locally
        self._agents[agent_id] = entry
        self._agent_sessions[agent_id] = session

        # Inject agent manager into session's timeline for DSL access
        session.timeline._agent_manager = self
        session.timeline._agent_id = agent_id
        session.timeline._setup_agent_namespace()

        return AgentHandle(agent_id, self)

    def get_agent(self, agent_id: str) -> AgentEntry | None:
        """Get an agent entry by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent entry, or None if not found
        """
        # Try local cache first
        if agent_id in self._agents:
            return self._agents[agent_id]
        # Fall back to scratchpad
        return self._scratchpad_manager.get_agent(agent_id)

    def get_session(self, agent_id: str) -> Session | None:
        """Get the underlying session for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Session, or None if not found
        """
        return self._agent_sessions.get(agent_id)

    def list_agents(self) -> list[AgentEntry]:
        """List all active agents.

        Returns:
            List of agent entries
        """
        return self._scratchpad_manager.get_all_agents()

    async def update_agent_state(
        self,
        agent_id: str,
        state: AgentState,
        task: str | None = None,
    ) -> AgentEntry | None:
        """Update an agent's state.

        Args:
            agent_id: Agent ID
            state: New state
            task: Optional new task description

        Returns:
            Updated entry, or None if not found
        """
        entry = self._scratchpad_manager.update_agent(agent_id, state=state, task=task)
        if entry and agent_id in self._agents:
            self._agents[agent_id] = entry
        return entry

    async def send_message(
        self,
        sender: str,
        recipient: str,
        content: str,
        node_refs: list[str] | None = None,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a message between agents.

        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            content: Message content
            node_refs: Node IDs referenced in content
            reply_to: Message ID this is replying to
            metadata: Additional metadata

        Returns:
            Message ID
        """
        message = self._scratchpad_manager.send_message(
            sender=sender,
            recipient=recipient,
            content=content,
            node_refs=node_refs,
            reply_to=reply_to,
            metadata=metadata,
        )
        return message.id

    def get_messages(
        self,
        agent_id: str,
        status: str | None = "pending",
    ) -> list[AgentMessage]:
        """Get messages for an agent.

        Args:
            agent_id: Recipient agent ID
            status: Filter by status (None for all)

        Returns:
            List of matching messages
        """
        return self._scratchpad_manager.get_messages(agent_id, status)

    def mark_message_delivered(self, message_id: str) -> None:
        """Mark a message as delivered.

        Args:
            message_id: Message ID
        """
        self._scratchpad_manager.mark_message_status(message_id, "delivered")

    def mark_message_read(self, message_id: str) -> None:
        """Mark a message as read.

        Args:
            message_id: Message ID
        """
        self._scratchpad_manager.mark_message_status(message_id, "read")

    def share_node(self, node: ContextNode) -> str:
        """Register a node in the shared registry.

        Args:
            node: Node to share

        Returns:
            Node ID
        """
        self._shared_nodes[node.node_id] = node
        return node.node_id

    def get_shared_node(self, node_id: str) -> ContextNode | None:
        """Get a shared node by ID.

        Args:
            node_id: Node ID

        Returns:
            Node, or None if not found
        """
        return self._shared_nodes.get(node_id)

    def unshare_node(self, node_id: str) -> bool:
        """Remove a node from the shared registry.

        Args:
            node_id: Node ID

        Returns:
            True if removed, False if not found
        """
        if node_id in self._shared_nodes:
            del self._shared_nodes[node_id]
            return True
        return False

    async def pause_agent(self, agent_id: str) -> None:
        """Pause an agent.

        Args:
            agent_id: Agent ID
        """
        await self.update_agent_state(agent_id, AgentState.PAUSED)

    async def resume_agent(self, agent_id: str) -> None:
        """Resume a paused agent.

        Args:
            agent_id: Agent ID
        """
        await self.update_agent_state(agent_id, AgentState.RUNNING)

    async def terminate_agent(self, agent_id: str) -> None:
        """Terminate an agent.

        Cancels the agent's session and removes from registry.

        Args:
            agent_id: Agent ID
        """
        # Update state
        await self.update_agent_state(agent_id, AgentState.TERMINATED)

        # Cancel and close session
        session = self._agent_sessions.get(agent_id)
        if session:
            await session.cancel()
            await self._session_manager.close_session(session.session_id)

        # Remove from local caches
        self._agents.pop(agent_id, None)
        self._agent_sessions.pop(agent_id, None)

        # Unregister from scratchpad
        self._scratchpad_manager.unregister_agent(agent_id)

    def has_pending_messages(self, agent_id: str) -> bool:
        """Check if an agent has pending messages.

        Args:
            agent_id: Agent ID

        Returns:
            True if there are pending messages
        """
        messages = self.get_messages(agent_id, status="pending")
        return len(messages) > 0
