"""Agent node for representing agents in the context DAG.

This module defines AgentNode for agent awareness - showing the agent its own
identity and information about other agents (parent, children, peers).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from activecontext.agents.schema import AgentState
from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency

from .base import ContextNode
from .enums import AgentRelation


@dataclass(kw_only=True)
class AgentNode(ContextNode):
    """Represents an agent in the context (self or another agent).

    Used for agent awareness - showing the agent its own identity and
    information about other agents (parent, children, peers).

    Attributes:
        agent_id: The agent's ID
        agent_type: Type of agent (explorer, summarizer, etc.)
        relation: Relationship to viewing agent ("self", "parent", "child", "peer")
        task: The agent's task description
        agent_state: Current state (spawned, running, waiting, etc.)
        session_id: Underlying session ID
        message_count: Number of pending messages for this agent
    """

    agent_id: str = ""
    agent_type: str = "default"
    relation: AgentRelation = AgentRelation.SELF
    task: str = ""
    agent_state: AgentState = AgentState.RUNNING
    session_id: str = ""
    message_count: int = 0

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "relation": self.relation.value,
            "task": self.task,
            "agent_state": self.agent_state.value,
            "message_count": self.message_count,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render type, state, task, and messages."""
        parts: list[str] = []
        parts.append(f"  Type: {self.agent_type}\n")
        parts.append(f"  State: {self.agent_state.value}\n")

        if self.task:
            parts.append(f"  Task: {self.task}\n")

        if self.message_count > 0:
            parts.append(f"  Messages: {self.message_count} pending\n")

        return "".join(parts)

    def update_state(self, agent_state: AgentState) -> AgentNode:
        """Update the agent's state."""
        old_state = self.agent_state
        self.agent_state = agent_state
        if old_state != agent_state:
            self.mark_changed(f"Agent state: {old_state.value} -> {agent_state.value}")
        return self

    def update_message_count(self, count: int) -> AgentNode:
        """Update pending message count."""
        old_count = self.message_count
        self.message_count = count
        if old_count != count:
            self.mark_changed(f"Messages: {old_count} -> {count}")
        return self

    def render_digest(self) -> str:
        """Return 'Agent: id [STATE] Nm' format."""
        return f"Agent: {self.agent_id} [{self.agent_state.value.upper()}] {self.message_count}m"

    def to_dict(self) -> dict[str, Any]:
        """Serialize AgentNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "relation": self.relation.value,
                "task": self.task,
                "agent_state": self.agent_state.value,
                "session_id": self.session_id,
                "message_count": self.message_count,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AgentNode:
        """Deserialize AgentNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "all")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", "default"),
            relation=AgentRelation(data.get("relation", "self")),
            task=data.get("task", ""),
            agent_state=AgentState(data.get("agent_state", "running")),
            session_id=data.get("session_id", ""),
            message_count=data.get("message_count", 0),
        )
        return node
