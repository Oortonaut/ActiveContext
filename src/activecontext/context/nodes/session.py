"""SessionNode - Session-level metadata for agent situational awareness."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class SessionNode(ContextNode):
    """Represents session-level metadata for agent situational awareness.

    This node is auto-created as a root node in each session and placed early
    in the context projection. It provides the LLM agent with visibility into:
    - Recent token usage trends (to adjust verbosity)
    - Execution timing statistics (to notice slow operations)
    - Context graph composition (what's currently loaded)
    - Cumulative session statistics

    The node updates automatically each tick with TickFrequency.turn().

    Attributes:
        token_history: Token counts for last N turns
        token_min: Minimum tokens in history
        token_max: Maximum tokens in history
        token_avg: Average tokens in history
        turn_durations_ms: Execution time for last N turns
        time_min_ms: Minimum turn duration
        time_max_ms: Maximum turn duration
        total_statements_executed: Cumulative statement count
        total_tokens_consumed: Cumulative token usage
        session_start_time: Unix timestamp when session started
        node_count_by_type: Current graph composition by node type
        running_node_count: Number of nodes in "running" mode
        graph_depth: Maximum depth of the context graph
        recent_actions: Short descriptions of last N actions
    """

    # Token tracking (rolling window)
    token_history: list[int] = field(default_factory=list)
    token_min: int = 0
    token_max: int = 0
    token_avg: float = 0.0

    # Timing records (rolling window)
    turn_durations_ms: list[float] = field(default_factory=list)
    time_min_ms: float = 0.0
    time_max_ms: float = 0.0

    # Cumulative stats
    total_statements_executed: int = 0
    total_tokens_consumed: int = 0
    session_start_time: float = field(default_factory=time.time)
    turn_count: int = 0

    # Context graph snapshot
    node_count_by_type: dict[str, int] = field(default_factory=dict)
    running_node_count: int = 0
    graph_depth: int = 0

    # Recent actions (short descriptions)
    recent_actions: list[str] = field(default_factory=list)

    # Configuration
    history_depth: int = 10  # How many turns to track

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens_consumed,
            "total_statements": self.total_statements_executed,
            "running_nodes": self.running_node_count,
            "graph_depth": self.graph_depth,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render turn info, token info, graph stats, and recent actions."""
        parts: list[str] = []

        # Turn and timing info
        if self.turn_durations_ms:
            last_duration = self.turn_durations_ms[-1]
            parts.append(
                f"Turn: {self.turn_count} | Duration: {last_duration:.0f}ms "
                f"(avg: {sum(self.turn_durations_ms) / len(self.turn_durations_ms):.0f}ms, "
                f"range: {self.time_min_ms:.0f}-{self.time_max_ms:.0f}ms)\n"
            )
        else:
            parts.append(f"Turn: {self.turn_count}\n")

        # Token info
        if self.token_history:
            last_tokens = self.token_history[-1]
            parts.append(
                f"Tokens: {last_tokens:,} this turn "
                f"(avg: {self.token_avg:,.0f}, range: {self.token_min:,}-{self.token_max:,})\n"
            )
        parts.append(
            f"Total: {self.total_tokens_consumed:,} tokens across {self.turn_count} turns\n"
        )

        # Context graph summary
        if self.node_count_by_type:
            parts.append("\nContext Graph:\n")
            for node_type, count in sorted(self.node_count_by_type.items()):
                running_info = ""
                if node_type == "text" and self.running_node_count > 0:
                    running_info = f" ({self.running_node_count} running)"
                parts.append(f"  {node_type}s: {count}{running_info}\n")
            if self.graph_depth > 0:
                parts.append(f"  depth: {self.graph_depth}\n")

        # Recent actions
        if self.recent_actions:
            parts.append("\nRecent Actions:\n")
            for i, action in enumerate(self.recent_actions[-5:]):
                turn_idx = self.turn_count - (len(self.recent_actions[-5:]) - i - 1)
                parts.append(f"  [T{turn_idx}] {action}\n")

        return "".join(parts)

    def record_turn(
        self,
        tokens_used: int,
        duration_ms: float,
        action_description: str | None = None,
    ) -> SessionNode:
        """Record statistics for a completed turn.

        Args:
            tokens_used: Tokens consumed this turn
            duration_ms: Turn execution time in milliseconds
            action_description: Optional short description of the turn's main action
        """
        self.turn_count += 1
        self.total_tokens_consumed += tokens_used

        # Update token history (rolling window)
        self.token_history.append(tokens_used)
        if len(self.token_history) > self.history_depth:
            self.token_history = self.token_history[-self.history_depth :]

        # Update token stats
        if self.token_history:
            self.token_min = min(self.token_history)
            self.token_max = max(self.token_history)
            self.token_avg = sum(self.token_history) / len(self.token_history)

        # Update timing history (rolling window)
        self.turn_durations_ms.append(duration_ms)
        if len(self.turn_durations_ms) > self.history_depth:
            self.turn_durations_ms = self.turn_durations_ms[-self.history_depth :]

        # Update timing stats
        if self.turn_durations_ms:
            self.time_min_ms = min(self.turn_durations_ms)
            self.time_max_ms = max(self.turn_durations_ms)

        # Record action
        if action_description:
            self.recent_actions.append(action_description)
            if len(self.recent_actions) > self.history_depth:
                self.recent_actions = self.recent_actions[-self.history_depth :]

        desc = f"Turn {self.turn_count}: {tokens_used} tokens"
        if action_description:
            desc += f" - {action_description[:50]}"
        self.mark_changed(desc)
        return self

    def record_statement(self) -> SessionNode:
        """Record that a statement was executed."""
        self.total_statements_executed += 1
        return self

    def update_graph_stats(self) -> SessionNode:
        """Update context graph statistics from the attached graph."""
        if not self._graph:
            return self

        # Count nodes by type
        type_counts: dict[str, int] = {}
        running_count = 0
        max_depth = 0

        for node_id in self._graph._nodes:
            node = self._graph.get_node(node_id)
            if node:
                node_type = node.node_type
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
                if node.mode == "running":
                    running_count += 1

        # Calculate graph depth via BFS from roots
        root_nodes = self._graph.get_roots()
        if root_nodes:
            visited: set[str] = set()
            queue: deque[tuple[str, int]] = deque((r.node_id, 1) for r in root_nodes)
            while queue:
                nid, depth = queue.popleft()
                if nid in visited:
                    continue
                visited.add(nid)
                max_depth = max(max_depth, depth)
                node = self._graph.get_node(nid)
                if node:
                    for child_id in node.child_order:
                        if child_id not in visited:
                            queue.append((child_id, depth + 1))

        self.node_count_by_type = type_counts
        self.running_node_count = running_count
        self.graph_depth = max_depth
        return self

    def Recompute(self) -> None:
        """Recompute graph statistics on tick."""
        self.update_graph_stats()
        super().Recompute()

    def render_digest(self) -> str:
        """Return 'Session: Turn N | N tokens' format."""
        return f"Session: Turn {self.turn_count} | {self.total_tokens_consumed:,} tokens"

    def to_dict(self) -> dict[str, Any]:
        """Serialize SessionNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "token_history": self.token_history,
                "token_min": self.token_min,
                "token_max": self.token_max,
                "token_avg": self.token_avg,
                "turn_durations_ms": self.turn_durations_ms,
                "time_min_ms": self.time_min_ms,
                "time_max_ms": self.time_max_ms,
                "total_statements_executed": self.total_statements_executed,
                "total_tokens_consumed": self.total_tokens_consumed,
                "session_start_time": self.session_start_time,
                "turn_count": self.turn_count,
                "node_count_by_type": self.node_count_by_type,
                "running_node_count": self.running_node_count,
                "graph_depth": self.graph_depth,
                "recent_actions": self.recent_actions,
                "history_depth": self.history_depth,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SessionNode:
        """Deserialize SessionNode from dict."""
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
            mode=data.get("mode", "running"),  # Default to running for session node
            tick_frequency=tick_freq or TickFrequency.turn(),
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            token_history=data.get("token_history", []),
            token_min=data.get("token_min", 0),
            token_max=data.get("token_max", 0),
            token_avg=data.get("token_avg", 0.0),
            turn_durations_ms=data.get("turn_durations_ms", []),
            time_min_ms=data.get("time_min_ms", 0.0),
            time_max_ms=data.get("time_max_ms", 0.0),
            total_statements_executed=data.get("total_statements_executed", 0),
            total_tokens_consumed=data.get("total_tokens_consumed", 0),
            session_start_time=data.get("session_start_time", time.time()),
            turn_count=data.get("turn_count", 0),
            node_count_by_type=data.get("node_count_by_type", {}),
            running_node_count=data.get("running_node_count", 0),
            graph_depth=data.get("graph_depth", 0),
            recent_actions=data.get("recent_actions", []),
            history_depth=data.get("history_depth", 10),
        )
        return node
