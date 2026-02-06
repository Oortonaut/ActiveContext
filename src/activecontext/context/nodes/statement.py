"""StatementNode and StatementResultNode - REPL statement tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


@trace_all_fields
@dataclass(kw_only=True)
class StatementNode(ContextNode):
    """A Python statement in the REPL timeline.

    Represents an executable Python statement, similar to how MCPServerNode
    represents an MCP connection. StatementResultNode children capture
    execution results.

    Attributes:
        statement_id: Unique identifier for this statement
        source: Python source code
        index: Position in timeline (0-based)
        timestamp: Unix timestamp when created
        status: "pending", "ok", or "error"
    """

    statement_id: str = ""
    source: str = ""
    index: int = 0
    timestamp: float = 0.0
    status: str = "pending"

    # Child tracking (MCPServerNode pattern)
    _result_nodes: dict[str, str] = field(default_factory=dict)  # exec_id → node_id

    default_hidden: bool = False

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "statement_id": self.statement_id,
            "index": self.index,
            "status": self.status,
            "source_length": len(self.source),
            "result_count": len(self._result_nodes),
            "expansion": self.default_expansion.value,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render statement source code."""
        lines = ["```python", self.source, "```"]
        return "\n".join(lines)

    def render_digest(self) -> str:
        """Return 'statement #N: STATUS: source_preview' format."""
        # Truncate long source for digest
        source_preview = self.source[:40]
        if len(self.source) > 40:
            source_preview += "..."
        return f"statement #{self.index}: [{self.status}] {source_preview}"

    def add_result(self, result_node_id: str, execution_id: str) -> None:
        """Track a result node as a child.

        Args:
            result_node_id: Node ID of the StatementResultNode
            execution_id: Unique execution ID
        """
        self._result_nodes[execution_id] = result_node_id

    def to_dict(self) -> dict[str, Any]:
        """Serialize StatementNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "statement_id": self.statement_id,
                "source": self.source,
                "index": self.index,
                "timestamp": self.timestamp,
                "status": self.status,
                "_result_nodes": self._result_nodes,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> StatementNode:
        """Deserialize StatementNode from dict."""
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
            statement_id=data.get("statement_id", ""),
            source=data.get("source", ""),
            index=data.get("index", 0),
            timestamp=data.get("timestamp", 0.0),
            status=data.get("status", "pending"),
        )
        node._result_nodes = data.get("_result_nodes", {})
        return node


@trace_all_fields
@dataclass(kw_only=True)
class StatementResultNode(ContextNode):
    """Result of executing a statement.

    Child node of StatementNode, holding execution outcome including
    stdout, stderr, exceptions, and namespace changes.

    Attributes:
        execution_id: Unique identifier for this execution
        statement_id: Back-reference to parent statement
        status: "ok", "error", "timeout", or "cancelled"
        stdout: Captured standard output
        stderr: Captured standard error
        exception: Exception details if status is "error"
        state_trace: Namespace changes (added/changed/deleted vars)
        duration_ms: Execution time in milliseconds
    """

    execution_id: str = ""
    statement_id: str = ""
    status: str = "ok"
    stdout: str = ""
    stderr: str = ""
    exception: dict[str, Any] | None = None
    state_trace: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    default_hidden: bool = False

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "execution_id": self.execution_id,
            "statement_id": self.statement_id,
            "status": self.status,
            "has_stdout": bool(self.stdout),
            "has_stderr": bool(self.stderr),
            "has_exception": self.exception is not None,
            "duration_ms": self.duration_ms,
            "expansion": self.default_expansion.value,
            "version": self.version,
        }

    def render_content(self) -> str:
        """Render execution result."""
        lines: list[str] = []

        if self.stdout:
            lines.append(f"stdout: {self.stdout}")
        if self.stderr:
            lines.append(f"stderr: {self.stderr}")
        if self.exception:
            exc_msg = self.exception.get("message", "Unknown error")
            exc_type = self.exception.get("type", "Error")
            lines.append(f"{exc_type}: {exc_msg}")
        if self.state_trace:
            added = self.state_trace.get("added", {})
            if added:
                lines.append(f"created: {', '.join(added.keys())}")
            changed = self.state_trace.get("changed", {})
            if changed:
                lines.append(f"changed: {', '.join(changed.keys())}")

        return "\n".join(lines) or "ok"

    def render_digest(self) -> str:
        """Return 'result: STATUS (duration)' format."""
        duration_str = f" ({self.duration_ms:.1f}ms)" if self.duration_ms > 0 else ""
        return f"result: [{self.status}]{duration_str}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize StatementResultNode to dict."""
        data = super().to_dict()
        data.update(
            {
                "execution_id": self.execution_id,
                "statement_id": self.statement_id,
                "status": self.status,
                "stdout": self.stdout,
                "stderr": self.stderr,
                "exception": self.exception,
                "state_trace": self.state_trace,
                "duration_ms": self.duration_ms,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> StatementResultNode:
        """Deserialize StatementResultNode from dict."""
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        return cls(
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
            execution_id=data.get("execution_id", ""),
            statement_id=data.get("statement_id", ""),
            status=data.get("status", "ok"),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exception=data.get("exception"),
            state_trace=data.get("state_trace", {}),
            duration_ms=data.get("duration_ms", 0.0),
        )
