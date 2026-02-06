"""PtyNode - Interactive PTY session management."""

from __future__ import annotations

import re as _re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from activecontext.context.graph import LinkedChildOrder
from activecontext.context.state import Expansion, TickFrequency
from activecontext.context.traceable import trace_all_fields

from .base import ContextNode


class PtyStatus(Enum):
    """Status of an interactive PTY session."""

    PENDING = "pending"  # Created, not yet spawned
    RUNNING = "running"  # PTY process is alive
    EXITED = "exited"  # Process exited normally
    KILLED = "killed"  # Force-killed or signalled
    ERROR = "error"  # Failed to spawn or internal error


# Max lines / bytes kept in the PTY scrollback ring buffer.
_PTY_MAX_LINES = 500
_PTY_MAX_BYTES = 100_000

# Regex that matches a single ANSI/VT100 escape sequence.
_ANSI_RE = _re.compile(
    r"\x1b(?:\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]|\][^\x07]*\x07|[\(\)][AB012])"
)


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


@trace_all_fields
@dataclass(kw_only=True)
class PtyNode(ContextNode):
    """Represents a long-lived interactive PTY session.

    The ``pty()`` DSL function creates a PtyNode and spawns the process in
    the background.  Output arrives incrementally via ``append_output()``
    (called by PtyManager at tick boundaries after Nagle batching).

    The scrollback is a ring buffer capped at ``_PTY_MAX_LINES`` lines and
    ``_PTY_MAX_BYTES`` total bytes.  ``render_content()`` returns the most
    recent ~50 lines with ANSI escapes stripped so the projection stays
    compact and token-friendly.

    Attributes:
        command: The command (e.g., ``"gdb"``, ``"python"``)
        args: Command arguments
        pty_status: Current lifecycle status
        exit_code: Process exit code (None until exited)
        signal: Signal name if killed (e.g., ``"SIGTERM"``)
        input_history: Lines the agent has sent via ``pty_send``
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    pty_status: PtyStatus = PtyStatus.PENDING
    exit_code: int | None = None
    signal: str | None = None

    # Input tracking — records what the agent has sent
    input_history: list[str] = field(default_factory=list)

    # Ring-buffer scrollback (not serialized; rebuilt from output on load)
    _scrollback_lines: list[str] = field(default_factory=list, init=False, repr=False)
    _scrollback_bytes: int = field(default=0, init=False, repr=False)
    _total_line_count: int = field(default=0, init=False, repr=False)

    # Raw output accumulator (serialized for session persistence)
    _raw_output: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        # PTY nodes default to CONTENT expansion (scrollback only, not ALL)
        if self.default_expansion == Expansion.ALL:
            object.__setattr__(self, "default_expansion", Expansion.CONTENT)

    # -- Node identity --------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True when the PTY session has ended."""
        return self.pty_status in (PtyStatus.EXITED, PtyStatus.KILLED, PtyStatus.ERROR)

    @property
    def error(self) -> str | None:
        """Error message if PTY failed, None otherwise."""
        if self.pty_status == PtyStatus.ERROR:
            return "PTY error"
        if self.pty_status == PtyStatus.KILLED and self.signal:
            return f"Killed by {self.signal}"
        return None

    @property
    def full_command(self) -> str:
        if self.args:
            return f"{self.command} {' '.join(self.args)}"
        return self.command

    # -- Output management (ring buffer) --------------------------------------

    def append_output(self, text: str) -> None:
        """Append new output text to the scrollback ring buffer.

        Splits on newlines, trims oldest lines when limits are exceeded,
        and calls ``mark_changed`` once per batch.
        """
        if not text:
            return

        self._raw_output += text

        new_lines = text.split("\n")
        # If the last element is empty it means text ended with \\n;
        # don't add an extra blank line.
        if new_lines and new_lines[-1] == "":
            new_lines.pop()

        for line in new_lines:
            self._scrollback_lines.append(line)
            self._scrollback_bytes += len(line) + 1  # +1 for implicit newline
            self._total_line_count += 1

        # Trim ring buffer
        while (
            len(self._scrollback_lines) > _PTY_MAX_LINES or self._scrollback_bytes > _PTY_MAX_BYTES
        ) and self._scrollback_lines:
            dropped = self._scrollback_lines.pop(0)
            self._scrollback_bytes -= len(dropped) + 1

        self.mark_changed(
            f"PTY output ({len(new_lines)} lines)",
            content=text[:500],
        )

    # -- Input tracking -------------------------------------------------------

    def record_input(self, text: str) -> None:
        """Record an input line sent by the agent."""
        self.input_history.append(text)

    # -- Lifecycle transitions ------------------------------------------------

    def set_running(self) -> PtyNode:
        """PENDING → RUNNING (called when backend.spawn() succeeds)."""
        self.pty_status = PtyStatus.RUNNING
        self.mark_changed(f"PTY '{self.command}' running")
        return self

    def set_exited(self, code: int, signal_name: str | None = None) -> PtyNode:
        """Mark the session as exited.

        Args:
            code: Process exit code.
            signal_name: Signal name if killed externally.
        """
        self.exit_code = code
        self.signal = signal_name
        if signal_name:
            self.pty_status = PtyStatus.KILLED
        elif code == 0:
            self.pty_status = PtyStatus.EXITED
        else:
            self.pty_status = PtyStatus.EXITED

        self.mark_changed(
            f"PTY '{self.command}' {self.pty_status.value} (exit={code})",
        )
        return self

    def set_error(self, message: str) -> PtyNode:
        """Mark the session as failed with an error message."""
        self.pty_status = PtyStatus.ERROR
        self.append_output(f"\n[ERROR] {message}\n")
        self.mark_changed(f"PTY '{self.command}' error: {message}")
        return self

    # -- Rendering ------------------------------------------------------------

    def GetDigest(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "type": self.node_type,
            "command": self.full_command,
            "status": self.pty_status.value,
            "exit_code": self.exit_code,
            "lines": self._total_line_count,
            "expansion": self.default_expansion.value,
            "mode": self.mode,
            "version": self.version,
        }

    def render_digest(self) -> str:
        cmd_display = (
            self.full_command[:40] + "..." if len(self.full_command) > 40 else self.full_command
        )
        return f"PTY: {cmd_display} [{self.pty_status.value.upper()}]"

    def get_wake_data(self) -> dict[str, Any]:
        """Return data dict for wake prompt template formatting."""
        return {
            "node_id": self.node_id,
            "command": self.full_command,
            "exit_code": self.exit_code,
            "status": self.pty_status.value,
        }

    def render_content(self) -> str:
        """Render the most recent scrollback lines (ANSI-stripped)."""
        # Show last 50 lines of scrollback
        tail = self._scrollback_lines[-50:]
        cleaned = [_strip_ansi(line) for line in tail]
        content = "\n".join(cleaned)
        if content and not content.endswith("\n"):
            content += "\n"

        # Append status footer
        if self.is_complete:
            content += f"--- exit_code={self.exit_code}"
            if self.signal:
                content += f", signal={self.signal}"
            content += " ---\n"
        elif self.pty_status == PtyStatus.RUNNING:
            hidden = self._total_line_count - len(tail)
            if hidden > 0:
                content += f"--- {hidden} earlier lines omitted ---\n"

        return content

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "command": self.command,
                "args": self.args,
                "pty_status": self.pty_status.value,
                "exit_code": self.exit_code,
                "signal": self.signal,
                "input_history": self.input_history,
                "raw_output": self._raw_output,
            }
        )
        return data

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> PtyNode:
        tick_freq = None
        if data.get("tick_frequency"):
            tick_freq = TickFrequency.from_dict(data["tick_frequency"])

        node = cls(
            node_id=data["node_id"],
            parent_ids=set(data.get("parent_ids", [])),
            child_order=LinkedChildOrder.from_list(
                data.get("child_order") or data.get("children_ids") or []
            ),
            default_expansion=Expansion(data.get("expansion", "content")),
            mode=data.get("mode", "paused"),
            tick_frequency=tick_freq,
            version=data.get("version", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            display_sequence=data.get("display_sequence"),
            originator=data.get("originator"),
            title=data.get("title", ""),
            command=data.get("command", ""),
            args=data.get("args", []),
            pty_status=PtyStatus(data.get("pty_status", "pending")),
            exit_code=data.get("exit_code"),
            signal=data.get("signal"),
            input_history=data.get("input_history", []),
        )
        # Rebuild scrollback from persisted raw output
        raw = data.get("raw_output", "")
        if raw:
            # Bypass mark_changed during deserialization
            node._raw_output = raw
            lines = raw.split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            node._scrollback_lines = lines[-_PTY_MAX_LINES:]
            node._scrollback_bytes = sum(len(ln) + 1 for ln in node._scrollback_lines)
            node._total_line_count = len(lines)
        return node
