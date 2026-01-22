"""Session replay from JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from acp_debug.transport.stdio import JsonRpcMessage


@dataclass
class RecordedMessage:
    """A recorded message from a session."""

    timestamp: float
    direction: str  # "c2a" or "a2c"
    message: JsonRpcMessage


class SessionPlayer:
    """Replays recorded ACP sessions from JSONL.

    Usage:
        player = SessionPlayer(Path("session.jsonl"))
        for record in player:
            print(f"{record.direction}: {record.message.method}")
    """

    def __init__(self, source: Path | IO[str]) -> None:
        """Initialize player.

        Args:
            source: Path to JSONL file or file-like object
        """
        self._source: IO[str]
        self._owns_file = False

        if isinstance(source, Path):
            self._source = open(source, encoding="utf-8")
            self._owns_file = True
        else:
            self._source = source

    def __iter__(self) -> Iterator[RecordedMessage]:
        """Iterate over recorded messages."""
        for line in self._source:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                yield RecordedMessage(
                    timestamp=data.get("ts", 0),
                    direction=data.get("dir", "c2a"),
                    message=JsonRpcMessage.from_dict(data.get("msg", {})),
                )
            except json.JSONDecodeError:
                continue

    def messages(self) -> list[RecordedMessage]:
        """Load all messages into memory."""
        return list(self)

    def filter_direction(self, direction: str) -> Iterator[RecordedMessage]:
        """Filter messages by direction.

        Args:
            direction: "c2a" or "a2c"
        """
        for record in self:
            if record.direction == direction:
                yield record

    def filter_method(self, method: str) -> Iterator[RecordedMessage]:
        """Filter messages by method name."""
        for record in self:
            if record.message.method == method:
                yield record

    def close(self) -> None:
        """Close the player."""
        if self._owns_file:
            self._source.close()

    def __enter__(self) -> SessionPlayer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


async def replay_to_agent(
    player: SessionPlayer,
    use_hooks: bool = True,
) -> None:
    """Replay recorded session to an agent.

    Args:
        player: Session player to read from
        use_hooks: Whether to process messages through extension chain
    """
    # TODO: Implement replay functionality
    pass
