"""Session recording to JSONL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import IO, Any

from acp_debug.transport.stdio import JsonRpcMessage


class SessionRecorder:
    """Records ACP messages to JSONL format.

    Format:
        {"ts": 1706000000.123, "dir": "c2a", "msg": {...}}

    Where:
        - ts: Unix timestamp with millisecond precision
        - dir: Direction ("c2a" = client→agent, "a2c" = agent→client)
        - msg: Raw JSON-RPC message
    """

    def __init__(self, output: Path | IO[str]) -> None:
        """Initialize recorder.

        Args:
            output: Path to output file or file-like object
        """
        self._output: IO[str]
        self._owns_file = False

        if isinstance(output, Path):
            self._output = open(output, "w", encoding="utf-8")
            self._owns_file = True
        else:
            self._output = output

    def record(self, direction: str, msg: JsonRpcMessage) -> None:
        """Record a message.

        Args:
            direction: "c2a" (client→agent) or "a2c" (agent→client)
            msg: The JSON-RPC message
        """
        record = {
            "ts": time.time(),
            "dir": direction,
            "msg": msg.to_dict(),
        }
        self._output.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._output.flush()

    def record_raw(self, direction: str, data: dict[str, Any]) -> None:
        """Record a raw message dictionary.

        Args:
            direction: "c2a" or "a2c"
            data: Raw message dictionary
        """
        record = {
            "ts": time.time(),
            "dir": direction,
            "msg": data,
        }
        self._output.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._output.flush()

    def close(self) -> None:
        """Close the recorder."""
        if self._owns_file:
            self._output.close()

    def __enter__(self) -> SessionRecorder:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
