"""CAP Serialization -- pluggable wire format encoding.

Defines the CAPSerializer protocol and concrete implementations for
different wire encodings. Transports compose with serializers:

- StdioTransport + JsonSerializer = current behavior (JSON-RPC 2.0)
- StdioTransport + MessagePackSerializer = compact binary over stdio
- Future transports compose the same way

All implementations MUST support JsonSerializer (the required baseline).
MessagePackSerializer is optional and requires the ``msgpack`` package.

Content negotiation: the ``negotiate_serialization`` helper selects the
best serializer given host and server preferences (see CAP spec Section 5).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CAPSerializer",
    "JsonSerializer",
    "MessagePackSerializer",
    "negotiate_serialization",
]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CAPSerializer(Protocol):
    """Serialization protocol for CAP messages.

    Each serializer converts between Python dicts (the logical JSON-RPC
    message) and ``bytes`` suitable for transmission over a transport.
    """

    @property
    def format_id(self) -> str:
        """Format identifier for content negotiation.

        Must be one of ``"json"``, ``"msgpack"``, or ``"protobuf"``.
        """
        ...

    def encode_request(self, method: str, params: Any, id: int | str) -> bytes:
        """Encode a JSON-RPC 2.0 request message.

        Args:
            method: RPC method name.
            params: Parameters (dataclass, dict, or None).
            id: Request identifier for response correlation.

        Returns:
            Serialized bytes ready for transmission.
        """
        ...

    def encode_notification(self, method: str, params: Any) -> bytes:
        """Encode a JSON-RPC 2.0 notification (no id, no response).

        Args:
            method: RPC method name.
            params: Parameters (dataclass, dict, or None).

        Returns:
            Serialized bytes ready for transmission.
        """
        ...

    def encode_response(self, result: Any, id: int | str) -> bytes:
        """Encode a JSON-RPC 2.0 success response.

        Args:
            result: Result value (dataclass, dict, or primitive).
            id: Request identifier this responds to.

        Returns:
            Serialized bytes ready for transmission.
        """
        ...

    def encode_error(
        self,
        code: int,
        message: str,
        id: int | str | None,
        data: Any = None,
    ) -> bytes:
        """Encode a JSON-RPC 2.0 error response.

        Args:
            code: Error code (from ErrorCodes).
            message: Human-readable error description.
            id: Request identifier (None for parse errors).
            data: Optional additional error context.

        Returns:
            Serialized bytes ready for transmission.
        """
        ...

    def decode(self, data: bytes) -> dict[str, Any]:
        """Decode a serialized message back to a Python dict.

        Args:
            data: Raw bytes from the transport.

        Returns:
            The logical JSON-RPC message as a dict.

        Raises:
            ValueError: If the data cannot be decoded.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_params(params: Any) -> Any:
    """Convert dataclass params to a plain dict, passthrough otherwise."""
    if params is None:
        return None
    if hasattr(params, "__dataclass_fields__"):
        return asdict(params)
    return params


def _normalize_result(result: Any) -> Any:
    """Convert dataclass results to a plain dict, passthrough otherwise."""
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    return result


# ---------------------------------------------------------------------------
# JsonSerializer
# ---------------------------------------------------------------------------


class JsonSerializer:
    """JSON-RPC 2.0 serialization (required baseline).

    Produces compact JSON with no unnecessary whitespace, matching
    the existing behavior of StdioTransport's inline serialization.
    Uses the same separators ``(",", ":")`` for minimal output size.
    """

    @property
    def format_id(self) -> str:
        return "json"

    def encode_request(self, method: str, params: Any, id: int | str) -> bytes:
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": id,
        }
        normalized = _normalize_params(params)
        if normalized is not None:
            msg["params"] = normalized
        return json.dumps(msg, separators=(",", ":")).encode("utf-8")

    def encode_notification(self, method: str, params: Any) -> bytes:
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        normalized = _normalize_params(params)
        if normalized is not None:
            msg["params"] = normalized
        return json.dumps(msg, separators=(",", ":")).encode("utf-8")

    def encode_response(self, result: Any, id: int | str) -> bytes:
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": id,
            "result": _normalize_result(result),
        }
        return json.dumps(msg, separators=(",", ":")).encode("utf-8")

    def encode_error(
        self,
        code: int,
        message: str,
        id: int | str | None,
        data: Any = None,
    ) -> bytes:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": error,
            "id": id,
        }
        return json.dumps(msg, separators=(",", ":")).encode("utf-8")

    def decode(self, data: bytes) -> dict[str, Any]:
        try:
            result = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got {type(result).__name__}")
        return result


# ---------------------------------------------------------------------------
# MessagePackSerializer
# ---------------------------------------------------------------------------


class MessagePackSerializer:
    """MessagePack serialization (compact binary).

    Same logical structure as JSON but binary-encoded using the
    ``msgpack`` package. Produces smaller messages with lower parse
    overhead -- recommended for bandwidth-constrained or high-frequency
    sync scenarios.

    Raises:
        ImportError: If ``msgpack`` is not installed. Install with
            ``pip install msgpack`` or ``uv add msgpack``.
    """

    def __init__(self) -> None:
        try:
            import msgpack as _msgpack  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MessagePackSerializer requires the 'msgpack' package. "
                "Install it with: pip install msgpack (or: uv add msgpack)"
            ) from None
        self._msgpack = _msgpack

    @property
    def format_id(self) -> str:
        return "msgpack"

    def encode_request(self, method: str, params: Any, id: int | str) -> bytes:
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "id": id,
        }
        normalized = _normalize_params(params)
        if normalized is not None:
            msg["params"] = normalized
        result: bytes = self._msgpack.packb(msg, use_bin_type=True)
        return result

    def encode_notification(self, method: str, params: Any) -> bytes:
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        normalized = _normalize_params(params)
        if normalized is not None:
            msg["params"] = normalized
        result: bytes = self._msgpack.packb(msg, use_bin_type=True)
        return result

    def encode_response(self, result: Any, id: int | str) -> bytes:
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": id,
            "result": _normalize_result(result),
        }
        encoded: bytes = self._msgpack.packb(msg, use_bin_type=True)
        return encoded

    def encode_error(
        self,
        code: int,
        message: str,
        id: int | str | None,
        data: Any = None,
    ) -> bytes:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "error": error,
            "id": id,
        }
        encoded: bytes = self._msgpack.packb(msg, use_bin_type=True)
        return encoded

    def decode(self, data: bytes) -> dict[str, Any]:
        try:
            result = self._msgpack.unpackb(data, raw=False)
        except Exception as e:
            raise ValueError(f"Invalid MessagePack: {e}") from e
        if not isinstance(result, dict):
            raise ValueError(f"Expected MessagePack map, got {type(result).__name__}")
        return result


# ---------------------------------------------------------------------------
# Content negotiation
# ---------------------------------------------------------------------------

# Registry of known format IDs to their serializer constructors.
_SERIALIZER_REGISTRY: dict[str, type[JsonSerializer] | type[MessagePackSerializer]] = {
    "json": JsonSerializer,
    "msgpack": MessagePackSerializer,
}


def negotiate_serialization(
    host_preferred: str,
    server_supported: list[str],
) -> CAPSerializer:
    """Select the best serializer given host and server preferences.

    Implements the propose-accept model from CAP spec Section 5.3:

    1. If the host's preferred format is supported by the server AND is
       available in this implementation, use it.
    2. Otherwise, fall back to ``"json"`` (the universal baseline).

    Args:
        host_preferred: Format identifier the host wants (e.g. ``"msgpack"``).
        server_supported: Format identifiers the server supports.

    Returns:
        An instantiated CAPSerializer for the negotiated format.

    Raises:
        RuntimeError: If even the JSON fallback cannot be instantiated
            (should never happen in practice).
    """
    # Check if the host's preference is supported by both sides
    if host_preferred in server_supported and host_preferred in _SERIALIZER_REGISTRY:
        try:
            return _SERIALIZER_REGISTRY[host_preferred]()
        except ImportError:
            # Optional dependency not installed -- fall through to JSON
            pass

    # JSON is always the fallback
    return JsonSerializer()
