"""Tests for CAP Serialization abstraction.

Covers:
- JsonSerializer encode/decode for all message types
- Round-trip: encode then decode produces equivalent message
- CAPSerializer runtime_checkable protocol conformance
- MessagePackSerializer not-installed error handling
- negotiate_serialization logic
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from activecontext.plugins.serialization import (
    CAPSerializer,
    JsonSerializer,
    MessagePackSerializer,
    negotiate_serialization,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class SampleParams:
    """A sample dataclass for testing encode with dataclass params."""

    node_id: str = "shell_1"
    calls: list[Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []


@dataclass
class SampleResult:
    """A sample dataclass for testing encode with dataclass results."""

    status: str = "ok"
    value: int = 42


# ---------------------------------------------------------------------------
# JsonSerializer tests
# ---------------------------------------------------------------------------


class TestJsonSerializer:
    """JsonSerializer encode/decode for all message types."""

    def setup_method(self) -> None:
        self.s = JsonSerializer()

    # -- format_id --

    def test_format_id(self) -> None:
        assert self.s.format_id == "json"

    # -- encode_request --

    def test_encode_request_with_dict_params(self) -> None:
        data = self.s.encode_request("node/create", {"node_type": "shell"}, id=1)
        msg = json.loads(data)
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "node/create"
        assert msg["id"] == 1
        assert msg["params"]["node_type"] == "shell"

    def test_encode_request_with_dataclass_params(self) -> None:
        params = SampleParams(node_id="sh_42", calls=[])
        data = self.s.encode_request("node/sync", params, id=5)
        msg = json.loads(data)
        assert msg["method"] == "node/sync"
        assert msg["id"] == 5
        assert msg["params"]["node_id"] == "sh_42"
        assert msg["params"]["calls"] == []

    def test_encode_request_with_none_params(self) -> None:
        data = self.s.encode_request("test", None, id=1)
        msg = json.loads(data)
        assert "params" not in msg

    def test_encode_request_with_string_id(self) -> None:
        data = self.s.encode_request("test", {"x": 1}, id="abc-123")
        msg = json.loads(data)
        assert msg["id"] == "abc-123"

    # -- encode_notification --

    def test_encode_notification_with_dict_params(self) -> None:
        data = self.s.encode_notification("shutdown", {"reason": "done"})
        msg = json.loads(data)
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "shutdown"
        assert msg["params"]["reason"] == "done"
        assert "id" not in msg

    def test_encode_notification_with_none_params(self) -> None:
        data = self.s.encode_notification("shutdown", None)
        msg = json.loads(data)
        assert msg["method"] == "shutdown"
        assert "params" not in msg

    def test_encode_notification_with_dataclass_params(self) -> None:
        params = SampleParams(node_id="sh_1")
        data = self.s.encode_notification("node/dirty", params)
        msg = json.loads(data)
        assert msg["params"]["node_id"] == "sh_1"
        assert "id" not in msg

    # -- encode_response --

    def test_encode_response_with_dict_result(self) -> None:
        data = self.s.encode_response({"status": "ok"}, id=3)
        msg = json.loads(data)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 3
        assert msg["result"]["status"] == "ok"
        assert "error" not in msg

    def test_encode_response_with_dataclass_result(self) -> None:
        result = SampleResult(status="complete", value=100)
        data = self.s.encode_response(result, id=7)
        msg = json.loads(data)
        assert msg["result"]["status"] == "complete"
        assert msg["result"]["value"] == 100

    def test_encode_response_with_none_result(self) -> None:
        data = self.s.encode_response(None, id=10)
        msg = json.loads(data)
        assert msg["result"] is None

    def test_encode_response_with_primitive_result(self) -> None:
        data = self.s.encode_response("hello", id=11)
        msg = json.loads(data)
        assert msg["result"] == "hello"

    # -- encode_error --

    def test_encode_error_basic(self) -> None:
        data = self.s.encode_error(-32000, "Node not found", id=5)
        msg = json.loads(data)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 5
        assert msg["error"]["code"] == -32000
        assert msg["error"]["message"] == "Node not found"
        assert "data" not in msg["error"]

    def test_encode_error_with_data(self) -> None:
        data = self.s.encode_error(-32001, "Unknown type", id=6, data={"node_type": "foobar"})
        msg = json.loads(data)
        assert msg["error"]["data"]["node_type"] == "foobar"

    def test_encode_error_with_none_id(self) -> None:
        """Parse errors use null id per JSON-RPC spec."""
        data = self.s.encode_error(-32700, "Parse error", id=None)
        msg = json.loads(data)
        assert msg["id"] is None

    # -- decode --

    def test_decode_valid_json(self) -> None:
        raw = b'{"jsonrpc":"2.0","method":"test","id":1}'
        msg = self.s.decode(raw)
        assert msg["method"] == "test"
        assert msg["id"] == 1

    def test_decode_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            self.s.decode(b"not json {{{")

    def test_decode_non_object_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            self.s.decode(b"[1, 2, 3]")

    # -- Compact output --

    def test_compact_encoding(self) -> None:
        """Ensure no unnecessary whitespace in output."""
        data = self.s.encode_request("test", {"key": "value"}, id=1)
        text = data.decode("utf-8")
        assert " " not in text  # compact separators, no spaces


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """Encode then decode produces equivalent message."""

    def setup_method(self) -> None:
        self.s = JsonSerializer()

    def test_round_trip_request(self) -> None:
        original_params = {"node_type": "shell", "args": ["pytest", "-v"]}
        encoded = self.s.encode_request("node/create", original_params, id=42)
        decoded = self.s.decode(encoded)
        assert decoded["jsonrpc"] == "2.0"
        assert decoded["method"] == "node/create"
        assert decoded["id"] == 42
        assert decoded["params"] == original_params

    def test_round_trip_notification(self) -> None:
        original_params = {"node_id": "shell_abc"}
        encoded = self.s.encode_notification("node/dirty", original_params)
        decoded = self.s.decode(encoded)
        assert decoded["method"] == "node/dirty"
        assert decoded["params"] == original_params
        assert "id" not in decoded

    def test_round_trip_response(self) -> None:
        original_result = {"state": {"is_complete": True}, "renders": {}}
        encoded = self.s.encode_response(original_result, id=99)
        decoded = self.s.decode(encoded)
        assert decoded["id"] == 99
        assert decoded["result"] == original_result

    def test_round_trip_error(self) -> None:
        encoded = self.s.encode_error(-32602, "Invalid params", id=7, data={"field": "node_type"})
        decoded = self.s.decode(encoded)
        assert decoded["id"] == 7
        assert decoded["error"]["code"] == -32602
        assert decoded["error"]["message"] == "Invalid params"
        assert decoded["error"]["data"]["field"] == "node_type"

    def test_round_trip_dataclass_params(self) -> None:
        params = SampleParams(node_id="sh_round", calls=[])
        encoded = self.s.encode_request("sync", params, id=10)
        decoded = self.s.decode(encoded)
        assert decoded["params"]["node_id"] == "sh_round"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestCAPSerializerProtocol:
    """CAPSerializer is runtime_checkable and satisfied by implementations."""

    def test_json_serializer_is_cap_serializer(self) -> None:
        s = JsonSerializer()
        assert isinstance(s, CAPSerializer)

    def test_protocol_is_runtime_checkable(self) -> None:
        """Verify the protocol decorator works at runtime."""

        class NotASerializer:
            pass

        assert not isinstance(NotASerializer(), CAPSerializer)

    def test_partial_implementation_not_conformant(self) -> None:
        """A class with only format_id does not satisfy the protocol."""

        class Partial:
            @property
            def format_id(self) -> str:
                return "partial"

        assert not isinstance(Partial(), CAPSerializer)


# ---------------------------------------------------------------------------
# MessagePackSerializer tests
# ---------------------------------------------------------------------------


class TestMessagePackSerializer:
    """MessagePack serializer tests -- import error and (when available) encode/decode."""

    def test_import_error_without_msgpack(self) -> None:
        """If msgpack is not installed, instantiation raises ImportError."""
        with patch.dict(sys.modules, {"msgpack": None}):
            with pytest.raises(ImportError, match="msgpack"):
                MessagePackSerializer()

    def test_import_error_message_is_helpful(self) -> None:
        """Error message tells the user how to install."""
        with patch.dict(sys.modules, {"msgpack": None}):
            with pytest.raises(ImportError, match="pip install msgpack"):
                MessagePackSerializer()


class TestMessagePackSerializerWithMsgpack:
    """Tests that run only when msgpack is installed."""

    @pytest.fixture(autouse=True)
    def _require_msgpack(self) -> None:
        pytest.importorskip("msgpack")

    def setup_method(self) -> None:
        self.s = MessagePackSerializer()

    def test_format_id(self) -> None:
        assert self.s.format_id == "msgpack"

    def test_is_cap_serializer(self) -> None:
        assert isinstance(self.s, CAPSerializer)

    def test_encode_decode_request(self) -> None:
        encoded = self.s.encode_request("node/sync", {"node_id": "sh_1"}, id=5)
        assert isinstance(encoded, bytes)
        decoded = self.s.decode(encoded)
        assert decoded["jsonrpc"] == "2.0"
        assert decoded["method"] == "node/sync"
        assert decoded["id"] == 5
        assert decoded["params"]["node_id"] == "sh_1"

    def test_encode_decode_notification(self) -> None:
        encoded = self.s.encode_notification("node/dirty", {"node_id": "x"})
        decoded = self.s.decode(encoded)
        assert decoded["method"] == "node/dirty"
        assert "id" not in decoded

    def test_encode_decode_response(self) -> None:
        result = {"state": {"running": True}, "tokens": {"collapsed": 10}}
        encoded = self.s.encode_response(result, id=3)
        decoded = self.s.decode(encoded)
        assert decoded["result"] == result
        assert decoded["id"] == 3

    def test_encode_decode_error(self) -> None:
        encoded = self.s.encode_error(-32000, "Not found", id=8, data={"id": "x"})
        decoded = self.s.decode(encoded)
        assert decoded["error"]["code"] == -32000
        assert decoded["error"]["data"]["id"] == "x"

    def test_decode_invalid_data_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid MessagePack"):
            self.s.decode(b"\xff\xfe invalid msgpack data")

    def test_round_trip_with_dataclass(self) -> None:
        params = SampleParams(node_id="mp_test", calls=[])
        encoded = self.s.encode_request("sync", params, id=20)
        decoded = self.s.decode(encoded)
        assert decoded["params"]["node_id"] == "mp_test"

    def test_none_params_omitted(self) -> None:
        encoded = self.s.encode_request("test", None, id=1)
        decoded = self.s.decode(encoded)
        assert "params" not in decoded

    def test_compact_vs_json(self) -> None:
        """MessagePack should produce smaller output than JSON for same message."""
        json_s = JsonSerializer()
        params = {"node_id": "shell_1", "calls": [], "cwd": "/home/user/project"}
        json_bytes = json_s.encode_request("node/sync", params, id=100)
        msgpack_bytes = self.s.encode_request("node/sync", params, id=100)
        assert len(msgpack_bytes) < len(json_bytes)


# ---------------------------------------------------------------------------
# Content negotiation tests
# ---------------------------------------------------------------------------


class TestNegotiateSerialization:
    """negotiate_serialization logic."""

    def test_json_when_both_support_json(self) -> None:
        s = negotiate_serialization("json", ["json"])
        assert isinstance(s, JsonSerializer)
        assert s.format_id == "json"

    def test_json_fallback_when_server_lacks_preference(self) -> None:
        """If host wants msgpack but server only has json, fall back to json."""
        s = negotiate_serialization("msgpack", ["json"])
        assert isinstance(s, JsonSerializer)

    def test_json_fallback_for_unknown_format(self) -> None:
        """Unknown format identifier falls back to json."""
        s = negotiate_serialization("protobuf", ["protobuf", "json"])
        # protobuf is not in the registry, so falls back to json
        assert isinstance(s, JsonSerializer)

    def test_json_fallback_when_host_preference_empty(self) -> None:
        s = negotiate_serialization("", ["json"])
        assert isinstance(s, JsonSerializer)

    def test_json_fallback_when_server_list_empty(self) -> None:
        s = negotiate_serialization("json", [])
        assert isinstance(s, JsonSerializer)

    def test_msgpack_when_both_support(self) -> None:
        """If msgpack is available and both support it, use it."""
        pytest.importorskip("msgpack")
        s = negotiate_serialization("msgpack", ["json", "msgpack"])
        assert isinstance(s, MessagePackSerializer)
        assert s.format_id == "msgpack"

    def test_msgpack_fallback_when_not_installed(self) -> None:
        """If msgpack requested but package not installed, fall back to json."""
        with patch.dict(sys.modules, {"msgpack": None}):
            s = negotiate_serialization("msgpack", ["json", "msgpack"])
            assert isinstance(s, JsonSerializer)

    def test_json_is_always_the_default(self) -> None:
        """Even with no preferences, json is returned."""
        s = negotiate_serialization("", [])
        assert isinstance(s, JsonSerializer)
