"""Tests for automatic trace generation via traceable fields.

Tests coverage for:
- src/activecontext/context/traceable.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

from activecontext.context.traceable import (
    TRACEABLE_KEY,
    TRACEABLE_FORMATTER_KEY,
    _EXCLUDED_FIELDS,
    format_value,
    get_field_formatter,
    get_traceable_fields,
    is_traceable,
    register_formatter,
    trace_all_fields,
    traceable,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class Status(Enum):
    """Test enum for traceable fields."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"


class MockGraph:
    """Mock graph for testing."""

    pass


@dataclass
class MockContextNode:
    """Mock ContextNode for testing traceable fields.

    Mimics the essential attributes of ContextNode that traceable checks.
    """

    node_id: str = "test_node"
    tracing: bool = True
    _graph: Any = None
    _mark_changed_calls: list = field(default_factory=list, repr=False)

    def _mark_changed(
        self,
        description: str = "",
        content: str | None = None,
        originator: str | None = None,
        field_name: str = "",
        prev_value: Any = "",
        curr_value: Any = "",
    ) -> None:
        """Track _mark_changed calls for verification."""
        self._mark_changed_calls.append(
            {
                "description": description,
                "content": content,
                "originator": originator,
                "field_name": field_name,
                "prev_value": prev_value,
                "curr_value": curr_value,
            }
        )


# =============================================================================
# format_value Tests
# =============================================================================


class TestFormatValue:
    """Tests for format_value function."""

    def test_format_none(self):
        """Test formatting None value."""
        assert format_value(None) == "None"

    def test_format_string(self):
        """Test formatting string value."""
        assert format_value("hello") == "hello"

    def test_format_int(self):
        """Test formatting integer value."""
        assert format_value(42) == "42"

    def test_format_float(self):
        """Test formatting float value."""
        assert format_value(3.14) == "3.14"

    def test_format_bool(self):
        """Test formatting boolean value."""
        assert format_value(True) == "True"
        assert format_value(False) == "False"

    def test_format_enum(self):
        """Test formatting enum value uses .value."""
        assert format_value(Status.RUNNING) == "running"

    def test_format_complex_object(self):
        """Test formatting complex object uses repr."""
        obj = {"key": "value"}
        result = format_value(obj)
        assert "key" in result
        assert "value" in result

    def test_format_long_repr_truncated(self):
        """Test that long repr is truncated to 50 chars."""
        long_list = list(range(100))
        result = format_value(long_list)
        assert len(result) <= 50


class TestRegisterFormatter:
    """Tests for custom type formatter registration."""

    def test_custom_formatter(self):
        """Test registering and using a custom formatter."""

        class CustomType:
            def __init__(self, value: int):
                self.value = value

        register_formatter(CustomType, lambda x: f"custom({x.value})")

        obj = CustomType(42)
        assert format_value(obj) == "custom(42)"


# =============================================================================
# traceable() Function Tests
# =============================================================================


class TestTraceableFunction:
    """Tests for traceable() helper function."""

    def test_returns_field_with_default(self):
        """Test that traceable() with default returns a field."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = traceable(default="pending")

        node = TestNode()
        assert node.status == "pending"

    def test_returns_field_with_default_factory(self):
        """Test that traceable() with default_factory returns a field."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            items: list = traceable(default_factory=list)

        node = TestNode()
        assert node.items == []
        # Verify each instance gets its own list
        node.items.append("a")
        node2 = TestNode()
        assert node2.items == []

    def test_field_has_traceable_metadata(self):
        """Test that traceable() sets the correct metadata."""
        from dataclasses import fields as dc_fields

        @dataclass
        class TestNode:
            status: str = traceable(default="pending")

        field_info = dc_fields(TestNode)[0]
        assert field_info.metadata.get(TRACEABLE_KEY) is True

    def test_field_has_formatter_metadata(self):
        """Test that traceable() sets formatter in metadata."""
        from dataclasses import fields as dc_fields

        fmt = lambda x: f"v={x}"

        @dataclass
        class TestNode:
            count: int = traceable(default=0, formatter=fmt)

        field_info = dc_fields(TestNode)[0]
        assert field_info.metadata.get(TRACEABLE_FORMATTER_KEY) is fmt


# =============================================================================
# is_traceable and get_traceable_fields Tests
# =============================================================================


class TestTraceableRegistry:
    """Tests for traceable field registry functions."""

    def test_is_traceable_returns_true_for_traceable_field(self):
        """Test is_traceable returns True for traceable fields."""

        @dataclass
        class TestNode(MockContextNode):
            status: str = traceable(default="pending")

        assert is_traceable(TestNode, "status") is True

    def test_is_traceable_returns_false_for_regular_field(self):
        """Test is_traceable returns False for regular fields."""

        @dataclass
        class TestNode(MockContextNode):
            name: str = "test"

        assert is_traceable(TestNode, "name") is False

    def test_get_traceable_fields(self):
        """Test get_traceable_fields returns all traceable fields."""

        @dataclass
        class TestNode(MockContextNode):
            status: str = traceable(default="pending")
            count: int = traceable(default=0)
            name: str = "test"  # Not traceable

        fields = get_traceable_fields(TestNode)
        assert "status" in fields
        assert "count" in fields
        assert "name" not in fields

    def test_get_field_formatter(self):
        """Test get_field_formatter returns the formatter."""
        fmt = lambda x: f"custom({x})"

        @dataclass
        class TestNode:
            status: str = traceable(default="pending", formatter=fmt)

        assert get_field_formatter(TestNode, "status") is fmt

    def test_get_field_formatter_returns_none_for_no_formatter(self):
        """Test get_field_formatter returns None when no formatter."""

        @dataclass
        class TestNode:
            status: str = traceable(default="pending")

        assert get_field_formatter(TestNode, "status") is None


# =============================================================================
# @trace_all_fields Decorator Tests
# =============================================================================


class TestTraceAllFieldsDecorator:
    """Tests for @trace_all_fields class decorator."""

    def test_traces_all_public_fields(self):
        """Test that all public fields are traced."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"
            count: int = 0

        node = TestNode()
        node._graph = MockGraph()
        node.status = "running"
        node.count = 5

        assert len(node._mark_changed_calls) == 2

    def test_traces_explicit_traceable_fields(self):
        """Test that traceable() fields are traced."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = traceable(default="pending")

        node = TestNode()
        node._graph = MockGraph()
        node.status = "running"

        assert len(node._mark_changed_calls) == 1
        call = node._mark_changed_calls[0]
        assert call["field_name"] == "status"
        assert call["prev_value"] == "pending"
        assert call["curr_value"] == "running"

    def test_uses_custom_formatter(self):
        """Test that custom formatter is used for traceable fields."""

        def custom_fmt(v: int) -> str:
            return f"count={v}"

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            count: int = traceable(default=0, formatter=custom_fmt)

        node = TestNode()
        node._graph = MockGraph()
        node.count = 5

        call = node._mark_changed_calls[0]
        assert call["prev_value"] == "count=0"
        assert call["curr_value"] == "count=5"

    def test_excludes_private_fields(self):
        """Test that underscore fields are not traced."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"
            _internal: str = "private"

        node = TestNode()
        node._graph = MockGraph()
        node._internal = "changed"

        # Only status would create trace, _internal should not
        assert all(
            "_internal" not in call["description"] for call in node._mark_changed_calls
        )

    def test_excludes_excluded_fields(self):
        """Test that _EXCLUDED_FIELDS are not traced."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            version: int = 0  # In _EXCLUDED_FIELDS
            status: str = "pending"

        node = TestNode()
        node._graph = MockGraph()
        node.version = 1
        node.status = "running"

        # Only status change should be traced
        assert len(node._mark_changed_calls) == 1
        assert "status" in node._mark_changed_calls[0]["description"]

    def test_no_trace_during_init(self):
        """Test that no traces during initialization."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"

        node = TestNode()
        node._graph = MockGraph()
        # No calls during init
        assert len(node._mark_changed_calls) == 0

    def test_no_trace_when_graph_none(self):
        """Test that no trace is created when _graph is None."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"

        node = TestNode()
        node.status = "running"
        assert len(node._mark_changed_calls) == 0

    def test_no_trace_when_tracing_disabled(self):
        """Test that no trace is created when tracing is False."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"

        node = TestNode()
        node._graph = MockGraph()
        node.tracing = False
        node.status = "running"
        assert len(node._mark_changed_calls) == 0

    def test_no_trace_when_value_unchanged(self):
        """Test that no trace when setting same value."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"

        node = TestNode()
        node._graph = MockGraph()
        node.status = "pending"  # Same as default

        assert len(node._mark_changed_calls) == 0

    def test_multiple_changes_create_multiple_traces(self):
        """Test that each change creates a trace."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: str = "pending"

        node = TestNode()
        node._graph = MockGraph()
        node.status = "running"
        node.status = "complete"

        assert len(node._mark_changed_calls) == 2

    def test_works_on_non_dataclass(self):
        """Test decorator returns unchanged class for non-dataclass."""

        @trace_all_fields
        class RegularClass:
            pass

        # Should not raise, just return unchanged
        assert RegularClass is not None

    def test_trace_with_enum(self):
        """Test trace formatting with enum values."""

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            status: Status = Status.PENDING

        node = TestNode()
        node._graph = MockGraph()
        node.status = Status.RUNNING

        assert len(node._mark_changed_calls) == 1
        call = node._mark_changed_calls[0]
        assert call["prev_value"] == "pending"
        assert call["curr_value"] == "running"


# =============================================================================
# _EXCLUDED_FIELDS Tests
# =============================================================================


class TestExcludedFields:
    """Tests for _EXCLUDED_FIELDS constant."""

    def test_contains_version_fields(self):
        """Test that version tracking fields are excluded."""
        assert "version" in _EXCLUDED_FIELDS
        assert "created_at" in _EXCLUDED_FIELDS
        assert "updated_at" in _EXCLUDED_FIELDS

    def test_contains_identity_fields(self):
        """Test that identity fields are excluded."""
        assert "node_id" in _EXCLUDED_FIELDS
        assert "parent_ids" in _EXCLUDED_FIELDS
        assert "children_ids" in _EXCLUDED_FIELDS

    def test_contains_internal_state(self):
        """Test that internal state fields are excluded."""
        assert "_graph" in _EXCLUDED_FIELDS
        assert "_last_trace" in _EXCLUDED_FIELDS


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestTraceableIntegration:
    """Integration tests combining traceable features."""

    def test_mixed_traceable_and_regular_fields(self):
        """Test class with both traceable and regular fields.

        With @trace_all_fields, all public fields are traced by default.
        Fields marked with traceable() can have custom formatters.
        """

        def custom_fmt(v: str) -> str:
            return f"traced({v})"

        @trace_all_fields
        @dataclass
        class TestNode(MockContextNode):
            traced_field: str = traceable(default="a", formatter=custom_fmt)
            regular_field: str = "b"

        node = TestNode()
        node._graph = MockGraph()

        node.traced_field = "x"
        node.regular_field = "y"

        # Both fields should create traces
        assert len(node._mark_changed_calls) == 2

        # traced_field uses custom formatter
        traced_call = next(
            c for c in node._mark_changed_calls if "traced_field" in c["description"]
        )
        assert traced_call["prev_value"] == "traced(a)"
        assert traced_call["curr_value"] == "traced(x)"

        # regular_field uses default formatter
        regular_call = next(
            c for c in node._mark_changed_calls if "regular_field" in c["description"]
        )
        assert regular_call["prev_value"] == "b"
        assert regular_call["curr_value"] == "y"

    def test_inheritance_with_trace_all_fields(self):
        """Test @trace_all_fields works with inheritance.

        Note: When both base and derived classes use @trace_all_fields,
        fields from the base class may be traced by both __setattr__ overrides.
        For best results, only apply the decorator to the final class.
        """

        @dataclass
        class BaseNode(MockContextNode):
            base_field: str = "base"

        @trace_all_fields
        @dataclass
        class DerivedNode(BaseNode):
            derived_field: str = "derived"

        node = DerivedNode()
        node._graph = MockGraph()

        node.base_field = "new_base"
        node.derived_field = "new_derived"

        # Both changes should be traced (once each)
        assert len(node._mark_changed_calls) == 2

    def test_trace_all_fields_on_derived_class(self):
        """Test @trace_all_fields on derived class only."""

        @dataclass
        class BaseNode(MockContextNode):
            base_field: str = "base"

        @trace_all_fields
        @dataclass
        class DerivedNode(BaseNode):
            derived_field: str = "derived"

        node = DerivedNode()
        node._graph = MockGraph()

        node.derived_field = "changed"

        # derived_field should be traced
        assert len(node._mark_changed_calls) == 1
        assert "derived_field" in node._mark_changed_calls[0]["description"]
