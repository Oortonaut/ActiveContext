"""Automatic trace generation for context node fields.

This module provides two mechanisms for automatic trace generation:

1. `traceable()` - Field-level annotation using dataclass field metadata:
   ```python
   @dataclass
   class MyNode(ContextNode):
       status: str = traceable(default="pending")
   ```

2. `@trace_all_fields` - Class decorator using __setattr__ override:
   ```python
   @trace_all_fields
   @dataclass
   class MyNode(ContextNode):
       status: str = "pending"
       count: int = 0
   ```

Both mechanisms automatically call `_mark_changed()` when a field value changes,
generating trace descriptions in the format "field_name: old_value -> new_value".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, field
from dataclasses import fields as dc_fields
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")

# Sentinel for traceable field metadata
TRACEABLE_KEY = "__traceable__"
TRACEABLE_FORMATTER_KEY = "__traceable_formatter__"

# Type formatters for trace descriptions
_type_formatters: dict[type, Callable[[Any], str]] = {}


def register_formatter(t: type, formatter: Callable[[Any], str]) -> None:
    """Register a formatter for a specific type.

    Formatters are used to convert values to strings for trace descriptions.
    They are checked in registration order, using isinstance().

    Args:
        t: The type to register a formatter for
        formatter: A callable that converts values of type t to strings
    """
    _type_formatters[t] = formatter


# Built-in formatters
register_formatter(Enum, lambda v: v.value if v is not None else "None")


def format_value(value: Any) -> str:
    """Format a value for trace description.

    Uses registered formatters for known types, falls back to str/repr.

    Args:
        value: The value to format

    Returns:
        String representation suitable for trace descriptions
    """
    if value is None:
        return "None"

    # Check for registered formatter (including parent types)
    for t, formatter in _type_formatters.items():
        if isinstance(value, t):
            return formatter(value)

    # Default: use str for simple types, repr for complex
    if isinstance(value, (str, int, float, bool)):
        return str(value)

    # Truncate long reprs
    return repr(value)[:50]


def traceable(
    default: T | type[MISSING] = MISSING,
    default_factory: Callable[[], T] | None = None,
    formatter: Callable[[T], str] | None = None,
    init: bool = True,
    repr: bool = True,
) -> T:
    """Mark a field as traceable with automatic change detection.

    Returns a dataclass field() with metadata marking it for auto-tracing.
    Use with @trace_all_fields decorator on the class to enable tracing.

    Usage:
        @trace_all_fields
        @dataclass
        class MyNode(ContextNode):
            expansion: Expansion = traceable(default=Expansion.ALL)
            children: list[str] = traceable(default_factory=list)

    Args:
        default: Default value for the field
        default_factory: Factory function for mutable defaults
        formatter: Custom function to format values for trace description
        init: Whether to include in __init__ (passed to field())
        repr: Whether to include in __repr__ (passed to field())

    Returns:
        A dataclass field with traceable metadata
    """
    metadata = {TRACEABLE_KEY: True}
    if formatter is not None:
        metadata[TRACEABLE_FORMATTER_KEY] = formatter

    if default_factory is not None:
        return field(  # type: ignore
            default_factory=default_factory,
            metadata=metadata,
            init=init,
            repr=repr,
        )
    elif default is not MISSING:
        return field(  # type: ignore
            default=default,
            metadata=metadata,
            init=init,
            repr=repr,
        )
    else:
        # No default - field is required
        return field(  # type: ignore
            metadata=metadata,
            init=init,
            repr=repr,
        )


def is_traceable(cls: type, field_name: str) -> bool:
    """Check if a field is marked as traceable.

    Args:
        cls: The class to check
        field_name: The field name to check

    Returns:
        True if the field has traceable metadata
    """
    try:
        for f in dc_fields(cls):
            if f.name == field_name:
                return f.metadata.get(TRACEABLE_KEY, False)
    except TypeError:
        pass
    return False


def get_traceable_fields(cls: type) -> set[str]:
    """Get all traceable fields for a class.

    Args:
        cls: The class to get traceable fields for

    Returns:
        Set of field names that are traceable
    """
    result: set[str] = set()
    try:
        for f in dc_fields(cls):
            if f.metadata.get(TRACEABLE_KEY, False):
                result.add(f.name)
    except TypeError:
        pass
    return result


def get_field_formatter(cls: type, field_name: str) -> Callable[[Any], str] | None:
    """Get the custom formatter for a traceable field, if any.

    Args:
        cls: The class containing the field
        field_name: The field name

    Returns:
        The custom formatter function, or None
    """
    try:
        for f in dc_fields(cls):
            if f.name == field_name:
                return f.metadata.get(TRACEABLE_FORMATTER_KEY)
    except TypeError:
        pass
    return None


# Fields that should never be auto-traced by @trace_all_fields
_EXCLUDED_FIELDS = frozenset(
    {
        # Identity fields - changes here are structural, not state changes
        "node_id",
        "parent_ids",
        "children_ids",
        "child_order",
        # Version tracking - updated by _mark_changed itself
        "version",
        "created_at",
        "updated_at",
        # Internal state
        "_graph",
        "_on_child_changed_hook",
        "_last_trace",
        "_last_trace_time",
        "_cached_children_tokens",
        # References
        "trace_sink",
        "content_id",
        "display_sequence",
    }
)


def trace_all_fields(cls: type[T]) -> type[T]:
    """Class decorator to auto-trace all mutable fields.

    Adds a __setattr__ override that automatically traces field changes.

    Traces all fields that:
    - Have traceable() metadata (always traced with custom formatter if provided)
    - Are public (don't start with underscore)
    - Are not in _EXCLUDED_FIELDS

    Usage:
        @trace_all_fields
        @dataclass
        class ShellNode(ContextNode):
            command: str = ""
            shell_status: ShellStatus = ShellStatus.PENDING
            # This field uses custom formatter
            count: int = traceable(default=0, formatter=lambda x: f"n={x}")

    Args:
        cls: The class to decorate

    Returns:
        The decorated class with __setattr__ override
    """
    # Get dataclass fields
    try:
        cls_fields = dc_fields(cls)
    except TypeError:
        # Not a dataclass, return unchanged
        return cls

    # Get original __setattr__ or use object's
    original_setattr = cls.__setattr__ if hasattr(cls, "__setattr__") else object.__setattr__

    # Build traced fields set and formatter map
    traced_fields: set[str] = set()
    field_formatters: dict[str, Callable[[Any], str]] = {}

    for f in cls_fields:
        name = f.name
        is_explicit_traceable = f.metadata.get(TRACEABLE_KEY, False)

        # Always include explicitly traceable fields
        if is_explicit_traceable:
            traced_fields.add(name)
            custom_fmt = f.metadata.get(TRACEABLE_FORMATTER_KEY)
            if custom_fmt is not None:
                field_formatters[name] = custom_fmt
            continue

        # Skip excluded fields for implicit tracing
        if name.startswith("_") or name in _EXCLUDED_FIELDS:
            continue

        traced_fields.add(name)

    def traced_setattr(self: Any, name: str, value: Any) -> None:
        """__setattr__ override that auto-traces field changes."""
        if name in traced_fields and hasattr(self, "_graph"):
            old_value = getattr(self, name, MISSING)
            original_setattr(self, name, value)

            if (
                old_value is not MISSING
                and old_value != value
                and getattr(self, "tracing", True)
                and getattr(self, "_graph", None) is not None
            ):
                # Use custom formatter if available
                fmt = field_formatters.get(name, format_value)
                old_str = fmt(old_value)
                new_str = fmt(value)
                self._mark_changed(
                    description=f"{name}: {old_str} -> {new_str}",
                    field_name=name,
                    prev_value=old_str,
                    curr_value=new_str,
                )
        else:
            original_setattr(self, name, value)

    cls.__setattr__ = traced_setattr  # type: ignore
    return cls
