"""Exposed marker -- identifies agent-facing API surface.

This module provides a decorator and utility for marking methods and properties
as part of the agent-facing API. The help system uses this to determine which
members to document.

Usage:
    from activecontext.context.exposed import exposed, get_exposed

    class MyNode(ContextNode):
        @exposed
        def SetState(self, state: Expansion) -> MyNode:
            '''Set the node's expansion state.'''
            ...

        @exposed
        @property
        def is_complete(self) -> bool:
            '''Whether this node has completed.'''
            ...

    # Get all exposed member names
    names = get_exposed(MyNode)  # {"SetState", "is_complete"}
"""

from __future__ import annotations

from typing import Any

EXPOSED_ATTR = "_exposed"


def exposed(obj: Any) -> Any:
    """Decorator marking a method/property as agent-facing.

    Can be applied to methods or properties. For properties, the
    decorator should be applied before @property::

        @property
        @exposed
        def is_complete(self) -> bool:
            ...

    Or after @property (the fget function is marked internally)::

        @exposed
        @property
        def is_complete(self) -> bool:
            ...

    Args:
        obj: The method or property to mark.

    Returns:
        The same object, with _exposed attribute set to True.
    """
    if isinstance(obj, property):
        # Cannot set attributes on property objects directly,
        # so mark the underlying fget function instead
        if obj.fget is not None:
            setattr(obj.fget, EXPOSED_ATTR, True)
        return obj
    setattr(obj, EXPOSED_ATTR, True)
    return obj


def is_exposed(obj: Any) -> bool:
    """Check if an object is marked as exposed.

    Handles properties by checking the underlying fget function.

    Args:
        obj: The object to check.

    Returns:
        True if the object is marked as exposed.
    """
    if isinstance(obj, property):
        return getattr(obj.fget, EXPOSED_ATTR, False)
    return getattr(obj, EXPOSED_ATTR, False)


def get_exposed(cls: type) -> set[str]:
    """Get all exposed member names from a class.

    Inspects the class hierarchy (including inherited members) and returns
    the names of all members marked with @exposed.

    Args:
        cls: The class to inspect.

    Returns:
        Set of member names that are marked as exposed.
    """
    result: set[str] = set()
    for name in dir(cls):
        if name.startswith("_"):
            continue
        # Use class-level lookup to get descriptors (properties)
        for klass in cls.__mro__:
            if name in klass.__dict__:
                member = klass.__dict__[name]
                if is_exposed(member):
                    result.add(name)
                break
        else:
            # Fallback: getattr for inherited attributes
            member = getattr(cls, name, None)
            if member and is_exposed(member):
                result.add(name)
    return result
