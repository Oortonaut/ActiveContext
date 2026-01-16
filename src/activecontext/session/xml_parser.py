"""XML tag parser for alternative command encoding.

Some LLM models may prefer XML-style tags over Python syntax.
This module converts XML tags to equivalent Python statements.

Syntax examples:
    <!-- Object constructors (name becomes variable) -->
    <view name="v" path="main.py" tokens="2000"/>
    <group name="g" tokens="300" lod="2">
        <member ref="v"/>
        <member ref="w"/>
    </group>
    <topic name="t" title="Feature X" tokens="1000"/>
    <artifact name="a" type="code" content="..." language="python"/>

    <!-- Method calls (self refers to variable) -->
    <SetLod self="v" level="1"/>
    <SetTokens self="v" count="500"/>
    <Run self="v" freq="Sync"/>

    <!-- Utility functions -->
    <ls/>
    <show self="v"/>
    <done message="Task complete"/>

    <!-- DAG manipulation -->
    <link child="v" parent="g"/>
    <unlink child="v" parent="g"/>
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

# Tags that create objects and bind to a variable name
CONSTRUCTOR_TAGS = {"view", "group", "topic", "artifact"}

# Tags that are utility functions (no self, no variable binding)
UTILITY_TAGS = {"ls", "show", "done", "link", "unlink"}

# Map XML attribute names to Python argument names for constructors
CONSTRUCTOR_ARG_MAP: dict[str, dict[str, str]] = {
    "view": {
        "path": "__positional__",  # First positional arg
    },
    "group": {},
    "topic": {
        "title": "__positional__",
    },
    "artifact": {
        "type": "__positional__",  # Maps to artifact_type
    },
}

# Attributes to exclude from kwargs (handled specially)
SPECIAL_ATTRS = {"name", "self", "ref"}


def is_xml_command(source: str) -> bool:
    """Check if the source looks like an XML command.

    Returns True if the source starts with '<' and appears to be
    a valid XML element (not a Python less-than comparison).
    """
    source = source.strip()
    if not source.startswith("<"):
        return False

    # Quick heuristics to distinguish XML from Python
    # XML: <tag ...> or <tag/>
    # Python: x < y, x <= y, x << y
    return bool(re.match(r"^<[a-zA-Z_][a-zA-Z0-9_]*[\s/>]", source))


def parse_xml_to_python(source: str) -> str:
    """Convert XML command(s) to Python statement(s).

    Args:
        source: XML source string (single element or multiple)

    Returns:
        Equivalent Python statement(s)

    Raises:
        ValueError: If XML is malformed or unsupported
    """
    source = source.strip()

    # Wrap in root element if multiple top-level elements
    # (XML requires single root)
    tag_count = source.count("<")
    if (
        tag_count > 1
        and not source.startswith("<commands")
        and not (source.endswith("/>") and tag_count == 1)
    ):
        source = f"<commands>{source}</commands>"

    try:
        root = ET.fromstring(source)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}") from e

    # If wrapped, process children; otherwise process the single element
    if root.tag == "commands":
        statements = [_element_to_python(child) for child in root]
    else:
        statements = [_element_to_python(root)]

    return "\n".join(statements)


def _element_to_python(elem: ET.Element) -> str:
    """Convert a single XML element to a Python statement."""
    tag = elem.tag
    attrs = dict(elem.attrib)

    # Constructor tags: <view name="v" path="main.py" .../>
    if tag in CONSTRUCTOR_TAGS:
        return _constructor_to_python(tag, attrs, elem)

    # Utility tags: <ls/>, <done message="..."/>, <link child="v" parent="g"/>
    if tag in UTILITY_TAGS:
        return _utility_to_python(tag, attrs)

    # Method calls: <SetLod self="v" level="1"/>
    # (PascalCase or any other tag with 'self' attribute)
    if "self" in attrs:
        return _method_call_to_python(tag, attrs)

    raise ValueError(f"Unknown XML tag: <{tag}>. Expected constructor, utility, or method call.")


def _constructor_to_python(tag: str, attrs: dict[str, str], elem: ET.Element) -> str:
    """Convert constructor tag to Python assignment."""
    # Extract variable name
    var_name = attrs.pop("name", None)
    if not var_name:
        raise ValueError(f"<{tag}> requires 'name' attribute for variable binding")

    # Build argument list
    args: list[str] = []
    kwargs: list[str] = []

    # Handle positional argument if defined
    arg_map = CONSTRUCTOR_ARG_MAP.get(tag, {})
    for xml_attr, py_arg in arg_map.items():
        if xml_attr in attrs:
            val = attrs.pop(xml_attr)
            if py_arg == "__positional__":
                args.append(_format_value(val))

    # Handle <member ref="..."/> children for group
    if tag == "group":
        for child in elem:
            if child.tag == "member" and "ref" in child.attrib:
                # Member refs are variable names, not strings
                args.append(child.attrib["ref"])

    # Remaining attributes become kwargs
    for key, val in attrs.items():
        if key not in SPECIAL_ATTRS:
            kwargs.append(f"{key}={_format_value(val)}")

    # Build call
    all_args = ", ".join(args + kwargs)
    return f"{var_name} = {tag}({all_args})"


def _utility_to_python(tag: str, attrs: dict[str, str]) -> str:
    """Convert utility function tag to Python call."""
    # Special handling for show which takes self
    if tag == "show" and "self" in attrs:
        target = attrs.pop("self")
        kwargs = [f"{k}={_format_value(v)}" for k, v in attrs.items()]
        args_str = ", ".join([target] + kwargs)
        return f"show({args_str})"

    # link/unlink take child and parent as positional (variable refs)
    if tag in ("link", "unlink"):
        child = attrs.get("child")
        parent = attrs.get("parent")
        if not child or not parent:
            raise ValueError(f"<{tag}> requires 'child' and 'parent' attributes")
        return f"{tag}({child}, {parent})"

    # ls, done - all attrs are kwargs
    kwargs = [f"{k}={_format_value(v)}" for k, v in attrs.items() if k not in SPECIAL_ATTRS]
    return f"{tag}({', '.join(kwargs)})"


def _method_call_to_python(tag: str, attrs: dict[str, str]) -> str:
    """Convert method call tag to Python method invocation."""
    target = attrs.pop("self")

    # Build kwargs from remaining attributes
    kwargs: list[str] = []
    for key, val in attrs.items():
        if key not in SPECIAL_ATTRS:
            kwargs.append(f"{key}={_format_value(val)}")

    return f"{target}.{tag}({', '.join(kwargs)})"


def _format_value(val: str) -> str:
    """Format a value for Python code.

    Attempts to preserve type:
    - Numbers remain unquoted
    - Booleans (true/false) become True/False
    - Everything else is quoted as string
    """
    # Check for boolean
    if val.lower() == "true":
        return "True"
    if val.lower() == "false":
        return "False"

    # Check for integer
    if re.match(r"^-?\d+$", val):
        return val

    # Check for float
    if re.match(r"^-?\d+\.\d+$", val):
        return val

    # String - use repr for proper escaping
    return repr(val)
