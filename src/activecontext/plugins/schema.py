"""CAP Schema Discovery — introspection, validation, and conversion.

This module provides utilities for working with node type schemas:

- **Introspection**: Extract a NodeTypeSchema from a Python ContextNode
  subclass by inspecting its dataclass fields, properties, and methods.
- **JSON Schema conversion**: Convert between CAP schema format and
  JSON Schema (shared with MCP tool discovery).
- **Validation**: Check schemas for completeness and consistency.
- **DSL generation**: Produce Python-style signatures for LLM prompts.

Usage:
    from activecontext.plugins.schema import introspect_node_class

    schema = introspect_node_class(ShellNode)
    # schema.node_type == "shell"
    # schema.constructor.positional == [ParamSchema(name="command", ...)]
    # schema.properties == [PropertySchema(name="is_complete", ...), ...]
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, get_type_hints

from activecontext.plugins.wire import (
    MISSING,
    ConstructorSchema,
    MethodSchema,
    NodeTypeSchema,
    ParamSchema,
    PropertySchema,
)

# Fields inherited from ContextNode that are NOT part of a plugin's
# public API.  These are managed by the host (graph structure, rendering
# state, lifecycle metadata) and should be excluded from schemas.
_BASE_FIELDS: frozenset[str] = frozenset(
    {
        # Identity / graph structure
        "node_id",
        "parent_ids",
        "child_order",
        # Rendering / view state (owned by NodeView, not the plugin)
        "expansion",
        "mode",
        "tick_frequency",
        # Versioning
        "version",
        "created_at",
        "updated_at",
        # Metadata managed by host
        "tags",
        "originator",
        "title",
        "display_sequence",
        "content_id",
        # Notifications
        "notification_level",
        "is_subscription_point",
        # Tracing
        "tracing",
    }
)

# Python type → schema type string mapping
_TYPE_MAP: dict[type, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    bytes: "bytes",
    list: "list",
    dict: "dict",
    type(None): "None",
}


def _type_to_str(tp: Any) -> str:
    """Convert a Python type annotation to a schema type string."""
    if tp is inspect.Parameter.empty or tp is Any:
        return "Any"

    # Direct match
    if tp in _TYPE_MAP:
        return _TYPE_MAP[tp]

    # String annotations (from __future__ annotations)
    if isinstance(tp, str):
        return tp

    # typing generics (list[str], dict[str, Any], etc.)
    origin = getattr(tp, "__origin__", None)
    args = getattr(tp, "__args__", None)

    if origin is not None:
        origin_name = _TYPE_MAP.get(origin, getattr(origin, "__name__", str(origin)))
        if args:
            arg_strs = ", ".join(_type_to_str(a) for a in args)
            return f"{origin_name}[{arg_strs}]"
        return str(origin_name)

    # Fallback: use __name__ or str()
    return getattr(tp, "__name__", str(tp))


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def introspect_node_class(cls: type) -> NodeTypeSchema:
    """Extract a NodeTypeSchema from a ContextNode subclass.

    Inspects the class's dataclass fields (excluding base ContextNode
    fields), public properties, and public methods to build a complete
    schema.

    Args:
        cls: A ContextNode subclass (must be a dataclass).

    Returns:
        NodeTypeSchema with constructor, properties, and methods populated.

    Raises:
        TypeError: If cls is not a dataclass.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")

    # Get node_type from class
    node_type = _get_node_type(cls)

    # Get type hints for the class
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    # Extract constructor params from dataclass fields
    constructor = _extract_constructor(cls, hints)

    # Extract readable/writable properties
    properties = _extract_properties(cls, hints)

    # Extract public methods
    methods = _extract_methods(cls)

    # Description from class docstring
    description = (cls.__doc__ or "").strip().split("\n")[0]

    return NodeTypeSchema(
        node_type=node_type,
        description=description,
        constructor=constructor,
        properties=properties,
        methods=methods,
    )


def _get_node_type(cls: type) -> str:
    """Get node_type from a class, either from property or instance."""
    # Check if it's a property that returns a constant
    for klass in cls.__mro__:
        if "node_type" in klass.__dict__:
            attr = klass.__dict__["node_type"]
            if isinstance(attr, property) and attr.fget is not None:
                # Try to get the return value from source inspection
                try:
                    source = inspect.getsource(attr.fget)
                    # Look for 'return "something"' pattern
                    for line in source.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("return "):
                            val = stripped[7:].strip().strip("\"'")
                            if val and val.isidentifier():
                                return val
                except (OSError, TypeError):
                    pass
            break

    # Fallback: try instantiating with defaults
    try:
        instance: Any = object.__new__(cls)
        if hasattr(instance, "node_type"):
            return str(instance.node_type)
    except Exception:
        pass

    # Last resort: derive from class name
    name = cls.__name__
    if name.endswith("Node"):
        name = name[:-4]
    # CamelCase to snake_case
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)


def _extract_constructor(cls: type, hints: dict[str, Any]) -> ConstructorSchema:
    """Extract constructor schema from dataclass fields."""
    fields = dataclasses.fields(cls)
    positional: list[ParamSchema] = []
    named: list[ParamSchema] = []

    for f in fields:
        if f.name in _BASE_FIELDS or f.name.startswith("_"):
            continue

        type_str = _type_to_str(hints.get(f.name, "Any"))

        # Determine if the field has a default
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            default = MISSING

        param = ParamSchema(
            name=f.name,
            type=type_str,
            default=default,
        )

        if default is MISSING:
            positional.append(param)
        else:
            named.append(param)

    return ConstructorSchema(positional=positional, named=named)


def _extract_properties(cls: type, hints: dict[str, Any]) -> list[PropertySchema]:
    """Extract public properties from a class."""
    properties: list[PropertySchema] = []

    for name in dir(cls):
        if name.startswith("_") or name in _BASE_FIELDS:
            continue

        attr = getattr(cls, name, None)
        if not isinstance(attr, property):
            continue

        type_str = _type_to_str(hints.get(name, "Any"))
        readable = attr.fget is not None
        writable = attr.fset is not None
        doc = (attr.fget.__doc__ or "").strip() if attr.fget else ""

        properties.append(
            PropertySchema(
                name=name,
                type=type_str,
                readable=readable,
                writable=writable,
                description=doc,
            )
        )

    return sorted(properties, key=lambda p: p.name)


def _extract_methods(cls: type) -> list[MethodSchema]:
    """Extract public DSL-callable methods from a class.

    Includes methods that:
    - Are public (no _ prefix)
    - Are not inherited from object or ContextNode base
    - Are not dunder methods
    - Are not classmethods or staticmethods
    """
    methods: list[MethodSchema] = []

    # Methods to exclude (from ContextNode protocol / base)
    excluded = {
        "render_header",
        "render_content",
        "render_digest",
        "get_token_breakdown",
        "get_digest",
        "tick",
        "notify_parents",
        "to_dict",
        "from_dict",
        # PascalCase originals
        "Recompute",
        "GetDigest",
        "Render",
    }

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if name in excluded:
            continue

        # Skip methods defined on object
        if hasattr(object, name):
            continue

        sig = inspect.signature(method)
        params: list[ParamSchema] = []
        returns = "None"

        for pname, param in sig.parameters.items():
            if pname == "self":
                continue

            type_str = _type_to_str(param.annotation)

            if param.default is inspect.Parameter.empty:
                default: Any = MISSING
            else:
                default = param.default

            params.append(
                ParamSchema(
                    name=pname,
                    type=type_str,
                    default=default,
                )
            )

        # Check return annotation
        if sig.return_annotation is not inspect.Signature.empty:
            ret = sig.return_annotation
            # Chainable if returns Self or the class name
            ret_str = _type_to_str(ret)
            returns = ret_str
        else:
            returns = "None"

        doc = (method.__doc__ or "").strip().split("\n")[0]

        methods.append(
            MethodSchema(
                name=name,
                params=params,
                returns=returns,
                description=doc,
            )
        )

    return sorted(methods, key=lambda m: m.name)


# ---------------------------------------------------------------------------
# JSON Schema conversion (MCP-compatible)
# ---------------------------------------------------------------------------


def to_json_schema(schema: NodeTypeSchema) -> dict[str, Any]:
    """Convert a NodeTypeSchema to JSON Schema format.

    Produces a schema compatible with MCP tool input_schema format:
    {
        "type": "object",
        "properties": { ... },
        "required": [ ... ]
    }

    This maps the constructor parameters to JSON Schema properties,
    matching how MCP tools describe their inputs.

    Args:
        schema: The node type schema to convert.

    Returns:
        JSON Schema dict suitable for MCP tool registration.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Positional params are required
    for param in schema.constructor.positional:
        properties[param.name] = _param_to_json_schema(param)
        required.append(param.name)

    # Named params have defaults (optional)
    for param in schema.constructor.named:
        prop = _param_to_json_schema(param)
        if param.default is not MISSING and param.default is not None:
            prop["default"] = _serialize_default(param.default)
        properties[param.name] = prop

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required

    return result


# CAP type string → JSON Schema type
_JSON_SCHEMA_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "bytes": "string",
    "None": "null",
    "Any": "object",
}


def _param_to_json_schema(param: ParamSchema) -> dict[str, Any]:
    """Convert a ParamSchema to a JSON Schema property."""
    prop: dict[str, Any] = {}

    # Map type
    base_type = param.type.split("[")[0]  # strip generics
    json_type = _JSON_SCHEMA_TYPE_MAP.get(base_type)

    if json_type:
        prop["type"] = json_type
    elif base_type == "list":
        prop["type"] = "array"
        # Try to extract item type from list[X]
        if "[" in param.type and "]" in param.type:
            inner = param.type[param.type.index("[") + 1 : param.type.rindex("]")]
            inner_type = _JSON_SCHEMA_TYPE_MAP.get(inner)
            if inner_type:
                prop["items"] = {"type": inner_type}
    elif base_type == "dict":
        prop["type"] = "object"
    else:
        prop["type"] = "string"  # safe fallback

    if param.description:
        prop["description"] = param.description

    return prop


def _serialize_default(value: Any) -> Any:
    """Serialize a default value for JSON Schema."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def from_json_schema(
    node_type: str,
    input_schema: dict[str, Any],
    description: str = "",
) -> NodeTypeSchema:
    """Convert an MCP-style JSON Schema to a NodeTypeSchema.

    Parses the properties/required structure to reconstruct constructor
    parameters.

    Args:
        node_type: The node type identifier.
        input_schema: JSON Schema dict with properties and required.
        description: Human-readable description.

    Returns:
        NodeTypeSchema with constructor populated from the schema.
    """
    properties = input_schema.get("properties", {})
    required_names = set(input_schema.get("required", []))

    positional: list[ParamSchema] = []
    named: list[ParamSchema] = []

    for pname, pschema in properties.items():
        type_str = _json_schema_type_to_cap(pschema)
        desc = pschema.get("description", "")

        if pname in required_names:
            positional.append(
                ParamSchema(
                    name=pname,
                    type=type_str,
                    default=MISSING,
                    description=desc,
                )
            )
        else:
            default = pschema.get("default", None)
            named.append(
                ParamSchema(
                    name=pname,
                    type=type_str,
                    default=default,
                    description=desc,
                )
            )

    return NodeTypeSchema(
        node_type=node_type,
        description=description,
        constructor=ConstructorSchema(positional=positional, named=named),
    )


# JSON Schema type → CAP type string
_CAP_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
    "object": "dict",
    "array": "list",
}


def _json_schema_type_to_cap(pschema: dict[str, Any]) -> str:
    """Convert a JSON Schema property to a CAP type string."""
    json_type = pschema.get("type", "object")
    base = _CAP_TYPE_MAP.get(json_type, "Any")

    if json_type == "array":
        items = pschema.get("items", {})
        item_type = _CAP_TYPE_MAP.get(items.get("type", ""), "Any")
        return f"list[{item_type}]"

    return base


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_schema(schema: NodeTypeSchema) -> list[str]:
    """Validate a NodeTypeSchema for completeness and consistency.

    Returns a list of warning/error messages. Empty list means valid.

    Checks:
    - node_type is non-empty and a valid identifier
    - No duplicate parameter names in constructor
    - No duplicate property names
    - No duplicate method names
    - Required params come before optional in positional list
    - Method param names don't shadow constructor params
    """
    errors: list[str] = []

    # node_type
    if not schema.node_type:
        errors.append("node_type is empty")
    elif not schema.node_type.replace("_", "").isalnum():
        errors.append(f"node_type '{schema.node_type}' contains invalid characters")

    # Constructor: no duplicate names
    ctor_names: list[str] = []
    for p in schema.constructor.positional:
        if p.name in ctor_names:
            errors.append(f"Duplicate constructor param: '{p.name}'")
        ctor_names.append(p.name)
    if schema.constructor.variadic:
        ctor_names.append(schema.constructor.variadic.name)
    for p in schema.constructor.named:
        if p.name in ctor_names:
            errors.append(f"Duplicate constructor param: '{p.name}'")
        ctor_names.append(p.name)

    # Positional: required before optional
    seen_optional = False
    for p in schema.constructor.positional:
        if p.default is MISSING:
            if seen_optional:
                errors.append(f"Required positional param '{p.name}' after optional")
        else:
            seen_optional = True

    # Properties: no duplicates
    prop_names: set[str] = set()
    for prop in schema.properties:
        if prop.name in prop_names:
            errors.append(f"Duplicate property: '{prop.name}'")
        prop_names.add(prop.name)

    # Methods: no duplicates
    method_names: set[str] = set()
    for m in schema.methods:
        if m.name in method_names:
            errors.append(f"Duplicate method: '{m.name}'")
        method_names.add(m.name)

    return errors


# ---------------------------------------------------------------------------
# DSL signature generation
# ---------------------------------------------------------------------------


def generate_dsl_signature(schema: NodeTypeSchema) -> str:
    """Generate a Python-style DSL signature string.

    Produces a one-line signature suitable for LLM prompts and
    documentation:

        shell(command, *args, timeout=120, cwd="")

    Args:
        schema: The node type schema.

    Returns:
        Formatted signature string.
    """
    parts: list[str] = []

    # Positional params
    for p in schema.constructor.positional:
        parts.append(p.name)

    # Variadic
    if schema.constructor.variadic:
        parts.append(f"*{schema.constructor.variadic.name}")

    # Named params
    for p in schema.constructor.named:
        if p.default is MISSING:
            parts.append(f"{p.name}")
        else:
            parts.append(f"{p.name}={_format_default(p.default)}")

    return f"{schema.node_type}({', '.join(parts)})"


def generate_dsl_doc(schema: NodeTypeSchema) -> str:
    """Generate a multi-line DSL documentation block.

    Produces a formatted documentation string with signature,
    description, parameters, properties, and methods:

        shell(command, *args, timeout=120, cwd="")
          Async shell command execution.

          Parameters:
            command (str): [required]
            *args (str): Variadic arguments
            timeout (int): Command timeout in seconds (default: 120)

          Properties (read-only):
            is_complete (bool): Whether the command has finished
            output (str): Command output

          Methods:
            cancel(): Cancel the running command
            set_timeout(seconds: int) -> ShellNode: Set timeout [chainable]
    """
    lines: list[str] = []

    # Signature
    lines.append(generate_dsl_signature(schema))

    # Description
    if schema.description:
        lines.append(f"  {schema.description}")

    # Parameters
    all_params = (
        schema.constructor.positional
        + ([schema.constructor.variadic] if schema.constructor.variadic else [])
        + schema.constructor.named
    )
    if all_params:
        lines.append("")
        lines.append("  Parameters:")
        for p in schema.constructor.positional:
            lines.append(f"    {p.name} ({p.type}): {p.description or '[required]'}")
        if schema.constructor.variadic:
            v = schema.constructor.variadic
            lines.append(f"    *{v.name} ({v.type}): {v.description or 'Variadic'}")
        for p in schema.constructor.named:
            default_str = _format_default(p.default)
            desc = p.description or ""
            if desc:
                lines.append(f"    {p.name} ({p.type}): {desc} (default: {default_str})")
            else:
                lines.append(f"    {p.name} ({p.type}): default: {default_str}")

    # Properties
    if schema.properties:
        read_only = [prop for prop in schema.properties if prop.readable and not prop.writable]
        read_write = [prop for prop in schema.properties if prop.readable and prop.writable]

        if read_only:
            lines.append("")
            lines.append("  Properties (read-only):")
            for prop in read_only:
                lines.append(f"    {prop.name} ({prop.type}): {prop.description}")

        if read_write:
            lines.append("")
            lines.append("  Properties (read-write):")
            for prop in read_write:
                lines.append(f"    {prop.name} ({prop.type}): {prop.description}")

    # Methods
    if schema.methods:
        lines.append("")
        lines.append("  Methods:")
        for m in schema.methods:
            sig = _format_method_sig(m)
            desc = m.description or ""
            chain = " [chainable]" if m.chainable else ""
            lines.append(f"    {sig}: {desc}{chain}")

    return "\n".join(lines)


def _format_default(value: Any) -> str:
    """Format a default value for display."""
    if value is MISSING:
        return "MISSING"
    if isinstance(value, str):
        return repr(value)
    return repr(value)


def _format_method_sig(method: MethodSchema) -> str:
    """Format a method signature."""
    parts: list[str] = []
    for p in method.params:
        if p.default is MISSING:
            parts.append(f"{p.name}: {p.type}")
        else:
            parts.append(f"{p.name}: {p.type} = {_format_default(p.default)}")

    ret = ""
    if method.returns and method.returns != "None":
        ret = f" -> {method.returns}"

    return f"{method.name}({', '.join(parts)}){ret}"


# ---------------------------------------------------------------------------
# Multi-schema documentation
# ---------------------------------------------------------------------------


def generate_plugin_docs(schemas: list[NodeTypeSchema], title: str = "Plugin Node Types") -> str:
    """Generate markdown documentation for multiple plugin node types.

    Produces a comprehensive reference document suitable for LLM prompts
    or user documentation. Groups schemas by node_type and includes full
    DSL signatures, parameters, properties, and methods.

    Args:
        schemas: List of NodeTypeSchemas to document.
        title: Title for the generated document.

    Returns:
        Formatted markdown documentation string.
    """
    if not schemas:
        return f"# {title}\n\nNo plugin node types registered.\n"

    lines: list[str] = []
    lines.append(f"# {title}\n")
    lines.append(
        "Functions available from plugin servers. "
        "These extend the built-in DSL with additional node types.\n"
    )

    for schema in sorted(schemas, key=lambda s: s.node_type):
        lines.append("")
        lines.append(f"## {schema.node_type}")
        lines.append("")

        # Signature
        sig = generate_dsl_signature(schema)
        lines.append("```python")
        lines.append(sig)
        lines.append("```")
        lines.append("")

        # Description
        if schema.description:
            lines.append(schema.description)
            lines.append("")

        # Parameters
        all_params = (
            schema.constructor.positional
            + ([schema.constructor.variadic] if schema.constructor.variadic else [])
            + schema.constructor.named
        )
        if all_params:
            lines.append("**Parameters:**")
            lines.append("")
            for p in schema.constructor.positional:
                desc = p.description or "[required]"
                lines.append(f"- `{p.name}` (`{p.type}`): {desc}")
            if schema.constructor.variadic:
                v = schema.constructor.variadic
                desc = v.description or "Variadic arguments"
                lines.append(f"- `*{v.name}` (`{v.type}`): {desc}")
            for p in schema.constructor.named:
                default_str = _format_default(p.default)
                desc = p.description or ""
                if desc:
                    lines.append(f"- `{p.name}` (`{p.type}`): {desc} (default: {default_str})")
                else:
                    lines.append(f"- `{p.name}` (`{p.type}`): default: {default_str}")
            lines.append("")

        # Properties
        if schema.properties:
            read_only = [prop for prop in schema.properties if prop.readable and not prop.writable]
            read_write = [prop for prop in schema.properties if prop.readable and prop.writable]

            if read_only:
                lines.append("**Properties (read-only):**")
                lines.append("")
                for prop in read_only:
                    desc = prop.description or ""
                    lines.append(f"- `{prop.name}` (`{prop.type}`): {desc}")
                lines.append("")

            if read_write:
                lines.append("**Properties (read-write):**")
                lines.append("")
                for prop in read_write:
                    desc = prop.description or ""
                    lines.append(f"- `{prop.name}` (`{prop.type}`): {desc}")
                lines.append("")

        # Methods
        if schema.methods:
            lines.append("**Methods:**")
            lines.append("")
            for m in schema.methods:
                sig = _format_method_sig(m)
                desc = m.description or ""
                chain = " [chainable]" if m.chainable else ""
                lines.append(f"- `{sig}`: {desc}{chain}")
            lines.append("")

    return "\n".join(lines)
