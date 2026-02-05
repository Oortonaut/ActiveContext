"""Tests for CAP Schema Discovery.

Verifies that:
1. Introspection extracts correct schemas from ContextNode subclasses
2. JSON Schema conversion round-trips correctly
3. Validation catches schema problems
4. DSL signature/doc generation is well-formed
"""

from __future__ import annotations

import pytest

from activecontext.context.nodes import (
    ArtifactNode,
    ShellNode,
    TopicNode,
)
from activecontext.plugins.schema import (
    from_json_schema,
    generate_dsl_doc,
    generate_dsl_signature,
    introspect_node_class,
    to_json_schema,
    validate_schema,
)
from activecontext.plugins.wire import (
    MISSING,
    ConstructorSchema,
    MethodSchema,
    NodeTypeSchema,
    ParamSchema,
    PropertySchema,
)


class TestIntrospection:
    """Extract schemas from real node classes."""

    def test_shell_node_type(self) -> None:
        schema = introspect_node_class(ShellNode)
        assert schema.node_type == "ShellNode"

    def test_shell_constructor_has_command(self) -> None:
        schema = introspect_node_class(ShellNode)
        # command has a default of "" so it appears as named
        all_names = [p.name for p in schema.constructor.positional] + [
            p.name for p in schema.constructor.named
        ]
        assert "command" in all_names

    def test_shell_excludes_base_fields(self) -> None:
        """Base ContextNode fields are excluded from schema."""
        schema = introspect_node_class(ShellNode)
        all_names = [p.name for p in schema.constructor.positional] + [
            p.name for p in schema.constructor.named
        ]
        assert "node_id" not in all_names
        assert "parent_ids" not in all_names
        assert "expansion" not in all_names
        assert "version" not in all_names
        assert "tracing" not in all_names

    def test_shell_has_args_field(self) -> None:
        schema = introspect_node_class(ShellNode)
        named_names = [p.name for p in schema.constructor.named]
        assert "args" in named_names

    def test_topic_node_type(self) -> None:
        schema = introspect_node_class(TopicNode)
        assert schema.node_type == "TopicNode"

    def test_artifact_node_type(self) -> None:
        schema = introspect_node_class(ArtifactNode)
        assert schema.node_type == "ArtifactNode"

    def test_description_from_docstring(self) -> None:
        schema = introspect_node_class(ShellNode)
        # ShellNode should have some docstring
        assert isinstance(schema.description, str)

    def test_not_a_dataclass_raises(self) -> None:
        class NotADataclass:
            pass

        with pytest.raises(TypeError, match="not a dataclass"):
            introspect_node_class(NotADataclass)

    def test_properties_are_extracted(self) -> None:
        schema = introspect_node_class(ShellNode)
        prop_names = [p.name for p in schema.properties]
        # ShellNode has properties like is_complete, is_success, etc.
        assert "node_type" in prop_names or "is_complete" in prop_names

    def test_shell_named_params_have_defaults(self) -> None:
        """Named params should all have non-MISSING defaults."""
        schema = introspect_node_class(ShellNode)
        for p in schema.constructor.named:
            assert p.default is not MISSING, f"Named param '{p.name}' has no default"


class TestJsonSchemaConversion:
    """Convert between CAP and JSON Schema formats."""

    def test_to_json_schema_basic(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="name", type="str")],
                named=[
                    ParamSchema(name="count", type="int", default=10),
                    ParamSchema(name="verbose", type="bool", default=False),
                ],
            ),
        )
        js = to_json_schema(schema)
        assert js["type"] == "object"
        assert "name" in js["properties"]
        assert js["properties"]["name"]["type"] == "string"
        assert js["required"] == ["name"]
        assert js["properties"]["count"]["type"] == "integer"
        assert js["properties"]["count"]["default"] == 10
        assert js["properties"]["verbose"]["type"] == "boolean"

    def test_to_json_schema_list_type(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            constructor=ConstructorSchema(
                named=[ParamSchema(name="items", type="list[str]", default=[])],
            ),
        )
        js = to_json_schema(schema)
        prop = js["properties"]["items"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_to_json_schema_no_required(self) -> None:
        """All optional params → no required field."""
        schema = NodeTypeSchema(
            node_type="test",
            constructor=ConstructorSchema(
                named=[ParamSchema(name="x", type="int", default=0)],
            ),
        )
        js = to_json_schema(schema)
        assert "required" not in js

    def test_from_json_schema_basic(self) -> None:
        js = {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"},
                "timeout": {"type": "integer", "default": 120},
            },
            "required": ["command"],
        }
        schema = from_json_schema("ShellNode", js, description="Shell command")
        assert schema.node_type == "ShellNode"
        assert schema.description == "Shell command"
        assert len(schema.constructor.positional) == 1
        assert schema.constructor.positional[0].name == "command"
        assert schema.constructor.positional[0].type == "str"
        assert len(schema.constructor.named) == 1
        assert schema.constructor.named[0].default == 120

    def test_roundtrip(self) -> None:
        """to_json_schema → from_json_schema preserves structure."""
        original = NodeTypeSchema(
            node_type="custom",
            description="A custom node",
            constructor=ConstructorSchema(
                positional=[
                    ParamSchema(name="name", type="str", description="Node name"),
                ],
                named=[
                    ParamSchema(name="count", type="int", default=5),
                    ParamSchema(name="flag", type="bool", default=True),
                ],
            ),
        )
        js = to_json_schema(original)
        restored = from_json_schema("custom", js, description="A custom node")

        assert restored.node_type == original.node_type
        assert len(restored.constructor.positional) == 1
        assert restored.constructor.positional[0].name == "name"
        assert restored.constructor.positional[0].type == "str"
        assert len(restored.constructor.named) == 2

    def test_from_json_schema_array_type(self) -> None:
        js = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        schema = from_json_schema("test", js)
        assert schema.constructor.named[0].type == "list[str]"

    def test_real_node_roundtrip(self) -> None:
        """Introspect a real node, convert to JSON Schema and back."""
        original = introspect_node_class(ShellNode)
        js = to_json_schema(original)
        restored = from_json_schema(original.node_type, js)

        # Same number of positional params
        assert len(restored.constructor.positional) == len(original.constructor.positional)
        # Same positional names
        assert [p.name for p in restored.constructor.positional] == [
            p.name for p in original.constructor.positional
        ]


class TestValidation:
    """Schema validation catches problems."""

    def test_valid_schema(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="x", type="int")],
            ),
        )
        errors = validate_schema(schema)
        assert errors == []

    def test_empty_node_type(self) -> None:
        schema = NodeTypeSchema(node_type="")
        errors = validate_schema(schema)
        assert any("empty" in e for e in errors)

    def test_invalid_node_type_chars(self) -> None:
        schema = NodeTypeSchema(node_type="my-node!")
        errors = validate_schema(schema)
        assert any("invalid characters" in e for e in errors)

    def test_duplicate_constructor_params(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            constructor=ConstructorSchema(
                positional=[
                    ParamSchema(name="x"),
                    ParamSchema(name="x"),
                ],
            ),
        )
        errors = validate_schema(schema)
        assert any("Duplicate constructor param" in e for e in errors)

    def test_duplicate_properties(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            properties=[
                PropertySchema(name="foo"),
                PropertySchema(name="foo"),
            ],
        )
        errors = validate_schema(schema)
        assert any("Duplicate property" in e for e in errors)

    def test_duplicate_methods(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            methods=[
                MethodSchema(name="bar"),
                MethodSchema(name="bar"),
            ],
        )
        errors = validate_schema(schema)
        assert any("Duplicate method" in e for e in errors)

    def test_real_node_validates(self) -> None:
        """Introspected real nodes should validate cleanly."""
        schema = introspect_node_class(ShellNode)
        errors = validate_schema(schema)
        assert errors == [], f"ShellNode validation errors: {errors}"

    def test_underscore_in_node_type_ok(self) -> None:
        """node_type with underscores is valid."""
        schema = NodeTypeSchema(node_type="mcp_server")
        errors = validate_schema(schema)
        assert errors == []


class TestDSLGeneration:
    """DSL signature and documentation generation."""

    def test_simple_signature(self) -> None:
        schema = NodeTypeSchema(
            node_type="topic",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="title", type="str")],
            ),
        )
        sig = generate_dsl_signature(schema)
        assert sig == "topic(title)"

    def test_signature_with_defaults(self) -> None:
        schema = NodeTypeSchema(
            node_type="shell",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="command")],
                variadic=ParamSchema(name="args"),
                named=[
                    ParamSchema(name="timeout", type="int", default=120),
                    ParamSchema(name="cwd", type="str", default=""),
                ],
            ),
        )
        sig = generate_dsl_signature(schema)
        assert sig == "shell(command, *args, timeout=120, cwd='')"

    def test_signature_all_optional(self) -> None:
        schema = NodeTypeSchema(
            node_type="lock",
            constructor=ConstructorSchema(
                named=[ParamSchema(name="path", type="str", default="")],
            ),
        )
        sig = generate_dsl_signature(schema)
        assert sig == "lock(path='')"

    def test_doc_contains_signature(self) -> None:
        schema = NodeTypeSchema(
            node_type="topic",
            description="Conversation topic marker.",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="title", type="str")],
            ),
        )
        doc = generate_dsl_doc(schema)
        assert doc.startswith("topic(title)")
        assert "Conversation topic marker." in doc

    def test_doc_includes_properties(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            properties=[
                PropertySchema(
                    name="value",
                    type="int",
                    readable=True,
                    writable=False,
                    description="Current value",
                ),
            ],
        )
        doc = generate_dsl_doc(schema)
        assert "Properties (read-only):" in doc
        assert "value (int)" in doc

    def test_doc_includes_methods(self) -> None:
        schema = NodeTypeSchema(
            node_type="test",
            methods=[
                MethodSchema(
                    name="reset",
                    description="Reset to defaults",
                ),
                MethodSchema(
                    name="set_value",
                    params=[ParamSchema(name="v", type="int")],
                    returns="TestNode",
                    chainable=True,
                    description="Set the value",
                ),
            ],
        )
        doc = generate_dsl_doc(schema)
        assert "Methods:" in doc
        assert "reset()" in doc
        assert "[chainable]" in doc

    def test_real_node_doc_generation(self) -> None:
        """Generate docs from a real introspected node."""
        schema = introspect_node_class(ShellNode)
        doc = generate_dsl_doc(schema)
        assert doc.startswith("ShellNode(")
        assert len(doc) > 50  # should be non-trivial


class TestPluginDocsGeneration:
    """Test multi-schema documentation generation."""

    def test_empty_schemas(self) -> None:
        """Empty schema list produces minimal doc."""
        from activecontext.plugins.schema import generate_plugin_docs

        doc = generate_plugin_docs([])
        assert "No plugin node types registered" in doc

    def test_single_schema(self) -> None:
        """Single schema produces formatted docs."""
        from activecontext.plugins.schema import generate_plugin_docs

        schema = NodeTypeSchema(
            node_type="test_plugin",
            description="A test plugin node",
            constructor=ConstructorSchema(
                positional=[ParamSchema(name="name", type="str", description="Node name")],
                named=[ParamSchema(name="count", type="int", default=5)],
            ),
            properties=[
                PropertySchema(
                    name="value",
                    type="int",
                    readable=True,
                    writable=False,
                    description="Current value",
                )
            ],
            methods=[
                MethodSchema(
                    name="reset",
                    description="Reset to defaults",
                    returns="None",
                )
            ],
        )
        doc = generate_plugin_docs([schema])
        assert "# Plugin Node Types" in doc  # Default title
        assert "## test_plugin" in doc
        assert "A test plugin node" in doc
        assert "name" in doc
        assert "count" in doc
        assert "value" in doc
        assert "reset()" in doc

    def test_multiple_schemas_sorted(self) -> None:
        """Multiple schemas are sorted by node_type."""
        from activecontext.plugins.schema import generate_plugin_docs

        schema1 = NodeTypeSchema(node_type="zebra", description="Last alphabetically")
        schema2 = NodeTypeSchema(node_type="apple", description="First alphabetically")
        schema3 = NodeTypeSchema(node_type="banana", description="Middle")

        doc = generate_plugin_docs([schema1, schema2, schema3])
        # Find positions of each type in the doc
        apple_pos = doc.find("## apple")
        banana_pos = doc.find("## banana")
        zebra_pos = doc.find("## zebra")

        assert apple_pos < banana_pos < zebra_pos

    def test_custom_title(self) -> None:
        """Custom title is used in doc."""
        from activecontext.plugins.schema import generate_plugin_docs

        schema = NodeTypeSchema(node_type="test")
        doc = generate_plugin_docs([schema], title="My Custom Plugins")
        assert "# My Custom Plugins" in doc

    def test_real_node_multi_doc(self) -> None:
        """Generate docs from real introspected nodes."""
        from activecontext.plugins.schema import generate_plugin_docs

        shell_schema = introspect_node_class(ShellNode)
        topic_schema = introspect_node_class(TopicNode)
        artifact_schema = introspect_node_class(ArtifactNode)

        doc = generate_plugin_docs([shell_schema, topic_schema, artifact_schema])
        assert "## ArtifactNode" in doc
        assert "## ShellNode" in doc
        assert "## TopicNode" in doc
        # Verify sorted order
        assert doc.find("## ArtifactNode") < doc.find("## ShellNode") < doc.find("## TopicNode")


class TestSchemaDiscoveryIntegration:
    """Integration tests for the full schema discovery flow."""

    def test_introspect_to_json_schema(self) -> None:
        """Full flow: introspect -> convert to JSON Schema -> validate."""
        schema = introspect_node_class(ShellNode)
        js = to_json_schema(schema)
        errors = validate_schema(schema)

        assert errors == []
        assert js["type"] == "object"
        assert "properties" in js

    def test_schema_roundtrip_preserves_info(self) -> None:
        """Introspect -> JSON Schema -> back preserves essential info."""
        original = introspect_node_class(TopicNode)
        js = to_json_schema(original)
        restored = from_json_schema(original.node_type, js, description=original.description)

        # Same node_type
        assert restored.node_type == original.node_type
        # Same number of required params
        assert len(restored.constructor.positional) == len(original.constructor.positional)
        # Same positional param names
        assert [p.name for p in restored.constructor.positional] == [
            p.name for p in original.constructor.positional
        ]

    def test_introspected_schema_generates_valid_signature(self) -> None:
        """Introspected schema produces valid DSL signature."""
        schema = introspect_node_class(ArtifactNode)
        sig = generate_dsl_signature(schema)

        # Should be a valid Python-like signature
        assert sig.startswith("ArtifactNode(")
        assert sig.endswith(")")
        # Should not have syntax errors (basic check)
        assert sig.count("(") == sig.count(")")

    def test_introspected_schema_generates_comprehensive_doc(self) -> None:
        """Introspected schema produces comprehensive documentation."""
        schema = introspect_node_class(ShellNode)
        doc = generate_dsl_doc(schema)

        # Should have signature
        assert "ShellNode(" in doc
        # Should have description
        assert len(doc) > 100
        # Should be multi-line
        assert "\n" in doc
