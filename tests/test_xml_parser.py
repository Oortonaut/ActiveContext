"""Tests for XML command parser."""

import pytest

from activecontext.session.xml_parser import (
    is_xml_command,
    parse_xml_to_python,
)


class TestIsXmlCommand:
    """Tests for XML detection."""

    def test_detects_simple_xml_tag(self):
        assert is_xml_command("<view name='v' path='main.py'/>")

    def test_detects_xml_with_whitespace(self):
        assert is_xml_command("  <view name='v'/>  ")

    def test_detects_self_closing_tag(self):
        assert is_xml_command("<ls/>")

    def test_detects_tag_with_children(self):
        assert is_xml_command("<group name='g'><member ref='v'/></group>")

    def test_rejects_python_less_than(self):
        assert not is_xml_command("x < y")

    def test_rejects_python_left_shift(self):
        assert not is_xml_command("x << 2")

    def test_rejects_python_comparison(self):
        assert not is_xml_command("if x < 10:")

    def test_rejects_plain_python(self):
        assert not is_xml_command("v = view('main.py')")


class TestConstructorTags:
    """Tests for constructor tags (view, group, topic, artifact)."""

    def test_view_basic(self):
        xml = '<view name="v" path="main.py"/>'
        py = parse_xml_to_python(xml)
        assert py == "v = view('main.py')"

    def test_view_with_options(self):
        xml = '<view name="v" path="main.py" tokens="2000" lod="1"/>'
        py = parse_xml_to_python(xml)
        assert "v = view('main.py'" in py
        assert "tokens=2000" in py
        assert "lod=1" in py

    def test_view_with_pos(self):
        xml = '<view name="v" path="main.py" pos="10:0"/>'
        py = parse_xml_to_python(xml)
        assert "pos='10:0'" in py

    def test_group_empty(self):
        xml = '<group name="g" tokens="500"/>'
        py = parse_xml_to_python(xml)
        assert py == "g = group(tokens=500)"

    def test_group_with_members(self):
        xml = '<group name="g" tokens="500"><member ref="v1"/><member ref="v2"/></group>'
        py = parse_xml_to_python(xml)
        assert "g = group(v1, v2" in py
        assert "tokens=500" in py

    def test_topic(self):
        xml = '<topic name="t" title="Feature X" tokens="1000"/>'
        py = parse_xml_to_python(xml)
        assert "t = topic('Feature X'" in py
        assert "tokens=1000" in py

    def test_artifact(self):
        xml = '<artifact name="a" type="code" content="print(1)" language="python"/>'
        py = parse_xml_to_python(xml)
        assert "a = artifact('code'" in py
        assert "content='print(1)'" in py
        assert "language='python'" in py

    def test_constructor_requires_name(self):
        xml = '<view path="main.py"/>'
        with pytest.raises(ValueError, match="requires 'name' attribute"):
            parse_xml_to_python(xml)


class TestMethodCalls:
    """Tests for method call tags with self attribute."""

    def test_set_lod(self):
        xml = '<SetLod self="v" level="2"/>'
        py = parse_xml_to_python(xml)
        assert py == "v.SetLod(level=2)"

    def test_set_tokens(self):
        xml = '<SetTokens self="v" count="500"/>'
        py = parse_xml_to_python(xml)
        assert py == "v.SetTokens(count=500)"

    def test_run(self):
        xml = '<Run self="v" freq="Sync"/>'
        py = parse_xml_to_python(xml)
        assert py == "v.Run(freq=TickFrequency.turn())"  # Sync maps to turn()

    def test_chained_methods_as_separate_tags(self):
        xml = '<SetLod self="v" level="1"/><SetTokens self="v" count="500"/>'
        py = parse_xml_to_python(xml)
        lines = py.strip().split("\n")
        assert len(lines) == 2
        assert "v.SetLod(level=1)" in lines[0]
        assert "v.SetTokens(count=500)" in lines[1]


class TestUtilityTags:
    """Tests for utility function tags."""

    def test_ls_empty(self):
        xml = "<ls/>"
        py = parse_xml_to_python(xml)
        assert py == "ls()"

    def test_show_with_self(self):
        xml = '<show self="v"/>'
        py = parse_xml_to_python(xml)
        assert py == "show(v)"

    def test_show_with_options(self):
        xml = '<show self="v" lod="2" tokens="100"/>'
        py = parse_xml_to_python(xml)
        assert "show(v" in py
        assert "lod=2" in py
        assert "tokens=100" in py

    def test_done_empty(self):
        xml = "<done/>"
        py = parse_xml_to_python(xml)
        assert py == "done()"

    def test_done_with_message(self):
        xml = '<done message="Task complete"/>'
        py = parse_xml_to_python(xml)
        assert py == "done(message='Task complete')"

    def test_link(self):
        xml = '<link child="v" parent="g"/>'
        py = parse_xml_to_python(xml)
        assert py == "link(v, g)"

    def test_unlink(self):
        xml = '<unlink child="v" parent="g"/>'
        py = parse_xml_to_python(xml)
        assert py == "unlink(v, g)"

    def test_link_requires_both_args(self):
        xml = '<link child="v"/>'
        with pytest.raises(ValueError, match="requires 'child' and 'parent'"):
            parse_xml_to_python(xml)


class TestValueFormatting:
    """Tests for proper value type conversion."""

    def test_integer_unquoted(self):
        xml = '<view name="v" path="f.py" tokens="2000"/>'
        py = parse_xml_to_python(xml)
        assert "tokens=2000" in py  # No quotes

    def test_float_unquoted(self):
        xml = '<SetWeight self="v" value="0.5"/>'
        py = parse_xml_to_python(xml)
        assert "value=0.5" in py

    def test_boolean_true(self):
        xml = '<SetFlag self="v" enabled="true"/>'
        py = parse_xml_to_python(xml)
        assert "enabled=True" in py

    def test_boolean_false(self):
        xml = '<SetFlag self="v" enabled="false"/>'
        py = parse_xml_to_python(xml)
        assert "enabled=False" in py

    def test_string_quoted(self):
        xml = '<view name="v" path="main.py"/>'
        py = parse_xml_to_python(xml)
        assert "'main.py'" in py

    def test_string_with_special_chars(self):
        xml = '<done message="It\'s done!"/>'
        py = parse_xml_to_python(xml)
        # Should properly escape
        assert "It" in py and "done" in py


class TestMultipleCommands:
    """Tests for multiple XML commands."""

    def test_multiple_tags(self):
        xml = """
        <view name="v" path="main.py"/>
        <SetLod self="v" level="1"/>
        <Run self="v" freq="Sync"/>
        """
        py = parse_xml_to_python(xml)
        lines = [l.strip() for l in py.strip().split("\n") if l.strip()]
        assert len(lines) == 3
        assert "v = view" in lines[0]
        assert "v.SetLod" in lines[1]
        assert "v.Run" in lines[2]


class TestErrorHandling:
    """Tests for error handling."""

    def test_malformed_xml(self):
        xml = "<view name='v'"  # Missing closing
        with pytest.raises(ValueError, match="Invalid XML"):
            parse_xml_to_python(xml)

    def test_unknown_tag_without_self(self):
        xml = '<unknown foo="bar"/>'
        with pytest.raises(ValueError, match="Unknown XML tag"):
            parse_xml_to_python(xml)
