"""Tests for core features: parsing, projection, messages."""

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.core.llm.provider import Message, Role
from activecontext.core.prompts import ParsedResponse, Segment, parse_response
from activecontext.session.protocols import Projection, ProjectionSection


def _executable(parsed: ParsedResponse) -> list[str]:
    """Extract executable segment contents (python/acrepl fenced + XML)."""
    return [
        s.content
        for s in parsed.segments
        if s.language == "python/acrepl" or s.kind == "xml"
    ]


class TestBlockSplitting:
    """Tests for structural block splitting."""

    def test_fenced_acrepl_block(self) -> None:
        """python/acrepl fenced blocks are captured with language tag."""
        text = """Here's some explanation.

```python/acrepl
x = 1 + 1
```

And more text."""

        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert len(fenced) == 1
        assert fenced[0].language == "python/acrepl"
        assert fenced[0].content == "x = 1 + 1"

    def test_plain_python_block_captured(self) -> None:
        """Regular python blocks are captured as fenced, not executable."""
        text = """Here's an example:

```python
# This is just documentation
def example():
    pass
```

Not executed."""

        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert len(fenced) == 1
        assert fenced[0].language == "python"
        # Not executable
        assert len(_executable(parsed)) == 0

    def test_other_language_blocks(self) -> None:
        """Other language blocks are captured with their language tags."""
        text = """Some bash:

```bash
echo "hello"
```

Some json:

```json
{"key": "value"}
```
"""

        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert len(fenced) == 2
        assert fenced[0].language == "bash"
        assert fenced[1].language == "json"
        # Not executable
        assert len(_executable(parsed)) == 0

    def test_multiple_acrepl_blocks(self) -> None:
        """Multiple python/acrepl blocks should all be captured."""
        text = """First:

```python/acrepl
a = 1
```

Second:

```python/acrepl
b = 2
```
"""

        parsed = parse_response(text)
        exe = _executable(parsed)
        assert len(exe) == 2
        assert exe[0] == "a = 1"
        assert exe[1] == "b = 2"

    def test_mixed_blocks(self) -> None:
        """Mix of acrepl and regular python — only acrepl is executable."""
        text = """Example (not executed):

```python
example_only = True
```

Now execute:

```python/acrepl
real_code = True
```
"""

        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert len(fenced) == 2  # both captured structurally
        exe = _executable(parsed)
        assert len(exe) == 1
        assert "real_code" in exe[0]

    def test_prose_segments(self) -> None:
        """Prose segments should be captured."""
        text = """Before code.

```python/acrepl
x = 1
```

After code."""

        parsed = parse_response(text)
        prose = parsed.prose_only
        assert "Before code" in prose
        assert "After code" in prose

    def test_unfenced_block(self) -> None:
        """Fenced block with no language tag."""
        text = """Text.

```
some content
```
"""
        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert len(fenced) == 1
        assert fenced[0].language == ""
        assert fenced[0].content == "some content"


class TestSegmentTypes:
    """Tests for Segment dataclass and mime-type assignment."""

    def test_segment_fields(self) -> None:
        """Segment has kind, content, mime_type, and language."""
        seg = Segment(kind="fenced", content="x=1", mime_type="text/x-python", language="python")
        assert seg.kind == "fenced"
        assert seg.content == "x=1"
        assert seg.mime_type == "text/x-python"
        assert seg.language == "python"

    def test_segment_language_defaults_empty(self) -> None:
        """Language defaults to empty string for non-fenced segments."""
        seg = Segment(kind="prose", content="hello", mime_type="text/markdown")
        assert seg.language == ""

    def test_fenced_python_mime_type(self) -> None:
        """Fenced python blocks get text/x-python mime type."""
        text = """Explanation.

```python/acrepl
x = 1
```
"""
        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert len(fenced) == 1
        assert fenced[0].mime_type == "text/x-python"
        assert fenced[0].language == "python/acrepl"

    def test_fenced_bash_mime_type(self) -> None:
        """Fenced bash blocks get text/x-shellscript mime type."""
        text = """```bash
echo hi
```
"""
        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert fenced[0].mime_type == "text/x-shellscript"
        assert fenced[0].language == "bash"

    def test_fenced_json_mime_type(self) -> None:
        """Fenced json blocks get application/json mime type."""
        text = """```json
{"a": 1}
```
"""
        parsed = parse_response(text)
        fenced = [s for s in parsed.segments if s.kind == "fenced"]
        assert fenced[0].mime_type == "application/json"

    def test_prose_mime_type(self) -> None:
        """Prose segments get text/markdown mime type."""
        parsed = parse_response("Just some text.")
        assert len(parsed.segments) == 1
        assert parsed.segments[0].kind == "prose"
        assert parsed.segments[0].mime_type == "text/markdown"

    def test_xml_command_mime_type(self) -> None:
        """XML DSL commands get application/xml mime type."""
        text = """I'll create a view.
<view name="v" path="main.py"/>
Done."""
        parsed = parse_response(text)
        xml_segs = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml_segs) == 1
        assert xml_segs[0].mime_type == "application/xml"
        assert 'view' in xml_segs[0].content

    def test_quoted_mime_type(self) -> None:
        """Blockquote segments get text/markdown mime type with 'quoted' kind."""
        text = """Here's the relevant code:
> def foo():
>     return bar
Needs fixing."""
        parsed = parse_response(text)
        quoted_segs = [s for s in parsed.segments if s.kind == "quoted"]
        assert len(quoted_segs) == 1
        assert quoted_segs[0].mime_type == "text/markdown"
        assert "> def foo():" in quoted_segs[0].content


class TestQuotedBlocks:
    """Tests for blockquote parsing."""

    def test_single_blockquote(self) -> None:
        """A markdown blockquote is extracted as a separate segment."""
        text = """Before quote.
> This is quoted.
> Second line.
After quote."""
        parsed = parse_response(text)
        kinds = [s.kind for s in parsed.segments]
        assert kinds == ["prose", "quoted", "prose"]

    def test_multiple_blockquotes(self) -> None:
        """Multiple blockquote sections become separate segments."""
        text = """First section.
> Quote one.
Middle text.
> Quote two.
End."""
        parsed = parse_response(text)
        kinds = [s.kind for s in parsed.segments]
        assert kinds == ["prose", "quoted", "prose", "quoted", "prose"]

    def test_blockquote_at_start(self) -> None:
        """Blockquote at the start of the response."""
        text = """> Quoted first line.
> Still quoted.
Then prose."""
        parsed = parse_response(text)
        kinds = [s.kind for s in parsed.segments]
        assert kinds == ["quoted", "prose"]

    def test_blockquote_at_end(self) -> None:
        """Blockquote at the end of the response."""
        text = """Prose first.
> Quoted last."""
        parsed = parse_response(text)
        kinds = [s.kind for s in parsed.segments]
        assert kinds == ["prose", "quoted"]

    def test_blockquote_not_executable(self) -> None:
        """Blockquotes are not executable segments."""
        text = """> This is a quote.
Not code."""
        parsed = parse_response(text)
        assert len(_executable(parsed)) == 0

    def test_blockquote_between_fenced_blocks(self) -> None:
        """Blockquote between fenced code blocks."""
        text = """Intro.

```python/acrepl
a = 1
```

> Quoting something.

```python/acrepl
b = 2
```
"""
        parsed = parse_response(text)
        kinds = [s.kind for s in parsed.segments]
        assert "quoted" in kinds
        assert _executable(parsed) == ["a = 1", "b = 2"]


class TestXmlCommandParsing:
    """Tests for XML DSL command detection in prose."""

    def test_self_closing_view(self) -> None:
        """Self-closing <view/> tag detected as xml segment."""
        text = """I'll show you the file.
<view name="v" path="main.py"/>
Here it is."""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 1
        assert '<view' in xml[0].content

    def test_self_closing_shell(self) -> None:
        """Self-closing <shell/> tag detected as xml segment."""
        text = """Running tests.
<shell cmd="pytest -v"/>
Done."""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 1
        assert '<shell' in xml[0].content

    def test_done_tag(self) -> None:
        """<done/> tag detected as xml segment."""
        text = """Task complete.
<done message="All finished"/>"""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 1
        assert '<done' in xml[0].content

    def test_multiple_xml_commands(self) -> None:
        """Multiple XML commands each become separate xml segments."""
        text = """Setting up.
<view name="v1" path="a.py"/>
<view name="v2" path="b.py"/>
Ready."""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 2

    def test_html_tags_not_detected(self) -> None:
        """Regular HTML tags should not be treated as DSL commands."""
        text = """Here is some <em>emphasized</em> text and a <div> block."""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 0

    def test_unknown_xml_tags_ignored(self) -> None:
        """XML tags not in the DSL set are ignored."""
        text = """Some <custom attr="val"/> tag."""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 0

    def test_xml_mixed_with_fenced_code(self) -> None:
        """XML commands and fenced code coexist."""
        text = """Setting up.
<view name="v" path="main.py"/>

```python/acrepl
x = v.content
```

<done message="finished"/>"""
        parsed = parse_response(text)
        exe = _executable(parsed)
        assert len(exe) == 3
        assert '<view' in exe[0]
        assert 'x = v.content' in exe[1]
        assert '<done' in exe[2]

    def test_xml_in_blockquote_not_detected(self) -> None:
        """XML inside a blockquote should stay as quoted text."""
        text = """> <view name="v" path="old.py"/>
Not code."""
        parsed = parse_response(text)
        xml_segs = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml_segs) == 0

    def test_link_tag(self) -> None:
        """<link> utility tag with closing tag on same line."""
        text = """Linking nodes.
<link parent="g1">child1</link>
Done."""
        parsed = parse_response(text)
        xml = [s for s in parsed.segments if s.kind == "xml"]
        assert len(xml) == 1
        assert '<link' in xml[0].content


class TestMessageActor:
    """Tests for actor field on messages."""

    def test_message_with_actor(self) -> None:
        """Messages can have an actor field."""
        msg = Message(role=Role.USER, content="hello", originator="user")
        assert msg.originator == "user"
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_message_without_actor(self) -> None:
        """Actor field defaults to None."""
        msg = Message(role=Role.ASSISTANT, content="hi")
        assert msg.originator is None

    def test_various_actors(self) -> None:
        """Different actor types."""
        actors = ["user", "agent", "agent:plan", "subagent:explorer", "tool:bash"]
        for actor in actors:
            msg = Message(role=Role.USER, content="test", originator=actor)
            assert msg.originator == actor


class TestProjection:
    """Tests for Projection dataclass."""

    def test_empty_projection_render(self) -> None:
        """Empty projection renders to empty string."""
        proj = Projection()
        assert proj.render() == ""

    def test_projection_with_sections(self) -> None:
        """Projection with sections renders content."""
        proj = Projection(
            sections=[
                ProjectionSection(
                    section_type="view",
                    source_id="v1",
                    content="=== test.py ===\n1 | print('hello')",
                    tokens_used=10,
                ),
                ProjectionSection(
                    section_type="conversation",
                    source_id="conv",
                    content="## Conversation\n**USER**: hi",
                    tokens_used=5,
                ),
            ]
        )

        rendered = proj.render()
        assert "test.py" in rendered
        assert "Conversation" in rendered
        assert "USER" in rendered

    def test_projection_skips_empty_sections(self) -> None:
        """Empty content sections are skipped."""
        proj = Projection(
            sections=[
                ProjectionSection(
                    section_type="view",
                    source_id="v1",
                    content="has content",
                    tokens_used=5,
                ),
                ProjectionSection(
                    section_type="view",
                    source_id="v2",
                    content="",  # Empty
                    tokens_used=0,
                ),
            ]
        )

        rendered = proj.render()
        assert "has content" in rendered
        # Should not have double newlines from empty section
        assert "\n\n\n" not in rendered


class TestDoneSignal:
    """Tests for the done() agent control function."""

    def test_done_sets_flag(self) -> None:
        """done() should set the done flag."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=".")
        assert not timeline.is_done()

        # Call done through namespace
        timeline._namespace["done"]("Task complete")

        assert timeline.is_done()
        assert timeline.get_done_message() == "Task complete"

    def test_done_reset(self) -> None:
        """reset_done() should clear the flag."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=".")
        timeline._namespace["done"]("First task")
        assert timeline.is_done()

        timeline.reset_done()
        assert not timeline.is_done()
        assert timeline.get_done_message() is None

    def test_done_without_message(self) -> None:
        """done() can be called without a message."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", context_graph=ContextGraph(), cwd=".")
        timeline._namespace["done"]()

        assert timeline.is_done()
        assert timeline.get_done_message() == ""


@pytest.mark.asyncio
async def test_clear_conversation() -> None:
    """Test clearing conversation history."""
    from activecontext import ActiveContext

    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Access internal session (AsyncSession wraps Session)
        internal = session._session

        # Simulate adding conversation (normally done via prompt)
        from activecontext.core.llm.provider import Message, Role

        internal._message_history.append(Message(role=Role.USER, content="test"))
        internal._message_history.append(Message(role=Role.ASSISTANT, content="response"))

        assert len(internal._message_history) == 2

        # Clear it via public API
        session.clear_message_history()
        assert len(internal._message_history) == 0


@pytest.mark.asyncio
async def test_get_context_objects() -> None:
    """Test get_context_objects method on Session."""
    from activecontext import ActiveContext

    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Create some views
        await session.execute('v1 = view("a.py")')
        await session.execute('v2 = view("b.py")')

        objects = session.get_context_objects()

        # Should have views (plus any initial context)
        assert len(objects) >= 2

        # All objects should have GetDigest
        for obj in objects.values():
            assert hasattr(obj, "GetDigest")
            digest = obj.GetDigest()
            assert "type" in digest
