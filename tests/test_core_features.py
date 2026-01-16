"""Tests for core features: parsing, projection, messages."""

import pytest

from activecontext.core.llm.provider import Message, Role
from activecontext.core.prompts import parse_response
from activecontext.session.protocols import Projection, ProjectionSection


class TestCodeParsing:
    """Tests for python/acrepl code block detection."""

    def test_parse_acrepl_block(self) -> None:
        """Only python/acrepl blocks should be detected as code."""
        text = """Here's some explanation.

```python/acrepl
x = 1 + 1
```

And more text."""

        parsed = parse_response(text)
        assert parsed.has_code
        assert len(parsed.code_blocks) == 1
        assert parsed.code_blocks[0] == "x = 1 + 1"

    def test_ignore_plain_python_block(self) -> None:
        """Regular python blocks should NOT be executed."""
        text = """Here's an example:

```python
# This is just documentation
def example():
    pass
```

Not executed."""

        parsed = parse_response(text)
        assert not parsed.has_code
        assert len(parsed.code_blocks) == 0

    def test_ignore_other_languages(self) -> None:
        """Other language blocks should be ignored."""
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
        assert not parsed.has_code

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
        assert parsed.has_code
        assert len(parsed.code_blocks) == 2
        assert parsed.code_blocks[0] == "a = 1"
        assert parsed.code_blocks[1] == "b = 2"

    def test_mixed_blocks(self) -> None:
        """Mix of acrepl and regular python - only acrepl executes."""
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
        assert parsed.has_code
        assert len(parsed.code_blocks) == 1
        assert "real_code" in parsed.code_blocks[0]
        assert "example_only" not in parsed.code_blocks[0]

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


class TestMessageActor:
    """Tests for actor field on messages."""

    def test_message_with_actor(self) -> None:
        """Messages can have an actor field."""
        msg = Message(role=Role.USER, content="hello", actor="user")
        assert msg.actor == "user"
        assert msg.role == Role.USER
        assert msg.content == "hello"

    def test_message_without_actor(self) -> None:
        """Actor field defaults to None."""
        msg = Message(role=Role.ASSISTANT, content="hi")
        assert msg.actor is None

    def test_various_actors(self) -> None:
        """Different actor types."""
        actors = ["user", "agent", "agent:plan", "subagent:explorer", "tool:bash"]
        for actor in actors:
            msg = Message(role=Role.USER, content="test", actor=actor)
            assert msg.actor == actor


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

        timeline = Timeline("test-session", cwd=".")
        assert not timeline.is_done()

        # Call done through namespace
        timeline._namespace["done"]("Task complete")

        assert timeline.is_done()
        assert timeline.get_done_message() == "Task complete"

    def test_done_reset(self) -> None:
        """reset_done() should clear the flag."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=".")
        timeline._namespace["done"]("First task")
        assert timeline.is_done()

        timeline.reset_done()
        assert not timeline.is_done()
        assert timeline.get_done_message() is None

    def test_done_without_message(self) -> None:
        """done() can be called without a message."""
        from activecontext.session.timeline import Timeline

        timeline = Timeline("test-session", cwd=".")
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

        internal._conversation.append(Message(role=Role.USER, content="test"))
        internal._conversation.append(Message(role=Role.ASSISTANT, content="response"))

        assert len(internal._conversation) == 2

        # Clear it via public API
        session.clear_conversation()
        assert len(internal._conversation) == 0


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
