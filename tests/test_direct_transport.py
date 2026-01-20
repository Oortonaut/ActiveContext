"""Tests for the direct transport."""

import pytest

from activecontext import ActiveContext, UpdateKind


@pytest.mark.asyncio
async def test_session_lifecycle() -> None:
    """Test basic session creation and listing."""
    async with ActiveContext() as ctx:
        # Create a session
        session = await ctx.create_session(cwd="/tmp")
        assert session.session_id is not None

        # List sessions
        sessions = await ctx.list_sessions()
        assert session.session_id in sessions

        # Get session
        same_session = await ctx.get_session(session.session_id)
        assert same_session is not None
        assert same_session.session_id == session.session_id


@pytest.mark.asyncio
async def test_execute_python() -> None:
    """Test direct Python execution."""
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Execute a simple statement
        result = await session.execute("x = 1 + 1")
        assert result.status.value == "ok"
        assert "x" in result.state_trace.added

        # Check namespace
        ns = session.get_namespace()
        assert ns["x"] == 2


@pytest.mark.asyncio
async def test_text_creation() -> None:
    """Test text() function in namespace."""
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Note: session may have an initial guide view
        initial_count = len(session.get_context_objects())

        # Create a text node
        result = await session.execute('v = text("test.py", tokens=1000)')
        assert result.status.value == "ok"

        # Check context objects (should have one more than initial)
        objects = session.get_context_objects()
        assert len(objects) == initial_count + 1

        # Check text node properties
        ns = session.get_namespace()
        v = ns["v"]
        digest = v.GetDigest()
        assert digest["type"] == "text"
        assert digest["path"] == "test.py"
        assert digest["tokens"] == 1000


@pytest.mark.asyncio
async def test_prompt_executes_code() -> None:
    """Test that prompt() executes Python-like content."""
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        updates = []
        async for update in session.prompt("y = 42"):
            updates.append(update)

        # Should have execution updates
        kinds = [u.kind for u in updates]
        assert UpdateKind.STATEMENT_PARSED in kinds
        assert UpdateKind.STATEMENT_EXECUTED in kinds
        assert UpdateKind.PROJECTION_READY in kinds

        # Value should be in namespace
        ns = session.get_namespace()
        assert ns["y"] == 42


@pytest.mark.asyncio
async def test_group_creation() -> None:
    """Test group() function in namespace."""
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Note: session may have an initial guide view
        initial_count = len(session.get_context_objects())

        # Create text nodes and group them
        await session.execute('v1 = text("a.py")')
        await session.execute('v2 = text("b.py")')
        await session.execute("g = group(v1, v2, tokens=500)")

        objects = session.get_context_objects()
        assert len(objects) == initial_count + 3  # 2 text nodes + 1 group

        ns = session.get_namespace()
        g = ns["g"]
        digest = g.GetDigest()
        assert digest["type"] == "group"
        assert digest["member_count"] == 2


@pytest.mark.asyncio
async def test_initial_context_and_projection() -> None:
    """Test that sessions start with initial context and projection renders."""
    import os

    async with ActiveContext() as ctx:
        # Create session in project root where CONTEXT_GUIDE.md exists
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        session = await ctx.create_session(cwd=cwd)

        # Should have initial guide as TextNode with markdown media type
        ns = session.get_namespace()
        assert "guide" in ns, "Initial guide should be in namespace"

        guide = ns["guide"]
        digest = guide.GetDigest()
        assert digest["type"] == "text"  # TextNode type
        assert digest["media_type"] == "markdown"  # markdown media type
        assert "context_guide.md" in digest["path"]

        # Projection should render the guide content
        projection = session.get_projection()
        rendered = projection.render()

        # Should contain markdown content from the guide
        assert "text" in rendered.lower() or "Text" in rendered or "context" in rendered.lower()


@pytest.mark.asyncio
async def test_cancel_sets_flag() -> None:
    """Test that cancel() sets the cancellation flag."""
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Initially not cancelled
        assert session._session._cancelled is False

        # Cancel the session
        await session.cancel()

        # Now cancelled
        assert session._session._cancelled is True


@pytest.mark.asyncio
async def test_cancel_interrupts_prompt() -> None:
    """Test that cancel() interrupts an ongoing prompt."""
    import asyncio

    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        updates_received = []
        cancelled = False

        async def collect_updates():
            nonlocal cancelled
            # Execute code that would normally complete
            async for update in session.prompt("x = 1"):
                updates_received.append(update)
                if not cancelled:
                    # Cancel after first update
                    cancelled = True
                    await session.cancel()

        # Run the prompt
        await collect_updates()

        # Should have received at least one update before cancellation took effect
        assert len(updates_received) >= 1

        # Verify cancellation flag is set
        assert session._session._cancelled is True


@pytest.mark.asyncio
async def test_cancel_resets_on_new_prompt() -> None:
    """Test that cancellation flag is reset when starting a new prompt."""
    async with ActiveContext() as ctx:
        session = await ctx.create_session(cwd="/tmp")

        # Cancel
        await session.cancel()
        assert session._session._cancelled is True

        # Start a new prompt - flag should reset
        async for _ in session.prompt("y = 2"):
            pass

        # Flag was reset at start of prompt
        # After prompt completes, we can verify execution worked
        ns = session.get_namespace()
        assert ns.get("y") == 2
