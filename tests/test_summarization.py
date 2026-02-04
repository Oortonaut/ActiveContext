"""Tests for TextNode LLM summarization."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import TextNode
from activecontext.core.llm.provider import CompletionResult, LLMProvider, Role
from activecontext.session.timeline import Timeline


@pytest.fixture
def mock_llm_provider() -> AsyncMock:
    """Create a mock LLM provider."""
    provider = AsyncMock(spec=LLMProvider)
    provider.model = "test-model"
    provider.complete.return_value = CompletionResult(
        content="This is a test summary of the file.",
        finish_reason="stop",
        usage={"prompt_tokens": 100, "completion_tokens": 10},
    )
    return provider


@pytest.fixture
def context_graph() -> ContextGraph:
    """Create a context graph."""
    return ContextGraph()


@pytest.fixture
async def timeline_with_llm(
    context_graph: ContextGraph,
    mock_llm_provider: AsyncMock,
    tmp_path: Path,
) -> Timeline:
    """Create a Timeline with LLM provider."""
    timeline = Timeline(
        session_id="test-session",
        context_graph=context_graph,
        cwd=str(tmp_path),
        llm_provider=mock_llm_provider,
    )
    return timeline


class TestTextNodeSummarization:
    """Tests for TextNode LLM summarization feature."""

    @pytest.mark.asyncio
    async def test_summarize_generates_summary(
        self,
        timeline_with_llm: Timeline,
        context_graph: ContextGraph,
        tmp_path: Path,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test that summarize() generates and caches a summary."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")

        # Create TextNode
        node = TextNode(path=str(test_file))
        context_graph.add_node(node)

        # Call summarize
        summary = await timeline_with_llm._summarize(node)

        # Verify summary was generated
        assert summary == "This is a test summary of the file."
        assert node.cached_summary == summary
        assert not node.summary_stale
        assert node.content_hash is not None

        # Verify LLM was called
        mock_llm_provider.complete.assert_called_once()
        call_args = mock_llm_provider.complete.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0].role == Role.USER
        assert "test.py" in messages[0].content

    @pytest.mark.asyncio
    async def test_summarize_uses_cached_summary(
        self,
        timeline_with_llm: Timeline,
        context_graph: ContextGraph,
        tmp_path: Path,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test that summarize() uses cached summary when available."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")

        # Create TextNode with cached summary
        node = TextNode(path=str(test_file))
        context_graph.add_node(node)

        # First call to generate and cache
        await timeline_with_llm._summarize(node)
        first_summary = node.cached_summary

        # Reset mock
        mock_llm_provider.complete.reset_mock()

        # Second call should use cache
        summary = await timeline_with_llm._summarize(node)

        # Verify cached summary was used (LLM not called again)
        assert summary == "This is a test summary of the file."
        assert summary == first_summary
        assert not node.summary_stale
        mock_llm_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_force_regenerates(
        self,
        timeline_with_llm: Timeline,
        context_graph: ContextGraph,
        tmp_path: Path,
        mock_llm_provider: AsyncMock,
    ) -> None:
        """Test that summarize(force=True) regenerates summary."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")

        # Create TextNode with cached summary
        node = TextNode(path=str(test_file))
        context_graph.add_node(node)

        # First call
        await timeline_with_llm._summarize(node)
        mock_llm_provider.complete.reset_mock()

        # Force regeneration
        summary = await timeline_with_llm._summarize(node, force=True)

        # Verify LLM was called again
        assert summary == "This is a test summary of the file."
        mock_llm_provider.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_without_llm_provider(
        self,
        context_graph: ContextGraph,
        tmp_path: Path,
    ) -> None:
        """Test that summarize() raises error when LLM provider is not available."""
        timeline = Timeline(
            session_id="test-session",
            context_graph=context_graph,
            cwd=str(tmp_path),
            llm_provider=None,
        )

        # Create TextNode
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")
        node = TextNode(path=str(test_file))
        context_graph.add_node(node)

        # Should raise ValueError
        with pytest.raises(ValueError, match="LLM provider not available"):
            await timeline._summarize(node)

    @pytest.mark.asyncio
    async def test_summarize_non_text_node_raises_error(
        self,
        timeline_with_llm: Timeline,
        context_graph: ContextGraph,
    ) -> None:
        """Test that summarize() raises error for non-TextNode."""
        from activecontext.context.nodes import GroupNode

        node = GroupNode()
        context_graph.add_node(node)

        with pytest.raises(ValueError, match="only works with TextNode"):
            await timeline_with_llm._summarize(node)

    @pytest.mark.asyncio
    async def test_textnode_render_summary_uses_cache(
        self,
        context_graph: ContextGraph,
        tmp_path: Path,
    ) -> None:
        """Test that TextNode.render_content() uses cached summary."""
        # Create TextNode with cached summary
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")
        node = TextNode(path=str(test_file))
        node.cached_summary = "Test summary"
        node.summary_stale = False
        context_graph.add_node(node)

        # Render content
        rendered = node.render_content(cwd=str(tmp_path))

        # Should contain the cached summary
        assert "Test summary" in rendered

    @pytest.mark.asyncio
    async def test_textnode_render_summary_stale(
        self,
        context_graph: ContextGraph,
        tmp_path: Path,
    ) -> None:
        """Test that TextNode.render_content() does not use stale summary."""
        # Create TextNode with stale summary
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")
        node = TextNode(path=str(test_file))
        node.cached_summary = "Old summary"
        node.summary_stale = True
        context_graph.add_node(node)

        # Render content
        rendered = node.render_content(cwd=str(tmp_path))

        # Should not use the stale summary text
        assert "Old summary" not in rendered

    @pytest.mark.asyncio
    async def test_textnode_token_breakdown_includes_summary(
        self,
        context_graph: ContextGraph,
        tmp_path: Path,
    ) -> None:
        """Test that TextNode.get_token_breakdown() includes summary tokens."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")
        node = TextNode(
            path=str(test_file),
            start_line=1,
            end_line=2,
        )
        node.cached_summary = "This is a test summary with several words in it."
        context_graph.add_node(node)

        breakdown = node.get_token_breakdown()

        # Content tokens should be non-zero when summary exists
        assert breakdown.content > 0
        assert breakdown.title > 0
        assert breakdown.detail > 0
