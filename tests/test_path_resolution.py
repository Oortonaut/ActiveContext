"""Tests for cross-platform path resolution.

Tests coverage for:
- Session.resolve_path() method
- Session._expand_path_roots() method
- Cross-platform path root expansion
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from activecontext.core.llm.litellm_provider import LiteLLMProvider
from activecontext.session.session_manager import Session, SessionManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock(spec=LiteLLMProvider)
    provider.model = "test-model"
    provider.stream = AsyncMock()
    return provider


@pytest.fixture
async def session_manager(mock_llm_provider):
    """Create SessionManager with mock LLM."""
    manager = SessionManager(default_llm=mock_llm_provider)
    yield manager
    # Cleanup
    for session_id in list(manager._sessions.keys()):
        await manager.close_session(session_id)


@pytest.fixture
async def test_session(session_manager):
    """Create a test session with a known cwd."""
    session = await session_manager.create_session(cwd="/test/project")
    return session


# =============================================================================
# Path Root Expansion Tests
# =============================================================================


class TestPathRootExpansion:
    """Test path root prefix expansion."""

    @pytest.mark.asyncio
    async def test_tilde_expansion(self, test_session):
        """Test ~ expands to user home directory."""
        home = os.path.expanduser("~")
        resolved, content = test_session.resolve_path("~/documents/file.txt")

        assert content is None
        expected = os.path.normpath(os.path.join(home, "documents", "file.txt"))
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_tilde_alone(self, test_session):
        """Test ~ alone expands to home directory."""
        home = os.path.expanduser("~")
        resolved, content = test_session.resolve_path("~")

        assert content is None
        assert resolved == os.path.normpath(home)

    @pytest.mark.asyncio
    async def test_home_brace_syntax(self, test_session):
        """Test {home} expands to user home directory."""
        home = os.path.expanduser("~")
        resolved, content = test_session.resolve_path("{home}/config.yaml")

        assert content is None
        expected = os.path.normpath(os.path.join(home, "config.yaml"))
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_home_brace_case_insensitive(self, test_session):
        """Test {HOME} and {Home} also work."""
        home = os.path.expanduser("~")

        resolved1, _ = test_session.resolve_path("{HOME}/file.txt")
        resolved2, _ = test_session.resolve_path("{Home}/file.txt")

        expected = os.path.normpath(os.path.join(home, "file.txt"))
        assert resolved1 == expected
        assert resolved2 == expected

    @pytest.mark.asyncio
    async def test_dollar_home_unix_style(self, test_session):
        """Test $HOME expands to user home directory (Unix style)."""
        home = os.path.expanduser("~")
        resolved, content = test_session.resolve_path("$HOME/scripts/run.sh")

        assert content is None
        expected = os.path.normpath(os.path.join(home, "scripts", "run.sh"))
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_userprofile_windows_style(self, test_session):
        """Test %USERPROFILE% expands to user home directory (Windows style)."""
        home = os.path.expanduser("~")
        resolved, content = test_session.resolve_path("%USERPROFILE%\\Documents\\file.txt")

        assert content is None
        expected = os.path.normpath(os.path.join(home, "Documents", "file.txt"))
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_userprofile_case_insensitive(self, test_session):
        """Test %USERPROFILE% is case-insensitive."""
        home = os.path.expanduser("~")

        resolved1, _ = test_session.resolve_path("%userprofile%/file.txt")
        resolved2, _ = test_session.resolve_path("%UserProfile%/file.txt")

        expected = os.path.normpath(os.path.join(home, "file.txt"))
        assert resolved1 == expected
        assert resolved2 == expected


class TestCwdPathRoots:
    """Test CWD/PROJECT path root expansion."""

    @pytest.mark.asyncio
    async def test_cwd_brace_syntax(self, test_session):
        """Test {cwd} expands to session working directory."""
        resolved, content = test_session.resolve_path("{cwd}/src/main.py")

        assert content is None
        expected = os.path.normpath("/test/project/src/main.py")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_cwd_brace_case_insensitive(self, test_session):
        """Test {CWD} and {Cwd} also work."""
        resolved1, _ = test_session.resolve_path("{CWD}/file.txt")
        resolved2, _ = test_session.resolve_path("{Cwd}/file.txt")

        expected = os.path.normpath("/test/project/file.txt")
        assert resolved1 == expected
        assert resolved2 == expected

    @pytest.mark.asyncio
    async def test_dollar_cwd_unix_style(self, test_session):
        """Test $CWD expands to session working directory."""
        resolved, content = test_session.resolve_path("$CWD/tests/test_main.py")

        assert content is None
        expected = os.path.normpath("/test/project/tests/test_main.py")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_project_brace_syntax(self, test_session):
        """Test {PROJECT} expands to project root (same as cwd)."""
        resolved, content = test_session.resolve_path("{PROJECT}/README.md")

        assert content is None
        expected = os.path.normpath("/test/project/README.md")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_project_case_insensitive(self, test_session):
        """Test {project} and {Project} also work."""
        resolved1, _ = test_session.resolve_path("{project}/file.txt")
        resolved2, _ = test_session.resolve_path("{Project}/file.txt")

        expected = os.path.normpath("/test/project/file.txt")
        assert resolved1 == expected
        assert resolved2 == expected

    @pytest.mark.asyncio
    async def test_cwd_alone(self, test_session):
        """Test {cwd} alone returns the cwd."""
        resolved, content = test_session.resolve_path("{cwd}")

        assert content is None
        assert resolved == os.path.normpath("/test/project")


class TestPathNormalization:
    """Test path separator normalization."""

    @pytest.mark.asyncio
    async def test_forward_slash_normalization(self, test_session):
        """Test forward slashes are normalized to native separator."""
        resolved, _ = test_session.resolve_path("{cwd}/src/module/file.py")

        # os.path.normpath handles separator conversion
        expected = os.path.normpath("/test/project/src/module/file.py")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_backslash_normalization(self, test_session):
        """Test backslashes are normalized to native separator."""
        resolved, _ = test_session.resolve_path("{cwd}\\src\\module\\file.py")

        expected = os.path.normpath("/test/project/src/module/file.py")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_mixed_separator_normalization(self, test_session):
        """Test mixed separators are normalized."""
        resolved, _ = test_session.resolve_path("{cwd}/src\\module/file.py")

        expected = os.path.normpath("/test/project/src/module/file.py")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_regular_path_normalization(self, test_session):
        """Test regular paths are also normalized."""
        resolved, _ = test_session.resolve_path("./src/../lib/utils.py")

        expected = os.path.normpath("./src/../lib/utils.py")
        assert resolved == expected


class TestPromptsPrefixPreserved:
    """Test that @prompts/ prefix still works."""

    @pytest.mark.asyncio
    async def test_prompts_prefix_returns_content(self, test_session):
        """Test @prompts/ prefix returns path and content."""
        resolved, content = test_session.resolve_path("@prompts/dsl_reference.md")

        assert resolved == "@prompts/dsl_reference"
        assert content is not None
        assert len(content) > 0
        assert "DSL" in content or "text(" in content  # Sanity check

    @pytest.mark.asyncio
    async def test_prompts_without_extension(self, test_session):
        """Test @prompts/ works without .md extension."""
        resolved, content = test_session.resolve_path("@prompts/dsl_reference")

        assert resolved == "@prompts/dsl_reference"
        assert content is not None


class TestRegularPaths:
    """Test that regular paths pass through unchanged (except normalization)."""

    @pytest.mark.asyncio
    async def test_relative_path_passthrough(self, test_session):
        """Test relative paths pass through with normalization."""
        resolved, content = test_session.resolve_path("src/main.py")

        assert content is None
        assert resolved == os.path.normpath("src/main.py")

    @pytest.mark.asyncio
    async def test_absolute_path_passthrough(self, test_session):
        """Test absolute paths pass through with normalization."""
        if os.name == "nt":
            path = "C:\\Users\\test\\file.txt"
            expected = os.path.normpath("C:\\Users\\test\\file.txt")
        else:
            path = "/home/test/file.txt"
            expected = os.path.normpath("/home/test/file.txt")

        resolved, content = test_session.resolve_path(path)

        assert content is None
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_dot_path_normalization(self, test_session):
        """Test . and .. in paths are normalized."""
        resolved, content = test_session.resolve_path("./src/../config/settings.yaml")

        assert content is None
        expected = os.path.normpath("./src/../config/settings.yaml")
        assert resolved == expected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_after_root(self, test_session):
        """Test path root with trailing slash but no path after."""
        home = os.path.expanduser("~")
        resolved, _ = test_session.resolve_path("~/")

        assert resolved == os.path.normpath(home)

    @pytest.mark.asyncio
    async def test_double_slash_after_root(self, test_session):
        """Test double slash after root is normalized."""
        resolved, _ = test_session.resolve_path("{cwd}//src//file.py")

        expected = os.path.normpath("/test/project/src/file.py")
        assert resolved == expected

    @pytest.mark.asyncio
    async def test_similar_prefix_not_matched(self, test_session):
        """Test that similar but not exact prefixes don't match."""
        # ~user should not match ~ (it's a different path on Unix)
        # {homedir} should not match {home}
        # $HOME_VAR should not match $HOME

        resolved1, _ = test_session.resolve_path("{homedir}/file.txt")
        resolved2, _ = test_session.resolve_path("$HOME_VAR/file.txt")

        # These should pass through as-is (with normalization)
        assert resolved1 == os.path.normpath("{homedir}/file.txt")
        assert resolved2 == os.path.normpath("$HOME_VAR/file.txt")
