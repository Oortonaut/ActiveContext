"""Comprehensive tests for src/activecontext/dashboard/data.py"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Any
from enum import Enum

from activecontext.dashboard.data import (
    get_llm_status,
    get_session_summary,
    get_context_data,
    get_timeline_data,
    get_projection_data,
    get_message_history_data,
    get_rendered_projection_data,
    get_client_capabilities_data,
    get_session_features_data,
    format_session_update,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock session with all required attributes."""
    session = MagicMock()
    session.session_id = "test-session-123"
    session.cwd = "/test/project"

    # Mock timeline
    session.timeline = MagicMock()
    session.timeline.get_statements.return_value = []
    session.timeline._executions = {}

    # Mock context graph (iterable)
    session.get_context_graph.return_value = iter([])

    # Mock projection
    projection = MagicMock()
    projection.sections = []
    projection.render.return_value = "rendered content"
    session.get_projection.return_value = projection

    # Mock llm
    session.llm = MagicMock()
    session.llm.model = "test-model"

    # Mock message history
    session._message_history = []

    return session


@pytest.fixture
def mock_model_info():
    """Create a mock model info object."""
    model = MagicMock()
    model.model_id = "claude-3-opus"
    model.name = "Claude 3 Opus"
    model.provider = "anthropic"
    model.description = "Most capable Claude model"
    return model


@pytest.fixture
def mock_node():
    """Create a mock node with GetDigest method."""
    node = MagicMock()
    node.GetDigest.return_value = {
        "type": "TextNode",
        "node_id": "text-node-1",
        "path": "/test/file.py",
        "state": "all",
    }
    node.parent_ids = set()
    node.children_ids = set()
    return node


@pytest.fixture
def mock_statement():
    """Create a mock statement."""
    stmt = MagicMock()
    stmt.statement_id = "stmt-1"
    stmt.index = 0
    stmt.source = "v = text('file.py')"
    stmt.timestamp = 1705312200.0
    return stmt


@pytest.fixture
def mock_execution():
    """Create a mock execution result."""
    execution = MagicMock()
    execution.status = MagicMock()
    execution.status.value = "success"
    execution.duration_ms = 42.5
    execution.exception = None
    return execution


@pytest.fixture
def mock_section():
    """Create a mock projection section."""
    section = MagicMock()
    section.section_type = "context"
    section.source_id = "node-1"
    section.tokens_used = 500
    section.state = MagicMock()
    section.state.name = "ALL"
    section.content = "section content"
    return section


# =============================================================================
# Tests for get_llm_status
# =============================================================================


class TestGetLLMStatus:
    """Tests for get_llm_status function."""

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_returns_current_model(self, mock_models, mock_providers, mock_model_info):
        """Should return the current model in the result."""
        mock_providers.return_value = ["anthropic", "openai"]
        mock_models.return_value = [mock_model_info]

        result = get_llm_status("claude-3-opus")

        assert result["current_model"] == "claude-3-opus"

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_returns_available_providers(self, mock_models, mock_providers):
        """Should return list of available providers."""
        mock_providers.return_value = ["anthropic", "openai", "gemini"]
        mock_models.return_value = []

        result = get_llm_status("test-model")

        assert result["available_providers"] == ["anthropic", "openai", "gemini"]

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_returns_available_models_as_dicts(self, mock_models, mock_providers, mock_model_info):
        """Should return list of available models as dictionaries."""
        mock_providers.return_value = ["anthropic"]
        mock_models.return_value = [mock_model_info]

        result = get_llm_status("claude-3-opus")

        assert len(result["available_models"]) == 1
        model_dict = result["available_models"][0]
        assert model_dict["model_id"] == "claude-3-opus"
        assert model_dict["name"] == "Claude 3 Opus"
        assert model_dict["provider"] == "anthropic"
        assert model_dict["description"] == "Most capable Claude model"

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_configured_true_when_providers_exist(self, mock_models, mock_providers):
        """Should set configured=True when providers are available."""
        mock_providers.return_value = ["anthropic"]
        mock_models.return_value = []

        result = get_llm_status("test")

        assert result["configured"] is True

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_configured_false_when_no_providers(self, mock_models, mock_providers):
        """Should set configured=False when no providers are available."""
        mock_providers.return_value = []
        mock_models.return_value = []

        result = get_llm_status("test")

        assert result["configured"] is False

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_handles_none_current_model(self, mock_models, mock_providers):
        """Should handle None as current_model."""
        mock_providers.return_value = []
        mock_models.return_value = []

        result = get_llm_status(None)

        assert result["current_model"] is None

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_handles_multiple_models(self, mock_models, mock_providers):
        """Should handle multiple models from different providers."""
        model1 = MagicMock()
        model1.model_id = "claude-3-opus"
        model1.name = "Claude 3 Opus"
        model1.provider = "anthropic"
        model1.description = "Claude model"

        model2 = MagicMock()
        model2.model_id = "gpt-4"
        model2.name = "GPT-4"
        model2.provider = "openai"
        model2.description = "OpenAI model"

        mock_providers.return_value = ["anthropic", "openai"]
        mock_models.return_value = [model1, model2]

        result = get_llm_status("claude-3-opus")

        assert len(result["available_models"]) == 2
        assert result["available_models"][0]["provider"] == "anthropic"
        assert result["available_models"][1]["provider"] == "openai"


# =============================================================================
# Tests for get_session_summary
# =============================================================================


class TestGetSessionSummary:
    """Tests for get_session_summary function."""

    def test_returns_session_id(self, mock_session):
        """Should return the session ID."""
        result = get_session_summary(mock_session, "model", "mode")

        assert result["session_id"] == "test-session-123"

    def test_returns_cwd_as_string(self, mock_session):
        """Should return the current working directory as a string."""
        from pathlib import Path
        mock_session.cwd = Path("/my/project/path")

        result = get_session_summary(mock_session, "model", "mode")

        # Path separators are OS-dependent
        assert result["cwd"] in ("/my/project/path", "\\my\\project\\path")
        assert isinstance(result["cwd"], str)

    def test_returns_model_from_parameter(self, mock_session):
        """Should return the provided model parameter."""
        result = get_session_summary(mock_session, "claude-3-opus", "mode")

        assert result["model"] == "claude-3-opus"

    def test_returns_model_from_session_llm_when_param_none(self, mock_session):
        """Should fallback to session.llm.model when model param is None."""
        mock_session.llm.model = "session-model"

        result = get_session_summary(mock_session, None, "mode")

        assert result["model"] == "session-model"

    def test_returns_none_model_when_no_llm(self, mock_session):
        """Should return None when no model param and no session.llm."""
        mock_session.llm = None

        result = get_session_summary(mock_session, None, "mode")

        assert result["model"] is None

    def test_returns_mode(self, mock_session):
        """Should return the provided mode."""
        result = get_session_summary(mock_session, "model", "plan")

        assert result["mode"] == "plan"

    def test_handles_none_mode(self, mock_session):
        """Should handle None mode."""
        result = get_session_summary(mock_session, "model", None)

        assert result["mode"] is None


# =============================================================================
# Tests for get_context_data
# =============================================================================


class TestGetContextData:
    """Tests for get_context_data function."""

    def test_returns_empty_for_no_nodes(self, mock_session):
        """Should return empty dict when no nodes exist."""
        mock_session.get_context_graph.return_value = iter([])

        result = get_context_data(mock_session)

        assert result["nodes_by_type"] == {}
        assert result["total"] == 0

    def test_groups_nodes_by_type(self, mock_session, mock_node):
        """Should group nodes by their type from GetDigest."""
        mock_session.get_context_graph.return_value = iter([mock_node])

        result = get_context_data(mock_session)

        assert "TextNode" in result["nodes_by_type"]
        assert len(result["nodes_by_type"]["TextNode"]) == 1
        assert result["total"] == 1

    def test_includes_parent_and_children_ids(self, mock_session, mock_node):
        """Should include parent_ids and children_ids in node digest."""
        mock_node.parent_ids = {"parent-1", "parent-2"}
        mock_node.children_ids = {"child-1"}
        mock_session.get_context_graph.return_value = iter([mock_node])

        result = get_context_data(mock_session)

        node_data = result["nodes_by_type"]["TextNode"][0]
        assert set(node_data["parent_ids"]) == {"parent-1", "parent-2"}
        assert node_data["children_ids"] == ["child-1"]

    def test_handles_nodes_without_parent_ids(self, mock_session, mock_node):
        """Should handle nodes without parent_ids attribute."""
        del mock_node.parent_ids
        mock_node.children_ids = set()
        mock_session.get_context_graph.return_value = iter([mock_node])

        result = get_context_data(mock_session)

        node_data = result["nodes_by_type"]["TextNode"][0]
        assert node_data["parent_ids"] == []

    def test_handles_nodes_without_children_ids(self, mock_session, mock_node):
        """Should handle nodes without children_ids attribute."""
        mock_node.parent_ids = set()
        del mock_node.children_ids
        mock_session.get_context_graph.return_value = iter([mock_node])

        result = get_context_data(mock_session)

        node_data = result["nodes_by_type"]["TextNode"][0]
        assert node_data["children_ids"] == []

    def test_handles_multiple_node_types(self, mock_session):
        """Should handle multiple different node types."""
        text_node = MagicMock()
        text_node.GetDigest.return_value = {"type": "TextNode", "node_id": "t1"}
        text_node.parent_ids = set()
        text_node.children_ids = set()

        group_node = MagicMock()
        group_node.GetDigest.return_value = {"type": "GroupNode", "node_id": "g1"}
        group_node.parent_ids = set()
        group_node.children_ids = set()

        shell_node = MagicMock()
        shell_node.GetDigest.return_value = {"type": "ShellNode", "node_id": "s1"}
        shell_node.parent_ids = set()
        shell_node.children_ids = set()

        mock_session.get_context_graph.return_value = iter([text_node, group_node, shell_node])

        result = get_context_data(mock_session)

        assert "TextNode" in result["nodes_by_type"]
        assert "GroupNode" in result["nodes_by_type"]
        assert "ShellNode" in result["nodes_by_type"]
        assert result["total"] == 3

    def test_handles_exception_getting_graph(self, mock_session):
        """Should return empty result when get_context_graph fails."""
        mock_session.get_context_graph.side_effect = Exception("Graph error")

        result = get_context_data(mock_session)

        assert result["nodes_by_type"] == {}
        assert result["total"] == 0

    def test_handles_node_without_GetDigest(self, mock_session):
        """Should skip nodes that fail to serialize."""
        bad_node = MagicMock()
        bad_node.GetDigest.side_effect = AttributeError("No GetDigest")

        good_node = MagicMock()
        good_node.GetDigest.return_value = {"type": "GoodNode", "node_id": "g1"}
        good_node.parent_ids = set()
        good_node.children_ids = set()

        mock_session.get_context_graph.return_value = iter([bad_node, good_node])

        result = get_context_data(mock_session)

        assert result["total"] == 2  # Both counted
        assert "GoodNode" in result["nodes_by_type"]  # Only good node serialized

    def test_handles_unknown_type_in_digest(self, mock_session):
        """Should use 'unknown' when type is missing from digest."""
        node = MagicMock()
        node.GetDigest.return_value = {"node_id": "n1"}  # No 'type' key
        node.parent_ids = set()
        node.children_ids = set()
        mock_session.get_context_graph.return_value = iter([node])

        result = get_context_data(mock_session)

        assert "unknown" in result["nodes_by_type"]


# =============================================================================
# Tests for get_timeline_data
# =============================================================================


class TestGetTimelineData:
    """Tests for get_timeline_data function."""

    def test_returns_empty_for_no_statements(self, mock_session):
        """Should return empty list when no statements exist."""
        mock_session.timeline.get_statements.return_value = []

        result = get_timeline_data(mock_session)

        assert result["statements"] == []
        assert result["count"] == 0

    def test_returns_statement_info(self, mock_session, mock_statement, mock_execution):
        """Should return statement information with execution details."""
        mock_session.timeline.get_statements.return_value = [mock_statement]
        mock_session.timeline._executions = {"stmt-1": [mock_execution]}

        result = get_timeline_data(mock_session)

        assert len(result["statements"]) == 1
        stmt = result["statements"][0]
        assert stmt["statement_id"] == "stmt-1"
        assert stmt["index"] == 0
        assert stmt["source"] == "v = text('file.py')"
        assert stmt["status"] == "success"
        assert stmt["duration_ms"] == 42.5
        assert stmt["has_error"] is False

    def test_returns_pending_status_when_no_executions(self, mock_session, mock_statement):
        """Should return pending status when statement has no executions."""
        mock_session.timeline.get_statements.return_value = [mock_statement]
        mock_session.timeline._executions = {}

        result = get_timeline_data(mock_session)

        stmt = result["statements"][0]
        assert stmt["status"] == "pending"
        assert stmt["duration_ms"] == 0
        assert stmt["has_error"] is False

    def test_returns_multiple_statements(self, mock_session, mock_execution):
        """Should return multiple statements in order."""
        stmt1 = MagicMock()
        stmt1.statement_id = "stmt-1"
        stmt1.index = 0
        stmt1.source = "first()"
        stmt1.timestamp = 1705312200.0

        stmt2 = MagicMock()
        stmt2.statement_id = "stmt-2"
        stmt2.index = 1
        stmt2.source = "second()"
        stmt2.timestamp = 1705312260.0

        mock_session.timeline.get_statements.return_value = [stmt1, stmt2]
        mock_session.timeline._executions = {
            "stmt-1": [mock_execution],
            "stmt-2": [mock_execution],
        }

        result = get_timeline_data(mock_session)

        assert len(result["statements"]) == 2
        assert result["statements"][0]["source"] == "first()"
        assert result["statements"][1]["source"] == "second()"
        assert result["count"] == 2

    def test_handles_statement_with_error(self, mock_session, mock_statement):
        """Should include error information in statement."""
        error_execution = MagicMock()
        error_execution.status = MagicMock()
        error_execution.status.value = "error"
        error_execution.duration_ms = 10.0
        error_execution.exception = Exception("NameError")

        mock_session.timeline.get_statements.return_value = [mock_statement]
        mock_session.timeline._executions = {"stmt-1": [error_execution]}

        result = get_timeline_data(mock_session)

        assert result["statements"][0]["has_error"] is True
        assert result["statements"][0]["status"] == "error"

    def test_uses_latest_execution(self, mock_session, mock_statement):
        """Should use the latest execution when multiple exist."""
        exec1 = MagicMock()
        exec1.status = MagicMock()
        exec1.status.value = "error"
        exec1.duration_ms = 10.0
        exec1.exception = Exception("First error")

        exec2 = MagicMock()
        exec2.status = MagicMock()
        exec2.status.value = "success"
        exec2.duration_ms = 50.0
        exec2.exception = None

        mock_session.timeline.get_statements.return_value = [mock_statement]
        mock_session.timeline._executions = {"stmt-1": [exec1, exec2]}

        result = get_timeline_data(mock_session)

        # Should use exec2 (latest)
        assert result["statements"][0]["status"] == "success"
        assert result["statements"][0]["duration_ms"] == 50.0
        assert result["statements"][0]["has_error"] is False

    def test_handles_exception(self, mock_session):
        """Should return empty result on exception."""
        mock_session.timeline = None

        result = get_timeline_data(mock_session)

        assert result["statements"] == []
        assert result["count"] == 0

    def test_skips_problematic_statements(self, mock_session, mock_statement, mock_execution):
        """Should skip statements that fail to process."""
        bad_stmt = MagicMock()
        bad_stmt.statement_id = "bad"
        # Accessing attributes raises exception
        type(bad_stmt).index = PropertyMock(side_effect=Exception("Bad attr"))

        mock_session.timeline.get_statements.return_value = [bad_stmt, mock_statement]
        mock_session.timeline._executions = {"stmt-1": [mock_execution]}

        result = get_timeline_data(mock_session)

        # Only good statement should be included
        assert result["count"] == 1
        assert result["statements"][0]["statement_id"] == "stmt-1"


# =============================================================================
# Tests for get_projection_data
# =============================================================================


class TestGetProjectionData:
    """Tests for get_projection_data function."""

    def test_returns_empty_for_no_sections(self, mock_session):
        """Should return empty sections when none exist."""
        projection = mock_session.get_projection.return_value
        projection.sections = []

        result = get_projection_data(mock_session)

        assert result["sections"] == []
        assert result["total_used"] == 0

    def test_returns_section_info(self, mock_session, mock_section):
        """Should return section information with token usage."""
        projection = mock_session.get_projection.return_value
        projection.sections = [mock_section]

        result = get_projection_data(mock_session)

        assert len(result["sections"]) == 1
        section = result["sections"][0]
        assert section["type"] == "context"
        assert section["source_id"] == "node-1"
        assert section["tokens_used"] == 500
        assert section["state"] == "all"

    def test_returns_total_tokens_used(self, mock_session):
        """Should return total tokens across all sections."""
        section1 = MagicMock()
        section1.section_type = "system"
        section1.source_id = "sys"
        section1.tokens_used = 200
        section1.state = MagicMock()
        section1.state.name = "DETAILS"

        section2 = MagicMock()
        section2.section_type = "context"
        section2.source_id = "ctx"
        section2.tokens_used = 800
        section2.state = MagicMock()
        section2.state.name = "ALL"

        projection = mock_session.get_projection.return_value
        projection.sections = [section1, section2]

        result = get_projection_data(mock_session)

        assert result["total_used"] == 1000
        assert len(result["sections"]) == 2

    def test_handles_section_without_state(self, mock_session):
        """Should handle sections with None state."""
        section = MagicMock()
        section.section_type = "context"
        section.source_id = "node-1"
        section.tokens_used = 100
        section.state = None

        projection = mock_session.get_projection.return_value
        projection.sections = [section]

        result = get_projection_data(mock_session)

        assert result["sections"][0]["state"] == "details"  # Default

    def test_handles_exception(self, mock_session):
        """Should return empty result on exception."""
        mock_session.get_projection.side_effect = Exception("Projection error")

        result = get_projection_data(mock_session)

        assert result["sections"] == []
        assert result["total_used"] == 0

    def test_skips_problematic_sections(self, mock_session, mock_section):
        """Should skip sections that fail to process."""
        bad_section = MagicMock()
        type(bad_section).section_type = PropertyMock(side_effect=Exception("Bad"))

        projection = mock_session.get_projection.return_value
        projection.sections = [bad_section, mock_section]

        result = get_projection_data(mock_session)

        # Only good section processed
        assert len(result["sections"]) == 1
        assert result["sections"][0]["type"] == "context"


# =============================================================================
# Tests for get_message_history_data
# =============================================================================


class TestGetMessageHistoryData:
    """Tests for get_message_history_data function."""

    def test_returns_empty_for_no_messages(self, mock_session):
        """Should return empty list when no messages exist."""
        mock_session._message_history = []

        result = get_message_history_data(mock_session)

        assert result["messages"] == []
        assert result["count"] == 0

    def test_handles_dict_messages(self, mock_session):
        """Should handle messages as dictionaries."""
        mock_session._message_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = get_message_history_data(mock_session)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["messages"][0]["id"] == "msg_0"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["id"] == "msg_1"
        assert result["count"] == 2

    def test_handles_object_messages_with_enum_role(self, mock_session):
        """Should handle message objects with Enum role."""
        class Role(Enum):
            USER = "user"
            ASSISTANT = "assistant"

        msg1 = MagicMock()
        msg1.role = Role.USER
        msg1.content = "Question"
        msg1.originator = None
        msg1.tool_name = None
        msg1.tool_args = None

        msg2 = MagicMock()
        msg2.role = Role.ASSISTANT
        msg2.content = "Answer"
        msg2.originator = "llm"
        msg2.tool_name = None
        msg2.tool_args = None

        mock_session._message_history = [msg1, msg2]

        result = get_message_history_data(mock_session)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["originator"] == "llm"

    def test_handles_mixed_message_formats(self, mock_session):
        """Should handle mixed dict and object messages."""
        obj_msg = MagicMock()
        obj_msg.role = "user"
        obj_msg.content = "Object message"
        obj_msg.originator = None
        obj_msg.tool_name = None
        obj_msg.tool_args = None

        mock_session._message_history = [
            {"role": "assistant", "content": "Dict message"},
            obj_msg,
        ]

        result = get_message_history_data(mock_session)

        assert len(result["messages"]) == 2
        assert result["count"] == 2

    def test_includes_tool_info_from_dict(self, mock_session):
        """Should include tool_name and tool_args from dict messages."""
        mock_session._message_history = [
            {
                "role": "assistant",
                "content": "Using tool",
                "tool_name": "read_file",
                "tool_args": {"path": "/test.py"},
            }
        ]

        result = get_message_history_data(mock_session)

        msg = result["messages"][0]
        assert msg["tool_name"] == "read_file"
        assert msg["tool_args"] == {"path": "/test.py"}

    def test_handles_missing_message_history_attr(self, mock_session):
        """Should handle session without _message_history."""
        del mock_session._message_history

        result = get_message_history_data(mock_session)

        assert result["messages"] == []
        assert result["count"] == 0

    def test_handles_none_message_history(self, mock_session):
        """Should handle None _message_history."""
        mock_session._message_history = None

        result = get_message_history_data(mock_session)

        # Should return empty since iterating None fails
        assert result["messages"] == []

    def test_skips_problematic_messages(self, mock_session):
        """Should skip messages that fail to process."""
        good_msg = {"role": "user", "content": "Hello"}
        bad_msg = MagicMock()
        type(bad_msg).role = PropertyMock(side_effect=Exception("Bad"))

        mock_session._message_history = [good_msg, bad_msg]

        result = get_message_history_data(mock_session)

        # Only good message processed
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "Hello"

    def test_handles_message_with_unknown_role(self, mock_session):
        """Should handle messages with missing role."""
        mock_session._message_history = [
            {"content": "No role"},  # Missing role key
        ]

        result = get_message_history_data(mock_session)

        assert result["messages"][0]["role"] == "unknown"


# =============================================================================
# Tests for get_rendered_projection_data
# =============================================================================


class TestGetRenderedProjectionData:
    """Tests for get_rendered_projection_data function."""

    @patch("activecontext.core.tokens.count_tokens")
    def test_returns_rendered_content(self, mock_count, mock_session):
        """Should return rendered projection content."""
        projection = mock_session.get_projection.return_value
        projection.render.return_value = "Full rendered projection"
        projection.sections = []
        mock_count.return_value = 100

        result = get_rendered_projection_data(mock_session)

        assert result["rendered"] == "Full rendered projection"

    @patch("activecontext.core.tokens.count_tokens")
    def test_returns_total_tokens(self, mock_count, mock_session):
        """Should return total token count."""
        projection = mock_session.get_projection.return_value
        projection.render.return_value = "content"
        projection.sections = []
        mock_count.return_value = 42

        result = get_rendered_projection_data(mock_session)

        assert result["total_tokens"] == 42

    @patch("activecontext.core.tokens.count_tokens")
    def test_returns_sections_with_content(self, mock_count, mock_session, mock_section):
        """Should return sections with their content."""
        projection = mock_session.get_projection.return_value
        projection.render.return_value = "rendered"
        projection.sections = [mock_section]
        mock_count.return_value = 50

        result = get_rendered_projection_data(mock_session)

        assert len(result["sections"]) == 1
        assert result["sections"][0]["content"] == "section content"
        assert result["sections"][0]["type"] == "context"
        assert result["section_count"] == 1

    def test_handles_render_exception(self, mock_session):
        """Should return error message when rendering fails."""
        mock_session.get_projection.side_effect = Exception("Render failed")

        result = get_rendered_projection_data(mock_session)

        assert "Error rendering projection" in result["rendered"]
        assert result["total_tokens"] == 0
        assert result["sections"] == []
        assert result["section_count"] == 0

    @patch("activecontext.core.tokens.count_tokens")
    def test_skips_problematic_sections(self, mock_count, mock_session, mock_section):
        """Should skip sections that fail to process."""
        bad_section = MagicMock()
        type(bad_section).section_type = PropertyMock(side_effect=Exception("Bad"))

        projection = mock_session.get_projection.return_value
        projection.render.return_value = "content"
        projection.sections = [bad_section, mock_section]
        mock_count.return_value = 10

        result = get_rendered_projection_data(mock_session)

        # Only good section included
        assert len(result["sections"]) == 1
        assert result["sections"][0]["type"] == "context"


# =============================================================================
# Tests for get_client_capabilities_data
# =============================================================================


class TestGetClientCapabilitiesData:
    """Tests for get_client_capabilities_data function."""

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_transport_info(self, mock_protocol, mock_client, mock_transport):
        """Should return transport information."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = {"name": "Rider", "version": "2024.3"}
        mock_protocol.return_value = "1.0"

        result = get_client_capabilities_data()

        assert result["transport"]["type"] == "stdio"
        assert result["transport"]["is_acp"] is False

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_is_acp_true_for_acp_transport(self, mock_protocol, mock_client, mock_transport):
        """Should set is_acp=True for acp transport type."""
        mock_transport.return_value = "acp"
        mock_client.return_value = {}
        mock_protocol.return_value = "1.0"

        result = get_client_capabilities_data()

        assert result["transport"]["is_acp"] is True

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_client_info(self, mock_protocol, mock_client, mock_transport):
        """Should return client information."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = {"name": "VSCode", "version": "1.85"}
        mock_protocol.return_value = "2.0"

        result = get_client_capabilities_data()

        assert result["client"]["name"] == "VSCode"
        assert result["client"]["version"] == "1.85"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_protocol_version(self, mock_protocol, mock_client, mock_transport):
        """Should return protocol version."""
        mock_transport.return_value = "direct"
        mock_client.return_value = None
        mock_protocol.return_value = "3.1"

        result = get_client_capabilities_data()

        assert result["protocol_version"] == "3.1"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_handles_no_client_info(self, mock_protocol, mock_client, mock_transport):
        """Should handle missing client info."""
        mock_transport.return_value = "direct"
        mock_client.return_value = None
        mock_protocol.return_value = "1.0"

        result = get_client_capabilities_data()

        assert result["client"] is None


# =============================================================================
# Tests for get_session_features_data
# =============================================================================


class TestGetSessionFeaturesData:
    """Tests for get_session_features_data function."""

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_model_from_param(self, mock_protocol, mock_client, mock_transport, mock_session):
        """Should return model from parameter when provided."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = None
        mock_protocol.return_value = "1.0"

        result = get_session_features_data(mock_session, "claude-opus", "plan")

        assert result["model"] == "claude-opus"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_model_from_session(self, mock_protocol, mock_client, mock_transport, mock_session):
        """Should return model from session.llm when param is None."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = None
        mock_protocol.return_value = "1.0"
        mock_session.llm.model = "session-model"

        result = get_session_features_data(mock_session, None, "plan")

        assert result["model"] == "session-model"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_mode(self, mock_protocol, mock_client, mock_transport, mock_session):
        """Should return mode from parameter."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = None
        mock_protocol.return_value = "1.0"

        result = get_session_features_data(mock_session, "model", "edit")

        assert result["mode"] == "edit"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_cwd(self, mock_protocol, mock_client, mock_transport, mock_session):
        """Should return cwd as string."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = None
        mock_protocol.return_value = "1.0"
        mock_session.cwd = "/my/path"

        result = get_session_features_data(mock_session, "model", "mode")

        assert result["cwd"] == "/my/path"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_title(self, mock_protocol, mock_client, mock_transport, mock_session):
        """Should return session title."""
        mock_transport.return_value = "stdio"
        mock_client.return_value = None
        mock_protocol.return_value = "1.0"
        mock_session.title = "My Session"

        result = get_session_features_data(mock_session, "model", "mode")

        assert result["title"] == "My Session"

    @patch("activecontext.dashboard.server.get_transport_type")
    @patch("activecontext.dashboard.server.get_client_info")
    @patch("activecontext.dashboard.server.get_protocol_version")
    def test_returns_transport_and_client_info(self, mock_protocol, mock_client, mock_transport, mock_session):
        """Should return transport and client information."""
        mock_transport.return_value = "acp"
        mock_client.return_value = {"name": "Rider", "version": "2024.3"}
        mock_protocol.return_value = "2.0"

        result = get_session_features_data(mock_session, "model", "mode")

        assert result["transport"]["type"] == "acp"
        assert result["transport"]["is_acp"] is True
        assert result["protocol_version"] == "2.0"
        assert result["client"]["name"] == "Rider"


# =============================================================================
# Tests for format_session_update
# =============================================================================


class TestFormatSessionUpdate:
    """Tests for format_session_update function."""

    def test_formats_websocket_message(self):
        """Should format WebSocket message with all fields."""
        result = format_session_update(
            kind="context_update",
            session_id="session-123",
            payload={"nodes": 5},
            timestamp=1705312200.0,
        )

        assert result["type"] == "update"
        assert result["kind"] == "context_update"
        assert result["session_id"] == "session-123"
        assert result["payload"] == {"nodes": 5}
        assert result["timestamp"] == 1705312200.0

    def test_formats_various_kinds(self):
        """Should format various update kinds."""
        kinds = ["timeline_update", "projection_update", "status_change", "error"]

        for kind in kinds:
            result = format_session_update(
                kind=kind,
                session_id="123",
                payload={},
                timestamp=0.0,
            )
            assert result["type"] == "update"
            assert result["kind"] == kind

    def test_handles_complex_payload(self):
        """Should handle complex nested payload."""
        payload = {
            "nodes": [
                {"id": "1", "type": "TextNode"},
                {"id": "2", "type": "GroupNode"},
            ],
            "metadata": {
                "count": 2,
                "updated": True,
            },
        }

        result = format_session_update(
            kind="update",
            session_id="123",
            payload=payload,
            timestamp=0.0,
        )

        assert result["payload"]["nodes"][0]["type"] == "TextNode"
        assert result["payload"]["metadata"]["count"] == 2

    def test_handles_empty_payload(self):
        """Should handle empty payload dict."""
        result = format_session_update(
            kind="heartbeat",
            session_id="123",
            payload={},
            timestamp=0.0,
        )

        assert result["payload"] == {}

    def test_handles_empty_session_id(self):
        """Should handle empty session_id."""
        result = format_session_update(
            kind="update",
            session_id="",
            payload={},
            timestamp=0.0,
        )

        assert result["session_id"] == ""

    def test_timestamp_is_float(self):
        """Should accept float timestamp."""
        result = format_session_update(
            kind="update",
            session_id="123",
            payload={},
            timestamp=1705312200.123456,
        )

        assert result["timestamp"] == 1705312200.123456

    def test_payload_with_special_characters(self):
        """Should handle payload with special characters."""
        payload = {
            "path": "C:\\Users\\Test\\file.py",
            "content": '<script>alert("xss")</script>',
            "newlines": "line1\nline2\r\nline3",
        }

        result = format_session_update(
            kind="update",
            session_id="123",
            payload=payload,
            timestamp=0.0,
        )

        assert result["payload"]["path"] == "C:\\Users\\Test\\file.py"
        assert result["payload"]["content"] == '<script>alert("xss")</script>'


# =============================================================================
# Integration-style tests
# =============================================================================


class TestDataModuleIntegration:
    """Integration-style tests for the data module."""

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    @patch("activecontext.core.tokens.count_tokens")
    def test_full_session_data_collection(
        self, mock_count, mock_models, mock_providers, mock_session, mock_node, mock_statement, mock_execution, mock_section
    ):
        """Should collect all session data without errors."""
        # Setup LLM mocks
        model_info = MagicMock()
        model_info.model_id = "claude-3-opus"
        model_info.name = "Claude 3 Opus"
        model_info.provider = "anthropic"
        model_info.description = "Test"
        mock_providers.return_value = ["anthropic"]
        mock_models.return_value = [model_info]
        mock_count.return_value = 100

        # Setup session mocks
        mock_session.get_context_graph.return_value = iter([mock_node])
        mock_session.timeline.get_statements.return_value = [mock_statement]
        mock_session.timeline._executions = {"stmt-1": [mock_execution]}

        projection = mock_session.get_projection.return_value
        projection.sections = [mock_section]
        projection.render.return_value = "Rendered"

        mock_session._message_history = [{"role": "user", "content": "Hello"}]

        # Collect all data
        llm_status = get_llm_status("claude-3-opus")
        summary = get_session_summary(mock_session, "claude-3-opus", "plan")
        context = get_context_data(mock_session)
        timeline = get_timeline_data(mock_session)
        projection_data = get_projection_data(mock_session)
        messages = get_message_history_data(mock_session)
        rendered = get_rendered_projection_data(mock_session)

        # Verify all returned valid data
        assert llm_status["current_model"] == "claude-3-opus"
        assert summary["session_id"] == "test-session-123"
        assert context["total"] == 1
        assert timeline["count"] == 1
        assert len(projection_data["sections"]) == 1
        assert messages["count"] == 1
        assert rendered["rendered"] == "Rendered"

    def test_handles_completely_empty_session(self, mock_session):
        """Should handle a session with no data gracefully."""
        mock_session.timeline.get_statements.return_value = []
        mock_session.timeline._executions = {}
        mock_session._message_history = []
        mock_session.get_context_graph.return_value = iter([])

        projection = mock_session.get_projection.return_value
        projection.sections = []
        projection.render.return_value = ""

        # All functions should work
        context = get_context_data(mock_session)
        timeline = get_timeline_data(mock_session)
        projection_data = get_projection_data(mock_session)
        messages = get_message_history_data(mock_session)

        assert context["total"] == 0
        assert timeline["statements"] == []
        assert projection_data["sections"] == []
        assert messages["messages"] == []


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for the data module."""

    def test_node_with_unknown_type(self, mock_session):
        """Should handle nodes without a type in digest."""
        node = MagicMock()
        node.GetDigest.return_value = {"node_id": "n1"}  # No 'type'
        node.parent_ids = set()
        node.children_ids = set()
        mock_session.get_context_graph.return_value = iter([node])

        result = get_context_data(mock_session)

        assert "unknown" in result["nodes_by_type"]
        assert result["total"] == 1

    def test_statement_with_very_long_source(self, mock_session, mock_execution):
        """Should handle statements with very long source strings."""
        stmt = MagicMock()
        stmt.statement_id = "long"
        stmt.index = 0
        stmt.source = "x = " + "a" * 10000
        stmt.timestamp = 0.0

        mock_session.timeline.get_statements.return_value = [stmt]
        mock_session.timeline._executions = {"long": [mock_execution]}

        result = get_timeline_data(mock_session)

        assert len(result["statements"]) == 1
        assert len(result["statements"][0]["source"]) > 10000

    def test_message_with_unicode_content(self, mock_session):
        """Should handle messages with unicode content."""
        mock_session._message_history = [
            {"role": "user", "content": "Hello 你好 مرحبا"},
            {"role": "assistant", "content": "Unicode: αβγδε"},
        ]

        result = get_message_history_data(mock_session)

        assert len(result["messages"]) == 2
        assert "你好" in result["messages"][0]["content"]
        assert "αβγδε" in result["messages"][1]["content"]

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_llm_status_with_empty_string_model(self, mock_models, mock_providers):
        """Should handle empty string as current model."""
        mock_providers.return_value = []
        mock_models.return_value = []

        result = get_llm_status("")

        assert result["current_model"] == ""

    def test_session_with_pathlib_cwd(self, mock_session):
        """Should convert pathlib.Path cwd to string."""
        from pathlib import Path
        mock_session.cwd = Path("/test/path")

        result = get_session_summary(mock_session, "model", "mode")

        # Path separators are OS-dependent
        assert result["cwd"] in ("/test/path", "\\test\\path")
        assert isinstance(result["cwd"], str)

    @patch("activecontext.dashboard.data.get_available_providers")
    @patch("activecontext.dashboard.data.get_available_models")
    def test_llm_status_handles_model_without_description(self, mock_models, mock_providers):
        """Should handle models without description attribute."""
        model = MagicMock()
        model.model_id = "test"
        model.name = "Test"
        model.provider = "test"
        model.description = None

        mock_providers.return_value = ["test"]
        mock_models.return_value = [model]

        result = get_llm_status("test")

        assert result["available_models"][0]["description"] is None

    def test_context_data_with_large_node_count(self, mock_session):
        """Should handle a large number of nodes efficiently."""
        nodes = []
        for i in range(1000):
            node = MagicMock()
            node.GetDigest.return_value = {"type": f"Type{i % 10}", "node_id": f"n{i}"}
            node.parent_ids = set()
            node.children_ids = set()
            nodes.append(node)

        mock_session.get_context_graph.return_value = iter(nodes)

        result = get_context_data(mock_session)

        assert result["total"] == 1000
        # 10 different types (Type0 through Type9)
        assert len(result["nodes_by_type"]) == 10
