"""Tests for MIME type support in dashboard and agent consumption.

Tests p2-003c implementation:
- AgentMessage schema with content_type and mime_type
- Dashboard data formatters extracting MIME type
- ACP agent handling different content block types
- MessageNode storing MIME type information
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from activecontext.agents.schema import AgentMessage
from activecontext.context.nodes import MessageNode
from activecontext.context.state import Expansion
from activecontext.dashboard.data import get_message_history_data
from activecontext.transport.acp.agent import extract_content_from_blocks

# Test imports for ACP content blocks
try:
    from acp.schema import AudioContentBlock, ImageContentBlock, TextContentBlock

    HAS_ACP = True
except ImportError:
    HAS_ACP = False
    TextContentBlock = None
    ImageContentBlock = None
    AudioContentBlock = None


class TestAgentMessageSchema:
    """Test AgentMessage schema with MIME type fields."""

    def test_agent_message_default_content_type(self):
        """Test that AgentMessage defaults to text content type."""
        msg = AgentMessage(
            id="msg_1",
            sender="agent_1",
            recipient="agent_2",
            content="Hello world",
        )
        assert msg.content_type == "text"
        assert msg.mime_type is None

    def test_agent_message_image_content_type(self):
        """Test AgentMessage with image content type."""
        msg = AgentMessage(
            id="msg_2",
            sender="agent_1",
            recipient="agent_2",
            content="base64_image_data",
            content_type="image",
            mime_type="image/png",
        )
        assert msg.content_type == "image"
        assert msg.mime_type == "image/png"

    def test_agent_message_to_dict_includes_mime_type(self):
        """Test that to_dict includes MIME type fields."""
        msg = AgentMessage(
            id="msg_3",
            sender="agent_1",
            recipient="agent_2",
            content="Hello",
            content_type="text",
            mime_type=None,
        )
        data = msg.to_dict()
        assert "content_type" in data
        assert data["content_type"] == "text"
        assert "mime_type" in data
        assert data["mime_type"] is None

    def test_agent_message_from_dict_with_mime_type(self):
        """Test that from_dict properly restores MIME type fields."""
        data = {
            "id": "msg_4",
            "sender": "agent_1",
            "recipient": "agent_2",
            "content": "audio_data",
            "created_at": "2024-01-01T00:00:00+00:00",
            "content_type": "audio",
            "mime_type": "audio/wav",
        }
        msg = AgentMessage.from_dict(data)
        assert msg.content_type == "audio"
        assert msg.mime_type == "audio/wav"

    def test_agent_message_from_dict_defaults(self):
        """Test that from_dict handles missing MIME type fields gracefully."""
        data = {
            "id": "msg_5",
            "sender": "agent_1",
            "recipient": "agent_2",
            "content": "Hello",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        msg = AgentMessage.from_dict(data)
        assert msg.content_type == "text"
        assert msg.mime_type is None


class TestMessageNodeMimeType:
    """Test MessageNode with MIME type support."""

    def test_message_node_default_content_type(self):
        """Test that MessageNode defaults to text content type."""
        node = MessageNode(
            node_id="msg_1",
            role="user",
            content="Hello world",
            originator="user",
        )
        assert node.content_type == "text"
        assert node.mime_type is None

    def test_message_node_image_content_type(self):
        """Test MessageNode with image content type."""
        node = MessageNode(
            node_id="msg_2",
            role="user",
            content="base64_image_data",
            originator="user",
            content_type="image",
            mime_type="image/jpeg",
        )
        assert node.content_type == "image"
        assert node.mime_type == "image/jpeg"

    def test_message_node_get_digest_includes_mime_type(self):
        """Test that GetDigest includes MIME type fields."""
        node = MessageNode(
            node_id="msg_3",
            role="user",
            content="Hello",
            originator="user",
            expansion=Expansion.ALL,
            content_type="text",
            mime_type=None,
        )
        digest = node.GetDigest()
        assert "content_type" in digest
        assert digest["content_type"] == "text"
        assert "mime_type" in digest
        assert digest["mime_type"] is None


class TestDashboardMimeTypeData:
    """Test dashboard data formatters with MIME type support."""

    def test_get_message_history_data_includes_mime_type(self):
        """Test that get_message_history_data includes MIME type fields."""
        # Create mock session with message history
        mock_session = MagicMock()
        mock_session._message_history = [
            {
                "role": "user",
                "content": "Hello",
                "content_type": "text",
                "mime_type": None,
            },
            {
                "role": "assistant",
                "content": "Response",
                "content_type": "text",
                "mime_type": None,
            },
        ]

        result = get_message_history_data(mock_session)
        assert result["count"] == 2
        assert all("content_type" in msg for msg in result["messages"])
        assert all("mime_type" in msg for msg in result["messages"])

    def test_get_message_history_data_with_image_message(self):
        """Test message history with image content."""
        mock_session = MagicMock()
        mock_session._message_history = [
            {
                "role": "user",
                "content": "base64_image_data",
                "content_type": "image",
                "mime_type": "image/png",
            },
        ]

        result = get_message_history_data(mock_session)
        assert result["count"] == 1
        msg = result["messages"][0]
        assert msg["content_type"] == "image"
        assert msg["mime_type"] == "image/png"


@pytest.mark.skipif(not HAS_ACP, reason="ACP not installed")
class TestACPContentBlockExtraction:
    """Test ACP content block extraction with MIME type awareness."""

    def test_extract_text_content_block(self):
        """Test extracting text from TextContentBlock."""
        blocks = [TextContentBlock(text="Hello world", type="text")]
        content, content_type, mime_type = extract_content_from_blocks(blocks)
        assert content == "Hello world"
        assert content_type == "text"
        assert mime_type is None

    def test_extract_image_content_block(self):
        """Test extracting image from ImageContentBlock."""
        blocks = [ImageContentBlock(data="base64_data", mime_type="image/png", type="image")]
        content, content_type, mime_type = extract_content_from_blocks(blocks)
        assert content == "base64_data"
        assert content_type == "image"
        assert mime_type == "image/png"

    def test_extract_audio_content_block(self):
        """Test extracting audio from AudioContentBlock."""
        blocks = [AudioContentBlock(data="audio_data", mime_type="audio/wav", type="audio")]
        content, content_type, mime_type = extract_content_from_blocks(blocks)
        assert content == "audio_data"
        assert content_type == "audio"
        assert mime_type == "audio/wav"

    def test_extract_multiple_text_blocks(self):
        """Test extracting multiple text blocks concatenates them."""
        blocks = [
            TextContentBlock(text="Hello ", type="text"),
            TextContentBlock(text="world", type="text"),
        ]
        content, content_type, mime_type = extract_content_from_blocks(blocks)
        assert content == "Hello world"
        assert content_type == "text"
        assert mime_type is None

    def test_extract_mixed_blocks_prefers_last_type(self):
        """Test that mixed blocks use the last block's type."""
        blocks = [
            TextContentBlock(text="Text first", type="text"),
            ImageContentBlock(data="image_data", mime_type="image/jpeg", type="image"),
        ]
        content, content_type, mime_type = extract_content_from_blocks(blocks)
        # Content is concatenated
        assert "Text first" in content
        assert "image_data" in content
        # Type is from the last block
        assert content_type == "image"
        assert mime_type == "image/jpeg"
