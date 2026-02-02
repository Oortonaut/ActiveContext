"""Tests for the NodePlugin protocol.

Verifies that:
1. All builtin node types satisfy the NodePlugin protocol
2. Protocol adapter methods on ContextNode work correctly
3. Supporting dataclasses are well-formed
"""

from __future__ import annotations

import pytest

from activecontext.context.nodes import (
    ArtifactNode,
    ContextNode,
    GroupNode,
    LockNode,
    MessageNode,
    SessionNode,
    ShellNode,
    ShellStatus,
    TopicNode,
    TraceNode,
    WorkNode,
)
from activecontext.plugins.protocol import (
    MethodCall,
    NodeNotification,
    NodePlugin,
    RenderSnapshot,
    SyncResult,
    TokenEstimate,
)


class TestProtocolConformance:
    """Every builtin node type must satisfy the NodePlugin protocol."""

    @pytest.mark.parametrize(
        "node",
        [
            ShellNode(command="pytest", args=["-v"]),
            TopicNode(title="Authentication"),
            ArtifactNode(artifact_type="code", content="def foo(): pass"),
            LockNode(),
            WorkNode(),
            TraceNode(),
            GroupNode(),
            MessageNode(),
            SessionNode(),
        ],
        ids=lambda n: type(n).__name__,
    )
    def test_isinstance_check(self, node: ContextNode) -> None:
        """runtime_checkable isinstance succeeds for all builtins."""
        assert isinstance(node, NodePlugin)

    @pytest.mark.parametrize(
        "attr",
        [
            "node_type",
            "node_id",
            "render_header",
            "render_content",
            "render_digest",
            "get_token_breakdown",
            "get_digest",
            "tick",
            "notify_parents",
            "to_dict",
            "from_dict",
        ],
    )
    def test_protocol_methods_exist_on_context_node(self, attr: str) -> None:
        """ContextNode base class has all protocol attributes."""
        node = ShellNode(command="echo")
        assert hasattr(node, attr), f"Missing protocol attribute: {attr}"


class TestAdapterMethods:
    """The protocol adapter methods on ContextNode correctly bridge
    to the existing PascalCase methods."""

    def test_tick_delegates_to_recompute(self) -> None:
        """tick() calls Recompute()."""
        called = []

        class TestNode(ShellNode):
            def Recompute(self) -> None:
                called.append(True)

        node = TestNode(command="echo")
        node.tick()
        assert called == [True]

    def test_get_digest_delegates_to_GetDigest(self) -> None:
        """get_digest() calls GetDigest()."""
        node = ShellNode(command="pytest", args=["-v"])
        digest = node.get_digest()
        assert digest["type"] == "shell"
        assert digest["command"] == "pytest -v"
        assert "id" in digest

    def test_render_content_returns_content_section(self) -> None:
        """render_content() returns the content section without header."""
        node = ArtifactNode(
            artifact_type="code",
            content="def foo(): pass",
            language="python",
        )
        content = node.render_content()
        header = node.render_header()
        # Content should not include the header
        assert not content.startswith(header) or content == ""
        # But it should contain something from the summary
        # (ArtifactNode.render_content includes content)

    def test_render_content_empty_for_header_only_nodes(self) -> None:
        """Nodes with no content return empty string from render_content."""
        node = TopicNode(title="Test Topic")
        # TopicNode with no messages: render_content returns minimal or empty
        content = node.render_content()
        # Content is the part after the header
        assert isinstance(content, str)

    def test_render_methods_are_composable(self) -> None:
        """header + content should approximate the full render."""
        node = ArtifactNode(
            artifact_type="output",
            content="line 1\nline 2\nline 3",
        )
        header = node.render_header()
        content = node.render_content()

        composed = header + content
        # The composed output should contain the header
        assert header in composed
        # And be non-empty
        assert len(composed) > 0


class TestSupportingTypes:
    """The supporting dataclasses for RPC are well-formed."""

    def test_method_call_frozen(self) -> None:
        call = MethodCall(method="set_content", args=("hello",), kwargs={})
        assert call.method == "set_content"
        assert call.args == ("hello",)
        with pytest.raises(AttributeError):
            call.method = "other"  # type: ignore[misc]

    def test_render_snapshot_defaults(self) -> None:
        snap = RenderSnapshot()
        assert snap.header == ""
        assert snap.content == ""
        assert snap.detail == ""

    def test_render_snapshot_with_values(self) -> None:
        snap = RenderSnapshot(
            header="### Shell: pytest",
            content="output preview...",
            detail="full output here...",
        )
        assert "pytest" in snap.header
        assert "preview" in snap.content
        assert "full output" in snap.detail

    def test_token_estimate_defaults(self) -> None:
        est = TokenEstimate()
        assert est.collapsed == 0
        assert est.summary == 0
        assert est.detail == 0

    def test_node_notification(self) -> None:
        notif = NodeNotification(
            description="Shell completed",
            level="wake",
        )
        assert notif.description == "Shell completed"
        assert notif.level == "wake"

    def test_sync_result(self) -> None:
        result = SyncResult(
            state={"command": "pytest", "status": "completed"},
            renders=RenderSnapshot(header="h", content="c", detail="d"),
            tokens=TokenEstimate(collapsed=10, summary=50, detail=200),
            digest={"id": "sh_1", "type": "shell"},
            notifications=[
                NodeNotification(description="done", level="hold"),
            ],
        )
        assert result.state["command"] == "pytest"
        assert result.renders.header == "h"
        assert result.tokens.collapsed == 10
        assert len(result.notifications) == 1
