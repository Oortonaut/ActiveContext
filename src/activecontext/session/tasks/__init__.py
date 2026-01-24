"""Task implementations for Session.

This module contains specialized task types:
- BlockingConversationTask: Sync request/response (e.g., MCP menu, permission dialogs)
- StreamingTask: Event-driven monitoring (e.g., file watcher)
"""

from __future__ import annotations

from activecontext.session.tasks.blocking_conversation import BlockingConversationTask

__all__ = [
    "BlockingConversationTask",
]
