"""LSP Update Translator -- Converts SessionUpdate stream to LSP notifications.

This module translates ActiveContext session updates into LSP notifications,
supporting both standard LSP clients and LSP-aware clients that understand
custom $/activeContext/* methods.

Standard LSP clients receive:
- $/progress notifications for execution progress
- window/showMessage for errors

LSP-aware clients additionally receive:
- $/activeContext/output for response chunks
- $/activeContext/projectionReady when projection is built
- $/activeContext/nodeChanged when nodes are updated
"""

from __future__ import annotations

import logging
from typing import Any

from activecontext.session.protocols import SessionUpdate, UpdateKind

logger = logging.getLogger(__name__)


class UpdateTranslator:
    """Translates SessionUpdate stream to LSP notifications.

    Usage::

        translator = UpdateTranslator(lsp_server, lsp_aware=True)

        async for update in session.prompt("some prompt"):
            await translator.translate(update)
    """

    def __init__(self, lsp_server: Any, *, lsp_aware: bool = False) -> None:
        """Initialize the translator.

        Args:
            lsp_server: The LSPServer instance to send notifications through.
            lsp_aware: If True, send custom $/activeContext/* notifications.
                If False, only send standard LSP notifications.
        """
        self._server = lsp_server
        self._lsp_aware = lsp_aware
        self._progress_tokens: dict[str, str] = {}  # statement_id -> token

    async def translate(self, update: SessionUpdate) -> None:
        """Translate a SessionUpdate to the appropriate LSP notification(s).

        Args:
            update: The session update to translate.
        """
        kind = update.kind

        if kind == UpdateKind.RESPONSE_CHUNK:
            await self._handle_response_chunk(update)
        elif kind == UpdateKind.STATEMENT_EXECUTING:
            await self._handle_statement_executing(update)
        elif kind == UpdateKind.STATEMENT_EXECUTED:
            await self._handle_statement_executed(update)
        elif kind == UpdateKind.PROJECTION_READY:
            await self._handle_projection_ready(update)
        elif kind == UpdateKind.NODE_CHANGED:
            await self._handle_node_changed(update)
        elif kind == UpdateKind.ERROR:
            await self._handle_error(update)
        else:
            # Other update kinds (TICK_APPLIED, CONVERSATION_PROGRESS, etc.)
            # are not translated to LSP notifications
            logger.debug("Ignoring update kind %s", kind)

    async def _handle_response_chunk(self, update: SessionUpdate) -> None:
        """Handle RESPONSE_CHUNK update.

        LSP-aware: $/activeContext/output
        Standard: $/progress (if we can associate with a progress token)
        """
        chunk = update.payload.get("chunk", "")

        if self._lsp_aware:
            await self._server.send_notification(
                "$/activeContext/output",
                {
                    "sessionId": update.session_id,
                    "content": chunk,
                    "timestamp": update.timestamp,
                },
            )
        else:
            # For standard LSP clients, we could send $/progress,
            # but we don't have a good progress token here.
            # Just log for now.
            logger.debug("Response chunk (no LSP translation): %s", chunk[:100])

    async def _handle_statement_executing(self, update: SessionUpdate) -> None:
        """Handle STATEMENT_EXECUTING update.

        Sends $/progress begin notification with the statement being executed.
        """
        statement_id = update.payload.get("statement_id", "unknown")
        source = update.payload.get("source", "")

        # Create a unique progress token for this statement
        token = f"statement-{statement_id}"
        self._progress_tokens[statement_id] = token

        await self._server.send_notification(
            "$/progress",
            {
                "token": token,
                "value": {
                    "kind": "begin",
                    "title": "Executing statement",
                    "message": source[:100],  # Truncate long statements
                },
            },
        )

    async def _handle_statement_executed(self, update: SessionUpdate) -> None:
        """Handle STATEMENT_EXECUTED update.

        Sends $/progress end notification with the execution result.
        """
        statement_id = update.payload.get("statement_id", "unknown")
        status = update.payload.get("status", "ok")
        error = update.payload.get("error")

        # Get the progress token we created in STATEMENT_EXECUTING
        token = self._progress_tokens.pop(statement_id, f"statement-{statement_id}")

        message = "Completed"
        if status != "ok":
            message = f"Failed: {error}" if error else f"Failed ({status})"

        await self._server.send_notification(
            "$/progress",
            {
                "token": token,
                "value": {
                    "kind": "end",
                    "message": message,
                },
            },
        )

    async def _handle_projection_ready(self, update: SessionUpdate) -> None:
        """Handle PROJECTION_READY update.

        LSP-aware: $/activeContext/projectionReady
        Standard: No notification (not relevant to standard LSP clients)
        """
        if self._lsp_aware:
            await self._server.send_notification(
                "$/activeContext/projectionReady",
                {
                    "sessionId": update.session_id,
                    "timestamp": update.timestamp,
                    "projection": update.payload.get("projection", {}),
                },
            )

    async def _handle_node_changed(self, update: SessionUpdate) -> None:
        """Handle NODE_CHANGED update.

        LSP-aware: $/activeContext/nodeChanged
        Standard: No notification (not relevant to standard LSP clients)
        """
        if self._lsp_aware:
            await self._server.send_notification(
                "$/activeContext/nodeChanged",
                {
                    "sessionId": update.session_id,
                    "nodeId": update.payload.get("node_id", ""),
                    "change": update.payload.get("change", ""),
                    "timestamp": update.timestamp,
                },
            )

    async def _handle_error(self, update: SessionUpdate) -> None:
        """Handle ERROR update.

        Sends window/showMessage notification with error details.
        """
        error_message = update.payload.get("error", "Unknown error")
        details = update.payload.get("details", "")

        message = error_message
        if details:
            message = f"{error_message}: {details}"

        await self._server.send_notification(
            "window/showMessage",
            {
                "type": 1,  # Error = 1
                "message": message,
            },
        )
