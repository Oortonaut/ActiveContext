"""LSP Server -- Language Server Protocol lifecycle.

This module implements the core LSP server handling:
- initialize/initialized handshake with capability negotiation
- shutdown/exit lifecycle
- Request dispatch with method routing
- $/cancelRequest support
- JSON-RPC error responses for unknown methods

The server follows the LSP specification for the base protocol:
1. Client sends 'initialize' request with client capabilities
2. Server responds with server capabilities
3. Client sends 'initialized' notification
4. Normal message exchange begins
5. Client sends 'shutdown' request, server responds
6. Client sends 'exit' notification, server stops
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

from activecontext.transport.lsp.document_store import DocumentStore
from activecontext.transport.lsp.framing import read_message, write_message

logger = logging.getLogger(__name__)


class DocumentSyncHandler(Protocol):
    """Protocol for document sync callbacks.

    Implement this protocol to receive notifications about document
    open/change/close events and integrate with ContextGraph or other
    document management systems.
    """

    async def on_document_opened(self, uri: str, language_id: str, version: int, text: str) -> None:
        """Called when a document is opened.

        Args:
            uri: Document URI
            language_id: Language identifier (e.g., "python")
            version: Initial version number
            text: Document content
        """
        ...

    async def on_document_changed(self, uri: str, version: int, text: str) -> None:
        """Called when a document is changed.

        Args:
            uri: Document URI
            version: New version number
            text: Updated document content
        """
        ...

    async def on_document_closed(self, uri: str) -> None:
        """Called when a document is closed.

        Args:
            uri: Document URI
        """
        ...

    async def on_document_saved(self, uri: str, text: str | None = None) -> None:
        """Called when a document is saved.

        Args:
            uri: Document URI
            text: Optional document content (from didSave params if included)
        """
        ...


class LSPServer:
    """Core LSP server handling lifecycle and request dispatch.

    Usage::

        server = LSPServer()

        # Register custom handlers
        server.on("textDocument/didOpen", handle_did_open)
        server.on("textDocument/completion", handle_completion)

        # Add custom capabilities
        server.set_capabilities({"completionProvider": {"triggerCharacters": ["."]}})

        # Run the server
        await server.serve(reader, writer)
    """

    def __init__(self, *, document_sync_handler: DocumentSyncHandler | None = None) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._initialized = False
        self._shutdown_requested = False
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._capabilities: dict[str, Any] = {}
        self._document_store = DocumentStore()
        self._document_sync_handler = document_sync_handler

        # Register built-in document sync handlers
        self.on("textDocument/didOpen", self._handle_did_open)
        self.on("textDocument/didChange", self._handle_did_change)
        self.on("textDocument/didClose", self._handle_did_close)
        self.on("textDocument/didSave", self._handle_did_save)

    @property
    def initialized(self) -> bool:
        """Whether the server has completed the initialize/initialized handshake."""
        return self._initialized

    @property
    def shutdown_requested(self) -> bool:
        """Whether a shutdown has been requested."""
        return self._shutdown_requested

    @property
    def document_store(self) -> DocumentStore:
        """Access the document store for querying open documents."""
        return self._document_store

    def on(self, method: str, handler: Callable[..., Any]) -> None:
        """Register a handler for an LSP method.

        Args:
            method: The LSP method name (e.g., "textDocument/didOpen").
            handler: Async callable that receives params dict and returns result.
                For notifications (no id), the return value is ignored.
        """
        self._handlers[method] = handler

    def set_capabilities(self, capabilities: dict[str, Any]) -> None:
        """Set additional server capabilities to advertise during initialization.

        These are merged into the default capabilities returned in the
        initialize response. Call before serve() to take effect.

        Args:
            capabilities: Dictionary of LSP server capabilities.
        """
        self._capabilities.update(capabilities)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Main server loop -- reads and dispatches messages until exit.

        Args:
            reader: Async stream reader for incoming LSP messages.
            writer: Async stream writer for outgoing LSP messages.
        """
        self._reader = reader
        self._writer = writer

        try:
            while not self._shutdown_requested:
                msg = await read_message(self._reader)
                if msg is None:
                    logger.debug("EOF on input stream, stopping server")
                    break
                await self._dispatch(msg)
        finally:
            self._reader = None
            self._writer = None

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route incoming message to the appropriate handler.

        Handles built-in lifecycle methods (initialize, initialized, shutdown,
        exit, $/cancelRequest) directly. Other methods are routed to registered
        handlers. Unknown methods with an id receive a MethodNotFound error.
        """
        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            await self._handle_initialize(msg)
        elif method == "initialized":
            self._initialized = True
            logger.debug("Client sent initialized notification")
        elif method == "shutdown":
            await self._handle_shutdown(msg)
        elif method == "exit":
            self._shutdown_requested = True
            logger.debug("Client sent exit notification")
        elif method == "$/cancelRequest":
            await self._handle_cancel(msg)
        elif method in self._handlers:
            handler = self._handlers[method]
            result = await handler(msg.get("params", {}))
            if msg_id is not None:
                await self._send_response(msg_id, result)
        elif msg_id is not None:
            # Request with unknown method -- return error
            await self._send_error(msg_id, -32601, f"Method not found: {method}")
        else:
            # Unknown notification -- ignore per LSP spec
            logger.debug("Ignoring unknown notification: %s", method)

    async def _handle_initialize(self, msg: dict[str, Any]) -> None:
        """Handle initialize request -- negotiate capabilities.

        Builds the server capabilities object combining defaults with
        any capabilities set via set_capabilities(), and returns the
        InitializeResult to the client.
        """
        capabilities: dict[str, Any] = {
            "textDocumentSync": 1,  # Full sync
            "workspace": {"workspaceFolders": {"supported": True}},
        }
        capabilities.update(self._capabilities)

        result = {
            "capabilities": capabilities,
            "serverInfo": {
                "name": "activecontext",
                "version": "0.1.0",
            },
        }
        await self._send_response(msg["id"], result)
        logger.debug("Sent initialize response with capabilities: %s", capabilities)

    async def _handle_shutdown(self, msg: dict[str, Any]) -> None:
        """Handle shutdown request -- respond with null and flag for exit."""
        await self._send_response(msg["id"], None)
        self._shutdown_requested = True
        logger.debug("Shutdown requested, waiting for exit")

    async def _handle_cancel(self, msg: dict[str, Any]) -> None:
        """Handle $/cancelRequest notification (best effort).

        Currently a no-op since we process requests synchronously.
        When async request handling is added, this should cancel the
        matching in-flight request.
        """
        params = msg.get("params", {})
        cancel_id = params.get("id")
        logger.debug("Cancel requested for id=%s (best effort)", cancel_id)

    async def _send_response(self, id: int | str, result: Any) -> None:
        """Send a successful JSON-RPC response."""
        assert self._writer is not None, "Cannot send response: no writer"
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": id, "result": result}
        await write_message(self._writer, msg)

    async def _send_error(self, id: int | str, code: int, message: str) -> None:
        """Send a JSON-RPC error response."""
        assert self._writer is not None, "Cannot send error: no writer"
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": code, "message": message},
        }
        await write_message(self._writer, msg)

    async def send_notification(self, method: str, params: Any = None) -> None:
        """Send a notification to the client.

        Args:
            method: The notification method name.
            params: Optional notification parameters.
        """
        assert self._writer is not None, "Cannot send notification: no writer"
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await write_message(self._writer, msg)

    # -----------------------------------------------------------------------
    # Document sync handlers
    # -----------------------------------------------------------------------

    async def _handle_did_open(self, params: dict[str, Any]) -> None:
        """Handle textDocument/didOpen notification.

        Opens a document in the store and notifies the sync handler.
        """
        text_document = params.get("textDocument", {})
        uri = text_document.get("uri", "")
        language_id = text_document.get("languageId", "")
        version = text_document.get("version", 0)
        text = text_document.get("text", "")

        try:
            self._document_store.open(uri, language_id, version, text)
            logger.debug("Opened document: %s (version %d)", uri, version)

            if self._document_sync_handler:
                await self._document_sync_handler.on_document_opened(
                    uri, language_id, version, text
                )
        except ValueError as e:
            logger.warning("Failed to open document: %s", e)

    async def _handle_did_change(self, params: dict[str, Any]) -> None:
        """Handle textDocument/didChange notification.

        Applies changes to the document and notifies the sync handler.
        """
        text_document = params.get("textDocument", {})
        uri = text_document.get("uri", "")
        version = text_document.get("version", 0)
        content_changes = params.get("contentChanges", [])

        try:
            doc = self._document_store.change(uri, content_changes, version)
            logger.debug("Changed document: %s (version %d)", uri, version)

            if self._document_sync_handler:
                await self._document_sync_handler.on_document_changed(uri, version, doc.text)
        except ValueError as e:
            logger.warning("Failed to change document: %s", e)

    async def _handle_did_close(self, params: dict[str, Any]) -> None:
        """Handle textDocument/didClose notification.

        Closes the document and notifies the sync handler.
        """
        text_document = params.get("textDocument", {})
        uri = text_document.get("uri", "")

        try:
            self._document_store.close(uri)
            logger.debug("Closed document: %s", uri)

            if self._document_sync_handler:
                await self._document_sync_handler.on_document_closed(uri)
        except ValueError as e:
            logger.warning("Failed to close document: %s", e)

    async def _handle_did_save(self, params: dict[str, Any]) -> None:
        """Handle textDocument/didSave notification.

        Optionally notifies the sync handler with save event.
        """
        text_document = params.get("textDocument", {})
        uri = text_document.get("uri", "")
        text = params.get("text")  # Optional, depends on includeText

        logger.debug("Saved document: %s", uri)

        if self._document_sync_handler:
            await self._document_sync_handler.on_document_saved(uri, text)
