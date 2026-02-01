"""LSP transport layer for ActiveContext.

This module provides Language Server Protocol support for ActiveContext,
enabling IDE integration via LSP instead of or alongside ACP.
"""

from activecontext.transport.lsp.document_store import Document, DocumentStore
from activecontext.transport.lsp.framing import (
    LSPFramingError,
    parse_header,
    read_message,
    write_message,
)
from activecontext.transport.lsp.server import DocumentSyncHandler, LSPServer
from activecontext.transport.lsp.update_translator import UpdateTranslator

__all__ = [
    "Document",
    "DocumentStore",
    "DocumentSyncHandler",
    "LSPFramingError",
    "LSPServer",
    "UpdateTranslator",
    "parse_header",
    "read_message",
    "write_message",
]
