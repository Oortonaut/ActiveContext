"""LSP transport layer for ActiveContext.

This module provides Language Server Protocol support for ActiveContext,
enabling IDE integration via LSP instead of or alongside ACP.
"""

from activecontext.transport.lsp.framing import (
    LSPFramingError,
    parse_header,
    read_message,
    write_message,
)

__all__ = [
    "LSPFramingError",
    "parse_header",
    "read_message",
    "write_message",
]
