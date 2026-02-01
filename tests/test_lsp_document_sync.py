"""Tests for LSP server document sync handlers."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from activecontext.transport.lsp.server import LSPServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lsp_frame(msg: dict[str, Any]) -> bytes:
    """Encode a JSON-RPC message with Content-Length framing."""
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _feed_messages(reader: asyncio.StreamReader, *messages: dict[str, Any]) -> None:
    """Feed one or more LSP-framed messages into a StreamReader."""
    for msg in messages:
        reader.feed_data(_make_lsp_frame(msg))
    reader.feed_eof()


class MockTransport:
    """Minimal mock transport that captures written bytes."""

    def __init__(self) -> None:
        self.data = bytearray()

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default

    def is_closing(self) -> bool:
        return False

    def write(self, data: bytes) -> None:
        self.data.extend(data)

    def close(self) -> None:
        pass


def _make_mock_writer() -> tuple[asyncio.StreamWriter, MockTransport]:
    """Create a StreamWriter backed by MockTransport."""
    mock_transport = MockTransport()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    writer = asyncio.StreamWriter(mock_transport, protocol, reader, asyncio.get_event_loop())
    return writer, mock_transport


class MockDocumentSyncHandler:
    """Mock handler that records all document sync events."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, str, int, str]] = []
        self.changed: list[tuple[str, int, str]] = []
        self.closed: list[str] = []
        self.saved: list[tuple[str, str | None]] = []

    async def on_document_opened(self, uri: str, language_id: str, version: int, text: str) -> None:
        self.opened.append((uri, language_id, version, text))

    async def on_document_changed(self, uri: str, version: int, text: str) -> None:
        self.changed.append((uri, version, text))

    async def on_document_closed(self, uri: str) -> None:
        self.closed.append(uri)

    async def on_document_saved(self, uri: str, text: str | None = None) -> None:
        self.saved.append((uri, text))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDidOpen:
    """Test textDocument/didOpen handling."""

    @pytest.mark.asyncio
    async def test_did_open_stores_document(self) -> None:
        """didOpen notification stores document in document store."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "print('hello')",
                    }
                },
            },
        )

        await server.serve(reader, writer)

        assert server.document_store.is_open("file:///test.py")
        doc = server.document_store.get("file:///test.py")
        assert doc is not None
        assert doc.language_id == "python"
        assert doc.version == 1
        assert doc.text == "print('hello')"

    @pytest.mark.asyncio
    async def test_did_open_calls_handler(self) -> None:
        """didOpen notification calls the document sync handler."""
        handler = MockDocumentSyncHandler()
        server = LSPServer(document_sync_handler=handler)
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "content",
                    }
                },
            },
        )

        await server.serve(reader, writer)

        assert len(handler.opened) == 1
        uri, language_id, version, text = handler.opened[0]
        assert uri == "file:///test.py"
        assert language_id == "python"
        assert version == 1
        assert text == "content"

    @pytest.mark.asyncio
    async def test_did_open_multiple_documents(self) -> None:
        """Multiple didOpen notifications track multiple documents."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///a.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "a",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///b.ts",
                        "languageId": "typescript",
                        "version": 1,
                        "text": "b",
                    }
                },
            },
        )

        await server.serve(reader, writer)

        assert server.document_store.count() == 2
        assert server.document_store.is_open("file:///a.py")
        assert server.document_store.is_open("file:///b.ts")


class TestDidChange:
    """Test textDocument/didChange handling."""

    @pytest.mark.asyncio
    async def test_did_change_updates_document(self) -> None:
        """didChange notification updates document content and version."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "old",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": "file:///test.py", "version": 2},
                    "contentChanges": [{"text": "new"}],
                },
            },
        )

        await server.serve(reader, writer)

        doc = server.document_store.get("file:///test.py")
        assert doc is not None
        assert doc.text == "new"
        assert doc.version == 2

    @pytest.mark.asyncio
    async def test_did_change_calls_handler(self) -> None:
        """didChange notification calls the document sync handler."""
        handler = MockDocumentSyncHandler()
        server = LSPServer(document_sync_handler=handler)
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "old",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": "file:///test.py", "version": 2},
                    "contentChanges": [{"text": "new"}],
                },
            },
        )

        await server.serve(reader, writer)

        assert len(handler.changed) == 1
        uri, version, text = handler.changed[0]
        assert uri == "file:///test.py"
        assert version == 2
        assert text == "new"

    @pytest.mark.asyncio
    async def test_did_change_multiple_changes(self) -> None:
        """Multiple contentChanges are applied in order."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "v1",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": "file:///test.py", "version": 2},
                    "contentChanges": [{"text": "v2"}, {"text": "v3"}],
                },
            },
        )

        await server.serve(reader, writer)

        doc = server.document_store.get("file:///test.py")
        assert doc is not None
        assert doc.text == "v3"


class TestDidClose:
    """Test textDocument/didClose handling."""

    @pytest.mark.asyncio
    async def test_did_close_removes_document(self) -> None:
        """didClose notification removes document from store."""
        server = LSPServer()
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "content",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didClose",
                "params": {"textDocument": {"uri": "file:///test.py"}},
            },
        )

        await server.serve(reader, writer)

        assert not server.document_store.is_open("file:///test.py")

    @pytest.mark.asyncio
    async def test_did_close_calls_handler(self) -> None:
        """didClose notification calls the document sync handler."""
        handler = MockDocumentSyncHandler()
        server = LSPServer(document_sync_handler=handler)
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "content",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didClose",
                "params": {"textDocument": {"uri": "file:///test.py"}},
            },
        )

        await server.serve(reader, writer)

        assert len(handler.closed) == 1
        assert handler.closed[0] == "file:///test.py"


class TestDidSave:
    """Test textDocument/didSave handling."""

    @pytest.mark.asyncio
    async def test_did_save_calls_handler(self) -> None:
        """didSave notification calls the document sync handler."""
        handler = MockDocumentSyncHandler()
        server = LSPServer(document_sync_handler=handler)
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "content",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didSave",
                "params": {"textDocument": {"uri": "file:///test.py"}},
            },
        )

        await server.serve(reader, writer)

        assert len(handler.saved) == 1
        uri, text = handler.saved[0]
        assert uri == "file:///test.py"
        assert text is None  # No includeText

    @pytest.mark.asyncio
    async def test_did_save_with_text(self) -> None:
        """didSave with includeText passes text to handler."""
        handler = MockDocumentSyncHandler()
        server = LSPServer(document_sync_handler=handler)
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "saved content",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didSave",
                "params": {
                    "textDocument": {"uri": "file:///test.py"},
                    "text": "saved content",
                },
            },
        )

        await server.serve(reader, writer)

        uri, text = handler.saved[0]
        assert uri == "file:///test.py"
        assert text == "saved content"


class TestFullDocumentLifecycle:
    """Test complete document lifecycle."""

    @pytest.mark.asyncio
    async def test_open_change_save_close_cycle(self) -> None:
        """Complete lifecycle: open -> change -> save -> close."""
        handler = MockDocumentSyncHandler()
        server = LSPServer(document_sync_handler=handler)
        reader = asyncio.StreamReader()
        writer, transport = _make_mock_writer()

        _feed_messages(
            reader,
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {
                    "textDocument": {
                        "uri": "file:///test.py",
                        "languageId": "python",
                        "version": 1,
                        "text": "v1",
                    }
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didChange",
                "params": {
                    "textDocument": {"uri": "file:///test.py", "version": 2},
                    "contentChanges": [{"text": "v2"}],
                },
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didSave",
                "params": {"textDocument": {"uri": "file:///test.py"}},
            },
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didClose",
                "params": {"textDocument": {"uri": "file:///test.py"}},
            },
        )

        await server.serve(reader, writer)

        # Verify all events were called
        assert len(handler.opened) == 1
        assert len(handler.changed) == 1
        assert len(handler.saved) == 1
        assert len(handler.closed) == 1

        # Document should be closed now
        assert not server.document_store.is_open("file:///test.py")
