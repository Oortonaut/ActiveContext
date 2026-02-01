"""Tests for LSP document store."""

from __future__ import annotations

import pytest

from activecontext.transport.lsp.document_store import Document, DocumentStore


class TestDocumentStore:
    """Test DocumentStore operations."""

    def test_open_document(self) -> None:
        """Opening a document creates and stores it."""
        store = DocumentStore()
        doc = store.open("file:///test.py", "python", 1, "print('hello')")

        assert doc.uri == "file:///test.py"
        assert doc.language_id == "python"
        assert doc.version == 1
        assert doc.text == "print('hello')"
        assert store.is_open("file:///test.py")
        assert store.count() == 1

    def test_open_duplicate_raises(self) -> None:
        """Opening an already-open document raises ValueError."""
        store = DocumentStore()
        store.open("file:///test.py", "python", 1, "content")

        with pytest.raises(ValueError, match="already open"):
            store.open("file:///test.py", "python", 2, "new content")

    def test_close_document(self) -> None:
        """Closing a document removes it from the store."""
        store = DocumentStore()
        store.open("file:///test.py", "python", 1, "content")

        store.close("file:///test.py")

        assert not store.is_open("file:///test.py")
        assert store.count() == 0

    def test_close_unopened_raises(self) -> None:
        """Closing an unopened document raises ValueError."""
        store = DocumentStore()

        with pytest.raises(ValueError, match="not open"):
            store.close("file:///missing.py")

    def test_get_document(self) -> None:
        """Getting a document returns it if open."""
        store = DocumentStore()
        store.open("file:///test.py", "python", 1, "content")

        doc = store.get("file:///test.py")
        assert doc is not None
        assert doc.uri == "file:///test.py"

        missing = store.get("file:///other.py")
        assert missing is None

    def test_change_full_sync(self) -> None:
        """Changing a document with full sync updates content."""
        store = DocumentStore()
        store.open("file:///test.py", "python", 1, "old content")

        changes = [{"text": "new content"}]
        doc = store.change("file:///test.py", changes, 2)

        assert doc.text == "new content"
        assert doc.version == 2

    def test_change_unopened_raises(self) -> None:
        """Changing an unopened document raises ValueError."""
        store = DocumentStore()

        with pytest.raises(ValueError, match="not open"):
            store.change("file:///missing.py", [{"text": "content"}], 1)

    def test_change_multiple_changes(self) -> None:
        """Multiple changes are applied in order (full sync)."""
        store = DocumentStore()
        store.open("file:///test.py", "python", 1, "version 1")

        # Each change overwrites (full sync behavior)
        changes = [
            {"text": "version 2"},
            {"text": "version 3"},
        ]
        doc = store.change("file:///test.py", changes, 3)

        assert doc.text == "version 3"
        assert doc.version == 3

    def test_all_uris(self) -> None:
        """all_uris returns all open document URIs."""
        store = DocumentStore()
        store.open("file:///a.py", "python", 1, "a")
        store.open("file:///b.py", "python", 1, "b")
        store.open("file:///c.py", "python", 1, "c")

        uris = store.all_uris()
        assert len(uris) == 3
        assert "file:///a.py" in uris
        assert "file:///b.py" in uris
        assert "file:///c.py" in uris


class TestDocument:
    """Test Document operations."""

    def test_apply_change_full_sync(self) -> None:
        """apply_change with full text updates content and version."""
        doc = Document("file:///test.py", "python", 1, "old")

        doc.apply_change({"text": "new"}, 2)

        assert doc.text == "new"
        assert doc.version == 2

    def test_apply_change_with_range_treated_as_full(self) -> None:
        """apply_change with range is currently treated as full sync."""
        doc = Document("file:///test.py", "python", 1, "original")

        # Even with a range, we treat as full sync for now
        change = {
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 4},
            },
            "text": "replacement",
        }
        doc.apply_change(change, 2)

        assert doc.text == "replacement"
        assert doc.version == 2

    def test_metadata_storage(self) -> None:
        """Documents can store arbitrary metadata."""
        doc = Document("file:///test.py", "python", 1, "content")

        doc.metadata["node_id"] = "text_42"
        doc.metadata["custom_field"] = {"nested": "data"}

        assert doc.metadata["node_id"] == "text_42"
        assert doc.metadata["custom_field"]["nested"] == "data"


class TestDocumentStoreMultipleDocuments:
    """Test DocumentStore with multiple documents."""

    def test_independent_documents(self) -> None:
        """Multiple documents can be managed independently."""
        store = DocumentStore()

        store.open("file:///a.py", "python", 1, "a content")
        store.open("file:///b.ts", "typescript", 1, "b content")

        # Change one doesn't affect the other
        store.change("file:///a.py", [{"text": "a changed"}], 2)

        doc_a = store.get("file:///a.py")
        doc_b = store.get("file:///b.ts")

        assert doc_a is not None
        assert doc_a.text == "a changed"
        assert doc_a.version == 2

        assert doc_b is not None
        assert doc_b.text == "b content"
        assert doc_b.version == 1

    def test_close_one_keeps_others(self) -> None:
        """Closing one document doesn't affect others."""
        store = DocumentStore()

        store.open("file:///a.py", "python", 1, "a")
        store.open("file:///b.py", "python", 1, "b")
        store.open("file:///c.py", "python", 1, "c")

        store.close("file:///b.py")

        assert store.is_open("file:///a.py")
        assert not store.is_open("file:///b.py")
        assert store.is_open("file:///c.py")
        assert store.count() == 2
