"""Document storage and versioning for LSP.

This module provides the DocumentStore class for tracking open documents
with their content and version numbers as managed by LSP clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document tracked by the LSP server.

    Attributes:
        uri: The document URI (e.g., "file:///path/to/file.py")
        language_id: Language identifier (e.g., "python", "typescript")
        version: Document version number (incremented on changes)
        text: Current document text content
        metadata: Optional metadata storage for extension data
    """

    uri: str
    language_id: str
    version: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_change(
        self,
        change: dict[str, Any],
        new_version: int,
    ) -> None:
        """Apply a text document change and update version.

        For full sync (textDocumentSync = 1), change contains only "text".
        For incremental sync, change contains "range" and "text".

        Args:
            change: Change object from didChange notification params
            new_version: New version number to set
        """
        if "range" in change:
            # Incremental change (not supported yet - would need range logic)
            # For now we treat this as full sync
            self.text = change["text"]
        else:
            # Full document sync
            self.text = change["text"]

        self.version = new_version


class DocumentStore:
    """Storage for open LSP documents with version tracking.

    Manages documents opened via textDocument/didOpen and closed via
    textDocument/didClose. Tracks version numbers and content changes.
    """

    def __init__(self) -> None:
        """Initialize empty document store."""
        self._documents: dict[str, Document] = {}

    def open(
        self,
        uri: str,
        language_id: str,
        version: int,
        text: str,
    ) -> Document:
        """Open a new document.

        Args:
            uri: Document URI
            language_id: Language identifier
            version: Initial version number
            text: Initial document content

        Returns:
            The created Document instance

        Raises:
            ValueError: If document is already open
        """
        if uri in self._documents:
            raise ValueError(f"Document already open: {uri}")

        doc = Document(
            uri=uri,
            language_id=language_id,
            version=version,
            text=text,
        )
        self._documents[uri] = doc
        return doc

    def change(
        self,
        uri: str,
        changes: list[dict[str, Any]],
        version: int,
    ) -> Document:
        """Apply changes to an open document.

        Args:
            uri: Document URI
            changes: List of changes (for full sync, typically 1 change with "text")
            version: New version number

        Returns:
            The updated Document instance

        Raises:
            ValueError: If document is not open
        """
        if uri not in self._documents:
            raise ValueError(f"Document not open: {uri}")

        doc = self._documents[uri]

        # Apply all changes in order
        for change in changes:
            doc.apply_change(change, version)

        return doc

    def close(self, uri: str) -> None:
        """Close a document.

        Args:
            uri: Document URI to close

        Raises:
            ValueError: If document is not open
        """
        if uri not in self._documents:
            raise ValueError(f"Document not open: {uri}")

        del self._documents[uri]

    def get(self, uri: str) -> Document | None:
        """Get a document by URI.

        Args:
            uri: Document URI

        Returns:
            Document instance if open, None otherwise
        """
        return self._documents.get(uri)

    def is_open(self, uri: str) -> bool:
        """Check if a document is open.

        Args:
            uri: Document URI

        Returns:
            True if document is open, False otherwise
        """
        return uri in self._documents

    def all_uris(self) -> list[str]:
        """Get all open document URIs.

        Returns:
            List of URIs for all open documents
        """
        return list(self._documents.keys())

    def count(self) -> int:
        """Get the number of open documents.

        Returns:
            Count of open documents
        """
        return len(self._documents)
