"""The interface every document source implements.

The rest of the add-on (sync, extraction) only talks to a ``DocumentSource``, never to
Paperless directly. That keeps Paperless details in one module and leaves room for a
folder-based source later without touching the callers.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Document:
    """The parts of a stored document the add-on needs. Holds text, never the file."""

    id: int
    title: str
    # OCR text. Can be empty, e.g. while Paperless is still processing the document.
    # repr=False keeps personal data (names, ID numbers) out of logs and tracebacks.
    content: str = field(repr=False)
    modified: datetime
    document_type_id: int | None
    tag_ids: tuple[int, ...]


class DocumentSource(Protocol):
    """Read access to a document archive."""

    def iter_documents(self, modified_after: datetime | None = None) -> Iterator[Document]:
        """Yield documents, oldest modification first.

        With ``modified_after``, only documents modified strictly after that moment are
        returned, which is what incremental sync uses as its cursor.
        """
        ...

    def get_document(self, document_id: int) -> Document:
        """Return one document by its ID."""
        ...
