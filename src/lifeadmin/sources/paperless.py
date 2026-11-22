"""Read documents from Paperless-ngx over its REST API.

API details checked against the Paperless-ngx docs (``docs/api.md``) and the OpenAPI schema
served by the pinned 3.5.1 image:

- Auth: ``Authorization: Token <token>``.
- Versioning: ``Accept: application/json; version=10``. We pin the version so a Paperless
  upgrade that changes the default can't silently change the response shape.
- Lists are paginated as ``{"count", "next", "previous", "results"}`` and accept
  ``page_size``, ``ordering`` and ``modified__gt``.

Retries and backoff live in the sync flow, not here: this module reports a failure once,
clearly, and lets the caller decide what to do.
"""

from collections.abc import Iterator, Mapping
from datetime import datetime
from types import TracebackType
from typing import Any, Self

import httpx

from lifeadmin.config import Settings
from lifeadmin.sources.base import Document

API_VERSION = 10
PAGE_SIZE = 100
TIMEOUT_SECONDS = 30.0
DOCUMENTS_PATH = "/api/documents/"


class PaperlessError(Exception):
    """A request to Paperless failed. The message never contains the API token."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PaperlessSource:
    """A ``DocumentSource`` backed by a Paperless-ngx instance.

    ``transport`` exists for tests: pass an ``httpx.MockTransport`` to answer requests
    without a real server. The client is still built here, so tests exercise the real
    headers and base URL.
    """

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=settings.paperless_url,
            headers={
                "Authorization": f"Token {settings.paperless_token}",
                "Accept": f"application/json; version={API_VERSION}",
            },
            timeout=TIMEOUT_SECONDS,
            transport=transport,
        )

    def iter_documents(self, modified_after: datetime | None = None) -> Iterator[Document]:
        params: Mapping[str, Any] = {"ordering": "modified", "page_size": PAGE_SIZE}
        if modified_after is not None:
            # A naive datetime would be read in Paperless's own time zone, which may not
            # be ours. Requiring an offset makes the cursor mean the same thing everywhere.
            if modified_after.tzinfo is None:
                raise ValueError("modified_after must be timezone-aware")
            params = {**params, "modified__gt": modified_after.isoformat()}

        while True:
            page = self._get_json(DOCUMENTS_PATH, params)
            try:
                results = page["results"]
                next_url = page.get("next")
            except (KeyError, TypeError, AttributeError) as exc:
                raise PaperlessError("Unexpected response shape from Paperless") from exc

            for item in results:
                yield _to_document(item)

            if not next_url:
                return
            # Reuse only the query string of "next" (page number plus our filters) and keep
            # requesting our own base URL. Paperless builds "next" from the host it was
            # reached through, which can differ inside Docker, and following an arbitrary
            # URL would send the token to whatever host it names.
            params = httpx.URL(next_url).params

    def get_document(self, document_id: int) -> Document:
        return _to_document(self._get_json(f"{DOCUMENTS_PATH}{document_id}/"))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _get_json(
        self, path: str, params: Mapping[str, Any] | httpx.QueryParams | None = None
    ) -> Any:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            # Connection refused, DNS failure, timeout. Name the kind of error only.
            raise PaperlessError(
                f"Could not reach Paperless ({type(exc).__name__}); check PAPERLESS_URL"
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise PaperlessError(
                f"Paperless rejected the API token (HTTP {status}); check PAPERLESS_TOKEN "
                "and that its user can view documents",
                status_code=status,
            )
        if response.is_error:
            raise PaperlessError(
                f"Paperless returned HTTP {status} for {response.request.url.path}",
                status_code=status,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise PaperlessError(f"Paperless returned non-JSON for {path}") from exc


def _to_document(item: Any) -> Document:
    """Map one API document object to a ``Document``."""
    try:
        return Document(
            id=item["id"],
            title=item["title"],
            # Treat a missing or null content field as "no OCR text yet".
            content=item.get("content") or "",
            # Paperless sends ISO 8601 with an offset; Python 3.12 parses it directly.
            modified=datetime.fromisoformat(item["modified"]),
            document_type_id=item.get("document_type"),
            tag_ids=tuple(item.get("tags") or ()),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise PaperlessError("Unexpected document format from Paperless") from exc
