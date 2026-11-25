"""Read documents from Paperless-ngx over its REST API.

API details checked against the Paperless-ngx docs (``docs/api.md``) and the OpenAPI schema
served by the pinned 3.5.1 image:

- Auth: ``Authorization: Token <token>``.
- Versioning: ``Accept: application/json; version=10``. We pin the version so a Paperless
  upgrade that changes the default can't silently change the response shape.
- Lists are paginated as ``{"count", "next", "previous", "results"}`` and accept
  ``page_size``, ``ordering`` and ``modified__gt``.

Writing back (checked against the 3.5.1 source):

- ``PATCH /api/documents/{id}/`` with ``custom_fields`` **deletes every custom field not in
  the request** (``documents/serialisers.py``, ``DocumentSerializer.update``), so it would
  wipe the user's own fields. We never use it. ``POST /api/documents/bulk_edit/`` with
  ``modify_custom_fields`` updates only the fields named, and ``add_tag`` only adds.
- ``modify_custom_fields`` requires both ``add_custom_fields`` ({field id: value}) and
  ``remove_custom_fields`` (a list, which we always leave empty).
- A select field's value is the ``id`` of one of its ``extra_data.select_options``
  (``{"id", "label"}``); Paperless generates the id when an option is created.
- ``bulk_edit`` is asynchronous: it answers ``{"result": "OK", "task_id": ...}`` at once
  and a background worker applies the edit later (seen taking ~5 s per queued edit on the
  dev stack). ``GET /api/tasks/?task_id=`` reports ``pending``, ``started``, ``success``,
  ``failure`` or ``revoked``, so ``write_extraction`` waits for ``success`` before it
  returns: the sync cursor must only move once the write has really happened.

Retries and backoff live in the sync flow, not here: this module reports a failure once,
clearly, and lets the caller decide what to do.
"""

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import TracebackType
from typing import Any, Self

import httpx

from lifeadmin.config import Settings
from lifeadmin.extract.llm import DOC_KINDS
from lifeadmin.sources.base import Document

API_VERSION = 10
PAGE_SIZE = 100
TIMEOUT_SECONDS = 30.0
DOCUMENTS_PATH = "/api/documents/"
BULK_EDIT_PATH = "/api/documents/bulk_edit/"
CUSTOM_FIELDS_PATH = "/api/custom_fields/"
TAGS_PATH = "/api/tags/"
TASKS_PATH = "/api/tasks/"
TASK_POLL_SECONDS = 0.5
TASK_TIMEOUT_SECONDS = 60.0

EXPIRY_FIELD = "expiry_date"
KIND_FIELD = "doc_kind"
VERIFY_TAG = "lifeadmin:verify"


class PaperlessError(Exception):
    """A request to Paperless failed. The message never contains the API token."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PaperlessSchema:
    """Ids of the custom fields and tag the add-on writes, as found or created in Paperless."""

    expiry_field_id: int
    kind_field_id: int
    kind_option_ids: Mapping[str, str]  # doc_kind -> select option id
    verify_tag_id: int


class PaperlessSource:
    """A ``DocumentSource`` backed by a Paperless-ngx instance.

    ``transport`` exists for tests: pass an ``httpx.MockTransport`` to answer requests
    without a real server. The client is still built here, so tests exercise the real
    headers and base URL.
    """

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Injected so tests can wait for "tasks" without real time passing.
        self._sleep = sleep
        self._clock = clock
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

    # --- Writing back ---

    def ensure_schema(self) -> PaperlessSchema:
        """Find or create the ``expiry_date`` and ``doc_kind`` fields and the verify tag.

        Safe to run on every start-up. Names match ignoring case, so a field the user made
        by hand is reused. One with our name but another type is refused, not changed.
        """
        expiry = self._find_or_create(
            CUSTOM_FIELDS_PATH, EXPIRY_FIELD, {"name": EXPIRY_FIELD, "data_type": "date"}
        )
        _require_type(expiry, "date")

        wanted = [{"label": kind} for kind in DOC_KINDS]
        kind = self._find_or_create(
            CUSTOM_FIELDS_PATH,
            KIND_FIELD,
            {"name": KIND_FIELD, "data_type": "select", "extra_data": {"select_options": wanted}},
        )
        _require_type(kind, "select")
        options = _select_options(kind)
        missing = [k for k in DOC_KINDS if k not in options]
        if missing:
            # Send every existing option back with its id, so values already set on
            # documents keep pointing at the same option.
            existing = (kind.get("extra_data") or {}).get("select_options") or []
            kind = self._send_json(
                "PATCH",
                f"{CUSTOM_FIELDS_PATH}{kind['id']}/",
                {"extra_data": {"select_options": [*existing, *({"label": k} for k in missing)]}},
            )
            options = _select_options(kind)

        tag = self._find_or_create(TAGS_PATH, VERIFY_TAG, {"name": VERIFY_TAG})
        try:
            return PaperlessSchema(
                expiry_field_id=expiry["id"],
                kind_field_id=kind["id"],
                kind_option_ids={k: options[k] for k in DOC_KINDS},
                verify_tag_id=tag["id"],
            )
        except (KeyError, TypeError) as exc:
            raise PaperlessError("Unexpected custom field or tag format from Paperless") from exc

    def write_extraction(
        self,
        document_id: int,
        schema: PaperlessSchema,
        *,
        doc_kind: str | None,
        expiry_date: date | None,
        needs_verify: bool,
    ) -> None:
        """Set ``doc_kind`` and ``expiry_date`` on a document, and tag it if it needs a check.

        Returns once Paperless reports every edit applied; raises ``PaperlessError`` if one
        fails or doesn't finish within ``TASK_TIMEOUT_SECONDS``.

        Only adds: a ``None`` value leaves the field as it is, and the verify tag is never
        removed here (a user removing it is how they mark a document verified). Other
        custom fields and tags are not touched.
        """
        values: dict[str, str] = {}
        if expiry_date is not None:
            values[str(schema.expiry_field_id)] = expiry_date.isoformat()
        if doc_kind is not None:
            if doc_kind not in schema.kind_option_ids:
                raise ValueError(f"Unknown doc_kind {doc_kind!r}")
            values[str(schema.kind_field_id)] = schema.kind_option_ids[doc_kind]

        task_ids = []
        if values:
            task_ids.append(
                self._bulk_edit(
                    document_id,
                    "modify_custom_fields",
                    {"add_custom_fields": values, "remove_custom_fields": []},
                )
            )
        if needs_verify:
            task_ids.append(self._bulk_edit(document_id, "add_tag", {"tag": schema.verify_tag_id}))
        self._wait_for_tasks(task_ids)

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

    def _bulk_edit(self, document_id: int, method: str, parameters: dict[str, Any]) -> str:
        """Queue one bulk edit and return its task id."""
        reply = self._send_json(
            "POST",
            BULK_EDIT_PATH,
            {"documents": [document_id], "method": method, "parameters": parameters},
        )
        try:
            return str(reply["task_id"])
        except (KeyError, TypeError) as exc:
            raise PaperlessError("Unexpected bulk edit response from Paperless") from exc

    def _wait_for_tasks(self, task_ids: list[str]) -> None:
        deadline = self._clock() + TASK_TIMEOUT_SECONDS
        pending = list(task_ids)
        while pending:
            still = []
            for task_id in pending:
                status = self._task_status(task_id)
                if status in ("failure", "revoked"):
                    raise PaperlessError(f"Paperless task {task_id} ended with status {status!r}")
                if status != "success":
                    still.append(task_id)  # not listed yet, pending or started
            pending = still
            if not pending:
                return
            if self._clock() >= deadline:
                raise PaperlessError(
                    f"Paperless tasks not finished after {TASK_TIMEOUT_SECONDS:.0f}s: "
                    f"{', '.join(pending)}; is its worker running?"
                )
            self._sleep(TASK_POLL_SECONDS)

    def _task_status(self, task_id: str) -> str | None:
        page = self._get_json(TASKS_PATH, {"task_id": task_id})
        try:
            results = page["results"]
            return results[0]["status"] if results else None
        except (KeyError, TypeError, IndexError) as exc:
            raise PaperlessError("Unexpected task format from Paperless") from exc

    def _find_or_create(self, path: str, name: str, body: dict[str, Any]) -> dict[str, Any]:
        page = self._get_json(path, {"name__iexact": name})
        try:
            results = page["results"]
        except (KeyError, TypeError) as exc:
            raise PaperlessError("Unexpected response shape from Paperless") from exc
        if len(results) > 1:
            raise PaperlessError(f"Paperless has more than one {name!r}; keep only one")
        if results:
            return results[0]
        return self._send_json("POST", path, body)

    def _get_json(
        self, path: str, params: Mapping[str, Any] | httpx.QueryParams | None = None
    ) -> Any:
        return self._request("GET", path, params=params)

    def _send_json(self, method: str, path: str, body: dict[str, Any]) -> Any:
        return self._request(method, path, json=body)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | httpx.QueryParams | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(method, path, params=params, json=json)
        except httpx.HTTPError as exc:
            # Connection refused, DNS failure, timeout. Name the kind of error only.
            raise PaperlessError(
                f"Could not reach Paperless ({type(exc).__name__}); check PAPERLESS_URL"
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise PaperlessError(
                f"Paperless rejected the API token (HTTP {status}); check PAPERLESS_TOKEN "
                "and that its user can view and change documents",
                status_code=status,
            )
        if response.is_error:
            # The response body is left out on purpose: validation errors can echo values.
            raise PaperlessError(
                f"Paperless returned HTTP {status} for {method} {response.request.url.path}",
                status_code=status,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise PaperlessError(f"Paperless returned non-JSON for {path}") from exc


def _require_type(field: dict[str, Any], data_type: str) -> None:
    if field.get("data_type") != data_type:
        raise PaperlessError(
            f"Paperless custom field {field.get('name')!r} has type "
            f"{field.get('data_type')!r}, expected {data_type!r}; rename or delete it"
        )


def _select_options(field: dict[str, Any]) -> dict[str, str]:
    """Option label -> id. Labels we don't know (added by the user) are kept but unused."""
    options = (field.get("extra_data") or {}).get("select_options") or []
    try:
        return {o["label"]: o["id"] for o in options}
    except (KeyError, TypeError) as exc:
        raise PaperlessError("Unexpected select options format from Paperless") from exc


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
