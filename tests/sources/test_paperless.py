"""PaperlessSource against a fake Paperless (httpx.MockTransport); no real server."""

import json
from datetime import UTC, date, datetime, timedelta, timezone

import httpx
import pytest

from lifeadmin.config import Settings
from lifeadmin.extract.llm import DOC_KINDS
from lifeadmin.sources.paperless import (
    API_VERSION,
    PAGE_SIZE,
    TASK_POLL_SECONDS,
    TASK_TIMEOUT_SECONDS,
    PaperlessError,
    PaperlessSchema,
    PaperlessSource,
)

# Synthetic values only; never a real token.
FAKE_TOKEN = "fake-token-0123456789"
BASE_URL = "http://paperless.test:8000"
SETTINGS = Settings(paperless_url=BASE_URL, paperless_token=FAKE_TOKEN)


def api_document(doc_id: int, **overrides) -> dict:
    """A document object shaped like Paperless's API response, with made-up data."""
    return {
        "id": doc_id,
        "title": f"Sample document {doc_id}",
        "content": f"SYNTHETIC OCR TEXT {doc_id}",
        "modified": "2026-03-07T09:30:00.123456-05:00",
        "document_type": 2,
        "tags": [1, 4],
    } | overrides


def page(results: list[dict], next_url: str | None = None) -> dict:
    return {"count": len(results), "next": next_url, "previous": None, "results": results}


def source_with(handler) -> PaperlessSource:
    return PaperlessSource(SETTINGS, transport=httpx.MockTransport(handler))


def test_sends_token_and_pinned_api_version():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=page([]))

    list(source_with(handler).iter_documents())

    (request,) = seen
    assert request.headers["Authorization"] == f"Token {FAKE_TOKEN}"
    assert request.headers["Accept"] == f"application/json; version={API_VERSION}"
    assert str(request.url).startswith(f"{BASE_URL}/api/documents/")


def test_follows_pagination_in_order():
    pages = {
        None: page([api_document(1), api_document(2)], f"{BASE_URL}/api/documents/?page=2"),
        "2": page([api_document(3)], f"{BASE_URL}/api/documents/?page=3"),
        "3": page([api_document(4)]),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pages[request.url.params.get("page")])

    docs = list(source_with(handler).iter_documents())

    assert [d.id for d in docs] == [1, 2, 3, 4]


def test_next_link_keeps_our_host_and_filters():
    # Paperless can build "next" with a different host (e.g. its container name). We must
    # keep calling the configured URL, so the token never goes anywhere else.
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            next_url = (
                "http://other-host:9999/api/documents/"
                "?modified__gt=2026-01-01T00%3A00%3A00%2B00%3A00&ordering=modified&page=2"
            )
            return httpx.Response(200, json=page([api_document(1)], next_url))
        return httpx.Response(200, json=page([api_document(2)]))

    cursor = datetime(2026, 1, 1, tzinfo=UTC)
    list(source_with(handler).iter_documents(modified_after=cursor))

    second = requests[1]
    assert second.url.host == "paperless.test"
    assert second.url.port == 8000
    assert second.url.params["page"] == "2"
    assert second.url.params["modified__gt"] == "2026-01-01T00:00:00+00:00"
    assert second.headers["Authorization"] == f"Token {FAKE_TOKEN}"


def test_first_request_without_cursor_has_no_date_filter():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=page([]))

    list(source_with(handler).iter_documents())

    params = seen[0].url.params
    assert "modified__gt" not in params
    assert params["ordering"] == "modified"
    assert params["page_size"] == str(PAGE_SIZE)


def test_cursor_becomes_modified_gt_filter():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=page([]))

    toronto = timezone(timedelta(hours=-5))
    cursor = datetime(2026, 3, 1, 12, 0, tzinfo=toronto)
    list(source_with(handler).iter_documents(modified_after=cursor))

    assert seen[0].url.params["modified__gt"] == "2026-03-01T12:00:00-05:00"


def test_naive_cursor_is_rejected():
    source = source_with(lambda request: httpx.Response(200, json=page([])))

    with pytest.raises(ValueError, match="timezone-aware"):
        list(source.iter_documents(modified_after=datetime(2026, 3, 1)))


def test_maps_api_fields_to_document():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=page([api_document(7)]))

    (doc,) = source_with(handler).iter_documents()

    assert doc.id == 7
    assert doc.title == "Sample document 7"
    assert doc.content == "SYNTHETIC OCR TEXT 7"
    assert doc.modified == datetime(2026, 3, 7, 14, 30, 0, 123456, tzinfo=UTC)
    assert doc.document_type_id == 2
    assert doc.tag_ids == (1, 4)


@pytest.mark.parametrize("content", ["", None])
def test_empty_or_null_content_becomes_empty_string(content):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=page([api_document(1, content=content)]))

    (doc,) = source_with(handler).iter_documents()

    assert doc.content == ""


def test_document_without_type_or_tags():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=page([api_document(1, document_type=None, tags=[])]))

    (doc,) = source_with(handler).iter_documents()

    assert doc.document_type_id is None
    assert doc.tag_ids == ()


def test_get_document_fetches_by_id():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/documents/42/"
        return httpx.Response(200, json=api_document(42))

    doc = source_with(handler).get_document(42)

    assert doc.id == 42


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_points_at_token_without_leaking_it(status):
    source = source_with(lambda request: httpx.Response(status, json={"detail": "Invalid"}))

    with pytest.raises(PaperlessError, match="PAPERLESS_TOKEN") as excinfo:
        list(source.iter_documents())

    assert excinfo.value.status_code == status
    assert FAKE_TOKEN not in str(excinfo.value)


def test_missing_document_reports_404():
    source = source_with(lambda request: httpx.Response(404, json={"detail": "Not found."}))

    with pytest.raises(PaperlessError, match="HTTP 404 for GET /api/documents/99/") as excinfo:
        source.get_document(99)

    assert excinfo.value.status_code == 404


def test_server_error_is_reported():
    source = source_with(lambda request: httpx.Response(500, text="boom"))

    with pytest.raises(PaperlessError, match="HTTP 500") as excinfo:
        list(source.iter_documents())

    assert excinfo.value.status_code == 500


def test_connection_failure_is_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(PaperlessError, match="Could not reach Paperless") as excinfo:
        list(source_with(handler).iter_documents())

    assert excinfo.value.status_code is None
    assert FAKE_TOKEN not in str(excinfo.value)


def test_non_json_response_is_reported():
    source = source_with(lambda request: httpx.Response(200, text="<html>login</html>"))

    with pytest.raises(PaperlessError, match="non-JSON"):
        list(source.iter_documents())


def test_malformed_document_is_reported():
    source = source_with(lambda request: httpx.Response(200, json=page([{"id": 1}])))

    with pytest.raises(PaperlessError, match="Unexpected document format"):
        list(source.iter_documents())


def test_unexpected_page_shape_is_reported():
    source = source_with(lambda request: httpx.Response(200, json=["not", "a", "page"]))

    with pytest.raises(PaperlessError, match="Unexpected response shape"):
        list(source.iter_documents())


def test_context_manager_closes_client():
    with source_with(lambda request: httpx.Response(200, json=page([]))) as source:
        pass

    with pytest.raises(RuntimeError):
        source.get_document(1)


# --- Writing back ---


class FakePaperless:
    """Just enough of Paperless's custom-field, tag and bulk-edit API, with state."""

    def __init__(self, fields=(), tags=(), task_statuses=("success",)):
        self.fields = {f["id"]: f for f in fields}
        self.tags = {t["id"]: t for t in tags}
        self.requests: list[httpx.Request] = []
        self.bulk_edits: list[dict] = []
        self.next_id = 100
        # Statuses each task reports on successive polls; the last one repeats.
        # None means "not in the task list yet".
        self.task_statuses = list(task_statuses)
        self.polls: dict[str, int] = {}
        self.sleeps: list[float] = []
        self.now = 0.0

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        for base, store in (("/api/custom_fields/", self.fields), ("/api/tags/", self.tags)):
            if path == base and method == "GET":
                name = request.url.params["name__iexact"].lower()
                return httpx.Response(
                    200, json=page([o for o in store.values() if o["name"].lower() == name])
                )
            if path == base and method == "POST":
                obj = {"id": self._new_id(), **body}
                for opt in (obj.get("extra_data") or {}).get("select_options", []):
                    opt.setdefault("id", f"opt-{self._new_id()}")
                store[obj["id"]] = obj
                return httpx.Response(201, json=obj)
            if path.startswith(base) and method == "PATCH":
                obj = store[int(path.rstrip("/").rsplit("/", 1)[1])]
                obj.update(body)
                for opt in obj["extra_data"]["select_options"]:
                    opt.setdefault("id", f"opt-{self._new_id()}")
                return httpx.Response(200, json=obj)
        if path == "/api/documents/bulk_edit/" and method == "POST":
            self.bulk_edits.append(body)
            return httpx.Response(
                200, json={"result": "OK", "task_id": f"task-{len(self.bulk_edits)}"}
            )
        if path == "/api/tasks/" and method == "GET":
            task_id = request.url.params["task_id"]
            n = self.polls[task_id] = self.polls.get(task_id, 0) + 1
            status = self.task_statuses[min(n, len(self.task_statuses)) - 1]
            results = [] if status is None else [{"task_id": task_id, "status": status}]
            return httpx.Response(200, json=page(results))
        return httpx.Response(404)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def source(self) -> PaperlessSource:
        return PaperlessSource(
            SETTINGS,
            transport=httpx.MockTransport(self),
            sleep=self.sleep,
            clock=lambda: self.now,
        )


def kind_field(options):
    return {
        "id": 7,
        "name": "Doc_Kind",
        "data_type": "select",
        "extra_data": {"select_options": options},
    }


def test_ensure_schema_creates_fields_and_tag():
    fake = FakePaperless()

    schema = fake.source().ensure_schema()

    by_name = {f["name"]: f for f in fake.fields.values()}
    assert by_name["expiry_date"]["data_type"] == "date"
    assert by_name["doc_kind"]["data_type"] == "select"
    labels = [o["label"] for o in by_name["doc_kind"]["extra_data"]["select_options"]]
    assert labels == list(DOC_KINDS)
    assert [t["name"] for t in fake.tags.values()] == ["lifeadmin:verify"]
    assert schema.expiry_field_id == by_name["expiry_date"]["id"]
    assert set(schema.kind_option_ids) == set(DOC_KINDS)


def test_ensure_schema_reuses_existing_and_creates_nothing():
    options = [{"id": f"o-{k}", "label": k} for k in DOC_KINDS]
    fake = FakePaperless(
        fields=[{"id": 3, "name": "EXPIRY_DATE", "data_type": "date"}, kind_field(options)],
        tags=[{"id": 9, "name": "LifeAdmin:Verify"}],
    )

    schema = fake.source().ensure_schema()

    assert {r.method for r in fake.requests} == {"GET"}
    assert schema == PaperlessSchema(3, 7, {k: f"o-{k}" for k in DOC_KINDS}, 9)


def test_ensure_schema_adds_missing_options_and_keeps_existing_ids():
    user_option = {"id": "user-1", "label": "library_card"}
    fake = FakePaperless(
        fields=[
            {"id": 3, "name": "expiry_date", "data_type": "date"},
            kind_field([{"id": "keep-me", "label": "passport"}, user_option]),
        ]
    )

    schema = fake.source().ensure_schema()

    options = fake.fields[7]["extra_data"]["select_options"]
    assert options[:2] == [{"id": "keep-me", "label": "passport"}, user_option]
    assert {o["label"] for o in options} == {*DOC_KINDS, "library_card"}
    assert schema.kind_option_ids["passport"] == "keep-me"


def test_ensure_schema_refuses_a_field_of_another_type():
    fake = FakePaperless(fields=[{"id": 3, "name": "expiry_date", "data_type": "string"}])

    with pytest.raises(PaperlessError, match="'expiry_date' has type 'string'"):
        fake.source().ensure_schema()

    assert all(r.method == "GET" for r in fake.requests)


def test_ensure_schema_refuses_duplicate_names():
    fake = FakePaperless(
        tags=[{"id": 1, "name": "lifeadmin:verify"}, {"id": 2, "name": "LIFEADMIN:VERIFY"}]
    )
    fake.fields = {
        3: {"id": 3, "name": "expiry_date", "data_type": "date"},
        7: kind_field([{"id": f"o-{k}", "label": k} for k in DOC_KINDS]),
    }

    with pytest.raises(PaperlessError, match="more than one 'lifeadmin:verify'"):
        fake.source().ensure_schema()


SCHEMA = PaperlessSchema(3, 7, {k: f"o-{k}" for k in DOC_KINDS}, 9)


def test_write_extraction_sets_fields_with_bulk_edit():
    fake = FakePaperless()

    fake.source().write_extraction(
        42, SCHEMA, doc_kind="passport", expiry_date=date(2031, 6, 15), needs_verify=False
    )

    assert fake.bulk_edits == [
        {
            "documents": [42],
            "method": "modify_custom_fields",
            "parameters": {
                "add_custom_fields": {"3": "2031-06-15", "7": "o-passport"},
                "remove_custom_fields": [],
            },
        }
    ]


def test_write_extraction_adds_verify_tag_when_needed():
    fake = FakePaperless()

    fake.source().write_extraction(
        42, SCHEMA, doc_kind="passport", expiry_date=date(2031, 6, 15), needs_verify=True
    )

    assert fake.bulk_edits[-1] == {"documents": [42], "method": "add_tag", "parameters": {"tag": 9}}


def test_write_extraction_leaves_missing_values_alone():
    fake = FakePaperless()

    fake.source().write_extraction(42, SCHEMA, doc_kind="t4", expiry_date=None, needs_verify=False)

    assert fake.bulk_edits[0]["parameters"]["add_custom_fields"] == {"7": "o-t4"}


def test_write_extraction_with_nothing_to_write_sends_only_the_tag():
    fake = FakePaperless()

    fake.source().write_extraction(42, SCHEMA, doc_kind=None, expiry_date=None, needs_verify=True)

    assert [e["method"] for e in fake.bulk_edits] == ["add_tag"]


def test_write_extraction_rejects_unknown_kind():
    with pytest.raises(ValueError, match="library_card"):
        FakePaperless().source().write_extraction(
            42, SCHEMA, doc_kind="library_card", expiry_date=None, needs_verify=False
        )


def test_writing_never_patches_a_document():
    fake = FakePaperless()
    source = fake.source()

    schema = source.ensure_schema()
    source.write_extraction(42, schema, doc_kind="noa", expiry_date=None, needs_verify=True)

    # PATCH /api/documents/{id}/ with custom_fields would delete the user's other fields.
    assert not [r for r in fake.requests if r.url.path.startswith("/api/documents/42")]
    assert all(r.method != "PATCH" or "/custom_fields/" in r.url.path for r in fake.requests)


@pytest.mark.parametrize(
    ("status", "match"), [(400, "HTTP 400 for POST"), (403, "PAPERLESS_TOKEN")]
)
def test_write_errors_name_status_but_never_the_body(status, match):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"add_custom_fields": ["2031-06-15 is invalid"]})

    source = PaperlessSource(SETTINGS, transport=httpx.MockTransport(handler))

    with pytest.raises(PaperlessError, match=match) as excinfo:
        source.write_extraction(
            42, SCHEMA, doc_kind="passport", expiry_date=date(2031, 6, 15), needs_verify=False
        )

    assert excinfo.value.status_code == status
    assert "2031" not in str(excinfo.value)


def write_passport(fake: FakePaperless, needs_verify: bool = False) -> None:
    fake.source().write_extraction(
        42, SCHEMA, doc_kind="passport", expiry_date=date(2031, 6, 15), needs_verify=needs_verify
    )


def test_write_waits_until_every_task_succeeds():
    fake = FakePaperless(task_statuses=(None, "pending", "started", "success"))

    write_passport(fake, needs_verify=True)

    assert fake.polls == {"task-1": 4, "task-2": 4}
    assert fake.sleeps == [TASK_POLL_SECONDS] * 3


def test_write_returns_without_sleeping_when_tasks_are_already_done():
    fake = FakePaperless()

    write_passport(fake)

    assert fake.sleeps == []


@pytest.mark.parametrize("status", ["failure", "revoked"])
def test_failed_task_raises(status):
    fake = FakePaperless(task_statuses=("pending", status))

    with pytest.raises(PaperlessError, match=f"task-1 ended with status '{status}'"):
        write_passport(fake)


def test_task_that_never_finishes_times_out():
    fake = FakePaperless(task_statuses=("pending",))

    with pytest.raises(PaperlessError, match="not finished after 60s: task-1"):
        write_passport(fake)

    assert fake.now >= TASK_TIMEOUT_SECONDS


def test_bulk_edit_reply_without_task_id_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": "OK"})

    source = PaperlessSource(SETTINGS, transport=httpx.MockTransport(handler))

    with pytest.raises(PaperlessError, match="Unexpected bulk edit response"):
        source.write_extraction(42, SCHEMA, doc_kind="t4", expiry_date=None, needs_verify=False)
