"""PaperlessSource against a fake Paperless (httpx.MockTransport); no real server."""

from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest

from lifeadmin.config import Settings
from lifeadmin.sources.paperless import API_VERSION, PAGE_SIZE, PaperlessError, PaperlessSource

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

    with pytest.raises(PaperlessError, match="HTTP 404 for /api/documents/99/") as excinfo:
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
