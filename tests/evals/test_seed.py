"""Seeding the dev Paperless, against a fake server (httpx.MockTransport)."""

import httpx
import pytest

from evals.seed import UPLOAD_PATH, upload_samples
from lifeadmin.config import Settings
from lifeadmin.sources.paperless import PaperlessError

FAKE_TOKEN = "fake-token-0123456789"
SETTINGS = Settings(paperless_url="http://paperless.test:8000", paperless_token=FAKE_TOKEN)


@pytest.fixture
def pngs(tmp_path):
    paths = []
    for name in ("passport-01", "t4-01"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"\x89PNG fake image " + name.encode())
        paths.append(path)
    return paths


def test_uploads_each_file_with_title_and_token(pngs):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=f"task-{len(seen)}")

    task_ids = upload_samples(SETTINGS, pngs, transport=httpx.MockTransport(handler))

    assert task_ids == ["task-1", "task-2"]
    first = seen[0]
    assert first.method == "POST"
    assert first.url.path == UPLOAD_PATH
    assert first.headers["Authorization"] == f"Token {FAKE_TOKEN}"
    body = first.read()
    assert b'name="title"\r\n\r\npassport-01' in body
    assert b'name="document"; filename="passport-01.png"' in body
    assert b"fake image passport-01" in body


def test_http_error_names_file_not_token(pngs):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"document": ["bad file"]})

    with pytest.raises(PaperlessError, match="passport-01.png.*HTTP 400") as exc_info:
        upload_samples(SETTINGS, pngs, transport=httpx.MockTransport(handler))
    assert FAKE_TOKEN not in str(exc_info.value)


def test_connection_error_is_reported(pngs):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(PaperlessError, match="Could not reach Paperless"):
        upload_samples(SETTINGS, pngs, transport=httpx.MockTransport(handler))
