"""OllamaExtractor against a fake Ollama (httpx.MockTransport); no real model is called."""

import json
from datetime import date

import httpx
import pytest

from lifeadmin.config import Settings
from lifeadmin.extract.llm import (
    DOC_KINDS,
    MAX_TEXT_CHARS,
    SCHEMA,
    LlmBadResponse,
    LlmResult,
    LlmUnavailable,
    OllamaExtractor,
)

OLLAMA_URL = "http://ollama.test:11434"
MODEL = "test-model:1b"
SETTINGS = Settings(
    paperless_url="http://paperless.test",
    paperless_token="fake-token",
    ollama_url=OLLAMA_URL,
    ollama_model=MODEL,
)
# Synthetic OCR text; the name and dates are made up.
SAMPLE_TEXT = "ONTARIO DRIVER'S LICENCE\nTESTER, JANE Q\nISS 2021/04/15 EXP 2026/04/15"


def chat_reply(fields: object) -> dict:
    """An /api/chat response shaped like Ollama's, with ``fields`` as the JSON content."""
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": json.dumps(fields)},
        "done": True,
    }


def fields(**overrides) -> dict:
    return {
        "doc_kind": "drivers_licence",
        "holder": "Jane Q Tester",
        "issue_date": "2021-04-15",
        "expiry_date": "2026-04-15",
    } | overrides


def extractor_with(handler) -> OllamaExtractor:
    return OllamaExtractor(SETTINGS, transport=httpx.MockTransport(handler))


def replying(body: dict, status: int = 200):
    return lambda request: httpx.Response(status, json=body)


def test_request_uses_schema_temperature_zero_and_no_thinking():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=chat_reply(fields()))

    extractor_with(handler).extract(SAMPLE_TEXT)

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == f"{OLLAMA_URL}/api/chat"
    body = json.loads(request.content)
    assert body["model"] == MODEL
    assert body["format"] == SCHEMA
    assert body["stream"] is False
    assert body["think"] is False
    assert body["options"] == {"temperature": 0}
    assert body["messages"][-1] == {"role": "user", "content": SAMPLE_TEXT}


def test_schema_and_prompt_never_ask_for_confidence():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "confidence" not in request.content.decode().lower()
        return httpx.Response(200, json=chat_reply(fields()))

    extractor_with(handler).extract(SAMPLE_TEXT)
    assert "confidence" not in SCHEMA["properties"]


def test_schema_kinds_match_the_data_model():
    assert set(DOC_KINDS) == {
        "passport",
        "pr_card",
        "drivers_licence",
        "health_card",
        "payslip",
        "t4",
        "noa",
        "employment_letter",
        "credit_report",
        "bank_statement",
    }
    assert SCHEMA["properties"]["doc_kind"]["enum"] == [*DOC_KINDS, None]


def test_maps_valid_reply_to_result():
    result = extractor_with(replying(chat_reply(fields()))).extract(SAMPLE_TEXT)

    assert result == LlmResult(
        doc_kind="drivers_licence",
        holder="Jane Q Tester",
        issue_date=date(2021, 4, 15),
        expiry_date=date(2026, 4, 15),
    )


def test_null_fields_stay_none():
    reply = chat_reply(fields(doc_kind=None, holder=None, issue_date=None, expiry_date=None))

    result = extractor_with(replying(reply)).extract(SAMPLE_TEXT)

    assert result == LlmResult(doc_kind=None, holder=None, issue_date=None, expiry_date=None)


def test_unknown_kind_becomes_none_and_keeps_other_fields():
    reply = chat_reply(fields(doc_kind="driver_license"))

    result = extractor_with(replying(reply)).extract(SAMPLE_TEXT)

    assert result.doc_kind is None
    assert result.expiry_date == date(2026, 4, 15)


@pytest.mark.parametrize(
    "bad",
    [
        "2031-02-30",  # not a real day
        "2031-13-01",  # not a real month
        "2031/02/14",  # wrong separator
        "20310214",  # compact ISO form, which fromisoformat would accept
        "2031-W07-1",  # ISO week date, likewise
        "14 FEB 2031",
        "",
        20310214,
    ],
)
def test_invalid_dates_become_none(bad):
    reply = chat_reply(fields(expiry_date=bad))

    result = extractor_with(replying(reply)).extract(SAMPLE_TEXT)

    assert result.expiry_date is None
    assert result.issue_date == date(2021, 4, 15)


def test_holder_whitespace_is_collapsed_and_blank_is_none():
    spaced = extractor_with(replying(chat_reply(fields(holder="  Jane\n Q  Tester ")))).extract(
        SAMPLE_TEXT
    )
    blank = extractor_with(replying(chat_reply(fields(holder="   ")))).extract(SAMPLE_TEXT)

    assert spaced.holder == "Jane Q Tester"
    assert blank.holder is None


def test_long_text_is_cut_to_the_cap():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_reply(fields()))

    extractor_with(handler).extract("A" * (MAX_TEXT_CHARS + 500))

    assert seen[0]["messages"][-1]["content"] == "A" * MAX_TEXT_CHARS


@pytest.mark.parametrize("status", [404, 500, 501, 503])
def test_http_errors_raise_unavailable(status):
    with pytest.raises(LlmUnavailable) as info:
        extractor_with(replying({"error": "synthetic failure"}, status)).extract(SAMPLE_TEXT)

    assert info.value.status_code == status


def test_501_message_points_at_structured_output():
    with pytest.raises(LlmUnavailable, match="structured output"):
        extractor_with(replying({"error": "x"}, 501)).extract(SAMPLE_TEXT)


@pytest.mark.parametrize(
    "error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")], ids=["connect", "timeout"]
)
def test_transport_errors_raise_unavailable(error):
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(LlmUnavailable, match="OLLAMA_URL") as info:
        extractor_with(handler).extract(SAMPLE_TEXT)

    assert info.value.status_code is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"done": True}),  # no message
        httpx.Response(200, json={"message": {"content": "{not json"}}),
        httpx.Response(200, json=chat_reply(["a", "list"])),
    ],
    ids=["body-not-json", "no-message", "content-not-json", "content-not-object"],
)
def test_malformed_replies_raise_bad_response(response):
    with pytest.raises(LlmBadResponse):
        extractor_with(lambda request: response).extract(SAMPLE_TEXT)


def test_context_manager_closes_client():
    with extractor_with(replying(chat_reply(fields()))) as extractor:
        extractor.extract(SAMPLE_TEXT)

    assert extractor._client.is_closed
