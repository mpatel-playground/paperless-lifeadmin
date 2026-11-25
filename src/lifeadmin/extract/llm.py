"""Ollama fallback: ask a local model for the fields MRZ and regex couldn't find.

This is the third extraction step, so it only runs on documents the first two missed. Three
rules shape it:

- **Schema-constrained output.** ``format`` carries a JSON schema, so Ollama restricts the
  model's tokens to valid JSON in our shape. Without it the model invents labels such as
  ``driver_license``. Requests use temperature 0 so the same text gives the same answer, and
  ``think: false`` because thinking doubled latency in testing without changing answers.
- **No self-reported confidence.** The model is never asked how sure it is. That number
  stayed high even on corrupted OCR, so the confidence scorer computes one instead.
- **Trust nothing that comes back.** The reply is checked again here even though the schema
  should already enforce it. An unknown kind or an impossible date becomes ``None`` for that
  field instead of failing the whole document.

Request shape for ``POST /api/chat`` (confirmed against the local Ollama app): ``format``
takes a JSON schema, ``think`` is a top-level boolean, sampling settings go under
``options``, and with ``stream: false`` the reply is one object whose ``message.content``
holds the JSON as a string.
"""

import json
from dataclasses import dataclass
from datetime import date
from types import TracebackType
from typing import Any, Literal, Self, get_args

import httpx

from lifeadmin.config import Settings

LlmDocKind = Literal[
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
]
DOC_KINDS: tuple[str, ...] = get_args(LlmDocKind)

CHAT_PATH = "/api/chat"
# A 9B model on a laptop can take tens of seconds on a long document, and the first call
# after start-up also loads the model into memory.
TIMEOUT_SECONDS = 120.0
# Enough for any ID card or slip. Longer documents (multi-page statements) are cut so one
# file can't stall extraction; the dates we want are almost always near the top.
MAX_TEXT_CHARS = 8000

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_kind": {"enum": [*DOC_KINDS, None]},
        "holder": {"type": ["string", "null"]},
        "issue_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "expiry_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    },
    "required": ["doc_kind", "holder", "issue_date", "expiry_date"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You read OCR text from Canadian personal documents, which may be in English, French or "
    "both, and may contain OCR errors. Return JSON with:\n"
    "- doc_kind: one of " + ", ".join(DOC_KINDS) + ", or null if none fits.\n"
    "- holder: the full name of the person the document belongs to, or null.\n"
    "- issue_date: the date the document was issued, as YYYY-MM-DD, or null.\n"
    "- expiry_date: the date the document expires, as YYYY-MM-DD, or null.\n"
    "Use only what the text says. Never use a date of birth as an issue or expiry date. "
    "If a field is missing or unreadable, use null; do not guess."
)


class LlmError(Exception):
    """Base class for LLM fallback failures."""


class LlmUnavailable(LlmError):
    """Ollama couldn't answer: not running, timed out, or failed on its side.

    Worth retrying later, so the caller should leave the document unprocessed rather than
    record an empty result.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LlmBadResponse(LlmError):
    """Ollama answered, but not with JSON in the expected shape."""


@dataclass(frozen=True)
class LlmResult:
    """What the model claims. Unverified: the confidence scorer checks it against the text."""

    doc_kind: LlmDocKind | None
    holder: str | None
    issue_date: date | None
    expiry_date: date | None


class OllamaExtractor:
    """Calls a local Ollama model to extract document fields.

    ``transport`` exists for tests: pass an ``httpx.MockTransport`` to answer requests
    without a real server.
    """

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._model = settings.ollama_model
        self._client = httpx.Client(
            base_url=settings.ollama_url, timeout=TIMEOUT_SECONDS, transport=transport
        )

    def extract(self, text: str) -> LlmResult:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text[:MAX_TEXT_CHARS]},
            ],
            "format": SCHEMA,
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._client.post(CHAT_PATH, json=body)
        except httpx.HTTPError as exc:
            raise LlmUnavailable(
                f"Could not reach Ollama ({type(exc).__name__}); check OLLAMA_URL"
            ) from exc

        status = response.status_code
        if status == 501:
            # Seen with an Ollama build that lacked schema support.
            raise LlmUnavailable(
                "Ollama can't do structured output (HTTP 501); use an official Ollama build",
                status_code=status,
            )
        if status == 404:
            raise LlmUnavailable(
                f"Ollama has no model {self._model!r} (HTTP 404); run `ollama pull` "
                "or check OLLAMA_MODEL",
                status_code=status,
            )
        if response.is_error:
            raise LlmUnavailable(f"Ollama returned HTTP {status}", status_code=status)

        try:
            content = response.json()["message"]["content"]
            fields = json.loads(content)
        except (ValueError, KeyError, TypeError) as exc:
            raise LlmBadResponse("Ollama reply was not JSON in the expected shape") from exc
        if not isinstance(fields, dict):
            raise LlmBadResponse("Ollama reply was not a JSON object")

        return LlmResult(
            doc_kind=_kind(fields.get("doc_kind")),
            holder=_holder(fields.get("holder")),
            issue_date=_date(fields.get("issue_date")),
            expiry_date=_date(fields.get("expiry_date")),
        )

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


def _kind(value: Any) -> LlmDocKind | None:
    return value if value in DOC_KINDS else None


def _holder(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    # Collapse runs of whitespace and newlines that OCR leaves inside names.
    return " ".join(value.split()) or None


def _date(value: Any) -> date | None:
    """Accept only a strict YYYY-MM-DD calendar date; anything else becomes ``None``."""
    if not isinstance(value, str) or len(value) != 10:
        return None
    try:
        # fromisoformat also takes "20310230" or week dates; the length check above plus
        # the dashes check below keep it to the one form the schema asks for.
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if value[4] == "-" and value[7] == "-" else None
