"""Combine MRZ, regex and LLM results into one answer with a computed confidence.

The confidence is the add-on's own estimate, built from checks it can verify. The LLM is
never asked how sure it is. The signals, from ``docs/DESIGN.md``:

- **Method:** a valid MRZ keeps its own confidence (1.0 less OCR-fix penalties). A labelled
  regex date starts high; an LLM-only answer starts below the cut-off.
- **Grounding:** an LLM date must be printed somewhere in the OCR text.
- **Agreement:** regex and LLM giving the same answer raises the score; different answers
  send the document to the verify queue.
- **Plausibility:** expiry after issue, a validity period that fits the kind, no dates
  implausibly far out.
- **OCR corrections:** a date that only reads after undoing OCR swaps costs a little.

``confidence`` covers what is written back to Paperless: the kind and the expiry date
(for payslips, T4s and NOAs, the issue date instead, when there is one). Below
``VERIFY_BELOW`` the document gets the ``lifeadmin:verify`` tag.

Weights below cite the eval run they come from (``make eval``, 119 cases: 17 samples, clean
plus 3 light and 3 heavy OCR-noise copies each, qwen3.5:9b). They are deliberately
conservative where the eval has few cases.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from lifeadmin.extract.llm import LlmResult
from lifeadmin.extract.mrz import MrzResult
from lifeadmin.extract.patterns import DateMatch, FoundDate, KindMatch, find_all_dates

Method = Literal["mrz", "regex", "llm"]

VERIFY_BELOW = 0.8

# Kinds that never carry an expiry date; an expiry found on one is discarded.
NO_EXPIRY_KINDS = ("payslip", "t4", "noa", "employment_letter", "credit_report", "bank_statement")

# --- Dates ---
# Regex and LLM gave the same date: right 49/49 times.
AGREE = 0.97
# A labelled regex date with no LLM answer: right 5/5 when the LLM gave none, and 30/30
# when printed as-is.
REGEX_BASE = 0.9
# Regex date needed OCR swaps undone. Expiry dates were right 24/25 (the miss, 2081, fails
# plausibility), but two payslip pay dates were accepted wrong at a 0.05 penalty: heavy
# noise turned "Oct 16" into "Oct l5", one fixable swap plus a digit-for-digit one no check
# can see. So a corrected date needs a second method: alone it lands below the cut-off
# (0.75), confirmed by the LLM it still passes (0.82).
OCR_FIX_PENALTY = 0.15
# 03/04/2029: day and month could be swapped. Always a human check.
AMBIGUOUS_PENALTY = 0.2
# More than one different date under the same kind of label.
SEVERAL_DATES_PENALTY = 0.2
# LLM only, date printed in the text. Below the cut-off: one such case in the eval, and it
# was wrong, so an unconfirmed LLM date always gets a human check.
LLM_GROUNDED = 0.75
LLM_GROUNDED_AFTER_FIX = 0.65
# LLM date not printed anywhere in the text: likely invented or misread.
LLM_UNGROUNDED = 0.4
# Regex and LLM disagree.
DISAGREE_CAP = 0.6
# An ID kind with no expiry found. Some cards have none (old Ontario health cards), but a
# person should confirm that.
MISSING_EXPIRY = 0.5

# --- Plausibility ---
IMPLAUSIBLE_CAP = 0.5
MAX_YEARS_AHEAD = 11  # a 10-year passport issued today, plus a margin
MAX_YEARS_BEHIND = 30
# Allowed issue-to-expiry periods in days, per kind. Canadian passports run 5 or 10 years
# and PR cards usually 5 (sometimes 1); Ontario licences and health cards renew on a
# birthday, so their period varies.
_YEAR = 365.25
_SLACK = 45


def _around(years: int) -> tuple[float, float]:
    return (years * _YEAR - _SLACK, years * _YEAR + _SLACK)


VALIDITY_DAYS: dict[str, tuple[tuple[float, float], ...]] = {
    "passport": (_around(5), _around(10)),
    "pr_card": (_around(1), _around(5)),
    "drivers_licence": ((0.5 * _YEAR, 6 * _YEAR),),
    "health_card": ((0.5 * _YEAR, 6 * _YEAR),),
}

# --- Kind ---
# Regex and LLM named the same kind.
KIND_AGREE = 0.98
# Regex found both a card-number format and a title.
KIND_NUMBER_AND_TITLE = 0.95
KIND_REGEX_ONE_SIGNAL = 0.9
# LLM only: right 119/119 in the eval, but without corroboration it stays just above the
# cut-off.
KIND_LLM_ONLY = 0.85
KIND_DISAGREE_CAP = 0.5
# A card number that only matched after OCR swaps; the kind is rarely wrong for it.
KIND_OCR_FIX_PENALTY = 0.05


@dataclass(frozen=True)
class Extraction:
    doc_kind: str | None
    holder: str | None = field(repr=False)  # personal data: kept out of logs
    issue_date: date | None
    expiry_date: date | None
    extracted_by: Method | None
    confidence: float
    reasons: tuple[str, ...]  # why the confidence is what it is, for the verify queue

    @property
    def needs_verify(self) -> bool:
        return self.confidence < VERIFY_BELOW


@dataclass(frozen=True)
class _Scored:
    value: object
    confidence: float
    method: Method | None
    reasons: tuple[str, ...] = ()


def score(
    text: str,
    *,
    mrz: MrzResult | None,
    dates: list[DateMatch],
    kind: KindMatch | None,
    llm: LlmResult | None,
    today: date,
) -> Extraction:
    """Pick each field's value and compute the document's confidence.

    ``llm`` is ``None`` when the LLM wasn't run; regex answers then stand on their own.
    """
    printed = find_all_dates(text)
    kind_s = _score_kind(mrz, kind, llm)
    doc_kind = kind_s.value

    issue_s = _score_date(
        "issue", [d for d in dates if d.field == "issue"], llm.issue_date if llm else None, printed
    )
    # Only a trusted issue date may cast doubt on an expiry date. Otherwise an unconfirmed
    # LLM issue date could pull down a check-digit-verified MRZ expiry.
    trusted_issue = issue_s.value if issue_s.confidence >= VERIFY_BELOW else None
    if mrz is not None:
        expiry_s = _Scored(mrz.expiry_date, mrz.confidence, "mrz")
    else:
        expiry_s = _score_date(
            "expiry",
            [d for d in dates if d.field == "expiry"],
            llm.expiry_date if llm else None,
            printed,
            issue=trusted_issue,
            kind=doc_kind,
            today=today,
        )

    reasons = [*kind_s.reasons, *issue_s.reasons]
    if doc_kind in NO_EXPIRY_KINDS:
        if expiry_s.value is not None:
            reasons.append(f"expiry date ignored: a {doc_kind} has none")
        expiry_s = _Scored(None, 1.0, None)
        # The issue date (pay date, date issued) is what these documents are dated by.
        date_conf = issue_s.confidence if issue_s.value is not None else 1.0
    else:
        reasons += expiry_s.reasons
        if expiry_s.value is None:
            reasons.append("no expiry date found")
            expiry_s = _Scored(None, MISSING_EXPIRY, None)
        else:
            expiry_s = _plausibility(expiry_s, trusted_issue, doc_kind, today, reasons)
        date_conf = expiry_s.confidence

    confidence = min(kind_s.confidence, date_conf) if doc_kind else 0.0
    if doc_kind is None:
        reasons.append("document kind unknown")

    holder = mrz.holder if mrz else (llm.holder if llm else None)
    by = expiry_s.method or (issue_s.method if issue_s.value else None) or kind_s.method
    return Extraction(
        doc_kind=doc_kind,
        holder=holder,
        issue_date=issue_s.value,
        expiry_date=expiry_s.value,
        extracted_by=by,
        confidence=round(max(0.0, confidence), 2),
        reasons=tuple(reasons),
    )


def _score_kind(mrz: MrzResult | None, kind: KindMatch | None, llm: LlmResult | None) -> _Scored:
    if mrz is not None and mrz.doc_kind is not None:
        return _Scored(mrz.doc_kind, mrz.confidence, "mrz")
    llm_kind = llm.doc_kind if llm else None
    if kind is not None and llm_kind is not None:
        if kind.doc_kind == llm_kind:
            return _Scored(kind.doc_kind, KIND_AGREE, "regex")
        return _Scored(
            kind.doc_kind,
            KIND_DISAGREE_CAP,
            "regex",
            (f"kind: regex says {kind.doc_kind}, llm says {llm_kind}",),
        )
    if kind is not None:
        conf = KIND_NUMBER_AND_TITLE if kind.by_number and kind.by_title else KIND_REGEX_ONE_SIGNAL
        if kind.ocr_corrected:
            conf -= KIND_OCR_FIX_PENALTY
        return _Scored(kind.doc_kind, conf, "regex")
    if llm_kind is not None:
        return _Scored(llm_kind, KIND_LLM_ONLY, "llm", ("kind from llm only",))
    return _Scored(None, 0.0, None)


def _score_date(
    name: str,
    matches: list[DateMatch],
    llm_date: date | None,
    printed: list[FoundDate],
    *,
    issue: object = None,
    kind: object = None,
    today: date | None = None,
) -> _Scored:
    """One date field from its labelled regex matches and the LLM's answer."""
    first = matches[0] if matches else None
    regex_date = first.value if first else None

    if regex_date is not None and llm_date is not None and regex_date != llm_date:
        # Prefer the reading that passes the plausibility checks; regex on a tie.
        value, method = regex_date, "regex"
        if (
            today is not None
            and _implausible(regex_date, issue, kind, today)
            and not _implausible(llm_date, issue, kind, today)
        ):
            value, method = llm_date, "llm"
        return _Scored(
            value,
            DISAGREE_CAP,
            method,
            (f"{name} date: regex and llm disagree",),
        )

    if regex_date is not None:
        reasons = []
        conf = AGREE if llm_date == regex_date else REGEX_BASE
        if first.ocr_corrected:
            conf -= OCR_FIX_PENALTY
            reasons.append(f"{name} date read after OCR fixes")
        if first.ambiguous:
            conf -= AMBIGUOUS_PENALTY
            reasons.append(f"{name} date: day and month could be swapped")
        if len({m.value for m in matches}) > 1:
            conf -= SEVERAL_DATES_PENALTY
            reasons.append(f"several different {name} dates")
        return _Scored(regex_date, conf, "regex", tuple(reasons))

    if llm_date is not None:
        grounding = _grounding(llm_date, printed)
        if grounding == "as printed":
            return _Scored(llm_date, LLM_GROUNDED, "llm", (f"{name} date from llm only",))
        if grounding == "after fixes":
            return _Scored(
                llm_date,
                LLM_GROUNDED_AFTER_FIX,
                "llm",
                (f"{name} date from llm only, found in text after OCR fixes",),
            )
        return _Scored(
            llm_date, LLM_UNGROUNDED, "llm", (f"{name} date from llm is not in the text",)
        )

    return _Scored(None, 1.0, None)


def _grounding(value: date, printed: list[FoundDate]) -> Literal["as printed", "after fixes", None]:
    """Whether ``value`` is printed in the text, and whether only after OCR fixes."""
    best = None
    for d in printed:
        matches = d.value == value or (d.ambiguous and _swap_day_month(d.value) == value)
        if not matches:
            continue
        if not d.ocr_corrected and not d.ambiguous:
            return "as printed"
        best = "after fixes"
    return best


def _swap_day_month(d: date) -> date | None:
    try:
        return d.replace(month=d.day, day=d.month)
    except ValueError:
        return None


def _implausible(expiry: date, issue: object, kind: object, today: date) -> str | None:
    """Why ``expiry`` can't be right, or ``None`` if it passes every check."""
    if expiry > today + timedelta(days=round(MAX_YEARS_AHEAD * _YEAR)):
        return f"expiry date more than {MAX_YEARS_AHEAD} years away"
    if expiry < today - timedelta(days=round(MAX_YEARS_BEHIND * _YEAR)):
        return f"expiry date more than {MAX_YEARS_BEHIND} years ago"
    if isinstance(issue, date):
        if expiry <= issue:
            return "expiry date is not after the issue date"
        periods = VALIDITY_DAYS.get(str(kind))
        days = (expiry - issue).days
        if periods and not any(lo <= days <= hi for lo, hi in periods):
            return f"validity period of {days / _YEAR:.1f} years is unusual for a {kind}"
    return None


def _plausibility(
    expiry: _Scored, issue: object, kind: object, today: date, reasons: list[str]
) -> _Scored:
    problem = _implausible(expiry.value, issue, kind, today)
    if problem is None:
        return expiry
    reasons.append(problem)
    return _Scored(expiry.value, min(expiry.confidence, IMPLAUSIBLE_CAP), expiry.method)
