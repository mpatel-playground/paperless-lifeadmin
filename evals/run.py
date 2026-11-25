"""Measure each extraction method against the answer key in ``evals/labels.toml``.

Run with ``make eval`` (or ``make eval ARGS=--no-llm`` without Ollama). Input is the real
Paperless OCR text saved by ``make eval-fetch``, plus seeded noised copies of it from
``evals.noise``.

Each method runs on its own, so the numbers show what MRZ, regex and the LLM are each
worth. That is the input for the confidence scorer's weights: the "signals" section reports
how often a method was right when some checkable condition held (regex and LLM agreed, a
date needed OCR fixes, ...). The scorer, and accuracy at each confidence level, come later.

How a field is scored:

- The answer key has a value: ``correct``, ``wrong`` or ``missing`` (no answer).
- The answer key has no value (a T4 has no expiry date): ``null_ok`` if the method also
  gives none, ``false_pos`` if it invents one. A made-up expiry date is the worst error the
  add-on can make, so it is counted on its own.
- A method is only scored on fields it attempts (regex doesn't read names, MRZ carries no
  issue date). MRZ on a document without one should return nothing.
"""

import argparse
import json
import sys
import unicodedata
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from evals.fetch import OCR_DIR
from evals.generate import EVALS_DIR, Sample, load_samples
from evals.noise import add_noise
from lifeadmin.config import load_settings
from lifeadmin.extract.llm import LlmBadResponse, LlmUnavailable, OllamaExtractor
from lifeadmin.extract.mrz import parse_mrz
from lifeadmin.extract.patterns import find_dates, find_kind

RESULTS_PATH = EVALS_DIR / "results" / "latest.json"
NOISE_COPIES = 2

FieldName = Literal["doc_kind", "holder", "issue_date", "expiry_date"]
FIELDS: tuple[FieldName, ...] = ("doc_kind", "holder", "issue_date", "expiry_date")
ATTEMPTS: dict[str, tuple[FieldName, ...]] = {
    "mrz": ("doc_kind", "holder", "expiry_date"),
    "regex": ("doc_kind", "issue_date", "expiry_date"),
    "llm": FIELDS,
}
MRZ_KINDS = ("passport", "pr_card")

Outcome = Literal["correct", "wrong", "missing", "null_ok", "false_pos"]
OUTCOMES: tuple[Outcome, ...] = ("correct", "wrong", "missing", "null_ok", "false_pos")


@dataclass(frozen=True)
class Case:
    sample: Sample
    variant: str  # "clean", "noise-1", ...
    text: str = field(repr=False)

    @property
    def noisy(self) -> bool:
        return self.variant != "clean"


@dataclass(frozen=True)
class Prediction:
    doc_kind: str | None = None
    holder: str | None = None
    issue_date: date | None = None
    expiry_date: date | None = None
    # Extra facts a method reports, kept for the signal tallies.
    confidence: float | None = None  # MRZ only
    ocr_corrected: bool = False  # regex: the chosen expiry date needed OCR swaps undone
    expiry_candidates: tuple[date, ...] = ()  # regex: every labelled expiry date
    error: str | None = None  # LLM: the reply couldn't be used


Predictor = Callable[[str], Prediction | None]


@dataclass(frozen=True)
class Row:
    sample_id: str
    variant: str
    method: str
    field: FieldName
    expected: str | None
    predicted: str | None
    outcome: Outcome


# --- Cases ---


def load_ocr(sample_ids: Iterable[str], ocr_dir: Path = OCR_DIR) -> dict[str, str]:
    texts = {}
    for sample_id in sample_ids:
        path = ocr_dir / f"{sample_id}.txt"
        if not path.exists():
            raise SystemExit(f"Missing {path}; run `make eval-fetch` first")
        texts[sample_id] = path.read_text(encoding="utf-8")
    return texts


def build_cases(
    samples: Sequence[Sample], texts: Mapping[str, str], copies: int = NOISE_COPIES
) -> list[Case]:
    """The clean text of each sample plus ``copies`` noised versions of it."""
    cases = []
    for s in samples:
        cases.append(Case(s, "clean", texts[s.id]))
        for n in range(1, copies + 1):
            # crc32, not hash(): Python salts str hashes per process, which would change
            # the noise on every run.
            seed = zlib.crc32(f"{s.id}:{n}".encode())
            cases.append(Case(s, f"noise-{n}", add_noise(texts[s.id], seed)))
    return cases


# --- Methods ---


def predict_mrz(text: str) -> Prediction | None:
    result = parse_mrz(text)
    if result is None:
        return None
    return Prediction(
        doc_kind=result.doc_kind,
        holder=result.holder,
        expiry_date=result.expiry_date,
        confidence=result.confidence,
    )


def predict_regex(text: str) -> Prediction | None:
    """The first labelled date of each field, and the kind from card formats and titles."""
    dates = find_dates(text)
    kind = find_kind(text)
    if not dates and kind is None:
        return None
    issue = next((d for d in dates if d.field == "issue"), None)
    expiry = next((d for d in dates if d.field == "expiry"), None)
    return Prediction(
        doc_kind=kind.doc_kind if kind else None,
        issue_date=issue.value if issue else None,
        expiry_date=expiry.value if expiry else None,
        ocr_corrected=bool(expiry and expiry.ocr_corrected),
        expiry_candidates=tuple(d.value for d in dates if d.field == "expiry"),
    )


def llm_predictor(extractor: OllamaExtractor) -> Predictor:
    def predict(text: str) -> Prediction:
        try:
            r = extractor.extract(text)
        except LlmBadResponse as exc:
            return Prediction(error=str(exc))
        return Prediction(
            doc_kind=r.doc_kind, holder=r.holder, issue_date=r.issue_date, expiry_date=r.expiry_date
        )

    return predict


# --- Scoring ---


def normalise_holder(name: str) -> frozenset[str]:
    """Name words ignoring case, accents, punctuation and order.

    The MRZ prints ``RAMAN COTE PRIYA`` and the LLM may answer ``RAMAN-COTE, PRIYA``; both
    are the label ``Priya Raman-Côté``.
    """
    folded = unicodedata.normalize("NFKD", name.casefold())
    letters = "".join(c if c.isalnum() else " " for c in folded if not unicodedata.combining(c))
    return frozenset(letters.split())


def _same(field_name: FieldName, expected: object, predicted: object) -> bool:
    if field_name == "holder":
        return normalise_holder(str(expected)) == normalise_holder(str(predicted))
    return expected == predicted


def outcome(field_name: FieldName, expected: object, predicted: object) -> Outcome:
    if expected is None:
        return "null_ok" if predicted is None else "false_pos"
    if predicted is None:
        return "missing"
    return "correct" if _same(field_name, expected, predicted) else "wrong"


def expected_value(method: str, sample: Sample, field_name: FieldName) -> object:
    if method == "mrz" and sample.kind not in MRZ_KINDS:
        return None  # no MRZ printed, so the right answer is no answer
    return sample.kind if field_name == "doc_kind" else getattr(sample, field_name)


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def evaluate(
    cases: Iterable[Case], predictors: Mapping[str, Predictor]
) -> tuple[list[Row], dict[tuple[str, str, str], Prediction | None]]:
    """Score every method on every case. Also returns the raw predictions for the signals."""
    rows: list[Row] = []
    raw: dict[tuple[str, str, str], Prediction | None] = {}
    for case in cases:
        for method, predict in predictors.items():
            pred = predict(case.text)
            raw[(case.sample.id, case.variant, method)] = pred
            for f in ATTEMPTS[method]:
                expected = expected_value(method, case.sample, f)
                predicted = getattr(pred, f) if pred else None
                rows.append(
                    Row(
                        case.sample.id,
                        case.variant,
                        method,
                        f,
                        _text(expected),
                        _text(predicted),
                        outcome(f, expected, predicted),
                    )
                )
    return rows, raw


def summarise(rows: Iterable[Row]) -> dict[str, dict[str, dict[str, Counter[str]]]]:
    """Outcome counts by method, then field, then "clean" / "noisy"."""
    table: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for r in rows:
        group = "clean" if r.variant == "clean" else "noisy"
        table[r.method][r.field][group][r.outcome] += 1
    return table


@dataclass
class Tally:
    """How often an answer was right under one condition."""

    right: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.right += ok
        self.total += 1

    def __str__(self) -> str:
        if not self.total:
            return "  -"
        return f"{self.right:>3}/{self.total:<3} ({100 * self.right / self.total:.0f}%)"


def signals(
    cases: Iterable[Case], raw: Mapping[tuple[str, str, str], Prediction | None]
) -> dict[str, Tally]:
    """Expiry-date accuracy under each condition the scorer can check without a label.

    "Right" includes correctly giving no date for a document without one.
    """
    t: dict[str, Tally] = defaultdict(Tally)
    for case in cases:
        key = (case.sample.id, case.variant)
        want = case.sample.expiry_date
        mrz = raw.get((*key, "mrz"))
        regex = raw.get((*key, "regex"))
        llm = raw.get((*key, "llm"))

        if mrz:
            t["mrz: result, confidence 1.0" if mrz.confidence == 1.0 else "mrz: result, < 1.0"].add(
                mrz.expiry_date == want
            )

        regex_date = regex.expiry_date if regex else None
        if regex_date:
            t["regex: date as printed" if not regex.ocr_corrected else "regex: OCR-corrected"].add(
                regex_date == want
            )
            if len(set(regex.expiry_candidates)) > 1:
                t["regex: several different expiry dates"].add(regex_date == want)

        if (case.sample.id, case.variant, "llm") not in raw:
            continue
        llm_date = llm.expiry_date if llm else None
        if regex_date and llm_date:
            if regex_date == llm_date:
                t["llm+regex agree: shared date"].add(llm_date == want)
            else:
                t["llm+regex disagree: llm date"].add(llm_date == want)
                t["llm+regex disagree: regex date"].add(regex_date == want)
        elif llm_date:
            t["llm only (regex none): llm date"].add(llm_date == want)
        elif regex_date:
            t["regex only (llm none): regex date"].add(regex_date == want)
        else:
            t["neither gives a date: none is right"].add(want is None)
        if llm and llm.error:
            t["llm: unusable reply"].add(False)
    return dict(t)


# --- Report ---


def format_report(rows: Sequence[Row], tallies: Mapping[str, Tally]) -> str:
    table = summarise(rows)
    out = [
        "Accuracy by method and field. Has value: correct / wrong / missing.",
        "No value in answer key: null_ok / false_pos (an invented value).",
        "",
        f"{'method':<6} {'field':<12} {'set':<6} {'correct':>8} {'wrong':>6} {'missing':>8}"
        f" {'null_ok':>8} {'false+':>7}",
    ]
    for method, fields in table.items():
        for f, groups in fields.items():
            for group in ("clean", "noisy"):
                c = groups.get(group, Counter())
                has = c["correct"] + c["wrong"] + c["missing"]
                acc = f"{100 * c['correct'] / has:.0f}%" if has else "-"
                out.append(
                    f"{method:<6} {f:<12} {group:<6} {c['correct']:>3}/{has:<3} {acc:>4}"
                    f" {c['wrong']:>2} {c['missing']:>8} {c['null_ok']:>8} {c['false_pos']:>7}"
                )
    out += ["", "Signals (expiry date right / cases):"]
    out += [f"  {name:<40} {tally}" for name, tally in sorted(tallies.items())]

    errors = [r for r in rows if r.outcome in ("wrong", "false_pos")]
    if errors:
        out += ["", "Wrong answers:"]
        out += [
            f"  {r.method:<5} {r.sample_id:<19} {r.variant:<8} {r.field:<12}"
            f" want {r.expected}  got {r.predicted}"
            for r in errors
        ]
    return "\n".join(out)


def write_results(
    rows: Sequence[Row], tallies: Mapping[str, Tally], path: Path = RESULTS_PATH
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "rows": [asdict(r) for r in rows],
        "signals": {k: {"right": v.right, "total": v.total} for k, v in tallies.items()},
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-llm", action="store_true", help="skip the Ollama fallback")
    args = parser.parse_args(argv)

    samples = load_samples()
    cases = build_cases(samples, load_ocr(s.id for s in samples))
    predictors: dict[str, Predictor] = {"mrz": predict_mrz, "regex": predict_regex}

    extractor = None
    if not args.no_llm:
        # Only the Ollama settings are used; the eval never talks to Paperless.
        settings = load_settings()
        extractor = OllamaExtractor(settings)
        predictors["llm"] = llm_predictor(extractor)
        print(f"Running {len(cases)} cases with {settings.ollama_model}...", file=sys.stderr)

    try:
        rows, raw = evaluate(cases, predictors)
    except LlmUnavailable as exc:
        raise SystemExit(f"{exc}\nStart Ollama, or run with --no-llm") from exc
    finally:
        if extractor:
            extractor.close()

    tallies = signals(cases, raw)
    print(format_report(rows, tallies))
    write_results(rows, tallies, RESULTS_PATH)
    print(f"\nFull results: {RESULTS_PATH.relative_to(EVALS_DIR.parent)}", file=sys.stderr)


if __name__ == "__main__":
    main()
