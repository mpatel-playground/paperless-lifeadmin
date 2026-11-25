"""Scoring rules and tallies of the eval runner, with fake predictors; no Ollama."""

import json
from datetime import date

import httpx
import pytest

from evals.generate import Sample
from evals.run import (
    Case,
    Prediction,
    build_cases,
    evaluate,
    expected_value,
    format_report,
    llm_predictor,
    load_ocr,
    main,
    normalise_holder,
    outcome,
    predict_regex,
    signals,
    summarise,
    write_results,
)
from lifeadmin.config import Settings
from lifeadmin.extract.llm import OllamaExtractor

# Made-up people and dates.
PASSPORT = Sample(
    id="passport-x",
    kind="passport",
    holder="Priya Raman-Côté",
    issue_date=date(2021, 6, 15),
    expiry_date=date(2031, 6, 15),
)
T4 = Sample(id="t4-x", kind="t4", holder="Daniel Okafor", tax_year=2025)


def test_holder_ignores_case_accents_punctuation_and_order():
    assert normalise_holder("Priya Raman-Côté") == normalise_holder("RAMAN COTE PRIYA")
    assert normalise_holder("OKAFOR, DANIEL") == normalise_holder("Daniel Okafor")
    assert normalise_holder("Daniel Okafor") != normalise_holder("Daniel Q Okafor")


@pytest.mark.parametrize(
    ("expected", "predicted", "want"),
    [
        (date(2031, 6, 15), date(2031, 6, 15), "correct"),
        (date(2031, 6, 15), date(2031, 6, 16), "wrong"),
        (date(2031, 6, 15), None, "missing"),
        (None, None, "null_ok"),
        (None, date(2031, 6, 15), "false_pos"),
    ],
)
def test_outcome(expected, predicted, want):
    assert outcome("expiry_date", expected, predicted) == want


def test_holder_outcome_uses_normalised_names():
    assert outcome("holder", "Priya Raman-Côté", "RAMAN COTE PRIYA") == "correct"


def test_mrz_expects_nothing_on_documents_without_an_mrz():
    assert expected_value("mrz", T4, "doc_kind") is None
    assert expected_value("mrz", PASSPORT, "doc_kind") == "passport"
    assert expected_value("regex", T4, "doc_kind") == "t4"
    assert expected_value("llm", T4, "expiry_date") is None


def test_build_cases_adds_seeded_noisy_copies():
    text = "Date of expiry: 2031-06-15 " * 20
    first = build_cases([PASSPORT], {PASSPORT.id: text}, copies=2)
    second = build_cases([PASSPORT], {PASSPORT.id: text}, copies=2)

    assert [c.variant for c in first] == ["clean", "noise-1", "noise-2"]
    assert first[0].text == text
    assert [c.text for c in first] == [c.text for c in second]
    assert first[1].text != first[2].text


def test_load_ocr_reports_missing_file(tmp_path):
    (tmp_path / "a.txt").write_text("text a")

    assert load_ocr(["a"], tmp_path) == {"a": "text a"}
    with pytest.raises(SystemExit, match="eval-fetch"):
        load_ocr(["a", "b"], tmp_path)


def test_predict_regex_takes_first_labelled_dates():
    pred = predict_regex("Issued: 2021-06-15\nExpiry: 2031-06-15\nExpiry: 2032-01-01")

    assert pred.issue_date == date(2021, 6, 15)
    assert pred.expiry_date == date(2031, 6, 15)
    assert pred.expiry_candidates == (date(2031, 6, 15), date(2032, 1, 1))


def test_predict_regex_returns_none_without_anything():
    assert predict_regex("nothing useful here") is None


def test_evaluate_scores_only_attempted_fields():
    cases = [Case(PASSPORT, "clean", "text")]
    predictors = {
        "regex": lambda text: Prediction(doc_kind="passport", expiry_date=date(2031, 6, 15)),
        "mrz": lambda text: None,
    }

    rows, raw = evaluate(cases, predictors)

    regex_fields = {r.field: r.outcome for r in rows if r.method == "regex"}
    assert regex_fields == {
        "doc_kind": "correct",
        "issue_date": "missing",
        "expiry_date": "correct",
    }
    mrz_fields = {r.field: r.outcome for r in rows if r.method == "mrz"}
    assert set(mrz_fields) == {"doc_kind", "holder", "expiry_date"}
    assert set(mrz_fields.values()) == {"missing"}
    assert raw[("passport-x", "clean", "mrz")] is None


def test_summarise_splits_clean_and_noisy():
    cases = [Case(T4, "clean", ""), Case(T4, "noise-1", ""), Case(T4, "noise-2", "")]
    invented = Prediction(expiry_date=date(2030, 1, 1))
    rows, _ = evaluate(cases, {"llm": lambda text: invented if text == "" else None})

    expiry = summarise(rows)["llm"]["expiry_date"]
    assert expiry["clean"]["false_pos"] == 1
    assert expiry["noisy"]["false_pos"] == 2


def test_signals_track_agreement_between_regex_and_llm():
    agree = Case(PASSPORT, "clean", "agree")
    disagree = Case(PASSPORT, "noise-1", "disagree")
    right, wrong = date(2031, 6, 15), date(2031, 6, 16)
    predictors = {
        "regex": lambda text: Prediction(expiry_date=right),
        "llm": lambda text: Prediction(expiry_date=right if text == "agree" else wrong),
    }

    _, raw = evaluate([agree, disagree], predictors)
    tallies = signals([agree, disagree], raw)

    assert (
        tallies["llm+regex agree: shared date"].right,
        tallies["llm+regex agree: shared date"].total,
    ) == (1, 1)
    assert tallies["llm+regex disagree: llm date"].right == 0
    assert tallies["llm+regex disagree: regex date"].right == 1


def test_signals_skip_llm_tallies_when_llm_not_run():
    case = Case(PASSPORT, "clean", "x")
    _, raw = evaluate([case], {"regex": lambda text: Prediction(expiry_date=date(2031, 6, 15))})

    tallies = signals([case], raw)

    assert not any(name.startswith("llm") for name in tallies)
    assert tallies["regex: date as printed"].right == 1


def test_report_lists_wrong_answers_and_results_file_is_json(tmp_path):
    case = Case(T4, "clean", "")
    rows, raw = evaluate([case], {"llm": lambda text: Prediction(expiry_date=date(2030, 1, 1))})
    tallies = signals([case], raw)

    report = format_report(rows, tallies)
    assert "t4-x" in report and "got 2030-01-01" in report

    path = tmp_path / "out" / "latest.json"
    write_results(rows, tallies, path)
    data = json.loads(path.read_text())
    assert any(r["outcome"] == "false_pos" for r in data["rows"])


def test_llm_predictor_turns_bad_reply_into_error_prediction():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "not json"}})

    settings = Settings(paperless_url="http://p.test", paperless_token="fake")
    extractor = OllamaExtractor(settings, transport=httpx.MockTransport(handler))

    pred = llm_predictor(extractor)("text")

    assert pred.error and pred.expiry_date is None


def test_main_without_llm_never_needs_settings(tmp_path, monkeypatch, capsys):
    import evals.run as run

    monkeypatch.setattr(run, "load_samples", lambda: [PASSPORT])
    monkeypatch.setattr(run, "load_ocr", lambda ids: {PASSPORT.id: "Expiry: 2031-06-15"})
    monkeypatch.setattr(run, "RESULTS_PATH", tmp_path / "evals" / "results" / "latest.json")
    monkeypatch.setattr(run, "EVALS_DIR", tmp_path / "evals")
    monkeypatch.setattr(run, "load_settings", lambda: pytest.fail("settings loaded"))

    main(["--no-llm"])

    out = capsys.readouterr().out
    assert "regex" in out and "llm " not in out
