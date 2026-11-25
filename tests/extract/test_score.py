"""Confidence scoring: one test per rule, with hand-built inputs (made-up data)."""

from datetime import date

import pytest

from lifeadmin.extract import score as s
from lifeadmin.extract.llm import LlmResult
from lifeadmin.extract.mrz import MrzResult
from lifeadmin.extract.patterns import DateMatch, KindMatch
from lifeadmin.extract.score import VERIFY_BELOW, score

TODAY = date(2026, 11, 1)
ISSUE = date(2024, 3, 14)
EXPIRY = date(2029, 3, 14)
TEXT = "ONTARIO DRIVER'S LICENCE\n4a ISS 2024/03/14\n4b EXP 2029/03/14"
LICENCE = KindMatch("drivers_licence", by_number=True, by_title=True, ocr_corrected=False)


def labelled(field, value, *, corrected=False, ambiguous=False):
    return DateMatch(field, value, "EXP" if field == "expiry" else "ISS", corrected, ambiguous)


def llm(**fields):
    base = {
        "doc_kind": "drivers_licence",
        "holder": "Jane Q Tester",
        "issue_date": ISSUE,
        "expiry_date": EXPIRY,
    }
    return LlmResult(**(base | fields))


def run(text=TEXT, *, mrz=None, dates=None, kind=LICENCE, llm_result=None):
    if dates is None:
        dates = [labelled("issue", ISSUE), labelled("expiry", EXPIRY)]
    return score(text, mrz=mrz, dates=dates, kind=kind, llm=llm_result, today=TODAY)


# --- Method ---


def test_valid_mrz_keeps_its_own_confidence():
    mrz = MrzResult("TD3", "passport", "CAN", "TESTER JANE", date(2031, 6, 15), 0.98)

    result = run("passport text", mrz=mrz, dates=[], kind=None, llm_result=None)

    assert result.doc_kind == "passport"
    assert result.expiry_date == date(2031, 6, 15)
    assert result.holder == "TESTER JANE"
    assert result.extracted_by == "mrz"
    assert result.confidence == 0.98


def test_mrz_wins_over_a_disagreeing_llm():
    mrz = MrzResult("TD3", "passport", "CAN", "TESTER JANE", date(2031, 6, 15), 1.0)

    result = run(mrz=mrz, dates=[], kind=None, llm_result=llm(expiry_date=date(2031, 6, 16)))

    assert result.expiry_date == date(2031, 6, 15)
    assert result.confidence == 1.0


def test_regex_alone_passes_the_cut_off():
    result = run()

    assert result.expiry_date == EXPIRY
    assert result.extracted_by == "regex"
    assert result.confidence == s.REGEX_BASE
    assert not result.needs_verify


# --- Agreement ---


def test_agreement_raises_the_score():
    assert run(llm_result=llm()).confidence == s.AGREE


def test_disagreement_always_goes_to_verify():
    result = run(llm_result=llm(expiry_date=date(2029, 3, 18)))

    assert result.expiry_date == EXPIRY  # regex wins a tie
    assert result.confidence < VERIFY_BELOW
    assert "expiry date: regex and llm disagree" in result.reasons


def test_disagreement_prefers_the_plausible_reading():
    dates = [labelled("issue", ISSUE), labelled("expiry", date(2079, 3, 14), corrected=True)]

    result = run(dates=dates, llm_result=llm())

    assert result.expiry_date == EXPIRY
    assert result.extracted_by == "llm"
    assert result.needs_verify


def test_kind_disagreement_goes_to_verify():
    result = run(llm_result=llm(doc_kind="health_card"))

    assert result.doc_kind == "drivers_licence"
    assert result.needs_verify


# --- OCR corrections, ambiguity, several dates ---


def test_ocr_corrected_regex_date_alone_goes_to_verify():
    dates = [labelled("issue", ISSUE), labelled("expiry", EXPIRY, corrected=True)]

    result = run(dates=dates)

    assert result.confidence == pytest.approx(s.REGEX_BASE - s.OCR_FIX_PENALTY)
    assert result.needs_verify
    assert "expiry date read after OCR fixes" in result.reasons


def test_ocr_corrected_regex_date_confirmed_by_llm_passes():
    dates = [labelled("issue", ISSUE), labelled("expiry", EXPIRY, corrected=True)]

    result = run(dates=dates, llm_result=llm())

    assert result.confidence == pytest.approx(s.AGREE - s.OCR_FIX_PENALTY)
    assert not result.needs_verify


def test_ocr_corrected_card_number_costs_the_kind_only_a_little():
    corrected = KindMatch("drivers_licence", by_number=True, by_title=True, ocr_corrected=True)

    result = run(kind=corrected)

    assert result.confidence == pytest.approx(s.KIND_NUMBER_AND_TITLE - s.KIND_OCR_FIX_PENALTY)
    assert not result.needs_verify


def test_ambiguous_day_month_goes_to_verify():
    dates = [labelled("issue", ISSUE), labelled("expiry", EXPIRY, ambiguous=True)]

    assert run(dates=dates).needs_verify


def test_several_different_expiry_dates_go_to_verify():
    dates = [labelled("issue", ISSUE), labelled("expiry", EXPIRY), labelled("expiry", ISSUE)]

    result = run(dates=dates)

    assert result.needs_verify
    assert "several different expiry dates" in result.reasons


def test_the_same_date_twice_is_not_several():
    dates = [labelled("issue", ISSUE), labelled("expiry", EXPIRY), labelled("expiry", EXPIRY)]

    assert not run(dates=dates).needs_verify


# --- Grounding (LLM only) ---


def test_llm_only_date_printed_in_text_still_goes_to_verify():
    result = run(dates=[], llm_result=llm())

    assert result.expiry_date == EXPIRY
    assert result.extracted_by == "llm"
    assert result.needs_verify
    assert result.confidence == s.LLM_GROUNDED


def test_ungrounded_llm_date_never_passes():
    result = run(dates=[], llm_result=llm(expiry_date=date(2029, 3, 17)))

    assert result.confidence <= s.LLM_UNGROUNDED
    assert "expiry date from llm is not in the text" in result.reasons


def test_llm_date_found_only_after_ocr_fixes_scores_lower():
    text = "4b EXP 2O29/O3/14\n4a ISS 2024/03/14"

    result = run(text, dates=[], llm_result=llm())

    assert result.confidence == s.LLM_GROUNDED_AFTER_FIX


def test_llm_date_matching_an_ambiguous_date_read_the_other_way_is_grounded_after_fix():
    text = "Expires 03/04/2029"  # day first reads 3 April; the LLM said 4 March

    result = run(text, dates=[], kind=None, llm_result=llm(expiry_date=date(2029, 3, 4)))

    assert result.expiry_date == date(2029, 3, 4)
    assert "expiry date from llm only, found in text after OCR fixes" in result.reasons


# --- Plausibility ---


@pytest.mark.parametrize(
    ("expiry", "reason"),
    [
        (date(2081, 3, 14), "more than 11 years away"),
        (date(1981, 3, 14), "more than 30 years ago"),
        (date(2023, 3, 14), "not after the issue date"),
        (date(2034, 3, 14), "validity period of 10.0 years is unusual for a drivers_licence"),
    ],
)
def test_implausible_expiry_goes_to_verify(expiry, reason):
    result = run(dates=[labelled("issue", ISSUE), labelled("expiry", expiry)])

    assert result.needs_verify
    assert any(reason in r for r in result.reasons)


@pytest.mark.parametrize("years", [5, 10])
def test_passport_validity_of_5_or_10_years_is_plausible(years):
    issue = date(2021, 6, 15)
    expiry = issue.replace(year=2021 + years)
    passport = KindMatch("passport", by_number=False, by_title=True, ocr_corrected=False)
    dates = [labelled("issue", issue), labelled("expiry", expiry)]

    assert not run(dates=dates, kind=passport).needs_verify


def test_passport_valid_for_7_years_is_implausible():
    passport = KindMatch("passport", by_number=False, by_title=True, ocr_corrected=False)
    dates = [labelled("issue", date(2021, 6, 15)), labelled("expiry", date(2028, 6, 15))]

    assert run(dates=dates, kind=passport).needs_verify


def test_id_without_expiry_goes_to_verify():
    result = run(dates=[labelled("issue", ISSUE)])

    assert result.expiry_date is None
    assert result.needs_verify
    assert "no expiry date found" in result.reasons


# --- Documents without an expiry ---


PAYSLIP = KindMatch("payslip", by_number=False, by_title=True, ocr_corrected=False)


def test_expiry_on_a_payslip_is_discarded():
    dates = [labelled("issue", date(2026, 10, 16)), labelled("expiry", date(2027, 1, 1))]

    result = run(dates=dates, kind=PAYSLIP)

    assert result.expiry_date is None
    assert result.issue_date == date(2026, 10, 16)
    assert not result.needs_verify
    assert "expiry date ignored: a payslip has none" in result.reasons


def test_t4_without_any_date_is_fine():
    t4 = KindMatch("t4", by_number=False, by_title=True, ocr_corrected=False)

    result = run(
        dates=[], kind=t4, llm_result=llm(doc_kind="t4", issue_date=None, expiry_date=None)
    )

    assert result.confidence == s.KIND_AGREE
    assert result.extracted_by == "regex"


def test_payslip_issue_date_disagreement_goes_to_verify():
    dates = [labelled("issue", date(2026, 10, 16))]
    reply = llm(doc_kind="payslip", issue_date=date(2026, 10, 15), expiry_date=None)

    assert run(dates=dates, kind=PAYSLIP, llm_result=reply).needs_verify


# --- Kind ---


def test_kind_from_one_regex_signal():
    title_only = KindMatch("drivers_licence", by_number=False, by_title=True, ocr_corrected=False)

    assert run(kind=title_only).confidence == s.KIND_REGEX_ONE_SIGNAL


def test_kind_from_llm_only():
    result = run(kind=None, llm_result=llm(expiry_date=EXPIRY))

    assert result.doc_kind == "drivers_licence"
    assert result.confidence == s.KIND_LLM_ONLY
    assert "kind from llm only" in result.reasons


def test_unknown_kind_scores_zero():
    result = run(kind=None, llm_result=None)

    assert result.doc_kind is None
    assert result.confidence == 0.0
    assert "document kind unknown" in result.reasons


def test_mrz_without_a_kind_falls_back_to_regex_kind():
    mrz = MrzResult("TD1", None, "USA", "TESTER JANE", EXPIRY, 0.7)

    result = run(mrz=mrz)

    assert result.doc_kind == "drivers_licence"
    assert result.confidence == 0.7  # the MRZ expiry keeps the MRZ's capped confidence


def test_holder_is_never_in_repr():
    assert "Jane" not in repr(run(llm_result=llm()))


def test_untrusted_issue_date_cannot_pull_down_an_mrz_expiry():
    mrz = MrzResult("TD3", "passport", "CAN", "TESTER JANE", date(2031, 6, 15), 1.0)
    # The LLM's issue date gives an odd 7-year passport, but it is LLM-only (unconfirmed).
    reply = llm(doc_kind="passport", issue_date=ISSUE, expiry_date=date(2031, 6, 15))

    result = run(mrz=mrz, dates=[], kind=None, llm_result=reply)

    assert result.confidence == 1.0
    assert not any("validity period" in r for r in result.reasons)


def test_trusted_issue_date_does_flag_an_odd_validity_period():
    mrz = MrzResult("TD3", "passport", "CAN", "TESTER JANE", date(2031, 6, 15), 1.0)

    result = run(mrz=mrz, dates=[labelled("issue", ISSUE)], kind=None)

    assert result.needs_verify
    assert any("validity period" in r for r in result.reasons)
