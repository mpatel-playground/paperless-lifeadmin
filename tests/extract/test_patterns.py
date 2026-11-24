"""Labelled date patterns. All names, numbers and dates are synthetic."""

from datetime import date

import pytest

from evals.generate import document_lines, load_samples
from lifeadmin.extract.patterns import DateMatch, find_dates


def only(text: str) -> DateMatch:
    matches = find_dates(text)
    assert len(matches) == 1, matches
    return matches[0]


# --- Labels ---


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("Date of expiry: 2031/06/15", "expiry"),
        ("Date d'expiration: 2031/06/15", "expiry"),
        ("Date d’expiration: 2031/06/15", "expiry"),  # typographic apostrophe
        ("Expiry: 2031/06/15", "expiry"),
        ("Expiration: 2031/06/15", "expiry"),
        ("4b EXP 2031/06/15", "expiry"),
        ("Exp. 2031/06/15", "expiry"),
        ("Valid until: 2031/06/15", "expiry"),
        ("Valide jusqu'au: 2031/06/15", "expiry"),
        ("Date of issue: 2031/06/15", "issue"),
        ("Date de délivrance: 2031/06/15", "issue"),
        ("Issued: 2031/06/15", "issue"),
        ("Délivrée: 2031/06/15", "issue"),
        ("4a ISS 2031/06/15", "issue"),
        ("Date issued: 2031/06/15", "issue"),
        ("Issue date: 2031/06/15", "issue"),
        ("Pay date: 2031/06/15", "issue"),
    ],
)
def test_labels(text, field):
    match = only(text)
    assert match.field == field
    assert match.value == date(2031, 6, 15)


@pytest.mark.parametrize(
    "text",
    [
        "DATE OF EXPIRY: 2031/06/15",
        "date of expiry: 2031/06/15",
        "Date d'expiration: 2031/06/15",
        "DATE D'EXPIRATION: 2031/06/15",
        "Valide jusqu’au: 2031/06/15",
    ],
)
def test_labels_ignore_case(text):
    assert only(text).field == "expiry"


@pytest.mark.parametrize(
    "text", ["Delivree: 2031/06/15", "DÉLIVRÉE: 2031/06/15", "Delivre: 2031/06/15"]
)
def test_labels_ignore_accents(text):
    # Paperless drops accents inconsistently (CÔTÉ came back as COTE).
    assert only(text).field == "issue"


def test_label_text_is_returned_without_accents():
    assert only("Délivrée: 2031/06/15").label == "Delivree"


@pytest.mark.parametrize(
    "text",
    [
        "Date of birth: 1990/03/14",
        "Date de naissance: 14 MAR /MARS 1990",
        "3 DOB/DDN 1990/03/14    15 SEX/SEXE F",
        "Born / Né(e) le: 14/03/1990",
        "Pay period: 2026-09-30 to 2026-10-13",
        "2031/06/15",
        "EXPENSES 2031/06/15",
        "Tax year: 2025",
    ],
)
def test_unlabelled_dates_are_ignored(text):
    assert find_dates(text) == []


# --- Date formats ---


@pytest.mark.parametrize(
    "printed",
    [
        "2031/06/15",
        "2031-06-15",
        "2031.06.15",
        "15/06/2031",
        "15-06-2031",
        "15 JUN /JUIN 2031",
        "15 JUN 2031",
        "15 JUIN 2031",
        "15 juin 2031",
        "15 June 2031",
        "June 15, 2031",
        "Jun 15, 2031",
        "Jun. 15, 2031",
        "15-JUN-2031",
    ],
)
def test_date_formats(printed):
    match = only(f"Expiry: {printed}")
    assert match.value == date(2031, 6, 15)
    assert not match.ocr_corrected
    assert not match.ambiguous


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("05 FEB /FÉV 2027", date(2027, 2, 5)),
        ("05 FÉV 2027", date(2027, 2, 5)),
        ("05 FEV 2027", date(2027, 2, 5)),
        ("05 AOÛT 2027", date(2027, 8, 5)),
        ("05 DÉC 2027", date(2027, 12, 5)),
        ("05 mai 2027", date(2027, 5, 5)),
        ("05 SEPT 2027", date(2027, 9, 5)),
        ("05 juillet 2027", date(2027, 7, 5)),
        ("April 28, 2026", date(2026, 4, 28)),
    ],
)
def test_month_names(printed, expected):
    assert only(f"Expiry: {printed}").value == expected


@pytest.mark.parametrize(
    "printed",
    ["31/02/2029", "2029/02/30", "2029/13/01", "00/05/2029", "31 APR 2029", "15/06/0031"],
)
def test_impossible_dates_are_rejected(printed):
    assert find_dates(f"Expiry: {printed}") == []


def test_ambiguous_day_first_date_is_flagged():
    match = only("Valid until: 03/04/2029")
    assert match.value == date(2029, 4, 3)
    assert match.ambiguous


@pytest.mark.parametrize("printed", ["15/06/2031", "04/04/2029", "2029/03/04"])
def test_unambiguous_dates_are_not_flagged(printed):
    assert not only(f"Valid until: {printed}").ambiguous


def test_month_first_is_used_when_day_first_is_impossible():
    match = only("Valid until: 06/15/2031")
    assert match.value == date(2031, 6, 15)
    assert not match.ambiguous


def test_dates_inside_longer_numbers_are_ignored():
    assert find_dates("Expiry: 12031/06/15") == []
    assert find_dates("Expiry: 2031/06/155") == []


# --- OCR noise ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4b EXP 2O29/O3/l4", date(2029, 3, 14)),  # O for 0, l for 1
        ("Expiry: 2O28/04/1l", date(2028, 4, 11)),
        ("Valid until: 29/O7/2O28", date(2028, 7, 29)),
        ("Date of expiry: l5 JUN /JUIN 2O31", date(2031, 6, 15)),
        ("Expiry: 2031/06/I5", date(2031, 6, 15)),  # I for 1
        ("Expiry: 2031/06/1S", date(2031, 6, 15)),  # S for 5
        ("Expiry: 2O3l-O6-l5", date(2031, 6, 15)),
    ],
)
def test_ocr_swaps_are_corrected_and_flagged(text, expected):
    match = only(text)
    assert match.value == expected
    assert match.ocr_corrected


def test_stray_symbol_after_date():
    match = only("Expiry: 2028/04/11§")
    assert match.value == date(2028, 4, 11)


def test_ocr_swaps_do_not_turn_words_into_dates():
    assert find_dates("Expiry: DOB/SIS/BOZO") == []


# --- Layout ---


def test_bilingual_labels_share_one_date():
    matches = find_dates("Expiry / Expiration: 2028/04/11")
    assert [(m.field, m.value, m.label) for m in matches] == [
        ("expiry", date(2028, 4, 11), "Expiry"),
    ]


def test_french_label_first():
    match = only("Date d'expiration / Date of expiry: 05 SEP /SEPT 2026")
    assert match.label == "Date d'expiration"
    assert match.value == date(2026, 9, 5)


def test_issue_and_expiry_on_one_line():
    matches = find_dates("ISS 2024/03/14   EXP 2029/03/14")
    assert [(m.field, m.value) for m in matches] == [
        ("issue", date(2024, 3, 14)),
        ("expiry", date(2029, 3, 14)),
    ]


def test_label_does_not_take_a_date_before_it():
    assert find_dates("2029/03/14 EXP") == []


def test_date_on_the_next_line():
    match = only("Valid until:\n\n29/07/2028\n")
    assert match.value == date(2028, 7, 29)


def test_next_line_with_its_own_label_is_not_borrowed():
    matches = find_dates("Valid until:\nIssued: 29/07/2023")
    assert [(m.field, m.value) for m in matches] == [("issue", date(2023, 7, 29))]


def test_dates_far_below_a_label_are_ignored():
    assert find_dates("Valid until:\nOntario\n29/07/2028") == []


def test_every_labelled_date_is_returned_in_order():
    text = "Expiry: 2028/04/11\nsome text\nExpiry: 2029/04/11"
    assert [m.value for m in find_dates(text)] == [date(2028, 4, 11), date(2029, 4, 11)]


def test_empty_text():
    assert find_dates("") == []


# --- Synthetic eval samples ---


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s.id)
def test_eval_samples(sample):
    text = "\n".join(line for line, _ in document_lines(sample))
    matches = find_dates(text)
    assert [m.value for m in matches if m.field == "issue"] == (
        [sample.issue_date] if sample.issue_date else []
    )
    assert [m.value for m in matches if m.field == "expiry"] == (
        [sample.expiry_date] if sample.expiry_date else []
    )
    assert not any(m.ocr_corrected or m.ambiguous for m in matches)
