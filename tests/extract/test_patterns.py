"""Labelled date and document-kind patterns. All names, numbers and dates are synthetic."""

from datetime import date

import pytest

from evals.generate import document_lines, load_samples
from lifeadmin.extract.patterns import (
    DateMatch,
    FoundDate,
    KindMatch,
    find_all_dates,
    find_dates,
    find_kind,
)


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


# --- Document kind: card numbers ---


@pytest.mark.parametrize(
    "number",
    [
        "B4172-39058-05314",
        "O1638-27450-51102",  # the first character is a letter: surname initial O
        "B4172 39058 05314",  # OCR split the groups with spaces
    ],
)
def test_drivers_licence_number(number):
    assert find_kind(f"4d DL/NO. {number}") == KindMatch(
        "drivers_licence", by_number=True, by_title=False, ocr_corrected=False
    )


@pytest.mark.parametrize("number", ["5820-417-396-KT", "7314-062-958", "7314 062 958"])
def test_health_card_number(number):
    assert find_kind(number) == KindMatch(
        "health_card", by_number=True, by_title=False, ocr_corrected=False
    )


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("B4l72-39O58-O5314", "drivers_licence"),
        ("84172-39058-05314", "drivers_licence"),  # surname initial B read as 8
        ("582O-4l7-396-KT", "health_card"),
    ],
)
def test_card_numbers_with_ocr_swaps_are_flagged(text, kind):
    match = find_kind(text)
    assert match.doc_kind == kind
    assert match.ocr_corrected


def test_one_clean_number_is_enough():
    match = find_kind("5820-417-396-KT\n582O-4l7-396-KT")
    assert match.doc_kind == "health_card"
    assert not match.ocr_corrected


@pytest.mark.parametrize(
    "text",
    [
        "1234567890",  # bare digit runs: account numbers look the same
        "B417239058053140",
        "15820-417-396",  # inside a longer number
        "5820-417-3961",
        "XB4172-39058-05314",
        "416-555-1234",  # a phone number
        "2026-09-30 to 2026-10-13",
        "SOIL-BOZO-ZIS",  # only digits after OCR swaps
        "BSOIZ-OOIZS-SSOBI",
    ],
)
def test_other_numbers_are_not_card_numbers(text):
    assert find_kind(text) is None


# --- Document kind: titles ---


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("CANADA PASSPORT / PASSEPORT CANADA", "passport"),
        ("PASSEPORT CANADA / CANADA PASSPORT", "passport"),
        ("PASSPORT PASSEPORT", "passport"),
        ("PERMANENT RESIDENT CARD", "pr_card"),
        ("CARTE DE RÉSIDENT PERMANENT", "pr_card"),
        ("CARTE DE RESIDENT PERMANENT", "pr_card"),
        ("ONTARIO  DRIVER'S LICENCE", "drivers_licence"),
        ("Driver’s License", "drivers_licence"),
        ("PERMIS DE CONDUIRE", "drivers_licence"),
        ("Ontario  Health / Santé", "health_card"),
        ("Santé / Health", "health_card"),
        ("ONTARIO SANTE", "health_card"),
        ("Northwind Maple Ltd. - Earnings Statement", "payslip"),
        ("Pay stub", "payslip"),
        ("Payslip", "payslip"),
        ("Talon de paie", "payslip"),
        ("T4 Statement of Remuneration Paid", "t4"),
        ("T4 État de la rémunération payée", "t4"),
        ("Canada Revenue Agency - Notice of Assessment", "noa"),
        ("Avis de cotisation", "noa"),
    ],
)
def test_titles(title, kind):
    assert find_kind(title) == KindMatch(kind, by_number=False, by_title=True, ocr_corrected=False)


@pytest.mark.parametrize(
    "text",
    ["passport photos required", "Health benefits", "Statement", "Assessment", "permis"],
)
def test_single_words_are_not_titles(text):
    assert find_kind(text) is None


def test_title_word_does_not_override_a_card_number():
    match = find_kind("Bring passport photos\n4d DL/NO. B4172-39058-05314")
    assert match.doc_kind == "drivers_licence"


def test_title_and_number_agree():
    match = find_kind("Ontario  Health / Santé\n5820-417-396-KT")
    assert match == KindMatch("health_card", by_number=True, by_title=True, ocr_corrected=False)


@pytest.mark.parametrize(
    "text",
    [
        "Ontario  Health / Santé\nB4172-39058-05314",  # title and number disagree
        "Earnings Statement\nNotice of Assessment",  # two titles
        "B4172-39058-05314\n5820-417-396-KT",  # two number formats
    ],
)
def test_conflicting_signals_give_no_kind(text):
    assert find_kind(text) is None


def test_no_kind_in_empty_text():
    assert find_kind("") is None


@pytest.mark.parametrize("sample", load_samples(), ids=lambda s: s.id)
def test_eval_sample_kinds(sample):
    text = "\n".join(line for line, _ in document_lines(sample))
    match = find_kind(text)
    assert match is not None
    assert match.doc_kind == sample.kind
    assert not match.ocr_corrected


# --- find_all_dates ---


def test_find_all_dates_needs_no_label():
    text = "DOB 1990/03/14\nsome line 15 JUN 2031\nPrinted 2026-01-02"

    assert [d.value for d in find_all_dates(text)] == [
        date(1990, 3, 14),
        date(2031, 6, 15),
        date(2026, 1, 2),
    ]


def test_find_all_dates_flags_ocr_fixes_and_ambiguity():
    assert find_all_dates("2O31-06-15") == [FoundDate(date(2031, 6, 15), True, False)]
    assert find_all_dates("03/04/2029") == [FoundDate(date(2029, 4, 3), False, True)]


def test_find_all_dates_ignores_accents_and_invalid_dates():
    assert [d.value for d in find_all_dates("15 FÉV 2031 and 31/02/2029")] == [date(2031, 2, 15)]
