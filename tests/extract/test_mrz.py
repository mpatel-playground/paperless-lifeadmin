"""MRZ check digits and parsing.

Specimen values are ICAO Doc 9303's fictional "Utopia" documents. Everything else comes
from the synthetic samples in ``evals/labels.toml``.
"""

from dataclasses import astuple
from datetime import date

import pytest

from evals.generate import document_lines, load_samples, mrz_text
from lifeadmin.extract.mrz import MrzResult, check_digit, parse_mrz

# --- check_digit ---


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("L898902C3", 6),  # TD3 specimen document number (letters and digits)
        ("740812", 2),  # date of birth
        ("120415", 9),  # date of expiry
        ("ZE184226B<<<<<", 1),  # personal number with fillers
        ("D23145890", 7),  # TD1 specimen document number
        # TD3 composite: document number, birth date, expiry and personal number, each
        # followed by its own check digit.
        ("L898902C36" + "7408122" + "1204159ZE184226B<<<<<1", 0),
    ],
)
def test_icao_specimen_values(field, expected):
    assert check_digit(field) == expected


def test_fillers_and_empty_count_as_zero():
    assert check_digit("<<<<<<<<<") == 0
    assert check_digit("") == 0


def test_weights_cycle_7_3_1():
    # A=10: 10*7 = 70 -> 0; "1"*3 = 7+3+1 = 11 -> 1; "11111" = 7+3+1+7+3 = 21 -> 1.
    assert check_digit("A") == 0
    assert check_digit("111") == 1
    assert check_digit("11111") == 1


@pytest.mark.parametrize("bad", ["l898902c3", "AB-12", "12 34", "É"])
def test_rejects_characters_outside_mrz_alphabet(bad):
    with pytest.raises(ValueError, match="Not an MRZ character"):
        check_digit(bad)


# --- parse_mrz: ICAO specimens ---

TD3_SPECIMEN = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
)
TD1_SPECIMEN = (
    "I<UTOD231458907<<<<<<<<<<<<<<<",
    "7408122F1204159UTO<<<<<<<<<<<6",
    "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
)


def test_td3_specimen():
    result = parse_mrz("\n".join(TD3_SPECIMEN))
    assert result == MrzResult(
        format="TD3",
        doc_kind="passport",
        issuing_state="UTO",
        holder="ANNA MARIA ERIKSSON",
        expiry_date=date(2012, 4, 15),
        confidence=1.0,
    )


def test_td1_specimen_from_unknown_issuer_goes_to_verify():
    result = parse_mrz("\n".join(TD1_SPECIMEN))
    assert result is not None
    assert result.format == "TD1"
    assert result.holder == "ANNA MARIA ERIKSSON"
    assert result.expiry_date == date(2012, 4, 15)
    # Only a CAN-issued TD1 maps to pr_card for now; anything else needs a human check.
    assert result.doc_kind is None
    assert result.confidence < 0.8


# --- parse_mrz: round trip of the synthetic samples ---

SAMPLES = load_samples()
MRZ_SAMPLES = [s for s in SAMPLES if s.kind in ("passport", "pr_card")]
OTHER_SAMPLES = [s for s in SAMPLES if s.kind not in ("passport", "pr_card")]


def _text(sample) -> str:
    return "\n".join(text for text, _ in document_lines(sample))


def _mrz(sample) -> list[str]:
    return [text for text, style in document_lines(sample) if style == "mrz"]


def _sample(sample_id: str):
    return next(s for s in SAMPLES if s.id == sample_id)


def _replace_mrz_line(sample, index: int, new_line: str) -> str:
    old = _mrz(sample)[index]
    text = _text(sample)
    assert text.count(old) == 1
    return text.replace(old, new_line)


def _swap_char(line: str, index: int, new: str) -> str:
    return line[:index] + new + line[index + 1 :]


@pytest.mark.parametrize("sample", MRZ_SAMPLES, ids=lambda s: s.id)
def test_round_trip_of_synthetic_samples(sample):
    result = parse_mrz(_text(sample))
    assert result is not None
    assert result.doc_kind == sample.kind
    assert result.expiry_date == sample.expiry_date
    assert result.holder == mrz_text(sample.holder).replace("<", " ")
    assert result.issuing_state == "CAN"
    assert result.confidence == 1.0


@pytest.mark.parametrize("sample", OTHER_SAMPLES, ids=lambda s: s.id)
def test_documents_without_mrz_return_none(sample):
    assert parse_mrz(_text(sample)) is None


@pytest.mark.parametrize("sample", MRZ_SAMPLES, ids=lambda s: s.id)
def test_result_never_holds_the_document_number(sample):
    result = parse_mrz(_text(sample))
    assert sample.number not in repr(result)
    assert all(sample.number not in str(value) for value in astuple(result))


def test_repr_hides_holder():
    result = parse_mrz("\n".join(TD3_SPECIMEN))
    assert "ERIKSSON" not in repr(result)


# --- parse_mrz: OCR noise seen from the dev Paperless (2026-11-22) ---


def test_nationality_zero_read_as_letter_o():
    # Paperless read the made-up nationality UTO as UT0 on a PR card.
    sample = _sample("pr-card-01")
    line2 = _mrz(sample)[1]
    assert line2[15:18] == "UTO"
    result = parse_mrz(_replace_mrz_line(sample, 1, line2.replace("UTO", "UT0")))
    assert result is not None
    assert result.expiry_date == sample.expiry_date
    # Nationality has no check digit, so the fix is a guess and costs confidence.
    assert result.confidence == pytest.approx(0.9)


def test_stray_symbol_after_mrz_line():
    sample = _sample("pr-card-02")
    line1 = _mrz(sample)[0]
    result = parse_mrz(_replace_mrz_line(sample, 0, line1 + "§"))
    assert result is not None
    assert result.expiry_date == sample.expiry_date
    assert result.confidence == 1.0


def test_dropped_accents_in_labels_do_not_matter():
    sample = _sample("passport-03")
    text = _text(sample).replace("Côté", "COTE").replace("é", "e")
    result = parse_mrz(text)
    assert result is not None
    assert result.holder == "PRIYA RAMAN COTE"


def test_payslip_with_el_for_ei_has_no_mrz():
    text = _text(_sample("payslip-01")).replace("EI premiums", "El premiums")
    assert parse_mrz(text) is None


# --- parse_mrz: other OCR layout noise ---


def test_spaces_inside_mrz_lines():
    sample = _sample("passport-01")
    text = _text(sample)
    for line in _mrz(sample):
        text = text.replace(line, f"{line[:20]} {line[20:30]}  {line[30:]}")
    result = parse_mrz(text)
    assert result is not None
    assert result.confidence == 1.0


def test_blank_line_between_mrz_lines():
    line1, line2 = TD3_SPECIMEN
    assert parse_mrz(f"{line1}\n\n{line2}") is not None


def test_lower_case_is_read_as_upper_case():
    line1, line2 = TD3_SPECIMEN
    result = parse_mrz(f"{line1.lower()}\n{line2.lower()}")
    assert result is not None
    # Upper-casing "l898902c3" gives back the specimen; the expiry is unaffected.
    assert result.expiry_date == date(2012, 4, 15)


# --- parse_mrz: corrections by field position ---


def test_letter_o_in_expiry_date_is_read_as_zero():
    sample = _sample("passport-01")  # expiry 310615
    line2 = _mrz(sample)[1]
    assert line2[21:27] == "310615"
    result = parse_mrz(_replace_mrz_line(sample, 1, _swap_char(line2, 23, "O")))
    assert result is not None
    assert result.expiry_date == sample.expiry_date
    # The expiry check digit confirms the fix, so it costs only a little.
    assert result.confidence == pytest.approx(0.98)


@pytest.mark.parametrize(
    ("sample_id", "line_index", "char_index", "ocr_char"),
    [
        ("passport-02", 1, 3, "O"),  # HB2|0|5577: zero read as letter O
        ("pr-card-01", 0, 7, "O"),  # PA|0|391827 on TD1 line 1
    ],
)
def test_one_swap_in_document_number_is_recovered(sample_id, line_index, char_index, ocr_char):
    sample = _sample(sample_id)
    line = _mrz(sample)[line_index]
    noisy = _replace_mrz_line(sample, line_index, _swap_char(line, char_index, ocr_char))
    result = parse_mrz(noisy)
    assert result is not None
    assert result.expiry_date == sample.expiry_date
    assert result.confidence == pytest.approx(0.98)


def test_ambiguous_document_number_swap_costs_confidence():
    # AK4|B|2913: swapping B->8, 2->Z or 1->I each satisfies the check digit, so we
    # can't tell which reading is right. The number isn't returned, so the MRZ is still
    # accepted, with a penalty for the tie.
    sample = _sample("passport-01")
    line2 = _mrz(sample)[1]
    result = parse_mrz(_replace_mrz_line(sample, 1, _swap_char(line2, 3, "B")))
    assert result is not None
    assert result.expiry_date == sample.expiry_date
    assert result.confidence == pytest.approx(0.93)


def test_document_number_beyond_one_swap_is_rejected():
    # Two wrong digits that no single OCR swap explains.
    line1, line2 = TD3_SPECIMEN
    assert parse_mrz(f"{line1}\n{_swap_char(_swap_char(line2, 3, '7'), 4, '7')}") is None


# --- parse_mrz: rejections ---


def test_wrong_expiry_digit_is_rejected():
    sample = _sample("passport-01")
    line2 = _mrz(sample)[1]
    wrong = "1" if line2[22] != "1" else "2"
    assert parse_mrz(_replace_mrz_line(sample, 1, _swap_char(line2, 22, wrong))) is None


def test_broken_composite_is_rejected_even_if_fields_pass():
    line1, line2 = TD3_SPECIMEN
    assert parse_mrz(f"{line1}\n{line2[:-1]}5") is None


def test_broken_td1_composite_is_rejected():
    line1, line2, line3 = TD1_SPECIMEN
    assert parse_mrz(f"{line1}\n{line2[:-1]}1\n{line3}") is None


def test_impossible_month_is_rejected_despite_valid_check_digits():
    line1, line2 = TD3_SPECIMEN
    expiry = "121315"  # month 13
    number, birth, personal = line2[0:10], line2[13:20], line2[28:43]
    expiry_field = expiry + str(check_digit(expiry))
    composite = check_digit(number + birth + expiry_field + personal)
    bad = f"{line2[:21]}{expiry_field}{personal}{composite}"
    assert len(bad) == 44
    assert parse_mrz(f"{line1}\n{bad}") is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        TD3_SPECIMEN[1],  # one line only
        f"{TD3_SPECIMEN[0]}\n{TD3_SPECIMEN[1][:40]}",  # truncated second line
        "\n".join(TD1_SPECIMEN[:2]),  # TD1 missing its name line
        f"{TD3_SPECIMEN[0]}\nNot an MRZ line\n{TD3_SPECIMEN[1]}",  # text in between
    ],
)
def test_incomplete_mrz_returns_none(text):
    assert parse_mrz(text) is None
