"""The synthetic sample generator: answer key, MRZ building and printed text.

The MRZ tests compare against the specimens published in ICAO Doc 9303 (parts 4 and 5),
so they check the generator and ``lifeadmin.extract.mrz.check_digit`` together against
an independent source. They fail until ``check_digit`` is implemented.
"""

import re
from datetime import date

import pytest

from evals.generate import (
    KINDS,
    MONTHS,
    NO_EXPIRY_KINDS,
    SPECIMEN_FOOTER,
    Sample,
    document_lines,
    generate,
    load_samples,
    mrz_name,
    mrz_text,
    td1_lines,
    td3_lines,
)

SAMPLES = load_samples()
MRZ_KINDS = {"passport": 44, "pr_card": 30}

# ICAO's fictional specimen holder, from Doc 9303.
ICAO = {
    "surname": "Eriksson",
    "given_names": "Anna Maria",
    "nationality": "UTO",
    "issuing_state": "UTO",
    "birth_date": date(1974, 8, 12),
    "sex": "F",
    "expiry_date": date(2012, 4, 15),
}


# --- Answer key ---


def test_answer_key_size_and_ids():
    ids = [s.id for s in SAMPLES]
    assert 15 <= len(ids) <= 20
    assert len(set(ids)) == len(ids), "sample ids must be unique"


def test_answer_key_covers_every_kind():
    assert {s.kind for s in SAMPLES} == set(KINDS)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.id)
def test_dates_are_consistent(sample: Sample):
    if sample.kind in NO_EXPIRY_KINDS:
        assert sample.expiry_date is None
    if sample.issue_date and sample.expiry_date:
        assert sample.expiry_date > sample.issue_date


def test_load_rejects_unknown_kind(tmp_path):
    path = tmp_path / "labels.toml"
    path.write_text('[[sample]]\nid = "x"\nkind = "library_card"\nholder = "A B"\n')
    with pytest.raises(ValueError, match="unknown kind"):
        load_samples(path)


def test_load_rejects_unknown_field(tmp_path):
    path = tmp_path / "labels.toml"
    path.write_text('[[sample]]\nid = "x"\nkind = "t4"\nholder = "A B"\nexpiry = 2030-01-01\n')
    with pytest.raises(TypeError):
        load_samples(path)


# --- MRZ ---


def test_mrz_text_transliterates_names():
    assert mrz_text("Raman-Côté") == "RAMAN<COTE"
    assert mrz_text("Élise") == "ELISE"
    assert mrz_text("O'Brien") == "OBRIEN"


def test_mrz_name_pads_and_truncates():
    assert mrz_name("Okafor", "Daniel", 20) == "OKAFOR<<DANIEL<<<<<<"
    assert mrz_name("Okafor", "Daniel", 10) == "OKAFOR<<DA"


def test_td3_matches_icao_specimen():
    lines = td3_lines(doc_number="L898902C3", personal_number="ZE184226B", **ICAO)
    assert lines == (
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    )


def test_td1_matches_icao_specimen():
    lines = td1_lines(doc_code="I<", doc_number="D23145890", **ICAO)
    assert lines == (
        "I<UTOD231458907<<<<<<<<<<<<<<<",
        "7408122F1204159UTO<<<<<<<<<<<6",
        "ERIKSSON<<ANNA<MARIA<<<<<<<<<<",
    )


@pytest.mark.parametrize("sample", [s for s in SAMPLES if s.kind in MRZ_KINDS], ids=lambda s: s.id)
def test_sample_mrz_shape_and_dates(sample: Sample):
    mrz = [text for text, style in document_lines(sample) if style == "mrz"]
    width = MRZ_KINDS[sample.kind]

    assert len(mrz) == (2 if sample.kind == "passport" else 3)
    for line in mrz:
        assert re.fullmatch(rf"[A-Z0-9<]{{{width}}}", line), line
    # The expiry in the MRZ is the labelled one (YYMMDD), so MRZ extraction can score it.
    assert sample.expiry_date.strftime("%y%m%d") in "".join(mrz)


# --- Printed text ---


def printed_forms(d: date) -> set[str]:
    """Every way the layouts may print a date, written out independently of the generator."""
    en, fr = MONTHS[d.month - 1]
    month = d.strftime("%B")
    return {
        d.strftime("%Y/%m/%d"),
        d.strftime("%d/%m/%Y"),
        f"{d.day:02d} {en} /{fr} {d.year}",
        f"{month} {d.day}, {d.year}",
        f"{month[:3]} {d.day}, {d.year}",
    }


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.id)
def test_labelled_dates_are_printed(sample: Sample):
    text = "\n".join(line for line, _ in document_lines(sample))
    for d in (sample.issue_date, sample.expiry_date):
        if d is not None:
            assert any(form in text for form in printed_forms(d)), d


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.id)
def test_every_document_is_marked_specimen(sample: Sample):
    assert document_lines(sample)[-1] == (SPECIMEN_FOOTER, "body")


def test_missing_required_field_names_it():
    with pytest.raises(ValueError, match="expiry_date"):
        document_lines(
            Sample(
                id="x",
                kind="drivers_licence",
                holder="A B",
                number="1",
                birth_date=date(1990, 1, 1),
                sex="F",
                issue_date=date(2020, 1, 1),
            )
        )


# --- Rendering ---


def test_generate_writes_one_png_per_sample(tmp_path):
    paths = generate(SAMPLES, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{s.id}.png" for s in SAMPLES)
    assert all(p.read_bytes().startswith(b"\x89PNG") for p in paths)
