"""MRZ check digits, against values from ICAO Doc 9303's fictional specimens."""

import pytest

from lifeadmin.extract.mrz import check_digit


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
