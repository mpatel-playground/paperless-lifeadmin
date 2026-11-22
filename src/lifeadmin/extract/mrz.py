"""Passport and ID card MRZ parsing (ICAO Doc 9303).

``evals/generate.py`` imports ``check_digit`` to build the MRZ lines of the synthetic
samples, so the eval set exercises this code too.
"""

import string

# ICAO 9303 part 3, section 4.9: digits keep their value, A-Z count 10-35, filler is 0.
_VALUES = {
    **{d: int(d) for d in string.digits},
    **{letter: 10 + i for i, letter in enumerate(string.ascii_uppercase)},
    "<": 0,
}
_WEIGHTS = (7, 3, 1)


def check_digit(data: str) -> int:
    """Return the ICAO 9303 check digit (0-9) for an MRZ field such as ``"L898902C3"``.

    Each character's value is multiplied by 7, 3, 1, 7, 3, 1, ... in turn; the check digit
    is the sum modulo 10.

    Accepts the MRZ character set only: ``0-9``, ``A-Z`` and the filler ``<``.
    Raises ``ValueError`` for anything else, including lower case, so OCR noise is caught
    here instead of producing a wrong digit.
    """
    total = 0
    for i, ch in enumerate(data):
        try:
            value = _VALUES[ch]
        except KeyError:
            raise ValueError(f"Not an MRZ character: {ch!r}") from None
        total += value * _WEIGHTS[i % 3]
    return total % 10
