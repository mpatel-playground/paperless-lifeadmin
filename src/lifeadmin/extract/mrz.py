"""Passport and ID card MRZ parsing (ICAO Doc 9303).

``parse_mrz`` finds the machine-readable lines of a passport (TD3: two lines of 44) or an
ID card (TD1: three lines of 30) in OCR text, undoes common OCR swaps, and accepts the
result only if every check digit passes.

``evals/generate.py`` imports ``check_digit`` to build the MRZ lines of the synthetic
samples, so the eval set exercises this code too.
"""

import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

# ICAO 9303 part 3, section 4.9: digits keep their value, A-Z count 10-35, filler is 0.
_VALUES = {
    **{d: int(d) for d in string.digits},
    **{letter: 10 + i for i, letter in enumerate(string.ascii_uppercase)},
    "<": 0,
}
_WEIGHTS = (7, 3, 1)

TD3_WIDTH = 44
TD1_WIDTH = 30

# OCR swaps undone by field position. A field that may only hold digits (dates, check
# digits) gets _TO_DIGIT; one that may only hold letters (codes, nationality, sex, names)
# gets _TO_LETTER. Document numbers can hold both, so they are handled separately.
_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}
_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}

# Confidence penalties. Provisional: the Week 2 evals set the final values.
# A fix inside a field covered by a check digit was confirmed by that digit, so it costs
# little. Names, nationality, sex and codes have no check digit, so a fix there is a guess.
COVERED_FIX_PENALTY = 0.02
UNCOVERED_FIX_PENALTY = 0.1
# More than one document number swap passed its check digit, so we don't know which
# reading is right. The number is never returned, so this only costs confidence.
AMBIGUOUS_NUMBER_PENALTY = 0.05
# A TD1 we can't map to a doc_kind stays below the 0.8 verify cut-off.
UNKNOWN_KIND_CAP = 0.7

_MRZ_RUN = re.compile(r"[A-Z0-9<]+")


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


@dataclass(frozen=True)
class MrzResult:
    """What the add-on keeps from a valid MRZ.

    The document number, birth date and personal number are read only to check the check
    digits and are deliberately not kept here.
    """

    format: Literal["TD3", "TD1"]
    doc_kind: Literal["passport", "pr_card"] | None
    issuing_state: str
    # As the MRZ spells it: upper case, no accents, e.g. "PRIYA RAMAN COTE".
    # repr=False keeps the name out of logs and tracebacks.
    holder: str = field(repr=False)
    expiry_date: date
    confidence: float


def parse_mrz(text: str) -> MrzResult | None:
    """Find and validate a TD3 or TD1 MRZ in OCR ``text``.

    Returns ``None`` if there is no MRZ or any check digit fails, so the caller falls back
    to regex and the LLM.
    """
    runs = _candidate_runs(text)
    for i in range(len(runs) - 1):
        if _is_td3_start(runs[i]) and len(runs[i + 1]) == TD3_WIDTH:
            result = _parse_td3(runs[i], runs[i + 1])
            if result:
                return result
    for i in range(len(runs) - 2):
        if _is_td1_start(runs[i]) and all(len(r) == TD1_WIDTH for r in runs[i + 1 : i + 3]):
            result = _parse_td1(runs[i], runs[i + 1], runs[i + 2])
            if result:
                return result
    return None


# --- Finding the lines ---


def _candidate_runs(text: str) -> list[str]:
    """The longest MRZ-alphabet run of each non-blank line, in order.

    OCR often splits MRZ lines with spaces and sometimes adds a stray character (``§``) at
    the end; removing whitespace and keeping the longest run handles both. Lines with no
    MRZ characters give ``""``, so they break the adjacency of the lines around them.
    """
    runs = []
    for raw in text.splitlines():
        line = "".join(raw.upper().split())
        if line:
            runs.append(max(_MRZ_RUN.findall(line), key=len, default=""))
    return runs


def _is_td3_start(run: str) -> bool:
    return len(run) == TD3_WIDTH and run[0] == "P"


def _is_td1_start(run: str) -> bool:
    # The first character is a letter field, so read "1" as "I" before deciding.
    return len(run) == TD1_WIDTH and _TO_LETTER.get(run[0], run[0]) in "IAC"


# --- Field correction ---


class _Fixes:
    """Applies the position-based OCR fixes and counts them for the confidence score."""

    def __init__(self) -> None:
        self.covered = 0
        self.uncovered = 0
        self.ambiguous = False

    def digits(self, value: str) -> str:
        # Every digit-only field is a date or a check digit, so a check digit covers it.
        fixed = _translate(value, _TO_DIGIT)
        self.covered += _count_changes(value, fixed)
        return fixed

    def letters(self, value: str) -> str:
        # No letter-only field is covered by a check digit.
        fixed = _translate(value, _TO_LETTER)
        self.uncovered += _count_changes(value, fixed)
        return fixed

    def number(self, number: str, check: str, composite_ok: Callable[[str], bool]) -> bool:
        """Whether the document number, or one OCR swap of it, passes its checks.

        A number can hold letters and digits, so position doesn't tell us whether ``8`` is
        an ``8`` or a ``B``. If the number as read fails, try swapping one character at a
        time. Several swaps can pass (the check digit is mod 10); since the number is never
        returned, any of them proves the rest of the MRZ is consistent, and the tie only
        costs confidence.

        Note: ``G`` (16) and ``6`` have the same value mod 10, so no check digit can tell
        them apart. Harmless while the number isn't used.
        """
        if _check_ok(number, check) and composite_ok(number):
            return True
        passing = 0
        for i, ch in enumerate(number):
            alt = _TO_DIGIT.get(ch) or _TO_LETTER.get(ch)
            if alt:
                candidate = number[:i] + alt + number[i + 1 :]
                passing += _check_ok(candidate, check) and composite_ok(candidate)
        if passing == 0:
            return False
        self.covered += 1
        self.ambiguous = passing > 1
        return True

    def confidence(self) -> float:
        penalty = self.covered * COVERED_FIX_PENALTY + self.uncovered * UNCOVERED_FIX_PENALTY
        if self.ambiguous:
            penalty += AMBIGUOUS_NUMBER_PENALTY
        return round(max(0.0, 1.0 - penalty), 2)


def _translate(value: str, table: dict[str, str]) -> str:
    return "".join(table.get(ch, ch) for ch in value)


def _count_changes(before: str, after: str) -> int:
    return sum(a != b for a, b in zip(before, after, strict=True))


def _check_ok(data: str, check: str) -> bool:
    return check.isdigit() and check_digit(data) == int(check)


# --- Field values ---


def _parse_date(yymmdd: str, *, birth: bool) -> date | None:
    """An MRZ ``YYMMDD`` date, or ``None`` if it isn't a real calendar date.

    Expiry dates are taken as 20YY. Birth dates are 20YY unless that is in the future,
    then 19YY.
    """
    if not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    year = 2000 + yy
    if birth and year > date.today().year:
        year -= 100
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def _holder(name_field: str) -> str:
    """``SURNAME<<GIVEN<NAMES<<<`` -> ``GIVEN NAMES SURNAME``."""
    surname, _, given = name_field.partition("<<")
    parts = [" ".join(given.replace("<", " ").split()), " ".join(surname.replace("<", " ").split())]
    return " ".join(p for p in parts if p)


def _is_code(value: str) -> bool:
    """Issuing state and nationality: three letters, or letters padded with ``<``."""
    return all(ch in string.ascii_uppercase or ch == "<" for ch in value)


def _doc_kind(fmt: str, issuing_state: str) -> Literal["passport", "pr_card"] | None:
    # Provisional: the PR card's MRZ document code is unconfirmed, so a TD1 is matched on
    # the issuing state instead. Revisit once checked against a real card.
    if fmt == "TD3":
        return "passport"
    return "pr_card" if issuing_state == "CAN" else None


def _result(
    fmt: Literal["TD3", "TD1"],
    *,
    issuing_state: str,
    nationality: str,
    sex: str,
    name_field: str,
    birth: str,
    expiry: str,
    fixes: _Fixes,
) -> MrzResult | None:
    """Plausibility checks shared by both formats, then the result."""
    expiry_date = _parse_date(expiry, birth=False)
    if (
        expiry_date is None
        or _parse_date(birth, birth=True) is None
        or sex not in "MFX<"
        or not _is_code(issuing_state)
        or not _is_code(nationality)
    ):
        return None
    holder = _holder(name_field)
    if not holder:
        return None
    doc_kind = _doc_kind(fmt, issuing_state)
    confidence = fixes.confidence()
    if doc_kind is None:
        confidence = min(confidence, UNKNOWN_KIND_CAP)
    return MrzResult(
        format=fmt,
        doc_kind=doc_kind,
        issuing_state=issuing_state,
        holder=holder,
        expiry_date=expiry_date,
        confidence=confidence,
    )


# --- Layouts (ICAO 9303 part 4 for TD3, part 5 for TD1) ---


def _parse_td3(line1: str, line2: str) -> MrzResult | None:
    fix = _Fixes()
    issuing_state = fix.letters(line1[2:5])
    name_field = fix.letters(line1[5:44])

    number, number_check = line2[0:9], fix.digits(line2[9])
    nationality = fix.letters(line2[10:13])
    birth, birth_check = fix.digits(line2[13:19]), fix.digits(line2[19])
    sex = fix.letters(line2[20])
    expiry, expiry_check = fix.digits(line2[21:27]), fix.digits(line2[27])
    personal, personal_check = line2[28:42], fix.digits(line2[42])
    composite = fix.digits(line2[43])

    if not (_check_ok(birth, birth_check) and _check_ok(expiry, expiry_check)):
        return None
    if set(personal) == {"<"}:
        # An unused personal number may take a filler or 0 as its check digit.
        personal_ok = personal_check in ("<", "0")
    else:
        personal_ok = _check_ok(personal, personal_check)
    if not personal_ok:
        return None

    def composite_ok(candidate: str) -> bool:
        data = candidate + number_check + birth + birth_check + expiry + expiry_check
        return _check_ok(data + personal + personal_check, composite)

    if not fix.number(number, number_check, composite_ok):
        return None

    return _result(
        "TD3",
        issuing_state=issuing_state,
        nationality=nationality,
        sex=sex,
        name_field=name_field,
        birth=birth,
        expiry=expiry,
        fixes=fix,
    )


def _parse_td1(line1: str, line2: str, line3: str) -> MrzResult | None:
    fix = _Fixes()
    fix.letters(line1[0:2])  # document code: counted if fixed, not kept
    issuing_state = fix.letters(line1[2:5])
    number, number_check = line1[5:14], fix.digits(line1[14])
    optional1 = line1[15:30]

    birth, birth_check = fix.digits(line2[0:6]), fix.digits(line2[6])
    sex = fix.letters(line2[7])
    expiry, expiry_check = fix.digits(line2[8:14]), fix.digits(line2[14])
    nationality = fix.letters(line2[15:18])
    optional2 = line2[18:29]
    composite = fix.digits(line2[29])

    name_field = fix.letters(line3)

    if not (_check_ok(birth, birth_check) and _check_ok(expiry, expiry_check)):
        return None

    def composite_ok(candidate: str) -> bool:
        # Line 1 from the document number on, then line 2's dates and optional data.
        data = candidate + number_check + optional1 + birth + birth_check
        return _check_ok(data + expiry + expiry_check + optional2, composite)

    # A "<" check digit marks the long-number extension, which v1 doesn't support; it
    # fails the check here and the document falls back to regex.
    if not fix.number(number, number_check, composite_ok):
        return None

    return _result(
        "TD1",
        issuing_state=issuing_state,
        nationality=nationality,
        sex=sex,
        name_field=name_field,
        birth=birth,
        expiry=expiry,
        fixes=fix,
    )
