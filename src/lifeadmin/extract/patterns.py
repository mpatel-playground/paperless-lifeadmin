"""Labelled issue and expiry dates in OCR text.

The second extraction step, for documents without a valid MRZ. ``find_dates`` returns
every date that follows an issue or expiry label ("Expiry", "Valid until",
"Date d'expiration", ...) and leaves choosing between them to the confidence scorer.

Labels are matched ignoring case and accents, because Paperless's OCR drops accents
inconsistently. Inside a date, common OCR swaps (``O`` for ``0``, ``l`` for ``1``, ...) are
accepted and flagged so the scorer can lower the confidence.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Literal

Field = Literal["issue", "expiry"]

# Labels as they read once accents are removed; matched ignoring case. Longer labels come
# first so "Date of expiry" wins over "Exp" at the same position. Birth-date labels are
# deliberately absent: a date after "DOB" or "Born" is never returned.
_LABELS: dict[Field, tuple[str, ...]] = {
    "expiry": (
        r"date d['’ ]?expiration",
        r"valide jusqu['’ ]?au",
        r"date of expiry",
        r"valid until",
        r"expiration",
        r"expiry",
        r"exp",
    ),
    "issue": (
        r"date de delivrance",
        r"date of issue",
        r"date issued",
        r"issue date",
        r"pay date",
        r"delivree?",
        r"issued",
        r"iss",
    ),
}
# One named group per field, e.g. (?P<expiry>...|...), so m.lastgroup gives the field.
# The look-arounds make labels whole words: "EXP" matches, "EXPENSES" doesn't.
_LABEL_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:"
    + "|".join(f"(?P<{field}>{'|'.join(labels)})" for field, labels in _LABELS.items())
    + r")(?![a-z])"
)

# A digit, or a character OCR commonly prints in its place. Deliberately case-sensitive:
# a lower-case "d" or "b" is far more likely a real letter than a misread digit.
_D = r"[0-9OoQDlIi|SBZ]"
_TO_DIGIT = str.maketrans("OoQDlIi|SBZ", "00001111582")

# English and French, full and abbreviated, accents removed (FÉV -> FEV, AOÛT -> AOUT).
_MONTHS = {
    "january": 1, "janvier": 1, "janv": 1, "jan": 1,
    "february": 2, "fevrier": 2, "fevr": 2, "fev": 2, "feb": 2,
    "march": 3, "mars": 3, "mar": 3,
    "april": 4, "avril": 4, "avr": 4, "apr": 4,
    "may": 5, "mai": 5,
    "june": 6, "juin": 6, "jun": 6,
    "july": 7, "juillet": 7, "juil": 7, "jul": 7,
    "august": 8, "aout": 8, "aug": 8,
    "september": 9, "septembre": 9, "sept": 9, "sep": 9,
    "october": 10, "octobre": 10, "oct": 10,
    "november": 11, "novembre": 11, "nov": 11,
    "december": 12, "decembre": 12, "dec": 12,
}  # fmt: skip
_MONTH = "(?i:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")(?![A-Za-z])"

# The outer look-arounds stop a date being cut out of a longer number or word.
_START = r"(?<![0-9A-Za-z])"
_END = r"(?![0-9A-Za-z])"
_SEP = r"[/.\-]"
_YMD = re.compile(
    rf"{_START}(?P<y>{_D}{{4}}){_SEP}(?P<m>{_D}{{1,2}}){_SEP}(?P<d>{_D}{{1,2}}){_END}"
)  # 2031/06/15, 2031-06-15
_DMY = re.compile(
    rf"{_START}(?P<d>{_D}{{1,2}}){_SEP}(?P<m>{_D}{{1,2}}){_SEP}(?P<y>{_D}{{4}}){_END}"
)  # 15/06/2031
_D_MON_Y = re.compile(
    rf"{_START}(?P<d>{_D}{{1,2}})[\s.\-]+(?P<mon>{_MONTH})\.?"
    rf"(?:\s*/\s*{_MONTH}\.?)?[\s,.\-]+(?P<y>{_D}{{4}}){_END}"
)  # 15 JUN /JUIN 2031, 15 mai 2031
_MON_D_Y = re.compile(
    rf"(?<![A-Za-z])(?P<mon>{_MONTH})\.?\s+(?P<d>{_D}{{1,2}}),?\s+(?P<y>{_D}{{4}}){_END}"
)  # June 15, 2031 and Jun 15, 2031
_DATE_RES = (_YMD, _DMY, _D_MON_Y, _MON_D_Y)

# Four-digit years outside this range are OCR garbage, not dates. Whether a date is
# plausible for the document is the confidence scorer's call, not this module's.
_MIN_YEAR, _MAX_YEAR = 1900, 2099


@dataclass(frozen=True)
class DateMatch:
    """A date that follows an issue or expiry label."""

    field: Field
    value: date
    label: str  # the label as printed, accents removed, e.g. "Valid until"
    ocr_corrected: bool  # only parsed after undoing OCR swaps such as O -> 0
    ambiguous: bool  # day and month could be swapped, e.g. 03/04/2029


@dataclass(frozen=True)
class _Date:
    start: int
    value: date
    ocr_corrected: bool
    ambiguous: bool


@dataclass(frozen=True)
class _Label:
    field: Field
    text: str
    start: int
    end: int  # the end of the last label merged into this one


def find_dates(text: str) -> list[DateMatch]:
    """Every labelled issue and expiry date in OCR ``text``, in document order.

    A label takes the first date after it on its line, stopping where the next label
    starts. If the last label on a line finds no date, the next non-blank line is tried,
    provided that line has no labels of its own: OCR often puts a value below its label.
    """
    lines = [line for line in _fold(text).splitlines() if line.strip()]
    dates = [_dates_in(line) for line in lines]
    labels = [_labels_in(line) for line in lines]
    used: set[tuple[int, int]] = set()  # (line index, date start): each date counts once
    results = []

    for i, line_labels in enumerate(labels):
        for j, label in enumerate(line_labels):
            is_last = j + 1 == len(line_labels)
            stop = len(lines[i]) if is_last else line_labels[j + 1].start
            found = _take(dates[i], i, label.end, stop, used)
            if found is None and is_last and i + 1 < len(lines) and not labels[i + 1]:
                found = _take(dates[i + 1], i + 1, 0, len(lines[i + 1]), used)
            if found is not None:
                results.append(
                    DateMatch(
                        field=label.field,
                        value=found.value,
                        label=label.text,
                        ocr_corrected=found.ocr_corrected,
                        ambiguous=found.ambiguous,
                    )
                )
    return results


def _take(
    dates: list[_Date], line: int, start: int, stop: int, used: set[tuple[int, int]]
) -> _Date | None:
    """The first unused date starting in ``[start, stop)``, marked as used."""
    for d in dates:
        if start <= d.start < stop and (line, d.start) not in used:
            used.add((line, d.start))
            return d
    return None


# --- Labels ---


def _fold(text: str) -> str:
    """Remove accents (``Délivrée`` -> ``Delivree``) without changing case."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _labels_in(line: str) -> list[_Label]:
    """The line's labels, left to right.

    Neighbouring labels for the same field with no date between them, as in the
    bilingual "Expiry / Expiration:", are merged so they share one date.
    """
    merged: list[_Label] = []
    for m in _LABEL_RE.finditer(line):
        field: Field = "expiry" if m.lastgroup == "expiry" else "issue"
        prev = merged[-1] if merged else None
        if prev and prev.field == field and not _dates_in(line[prev.end : m.start()]):
            merged[-1] = _Label(field, prev.text, prev.start, m.end())
        else:
            merged.append(_Label(field, m.group(), m.start(), m.end()))
    return merged


# --- Dates ---


def _dates_in(line: str) -> list[_Date]:
    """Every valid date in ``line``, left to right, without overlaps.

    Where two formats match overlapping text, the one starting first wins.
    """
    found = []
    for pattern in _DATE_RES:
        for m in pattern.finditer(line):
            parsed = _parse(m)
            if parsed:
                found.append((m.start(), m.end(), parsed))
    found.sort(key=lambda f: (f[0], -f[1]))
    dates: list[_Date] = []
    last_end = -1
    for start, end, (value, corrected, ambiguous) in found:
        if start >= last_end:
            dates.append(_Date(start, value, corrected, ambiguous))
            last_end = end
    return dates


def _parse(m: re.Match[str]) -> tuple[date, bool, bool] | None:
    """``(value, ocr_corrected, ambiguous)`` for a date match, or ``None`` if invalid."""
    raw = [m["y"], m["d"]] + ([] if "mon" in m.re.groupindex else [m["m"]])
    fixed = [part.translate(_TO_DIGIT) for part in raw]
    corrected = fixed != raw
    year, day = int(fixed[0]), int(fixed[1])
    if not _MIN_YEAR <= year <= _MAX_YEAR:
        return None
    if "mon" in m.re.groupindex:
        return _valid(year, _MONTHS[m["mon"].lower()], day, corrected, ambiguous=False)
    month = int(fixed[2])
    if m.re is _DMY and day != month and day <= 12 and month <= 12:
        # 03/04/2029 could be 3 April or March 4. Read day first, as Canadian cards do,
        # and let the scorer send it for a human check.
        return _valid(year, month, day, corrected, ambiguous=True)
    if m.re is _DMY and month > 12:
        # 06/15/2031 can only be month first.
        day, month = month, day
    return _valid(year, month, day, corrected, ambiguous=False)


def _valid(
    year: int, month: int, day: int, corrected: bool, *, ambiguous: bool
) -> tuple[date, bool, bool] | None:
    try:
        return date(year, month, day), corrected, ambiguous
    except ValueError:
        return None  # e.g. 31/02/2029
