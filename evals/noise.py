"""Damage OCR text the way Paperless's OCR has been seen to, for the noised eval cases.

Paperless's OCR is nearly perfect on our clean rendered samples, so on its own the eval
would say little about real scans. ``add_noise`` adds the errors seen on real output:

- digits and look-alike letters swapped (``0``/``O``, ``1``/``l``, ``5``/``S``, ``8``/``B``)
- accents dropped (``CÔTÉ`` -> ``COTE``)
- a stray ``§`` at the end of a line
- ``EI`` read as ``El``
- MRZ filler ``<`` read as ``c`` or ``e``

The ``heavy`` level also misreads digits as other digits (``3``/``8``, ``1``/``7``,
``6``/``5``). That is the dangerous error: the result is a valid-looking wrong date, which
no OCR-swap check can spot, so only grounding, agreement and plausibility can catch it.

Noise is seeded, so a case's text is the same on every run and results can be compared.
"""

import random
import unicodedata
from dataclasses import dataclass
from typing import Literal

Level = Literal["light", "heavy"]


@dataclass(frozen=True)
class Rates:
    """Probability of each kind of damage, per eligible character (or line)."""

    digit_to_letter: float
    letter_to_digit: float
    digit_to_digit: float
    accent_drop: float
    mrz_filler: float
    stray_mark: float  # per non-empty line


LEVELS: dict[Level, Rates] = {
    "light": Rates(
        digit_to_letter=0.12,
        letter_to_digit=0.04,
        digit_to_digit=0.0,
        accent_drop=0.5,
        mrz_filler=0.03,
        stray_mark=0.15,
    ),
    "heavy": Rates(
        digit_to_letter=0.35,
        letter_to_digit=0.12,
        digit_to_digit=0.06,
        accent_drop=0.9,
        mrz_filler=0.15,
        stray_mark=0.3,
    ),
}

DIGIT_TO_LETTER = {"0": "O", "1": "l", "5": "S", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "l": "1", "I": "1", "S": "5", "B": "8"}
DIGIT_TO_DIGIT = {"3": "8", "8": "3", "1": "7", "7": "1", "6": "5", "5": "6"}
MRZ_FILLER = ("c", "e")


def add_noise(text: str, seed: int, level: Level = "light") -> str:
    rates = LEVELS[level]
    rng = random.Random(seed)
    text = text.replace("EI", "El")
    out = []
    for ch in text:
        if ch in DIGIT_TO_LETTER and rng.random() < rates.digit_to_letter:
            ch = DIGIT_TO_LETTER[ch]
        elif ch in DIGIT_TO_DIGIT and rng.random() < rates.digit_to_digit:
            ch = DIGIT_TO_DIGIT[ch]
        elif ch in LETTER_TO_DIGIT and rng.random() < rates.letter_to_digit:
            ch = LETTER_TO_DIGIT[ch]
        elif ch == "<" and rng.random() < rates.mrz_filler:
            ch = rng.choice(MRZ_FILLER)
        elif not ch.isascii() and rng.random() < rates.accent_drop:
            ch = _strip_accent(ch)
        out.append(ch)
    lines = "".join(out).split("\n")
    return "\n".join(
        line + "§" if line.strip() and rng.random() < rates.stray_mark else line for line in lines
    )


def _strip_accent(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
