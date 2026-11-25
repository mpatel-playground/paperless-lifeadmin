"""Damage OCR text the way Paperless's OCR has been seen to, for the noised eval cases.

Paperless's OCR is nearly perfect on our clean rendered samples, so on its own the eval
would say little about real scans. ``add_noise`` adds the errors seen on real output:

- digits and look-alike letters swapped (``0``/``O``, ``1``/``l``, ``5``/``S``, ``8``/``B``)
- accents dropped (``CÔTÉ`` -> ``COTE``)
- a stray ``§`` at the end of a line
- ``EI`` read as ``El``

Noise is seeded, so a case's text is the same on every run and results can be compared.
"""

import random
import unicodedata

# Probability that each digit becomes its look-alike letter, and the reverse.
DIGIT_SWAP_RATE = 0.12
LETTER_SWAP_RATE = 0.04
ACCENT_DROP_RATE = 0.5  # per accented character
STRAY_MARK_RATE = 0.15  # per non-empty line

_DIGIT_TO_LETTER = {"0": "O", "1": "l", "5": "S", "8": "B"}
_LETTER_TO_DIGIT = {"O": "0", "l": "1", "I": "1", "S": "5", "B": "8"}


def add_noise(text: str, seed: int) -> str:
    rng = random.Random(seed)
    text = text.replace("EI", "El")
    out = []
    for ch in text:
        if ch in _DIGIT_TO_LETTER and rng.random() < DIGIT_SWAP_RATE:
            ch = _DIGIT_TO_LETTER[ch]
        elif ch in _LETTER_TO_DIGIT and rng.random() < LETTER_SWAP_RATE:
            ch = _LETTER_TO_DIGIT[ch]
        elif not ch.isascii() and rng.random() < ACCENT_DROP_RATE:
            ch = _strip_accent(ch)
        out.append(ch)
    lines = "".join(out).split("\n")
    return "\n".join(
        line + "§" if line.strip() and rng.random() < STRAY_MARK_RATE else line for line in lines
    )


def _strip_accent(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
