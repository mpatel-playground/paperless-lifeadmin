"""OCR noise for the eval: seeded, and only the kinds of damage seen on real output."""

import pytest

from evals.noise import DIGIT_TO_DIGIT, DIGIT_TO_LETTER, LETTER_TO_DIGIT, MRZ_FILLER, add_noise

# Synthetic text with every character the noise can touch.
TEXT = "\n".join(
    [
        "PASSEPORT / PASSPORT  Nom: RAMAN-CÔTÉ  Prénoms: ÉLISE",
        "Date of expiry: 15 JUN 2031  Issued: 2021-06-15  No. 0158 1850 5581 3766",
        "P<CANBERGERON<<ELISE<<<<<<<<<<<<<<<<<<<<<<<<",
        "AK482913<0CAN9003148F3106150<<<<<<<<<<<<<<<0",
        "EIRE SOLDIER LOBBY",
    ]
    * 4
)
ACCENTS = {("Ô", "O"), ("É", "E"), ("é", "e")}
LIGHT_ALLOWED = (
    set(DIGIT_TO_LETTER.items())
    | set(LETTER_TO_DIGIT.items())
    | {("<", c) for c in MRZ_FILLER}
    | {("I", "l")}  # EI -> El
    | ACCENTS
)
HEAVY_ALLOWED = LIGHT_ALLOWED | set(DIGIT_TO_DIGIT.items())


def changes(clean: str, noisy: str) -> set[tuple[str, str]]:
    """Every (before, after) character substitution, ignoring stray marks at line ends."""
    clean_lines, noisy_lines = clean.split("\n"), noisy.split("\n")
    assert len(noisy_lines) == len(clean_lines)
    found = set()
    for before, after in zip(clean_lines, noisy_lines, strict=True):
        after = after.removesuffix("§")
        assert len(after) == len(before)
        found |= {(a, b) for a, b in zip(before, after, strict=True) if a != b}
    return found


@pytest.mark.parametrize("level", ["light", "heavy"])
def test_same_seed_gives_same_text(level):
    assert add_noise(TEXT, 7, level) == add_noise(TEXT, 7, level)


def test_different_seeds_give_different_text():
    assert add_noise(TEXT, 1) != add_noise(TEXT, 2)


def test_applies_each_kind_of_damage():
    noisy = add_noise(TEXT, 3, "heavy")

    assert "El" in noisy and "EI" not in noisy
    assert "§" in noisy
    found = changes(TEXT, noisy)
    assert found & set(DIGIT_TO_LETTER.items())
    assert found & set(LETTER_TO_DIGIT.items())
    assert found & set(DIGIT_TO_DIGIT.items())
    assert found & {("<", c) for c in MRZ_FILLER}
    assert found & ACCENTS


@pytest.mark.parametrize("seed", range(10))
def test_light_noise_never_swaps_digit_for_digit(seed):
    assert changes(TEXT, add_noise(TEXT, seed, "light")) <= LIGHT_ALLOWED


@pytest.mark.parametrize("seed", range(10))
def test_heavy_noise_only_makes_known_substitutions(seed):
    assert changes(TEXT, add_noise(TEXT, seed, "heavy")) <= HEAVY_ALLOWED


def test_heavy_damages_more_than_light():
    light = sum(len(changes(TEXT, add_noise(TEXT, s, "light"))) for s in range(10))
    heavy = sum(len(changes(TEXT, add_noise(TEXT, s, "heavy"))) for s in range(10))
    assert heavy > light


def test_text_without_noisy_characters_is_unchanged_apart_from_stray_marks():
    text = "abc def\nxyz"
    assert add_noise(text, 1, "heavy").replace("§", "") == text
