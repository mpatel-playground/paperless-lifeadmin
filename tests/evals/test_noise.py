"""OCR noise for the eval: seeded, and only the kinds of damage seen on real output."""

from evals.noise import add_noise

# Synthetic text with every character the noise can touch.
TEXT = "\n".join(
    [
        "PASSEPORT / PASSPORT  Nom: RAMAN-CÔTÉ  Prénoms: ÉLISE",
        "Date of expiry: 15 JUN 2031  Issued: 2021-06-15  No. 0158 1850 5581",
        "P<CANBERGERON<<ELISE<<<<<<<<<<<<<<<<<<<<<<<<",
        "AK482913<0CAN9003148F3106150<<<<<<<<<<<<<<<0",
        "EIRE SOLDIER LOBBY",
    ]
    * 4
)


def test_same_seed_gives_same_text():
    assert add_noise(TEXT, 7) == add_noise(TEXT, 7)


def test_different_seeds_give_different_text():
    assert add_noise(TEXT, 1) != add_noise(TEXT, 2)


def test_applies_each_kind_of_damage():
    noisy = add_noise(TEXT, 3)

    assert "El" in noisy and "EI" not in noisy
    assert "§" in noisy
    assert "CÔTÉ" not in noisy or "ÉLISE" not in noisy  # some accents dropped
    digits_before = sum(c.isdigit() for c in TEXT)
    assert sum(c.isdigit() for c in noisy) != digits_before  # some swaps happened


def test_only_known_substitutions_and_stray_marks():
    noisy = add_noise(TEXT, 5)
    allowed = {
        ("0", "O"),
        ("1", "l"),
        ("5", "S"),
        ("8", "B"),
        ("O", "0"),
        ("l", "1"),
        ("I", "1"),
        ("I", "l"),  # EI -> El
        ("S", "5"),
        ("B", "8"),
        ("Ô", "O"),
        ("É", "E"),
        ("é", "e"),
    }
    clean_lines = TEXT.split("\n")
    noisy_lines = noisy.split("\n")
    assert len(noisy_lines) == len(clean_lines)
    for before, after in zip(clean_lines, noisy_lines, strict=True):
        after = after.removesuffix("§")
        assert len(after) == len(before)
        for a, b in zip(before, after, strict=True):
            assert a == b or (a, b) in allowed, (a, b)


def test_text_without_noisy_characters_is_unchanged_apart_from_stray_marks():
    text = "abc def\nxyz"
    assert add_noise(text, 1).replace("§", "") == text
