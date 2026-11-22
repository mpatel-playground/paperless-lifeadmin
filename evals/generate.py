"""Render the synthetic eval documents described in ``evals/labels.toml`` as PNGs.

Run with ``make samples``. Each sample becomes ``evals/samples/<id>.png``, which
``make seed`` uploads to the dev Paperless so it goes through real OCR.

The work is split in two so it can be tested without looking at pixels:

- ``document_lines`` turns a ``Sample`` into the text printed on the document (pure).
- ``render`` draws those lines onto an image.

MRZ lines are built from the ICAO 9303 layouts, using ``check_digit`` from
``lifeadmin.extract.mrz``. Building the samples therefore also exercises the parser's
check-digit code, and the tests compare our MRZ against ICAO's published specimens.
"""

import tomllib
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from lifeadmin.extract.mrz import check_digit

EVALS_DIR = Path(__file__).parent
LABELS_PATH = EVALS_DIR / "labels.toml"
SAMPLES_DIR = EVALS_DIR / "samples"

# Same values as tracked_document.doc_kind in docs/DESIGN.md (the subset we generate).
KINDS = ("passport", "pr_card", "drivers_licence", "health_card", "payslip", "t4", "noa")
# Kinds whose documents never carry an expiry date.
NO_EXPIRY_KINDS = ("payslip", "t4", "noa")

# Not verified against a real Canadian PR card: ICAO only says a TD1 code starts with
# I, A or C. Check a real card before relying on this in the parser.
PR_CARD_DOC_CODE = "I<"
ISSUING_STATE = "CAN"

# Shown on every document so a sample can never be mistaken for a real one.
SPECIMEN_FOOTER = "SPECIMEN - synthetic test document / document d'essai fictif"

Style = Literal["title", "body", "mrz"]
Line = tuple[str, Style]

MONTHS = (
    ("JAN", "JAN"),
    ("FEB", "FÉV"),
    ("MAR", "MARS"),
    ("APR", "AVR"),
    ("MAY", "MAI"),
    ("JUN", "JUIN"),
    ("JUL", "JUIL"),
    ("AUG", "AOÛT"),
    ("SEP", "SEPT"),
    ("OCT", "OCT"),
    ("NOV", "NOV"),
    ("DEC", "DÉC"),
)
MONTH_NAMES_EN = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class Sample:
    id: str
    kind: str
    holder: str
    issue_date: date | None = None
    expiry_date: date | None = None
    number: str | None = None
    birth_date: date | None = None
    sex: str | None = None
    tax_year: int | None = None
    lang: Literal["en", "fr"] = "en"

    @property
    def surname(self) -> str:
        return self.holder.rsplit(" ", 1)[-1]

    @property
    def given_names(self) -> str:
        return self.holder.rsplit(" ", 1)[0] if " " in self.holder else ""


def load_samples(path: Path = LABELS_PATH) -> list[Sample]:
    """Read the answer key. Unknown keys or kinds fail loudly rather than being ignored."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    samples = [Sample(**entry) for entry in raw["sample"]]
    for s in samples:
        if s.kind not in KINDS:
            raise ValueError(f"{s.id}: unknown kind {s.kind!r}")
    return samples


# --- MRZ (ICAO Doc 9303, parts 4 and 5) ---


def mrz_text(value: str) -> str:
    """Upper-case ``value`` and map it to the MRZ alphabet (A-Z, 0-9, ``<``).

    Accents are dropped (``É`` -> ``E``), and spaces and hyphens become ``<``. ICAO allows
    other transliterations for some letters; this simple form covers our sample names.
    """
    decomposed = unicodedata.normalize("NFKD", value.upper())
    out = []
    for ch in decomposed:
        if unicodedata.combining(ch):
            continue
        if ch in " -":
            out.append("<")
        elif ch.isascii() and ch.isalnum():
            out.append(ch)
    return "".join(out)


def mrz_name(surname: str, given_names: str, width: int) -> str:
    """``SURNAME<<GIVEN<NAMES`` padded with ``<`` (or truncated) to ``width``."""
    return f"{mrz_text(surname)}<<{mrz_text(given_names)}".ljust(width, "<")[:width]


def mrz_date(d: date) -> str:
    return d.strftime("%y%m%d")


def _with_check(field: str) -> str:
    return field + str(check_digit(field))


def td3_lines(
    *,
    doc_number: str,
    surname: str,
    given_names: str,
    nationality: str,
    birth_date: date,
    sex: str,
    expiry_date: date,
    issuing_state: str = ISSUING_STATE,
    personal_number: str = "",
) -> tuple[str, str]:
    """The two 44-character MRZ lines of a passport data page."""
    line1 = "P<" + issuing_state + mrz_name(surname, given_names, 39)

    number = _with_check(doc_number.ljust(9, "<"))
    birth = _with_check(mrz_date(birth_date))
    expiry = _with_check(mrz_date(expiry_date))
    personal = personal_number.ljust(14, "<")
    # An unused personal number field takes a filler as its check digit (9303 part 4).
    personal += str(check_digit(personal)) if personal_number else "<"

    composite = check_digit(number + birth + expiry + personal)
    line2 = f"{number}{nationality}{birth}{sex}{expiry}{personal}{composite}"
    return line1, line2


def td1_lines(
    *,
    doc_code: str,
    doc_number: str,
    surname: str,
    given_names: str,
    nationality: str,
    birth_date: date,
    sex: str,
    expiry_date: date,
    issuing_state: str = ISSUING_STATE,
) -> tuple[str, str, str]:
    """The three 30-character MRZ lines of an ID card."""
    number = _with_check(doc_number.ljust(9, "<"))
    optional1 = "<" * 15
    line1 = doc_code + issuing_state + number + optional1

    birth = _with_check(mrz_date(birth_date))
    expiry = _with_check(mrz_date(expiry_date))
    optional2 = "<" * 11
    # The composite covers line 1 from the document number on, plus the dates and
    # optional data of line 2 (9303 part 5).
    composite = check_digit(number + optional1 + birth + expiry + optional2)
    line2 = f"{birth}{sex}{expiry}{nationality}{optional2}{composite}"

    line3 = mrz_name(surname, given_names, 30)
    return line1, line2, line3


# --- Printed date formats (deliberately varied, like real documents) ---


def bilingual_month_date(d: date) -> str:
    """Passport style: ``15 JUN /JUIN 2031``."""
    en, fr = MONTHS[d.month - 1]
    return f"{d.day:02d} {en} /{fr} {d.year}"


def slash_ymd(d: date) -> str:
    return d.strftime("%Y/%m/%d")


def slash_dmy(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def long_en(d: date) -> str:
    return f"{MONTH_NAMES_EN[d.month - 1]} {d.day}, {d.year}"


def short_en(d: date) -> str:
    return f"{MONTH_NAMES_EN[d.month - 1][:3]} {d.day}, {d.year}"


# --- Document layouts ---


def _label(sample: Sample, en: str, fr: str) -> str:
    return f"{fr} / {en}" if sample.lang == "fr" else f"{en} / {fr}"


def _require(sample: Sample, *fields: str) -> None:
    missing = [f for f in fields if getattr(sample, f) is None]
    if missing:
        raise ValueError(f"{sample.id}: {sample.kind} needs {', '.join(missing)}")


def _passport(s: Sample) -> list[Line]:
    _require(s, "number", "birth_date", "sex", "issue_date", "expiry_date")
    mrz = td3_lines(
        doc_number=s.number,
        surname=s.surname,
        given_names=s.given_names,
        nationality=ISSUING_STATE,
        birth_date=s.birth_date,
        sex=s.sex,
        expiry_date=s.expiry_date,
    )
    return [
        (_label(s, "CANADA PASSPORT", "PASSEPORT CANADA"), "title"),
        (
            f"Type: P    {_label(s, 'Passport No.', 'N° du passeport')}: {s.number}",
            "body",
        ),
        (f"{_label(s, 'Surname', 'Nom')}: {s.surname.upper()}", "body"),
        (f"{_label(s, 'Given names', 'Prénoms')}: {s.given_names.upper()}", "body"),
        (f"{_label(s, 'Nationality', 'Nationalité')}: CANADIAN/CANADIENNE", "body"),
        (
            f"{_label(s, 'Date of birth', 'Date de naissance')}: "
            f"{bilingual_month_date(s.birth_date)}",
            "body",
        ),
        (f"{_label(s, 'Sex', 'Sexe')}: {s.sex}", "body"),
        (
            f"{_label(s, 'Date of issue', 'Date de délivrance')}: "
            f"{bilingual_month_date(s.issue_date)}",
            "body",
        ),
        (
            f"{_label(s, 'Date of expiry', "Date d'expiration")}: "
            f"{bilingual_month_date(s.expiry_date)}",
            "body",
        ),
        ("", "body"),
        *[(line, "mrz") for line in mrz],
    ]


def _pr_card(s: Sample) -> list[Line]:
    _require(s, "number", "birth_date", "sex", "issue_date", "expiry_date")
    mrz = td1_lines(
        doc_code=PR_CARD_DOC_CODE,
        doc_number=s.number,
        surname=s.surname,
        given_names=s.given_names,
        # A PR holder isn't a Canadian national; a fixed made-up code keeps it synthetic.
        nationality="UTO",
        birth_date=s.birth_date,
        sex=s.sex,
        expiry_date=s.expiry_date,
    )
    return [
        (_label(s, "PERMANENT RESIDENT CARD", "CARTE DE RÉSIDENT PERMANENT"), "title"),
        (f"{_label(s, 'Name', 'Nom')}: {s.holder.upper()}", "body"),
        (f"{_label(s, 'ID No.', "N° d'identification")}: {s.number}", "body"),
        (f"{_label(s, 'Date of birth', 'Date de naissance')}: {slash_ymd(s.birth_date)}", "body"),
        (f"{_label(s, 'Issued', 'Délivrée')}: {slash_ymd(s.issue_date)}", "body"),
        (f"{_label(s, 'Expiry', 'Expiration')}: {slash_ymd(s.expiry_date)}", "body"),
        ("", "body"),
        *[(line, "mrz") for line in mrz],
    ]


def _drivers_licence(s: Sample) -> list[Line]:
    _require(s, "number", "birth_date", "sex", "issue_date", "expiry_date")
    return [
        ("ONTARIO  DRIVER'S LICENCE / PERMIS DE CONDUIRE", "title"),
        (f"4d DL/NO. {s.number}", "body"),
        (f"1,2 {s.surname.upper()}, {s.given_names.upper()}", "body"),
        (f"4a ISS {slash_ymd(s.issue_date)}", "body"),
        (f"4b EXP {slash_ymd(s.expiry_date)}", "body"),
        (f"3 DOB/DDN {slash_ymd(s.birth_date)}    15 SEX/SEXE {s.sex}", "body"),
        ("9 CLASS/CAT G", "body"),
    ]


def _health_card(s: Sample) -> list[Line]:
    _require(s, "number", "birth_date")
    lines: list[Line] = [
        ("Ontario  Health / Santé", "title"),
        (s.holder.upper(), "body"),
        (s.number, "body"),
        (f"{_label(s, 'Born', 'Né(e) le')}: {slash_dmy(s.birth_date)}", "body"),
    ]
    if s.issue_date:
        lines.append((f"{_label(s, 'Issued', 'Délivrée')}: {slash_dmy(s.issue_date)}", "body"))
    if s.expiry_date:
        label = _label(s, "Valid until", "Valide jusqu'au")
        lines.append((f"{label}: {slash_dmy(s.expiry_date)}", "body"))
    return lines


def _payslip(s: Sample) -> list[Line]:
    _require(s, "issue_date")
    period_end = s.issue_date - timedelta(days=2)
    period_start = period_end - timedelta(days=13)
    return [
        ("Northwind Maple Ltd. - Earnings Statement", "title"),
        (f"Employee: {s.holder}", "body"),
        (f"Pay period: {period_start.isoformat()} to {period_end.isoformat()}", "body"),
        (f"Pay date: {short_en(s.issue_date)}", "body"),
        ("Regular earnings       80.00 h    2,846.15", "body"),
        ("CPP contributions                   -158.22", "body"),
        ("EI premiums                          -46.68", "body"),
        ("Federal income tax                  -301.44", "body"),
        ("Ontario income tax                  -142.87", "body"),
        ("Net pay                            2,196.94", "body"),
    ]


def _t4(s: Sample) -> list[Line]:
    _require(s, "tax_year")
    title = _label(s, "T4 Statement of Remuneration Paid", "T4 État de la rémunération payée")
    return [
        (title, "title"),
        (f"{_label(s, 'Year', 'Année')}: {s.tax_year}", "body"),
        (f"{_label(s, "Employer's name", "Nom de l'employeur")}: Northwind Maple Ltd.", "body"),
        (f"{_label(s, "Employee's name", "Nom de l'employé")}: {s.holder.upper()}", "body"),
        ("14  Employment income / Revenus d'emploi      71,480.00", "body"),
        ("16  CPP contributions / Cotisations au RPC     3,867.50", "body"),
        ("18  EI premiums / Cotisations d'AE               1,049.12", "body"),
        ("22  Income tax deducted / Impôt retenu        11,962.30", "body"),
    ]


def _noa(s: Sample) -> list[Line]:
    _require(s, "tax_year", "issue_date")
    return [
        ("Canada Revenue Agency - Notice of Assessment", "title"),
        (f"Tax year: {s.tax_year}", "body"),
        (f"Date issued: {long_en(s.issue_date)}", "body"),
        (f"Name: {s.holder}", "body"),
        ("Line 15000  Total income                    71,480.00", "body"),
        ("Line 26000  Taxable income                  68,912.00", "body"),
        ("Line 43500  Total payable                   11,804.16", "body"),
        ("Balance                                       -158.14", "body"),
        ("RRSP deduction limit for next year         13,207.00", "body"),
    ]


LAYOUTS = {
    "passport": _passport,
    "pr_card": _pr_card,
    "drivers_licence": _drivers_licence,
    "health_card": _health_card,
    "payslip": _payslip,
    "t4": _t4,
    "noa": _noa,
}


def document_lines(sample: Sample) -> list[Line]:
    """The text printed on ``sample``'s document, top to bottom, with a SPECIMEN footer."""
    return [*LAYOUTS[sample.kind](sample), ("", "body"), (SPECIMEN_FOOTER, "body")]


# --- Rendering ---

WIDTH = 1800
MARGIN = 80
LINE_GAP = 18
FONT_SIZES: dict[Style, int] = {"title": 52, "body": 38, "mrz": 44}
# System fonts, macOS first, then Linux (CI). The first one found is used. Pillow's bundled
# font is only a fallback: it has no accented letters, so French labels come out as boxes.
TEXT_FONT_PATHS = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
# A monospaced face for the MRZ, as on real documents.
MONO_FONT_PATHS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


def _font(paths: tuple[str, ...], size: int) -> ImageFont.FreeTypeFont:
    path = next((p for p in paths if Path(p).exists()), None)
    if path is None:
        warnings.warn("No system font found; accented letters will not render", stacklevel=2)
        return ImageFont.load_default(size=size)
    return ImageFont.truetype(path, size)


def _fonts() -> dict[Style, ImageFont.FreeTypeFont]:
    return {
        "title": _font(TEXT_FONT_PATHS, FONT_SIZES["title"]),
        "body": _font(TEXT_FONT_PATHS, FONT_SIZES["body"]),
        "mrz": _font(MONO_FONT_PATHS, FONT_SIZES["mrz"]),
    }


def render(lines: list[Line]) -> Image.Image:
    """Black text on a white page, one line per row, sized to fit the content."""
    fonts = _fonts()
    heights = [FONT_SIZES[style] + LINE_GAP for _, style in lines]
    image = Image.new("L", (WIDTH, sum(heights) + 2 * MARGIN), color=255)
    draw = ImageDraw.Draw(image)
    y = MARGIN
    for (text, style), height in zip(lines, heights, strict=True):
        draw.text((MARGIN, y), text, font=fonts[style], fill=0)
        y += height
    return image


def generate(samples: list[Sample], out_dir: Path = SAMPLES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample in samples:
        path = out_dir / f"{sample.id}.png"
        render(document_lines(sample)).save(path)
        paths.append(path)
    return paths


def main() -> None:
    paths = generate(load_samples())
    print(f"Wrote {len(paths)} samples to {SAMPLES_DIR.relative_to(Path.cwd())}/")


if __name__ == "__main__":
    main()
