"""Save the Paperless OCR text of each seeded sample to ``evals/ocr/<id>.txt``.

Run with ``make eval-fetch`` after ``make seed`` and once Paperless has finished OCR. The
text files are committed: the samples are synthetic, and having the real OCR output on disk
lets ``make eval`` (and CI) run without a Paperless instance.

Samples are matched by title, which ``make seed`` sets to the sample id.
"""

from collections.abc import Iterable
from pathlib import Path

from evals.generate import EVALS_DIR, load_samples
from lifeadmin.config import load_settings
from lifeadmin.sources.base import DocumentSource
from lifeadmin.sources.paperless import PaperlessSource

OCR_DIR = EVALS_DIR / "ocr"


class FetchError(RuntimeError):
    """A sample is missing, duplicated or has no OCR text yet."""


def fetch_ocr(source: DocumentSource, sample_ids: Iterable[str]) -> dict[str, str]:
    """OCR text for every sample id. Fails on the first problem rather than saving a gap."""
    wanted = set(sample_ids)
    found: dict[str, str] = {}
    for doc in source.iter_documents():
        if doc.title not in wanted:
            continue
        if doc.title in found:
            raise FetchError(f"{doc.title}: more than one Paperless document has this title")
        if not doc.content.strip():
            raise FetchError(f"{doc.title}: no OCR text yet; wait for Paperless to finish")
        found[doc.title] = doc.content

    missing = sorted(wanted - found.keys())
    if missing:
        raise FetchError(f"Not in Paperless (run `make seed`?): {', '.join(missing)}")
    return found


def write_ocr(texts: dict[str, str], out_dir: Path = OCR_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, text in texts.items():
        # Always end with one newline, so a re-fetch of unchanged text gives no git diff.
        (out_dir / f"{sample_id}.txt").write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def main() -> None:
    ids = [s.id for s in load_samples()]
    with PaperlessSource(load_settings()) as source:
        texts = fetch_ocr(source, ids)
    write_ocr(texts)
    print(f"Saved OCR text for {len(texts)} samples to {OCR_DIR}")


if __name__ == "__main__":
    main()
