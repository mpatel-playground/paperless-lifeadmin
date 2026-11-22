"""Upload the generated samples to the dev Paperless so they go through real OCR.

Run with ``make seed`` (after ``make up`` and ``make samples``). It uses ``PAPERLESS_URL``
and ``PAPERLESS_TOKEN`` from ``.env``.

Uses ``POST /api/documents/post_document/`` (multipart: ``document`` file plus ``title``).
Paperless answers with a consumption task id and does the OCR in the background. Running
it twice is harmless: Paperless rejects files it already has (same checksum) as duplicates.
"""

from collections.abc import Iterable
from pathlib import Path

import httpx

from evals.generate import SAMPLES_DIR
from lifeadmin.config import Settings, load_settings
from lifeadmin.sources.paperless import API_VERSION, PaperlessError

UPLOAD_PATH = "/api/documents/post_document/"
TIMEOUT_SECONDS = 60.0


def upload_samples(
    settings: Settings, paths: Iterable[Path], transport: httpx.BaseTransport | None = None
) -> list[str]:
    """Upload each PNG, titled by its sample id. Returns the Paperless task ids."""
    task_ids = []
    with httpx.Client(
        base_url=settings.paperless_url,
        headers={
            "Authorization": f"Token {settings.paperless_token}",
            "Accept": f"application/json; version={API_VERSION}",
        },
        timeout=TIMEOUT_SECONDS,
        transport=transport,
    ) as client:
        for path in paths:
            try:
                response = client.post(
                    UPLOAD_PATH,
                    data={"title": path.stem},
                    files={"document": (path.name, path.read_bytes(), "image/png")},
                )
            except httpx.HTTPError as exc:
                raise PaperlessError(
                    f"Could not reach Paperless ({type(exc).__name__}); check PAPERLESS_URL"
                ) from exc
            if response.is_error:
                raise PaperlessError(
                    f"Upload of {path.name} failed with HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            task_ids.append(response.json())
    return task_ids


def main() -> None:
    paths = sorted(SAMPLES_DIR.glob("*.png"))
    if not paths:
        raise SystemExit("No samples found; run `make samples` first")
    task_ids = upload_samples(load_settings(), paths)
    print(f"Queued {len(task_ids)} documents for OCR in Paperless")


if __name__ == "__main__":
    main()
