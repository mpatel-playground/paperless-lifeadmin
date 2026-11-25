"""Fetching sample OCR text, from a fake DocumentSource; no Paperless."""

from datetime import UTC, datetime

import pytest

from evals.fetch import FetchError, fetch_ocr, write_ocr
from lifeadmin.sources.base import Document


class FakeSource:
    def __init__(self, docs: list[tuple[str, str]]) -> None:
        self.docs = [
            Document(
                id=i,
                title=title,
                content=content,
                modified=datetime(2026, 1, 1, tzinfo=UTC),
                document_type_id=None,
                tag_ids=(),
            )
            for i, (title, content) in enumerate(docs, start=1)
        ]

    def iter_documents(self, modified_after=None):
        return iter(self.docs)

    def get_document(self, document_id):
        raise NotImplementedError


def test_matches_samples_by_title_and_ignores_other_documents():
    source = FakeSource([("passport-01", "MRZ TEXT"), ("holiday photo", "x"), ("t4-01", "T4")])

    assert fetch_ocr(source, ["passport-01", "t4-01"]) == {
        "passport-01": "MRZ TEXT",
        "t4-01": "T4",
    }


def test_missing_sample_fails_and_names_it():
    source = FakeSource([("passport-01", "text")])

    with pytest.raises(FetchError, match="t4-01"):
        fetch_ocr(source, ["passport-01", "t4-01"])


def test_empty_ocr_fails():
    with pytest.raises(FetchError, match="no OCR text"):
        fetch_ocr(FakeSource([("t4-01", "  \n")]), ["t4-01"])


def test_duplicate_title_fails():
    with pytest.raises(FetchError, match="more than one"):
        fetch_ocr(FakeSource([("t4-01", "a"), ("t4-01", "b")]), ["t4-01"])


def test_write_ocr_ends_each_file_with_one_newline(tmp_path):
    write_ocr({"t4-01": "line one\nline two\n\n\n", "noa-01": "x"}, tmp_path)

    assert (tmp_path / "t4-01.txt").read_text() == "line one\nline two\n"
    assert (tmp_path / "noa-01.txt").read_text() == "x\n"
