import dataclasses
from datetime import UTC, datetime

import pytest

from lifeadmin.sources.base import Document


def make_document(**overrides) -> Document:
    fields = {
        "id": 1,
        "title": "Sample passport",
        "content": "P<CANSAMPLE<<ALEX<<<<<<<<",
        "modified": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        "document_type_id": None,
        "tag_ids": (),
    }
    return Document(**(fields | overrides))


def test_repr_hides_ocr_text():
    assert "SAMPLE" not in repr(make_document())


def test_document_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_document().title = "changed"  # type: ignore[misc]
