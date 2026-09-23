from pathlib import Path

import pytest

from app import config
from app.ingestion.loader import load_document


def _first_file(pattern: str) -> Path | None:
    matches = list(config.RAW_CORPUS_DIR.glob(pattern))
    return matches[0] if matches else None


def test_text_layer_pdf_extracts_without_ocr():
    path = _first_file("*Lumley*")
    if path is None:
        pytest.skip("Add a text-layer sample PDF (e.g. Lumley v Gye) to data/raw_corpus/ to run this test")
    doc = load_document(path)
    assert doc.used_ocr is False
    assert len(doc.text.strip()) > 100


def test_scanned_pdf_falls_back_to_ocr():
    path = _first_file("*broome*")
    if path is None:
        pytest.skip("Add a scanned sample PDF (e.g. Broome v Cassell) to data/raw_corpus/ to run this test")
    doc = load_document(path)
    assert doc.used_ocr is True
    assert len(doc.text.strip()) > 100
