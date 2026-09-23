from dataclasses import dataclass
from pathlib import Path

import docx
import fitz  # pymupdf
import pdfplumber
import pytesseract
from PIL import Image

from app import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

MIN_CHARS_TO_TRUST_TEXT_LAYER = 20  # first-page heuristic, matches what we used to scan the corpus


@dataclass
class LoadedDocument:
    source_path: str
    text: str
    page_count: int
    used_ocr: bool


def _has_text_layer(pdf: "pdfplumber.PDF") -> bool:
    first_page_text = pdf.pages[0].extract_text() or ""
    return len(first_page_text.strip()) > MIN_CHARS_TO_TRUST_TEXT_LAYER


def _extract_pdf_text_layer(path: Path) -> tuple[str, int]:
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n\n".join(pages), len(pdf.pages)


def _extract_pdf_via_ocr(path: Path, dpi: int = 300) -> tuple[str, int]:
    """Render each page to an image with pymupdf, then OCR it with Tesseract.
    Slower than the text-layer path - only used when necessary."""
    doc = fitz.open(path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    page_texts = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        page_texts.append(pytesseract.image_to_string(img))
    page_count = doc.page_count
    doc.close()
    return "\n\n".join(page_texts), page_count


def load_pdf(path: Path) -> LoadedDocument:
    with pdfplumber.open(path) as pdf:
        text_layer_ok = _has_text_layer(pdf)

    if text_layer_ok:
        text, page_count = _extract_pdf_text_layer(path)
        return LoadedDocument(str(path), text, page_count, used_ocr=False)

    text, page_count = _extract_pdf_via_ocr(path)
    return LoadedDocument(str(path), text, page_count, used_ocr=True)


def load_docx(path: Path) -> LoadedDocument:
    d = docx.Document(path)
    text = "\n".join(p.text for p in d.paragraphs)
    return LoadedDocument(str(path), text, page_count=1, used_ocr=False)


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def load_corpus(corpus_dir: Path = config.RAW_CORPUS_DIR) -> list[LoadedDocument]:
    """Loads every .pdf/.docx in the corpus directory. Skips exact-duplicate
    filenames like the '(1).pdf' Caparo duplicate we found in the raw dataset -
    dedupe on (file size, first 500 chars) so a genuine re-issued judgment
    with the same name isn't silently dropped."""
    seen_fingerprints: set[tuple[int, str]] = set()
    docs: list[LoadedDocument] = []

    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() not in (".pdf", ".docx"):
            continue
        loaded = load_document(path)
        fingerprint = (path.stat().st_size, loaded.text[:500])
        if fingerprint in seen_fingerprints:
            print(f"Skipping duplicate: {path.name}")
            continue
        seen_fingerprints.add(fingerprint)
        docs.append(loaded)

    return docs


if __name__ == "__main__":
    # Quick manual smoke test: point RAW_CORPUS_DIR at your extracted
    # "Cambridge Hackathon - Case Law Database" folder and run this file directly.
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} documents")
    ocr_count = sum(1 for d in corpus if d.used_ocr)
    print(f"OCR was needed for {ocr_count} of them")
