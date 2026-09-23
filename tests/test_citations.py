from pathlib import Path

import pytest

from app import config
from app.ingestion.citations import dedupe_citations, extract_citations
from app.ingestion.loader import load_document

# Ground truth we confirmed by hand: 7 real (in local corpus), 3 fabricated
# (no match anywhere, fictional parties), 2 real-but-outside-corpus.
EXPECTED_CITATIONS = {
    "(1853) 2 E&B 216",          # Lumley v Gye - real
    "[2007] UKHL 21",             # OBG v Allan - real
    "[1952] Ch 646",              # DC Thomson v Deakin - real
    "[2019] EWHC 1847 (Comm)",    # Fairfax v Brennan Holdings - FABRICATED
    "(1854) 9 Ex 341",            # Hadley v Baxendale - real
    "[2008] UKHL 48",             # Transfield v Mercator (The Achilleas) - real
    "[1972] 1 QB 60",             # Anglia Television v Reed - real
    "[2021] EWHC 3312 (Ch)",      # Stonegate v Redwood Procurement - FABRICATED
    "[1974] 1 WLR 798",           # Wrotham Park v Parkside Homes - real, not in corpus
    "[1975] AC 396",              # American Cyanamid v Ethicon - real
    "[1996] 1 All ER 853",        # Series 5 Software v Clarke - real, not in corpus
    "[2023] EWHC 892 (TCC)",      # Pemberton Aerospace v Delta Global - FABRICATED
}


def test_extracts_all_12_known_citations():
    path = config.RAW_CORPUS_DIR / "White and Case.pdf"
    if not path.exists():
        pytest.skip("Add 'White and Case.pdf' to data/raw_corpus/ to run this test")

    doc = load_document(path)
    citations = dedupe_citations(extract_citations(doc.text))
    found = {c.citation for c in citations}

    assert found == EXPECTED_CITATIONS, (
        f"Missing: {EXPECTED_CITATIONS - found}\nExtra/unexpected: {found - EXPECTED_CITATIONS}"
    )
