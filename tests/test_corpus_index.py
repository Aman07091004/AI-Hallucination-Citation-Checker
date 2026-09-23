import pytest

from app import config
from app.ingestion.citations import dedupe_citations, extract_citations
from app.ingestion.loader import load_document
from app.rag.corpus_index import build_registry, find_case_in_corpus

# Ground truth confirmed by hand + independent web search (see conversation
# history): these 7 should resolve locally, these 5 should not.
EXPECTED_MATCHED = {
    "(1853) 2 E&B 216", "[2007] UKHL 21", "[1952] Ch 646",
    "(1854) 9 Ex 341", "[2008] UKHL 48", "[1972] 1 QB 60", "[1975] AC 396",
}
EXPECTED_NOT_FOUND = {
    "[2019] EWHC 1847 (Comm)",   # fabricated
    "[2021] EWHC 3312 (Ch)",     # fabricated
    "[2023] EWHC 892 (TCC)",     # fabricated
    "[1974] 1 WLR 798",          # real, just not in this corpus
    "[1996] 1 All ER 853",       # real, just not in this corpus
}


def test_corpus_matching_matches_known_ground_truth():
    brief_path = config.RAW_CORPUS_DIR / "White and Case.pdf"
    if not brief_path.exists() or not list(config.RAW_CORPUS_DIR.glob("*.pdf")):
        pytest.skip("Add the full corpus + 'White and Case.pdf' to data/raw_corpus/ to run this test")

    doc = load_document(brief_path)
    citations = dedupe_citations(extract_citations(doc.text))
    registry = build_registry()

    matched = {c.citation for c in citations if find_case_in_corpus(c.case_name, registry).status == "MATCHED"}
    not_found = {c.citation for c in citations if find_case_in_corpus(c.case_name, registry).status == "NOT_FOUND"}

    assert matched == EXPECTED_MATCHED
    assert not_found == EXPECTED_NOT_FOUND
