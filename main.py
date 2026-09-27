from pathlib import Path

from app import config
from app.ingestion.citations import (
    dedupe_citations,
    extract_citations,
    get_citation_context,
    strip_self_reference,
)
from app.ingestion.loader import load_document
from app.rag.corpus_index import build_registry, find_case_in_corpus
from app.rag.vectorstore import hybrid_query_case, ingest_case
from app.verification.external_check import check_citation_exists, classify_external_result
from app.verification.llm_judge import judge_proposition


def verify_document(brief_path: Path) -> list[dict]:
    brief_doc = load_document(brief_path)
    citations = dedupe_citations(extract_citations(brief_doc.text))
    registry = build_registry()

    report = []

    for c in citations:
        match = find_case_in_corpus(c.case_name, registry)

        if match.status != "MATCHED":
            external = check_citation_exists(c.case_name, c.citation)
            verdict = classify_external_result(external)
            report.append(
                {
                    "case_name": c.case_name,
                    "citation": c.citation,
                    "verdict": verdict,
                    "confidence": external["confidence"],
                    "explanation": external["evidence"] or "No supporting evidence found.",
                }
            )
            continue

        ingest_case(match.matched_file, c.case_name)

        claimed_context = get_citation_context(brief_doc.text, c.char_offset)
        query_text = strip_self_reference(claimed_context, c.case_name, c.citation)
        top_hits = hybrid_query_case(match.matched_file, query_text, top_k=3)

        if not top_hits:
            report.append(
                {
                    "case_name": c.case_name,
                    "citation": c.citation,
                    "verdict": "UNCLEAR",
                    "confidence": "low",
                    "explanation": "Case found locally but no passages could be retrieved.",
                }
            )
            continue

        passages = [hit["text"] for hit in top_hits]
        judgment = judge_proposition(c.case_name, c.citation, claimed_context, passages)

        report.append(
            {
                "case_name": c.case_name,
                "citation": c.citation,
                "verdict": judgment["verdict"],
                "confidence": judgment["confidence"],
                "explanation": judgment["explanation"],
                "from_cache": judgment.get("from_cache", False),
            }
        )

    return report


def print_report(report: list[dict]) -> None:
    print(f"\n{'Case':45} {'Citation':25} {'Verdict':22} Confidence")
    print("-" * 110)
    for entry in report:
        print(
            f"{entry['case_name'][:44]:45} {entry['citation'][:24]:25} "
            f"{entry['verdict']:22} {entry['confidence']}"
        )
        print(f"    -> {entry['explanation']}")


if __name__ == "__main__":
    brief_path = config.RAW_CORPUS_DIR / "White and Case.pdf"
    report = verify_document(brief_path)
    print_report(report)

    verdict_counts = {}
    for entry in report:
        verdict_counts[entry["verdict"]] = verdict_counts.get(entry["verdict"], 0) + 1
    print(f"\nSummary: {verdict_counts}")
