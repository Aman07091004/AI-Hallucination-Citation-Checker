"""
The actual product: reads a legal document, checks every citation, and
produces a verdict for each one. This is where every piece we've built so
far - loader, citation extractor, corpus matcher, hybrid retrieval, LLM
judge - comes together into one report.

Verdict categories:
  - SUPPORTED                    : case exists locally, and the claim matches what it says
  - MISQUOTED                    : case exists locally, but the claim doesn't match
  - UNCLEAR                      : case exists locally, but retrieval couldn't find enough to judge
  - REAL_BUT_NOT_IN_LOCAL_CORPUS : not in our corpus, but an external check found it's a real case
                                    (e.g. Wrotham Park, Series 5 Software - both real, neither in our 58 files)
  - LIKELY_FABRICATED            : not in our corpus, and no external evidence it exists anywhere
  - UNCERTAIN                    : external check couldn't determine either way
"""

from pathlib import Path

from app import config
from app.ingestion.citations import (
    dedupe_citations,
    extract_authorities,
    extract_citations,
    get_citation_context,
    strip_self_reference,
)
from app.ingestion.loader import load_document
from app.rag.corpus_index import build_registry, find_case_in_corpus
from app.rag.vectorstore import hybrid_query_case, ingest_case
from app.verification.external_check import check_citation_exists, classify_external_result
from app.verification.llm_judge import judge_proposition
from app.verification.scoring import categorize, resolve_confidence


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


def verify_text_iter(text: str, max_citations: int | None = None):
    """Streaming counterpart to verify_document() for the job API: an
    upload could take a while to fully check (each authority may need an
    LLM call), so instead of returning one flat list at the end, this
    yields the full citation list as soon as it's extracted, then a result
    per citation as its verdict lands - so a client can show authorities
    immediately and fill in verdicts one by one rather than staring at a
    blank screen.

    Events yielded:
      {"event": "extracted", "citations": [...], "truncated": bool, "total_found": int}
      {"event": "result", "citation": {...}}   (one per citation, in order)
    """
    # Legislation isn't checked here - there's no statute corpus or judge for
    # it, only case law - so only case authorities enter the pipeline below.
    authorities = [a for a in dedupe_citations(extract_authorities(text)) if a.kind == "case"]
    total_found = len(authorities)
    truncated = max_citations is not None and total_found > max_citations
    if truncated:
        authorities = authorities[:max_citations]

    base_citations = [
        {
            "id": i,
            "kind": authority.kind,
            "name": authority.case_name,
            "citation": authority.citation,
            "spans": [list(span) for span in authority.spans],
            "status": "pending",
        }
        for i, authority in enumerate(authorities)
    ]
    yield {"event": "extracted", "citations": base_citations, "truncated": truncated, "total_found": total_found}

    registry = build_registry()

    for base, authority in zip(base_citations, authorities):
        result = dict(base, status="done")
        match = find_case_in_corpus(authority.case_name, registry)

        if match.status == "MATCHED":
            ingest_case(match.matched_file, authority.case_name)

            claimed_context = get_citation_context(text, authority.char_offset)
            query_text = strip_self_reference(claimed_context, authority.case_name, authority.citation)
            top_hits = hybrid_query_case(match.matched_file, query_text, top_k=3)

            if not top_hits:
                verdict = "UNCLEAR"
                category, depth = categorize(verdict)
                result.update(
                    verdict=verdict, category=category, depth=depth,
                    confidence_pct=resolve_confidence(None, "low", verdict),
                    explanation="Case found locally but no passages could be retrieved.",
                    evidence={}, scope_note="",
                )
            else:
                passages = [hit["text"] for hit in top_hits]
                judgment = judge_proposition(authority.case_name, authority.citation, claimed_context, passages)
                category, depth = categorize(judgment["verdict"])
                avg_retrieval = sum(hit["blended_score"] for hit in top_hits) / len(top_hits)
                result.update(
                    verdict=judgment["verdict"], category=category, depth=depth,
                    confidence_pct=resolve_confidence(
                        None, judgment["confidence"], judgment["verdict"], retrieval_score=avg_retrieval
                    ),
                    explanation=judgment["explanation"],
                    evidence={"passages": passages}, scope_note="",
                )
        else:
            external = check_citation_exists(authority.case_name, authority.citation)
            verdict = classify_external_result(external)
            category, depth = categorize(verdict)
            result.update(
                verdict=verdict, category=category, depth=depth,
                confidence_pct=resolve_confidence(None, external["confidence"], verdict),
                explanation=external["evidence"] or "No supporting evidence found.",
                evidence={"source": external["source"]} if external["source"] else {},
                scope_note=(
                    "Not found in the local case corpus; checked for real-world existence via "
                    "web search only, not verified against the case's actual text."
                ),
            )

        yield {"event": "result", "citation": result}


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
