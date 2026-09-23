import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

from app import config

_STOPWORDS = {
    "ltd", "plc", "inc", "co", "company", "and", "another", "others", "the",
    "v", "case", "law", "uk", "nondevolved", "england", "wales", "of",
    "corporation", "corp", "sa", "nv", "group", "limited",
}


def _normalize(name: str) -> set[str]:
    words = re.findall(r"[a-z]+", name.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


@dataclass
class CorpusEntry:
    file_path: Path
    normalized_tokens: set[str]


@dataclass
class MatchResult:
    status: str  # "MATCHED" or "NOT_FOUND"
    matched_file: Path | None
    score: float  # 0-100, rapidfuzz token_sort_ratio on the normalized tokens


def build_registry(corpus_dir: Path = config.RAW_CORPUS_DIR) -> list[CorpusEntry]:
    entries = []
    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() not in (".pdf", ".docx"):
            continue
        if path.stem == "brief_sample":  # exclude the brief itself, it's not a case in the corpus
            continue
        entries.append(CorpusEntry(path, _normalize(path.stem)))
    return entries


def find_case_in_corpus(
    case_name: str, registry: list[CorpusEntry], score_threshold: float = 70.0
) -> MatchResult:
    query_tokens = _normalize(case_name)
    if not query_tokens:
        return MatchResult("NOT_FOUND", None, 0.0)

    best_score = 0.0
    best_entry: CorpusEntry | None = None

    for entry in registry:
        # token_sort_ratio handles word-order differences and partial overlap
        # well for short party-name sets like these.
        score = fuzz.token_sort_ratio(" ".join(sorted(query_tokens)), " ".join(sorted(entry.normalized_tokens)))
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_score >= score_threshold:
        return MatchResult("MATCHED", best_entry.file_path, best_score)
    return MatchResult("NOT_FOUND", None, best_score)


if __name__ == "__main__":
    from app.ingestion.citations import dedupe_citations, extract_citations
    from app.ingestion.loader import load_document

    brief_path = config.RAW_CORPUS_DIR / "White and Case.pdf"
    doc = load_document(brief_path)
    citations = dedupe_citations(extract_citations(doc.text))
    registry = build_registry()

    print(f"Corpus has {len(registry)} candidate case files.\n")
    print(f"{'Case':55} {'Citation':30} {'Status':12} Score  Matched file")
    print("-" * 130)
    for c in citations:
        result = find_case_in_corpus(c.case_name, registry)
        matched_name = result.matched_file.name if result.matched_file else "-"
        print(f"{c.case_name[:54]:55} {c.citation[:29]:30} {result.status:12} {result.score:5.1f}  {matched_name}")
