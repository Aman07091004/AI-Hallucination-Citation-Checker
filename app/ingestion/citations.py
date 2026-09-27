import re
from dataclasses import dataclass

# Citation-shaped patterns, most specific first.
_CITATION_PATTERNS = [
    # [2007] UKHL 21   /   [2019] EWHC 1847 (Comm)   /   [2023] EWHC 892 (TCC)
    r"\[\d{4}\]\s+[A-Z]{2,6}(?:\s*\(?[A-Za-z]{2,6}\)?)?\s+\d+(?:\s*\([A-Za-z]{2,6}\))?",
    # [1974] 1 WLR 798   /   [1972] 1 QB 60   /   [1996] 1 All ER 853
    r"\[\d{4}\]\s+\d+\s+[A-Za-z ]{2,10}\s+\d+",
    # [1975] AC 396
    r"\[\d{4}\]\s+[A-Za-z]{1,4}\s+\d+",
    # (1853) 2 E&B 216   /   (1854) 9 Ex 341
    r"\(\d{4}\)\s+\d+\s+[A-Za-z&]{1,5}\s+\d+",
]
_CITATION_RE = re.compile("|".join(f"(?:{p})" for p in _CITATION_PATTERNS))

_CASE_NAME_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.,'\-]*(?:\s+[A-Za-z0-9&.,'\-]+){0,8}"  # party 1 (up to ~9 words)
    r"\s+v\.?\s+"                                              # " v " or " v. "
    r"[A-Za-z0-9&.,'\-]+(?:\s+[A-Za-z0-9&.,'\-]+){0,8})"        # party 2
    r"\s*(?:\([^)]{1,40}\))?\s*$"                              # optional trailing "(The Achilleas)" etc.
)

_LOOKBACK_WINDOW = 120  # chars to search before a citation for the case name

# Strips lead-in phrases like "House of Lords in ", "Crestholm relies upon ",
# "Court of Queen's Bench in " so only the actual case name remains.
_LEADIN_RE = re.compile(r".*\b(?:in|upon)\s+(?=[A-Z])", re.IGNORECASE | re.DOTALL)


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _clean_case_name(raw: str) -> str:
    raw = _normalize_whitespace(raw)
    m = _LEADIN_RE.match(raw)
    return raw[m.end():].strip() if m else raw


@dataclass
class ExtractedCitation:
    case_name: str
    citation: str
    char_offset: int  # position of the citation in the source text


def extract_citations(text: str) -> list[ExtractedCitation]:
    results: list[ExtractedCitation] = []

    for match in _CITATION_RE.finditer(text):
        citation_str = _normalize_whitespace(match.group())
        if _looks_like_noise(citation_str):
            continue
        window_start = max(0, match.start() - _LOOKBACK_WINDOW)
        preceding_text = text[window_start:match.start()]

        name_match = _CASE_NAME_RE.search(preceding_text)
        case_name = _clean_case_name(name_match.group(1)) if name_match else "UNKNOWN"

        results.append(
            ExtractedCitation(
                case_name=case_name,
                citation=citation_str,
                char_offset=match.start(),
            )
        )

    return results


# Words that occasionally get swallowed into a citation match when a PDF's
# table layout collapses into run-on text
# with irregular spacing.
_NOISE_WORDS = {"ground", "page", "note", "ref", "ยง"}


def _looks_like_noise(citation: str) -> bool:
    words = re.findall(r"[A-Za-z]+", citation)
    return any(w.lower() in _NOISE_WORDS for w in words)


def get_citation_context(text: str, char_offset: int, window: int = 350) -> str:
    """Returns the text surrounding a citation - this is what a brief
    actually CLAIMS the case says, which is what we'll check against the
    real case text during proposition verification. We look mostly
    backwards from the citation since UK legal writing typically states
    the proposition first, then cites authority for it immediately after -
    e.g. '...inducing a breach of contract is itself an actionable tort,
    as established in Lumley v Gye (1853) 2 E&B 216.'

    We strip a couple of known noise patterns before returning: broken
    Word cross-reference fields (e.g. 'Error! Unknown document property
    name.') that leaked into the PDF text and would otherwise pollute the
    embedding/BM25 query with irrelevant tokens."""
    start = max(0, char_offset - window)
    end = min(len(text), char_offset + 40)
    raw = text[start:end]
    raw = raw.replace("Error! Unknown document property name.", " ")
    return _normalize_whitespace(raw)


def strip_self_reference(context: str, case_name: str, citation: str) -> str:
    """Removes the case name and citation string from a context snippet
    before it's used as a search query.

    Why this matters: our context snippet is built by looking backwards
    from the citation itself, so it naturally still contains the case name
    and citation text. If we search using that as-is, BM25 will often just
    find whichever chunk repeats that same citation string most exactly -
    typically a case's own title/header page - rather than the chunk that
    actually supports the legal claim being made. That's not a genuine
    proposition check, it's closer to search leaking its own answer key.
    Stripping these out forces retrieval to work on the substance of the
    claim alone."""
    cleaned = context
    for token in (citation, case_name):
        if token:
            cleaned = cleaned.replace(token, " ")
    return _normalize_whitespace(cleaned)


def dedupe_citations(citations: list[ExtractedCitation]) -> list[ExtractedCitation]:
    """A brief often cites the same case twice - once with full reasoning in
    the body, once in a reference table (e.g. this brief's 'Citation Index').
    Whitespace/line-wrap differences mean exact string matching isn't enough,
    so we dedupe on the citation with all whitespace already normalized."""
    seen: set[str] = set()
    deduped = []
    for c in citations:
        if c.citation not in seen:
            seen.add(c.citation)
            deduped.append(c)
    return deduped


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from app.ingestion.loader import load_document

    if len(sys.argv) != 2:
        print("Usage: python -m app.ingestion.citations <path-to-pdf-or-docx>")
        sys.exit(1)

    doc = load_document(Path(sys.argv[1]))
    citations = dedupe_citations(extract_citations(doc.text))

    print(f"Found {len(citations)} unique citations:\n")
    for c in citations:
        print(f"  {c.case_name}  ->  {c.citation}")
