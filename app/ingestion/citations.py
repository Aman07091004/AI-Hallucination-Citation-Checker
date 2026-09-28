"""
Extracts (case name, citation string) pairs from UK legal document text.

Handles the citation formats we actually found in your corpus and the sample
brief:
  - Neutral citations:      [2007] UKHL 21 | [2019] EWHC 1847 (Comm) | [2023] EWHC 892 (TCC)
  - Law report citations:   [1974] 1 WLR 798 | [1975] AC 396 | [1972] 1 QB 60
  - Old-style citations:    (1853) 2 E&B 216 | (1854) 9 Ex 341

Strategy: find citation-shaped substrings first (they're structurally
distinctive and cheap to match reliably), then look backwards from each match
for the nearest "X v Y" case name immediately preceding it. This is more
robust than trying to match name+citation in one giant regex, because case
names have very irregular shapes (ampersands, "& Co", "plc", "Ltd", KC/QC
counsel names nearby, etc.) while citations do not.
"""

import re
from dataclasses import dataclass, field

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

# Case name immediately before a citation, e.g. "OBG Ltd v Allan " or
# "Transfield Shipping Inc v Mercator Shipping Inc (The Achilleas) "
_CASE_NAME_RE = re.compile(
    r"([A-Z][A-Za-z0-9&.,'\-]*(?:\s+[A-Za-z0-9&.,'\-]+){0,8}"  # party 1 (up to ~9 words)
    r"\s+v\.?\s+"                                              # " v " or " v. "
    r"[A-Za-z0-9&.,'\-]+(?:\s+[A-Za-z0-9&.,'\-]+){0,8})"        # party 2
    r"\s*(?:\([^)]{1,40}\))?\s*$"                              # optional trailing "(The Achilleas)" etc.
)

_LOOKBACK_WINDOW = 120  # chars to search before a citation for the case name

# Strips lead-in phrases like "House of Lords in ", "Crestholm relies upon ",
# "Court of Queen's Bench in " so only the actual case name remains. We take
# the LAST such connector before a capital letter, since case names
# themselves are sometimes preceded by court/procedural framing text.
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
    char_offset: int = -1  # position of the citation itself in the source text, for later grounding
    kind: str = "case"  # "case" or "legislation"
    spans: list[tuple[int, int]] = field(default_factory=list)  # (start, end) highlight ranges - a
    # citation mentioned more than once ends up with one entry and multiple spans, via dedupe_citations

    def __post_init__(self) -> None:
        # Old-style construction (extract_citations) only ever gave us char_offset -
        # derive a single-span fallback from it so both construction styles work.
        if not self.spans and self.char_offset >= 0:
            self.spans = [(self.char_offset, self.char_offset + len(self.citation))]
        if self.char_offset < 0 and self.spans:
            self.char_offset = self.spans[0][0]

    @property
    def span_start(self) -> int:
        return self.spans[0][0]

    @property
    def span_end(self) -> int:
        return self.spans[0][1]


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
# table layout (e.g. a "Citation Index" appendix) collapses into run-on text
# with irregular spacing. None of these are real court/report abbreviations.
_NOISE_WORDS = {"ground", "page", "note", "ref", "ยง"}


def _looks_like_noise(citation: str) -> bool:
    words = re.findall(r"[A-Za-z]+", citation)
    return any(w.lower() in _NOISE_WORDS for w in words)


def get_citation_context(text: str, char_offset: int, window: int = 350, forward: int | None = None) -> str:
    """Returns the text surrounding a citation - this is what a brief
    actually CLAIMS the case says, which is what we'll check against the
    real case text during proposition verification. We look mostly
    backwards from the citation since UK legal writing typically states
    the proposition first, then cites authority for it immediately after -
    e.g. '...inducing a breach of contract is itself an actionable tort,
    as established in Lumley v Gye (1853) 2 E&B 216.'

    Pass `forward` for the opposite phrasing - "X v Y held that <claim>" -
    where the claim comes AFTER the citation. We stop at the first
    paragraph break so we never pull in the start of the next point.

    We strip a couple of known noise patterns before returning: broken
    Word cross-reference fields (e.g. 'Error! Unknown document property
    name.') that leaked into the PDF text and would otherwise pollute the
    embedding/BM25 query with irrelevant tokens."""
    if forward is not None:
        end = min(len(text), char_offset + forward)
        raw = text[char_offset:end]
        paragraph_break = raw.find("\n\n")
        if paragraph_break != -1:
            raw = raw[:paragraph_break]
    else:
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
    so we dedupe on the citation with all whitespace already normalized.

    Repeat mentions aren't just dropped: their spans are merged onto the
    first-seen entry, so a frontend can still highlight every occurrence of
    an authority the checker only reports on once."""
    first_seen: dict[str, ExtractedCitation] = {}
    order: list[str] = []
    for c in citations:
        if c.citation not in first_seen:
            first_seen[c.citation] = c
            order.append(c.citation)
        else:
            first_seen[c.citation].spans = first_seen[c.citation].spans + c.spans
    return [first_seen[citation] for citation in order]


# --- richer extraction: real case names in arbitrary prose, plus legislation ------
#
# extract_citations() above anchors on a fixed "in X v Y" shape via one big
# regex. That's fine for the validated brief, but it breaks on ordinary
# prose lead-ins ("This rule derives from...", "Reliance is further placed
# on...") because the regex's own word-repeat group is happy to swallow that
# lead-in text as if it were part of the party name. extract_authorities()
# instead walks backward from " v " one word at a time and stops the moment
# it hits a word that doesn't look like part of a name - either a lowercase
# word that isn't a name-connector ("of"/"and"/"the"/"for"), or a known
# citation-signal word ("See", "In", "Held", ...) even if it's capitalised
# because it started a sentence.

_SIGNAL_WORDS = {"see", "in", "upon", "held", "per", "citing", "applying", "following", "compare", "accord", "cf"}
_NAME_CONNECTOR_WORDS = {"of", "and", "the", "for", "&"}
_V_SEPARATOR_RE = re.compile(r"\s+v\.?\s+")
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]{1,40}\)\s*$")


def _find_case_name(text: str, window_start: int, v_start: int) -> tuple[str, int] | None:
    """Walks backward, word by word, from just before " v " (v_start, an
    absolute offset into `text`) looking for where the first party's name
    actually starts. Returns (name, absolute_start_offset), or None if
    nothing name-like immediately precedes " v "."""
    segment = text[window_start:v_start]
    tokens = list(re.finditer(r"\S+", segment))

    kept = []
    for tok in reversed(tokens):
        bare = re.sub(r"[^\w&]", "", tok.group()).lower()
        if bare in _SIGNAL_WORDS:
            break
        if tok.group()[:1].isupper() or bare in _NAME_CONNECTOR_WORDS:
            kept.append(tok)
            if len(kept) >= 9:  # matches the ~9-word cap extract_citations() uses
                break
        else:
            break

    if not kept:
        return None
    first_token = kept[-1]
    name = _normalize_whitespace(segment[first_token.start():])
    return name, window_start + first_token.start()


def _find_case_authority(text: str, citation_start: int, citation_end: int) -> ExtractedCitation:
    window_start = max(0, citation_start - _LOOKBACK_WINDOW)
    preceding = text[window_start:citation_start]

    case_name = "UNKNOWN"
    span_start = citation_start

    v_matches = list(_V_SEPARATOR_RE.finditer(preceding))
    if v_matches:
        v_match = v_matches[-1]  # the one immediately before the citation, not an earlier one
        party2 = _TRAILING_PAREN_RE.sub("", preceding[v_match.end():]).strip()
        name_result = _find_case_name(text, window_start, window_start + v_match.start())

        if party2 and name_result is not None:
            party1, name_start = name_result
            case_name = f"{party1} v {party2}"
            span_start = name_start

    citation_str = _normalize_whitespace(text[citation_start:citation_end])
    return ExtractedCitation(
        case_name=case_name,
        citation=citation_str,
        char_offset=citation_start,
        kind="case",
        spans=[(span_start, citation_end)],
    )


def _extract_case_authorities(text: str) -> list[ExtractedCitation]:
    """Like extract_citations(), but with the more robust backward-walk
    name search above, and each result carries its highlight span (name
    through citation) rather than just the citation's own offset."""
    results = []
    for match in _CITATION_RE.finditer(text):
        citation_str = _normalize_whitespace(match.group())
        if _looks_like_noise(citation_str):
            continue
        results.append(_find_case_authority(text, match.start(), match.end()))
    return results


def extract_authorities(text: str) -> list[ExtractedCitation]:
    """Every authority a brief relies on - both cases and legislation,
    combined into one list, kind-tagged. Does not dedupe - repeat mentions
    come back as separate entries; run the result through
    dedupe_citations() to merge them."""
    return _extract_case_authorities(text) + extract_legislation(text)


# --- legislation: "section 37 of the Senior Courts Act 1981" style references ----

_ACT_YEAR_RE = re.compile(r"\bAct\s+(\d{4})\b")
_ACT_TITLE_RE = re.compile(
    r"((?:[A-Z][A-Za-z0-9'&/-]*|\([^()\n]{1,60}\))"
    r"(?:\s+(?:[A-Z][A-Za-z0-9'&/-]*|\([^()\n]{1,60}\)|of|and|for))*)"
    r"\s*$"
)
_SECTION_LEADIN_RE = re.compile(r"(?i:sections?\s+[0-9]+(?:\s*(?:,|and)\s*[0-9]+)*\s+of\s+the)\s*$")
_GENERIC_TITLE_WORDS = {"the", "a", "an"}


def extract_legislation(text: str) -> list[ExtractedCitation]:
    """Finds "<Title> Act <year>" references. The title search is bounded
    to the current sentence (stops at the nearest '.', '!', '?' or newline
    before the match) so it can never bleed across a sentence boundary -
    e.g. "It was held. Human Rights Act 1998 applies." must not pick up
    "held" as part of the title. A bare "The Act 2010" with no real title
    word is not legislation we can identify, so it's skipped rather than
    reported with a useless "The" as its name.

    When the reference reads "section 37 of the X Act 1981", the citation
    (used for dedup/lookup) is just "X Act 1981", but the highlight span
    extends back to include "section 37 of the" - it's part of what the
    brief actually said, even though it's not part of the Act's title."""
    results = []
    for match in _ACT_YEAR_RE.finditer(text):
        year = match.group(1)
        act_start = match.start()

        boundary = max(
            text.rfind(".", 0, act_start),
            text.rfind("!", 0, act_start),
            text.rfind("?", 0, act_start),
            text.rfind("\n", 0, act_start),
        )
        search_start = boundary + 1 if boundary != -1 else 0
        preceding = text[search_start:act_start]

        title_match = _ACT_TITLE_RE.search(preceding)
        if title_match is None:
            continue
        raw_title = title_match.group(1).strip()
        if raw_title.lower() in _GENERIC_TITLE_WORDS:
            continue

        title_start = search_start + title_match.start(1)
        span_start = title_start
        leadin_match = _SECTION_LEADIN_RE.search(text[search_start:title_start])
        if leadin_match:
            span_start = search_start + leadin_match.start()

        results.append(
            ExtractedCitation(
                case_name="",
                citation=f"{raw_title} Act {year}",
                char_offset=title_start,
                kind="legislation",
                spans=[(span_start, match.end())],
            )
        )
    return results


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
