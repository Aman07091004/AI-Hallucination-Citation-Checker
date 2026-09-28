"""
Turns a raw verdict string (from llm_judge.py or external_check.py) into the
three things the job API actually needs to display: a category bucket
("verified" / "misapplied" / "fabricated" / "unresolved"), a depth label
("full" - checked against the case's actual text, or "existence_only" - only
checked that the case is real), and a single 0-100 confidence number.

Categorisation is a fixed lookup, not inference - an unrecognised verdict
must fall back to "unresolved", never to "verified". A citation we don't
understand is exactly the case we most need to NOT wave through.
"""

import re

_VERDICT_TO_CATEGORY = {
    "SUPPORTED": ("verified", "full"),
    "MISQUOTED": ("misapplied", "full"),
    "UNCLEAR": ("unresolved", "full"),
    "LIKELY_FABRICATED": ("fabricated", "existence_only"),
    "REAL_BUT_NOT_IN_LOCAL_CORPUS": ("verified", "existence_only"),
    "UNCERTAIN": ("unresolved", "existence_only"),
}

_LABEL_CONFIDENCE_PCT = {"high": 90, "medium": 65, "low": 40}

# Retrieval quality ceilings: even a confident-sounding LLM judgment isn't
# trustworthy if the passages it was given weren't actually a good match for
# the claim - so a weak retrieval score caps how high the final confidence
# can go, regardless of what the model itself claimed.
_RETRIEVAL_CEILINGS = [
    (0.70, 100),
    (0.55, 90),
    (0.40, 75),
    (0.0, 60),
]


def categorize(verdict: str) -> tuple[str, str]:
    """Returns (category, depth). Unknown verdicts default to
    ("unresolved", "full") - never silently "verified"."""
    return _VERDICT_TO_CATEGORY.get(verdict, ("unresolved", "full"))


def parse_confidence_pct(value) -> int | None:
    """Normalises a confidence value of unknown shape (an int/float
    percentage, a fraction, a "NN%" string) into an int 0-100, or None if
    it isn't a number at all (e.g. a label like "high", which the caller
    should fall back to resolve_confidence()'s label table for)."""
    if isinstance(value, bool):  # bool is a subclass of int - exclude explicitly
        return None

    if isinstance(value, (int, float)):
        pct = value * 100 if 0 <= value <= 1 else value
        return max(0, min(100, round(pct)))

    if isinstance(value, str):
        match = re.search(r"\d+(\.\d+)?", value)
        if not match:
            return None
        return max(0, min(100, round(float(match.group()))))

    return None


def _retrieval_ceiling(retrieval_score: float) -> int:
    for threshold, ceiling in _RETRIEVAL_CEILINGS:
        if retrieval_score >= threshold:
            return ceiling
    return _RETRIEVAL_CEILINGS[-1][1]


def resolve_confidence(
    confidence_pct, confidence_label: str, verdict: str, retrieval_score: float | None = None
) -> int:
    """The single place confidence_pct actually gets decided: start from an
    explicit number if one was given, else fall back to the label
    ("high"/"medium"/"low"); cap it by retrieval quality if we have a
    score; and finally, never let an "unresolved" verdict claim more than
    coin-flip confidence - the category is the honest signal there, not
    the number."""
    pct = parse_confidence_pct(confidence_pct)
    if pct is None:
        pct = _LABEL_CONFIDENCE_PCT.get(confidence_label, 50)

    if retrieval_score is not None:
        pct = min(pct, _retrieval_ceiling(retrieval_score))

    category, _ = categorize(verdict)
    if category == "unresolved":
        pct = min(pct, 50)

    return max(0, min(100, round(pct)))
