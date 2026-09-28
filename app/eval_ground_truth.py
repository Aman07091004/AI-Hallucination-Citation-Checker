"""
Checks the pipeline's output on the sample brief against the hand-verified
answer key built at the start of this project (each citation independently
fact-checked, the three fabrications confirmed by web search).

Run after ANY change to prompts, retrieval, extraction or context windows:

    python -m app.eval_ground_truth

If a change moves this score, you find out immediately, rather than
discovering it later in a demo.
"""

from pathlib import Path

from app import config
from app.verify_brief import verify_document

# citation string -> (expected category, expected depth or None if not applicable)
GROUND_TRUTH = {
    "(1853) 2 E&B 216":        ("verified", "full"),            # Lumley v Gye
    "[2007] UKHL 21":          ("verified", "full"),            # OBG v Allan
    "[1952] Ch 646":           ("verified", "full"),            # DC Thomson v Deakin
    "[2019] EWHC 1847 (Comm)": ("fabricated", None),            # Fairfax v Brennan Holdings
    "(1854) 9 Ex 341":         ("verified", "full"),            # Hadley v Baxendale
    "[2008] UKHL 48":          ("misapplied", "full"),          # Transfield v Mercator (judgment call, kept deliberately)
    "[1972] 1 QB 60":          ("verified", "full"),            # Anglia Television v Reed
    "[2021] EWHC 3312 (Ch)":   ("fabricated", None),            # Stonegate v Redwood
    "[1974] 1 WLR 798":        ("verified", "existence_only"),  # Wrotham Park - real, not in corpus
    "[1975] AC 396":           ("verified", "full"),            # American Cyanamid v Ethicon
    "[1996] 1 All ER 853":     ("verified", "existence_only"),  # Series 5 Software - real, not in corpus
    "[2023] EWHC 892 (TCC)":   ("fabricated", None),            # Pemberton v Delta Global
}


def evaluate(report: list[dict]) -> tuple[int, list[str]]:
    by_citation = {e["citation"]: e for e in report}
    correct = 0
    lines = []
    for citation, (want_cat, want_depth) in GROUND_TRUTH.items():
        got = by_citation.get(citation)
        if got is None:
            lines.append(f"  MISSING  {citation}  (expected {want_cat})")
            continue
        ok = got["category"] == want_cat and (want_depth is None or got["depth"] == want_depth)
        correct += ok
        mark = "ok     " if ok else "WRONG  "
        lines.append(f"  {mark} {citation:26} expected {want_cat:11} got {got['category']:11} {got['confidence_pct']:>3}%  ({got['verdict']})")
    return correct, lines


if __name__ == "__main__":
    report = verify_document(config.RAW_CORPUS_DIR / "White and Case.pdf")
    correct, lines = evaluate(report)
    print("\n".join(lines))
    print(f"\nGround truth: {correct}/{len(GROUND_TRUTH)} correct")

    extra = [e for e in report if e["citation"] not in GROUND_TRUTH]
    if extra:
        print(f"\nAuthorities found beyond the answer key ({len(extra)}):")
        for e in extra:
            print(f"  {e['kind']:11} {e['citation']}  -> {e['category']}")
