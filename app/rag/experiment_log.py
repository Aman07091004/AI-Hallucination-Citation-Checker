"""
Logs every retrieval result to a CSV file, tagged with a "stage" label
(e.g. "pure_semantic", "hybrid_with_leakage", "hybrid_clean_query"), so
different versions of the pipeline can be compared side by side at the end
of the project instead of only existing as scrollback in a terminal.

Usage from vectorstore.py's __main__: pass a stage name as a command-line
argument each time you run an experiment you want to keep, e.g.:

    python -m app.rag.vectorstore hybrid_clean_query

If no stage name is given, results are logged under "unlabeled_run" - fine
for quick checks, but give it a real name whenever you want the run to
count toward your final comparison.
"""

import csv
from datetime import datetime
from pathlib import Path

from app import config

LOG_PATH: Path = config.PROCESSED_DIR / "retrieval_experiments.csv"

FIELDNAMES = [
    "timestamp",
    "stage",
    "case_name",
    "citation",
    "rank",
    "blended_score",
    "semantic_score",
    "bm25_score",
    "passage_preview",
]


def log_hit(stage: str, case_name: str, citation: str, rank: int, hit: dict) -> None:
    """Appends one retrieved passage's scores as a row. `hit` can come from
    either query_case() (has 'similarity') or hybrid_query_case() (has
    'blended_score' / 'semantic_score' / 'bm25_score') - whichever keys are
    missing are just left blank in that row, so old and new-style results
    can sit in the same file without breaking anything."""
    is_new_file = not LOG_PATH.exists()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new_file:
            writer.writeheader()

        preview = hit.get("text", "")[:200].replace("\n", " ").replace("\r", " ")
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "case_name": case_name,
                "citation": citation,
                "rank": rank,
                "blended_score": hit.get("blended_score", ""),
                "semantic_score": hit.get("semantic_score", hit.get("similarity", "")),
                "bm25_score": hit.get("bm25_score", ""),
                "passage_preview": preview,
            }
        )


def summarize_by_stage() -> None:
    """Quick console summary: average scores per stage, so you can eyeball
    whether a given stage was an improvement without opening the CSV."""
    if not LOG_PATH.exists():
        print("No experiment log yet - run a stage first.")
        return

    from collections import defaultdict

    stage_scores = defaultdict(list)
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = row["blended_score"] or row["semantic_score"]
            if score:
                stage_scores[row["stage"]].append(float(score))

    print(f"\n{'Stage':30} {'Avg score':10} {'# passages'}")
    print("-" * 55)
    for stage, scores in stage_scores.items():
        print(f"{stage:30} {sum(scores)/len(scores):<10.3f} {len(scores)}")


def export_comparison_csv(stage_order: list[str] | None = None) -> Path:
    """Pivots the raw long-format log into a wide comparison table: one row
    per (case, rank), one column per stage, holding just the score. This is
    the "look at everything side by side" view - the raw log stays in its
    original append-friendly long format so future runs can keep adding to
    it safely, and this pivot is regenerated fresh from it on demand.

    Score choice per row: blended_score if the stage had one (hybrid runs),
    otherwise semantic_score (the only score a pure-semantic run produced) -
    i.e. always the stage's own "primary" ranking score.
    """
    import pandas as pd

    df = pd.read_csv(LOG_PATH)
    df["score"] = df["blended_score"].where(df["blended_score"].notna(), df["semantic_score"])

    # Preserve first-seen case order (matches citation order in the brief)
    # rather than alphabetical, so the table reads in a natural sequence.
    case_order = list(dict.fromkeys(df["case_name"]))
    df["case_name"] = pd.Categorical(df["case_name"], categories=case_order, ordered=True)

    pivot = df.pivot_table(index=["case_name", "rank"], columns="stage", values="score", aggfunc="first")
    pivot = pivot.reset_index().sort_values(["case_name", "rank"])

    # Order stage columns as requested/observed, rather than alphabetically.
    all_stages = list(pivot.columns[2:])
    if stage_order:
        ordered = [s for s in stage_order if s in all_stages] + [s for s in all_stages if s not in stage_order]
    else:
        ordered = all_stages
    pivot = pivot[["case_name", "rank"] + ordered]

    comparison_path = LOG_PATH.parent / "retrieval_comparison.csv"
    pivot.to_csv(comparison_path, index=False)
    return comparison_path


if __name__ == "__main__":
    summarize_by_stage()
    path = export_comparison_csv(stage_order=["pure_semantic", "hybrid_with_leakage", "hybrid_clean_query"])
    print(f"\nWide comparison table written to {path}")
