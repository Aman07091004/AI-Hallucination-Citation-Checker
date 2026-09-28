"""
Builds a labeled dataset for the classical ML baseline, using real text
from our 58-case corpus rather than needing a large external benchmark.

Why we built our own instead of downloading one: the LePhantomCite dataset
we found earlier (1,300 labeled examples) would be the ideal source, but
it's only distributed via Hugging Face, which isn't reachable from this
project's sandboxed environment. Rather than blocking on that, we use the
same core technique that paper itself uses to build its own dataset:
inject "wrong proposition" errors into real citations by pairing a claim
with a passage from a DIFFERENT, unrelated case - a real, well-established
technique for generating negative examples when you don't have naturally-
occurring labeled failures.

Honesty about scale: even generating several examples per case, this
dataset is small (dozens of rows, not thousands). That's fine for a
teaching exercise about ML mechanics (train/test splits, precision/recall,
feature engineering) but should NOT be read as "our classifier is
production-ready" - a real production system would need a dataset at
LePhantomCite's scale (1,300+ examples) to trust its metrics.
"""

import random
from dataclasses import dataclass, asdict
from pathlib import Path

from app import config
from app.ingestion.citations import dedupe_citations, extract_citations, get_citation_context, strip_self_reference
from app.ingestion.loader import load_document
from app.rag.corpus_index import build_registry, find_case_in_corpus
from app.rag.vectorstore import hybrid_query_case, ingest_case

RANDOM_SEED = 42


@dataclass
class TrainingExample:
    case_name: str
    citation: str
    claim: str
    passage: str
    label: int  # 1 = SUPPORTED (real pairing), 0 = MISMATCHED (wrong-case pairing)
    bm25_score: float  # captured at retrieval time, against that case's FULL chunk corpus
    semantic_score: float  # also captured at retrieval time


def build_dataset(brief_path: Path, negatives_per_case: int = 3, positives_per_case: int = 3) -> list[TrainingExample]:
    """For each matched citation in the brief:
    - POSITIVE examples: the claim paired with its own top-N genuinely
      retrieved passages (label=1) - these are cases we've already
      validated are on-topic via the hybrid retrieval work.
    - NEGATIVE examples: the same claim paired with passages sampled from
      OTHER, unrelated matched cases (label=0) - simulating the "real
      citation, wrong proposition" hallucination type.
    """
    random.seed(RANDOM_SEED)

    brief_doc = load_document(brief_path)
    citations = dedupe_citations(extract_citations(brief_doc.text))
    registry = build_registry()

    matched = []
    for c in citations:
        match = find_case_in_corpus(c.case_name, registry)
        if match.status == "MATCHED":
            matched.append((c, match.matched_file))

    # Ingest all matched cases up front so we can pull "other case" passages for negatives
    for c, file_path in matched:
        ingest_case(file_path, c.case_name)

    examples = []

    for c, file_path in matched:
        claimed_context = get_citation_context(brief_doc.text, c.char_offset)
        query_text = strip_self_reference(claimed_context, c.case_name, c.citation)

        # Positives: this case's own top retrieved passages
        own_hits = hybrid_query_case(file_path, query_text, top_k=positives_per_case)
        for hit in own_hits:
            examples.append(
                TrainingExample(
                    c.case_name, c.citation, claimed_context, hit["text"], label=1,
                    bm25_score=hit["bm25_score"], semantic_score=hit["semantic_score"],
                )
            )

        # Negatives: passages from OTHER cases, queried with THIS claim
        other_cases = [(oc, of) for oc, of in matched if oc.case_name != c.case_name]
        sampled_others = random.sample(other_cases, min(negatives_per_case, len(other_cases)))
        for other_c, other_file in sampled_others:
            other_hits = hybrid_query_case(other_file, query_text, top_k=1)
            if other_hits:
                examples.append(
                    TrainingExample(
                        c.case_name, c.citation, claimed_context, other_hits[0]["text"], label=0,
                        bm25_score=other_hits[0]["bm25_score"], semantic_score=other_hits[0]["semantic_score"],
                    )
                )

    return examples


def save_dataset_csv(examples: list[TrainingExample], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_name", "citation", "claim", "passage", "label", "bm25_score", "semantic_score"]
        )
        writer.writeheader()
        for ex in examples:
            writer.writerow(asdict(ex))


if __name__ == "__main__":
    brief_path = config.RAW_CORPUS_DIR / "White and Case.pdf"
    dataset = build_dataset(brief_path)

    positives = sum(1 for e in dataset if e.label == 1)
    negatives = sum(1 for e in dataset if e.label == 0)
    print(f"Built {len(dataset)} examples: {positives} positive (SUPPORTED), {negatives} negative (MISMATCHED)")

    out_path = config.PROCESSED_DIR / "ml_training_data.csv"
    save_dataset_csv(dataset, out_path)
    print(f"Saved to {out_path}")
