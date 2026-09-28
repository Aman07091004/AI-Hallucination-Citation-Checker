"""
The classical ML baseline: given a (claim, passage) pair, predict whether
the passage actually supports the claim - the same question llm_judge.py
answers with an LLM, but here answered with a small, fast, free logistic
regression model trained on numeric features.

This is deliberately NOT trying to out-perform the LLM. The point is to
understand the fundamentals: what features would even help distinguish a
supported claim from a mismatched one, how to properly split data for
training vs evaluation, and how to read precision/recall/F1 rather than
just accuracy - plus a genuine, honest discussion of when a classical
model like this is actually useful in a real pipeline (hint: as a cheap
pre-filter, not a replacement for LLM judgment on hard cases).

Important fix baked into this design: bm25_score and semantic_score are
NOT recomputed here from an isolated (claim, passage) pair. An early
version did that and produced a real bug - BM25 needs proper document
frequency statistics from a real corpus to compute a meaningful score;
computing it fresh against a "corpus" of exactly one passage makes its
IDF term degenerate, and a genuinely relevant passage scored WORSE
(-2.43) than a totally unrelated one (0.0) in a direct test. Both scores
are instead captured once, correctly, at retrieval time in
build_training_data.py, where they're computed against each case's full
multi-chunk corpus - exactly like the real hybrid_query_case() function
already does elsewhere in this project.
"""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from app import config


def _word_overlap_ratio(claim: str, passage: str) -> float:
    """Simple lexical feature: what fraction of the claim's distinctive
    words also appear in the passage. Cheap to compute, no model needed -
    a good baseline feature precisely because it's easy to reason about,
    and unlike BM25, is well-defined for a single pair (it's just set
    overlap, no corpus-wide statistics involved)."""
    claim_words = set(re.findall(r"[a-z]{4,}", claim.lower()))  # 4+ letters - skip short filler words
    passage_words = set(re.findall(r"[a-z]{4,}", passage.lower()))
    if not claim_words:
        return 0.0
    return len(claim_words & passage_words) / len(claim_words)


def extract_features(example) -> dict:
    """Builds the feature vector for one training example. bm25_score and
    semantic_score come pre-computed from retrieval time (see module
    docstring for why); word_overlap_ratio and lengths are computed fresh
    here since they're well-defined per-pair."""
    return {
        "bm25_score": example.bm25_score,
        "semantic_score": example.semantic_score,
        "word_overlap_ratio": _word_overlap_ratio(example.claim, example.passage),
        "claim_length": len(example.claim.split()),
        "passage_length": len(example.passage.split()),
    }


def build_feature_matrix(examples: list, feature_subset: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [extract_features(ex) for ex in examples]
    feature_names = sorted(rows[0].keys())
    if feature_subset is not None:
        feature_names = [name for name in feature_names if name in feature_subset]
    X = np.array([[row[name] for name in feature_names] for row in rows])
    y = np.array([ex.label for ex in examples])
    return X, y, feature_names


def train_and_evaluate(examples: list, feature_subset: list[str] | None = None) -> dict:
    """Trains logistic regression with a stratified train/test split, and
    also evaluates a naive fixed-threshold baseline for comparison - so
    the value of actually training a model (vs eyeballing a cutoff) is
    visible in the numbers, not just assumed."""
    X, y, feature_names = build_feature_matrix(examples, feature_subset=feature_subset)

    # With a small dataset, a large test split (30%) still leaves enough
    # in each set to be meaningful, and stratify= keeps the class balance
    # consistent between train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Features used: {feature_names}")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}\n")
    print("Logistic Regression results:")
    print(classification_report(y_test, predictions, target_names=["MISMATCHED", "SUPPORTED"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, predictions))

    # Naive baseline: a single hand-picked threshold on the best available
    # relevance signal (properly-computed semantic_score this time), no
    # training at all. If the trained model isn't clearly better than
    # this, that's a real, honest finding worth reporting, not hiding.
    idx = feature_names.index("semantic_score")
    naive_predictions = (X_test[:, idx] > np.median(X[:, idx])).astype(int)
    print(f"\nNaive threshold baseline (median split on semantic_score):")
    print(classification_report(y_test, naive_predictions, target_names=["MISMATCHED", "SUPPORTED"]))

    coefficients = dict(zip(feature_names, model.coef_[0]))
    print(f"\nLearned feature weights: {coefficients}")

    return {
        "model": model,
        "feature_names": feature_names,
        "test_accuracy": model.score(X_test, y_test),
        "y_test": y_test,
        "predictions": predictions,
        "naive_predictions": naive_predictions,
    }


def summarize_features_by_class(examples: list) -> None:
    """Basic exploratory data analysis: look at the data before trusting
    any model's conclusions about it. Specifically checks whether BM25 and
    semantic score actually separate the two classes at all in the raw
    data, or whether a model's learned coefficient sign is more likely an
    overfitting artifact of a small training split."""
    X, y, feature_names = build_feature_matrix(examples)
    print("Per-class feature summary (mean, then median):")
    for i, name in enumerate(feature_names):
        supported_vals = X[y == 1, i]
        mismatched_vals = X[y == 0, i]
        print(
            f"  {name:20} SUPPORTED: mean={supported_vals.mean():.3f} median={np.median(supported_vals):.3f}  |  "
            f"MISMATCHED: mean={mismatched_vals.mean():.3f} median={np.median(mismatched_vals):.3f}"
        )


def cross_validate(examples: list, n_splits: int = 5, feature_subset: list[str] | None = None) -> np.ndarray:
    """A single train/test split on a small dataset has high variance -
    which specific rows land in the test set can swing accuracy by a lot
    just from luck. K-fold cross-validation trains and evaluates n_splits
    times on different partitions and reports the average, which is a far
    more trustworthy estimate of real performance than one split's number -
    especially important here given we only have ~42 examples total."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y, feature_names = build_feature_matrix(examples, feature_subset=feature_subset)
    n_splits = min(n_splits, min(np.bincount(y)))  # can't have more folds than the smallest class has examples

    model = LogisticRegression()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f"\n{n_splits}-fold cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Individual fold scores: {[round(s, 3) for s in scores]}")
    print(
        "This is a more trustworthy estimate than the single train/test split above - "
        "note how much it varies fold to fold, which is exactly the small-sample-size risk."
    )
    return scores


if __name__ == "__main__":
    import csv
    from app.ml.build_training_data import TrainingExample

    dataset_path = config.PROCESSED_DIR / "ml_training_data.csv"
    if not dataset_path.exists():
        print(f"No dataset found at {dataset_path} - run 'python -m app.ml.build_training_data' first.")
    else:
        examples = []
        with open(dataset_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                examples.append(
                    TrainingExample(
                        row["case_name"], row["citation"], row["claim"], row["passage"],
                        label=int(row["label"]),
                        bm25_score=float(row["bm25_score"]),
                        semantic_score=float(row["semantic_score"]),
                    )
                )
        train_and_evaluate(examples)
        print()
        summarize_features_by_class(examples)
        cross_validate(examples)
