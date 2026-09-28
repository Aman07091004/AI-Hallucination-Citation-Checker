"""
Saves the classical ML baseline's evaluation results as PNG images, so
they're easy to drop into a portfolio writeup, a slide, or PROGRESS.md
without re-running code or copy-pasting terminal output.

Uses matplotlib's non-interactive 'Agg' backend explicitly, since this
runs from the command line with no display available - without this, plot
generation can fail or hang on some systems trying to open a window.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from app import config

RESULTS_DIR = config.PROCESSED_DIR / "ml_results"


def save_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["MISMATCHED", "SUPPORTED"], yticklabels=["MISMATCHED", "SUPPORTED"], ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / filename}")


def save_cv_score_comparison(scores_by_label: dict[str, np.ndarray], filename: str = "cross_validation_scores.png") -> None:
    """scores_by_label e.g. {'Full features': array([...]), 'Trimmed features': array([...])} -
    shows each fold's score as a point plus the mean, so the fold-to-fold
    variance (the actual point of cross-validation) is visible, not just
    a single averaged bar that would hide it."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = range(len(scores_by_label))
    labels = list(scores_by_label.keys())

    for i, (label, scores) in enumerate(scores_by_label.items()):
        jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(scores))
        ax.scatter([i + j for j in jitter], scores, alpha=0.7, s=60, label="Individual folds" if i == 0 else None)
        ax.hlines(scores.mean(), i - 0.2, i + 0.2, colors="black", linewidth=2)

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Chance level (0.5)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("Cross-validation accuracy: fold-to-fold variance")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / filename}")


def save_feature_class_comparison(examples: list, filename: str = "feature_class_comparison.png") -> None:
    """Bar chart version of summarize_features_by_class() - visualizes
    whether each feature actually separates SUPPORTED from MISMATCHED in
    the raw data, which is what explains (or fails to explain) a trained
    model's coefficients."""
    from app.ml.baseline_classifier import build_feature_matrix

    X, y, feature_names = build_feature_matrix(examples)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    supported_means = [X[y == 1, i].mean() for i in range(len(feature_names))]
    mismatched_means = [X[y == 0, i].mean() for i in range(len(feature_names))]

    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, supported_means, width, label="SUPPORTED", color="#4c72b0")
    ax.bar(x + width / 2, mismatched_means, width, label="MISMATCHED", color="#c44e52")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.set_ylabel("Mean value")
    ax.set_title("Feature means by class - does this feature actually separate the classes?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"Saved {RESULTS_DIR / filename}")


if __name__ == "__main__":
    import csv

    from app.ml.baseline_classifier import cross_validate, train_and_evaluate
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
                        label=int(row["label"]), bm25_score=float(row["bm25_score"]), semantic_score=float(row["semantic_score"]),
                    )
                )

        result = train_and_evaluate(examples)
        save_confusion_matrix(result["y_test"], result["predictions"], "Logistic Regression - Full Features", "confusion_matrix_full.png")
        save_confusion_matrix(result["y_test"], result["naive_predictions"], "Naive Threshold Baseline", "confusion_matrix_naive.png")

        meaningful_features = ["bm25_score", "semantic_score", "word_overlap_ratio"]
        trimmed_result = train_and_evaluate(examples, feature_subset=meaningful_features)
        save_confusion_matrix(
            trimmed_result["y_test"], trimmed_result["predictions"], "Logistic Regression - Trimmed Features", "confusion_matrix_trimmed.png"
        )

        full_cv = cross_validate(examples)
        trimmed_cv = cross_validate(examples, feature_subset=meaningful_features)
        save_cv_score_comparison({"Full features": full_cv, "Trimmed features": trimmed_cv})

        save_feature_class_comparison(examples)

        print(f"\nAll charts saved to {RESULTS_DIR}")
