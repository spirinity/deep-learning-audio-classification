"""
Build per-category (5x5) confusion matrices for the ESC-50 method comparison.

ESC-50 has 50 fine-grained classes that roll up into 5 broad categories
(Animals, Natural soundscapes, Human non-speech, Interior/domestic,
Exterior/urban). A 50x50 confusion matrix is unreadable, so each true/predicted
label is first mapped to its parent category and a compact 5x5 matrix is built
per method. This answers: does the model confuse whole categories, or only pick
the wrong class *inside* the correct category?

Reads the offline predictions written by evaluate_methods.py
(data/demo/comparison_predictions.csv) -- no model load, no GPU required.

Run:
    python confusion_matrix.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from classifier import ESC50_CATEGORIES, LABEL_TO_CATEGORY

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "data" / "demo" / "comparison_predictions.csv"
IMG_DIR = BASE_DIR / "imgs"

CATEGORY_ORDER = list(ESC50_CATEGORIES.keys())
SHORT_LABELS = {
    "Animals": "Animals",
    "Natural soundscapes": "Natural",
    "Human non-speech": "Human",
    "Interior/domestic": "Interior",
    "Exterior/urban": "Exterior",
}
METHOD_ORDER = ["Zero-Shot CLAP", "Proto-LC", "Supervised MLP"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_PATH,
        help="Path to comparison_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=IMG_DIR,
        help="Directory for the saved PNG figures",
    )
    return parser.parse_args()


def to_category(label: str) -> str:
    """Map a fine-grained ESC-50 label to its parent category."""
    return LABEL_TO_CATEGORY.get(str(label).strip(), "Unknown")


def category_confusion(df: pd.DataFrame) -> np.ndarray:
    """Return a 5x5 count matrix (rows = true category, cols = predicted)."""
    true_cat = df["target_label"].map(to_category)
    pred_cat = df["prediction_label"].map(to_category)
    return confusion_matrix(true_cat, pred_cat, labels=CATEGORY_ORDER)


def plot_matrix(ax, matrix: np.ndarray, title: str):
    """Draw a single annotated, row-normalized heatmap onto ax."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0
    )

    ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = range(len(CATEGORY_ORDER))
    short = [SHORT_LABELS[c] for c in CATEGORY_ORDER]
    ax.set_xticks(ticks, short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(ticks, short, fontsize=8)
    ax.set_xlabel("Predicted category", fontsize=9)
    ax.set_ylabel("True category", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)

    for i in range(len(CATEGORY_ORDER)):
        for j in range(len(CATEGORY_ORDER)):
            count = matrix[i, j]
            pct = normalized[i, j] * 100
            ax.text(
                j,
                i,
                f"{count}\n{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[i, j] > 0.5 else "#222",
            )


def print_summary(method: str, matrix: np.ndarray):
    """Print per-category recall (diagonal / row total) for quick analysis."""
    total = matrix.sum()
    correct = np.trace(matrix)
    overall = correct / total if total else 0.0
    print(f"\n[{method}] category-level accuracy: {overall * 100:.2f}% ({correct}/{total})")
    for i, cat in enumerate(CATEGORY_ORDER):
        row_total = matrix[i].sum()
        recall = matrix[i, i] / row_total if row_total else 0.0
        print(f"  {cat:<22} recall {recall * 100:5.1f}%  ({matrix[i, i]}/{row_total})")


def main():
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: {args.predictions}\n"
            "Run evaluate_methods.py first to generate it."
        )

    df = pd.read_csv(args.predictions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    methods += [m for m in df["method"].unique() if m not in methods]

    matrices = {}
    for method in methods:
        matrix = category_confusion(df[df["method"] == method])
        matrices[method] = matrix
        print_summary(method, matrix)

        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        plot_matrix(ax, matrix, f"{method} — per-category confusion")
        fig.tight_layout()
        slug = method.lower().replace(" ", "_").replace("-", "_")
        out_path = args.out_dir / f"confusion_{slug}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")

    n = len(matrices)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.6))
        if n == 1:
            axes = [axes]
        for ax, method in zip(axes, methods):
            plot_matrix(ax, matrices[method], method)
        fig.suptitle("ESC-50 per-category confusion matrix (5x5)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        combined = args.out_dir / "confusion_all.png"
        fig.savefig(combined, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved combined figure {combined}")


if __name__ == "__main__":
    main()
