"""Plots (Agg backend so it works without a display)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .evaluation import EvalResult  # noqa: E402


def plot_evaluation(result: EvalResult, out_path: str | Path) -> None:
    s, y, m = result.scores, result.labels, result.metrics
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    bins = np.linspace(s.min(), s.max(), 31)
    axes[0].hist(s[y == 0], bins=bins, alpha=0.6, color="tab:green", label="good")
    axes[0].hist(s[y == 1], bins=bins, alpha=0.6, color="tab:red", label="defect")
    if m.get("threshold") is not None:
        axes[0].axvline(m["threshold"], color="k", ls="--", label="calibrated threshold")
    axes[0].set(title=f"Score distribution - {result.category}", xlabel="Image anomaly score")
    axes[0].legend()

    fpr, tpr, _ = roc_curve(y, s)
    axes[1].plot(fpr, tpr, lw=2, label=f"AUROC {m['image_auroc']:.4f}")
    axes[1].plot([0, 1], [0, 1], "k:", lw=1)
    axes[1].set(
        title="ROC (image level)", xlabel="False positive rate", ylabel="True positive rate"
    )
    axes[1].legend(loc="lower right")

    if "tp" in m:
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        axes[2].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            axes[2].text(j, i, str(v), ha="center", va="center", fontsize=14)
        axes[2].set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Good", "Defect"],
            yticklabels=["Good", "Defect"],
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion @ calibrated threshold (F1 {m['f1']:.3f})",
        )
    else:
        axes[2].axis("off")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_side_by_side(
    original: np.ndarray, overlay_img: np.ndarray, title: str, out_path: str | Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[1].imshow(overlay_img)
    axes[1].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
