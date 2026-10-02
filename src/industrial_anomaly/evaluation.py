"""Image-level and pixel-level metrics on the MVTec test split."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .data import category_dirs, list_images, load_mask
from .detector import AnomalyDetector

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    category: str
    metrics: dict
    scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)


def evaluate_category(detector: AnomalyDetector, root: str | Path, category: str) -> EvalResult:
    dirs = category_dirs(root, category)
    items: list[tuple[Path, str]] = []
    for defect_dir in sorted(p for p in dirs["test"].iterdir() if p.is_dir()):
        items += [(p, defect_dir.name) for p in list_images(defect_dir)]
    if not items:
        raise FileNotFoundError(f"No test images under {dirs['test']}")

    size = detector.cfg.image_size
    scores, labels, defects = [], [], []
    pix_scores, pix_masks, have_all_masks = [], [], True

    paths = [p for p, _ in items]
    for pred, (path, defect) in zip(detector.predict_paths(paths), items, strict=True):
        is_defect = defect != "good"
        scores.append(pred.score)
        labels.append(int(is_defect))
        defects.append(defect)

        h, w = pred.anomaly_map.shape
        if is_defect:
            mask = load_mask(dirs["gt"], defect, path)
            if mask is None:
                have_all_masks = False
                continue
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        pix_masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).ravel())
        pix_scores.append(cv2.resize(pred.anomaly_map, (size, size)).astype(np.float32).ravel())

    s, y = np.asarray(scores), np.asarray(labels)
    metrics: dict = {
        "category": category,
        "n_good": int((y == 0).sum()),
        "n_defect": int((y == 1).sum()),
        "image_auroc": float(roc_auc_score(y, s)),
        "image_ap": float(average_precision_score(y, s)),
        "pixel_auroc": None,
        "threshold": detector.threshold,
    }
    if have_all_masks and pix_masks:
        m = np.concatenate(pix_masks)
        if 0 < m.sum() < m.size:
            metrics["pixel_auroc"] = float(roc_auc_score(m, np.concatenate(pix_scores)))

    if detector.threshold is not None:
        pred_y = (s > detector.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred_y, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics.update(
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
            precision=float(precision),
            recall=float(recall),
            f1=float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            false_alarm_rate=float(fp / (fp + tn)) if fp + tn else 0.0,
            miss_rate=float(fn / (fn + tp)) if fn + tp else 0.0,
        )

    by_defect = {
        d: float(np.mean([v for v, dd in zip(s, defects, strict=True) if dd == d]))
        for d in sorted(set(defects))
    }
    metrics["mean_score_by_type"] = by_defect
    return EvalResult(category=category, metrics=metrics, scores=s, labels=y)
