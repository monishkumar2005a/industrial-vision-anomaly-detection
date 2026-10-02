"""Turn patch distances into an image score and an anomaly map."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def image_score(patch_distances: np.ndarray, top_ratio: float = 0.05) -> float:
    """Mean of the top few patch distances (at least one patch)."""
    k = max(1, int(round(top_ratio * patch_distances.size)))
    top = np.partition(patch_distances, -k)[-k:]
    return float(top.mean())


def anomaly_map(
    patch_distances: np.ndarray,
    grid: tuple[int, int],
    out_size: tuple[int, int],
    sigma: float = 4.0,
) -> np.ndarray:
    """Resize the patch grid to the image size, then blur it."""
    grid_map = patch_distances.reshape(grid).astype(np.float32)
    h, w = out_size
    up = cv2.resize(grid_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(up, sigma=sigma) if sigma > 0 else up


def normalize_map(amap: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale with fixed bounds from calibration so normal parts stay cold.

    Min-max scaling every image on its own makes even a good part show a hot spot.
    """
    return np.clip((amap - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def overlay(image_rgb: np.ndarray, norm_map: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, heat, alpha, 0)


def hotspot_summary(norm_map: np.ndarray, level: float = 0.5) -> dict | None:
    """Locate the largest hot region of a normalised map (no LLM needed).

    Returns bounding box and size in image pixels, or None if nothing reaches ``level``.
    """
    mask = (norm_map >= level).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[i])
    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area_pct": round(100.0 * area / mask.size, 2),
    }
