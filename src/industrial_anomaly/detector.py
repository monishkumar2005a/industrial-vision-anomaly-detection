"""Main class: fit, calibrate, predict, save, load."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .data import iter_batches, list_images
from .features import FeatureExtractor
from .memory_bank import MemoryBank
from .scoring import anomaly_map, image_score, normalize_map, overlay

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    score: float
    is_anomalous: bool | None  # None if there's no threshold yet
    anomaly_map: np.ndarray  # smoothed patch distances at original image size
    path: Path | None = None


class AnomalyDetector:
    def __init__(self, cfg: Config | None = None, device: str | torch.device | None = None) -> None:
        self.cfg = cfg or Config()
        self.device = device
        self.extractor = FeatureExtractor(self.cfg, device)
        self.bank = MemoryBank(self.cfg.pca_components, self.cfg.coreset_ratio, self.cfg.seed)
        self.threshold: float | None = None
        self.map_lo: float | None = None
        self.map_hi: float | None = None

    def fit(self, train_dir: str | Path) -> AnomalyDetector:
        paths = list_images(train_dir)
        if self.cfg.n_train_images is not None:
            paths = paths[: self.cfg.n_train_images]
        if not paths:
            raise FileNotFoundError(f"No training images in {train_dir}")

        # Keep a few normal images out of the memory bank. Anything inside the bank scores
        # ~0, so we need unseen normal images to calibrate the threshold.
        n_cal = int(round(len(paths) * self.cfg.calibration_fraction))
        if self.cfg.calibration_fraction > 0 and n_cal < 2:
            raise ValueError(
                "Calibration needs >= 2 held-out images; use more data or a larger fraction."
            )
        order = np.random.default_rng(self.cfg.seed).permutation(len(paths))
        cal_paths = [paths[i] for i in sorted(order[:n_cal])]
        fit_paths = [paths[i] for i in sorted(order[n_cal:])]

        feats = []
        for _, imgs in tqdm(
            iter_batches(fit_paths, self.cfg.batch_size),
            total=-(-len(fit_paths) // self.cfg.batch_size),
            desc="Building memory bank",
        ):
            feats.append(self.extractor.extract(imgs)[0])
        self.bank.fit(np.vstack(feats))

        if cal_paths:
            self._calibrate(cal_paths)
        return self

    def _calibrate(self, cal_paths: list[Path]) -> None:
        scores, lo, hi = [], np.inf, -np.inf
        for pred in self.predict_paths(cal_paths, show_progress=False):
            scores.append(pred.score)
            lo, hi = min(lo, float(pred.anomaly_map.min())), max(hi, float(pred.anomaly_map.max()))
        s = np.asarray(scores)
        self.threshold = float(s.mean() + self.cfg.threshold_sigma * s.std(ddof=1))
        self.map_lo, self.map_hi = lo, hi
        logger.info(
            "Calibrated on %d normal images: threshold=%.4f (mean %.4f, std %.4f)",
            len(s),
            self.threshold,
            s.mean(),
            s.std(ddof=1),
        )

    def predict_arrays(self, images: Sequence[np.ndarray]) -> list[Prediction]:
        out: list[Prediction] = []
        bs = self.cfg.batch_size
        for i in range(0, len(images), bs):
            chunk = list(images[i : i + bs])
            feats, grid = self.extractor.extract(chunk)
            dists = self.bank.search(feats).reshape(len(chunk), -1)
            for img, d in zip(chunk, dists, strict=True):
                h, w = img.shape[:2]
                score = image_score(d, self.cfg.anomaly_patch_ratio)
                amap = anomaly_map(d, grid, (h, w), self.cfg.smoothing_sigma)
                flag = None if self.threshold is None else score > self.threshold
                out.append(Prediction(score=score, is_anomalous=flag, anomaly_map=amap))
        return out

    def predict_paths(
        self, paths: Sequence[str | Path], show_progress: bool = True
    ) -> Iterator[Prediction]:
        paths = [Path(p) for p in paths]
        batches = iter_batches(paths, self.cfg.batch_size)
        if show_progress:
            batches = tqdm(batches, total=-(-len(paths) // self.cfg.batch_size), desc="Inference")
        for chunk, imgs in batches:
            for p, pred in zip(chunk, self.predict_arrays(imgs), strict=True):
                pred.path = p
                yield pred

    def normalized_map(self, pred: Prediction) -> np.ndarray:
        lo = self.map_lo if self.map_lo is not None else float(pred.anomaly_map.min())
        hi = self.map_hi if self.map_hi is not None else float(pred.anomaly_map.max())
        return normalize_map(pred.anomaly_map, lo, hi)

    def render(self, image_rgb: np.ndarray, pred: Prediction) -> np.ndarray:
        return overlay(image_rgb, self.normalized_map(pred))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.bank.save(d)
        meta = {
            "config": self.cfg.to_dict(),
            "threshold": self.threshold,
            "map_lo": self.map_lo,
            "map_hi": self.map_hi,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(
        cls, directory: str | Path, device: str | torch.device | None = None
    ) -> AnomalyDetector:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        det = cls(Config.from_dict(meta["config"]), device)
        det.bank = MemoryBank.load(d)
        det.threshold, det.map_lo, det.map_hi = meta["threshold"], meta["map_lo"], meta["map_hi"]
        return det
