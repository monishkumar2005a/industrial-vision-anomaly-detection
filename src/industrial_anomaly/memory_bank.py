"""Memory bank of normal patches: PCA to shrink them, FAISS to search them."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def greedy_coreset(features: np.ndarray, n_select: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center subsampling (the coreset idea from PatchCore). Returns row indices."""
    n = len(features)
    if n_select >= n:
        return np.arange(n)
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(n))]
    min_d = torch.cdist(x, x[selected[0]][None]).squeeze(1)
    for _ in range(n_select - 1):
        nxt = int(torch.argmax(min_d))
        selected.append(nxt)
        min_d = torch.minimum(min_d, torch.cdist(x, x[nxt][None]).squeeze(1))
    return np.asarray(selected)


class MemoryBank:
    def __init__(
        self, n_components: int = 128, coreset_ratio: float | None = None, seed: int = 42
    ) -> None:
        self.n_components = n_components
        self.coreset_ratio = coreset_ratio
        self.seed = seed
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.index: faiss.Index | None = None

    @property
    def size(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def fit(self, normal_features: np.ndarray) -> None:
        logger.info("Fitting PCA + index on features of shape %s", normal_features.shape)
        projected = self.pca.fit_transform(normal_features).astype("float32")
        if self.coreset_ratio is not None:
            n_select = max(1, int(len(projected) * self.coreset_ratio))
            keep = greedy_coreset(projected, n_select, self.seed)
            logger.info("Coreset: %d -> %d patches", len(projected), len(keep))
            projected = projected[keep]
        self.index = faiss.IndexFlatL2(self.n_components)
        self.index.add(np.ascontiguousarray(projected))

    def search(self, features: np.ndarray) -> np.ndarray:
        """Squared distance from each patch to its nearest normal patch."""
        if self.index is None:
            raise RuntimeError("MemoryBank is not fitted.")
        projected = np.ascontiguousarray(self.pca.transform(features).astype("float32"))
        distances, _ = self.index.search(projected, 1)
        return distances.ravel()

    def save(self, directory: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("MemoryBank is not fitted.")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, d / "pca.joblib")
        faiss.write_index(self.index, str(d / "index.faiss"))
        meta = {"n_components": self.n_components, "coreset_ratio": self.coreset_ratio}
        (d / "bank_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: str | Path) -> MemoryBank:
        d = Path(directory)
        meta = json.loads((d / "bank_meta.json").read_text())
        bank = cls(meta["n_components"], meta["coreset_ratio"])
        bank.pca = joblib.load(d / "pca.joblib")  # joblib uses pickle, only load models you trust
        bank.index = faiss.read_index(str(d / "index.faiss"))
        return bank
