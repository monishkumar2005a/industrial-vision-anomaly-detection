"""Fake MVTec-style data and an untrained backbone so tests run fast and offline."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from industrial_anomaly import AnomalyDetector, Config

SIZE = 128


def _normal_image(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32) / SIZE
    base = 90 + 60 * xx + 40 * yy
    stripes = 15 * np.sin(xx * 40)
    img = base + stripes + rng.normal(0, 3, (SIZE, SIZE))
    return np.clip(np.stack([img, img * 0.9, img * 0.8], -1), 0, 255).astype(np.uint8)


def _defect_image(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    img = _normal_image(rng)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    y, x = rng.integers(20, 90, 2)
    img[y : y + 24, x : x + 24] = 255
    mask[y : y + 24, x : x + 24] = 255
    return img, mask


def _write(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


@pytest.fixture(scope="session")
def mvtec_root(tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("mvtec")
    cat = root / "bottle"
    for i in range(16):
        _write(cat / "train" / "good" / f"{i:03d}.png", _normal_image(rng))
    for i in range(6):
        _write(cat / "test" / "good" / f"{i:03d}.png", _normal_image(rng))
    for i in range(6):
        img, mask = _defect_image(rng)
        _write(cat / "test" / "broken" / f"{i:03d}.png", img)
        _write(cat / "ground_truth" / "broken" / f"{i:03d}_mask.png", mask)
    return root


@pytest.fixture(scope="session")
def tiny_config() -> Config:
    return Config(
        pretrained=False,
        image_size=64,
        pca_components=16,
        n_train_images=None,
        calibration_fraction=0.25,
        batch_size=4,
        seed=0,
    )


@pytest.fixture(scope="session")
def fitted(mvtec_root, tiny_config) -> AnomalyDetector:
    import torch

    torch.manual_seed(0)  # weights are random without pretraining, so fix the seed
    det = AnomalyDetector(tiny_config, device="cpu")
    det.fit(mvtec_root / "bottle" / "train" / "good")
    return det
