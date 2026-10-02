"""Helpers for the MVTec AD folder layout."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def list_images(directory: str | Path) -> list[Path]:
    """
    Return image files recursively from a directory.

    This supports the MVTec AD layout where test images are
    stored inside defect-category subdirectories.
    """

    directory = Path(directory)

    if directory.is_file():
        if directory.suffix.lower() in IMAGE_EXTENSIONS:
            return [directory]
        return []

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB. cv2.imread returns None on failure, so raise instead."""

    bgr = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise ValueError(f"Could not read image: {path}")

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def iter_batches(
    paths: list[Path],
    batch_size: int,
) -> Iterator[tuple[list[Path], list[np.ndarray]]]:
    """
    Yield images in batches so we never hold
    a whole category in RAM.
    """

    for i in range(
        0,
        len(paths),
        batch_size,
    ):
        chunk = paths[i : i + batch_size]

        yield (
            chunk,
            [load_rgb(path) for path in chunk],
        )


def category_dirs(
    root: str | Path,
    category: str,
) -> dict[str, Path]:

    base = Path(root) / category

    return {
        "train": base / "train" / "good",
        "test": base / "test",
        "gt": base / "ground_truth",
    }


def load_mask(
    gt_dir: Path,
    defect: str,
    image_path: Path,
) -> np.ndarray | None:
    """
    Ground-truth mask for a defective test image.

    Returns None if there isn't one.
    """

    mask_path = gt_dir / defect / f"{image_path.stem}_mask.png"

    if not mask_path.exists():
        return None

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    return None if mask is None else (mask > 0).astype(np.uint8)
