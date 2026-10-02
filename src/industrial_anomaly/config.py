"""All the knobs in one place (these used to be the Colab #@param cells)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # Backbone
    backbone: str = "wide_resnet50_2"
    out_indices: tuple[int, ...] = (2, 3)
    pretrained: bool = True
    image_size: int = 224
    local_aggregation: int = 1  # kxk neighbour pooling from the PatchCore paper, 1 = off

    # Memory bank
    pca_components: int = 128
    n_train_images: int | None = 50  # None = use every training image
    coreset_ratio: float | None = None  # e.g. 0.1 keeps 10% of patches, None = keep all

    # Scoring
    anomaly_patch_ratio: float = 0.05  # image score = mean of the top 5% patch distances
    smoothing_sigma: float = 4.0  # blur applied after upsampling the map

    # Threshold calibration (uses held-out normal images)
    calibration_fraction: float = 0.2
    threshold_sigma: float = 3.0  # threshold = mean + k * std of the held-out scores

    # Runtime
    batch_size: int = 8
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["out_indices"] = list(self.out_indices)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = dict(d)
        d["out_indices"] = tuple(d.get("out_indices", (2, 3)))
        return cls(**d)
