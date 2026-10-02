"""Wide-ResNet50 patch features (layers 2 and 3)."""

from __future__ import annotations

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from .config import Config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """Turns RGB images into one feature vector per patch.

    The patch grid size is returned from extract() instead of being stored on the object.
    """

    def __init__(self, cfg: Config, device: str | torch.device | None = None) -> None:
        self.cfg = cfg
        self.device = resolve_device(device)
        self.model = (
            timm.create_model(
                cfg.backbone,
                pretrained=cfg.pretrained,
                features_only=True,
                out_indices=cfg.out_indices,
            )
            .to(self.device)
            .eval()
        )
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((cfg.image_size, cfg.image_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _aggregate(self, fmap: torch.Tensor) -> torch.Tensor:
        k = self.cfg.local_aggregation
        if k <= 1:
            return fmap
        return F.avg_pool2d(fmap, kernel_size=k, stride=1, padding=k // 2)

    @torch.inference_mode()
    def extract(self, images: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
        """Return (features [B*H*W, C] float32, (H, W) patch grid)."""
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        maps = [self._aggregate(m) for m in self.model(batch)]
        target = maps[0].shape[-2:]
        maps = [maps[0]] + [
            F.interpolate(m, size=target, mode="bilinear", align_corners=False) for m in maps[1:]
        ]
        feat = torch.cat(maps, dim=1)  # [B, C, H, W]
        b, c, h, w = feat.shape
        flat = feat.permute(0, 2, 3, 1).reshape(b * h * w, c)
        return flat.float().cpu().numpy(), (h, w)
