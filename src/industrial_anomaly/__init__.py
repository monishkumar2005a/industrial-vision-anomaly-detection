"""Industrial anomaly detection (PatchCore-style)."""

from .config import Config
from .detector import AnomalyDetector, Prediction

__all__ = ["AnomalyDetector", "Config", "Prediction"]
__version__ = "0.1.0"
