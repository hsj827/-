from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import Detector, ScoreMap
from .crack import components
from .template import Template


class StainDetector(Detector):
    """Local grey-level deviation from the good median, scaled by good-image spread."""

    name = "stain"

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        self.template = Template()

    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None:
        self.template.build(images, masks)
        self.mark_fitted()

    def pixel_reference(self) -> float | None:
        return float(self.cfg.get("z", 4.0))

    def score_map(self, image: np.ndarray, mask: np.ndarray) -> ScoreMap:
        window = int(self.cfg.get("window", 15))
        interior = self.template.interior(int(self.cfg.get("interior_erode", 3)))
        residual = np.abs(image - self.template.image).astype(np.float32)
        if window > 1:
            residual = cv2.GaussianBlur(residual, (0, 0), window / 3.0)
        z = residual / np.maximum(self.template.residual_sigma, 1e-3)
        z = z * interior
        keep = components(z > float(self.cfg.get("z", 4.0)),
                          int(self.cfg.get("min_area", 20)),
                          int(self.cfg.get("max_area", 20000)))
        score = z * keep
        return ScoreMap(
            source=self.name,
            score=score.astype(np.float32),
            image_score=float(z[keep].max()) if keep.any() else 0.0,
            mask=keep,
            meta={"keep_px": int(keep.sum())},
        )

    def state(self) -> dict[str, Any]:
        return self.template.state()

    def restore(self, state: dict[str, Any]) -> None:
        self.template.restore(state)
        self.mark_fitted()
