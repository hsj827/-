from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import Detector, ScoreMap
from .crack import components
from .template import Template


class MissingDetector(Detector):
    """Geometric detector: material absent from (or added to) the good outline."""

    name = "missing"

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        self.template = Template()

    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None:
        self.template.build(images, masks)
        self.mark_fitted()

    def pixel_reference(self) -> float | None:
        return float(self.template.tolerance_px)

    def score_map(self, image: np.ndarray, mask: np.ndarray) -> ScoreMap:
        dilated, eroded = self.template.contour(self.cfg.get("tolerance") or None)
        current = mask.astype(bool)

        lost = eroded & ~cv2.dilate(current.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        extra = current & ~dilated

        depth_lost = cv2.distanceTransform(lost.astype(np.uint8), cv2.DIST_L2, 3)
        depth_extra = cv2.distanceTransform(extra.astype(np.uint8), cv2.DIST_L2, 3)
        raw = np.maximum(depth_lost, depth_extra).astype(np.float32)

        keep = components(raw > 0, int(self.cfg.get("min_area", 60)),
                          int(self.cfg.get("max_area", 100000)))
        score = raw * (self.template.tolerance_px + keep.astype(np.float32))
        return ScoreMap(
            source=self.name,
            score=score.astype(np.float32),
            image_score=float(keep.sum()) if keep.any() else 0.0,
            mask=keep,
            meta={"lost_px": int(lost.sum()), "extra_px": int(extra.sum())},
        )

    def state(self) -> dict[str, Any]:
        return self.template.state()

    def restore(self, state: dict[str, Any]) -> None:
        self.template.restore(state)
        self.mark_fitted()
