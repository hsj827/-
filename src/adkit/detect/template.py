from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class Template:
    """Median-of-goods reference image and mask. Everything classical compares against."""

    def __init__(self) -> None:
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.residual_sigma: np.ndarray | None = None
        self.tolerance_px: int = 3
        self.n: int = 0

    def build(self, images: list[np.ndarray], masks: list[np.ndarray]) -> "Template":
        if not images:
            raise ValueError("template needs at least one good image")
        stack = np.stack(images).astype(np.float32)
        mask_stack = np.stack(masks).astype(np.float32)
        self.image = np.median(stack, axis=0).astype(np.float32)
        self.mask = (mask_stack.mean(axis=0) >= 0.5).astype(bool)
        if len(images) > 2:
            spread = np.median(np.abs(stack - self.image[None]), axis=0)
            self.residual_sigma = cv2.GaussianBlur(spread, (0, 0), 2.0) + 1e-3
        else:
            self.residual_sigma = np.full_like(self.image, 0.02, dtype=np.float32)
        self.tolerance_px = self._calibrate_tolerance(mask_stack)
        self.n = len(images)
        return self

    def _calibrate_tolerance(self, mask_stack: np.ndarray, coverage: float = 0.97) -> int:
        """Smallest dilation of the template that still contains the goods' own outlines."""
        base = self.mask.astype(np.uint8) * 255
        limit = 1.0 - coverage
        for radius in range(1, 13):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            dilated = cv2.dilate(base, kernel) > 0
            if all(float(((m > 0.5) & ~dilated).mean()) <= limit for m in mask_stack):
                return radius
        return 12

    def contour(self, tolerance: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(dilated_template_mask, eroded_template_mask) for boundary-difference tests."""
        return self._band(self.mask, tolerance)

    def _band(self, mask: np.ndarray, tolerance: int | None) -> tuple[np.ndarray, np.ndarray]:
        base = (mask.astype(np.uint8)) * 255
        k = int(self.tolerance_px if tolerance in (None, 0) else tolerance)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        return cv2.dilate(base, kernel) > 0, cv2.erode(base, kernel) > 0

    def interior(self, erode_px: int = 4) -> np.ndarray:
        mask = self.mask.astype(np.uint8) * 255
        if erode_px > 0:
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2))
        return mask > 0

    def state(self) -> dict[str, Any]:
        return {
            "template_image": self.image,
            "template_mask": self.mask.astype(np.uint8),
            "residual_sigma": self.residual_sigma,
            "tolerance_px": int(self.tolerance_px),
            "n": int(self.n),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.image = state.get("template_image")
        self.mask = np.asarray(state.get("template_mask"), bool)
        self.residual_sigma = state.get("residual_sigma")
        self.tolerance_px = int(state.get("tolerance_px", 3))
        self.n = int(state.get("n", 0))
