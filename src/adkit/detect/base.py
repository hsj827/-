from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScoreMap:
    source: str
    score: np.ndarray
    image_score: float
    mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def normalized(self, lo: float, hi: float,
                   pixel_ref: float | None = None) -> np.ndarray:
        """How many times the good-image extreme this pixel is. Deliberately not capped
        at 1.0 - the amount a detection exceeds the goods is the severity signal.

        A detector that is silent on every good image has no band to divide by: its
        percentile spread is exactly zero, and dividing by it would either zero the map or
        explode. Zeroing is worse than it looks, because it removes that detector from the
        fused pixel map altogether, so its findings can never appear as a defect region or
        as a source label even though the same detector votes clearly at image level. Fall
        back on the detector's own decision threshold instead, which means the same thing:
        1.0 is the point at which it would have called this image defective.
        """
        if hi - lo >= 1e-9:
            return np.maximum((self.score - lo) / (hi - lo), 0.0)
        if pixel_ref is None or pixel_ref <= 1e-9:
            return np.zeros_like(self.score)
        return np.maximum(self.score / float(pixel_ref), 0.0)

    def inside(self, mask: np.ndarray) -> np.ndarray:
        return mask_to_grid(mask, self.score.shape)


def mask_to_grid(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Restrict a score map to the part, resampling the mask for coarse patch grids."""
    if mask.shape == shape:
        return mask
    import cv2

    resized = cv2.resize(mask.astype(np.uint8) * 255, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return resized > 127


class Detector(ABC):
    """Uniform interface: fit on good canonical images only, emit a pixel score map."""

    name: str = "detector"

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}

    @abstractmethod
    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None: ...

    @abstractmethod
    def score_map(self, image: np.ndarray, mask: np.ndarray) -> ScoreMap: ...

    def pixel_reference(self) -> float | None:
        """The pixel value this detector itself treats as the defect decision.

        Used only when the good images give no distribution to normalise against.
        """
        return None

    def calibration_map(self, image: np.ndarray, mask: np.ndarray,
                        index: int | None = None) -> ScoreMap:
        """Score one training image *for the purpose of setting thresholds*.

        This must not be able to see the image it scores, and `score_map` can: a nearest
        neighbour bank contains the query's own patches, and a per-position mean contains the
        query's own feature, so a training image scores structurally lower than any part the
        fit has not met. Measured on a 20-good fit of a fixed-camera benchmark, the median
        unseen good scored 1.17x where the trained goods medianed 0.85x against the very same
        reference - so the reference is not a tight percentile of the right distribution, it
        is a percentile of a distribution the query cannot belong to. Inference keeps
        `score_map`; only the calibration numbers get excluded.
        """
        return self.score_map(image, mask)

    def state(self) -> dict[str, Any]:
        return {}

    def restore(self, state: dict[str, Any]) -> None:
        pass

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.state().items() if isinstance(v, np.ndarray)}
        header = {k: v for k, v in self.state().items() if not isinstance(v, np.ndarray)}
        if arrays:
            np.savez_compressed(directory / f"{self.name}.npz", **arrays)
        (directory / f"{self.name}.json").write_text(
            json.dumps(header, indent=2, default=str), encoding="utf-8"
        )

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        state: dict[str, Any] = {}
        json_path = directory / f"{self.name}.json"
        if json_path.exists():
            state = json.loads(json_path.read_text(encoding="utf-8"))
        npz_path = directory / f"{self.name}.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                state.update({k: data[k] for k in data.files})
        self.restore(state)

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_fitted", False)

    def mark_fitted(self) -> None:
        self._fitted = True
