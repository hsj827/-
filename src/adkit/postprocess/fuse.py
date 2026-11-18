from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..detect.base import ScoreMap
from .threshold import Calibration, DetectorCalib


@dataclass
class Fused:
    map: np.ndarray
    image_score: float
    is_defect: bool
    per_detector: dict[str, float] = field(default_factory=dict)
    sources: dict[str, np.ndarray] = field(default_factory=dict)
    raw: dict[str, ScoreMap] = field(default_factory=dict)
    inert: list[str] = field(default_factory=list)


RATIO_CAP = 999.0


def _resize_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if array.shape == shape:
        return array
    return cv2.resize(array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def fuse(score_maps: dict[str, ScoreMap], calib: Calibration,
         method: str = "max", weights: dict[str, float] | None = None) -> Fused:
    """Combine detector maps onto one grid. 1.0 == the good-image extreme.

    Patch-based detectors emit a coarse grid while the classical ones emit full
    resolution, so each map is normalised first and only then resampled.
    """
    if not score_maps:
        raise ValueError("no score maps to fuse")

    target = max((smap.score.shape for smap in score_maps.values()),
                 key=lambda shape: shape[0] * shape[1])
    maps: dict[str, np.ndarray] = {}
    for name, smap in score_maps.items():
        reference = calib.detectors.get(name, DetectorCalib())
        maps[name] = _resize_to(
            smap.normalized(reference.lo, reference.hi, reference.pixel_ref or None), target)

    stacked = np.stack(list(maps.values()))
    if method == "max":
        fused_map = stacked.max(axis=0)
    elif method == "weighted":
        w = np.array([(weights or {}).get(name, 1.0) for name in maps], dtype=np.float32)
        w = w / (w.sum() + 1e-9)
        fused_map = (stacked * w[:, None, None]).sum(axis=0)
    else:
        raise ValueError(f"unknown fusion method {method!r}")

    per_detector: dict[str, float] = {}
    uncalibrated: list[str] = []
    exceeded = False
    for name, smap in score_maps.items():
        reference = calib.detectors.get(name, DetectorCalib())
        per_detector[name] = min(smap.image_score / reference.image_ref, RATIO_CAP)
        # Compare against the good reference in absolute terms. Dividing alone is misleading
        # for a detector that stayed silent on every good image: its reference collapses to a
        # floor, so an ordinary reading reads back as a billion-fold exceedance.
        if smap.image_score > reference.image_ref * calib.image_margin:
            exceeded = True
        if reference.inert:
            uncalibrated.append(name)

    return Fused(
        map=fused_map.astype(np.float32),
        image_score=max(per_detector.values()),
        is_defect=exceeded,
        per_detector=per_detector,
        sources=maps,
        raw=score_maps,
        inert=uncalibrated,
    )
