from __future__ import annotations

import cv2
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config


@dataclass
class DetectorCalib:
    lo: float = 0.0
    hi: float = 1.0
    image_max: float = 0.0
    n_pixels: int = 0
    pixel_ref: float = 0.0
    # The high quantile of the per-image scores, kept beside the maximum because they are
    # different quantities and only one of them is stable as data arrives.
    image_p: float = 0.0

    @property
    def span(self) -> float:
        return max(self.hi - self.lo, 1e-6)

    @property
    def image_ref(self) -> float:
        """Image-level reference: how loudly a good part may score, not how loudly the
        loudest good part in this batch happened to score.

        Using the maximum makes the line a function of batch size: at 209 goods the maximum
        sits far above what an unseen part typically reaches, so a real defect at 1.14x the
        reference is judged good against a margin of 1.15 - measured on a benchmark bottle
        whose chip peaked at 1.68 against a pixel line of 2.66 while unseen goods topped out
        at 1.24. The more good images you collect, the blunter the system gets, which is the
        opposite of what adding data is for. A quantile converges instead of climbing, and
        the price it charges is explicit and stated in the fit notes: a fraction
        `100 - reference_percentile` of good parts is expected to cross the line.

        Falling back on the detector's own decision threshold matters. A defect-specific
        channel is silent on the goods by design - good parts have no cracks - and if the
        reference collapses to a floor, `score > reference * margin` becomes true for every
        image that produces anything at all, which turns the channel into an automatic NG
        vote. Measured on the production set: 4/4 unseen good parts saturated at the ratio
        cap. The detector's own cut is the only absolute line left in that case.
        """
        return max(self.image_p, self.hi, self.pixel_ref, 1e-6)

    @property
    def inert(self) -> bool:
        """True when nothing at all calibrates this detector.

        Silence on the good images is normal for a defect-specific detector - goods have no
        cracks - and no longer means unusable, because the detector's own decision threshold
        carries the pixel scale. Only a detector with neither is truly inert.
        """
        if max(self.image_p, self.image_max, self.hi) > 1e-9:
            return False
        return self.pixel_ref <= 1e-9


@dataclass
class Calibration:
    detectors: dict[str, DetectorCalib] = field(default_factory=dict)
    pixel_threshold: float = 1.0
    image_threshold: float = 1.0
    image_margin: float = 1.15
    pixel_cut: float = 1.5
    # The floor for the extent statistic: the most past-the-goods-extreme mass any good part
    # produces anywhere on itself, times the margin. Kept separate from `pixel_cut` because it
    # answers a different question - how much of the part is wrong, not how loud one pixel is.
    extent_cut: float = 0.0
    n_good: int = 0
    views: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["detectors"] = {k: asdict(v) for k, v in self.detectors.items()}
        return json.dumps(payload, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Calibration":
        payload = json.loads(text)
        payload["detectors"] = {k: DetectorCalib(**v) for k, v in payload.get("detectors", {}).items()}
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def calibrate(detectors, images: list[np.ndarray], masks: list[np.ndarray],
              cfg: Config, template_mask: np.ndarray | None = None) -> Calibration:
    """Fit thresholds from the good-image score distribution only."""
    pct = float(cfg.get("calibration.pixel_percentile", 99.9))
    margin = float(cfg.get("calibration.image_margin", 1.15))
    ref_pct = float(cfg.get("calibration.reference_percentile", 99.0))

    calib = Calibration(n_good=len(images))
    pixel_pools: dict[str, list[np.ndarray]] = {}
    image_scores: dict[str, list[float]] = {}

    for det in detectors:
        pixel_pools[det.name] = []
        image_scores[det.name] = []

    for i, (image, mask) in enumerate(zip(images, masks)):
        for det in detectors:
            # `calibration_map`, not `score_map`: a training image scored by a model that
            # contains it is a different, easier question than a part nobody has seen.
            smap = det.calibration_map(image, mask, i)
            region = mask
            if region.shape != smap.score.shape:
                region = cv2.resize(region.astype(np.uint8) * 255,
                                   (smap.score.shape[1], smap.score.shape[0]),
                                   interpolation=cv2.INTER_AREA) > 127
            values = smap.score[region] if region.any() else smap.score.ravel()
            pixel_pools[det.name].append(values.astype(np.float32))
            image_scores[det.name].append(smap.image_score)

    for det in detectors:
        pool = np.concatenate(pixel_pools[det.name]) if pixel_pools[det.name] else np.zeros(1, np.float32)
        scores = np.asarray(image_scores[det.name], dtype=np.float32)
        calib.detectors[det.name] = DetectorCalib(
            lo=float(np.percentile(pool, 50)),
            hi=float(np.percentile(pool, pct)),
            image_max=float(scores.max()) if scores.size else 0.0,
            n_pixels=int(pool.size),
            pixel_ref=float(det.pixel_reference() or 0.0),
            image_p=float(np.percentile(scores, ref_pct)) if scores.size else 0.0,
        )

    calib.pixel_threshold = 1.0
    calib.image_margin = margin
    calib.image_threshold = max((c.image_ref for c in calib.detectors.values()), default=1.0) * margin

    # The region threshold used to be a constant, which meant the pixel path could never be
    # trusted with the verdict: on a rig whose own good parts throw regions above that
    # constant, counting regions would simply invent defects. Probe the goods at a loose
    # value, take the loudest region they actually produce, and set the cut above it - so
    # that regions can decide something without the threshold being a guess.
    #
    # Both statistics are probed through `calibration_map`, for the same reason the image
    # reference is: asking a detector for the region it produces while holding the query inside
    # itself is a different question from the region an unseen part produces, and a cut has to
    # be set on the answer to the second one. Whether the resulting line overshoots or
    # undershoots the parts that will actually arrive is not something this function can see -
    # every image here is in the fit - so `Pipeline._holdout_metrics` measures it on parts a
    # fold never trained on, and the deployed cut is set from that.
    from .classify import classify
    from .fuse import fuse

    probe = float(cfg.get("calibration.region_probe", 1.0))
    floor_cut = float(cfg.get("fusion.pixel_cut", 1.5))
    peaks: list[float] = []
    masses: list[float] = []
    crossing_sources: list[list[str]] = []
    for i, (image, mask) in enumerate(zip(images, masks)):
        maps = {d.name: d.calibration_map(image, mask, i) for d in detectors}
        fused = fuse(maps, calib, method=str(cfg.get("fusion.method", "max")),
                     weights=cfg.get("fusion.weights"))
        found = classify(fused, cfg, pixel_threshold=probe, template_mask=template_mask,
                         extent_cut=1e-9)
        peaks.append(max((d.score for d in found), default=0.0))
        masses.append(max((d.mass for d in found), default=0.0))
        loudest = max(found, key=lambda d: d.score, default=None)
        crossing_sources.append(list(loudest.sources) if loudest else [])
    region_ref = max(peaks) if peaks else 0.0
    extent_ref = max(masses) if masses else 0.0
    # Quantiles, not maxima. A maximum is a function of batch size: it can only grow as more
    # good images arrive, so the line drifts away from the parts that will actually show up
    # and the system gets blunter with the very data it was supposed to benefit from. Measured
    # on a 209-good benchmark fit, the pixel line reached 2.66 and the extent line 0.0551 while
    # unseen good parts topped out at 1.24 - a real chip at 1.68 / 0.0068 was therefore judged
    # good and got no box, and the batch maximum was set by one image nobody would ship as a
    # reference. A quantile converges. What it costs is stated rather than hidden: by
    # construction about `100 - reference_percentile` percent of good parts sit above the line.
    calib.pixel_cut = max(float(np.percentile(peaks, ref_pct)) * margin, floor_cut) \
        if peaks else floor_cut
    calib.extent_cut = float(np.percentile(masses, ref_pct)) * margin if masses else 0.0
    calib.notes["region_ref"] = round(region_ref, 4)
    calib.notes["extent_ref"] = round(extent_ref, 5)
    calib.notes["region_p"] = round(float(np.percentile(peaks, ref_pct)) if peaks else 0.0, 4)
    calib.notes["extent_p"] = round(float(np.percentile(masses, ref_pct)) if masses else 0.0, 5)
    calib.notes["reference_percentile"] = ref_pct
    calib.notes["goods_over_pixel_cut"] = int(sum(1 for v in peaks
                                                  if v >= calib.pixel_cut / margin))
    calib.notes["goods_over_extent_cut"] = int(sum(1 for v in masses
                                                   if v >= calib.extent_cut / margin))
    # Say which channel owns the goods that cross the line. Without it the count is a number
    # the operator cannot act on: a phantom region whose label is `missing` because the fitted
    # rectangle's heading disagrees with its own median outline is a geometry problem, and the
    # same count coming from `stefann` would be a lighting or a second-face problem.
    #
    # This counts the sources of the loudest region on each crossing image, so one region that
    # two channels both saw is charged to both - it is not the same question as which channel
    # wins the over-threshold pixels. Measured on the deployed 209-good bottle bundle the
    # answer to the second question is unambiguous: all 48 pixels above the line argmaxed to
    # `missing`, whose outline evidence is the fitted quad and nothing else, while the four
    # unseen goods and the 14 can frames produced none at all.
    owners: dict[str, int] = {}
    for sources, peak in zip(crossing_sources, peaks):
        if peak >= calib.pixel_cut / margin and sources:
            for name in sources:
                owners[name] = owners.get(name, 0) + 1
    calib.notes["crossing_by_detector"] = owners
    return calib
