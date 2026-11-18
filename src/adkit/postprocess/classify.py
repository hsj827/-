from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import Config
from .fuse import Fused


@dataclass
class Defect:
    label: str
    bbox: tuple[int, int, int, int]
    area: float
    elongation: float
    max_dim: float
    score: float
    centroid: tuple[float, float]
    sources: list[str] = field(default_factory=list)
    # Pixels, cropped to bbox. The bbox is what a reviewer sees as a rectangle; the mask is
    # the claim being checked - whether the detector's boundary actually follows the defect.
    mask: np.ndarray | None = None
    source_masks: dict[str, np.ndarray] = field(default_factory=dict)
    # How much of the part is past the goods extreme, and by how much: the sum of
    # max(map - base, 0) over this component, divided by the part area. A dozen pixels at 2.1
    # and a tenth of the part at 1.2 are the same peak-free statistic; this separates them.
    mass: float = 0.0

    @property
    def mask_area(self) -> float:
        return float(self.mask.sum()) if self.mask is not None else 0.0

    @property
    def fill(self) -> float:
        """How much of the box the defect actually occupies."""
        if self.mask is None:
            return 0.0
        return float(self.mask.sum()) / max(1.0, float(self.mask.size))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "bbox": list(self.bbox),
            "area": self.area,
            "mask_area": round(self.mask_area),
            "fill": round(self.fill, 3),
            "elongation": round(self.elongation, 2),
            "max_dim": round(self.max_dim, 1),
            "score": round(self.score, 4),
            "centroid": [round(c, 1) for c in self.centroid],
            "sources": self.sources,
            "mass": round(self.mass, 5),
        }


def _features(component: np.ndarray) -> tuple[float, float, float, tuple[int, int, int, int], tuple[float, float]]:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0, (0, 0, 0, 0), (0.0, 0.0)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour)) or float(component.sum())
    rect = cv2.minAreaRect(contour)
    short, long_ = sorted(rect[1])
    elongation = long_ / short if short > 1e-3 else long_
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = rect[0]
    return area, elongation, long_, (int(x), int(y), int(w), int(h)), (float(cx), float(cy))


def _boundary_distance(template_mask: np.ndarray | None) -> np.ndarray | None:
    if template_mask is None:
        return None
    mask = template_mask.astype(np.uint8) * 255
    edge = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _inside_band(bbox: tuple[int, int, int, int], border: int,
                 width: int, height: int) -> bool:
    """True when the whole box sits in the outer `border`-pixel band of the canvas."""
    x, y, w, h = bbox
    return (x + w <= border + 1 or y + h <= border + 1
            or x >= width - border - 1 or y >= height - border - 1)


def _matches(rule: dict[str, Any], feats: dict[str, float]) -> bool:
    if "source" in rule and rule["source"] not in feats["sources"]:
        # Shape alone cannot tell a crack from a smudge that happens to be long: a stain
        # streak scores elongation 4 without the line detector ever firing on it.
        return False
    if "min_elongation" in rule and feats["elongation"] < float(rule["min_elongation"]):
        return False
    if "min_maxdim" in rule and feats["max_dim"] < float(rule["min_maxdim"]):
        return False
    if "min_area" in rule and feats["area"] < float(rule["min_area"]):
        return False
    if "max_area" in rule and feats["area"] > float(rule["max_area"]):
        return False
    if "near_boundary" in rule and feats.get("boundary_distance", 1e9) > float(rule["near_boundary"]):
        return False
    return True


def classify(fused: Fused, cfg: Config, pixel_threshold: float | None = None,
             template_mask: np.ndarray | None = None,
             extent_cut: float | None = None) -> list[Defect]:
    rules = cfg.get("classify.rules", {}) or {}
    default_label = str(cfg.get("classify.default", "stain"))
    cut = float(cfg.get("fusion.pixel_cut", 1.5)) if pixel_threshold is None else float(pixel_threshold)
    min_area = float(cfg.get("classify.min_area", 12))
    border = int(cfg.get("classify.border_px", 2))
    # 1.0 is by construction "as extreme as the loudest pixel any good image produced", so
    # pixels above it are the evidence and the amount above it is the strength.
    base = float(cfg.get("classify.extent_base", 1.0))
    extent = float(cfg.get("classify.extent_cut", 0.0)) if extent_cut is None else float(extent_cut)
    # An explicit off switch, because this rule trades false alarms for recall and the right
    # side of that trade is a production decision, not an algorithmic one: measured on the
    # cans it is 2/14 unseen good parts flagged at 15/16 defects caught versus 3/14 at 16/16.
    if not bool(cfg.get("classify.extent_evidence", True)):
        extent = 0.0
    threshold = min(cut, base) if extent > 0.0 else cut

    binary = (fused.map >= threshold).astype(np.uint8)
    if not binary.any():
        return []

    # Connect evidence that is nearby before measuring it. A spill is speckled, and each
    # speckle on its own is both below the peak cut and below the extent floor - the image
    # carries 0.0065 of extent evidence while no single component reaches the 0.00519 floor,
    # which is how a labelled contamination with a real signal stayed undetected. Dilation
    # merges them; the mass is still summed from the untouched map, so merged pixels add the
    # evidence they actually hold and nothing is invented. The floor moves with it, because
    # the goods are measured through this same function.
    merge = int(cfg.get("classify.merge_px", 3))
    if merge > 1:
        binary = cv2.dilate(binary, np.ones((merge, merge), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    boundary = _boundary_distance(template_mask)
    order = [name for name in rules.keys() if name != "default"]
    height, width = fused.map.shape[:2]
    part_area = float(template_mask.sum()) if template_mask is not None \
        else float(max(1, int(np.count_nonzero(fused.map >= base))))

    defects: list[Defect] = []
    for i in range(1, count):
        component = labels == i
        area, elongation, max_dim, bbox, centroid = _features(component)
        if area < min_area:
            continue
        # Reject a region only when the *whole* component lives inside the outer band. The
        # score map is already multiplied by the part mask, so anything at the canvas edge is
        # something the part itself reaches - and for a product that fills the frame, the old
        # "touches the edge" test threw away every real defect: on a bottle benchmark it
        # returned zero regions on all 63 labelled NG images, including one whose map peaked at
        # 4.01 against a cut of 2.43. Only a sliver confined to the band is a cropping seam.
        if border >= 0 and _inside_band(bbox, border, width, height):
            continue
        region_score = float(fused.map[component].max())
        # Two independent ways to be a defect. `strong` is the old one: somewhere in this
        # region a pixel is clearly past anything the goods produce. `extensive` is the one
        # the peak cannot see: a lot of the region sits a little past the goods extreme, which
        # is what a spill or a faint scratch looks like. Measured on a benchmark bottle, a
        # contamination covering a tenth of the part peaked at 1.30 against a cut of 2.43 -
        # invisible to the first rule - while its extent ran 18x the most any good part
        # produced. Both floors come from the leave-one-out calibration, so neither is a
        # guess, and neither is a percentile of a score the detector memorised.
        mass = float(np.clip(fused.map[component] - base, 0.0, None).sum() / part_area)
        strong = region_score >= cut
        extensive = extent > 0.0 and mass >= extent
        if not (strong or extensive):
            continue
        dist = float(boundary[component].min()) if boundary is not None else 1e9
        sources = [name for name, source_map in fused.sources.items()
                   if float(source_map[component].max()) > 0.0]
        feats = {"area": area, "elongation": elongation, "max_dim": max_dim,
                 "boundary_distance": dist, "sources": set(sources)}

        label = default_label
        for name in order:
            rule = rules.get(name) or {}
            if rule and _matches(rule, feats):
                label = name
                break

        x, y, w, h = bbox
        # The merged component decides *whether* this is a defect; the mask reported is the
        # pixels that actually exceed the threshold. The dilation halo is not evidence, and the
        # mask is the claim a reviewer checks with a mouse - shipping the grown shape would
        # make every box a few pixels fatter than the defect and teach the reflux loop to treat
        # the halo as the boundary.
        mask = (component & (fused.map >= threshold))[y:y + h, x:x + w]
        # A detector contributed to this defect where its own normalised map also clears the
        # cut, not merely where it is above zero - otherwise a faint tail of one channel gets
        # reported as a co-author of every defect.
        source_masks = {}
        admit = cut if strong else base
        for name, values in fused.sources.items():
            region = values[y:y + h, x:x + w]
            if region.shape != mask.shape:
                continue
            contributed = (region >= admit) & mask
            if contributed.any():
                source_masks[name] = contributed
        defects.append(
            Defect(
                label=label,
                bbox=bbox,
                area=area,
                elongation=elongation,
                max_dim=max_dim,
                score=region_score,
                centroid=centroid,
                sources=sources,
                mask=mask,
                source_masks=source_masks,
                mass=mass,
            )
        )
    return sorted(defects, key=lambda d: -d.score)
