from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config


def gradient_map(image: np.ndarray) -> np.ndarray:
    """Normalised edge magnitude.

    Dividing by the frame's own edge strength makes the model insensitive to exposure and
    lamp ageing, which is the whole reason to match on gradients instead of grey levels.
    """
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return (magnitude / (float(magnitude.std()) + 1e-6)).astype(np.float32)


@dataclass
class Pose:
    centre: tuple[float, float]
    size: tuple[float, float]
    peak: float
    best_turn: int = 0
    margin: float = 0.0


class ShapeModel:
    """Gradient shape model: where the part is, decided by counting its edge points.

    Position only. Grey-level correlation returns a perfect match on a patch of blank
    vignetted background, which is exactly what these frames offer, so the score has to be
    the fraction of the model's edge points that land on real edges.

    Which way the part faces is NOT decided here: the quarter-turn that wins this search is
    a property of the rotated mask, not of the component, and applying it to an already
    orientation-normalised canvas would double-rotate a sideways part. Heading is settled
    in canonical space, against the marking band.
    """

    def __init__(self, template: np.ndarray, centre: tuple[float, float],
                 angles: tuple[int, ...] = (0, 90, 180, 270),
                 grey: np.ndarray | None = None, min_points: int = 24):
        self.template = template.astype(np.float32)
        self.centre = centre
        self.angles = tuple(angles)
        self.min_points = min_points
        # Kept so a part the shape model cannot place can still fall back to grey-patch
        # matching rather than dropping straight to global thresholding.
        self.grey = grey if grey is not None else template

    def locate(self, gray: np.ndarray, query_percentile: float = 99.0) -> Pose:
        """Score = the fraction of the model's edge points that land on real edges.

        "Edge point" is one definition used on both sides of the comparison: a pixel above
        this frame's own gradient percentile. Percentiles taken separately per image do not
        work - the template is a tight crop so its 90th percentile is a weak edge, while
        the frame's 99th is a strong one, and the model then looks like it matches nothing.
        Normalising the gradient map by its own std already handles exposure, so the
        threshold is directly comparable between the crop and the frame.
        """
        map_ = gradient_map(np.clip(gray * 255.0, 0, 255).astype(np.uint8).astype(np.float32))
        threshold = float(np.percentile(map_, query_percentile))
        edges = (map_ > threshold).astype(np.float32)
        mask = (self.template > threshold).astype(np.float32)
        points = int(mask.sum())
        if points < self.min_points:
            raise ValueError("no shape points at this frame's edge threshold")

        ranked: list[tuple[float, Pose]] = []
        for turn in self.angles:
            patch = np.ascontiguousarray(np.rot90(mask, (turn // 90) % 4))
            if patch.shape[0] >= edges.shape[0] or patch.shape[1] >= edges.shape[1]:
                continue
            _, peak, _, loc = cv2.minMaxLoc(cv2.matchTemplate(edges, patch, cv2.TM_CCORR))
            ranked.append((float(peak) / points, Pose(
                centre=(float(loc[0] + patch.shape[1] / 2), float(loc[1] + patch.shape[0] / 2)),
                size=(float(patch.shape[1]), float(patch.shape[0])),
                peak=float(peak) / points, best_turn=int(turn))))
        if not ranked:
            raise ValueError("shape model is larger than the image")
        ranked.sort(key=lambda item: -item[0])
        best = ranked[0][1]
        best.margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 1.0)
        return best


def tight_part_rect(map_: np.ndarray, box: tuple[int, int, int, int], percentile: float) -> tuple:
    """Part rectangle from edge points, found in two passes.

    One percentile cannot do both jobs: a high one (99) bounds the part correctly but
    leaves too few points to match on, while a low one (90) gathers enough
    points yet swallows the lens vignette and yields a template bigger than the component.
    So the high pass sets the region and the low pass samples inside it.

    The size returned is the axis-aligned extent, not minAreaRect's own sides: a template
    cut to max/min of those two sides is a horizontal band across any part that happens to
    be standing up in the anchor image, and the model then matches bands instead of parts.
    """
    x, y, w, h = box
    sub = map_[y:y + h, x:x + w]
    if sub.size == 0:
        raise ValueError("empty search box")

    coarse_threshold = float(np.percentile(sub, 99.0))
    ys, xs = np.nonzero(sub > coarse_threshold)
    if xs.size < 16:
        raise ValueError("too few edge points to fit a shape")
    bx, by, bw, bh = cv2.boundingRect(np.stack([xs, ys], axis=1).astype(np.int32))

    inner = sub[by:by + bh, bx:bx + bw]
    threshold = float(np.percentile(inner, percentile))
    ys2, xs2 = np.nonzero(inner > threshold)
    if xs2.size < 16:
        ys2, xs2 = ys, xs
        bx = by = 0
    points = np.stack([xs2 + bx, ys2 + by], axis=1).astype(np.float32)
    ex, ey, ew, eh = cv2.boundingRect(points.astype(np.int32))
    # points are already in sub-window coordinates (xs2 was shifted by bx above), so only
    # the sub-window's own offset is still missing. Adding bx twice slides the box off the
    # part and teaches a template of blank background.
    return (float(ex + x + ew / 2), float(ey + y + eh / 2)), (float(ew), float(eh)), 0.0


def teach(gray: np.ndarray, cfg: Config, coarse_box: tuple[int, int, int, int]) -> ShapeModel:
    mcfg = cfg.get("preprocess.shape") or {}
    map_ = gradient_map(np.clip(gray * 255.0, 0, 255).astype(np.uint8).astype(np.float32))
    centre, (rw, rh), _ = tight_part_rect(map_, coarse_box, float(mcfg.get("edge_percentile", 99.5)))

    margin = float(mcfg.get("crop_margin", 0.08))
    crop_w = int(rw * (1.0 + margin))
    crop_h = int(rh * (1.0 + margin))
    x0 = max(0, int(centre[0] - crop_w / 2))
    y0 = max(0, int(centre[1] - crop_h / 2))
    template = map_[y0:y0 + crop_h, x0:x0 + crop_w]
    grey = gray[y0:y0 + crop_h, x0:x0 + crop_w]
    if min(template.shape) < 24:
        raise ValueError(f"taught template too small: {template.shape}")

    return ShapeModel(
        template=template,
        centre=(float(x0 + crop_w / 2), float(y0 + crop_h / 2)),
        angles=tuple(int(a) for a in mcfg.get("angles", [0, 90, 180, 270])),
        grey=grey,
    )
