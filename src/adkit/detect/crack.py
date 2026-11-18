from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import Detector, ScoreMap
from .template import Template


def _gauss_kernel(sigma: float, order: int) -> np.ndarray:
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    if order == 0:
        k = g
    elif order == 1:
        k = -(x / (sigma * sigma)) * g
    else:
        k = ((x * x) / (sigma * sigma) - 1.0) * g
    return (k / np.abs(k).sum()).astype(np.float32)


def line_response(image: np.ndarray, sigmas: list[float], polarity: str = "both") -> np.ndarray:
    """Steger's Hessian line measurement, max-normalised across scales.

    A crack in a dark body and a scratch on a dark lid have opposite contrast, so a
    detector wired for one polarity is blind to the other; `both` keeps whichever is
    locally stronger.
    """
    best = np.zeros_like(image, dtype=np.float32)
    for sigma in sigmas:
        g = _gauss_kernel(sigma, 0)
        d1 = _gauss_kernel(sigma, 1)
        d2 = _gauss_kernel(sigma, 2)
        hxx = cv2.sepFilter2D(image, cv2.CV_32F, d2, g)
        hyy = cv2.sepFilter2D(image, cv2.CV_32F, g, d2)
        hxy = cv2.sepFilter2D(image, cv2.CV_32F, d1, d1)
        root = np.sqrt(hxx * hxx + 2.0 * hxy * hxy + hyy * hyy)
        dark = -(hxx + hyy) + root
        bright = (hxx + hyy) + root
        if polarity == "dark":
            candidate = dark
        elif polarity == "bright":
            candidate = bright
        else:
            candidate = np.maximum(dark, bright)
        best = np.maximum(best, candidate.astype(np.float32) / (sigma * sigma))
    return best


def dark_line_response(image: np.ndarray, sigmas: list[float]) -> np.ndarray:
    return line_response(image, sigmas, "dark")


def components(mask: np.ndarray, min_area: int, max_area: int,
               min_elongation: float = 0.0) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros(mask.shape, np.uint8)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        if min_elongation > 0:
            contours, _ = cv2.findContours((labels == i).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
            short, long_ = sorted(rect[1])
            if short < 1e-3 or long_ / short < min_elongation:
                continue
        out[labels == i] = 1
    return out.astype(bool)


class CrackDetector(Detector):
    """Classical dark-line (crack) detector: Hessian ridge response + shape filtering."""

    name = "crack"

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        self.template = Template()
        self.response_cut: float | None = None
        self.response_sigma: np.ndarray | None = None

    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None:
        self.template.build(images, masks)
        sigmas = [float(s) for s in self.cfg.get("sigmas", [1.0, 2.0, 3.0])]
        polarity = str(self.cfg.get("polarity", "both"))
        region = self.template.interior(int(self.cfg.get("interior_erode", 3)))

        # What the line filter would fire on with no defect present at all. Comparing good
        # images against each other cannot measure this: the lid seam is itself a chain of
        # short bright ridges, so a few pixels of registration error turns it into the most
        # crack-like texture on the part, and it looks identical in every good image. The
        # only honest estimate is to shift the reference by the placement jitter this rig
        # actually has and see what response pure misalignment produces.
        jitter = int(self.cfg.get("jitter_px", 2))
        # The unshifted comparison must be excluded. Including it makes the noise envelope
        # at least as large as the response itself, so every fitted image is pinned at z<=1
        # and the reference becomes exactly 1.0 by construction - a decision boundary that
        # cannot be exceeded by anything the fit has seen, and is crossed by every part it
        # has not. Measured: 0/10 in-sample false alarms, 4/4 out-of-sample.
        offsets = [(dx, dy) for dx in range(-jitter, jitter + 1)
                   for dy in range(-jitter, jitter + 1) if 0 < dx * dx + dy * dy <= jitter * jitter]
        noise = np.zeros_like(self.template.image, dtype=np.float32)
        for dx, dy in offsets:
            shifted = cv2.warpAffine(self.template.image,
                                     np.float32([[1, 0, dx], [0, 1, dy]]),
                                     (self.template.image.shape[1], self.template.image.shape[0]))
            for image in images:
                np.maximum(noise, line_response(image - shifted, sigmas, polarity), out=noise)

        floor = float(np.percentile(noise[region], 50.0)) if region.any() else 0.0
        self.response_sigma = np.maximum(noise, floor * float(self.cfg.get("floor_ratio", 0.5))
                                         ) + 1e-6
        self.response_sigma = self.response_sigma.astype(np.float32)

        responses = [line_response(im - self.template.image, sigmas, polarity) for im in images]
        z = [(r / self.response_sigma).astype(np.float32) for r in responses]
        base = float(np.percentile(np.concatenate([zz[region] for zz in z]),
                                  float(self.cfg.get("score_percentile", 99.0))))

        # A percentile says nothing about the maximum, and the decision is a maximum over
        # connected components - so a cut taken from the fitted images' own p99 lets every
        # unseen good part through. Measured on the production set: cut at p99 gives 4/4
        # false alarms on parts held out of the fit, and the worst of them (3.49) beats the
        # weakest real defect (2.91). No reweighting of the statistic fixes that; the tail
        # itself is what was underestimated.
        #
        # So measure the tail's growth instead of guessing a multiplier: rebuild the model
        # on part of the goods and see how much further the rest reach.
        self.response_cut = base * self._generalisation_ratio(images, region)
        self.mark_fitted()

    def _generalisation_ratio(self, images: list[np.ndarray], region: np.ndarray) -> float:
        """How much higher a good image scores when the model did not see it.

        The fold supplies the *noise envelope and the cut*, not the template: the held-out
        residual is still taken against the deployed median. Making the fold supply its own
        template as well was tried, on the argument that an isolated model is what generalises
        or fails to, and it is wrong here for the same reason the pipeline's cross-fitted
        margin was reverted - a fold template is the median of fewer images, so it is blurrier
        and every residual against it is larger. The ratio then measures that small-model
        penalty instead of the tail, and the cut it produces is not a threshold any more:
        measured over four folds on the production set, unseen goods stayed 0/14 false alarms
        either way, but known defects fell from 12/16 caught to 1/16.
        """
        margin = float(self.cfg.get("cut_margin", 1.15))
        folds = int(self.cfg.get("calibration_folds", 2))
        if len(images) < folds * 2:
            return margin
        worst_seen, worst_held = 1.0, 1.0
        for fold in range(folds):
            held = list(range(fold, len(images), folds))
            rest = [i for i in range(len(images)) if i not in held]
            if not rest or not held:
                continue
            cut, sigma = self._fold_model([images[i] for i in rest], region)
            for group, indices in (("rest", rest), ("held", held)):
                peaks = []
                for i in indices:
                    z = line_response(images[i] - self.template.image,
                                      [float(s) for s in self.cfg.get("sigmas", [1.0, 2.0, 3.0])],
                                      str(self.cfg.get("polarity", "both"))) / sigma * region
                    keep = components(z > cut, int(self.cfg.get("min_area", 10)),
                                      int(self.cfg.get("max_area", 3000)),
                                      float(self.cfg.get("min_elongation", 4.0)))
                    peaks.append(float(z[keep].max()) if keep.any() else 0.0)
                seen = max(p for p, i in zip(peaks, indices) if i in rest) if group == "rest" else None
                if group == "rest":
                    worst_seen = max(worst_seen, max(peaks))
                else:
                    worst_held = max(worst_held, max(peaks))
        return max(1.0, worst_held / max(worst_seen, 1e-9)) * margin

    def _fold_model(self, subset: list[np.ndarray], region: np.ndarray) -> tuple[float, np.ndarray]:
        """Template, jitter noise map and p99 cut for one calibration fold."""
        reference = Template()
        reference.build(subset, [np.ones_like(im, dtype=bool) for im in subset])
        sigmas = [float(s) for s in self.cfg.get("sigmas", [1.0, 2.0, 3.0])]
        polarity = str(self.cfg.get("polarity", "both"))
        jitter = int(self.cfg.get("jitter_px", 2))
        offsets = [(dx, dy) for dx in range(-jitter, jitter + 1)
                   for dy in range(-jitter, jitter + 1) if 0 < dx * dx + dy * dy <= jitter * jitter]
        h, w = reference.image.shape
        noise = np.zeros(reference.image.shape, np.float32)
        for dx, dy in offsets:
            shifted = cv2.warpAffine(reference.image,
                                     np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))
            for image in subset:
                np.maximum(noise, line_response(image - shifted, sigmas, polarity), out=noise)
        base = np.maximum(noise, float(np.percentile(noise[region], 50.0)) * 0.5) + 1e-6
        zs = [line_response(image - reference.image, sigmas, polarity) / base * region
              for image in subset]
        cut = float(np.percentile(np.concatenate([z[region] for z in zs]),
                                 float(self.cfg.get("score_percentile", 99.0))))
        return cut, base

    def _z(self, image: np.ndarray) -> np.ndarray:
        sigmas = [float(s) for s in self.cfg.get("sigmas", [1.0, 2.0, 3.0])]
        polarity = str(self.cfg.get("polarity", "both"))
        resp = line_response(image - self.template.image, sigmas, polarity)
        return (resp / self.response_sigma).astype(np.float32)

    def pixel_reference(self) -> float | None:
        return self.response_cut

    def score_map(self, image: np.ndarray, mask: np.ndarray) -> ScoreMap:
        if self.response_cut is None:
            raise RuntimeError("CrackDetector not fitted")
        region = self.template.interior(int(self.cfg.get("interior_erode", 3)))
        z = self._z(image) * region

        candidate = z > self.response_cut
        keep = components(
            candidate,
            int(self.cfg.get("min_area", 10)),
            int(self.cfg.get("max_area", 3000)),
            float(self.cfg.get("min_elongation", 4.0)),
        )
        score = z * keep
        return ScoreMap(
            source=self.name,
            score=score.astype(np.float32),
            image_score=float(z[keep].max()) if keep.any() else 0.0,
            mask=keep,
            meta={"cut": self.response_cut, "n_px": int(keep.sum())},
        )

    def state(self) -> dict[str, Any]:
        return {**self.template.state(), "response_cut": float(self.response_cut or 0.0),
                "response_sigma": self.response_sigma}

    def restore(self, state: dict[str, Any]) -> None:
        self.template.restore(state)
        self.response_cut = float(state.get("response_cut", 0.0))
        sigma = state.get("response_sigma")
        if sigma is not None:
            self.response_sigma = np.asarray(sigma, np.float32)
        self.mark_fitted()
