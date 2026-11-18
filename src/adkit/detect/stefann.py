from __future__ import annotations

from typing import Any

import numpy as np

from .base import Detector, ScoreMap, mask_to_grid
from .features import Backbone


class STEFANN(Detector):
    """Per-position Mahalanobis distance over good-image patch features.

    Needs strict alignment (the canonicaliser provides it) and >= min_images goods,
    because a covariance is estimated at every grid position.
    """

    name = "stefann"

    def __init__(self, cfg: dict[str, Any] | None = None, backbone: Backbone | None = None):
        super().__init__(cfg)
        self.grid = tuple(int(v) for v in self.cfg.get("grid", [16, 16]))
        self.pca_dim = int(self.cfg.get("pca_dim", 24))
        self.shrinkage = float(self.cfg.get("shrinkage", 0.3))
        self.min_images = int(self.cfg.get("min_images", 4))
        self.backbone = backbone or Backbone(
            self.cfg.get("backbone", "resnet18"),
            tuple(self.cfg.get("layers", ["layer2", "layer3"])),
            bool(self.cfg.get("pretrained", True)),
        )
        self.components: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None
        # Per-position sums of the fitted coordinates, kept so a training image can be scored
        # against the mean and covariance *without itself*. Not persisted: only the fit-time
        # calibration needs them, and they are the size of the model.
        self._n = 0
        self._sum1: np.ndarray | None = None
        self._sum2: np.ndarray | None = None

    def _patches(self, image: np.ndarray) -> np.ndarray:
        feats = self.backbone.patches(
            image, self.grid,
            pool=int(self.cfg.get("pool", 3)),
            proj_dim=int(self.cfg.get("proj_dim", 64)),
            seed=int(self.cfg.get("proj_seed", 0)),
        )
        return feats

    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None:
        from sklearn.decomposition import PCA

        if len(images) < self.min_images:
            raise ValueError(
                f"STEFANN needs >= {self.min_images} good images per view "
                f"(got {len(images)}); use PatchCore for very small sets"
            )
        stack = np.stack([self._patches(im) for im in images], axis=0)  # (n, N, D)
        n, positions, dim = stack.shape
        pca_dim = min(self.pca_dim, max(2, n - 2), dim)

        components = np.zeros((positions, dim, pca_dim), np.float32)
        means = np.zeros((positions, dim), np.float32)
        precision = np.zeros((positions, pca_dim, pca_dim), np.float32)
        sum1 = np.zeros((positions, pca_dim), np.float64)
        sum2 = np.zeros((positions, pca_dim, pca_dim), np.float64)
        identity = np.eye(pca_dim, dtype=np.float64)

        for p in range(positions):
            block = stack[:, p, :]
            pca = PCA(n_components=pca_dim).fit(block)
            means[p] = pca.mean_
            components[p] = pca.components_.T
            reduced = pca.transform(block)
            cov = np.atleast_2d(np.cov(reduced, rowvar=False))
            scale = float(np.trace(cov)) / pca_dim
            shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * scale * identity
            precision[p] = np.linalg.inv(shrunk + 1e-6 * identity)
            sum1[p] = reduced.sum(axis=0)
            sum2[p] = reduced.T @ reduced

        self.components = components.astype(np.float32)
        self.mean = means.astype(np.float32)
        self.precision = precision.astype(np.float32)
        self._n = int(n)
        self._sum1 = sum1
        self._sum2 = sum2
        self.mark_fitted()

    def _reduced(self, image: np.ndarray) -> np.ndarray:
        feats = self._patches(image)
        return np.einsum("nD,nDd->nd", feats - self.mean, self.components.astype(np.float64))

    def _leave_one_out(self, reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and precision at every position with this image's own feature removed.

        The rotation is left as fitted - dropping one image out of n moves the principal axes
        by O(1/n), which is second-order next to the shift in the mean and covariance, and the
        point of the exercise is a first-order bias that measured 0.85 -> 1.17 on the median.
        """
        n = self._n - 1
        centre = (self._sum1 - reduced) / n
        second = self._sum2 - reduced[:, :, None] * reduced[:, None, :]
        cov = (second - n * centre[:, :, None] * centre[:, None, :]) / max(n - 1, 1)
        dim = cov.shape[-1]
        identity = np.eye(dim)
        scale = np.trace(cov, axis1=1, axis2=2)[:, None, None] / dim
        shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * scale * identity
        return centre.astype(np.float64), np.linalg.inv(shrunk + 1e-6 * identity)

    def score_map(self, image: np.ndarray, mask: np.ndarray,
                  exclude_index: int | None = None) -> ScoreMap:
        if self.components is None:
            raise RuntimeError("STEFANN not fitted")
        reduced = self._reduced(image)
        if exclude_index is None:
            centre = np.zeros_like(reduced)
            precision = self.precision.astype(np.float64)
        else:
            centre, precision = self._leave_one_out(reduced)
        delta = reduced - centre
        quad = np.einsum("ni,nij,nj->n", delta, precision, delta)
        score = np.sqrt(np.maximum(quad, 0.0)).reshape(self.grid).astype(np.float32)
        score = score * mask_to_grid(mask, score.shape)
        return ScoreMap(
            source=self.name,
            score=score,
            image_score=float(score.max()),
            meta={"grid": list(self.grid), "pca_dim": int(self.components.shape[2]),
                  "held_out": exclude_index is not None},
        )

    def calibration_map(self, image: np.ndarray, mask: np.ndarray,
                        index: int | None = None) -> ScoreMap:
        if index is None or self._sum2 is None or self._n < 4:
            return self.score_map(image, mask)
        return self.score_map(image, mask, exclude_index=index)

    def state(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "mean": self.mean,
            "precision": self.precision,
            "grid": np.array(self.grid),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.components = state.get("components")
        self.mean = state.get("mean")
        self.precision = state.get("precision")
        # No sums on disk: a bundle loaded from disk must not calibrate itself against stale
        # statistics it cannot have.
        self._sum1 = self._sum2 = None
        self._n = 0
        self.mark_fitted()
