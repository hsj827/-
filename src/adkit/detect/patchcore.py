from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .base import Detector, ScoreMap, mask_to_grid
from .features import Backbone


def farthest_point_sample(features: np.ndarray, target: int, seed: int = 0,
                          candidate_cap: int = 8000) -> np.ndarray:
    """Greedy k-center coreset over at most `candidate_cap` candidates.

    Above the cap the candidates are randomly pooled first, but the selection itself stays
    FPS. It used to stop at the pool - `n > cap` returned the random sample directly - which
    silently turned the coreset into random subsampling on any view with more than ~20 good
    images. That is the one part of PatchCore the method's coverage argument rests on, and it
    also left the bank at the full pool size, four times the operating point the published
    inference times assume.
    """
    n = features.shape[0]
    if target >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if n > candidate_cap:
        pool = rng.choice(n, size=candidate_cap, replace=False)
        local = farthest_point_sample(features[pool], min(target, candidate_cap), seed,
                                      candidate_cap=n + 1)
        return np.sort(pool[local])

    selected = [0]
    distances = np.full(n, np.inf, dtype=np.float32)
    for _ in range(target - 1):
        last = features[selected[-1]]
        step = np.maximum(((features - last) ** 2).sum(axis=1), 1e-12)
        distances = np.minimum(distances, step)
        selected.append(int(np.argmax(distances)))
    return np.array(sorted(set(selected)), dtype=np.int64)


class PatchCore(Detector):
    """kNN over a coreset of good-image patch descriptors. No gradient training."""

    name = "patchcore"

    def __init__(self, cfg: dict[str, Any] | None = None, backbone: Backbone | None = None):
        super().__init__(cfg)
        self.grid = tuple(int(v) for v in self.cfg.get("grid", [32, 32]))
        self.k = int(self.cfg.get("k", 9))
        self.coreset_ratio = float(self.cfg.get("coreset_ratio", 0.15))
        # The bank size is what the per-image cost is proportional to (top-k over the bank,
        # not the dot product), so it needs its own bound: `coreset_ratio` alone grows with
        # every good image added and reached 20000 rows on a 209-good view.
        self.max_memory = int(self.cfg.get("max_memory", 6000))
        self.candidate_cap = int(self.cfg.get("coreset_candidates", 8000))
        self.coreset = str(self.cfg.get("coreset", "fps"))
        self.smooth = int(self.cfg.get("smooth", 3))
        self.backbone = backbone or Backbone(
            self.cfg.get("backbone", "resnet18"),
            tuple(self.cfg.get("layers", ["layer2", "layer3"])),
            bool(self.cfg.get("pretrained", True)),
        )
        self.memory: np.ndarray | None = None
        self.owners: np.ndarray | None = None
        self._raw_dim: int | None = None

    def _patches(self, image: np.ndarray) -> np.ndarray:
        return self.backbone.patches(
            image, self.grid,
            pool=int(self.cfg.get("pool", 3)),
            proj_dim=int(self.cfg.get("proj_dim", 128)),
            seed=int(self.cfg.get("proj_seed", 0)),
        )

    def fit(self, images: list[np.ndarray], masks: list[np.ndarray]) -> None:
        if not images:
            raise ValueError("PatchCore needs at least one good image")
        patches = [self._patches(im) for im in images]
        bank = np.concatenate(patches, axis=0)
        # Which good each bank row came from. Without this the calibration cannot ask the
        # nearest neighbour question honestly: the query's own patches are in the bank, so a
        # trained image finds distance ~0 to itself and every unseen part looks worse by
        # comparison. Kept in memory only - a loaded bundle is never re-calibrated.
        owners = np.concatenate([np.full(p.shape[0], i, np.int32)
                                 for i, p in enumerate(patches)])
        self._raw_dim = bank.shape[1]
        target = max(1, min(int(bank.shape[0] * self.coreset_ratio), self.max_memory))
        pooled = bank.shape[0] > self.candidate_cap
        if target < bank.shape[0]:
            keep = farthest_point_sample(bank, target, int(self.cfg.get("proj_seed", 0)),
                                         self.candidate_cap)
            bank, owners = bank[keep], owners[keep]
        self.memory = bank.astype(np.float32)
        self.owners = owners
        self._coreset_pooled = pooled
        self.mark_fitted()

    def _knn(self, query: np.ndarray, own: np.ndarray | None) -> np.ndarray:
        """Nearest-neighbour similarities, a block of patches at a time.

        The whole product is `patches x bank` - at 1024 queries against a 20000-row memory
        that is 78 MB of float32 plus 156 MB of int64 partition indices, and a CPU-only box
        with 2 GB free fails inside it during calibration, when the fold models double the
        pressure. Nothing here needs the full matrix at once: the k-nearest neighbours of one
        patch say nothing about the next, so the work is blocked and the result is identical.
        """
        columns = self.memory.shape[0] if own is None else int((~own).sum())
        k = min(self.k, columns)
        block = max(1, int(self.cfg.get("knn_block", 256)))
        out = np.empty((query.shape[0], k), np.float32)
        for start in range(0, query.shape[0], block):
            similarity = query[start:start + block] @ self.memory.T
            if own is not None:
                similarity[:, own] = -np.inf
            idx = np.argpartition(similarity, -k, axis=1)[:, -k:]
            out[start:start + block] = np.take_along_axis(similarity, idx, axis=1)
        return out

    def score_map(self, image: np.ndarray, mask: np.ndarray,
                  exclude: int | None = None) -> ScoreMap:
        if self.memory is None:
            raise RuntimeError("PatchCore not fitted")
        query = self._patches(image)
        own = None
        if exclude is not None and self.owners is not None:
            own = self.owners == int(exclude)
            if not (own.any() and int((~own).sum()) >= 1):
                own = None  # nothing left to compare against; excluding it would score nothing
        # L2-normalised features: largest dot product == nearest neighbour.
        similarity = self._knn(query, own)
        distances = 2.0 - 2.0 * similarity
        score = np.sqrt(np.maximum(distances.mean(axis=1), 0.0)).reshape(self.grid)
        if self.smooth > 1:
            score = cv2.dilate(score, np.ones((self.smooth, self.smooth), np.uint8))
        score = score * mask_to_grid(mask, score.shape)
        return ScoreMap(
            source=self.name,
            score=score.astype(np.float32),
            image_score=float(score.max()),
            meta={"n_memory": int(self.memory.shape[0]), "grid": list(self.grid),
                  "coreset_pooled": bool(getattr(self, "_coreset_pooled", False))},
        )

    def calibration_map(self, image: np.ndarray, mask: np.ndarray,
                        index: int | None = None) -> ScoreMap:
        # `index` is the position in the list passed to `fit`, which is what `owners` records.
        # After `load` there are no owners, so a re-calibration from disk silently falls back
        # to the biased scoring - fit-time calibration is the only caller that matters.
        return self.score_map(image, mask, exclude=index)

    def state(self) -> dict[str, Any]:
        return {"memory": self.memory, "grid": np.array(self.grid), "raw_dim": self._raw_dim or 0}

    def restore(self, state: dict[str, Any]) -> None:
        self.memory = state.get("memory")
        self.owners = None
        self.mark_fitted()
