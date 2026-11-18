from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

BACKGROUND = 250


def _torch_works() -> bool:
    try:
        import torch
    except Exception:
        return False
    return True


TORCH_OK = _torch_works()


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_torch: requires a working torch install")


def pytest_runtest_setup(item):
    if "needs_torch" in item.keywords and not TORCH_OK:
        pytest.skip("torch cannot load here - install the Microsoft Visual C++ 2015-2022 x64 redistributable")


def _draw_local_part(*, crack: bool, stain: bool, missing: bool, seed: int,
                     variant: str = "side") -> np.ndarray:
    """The part drawn upright; every variant is this same object, just transformed.

    ``variant`` models a genuinely different face of the component: ``side`` shows the
    two metal terminations, ``face`` is the plain ceramic body seen from the other axis.
    """
    w, h = 150, 100
    pad = 30
    canvas = np.full((h + 2 * pad, w + 2 * pad), BACKGROUND, np.uint8)
    x0, y0 = pad, pad

    if variant == "face":
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), 120, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + w - 6, y0 + h - 6), 95, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), 60, 2)
        # A corner chamfer: without an asymmetric fiducial the orientation of a plain
        # rectangle is genuinely undecidable, and the pipeline is right to refuse.
        pts = np.array([[x0 + w, y0], [x0 + w, y0 + 26], [x0 + w - 26, y0]], np.int32)
        cv2.fillPoly(canvas, [pts], BACKGROUND)
        cv2.polylines(canvas, [pts], True, 60, 2)
        cv2.circle(canvas, (x0 + int(w * 0.15), y0 + int(h * 0.25)), 6, 30, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), 205, -1)
        term = int(w * 0.18)
        cv2.rectangle(canvas, (x0, y0), (x0 + term, y0 + h), 235, -1)
        cv2.rectangle(canvas, (x0 + w - term, y0), (x0 + w, y0 + h), 235, -1)
        body_x0 = x0 + int(w * 0.24)
        cv2.rectangle(canvas, (body_x0, y0 + int(h * 0.06)),
                      (x0 + w - body_x0 + x0, y0 + h - int(h * 0.06)), 45, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), 60, 2)
        # polarity notch breaks the 180-degree symmetry so flip resolution is testable
        cv2.circle(canvas, (x0 + int(term * 0.5), y0 + int(h * 0.25)), 5, 120, -1, cv2.LINE_AA)

    if crack:
        pts = np.array([[x0 + int(w * 0.42), y0 + 4], [x0 + int(w * 0.62), y0 + h - 4]], np.int32)
        cv2.polylines(canvas, [pts], False, 12, 2, cv2.LINE_AA)
    if stain:
        cv2.circle(canvas, (x0 + int(w * 0.12), y0 + int(h * 0.7)), 8, 15, -1, cv2.LINE_AA)
    if missing:
        cv2.circle(canvas, (x0 + w, y0), 16, BACKGROUND, -1, cv2.LINE_AA)

    rng = np.random.default_rng(seed)
    return np.clip(canvas.astype(np.float32) + rng.normal(0, 3.0, canvas.shape), 0, 255).astype(np.uint8)


def make_part(path: Path, *, centre=(320, 256), angle: float = 0.0, scale: float = 1.0,
              crack: bool = False, stain: bool = False, missing: bool = False,
              variant: str = "side", seed: int = 0) -> Path:
    local = _draw_local_part(crack=crack, stain=stain, missing=missing, seed=seed, variant=variant)
    matrix = cv2.getRotationMatrix2D((local.shape[1] / 2, local.shape[0] / 2), angle, scale)
    rotated = cv2.warpAffine(local, matrix, (local.shape[1], local.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=BACKGROUND)
    out = np.full((512, 640), BACKGROUND, np.uint8)
    origin = (int(centre[0] - rotated.shape[1] / 2), int(centre[1] - rotated.shape[0] / 2))
    out[origin[1]:origin[1] + rotated.shape[0], origin[0]:origin[0] + rotated.shape[1]] = rotated
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)
    return path


@pytest.fixture
def make():
    return make_part


@pytest.fixture
def good_set(tmp_path: Path) -> list[Path]:
    # one camera, one face: quarter-turns plus a few degrees of placement tilt
    angles = [0.0, 90.0, 180.0, 270.0, 4.0, -5.0, 93.0, 182.0]
    return [
        make_part(
            tmp_path / f"good_{i:02d}.png",
            centre=(320 + (i % 3) * 8 - 8, 256 + (i % 2) * 6 - 3),
            angle=angles[i % len(angles)],
            scale=1.0 if i % 4 else 0.98,
            seed=i,
        )
        for i in range(8)
    ]


@pytest.fixture
def canonicaliser(config):
    from adkit.pipeline import Pipeline

    def _fn(paths, root):
        pipe = Pipeline(config, root=root)
        prepared = [pipe.transform(p, view_hint="synth") for p in paths]
        reference = prepared[0].image
        return [pipe.transform(p, view_hint="synth", reference=reference) for p in paths]

    return _fn


@pytest.fixture
def config(tmp_path: Path):
    from adkit.config import Config

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    cfg.raw["preprocess"]["canonical"]["size"] = [128, 128]
    cfg.raw["preprocess"]["view"]["assignment"] = "manual"
    cfg.raw["preprocess"]["view"]["default"] = "synth"
    for name in ("patchcore", "stefann"):
        cfg.raw["detectors"][name].update({"pretrained": False, "grid": [8, 8], "proj_dim": 32})
    cfg.raw["detectors"]["patchcore"]["coreset_ratio"] = 1.0
    cfg.raw["detectors"]["stefann"]["pca_dim"] = 4
    cfg.raw["calibration"]["min_good_per_view"] = 4
    return cfg
