"""Measure how consistently this rig canonicalises the same part.

Prints, per good image, how the part was located, which quarter-turn the marking band
chose, and how well the resulting canvas matches the anchor's. Writes the canvases into a
contact sheet, because the question "did the parts all land the same way up" is one to
answer by looking at, not by trusting a scalar.

    python tools/verify_alignment.py [view]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from adkit.config import Config
from adkit.data import GOOD, by_view, discover
from adkit.imaging import read_frame, write_image
from adkit.pipeline import Pipeline
from adkit.preprocess import canonicalize, find_roi


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 1e-9 else 0.0


def main() -> int:
    view = sys.argv[1] if len(sys.argv) > 1 else "tight"
    cfg = Config.load(ROOT / "config" / "default.yaml")
    pipe = Pipeline(cfg, root=ROOT)
    goods = by_view(discover(ROOT / "data"))[view][GOOD]
    if not goods:
        print(f"no good images under data/{view}")
        return 1

    model, anchor = pipe._teach_model(goods)
    gray = read_frame(goods[anchor].path).gray
    reference = canonicalize(gray, find_roi(gray, cfg, model), cfg, None)

    out = ROOT / "_orient" / view
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    print(f"anchor {goods[anchor].path.name}\n")
    print(f"{'image':>26} {'method':>13} {'peak':>5} {'rect':>18} {'turn':>4} "
          f"{'margin':>7} {'ambig':>6} {'ncc':>6}")
    for i, sample in enumerate(goods):
        g = read_frame(sample.path).gray
        roi = find_roi(g, cfg, model)
        canon = canonicalize(g, roi, cfg, reference.image)
        meta = canon.geometry.meta
        score = ncc(canon.image, reference.image)
        scores.append(score)
        (_, _), (rw, rh), angle = roi.rect
        write_image(out / f"c{i:02d}.png", (canon.image * 255).astype(np.uint8))
        print(f"{sample.path.name:>26} {roi.method:>13} {roi.peak:>5.2f} "
              f"{rw:>5.0f}x{rh:<5.0f}@{angle:>5.1f} {canon.geometry.turns:>4} "
              f"{meta['orientation_margin']:>7.3f} "
              f"{str(meta['orientation_ambiguous']):>6} {score:>6.3f}")

    values = np.asarray(scores)
    print(f"\nncc vs anchor: mean {values.mean():.3f} min {values.min():.3f} "
          f"count>=0.9 {int((values >= 0.9).sum())}/{len(values)}")

    thumbs = [cv2.resize(read_frame(out / f"c{i:02d}.png").gray, (190, 190))
              for i in range(len(goods))]
    cols = 7
    bands = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    width = max(b.shape[1] for b in bands)
    bands = [np.pad(b, ((0, 0), (0, width - b.shape[1])), constant_values=255) for b in bands]
    gap = np.full((6, width), 255, np.uint8)
    sheet = np.vstack([b for pair in zip(bands, [gap] * len(bands)) for b in pair])
    write_image(out / "sheet.png", sheet)
    print(f"contact sheet -> {out / 'sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
