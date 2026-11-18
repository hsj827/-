"""Score the *deployed* bundle on the split nobody tuned it on.

`fit` reports fold numbers, which are an estimate of what a fold-sized model does. This is the
other question: what the model that was actually shipped does on the official test goods and
the labelled defects.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from adkit.config import Config                                  # noqa: E402
from adkit.imaging import list_images                            # noqa: E402
from adkit.pipeline import Pipeline                              # noqa: E402

view, base = "bottle", ROOT / "bottle"
cfg = Config.load()
pipe = Pipeline(cfg, ROOT)
bundle = pipe.load(view)[view]
print(f"{view}: pixel_cut {bundle.calibration.pixel_cut:.4f} "
      f"extent_cut {bundle.calibration.extent_cut:.5f}")

rows = []
started = time.time()
for path in list_images(base / "test" / "good"):
    r = pipe.predict(path, view)
    rows.append((path.name, r.is_defect, r.image_score, [d.label for d in r.defects],
                 dict(r.per_detector)))
fa = [r for r in rows if r[1]]
print(f"\nunseen test goods: {len(fa)}/{len(rows)} judged defective "
      f"(worst {max(r[2] for r in rows):.3f})")
for name, bad, score, labels, ratios in fa:
    print(f"  ! {name} score {score:.3f} labels={labels} "
          + " ".join(f"{k}={v:.2f}" for k, v in sorted(ratios.items(), key=lambda kv: -kv[1])[:3]))

print("\nknown defects, by class:")
total = caught = boxed = 0
for kind in sorted(p.name for p in (base / "test").iterdir() if p.is_dir() and p.name != "good"):
    paths = list_images(base / "test" / kind)
    hits = 0
    labels: dict[str, int] = {}
    for path in paths:
        r = pipe.predict(path, view)
        total += 1
        if r.is_defect:
            hits += 1
            caught += 1
            if r.defects:
                boxed += 1
        for d in r.defects:
            labels[d.label] = labels.get(d.label, 0) + 1
    print(f"  {kind:16s} {hits:3d}/{len(paths):3d}   labels={labels}")
print(f"\ntotal {caught}/{total} caught, {boxed} of them with at least one box; "
      f"{time.time() - started:.0f}s for {len(rows) + total} images")
