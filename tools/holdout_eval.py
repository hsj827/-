"""Out-of-sample validation of a fitted view on real images.

Fit numbers are in-sample by definition: the cut is taken from the same images that
trained the model, so '0 false alarms on the goods' proves only that the percentile was
computed, not that the detector generalises. This tool fits on a fold and scores only the
images that fold never saw, which is the only measurement that can be compared against a
production rejection rate.

    python tools/holdout_eval.py --view tight --ng image/NG

Every score below is printed as a ratio against the reference that *that* detector
produced on its own training goods, because the image-level decision is exactly that
comparison. Dividing everything by the largest reference in the bundle hides it: a channel
whose reference is a floor can exceed itself at a raw score a larger channel ignores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adkit.config import Config                                  # noqa: E402
from adkit.data import GOOD, Sample, by_view, discover           # noqa: E402
from adkit.imaging import list_images                            # noqa: E402
from adkit.pipeline import Pipeline, Result                      # noqa: E402


def ratios(res: Result, margin: float) -> str:
    """Per-detector exceedance, with the ones that voted NG marked."""
    parts = []
    for name, value in res.per_detector.items():
        parts.append(f"{name}={value:5.2f}{'*' if value > margin else ' '}")
    return "  ".join(parts)


def evaluate(view: str, ng_paths: list[Path], folds: int,
             names: list[str] | None = None, goods_dir: Path | None = None,
             config: Path | None = None) -> int:
    cfg = Config.load(config or ROOT / "config/default.yaml")
    pipe = Pipeline(cfg, ROOT)
    if goods_dir is not None:
        # The data directory is what the operator curates, and it can legitimately hold a
        # different product from the one being measured. Point the goods at a directory
        # explicitly and the number describes that product.
        goods = [Sample(p, view, GOOD) for p in list_images(goods_dir)]
    else:
        grouped = by_view(discover(pipe.data_dir))
        goods = grouped.get(view, {}).get(GOOD, [])
    if len(goods) < 2 * folds:
        print(f"{view}: {len(goods)} good images, need at least {2 * folds} for {folds} folds")
        return 1
    names = names or cfg.enabled_detectors
    margin = float(cfg.get("calibration.image_margin", 1.15))

    n_good = n_fa = n_ng = n_caught = 0
    worst = ("", 0.0)
    fa_by_detector: dict[str, int] = {}
    caught_by: dict[str, int] = {}
    for fold in range(folds):
        held_idx = set(range(fold, len(goods), folds))
        held = [goods[i] for i in sorted(held_idx)]
        train = [s for i, s in enumerate(goods) if i not in held_idx]
        warn: list[str] = []
        bundle, _ = pipe._build_bundle(view, train, names, warn)  # noqa: SLF001
        pipe.bundles[view] = bundle

        print(f"\n=== fold {fold}: fit on {len(train)}, score {len(held)} unseen "
              f"+ {len(ng_paths)} NG ===  pixel_cut {bundle.calibration.pixel_cut:.2f}")
        for w in warn:
            print(f"  ! {w}")
        for s in held:
            res = pipe.predict(s.path, view)
            n_good += 1
            if res.image_score > worst[1]:
                worst = (s.path.name, res.image_score)
            if res.is_defect:
                n_fa += 1
                for name, value in res.per_detector.items():
                    if value > margin:
                        fa_by_detector[name] = fa_by_detector.get(name, 0) + 1
            srcs = sorted({src for d in res.defects for src in d.sources})
            label = "NG " if res.is_defect else "ok"
            print(f"  {label} {s.path.name[:34]:34s} ncc {bundle.ncc(res.canonical):.3f} "
                  f"defects={len(res.defects)} {srcs}")
            print(f"        {ratios(res, margin)}")

        for path in ng_paths:
            res = pipe.predict(path, view)
            n_ng += 1
            srcs = sorted({src for d in res.defects for src in d.sources})
            if res.is_defect:
                n_caught += 1
                for name, value in res.per_detector.items():
                    if value > margin:
                        caught_by[name] = caught_by.get(name, 0) + 1
            print(f"  {'NG ' if res.is_defect else 'MIS'} {path.name[:34]:34s} srcs={srcs}")
            print(f"        {ratios(res, margin)}")

    print("\n--- summary ---")
    print(f"folds={folds} detectors={names} image_margin={margin}")
    print(f"unseen goods false alarms: {n_fa}/{n_good} "
          f"(worst unseen good {worst[0]} at {worst[1]:.2f}x its own reference)")
    print(f"  who voted: {fa_by_detector or '-'}")
    print(f"NG caught: {n_caught}/{n_ng}")
    print(f"  who voted: {caught_by or '-'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--view", default="tight")
    ap.add_argument("--ng", default=None, help="directory of known-defective images")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--detectors", default=None,
                    help="comma-separated subset, e.g. crack,stain,missing")
    ap.add_argument("--config", default=None, help="config file to load instead of the default")
    ap.add_argument("--goods", default=None,
                    help="score a goods directory instead of data/<view>/good")
    args = ap.parse_args()

    ng_paths: list[Path] = []
    if args.ng:
        base = Path(args.ng)
        base = base if base.is_absolute() else Path(args.root) / base
        ng_paths = list_images(base)
    names = args.detectors.split(",") if args.detectors else None
    goods_dir = None
    if args.goods:
        goods_dir = Path(args.goods)
        goods_dir = goods_dir if goods_dir.is_absolute() else Path(args.root) / goods_dir
    return evaluate(args.view, ng_paths, args.folds, names, goods_dir,
                    Path(args.config) if args.config else None)


if __name__ == "__main__":
    raise SystemExit(main())
