"""Measure whether the calibrated decision rule generalises, on data that can answer it.

The production can set has 14 good images that repeat at ncc 0.67, so a held-out part
exceeds the goods' own score maximum about half the time, and the measurement cannot tell a
wrong statistic apart from an unestimable one. This set can: 209 goods from a fixed camera,
defects labelled with pixel masks, and a test split no fit ever sees.

    python tools/mvtec_eval.py --sizes 20,50,100

Read the two numbers together. Out-of-sample false alarms that fall as the fit grows say the
percentile is a maximum in disguise and more goods fix it. False alarms that stay flat while
the goods repeat at ncc 0.99 say the reference itself is wrong and no amount of data helps.
Pixel AUROC is reported per detector because an image-level verdict can be right for the
wrong reason: a channel that separates goods from defects globally while its map peaks
anywhere says nothing about localisation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adkit.config import Config                                  # noqa: E402
from adkit.data import GOOD, Sample                              # noqa: E402
from adkit.pipeline import Pipeline, ViewBundle                  # noqa: E402
from adkit.postprocess import classify, fuse                     # noqa: E402
from adkit.preprocess import map_back                            # noqa: E402


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank statistic: does the map put the labelled defect pixels above the rest?

    `labels` decides which pixels are the positives - never a property of `scores`. Tying on
    the map instead is how this function first shipped: a channel that is silent (all zeros)
    then looks like it has no negatives at all and returns NaN, while a channel that merely
    lights up everywhere returns 1.0. Ties between equal scores get average ranks so a
    constant map correctly comes out at 0.5 rather than perfect.
    """
    flat = scores.astype(np.float64).ravel()
    lab = labels.astype(bool).ravel()
    positives, negatives = int(lab.sum()), int((~lab).sum())
    if flat.size == 0 or positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty(flat.size, np.float64)
    values, start = flat[order], 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[end] == values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[lab].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def read_mask(path: Path) -> np.ndarray:
    """Ground-truth masks live under the project's non-ASCII path; cv2.imread cannot open
    those, and it returns None rather than raising, so the project's own decoder is used."""
    from adkit.imaging import _imread_unicode
    raw = _imread_unicode(path)
    if raw is None:
        raise OSError(f"cannot decode ground-truth mask: {path}")
    return (raw[..., 0] if raw.ndim == 3 else raw) > 127


def region_table(fused, mask, base: float) -> list[tuple[float, float]]:
    """(area as a fraction of the part, peak map value) for every region above `base`.

    A single pixel cut has to be one number, so it is set by whichever good pixel is loudest
    - and a crack, which is long and loud over a handful of pixels, then shares the line with
    a spill, which is faint but covers a tenth of the part. Those two live in different places
    in the (size, peak) plane, so the plane is printed before picking a rule.
    """
    binary = (fused.map >= base).astype(np.uint8) * mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    total = float(mask.sum()) or 1.0
    out = []
    for i in range(1, count):
        component = labels == i
        area = float(component.sum())
        if area < 4:
            continue
        out.append((area / total, float(fused.map[component].max())))
    return sorted(out, reverse=True)


def envelope(rows: list[list[tuple[float, float]]], sizes: list[float]) -> list[float]:
    """Loudest peak any image produced at each region size or bigger."""
    out = []
    for size in sizes:
        peaks = [peak for regions in rows for area, peak in regions if area >= size]
        out.append(max(peaks) if peaks else float("nan"))
    return out


class Scorer:
    """One fitted view, scored the way `predict` scores it, with the pieces kept apart."""

    def __init__(self, pipe: Pipeline, bundle: ViewBundle):
        self.pipe, self.bundle = pipe, bundle
        self.margin = float(pipe.cfg.get("calibration.image_margin", 1.15))

    def __call__(self, path: Path, gt: Path | None = None) -> dict:
        bundle = self.bundle
        prepared = self.pipe.transform(path, view_hint=bundle.view, reference=bundle.reference,
                                       model=bundle.model)
        maps = {d.name: d.score_map(prepared.image, prepared.mask) for d in bundle.detectors}
        fused = fuse(maps, bundle.calibration, method=bundle.fusion_method,
                     weights=self.pipe.cfg.get("fusion.weights"))
        # exactly the call Pipeline.predict makes, including the extent line - a harness that
        # forgets it measures a rule nobody deployed
        regions = classify(fused, self.pipe.cfg, pixel_threshold=bundle.calibration.pixel_cut,
                           template_mask=bundle.template_mask,
                           extent_cut=bundle.calibration.extent_cut)
        row = {"verdict": bool(fused.is_defect or regions), "regions": len(regions),
               "envelope": region_table(fused, prepared.mask, 1.0),
               "ratios": dict(fused.per_detector),
               # 1.0 is by construction "as extreme as the loudest pixel any good image
               # produced", so clipping the map at it and averaging over the part asks a
               # question the maximum cannot answer: how *much* of this part is past the good
               # extreme, not how loud its single loudest pixel is. A registration spike of a
               # dozen pixels at 2.1 and a faint spill over 5% of the bottle at 1.2 are the
               # same number under a max, and very different people under this one.
               "excess": float(np.mean(np.clip(fused.map - 1.0, 0.0, None)[prepared.mask])),
               "voters": [n for n, v in fused.per_detector.items() if v > self.margin]}
        if gt is not None:
            truth = read_mask(gt)
            shape = truth.shape
            # Ranks and thresholds are different questions, and only one of them is answered
            # by AUROC. A map can rank the defect pixels top of the image - AUROC 0.98 - and
            # still never reach the region cut, which is a calibration failure masquerading as
            # a detection failure. These two numbers separate them: `gt_peak` is what the map
            # reaches *on the defect*, `off_peak` is what it reaches somewhere else. If
            # off_peak > gt_peak the localisation is wrong; if both are under the cut the
            # scale is.
            inside = map_back(prepared.mask.astype(np.float32), prepared.geometry,
                              shape) > 0.5
            full = map_back(fused.map, prepared.geometry, shape)
            over = truth & inside
            row["gt_peak"] = float(full[over].max()) if over.any() else float("nan")
            row["off_peak"] = float(full[inside & ~truth].max()) if (inside & ~truth).any() else 0.0
            row["cut"] = bundle.calibration.pixel_cut
            row["auroc"] = {name: auroc(map_back(m.score, prepared.geometry, shape)[inside],
                                        truth[inside]) for name, m in maps.items()}
            row["auroc"]["fused"] = auroc(full[inside], truth[inside])
        return row


def dataset_paths(base: Path) -> tuple[list[Path], list[Path], dict[str, list[tuple[Path, Path]]]]:
    train = sorted((base / "train" / "good").glob("*.png"))
    test = sorted((base / "test" / "good").glob("*.png"))
    classes = {}
    for kind in sorted(p for p in (base / "test").iterdir() if p.is_dir() and p.name != "good"):
        classes[kind.name] = [
            (p, base / "ground_truth" / kind.name / f"{p.stem}_mask.png")
            for p in sorted(kind.glob("*.png"))
            if (base / "ground_truth" / kind.name / f"{p.stem}_mask.png").exists()]
    return train, test, classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bottle")
    ap.add_argument("--view", default="bottle")
    ap.add_argument("--sizes", default="20,50,100", help="goods counts to fit on")
    ap.add_argument("--detectors", default=None)
    ap.add_argument("--holdout", type=int, default=20, help="unseen goods to score")
    ap.add_argument("--per-class", type=int, default=0, help="cap NG images per class")
    ap.add_argument("--config", default=None, help="config file to load instead of the default")
    ap.add_argument("--no-folds", action="store_true",
                    help="skip the fold protocol (cuts then come from the fit alone)")
    ap.add_argument("--detail", action="store_true",
                    help="print what the map reached on the defect for every missed image")
    args = ap.parse_args()

    base = Path(args.dataset) if Path(args.dataset).is_absolute() else ROOT / args.dataset
    train, test_good, classes = dataset_paths(base)

    cfg = Config.load(Path(args.config) if args.config else ROOT / "config/default.yaml")
    cfg.raw["preprocess"]["view"].update({"assignment": "manual", "default": args.view})
    # The fold protocol is exactly what this dataset replaces: the test split is a real
    # holdout, and refitting in folds on 200 goods would only cost an hour.
    # The fold protocol stays on by default: it is what sets the deployed region and extent
    # cuts in production, and the folds only ever see train goods, so the official test split
    # is still a third set nobody has tuned anything on. --no-folds measures the fit-only
    # estimate instead, which is a different and knowingly higher number.
    if args.no_folds:
        cfg.raw["calibration"]["min_images_per_fold"] = 10 ** 6
    names = args.detectors.split(",") if args.detectors else cfg.enabled_detectors

    print(f"{base}: {len(train)} train goods, {len(test_good)} test goods, "
          f"{ {k: len(v) for k, v in classes.items()} }, detectors={names}")

    for size in [int(s) for s in args.sizes.split(",")]:
        pipe = Pipeline(cfg, ROOT)
        goods = [Sample(p, args.view, GOOD) for p in train[:size]]
        started = time.time()
        warn: list[str] = []
        bundle, _ = pipe._build_bundle(args.view, goods, names, warn)  # noqa: SLF001
        pipe.bundles[args.view] = bundle
        fit_s = time.time() - started
        scorer = Scorer(pipe, bundle)

        probe = [pipe.transform(p, view_hint=args.view, reference=bundle.reference,
                                model=bundle.model) for p in test_good[:args.holdout]]
        alignment = [bundle.ncc(p.image) for p in probe]
        in_sample = [bundle.ncc(pipe.transform(s.path, view_hint=args.view,
                                               reference=bundle.reference,
                                               model=bundle.model).image) for s in goods]
        rows = [scorer(p) for p in test_good[:args.holdout]]
        flagged = [r for r in rows if r["verdict"]]
        voters: dict[str, int] = {}
        for r in flagged:
            for name in r["voters"]:
                voters[name] = voters.get(name, 0) + 1
        worst: dict[str, float] = {}
        for r in rows:
            for name, value in r["ratios"].items():
                worst[name] = max(worst.get(name, 0.0), value)
        print(f"\n=== fit on {size} goods ({fit_s:.0f}s) · {len(rows)} unseen goods ===")
        for w in warn:
            print(f"  ! {w}")
        # The two hypotheses look identical in the tail and differ in the body. If the goods
        # the fit *did* see also score low in the middle of their distribution, the reference
        # is biased by self-inclusion - the query's own features sit in the memory bank, its
        # own pixels are in the PCA - and no number of images removes it. If only the top
        # differs, the estimate is merely short of samples.
        seen = [scorer(s.path) for s in goods[:len(rows)]]
        print("  score distribution (ratio to the trained reference), each side once:")
        for name in sorted(names):
            body = np.array([[r["ratios"].get(name, 0.0) for r in seen],
                             [r["ratios"].get(name, 0.0) for r in rows]])
            print(f"    {name:10s} trained median {np.median(body[0]):.2f} "
                  f"max {body[0].max():.2f}  | unseen median {np.median(body[1]):.2f} "
                  f"max {body[1].max():.2f}")

        # Held-out registration is stated next to the alarm rate on purpose: if the unseen
        # goods align as well as the trained ones and still alarm, the statistic is wrong. If
        # they align worse, part of the rate is the pose model and not the threshold.
        print(f"  ncc: trained goods median {float(np.median(in_sample)):.3f}, "
              f"unseen goods median {float(np.median(alignment)):.3f}")
        print(f"  out-of-sample false alarms: {len(flagged)}/{len(rows)}  voters={voters or '-'}")
        print("  worst unseen good's ratio per channel: "
              + "  ".join(f"{k}={v:.2f}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1]))
              + f"   (margin {cfg.get('calibration.image_margin')})")
        cal = bundle.calibration
        print(f"  deployed cuts: peak {cal.pixel_cut:.2f}, extent {cal.extent_cut:.5f}   "
              f"fit-only estimate was {cal.notes.get('region_ref', 0):.2f} / "
              f"{cal.notes.get('extent_ref', 0):.5f}")
        # The candidate replacement for the single pixel cut: the loudest extent any unseen
        # good produces is the floor any defect statistic has to clear, so it is printed next
        # to every class rather than averaged away.
        goods_excess = max(r["excess"] for r in rows)
        print(f"  extent statistic: unseen goods max {goods_excess:.4f}, "
              f"median {np.median([r['excess'] for r in rows]):.4f}")

        sizes = [0.0002, 0.001, 0.005, 0.02, 0.05]
        goods_env = envelope([r["envelope"] for r in rows], sizes)
        print("  loudest peak at each region size (fraction of the part):")
        print("    " + "  ".join(f">={s:.4f}:{v:.2f}" for s, v in zip(sizes, goods_env))
              + "   <- unseen goods")

        for kind, pairs in classes.items():
            subset = pairs[:args.per_class] if args.per_class else pairs
            scored = [(p, scorer(p, g)) for p, g in subset]
            caught = sum(1 for _, r in scored if r["verdict"])
            mean = {name: float(np.nanmean([r["auroc"][name] for _, r in scored]))
                    for name in scored[0][1]["auroc"]}
            ranked = "  ".join(f"{k}={v:.3f}" for k, v in sorted(mean.items(),
                                                                 key=lambda kv: -kv[1]))
            defect_env = envelope([r["envelope"] for _, r in scored], sizes)
            print(f"  {kind:16s} verdict {caught:3d}/{len(scored):3d}   pixel AUROC {ranked}"
                  f"   excess goods-max {goods_excess:.4f} vs defect median "
                  f"{np.median([r['excess'] for _, r in scored]):.4f}")
            print("    " + "  ".join(f">={s:.4f}:{v:.2f}" for s, v in zip(sizes, defect_env))
                  + "   <- this class")
            if args.detail:
                for path, r in scored:
                    top = r["envelope"][0] if r["envelope"] else (0.0, 0.0)
                    flag = "ok  " if r["verdict"] else "MISS"
                    print(f"     {flag} {path.name:12s} regions={r['regions']} "
                          f"extent {r['excess']:.4f} largest region {top[0]*100:.2f}% of part "
                          f"at peak {top[1]:.2f} (cut {r.get('cut', float('nan')):.2f}) "
                          f"peak-on-defect {r['gt_peak']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
