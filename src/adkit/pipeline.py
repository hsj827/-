from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Config
from .data import GOOD, INBOX, NG, Sample, by_view, contact_sheet, discover, ensure_layout, write_manifest
from .detect import build_detector
from .detect.base import Detector
from .detect.features import Backbone
from .imaging import list_images, read_frame
from .postprocess import Calibration, calibrate, classify, fuse
from .preprocess import (ROI, ShapeModel, canonicalize, central_crop, classify_view,
                         find_roi, gradient_map, map_back, teach)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _coarse_box(shape: tuple[int, int], cfg: Config) -> tuple[int, int, int, int]:
    """The search window the shape model is taught inside.

    It must not come from find_roi: without a model that falls back to thresholding, which
    on these frames returns the vignette rather than the part and would teach a full-frame
    template. Edge energy alone already separates them.
    """
    h, w = shape
    reach = float(cfg.get("preprocess.roi.search_reach", 0.8))
    mx, my = int(w * (1 - reach) / 2), int(h * (1 - reach) / 2)
    return (mx, my, w - 2 * mx, h - 2 * my)


def _strided(items: range, want: int) -> list[int]:
    """Evenly spaced sample of `want` indices - deterministic, so fits are reproducible."""
    n = len(items)
    if want <= 0 or n <= want:
        return list(items)
    step = max(1, n // want)
    return [items[i] for i in range(0, n, step)][:want]


def _first_model(bundles: dict) -> ShapeModel | None:
    for bundle in bundles.values():
        if bundle.model is not None:
            return bundle.model
    return None


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    x, y = a - a.mean(), b - b.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12))


@dataclass
class Prepared:
    path: Path
    view: str
    label: str | None
    image: np.ndarray
    mask: np.ndarray
    roi: ROI
    geometry: Any
    annotation_ratio: float = 0.0

    @property
    def key(self) -> str:
        return self.path.stem


@dataclass
class ViewBundle:
    view: str
    detectors: list[Detector]
    calibration: Calibration
    reference: np.ndarray
    template_mask: np.ndarray
    fusion_method: str = "max"
    model: ShapeModel | None = None

    def ncc(self, image: np.ndarray) -> float:
        ref = self.reference
        if image.shape != ref.shape:
            image = cv2.resize(image, (ref.shape[1], ref.shape[0]))
        a, b = image - image.mean(), ref - ref.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
        return float((a * b).sum() / denom)


@dataclass
class Result:
    path: Path
    view: str
    is_defect: bool
    image_score: float
    per_detector: dict[str, float]
    defects: list[Any]
    canonical: np.ndarray
    fused_map: np.ndarray
    full_map: np.ndarray | None = None
    geometry: Any = None
    inert: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.path),
            "view": self.view,
            "is_defect": self.is_defect,
            "image_score": round(self.image_score, 4),
            "per_detector": {k: round(v, 4) for k, v in self.per_detector.items()},
            "uncalibrated": self.inert,
            "defects": [d.to_dict() for d in self.defects],
        }


class Pipeline:
    def __init__(self, cfg: Config | None = None, root: str | Path | None = None):
        self.cfg = cfg or Config.load()
        self.root = Path(root) if root else Path.cwd()
        self.data_dir = self.root / self.cfg.get("paths.data", "data")
        self.runs_dir = self.root / self.cfg.get("paths.runs", "runs")
        self.backbones: dict[tuple, Backbone] = {}
        self.bundles: dict[str, ViewBundle] = {}
        self.alignment: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ transform
    def _backbone(self, dcfg: dict[str, Any]) -> Backbone:
        key = (
            dcfg.get("backbone", "resnet18"),
            tuple(dcfg.get("layers", ["layer2", "layer3"])),
            bool(dcfg.get("pretrained", True)),
        )
        if key not in self.backbones:
            self.backbones[key] = Backbone(*key)
        return self.backbones[key]

    def transform(self, path: str | Path, view_hint: str | None = None,
                  reference: np.ndarray | None = None, label: str | None = None,
                  model=None) -> Prepared:
        path = Path(path)
        frame = read_frame(path, strip_annotation=bool(self.cfg.get("preprocess.annotation.strip", True)))
        reject = float(self.cfg.get("preprocess.annotation.reject_area_ratio", 0.02))
        if frame.annotation_ratio > reject:
            raise ValueError(
                f"{path.name}: {frame.annotation_ratio:.3%} annotated pixels - "
                "this looks like a marked-up screenshot, exclude it from the dataset"
            )
        roi = find_roi(frame.gray, self.cfg, model)
        view = classify_view(roi, self.cfg, view_hint)
        canon = canonicalize(frame.gray, roi, self.cfg, reference)
        return Prepared(path, view, label, canon.image, canon.mask, roi, canon.geometry,
                        annotation_ratio=frame.annotation_ratio)

    # ------------------------------------------------------------------ prepare
    def prepare(self, src: str | Path, views: list[str] | None = None) -> list[dict[str, Any]]:
        """Triage helper: canonicalise every image, guess its view, drop thumbnails for sorting."""
        rows: list[dict[str, Any]] = []
        paths = list_images(src)
        if not paths:
            raise FileNotFoundError(f"no images under {src}")
        for path in paths:
            try:
                prepared = self.transform(path)
            except Exception as exc:
                rows.append({"file": path.name, "status": f"error: {exc}"})
                continue
            rows.append({
                "file": path.name,
                "view": prepared.view,
                "bbox_aspect": round(prepared.roi.bbox_aspect, 3),
                "area_ratio": round(prepared.roi.area_ratio, 5),
                "fill": round(prepared.roi.fill, 3),
                "mean": round(prepared.roi.mean, 3),
                "edge_density": round(prepared.roi.edge_density, 4),
                "annotation_ratio": round(prepared.annotation_ratio, 5),
                "status": "ok",
            })

        detected = sorted({r["view"] for r in rows if r.get("status") == "ok"} | set(views or []))
        ensure_layout(self.data_dir, detected or ["default"])
        write_manifest(rows, self.runs_dir / "prepare_manifest.csv")

        ok = [r for r in rows if r.get("status") == "ok"]
        for view in detected:
            items = [(r["file"], Path(src) / r["file"]) for r in ok if r["view"] == view]
            contact_sheet(items, self.data_dir / view / INBOX / "contact_sheet.png")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "prepare_summary.json").write_text(
            json.dumps({"views": detected, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return rows

    # ------------------------------------------------------------------ fitting
    def _goods(self, view: str) -> list[Sample]:
        grouped = by_view(discover(self.data_dir))
        return grouped.get(view, {}).get(GOOD, [])

    def _teach_model(self, goods: list[Sample], view: str | None = None) -> tuple[ShapeModel, int]:
        """Teach the gradient shape model from whichever good image best registers the rest.

        The anchor is the frame every later image is aligned to, so a poor one is not a local
        problem: it lowers the registration of the whole good set, which raises the goods'
        own evidence, which raises every decision line. Measured on a 209-image view, moving
        from the best anchor (median ncc 0.845) to a mediocre one (0.814) cost 9 of 63 known
        defects, purely through the thresholds.

        It is therefore chosen on that measured quantity, in two stages, because the cheap
        proxy for it is not trustworthy: matching a central crop against other frames ranked
        candidates 159 / 070 / 058 depending on how many comparators were used, and its
        winner (159, ncc 0.813) registered *worse* than a candidate it did not pick first
        (048, 0.844). So the vote only shortlists; the ranking is done by actually teaching
        the model and registering a sample of goods against it.
        """
        if not goods:
            raise ValueError("no good images to build a shape model from")

        def gray_at(index: int) -> np.ndarray:
            return read_frame(goods[index].path).gray

        def gradient_at(index: int, factor: float) -> np.ndarray:
            g = gray_at(index)
            if factor < 1.0:
                g = cv2.resize(g, (max(1, int(g.shape[1] * factor)),
                                   max(1, int(g.shape[0] * factor))),
                               interpolation=cv2.INTER_AREA)
            return gradient_map(np.clip(g * 255.0, 0, 255).astype(np.uint8).astype(np.float32))

        anchor = 0
        self.anchor_quality = None
        n = len(goods)
        view = view or getattr(goods[0], "view", None)
        refine = int(self.cfg.get("preprocess.roi.anchor_refine", 10))
        probe_set = _strided(range(n), int(self.cfg.get("preprocess.roi.anchor_probe_images", 48)))

        def measured(index: int) -> float:
            """The median registration of the good set against `index` as the reference.

            This is the same work `fit` does two lines later, paid once on a sample instead of
            the whole set - and it is the only anchor criterion that measures what the anchor
            is actually used for.
            """
            candidate_model = teach(gray_at(index), self.cfg,
                                    _coarse_box(gray_at(index).shape, self.cfg))
            reference = self.transform(goods[index].path, view_hint=view,
                                       model=candidate_model).image
            qualities = []
            for j in probe_set:
                q = self.transform(goods[j].path, view_hint=view, reference=reference,
                                   model=candidate_model)
                qualities.append(_ncc(q.image, reference))
            return float(np.median(qualities))

        pinned = str(self.cfg.get("preprocess.roi.anchor_image", "") or "")
        if pinned:
            # Two things need the reference frame pinned while something else moves: an
            # operator who wants a refit to stay comparable with the model it replaces, and a
            # measurement that has to attribute a change to exactly one cause. The search is
            # skipped rather than overridden - ranking 209 candidates to then discard the
            # ranking is 6,688 image reads for nothing.
            hits = [i for i, s in enumerate(goods) if s.path.name == pinned]
            if not hits:
                raise ValueError(f"anchor_image {pinned!r} is not among this view's good images")
            anchor = hits[0]
            self.anchor_quality = measured(anchor) if n >= 3 else None
        elif n >= 3:
            ratio = float(self.cfg.get("preprocess.roi.template_crop", 0.26))
            factor = float(self.cfg.get("preprocess.roi.anchor_scale", 1.0))
            # Every image competes (one gradient map each); only the comparator set is
            # sampled. Cutting candidates was tried first and is the wrong knob: it cost 8 of
            # 63 known defects, because the medoid is rarely at a strided position and a
            # poorer anchor drags every registration - and therefore every threshold - down.
            candidates = _strided(range(n), int(self.cfg.get("preprocess.roi.anchor_candidates", 0)))
            comparators = set(_strided(range(n), int(self.cfg.get("preprocess.roi.anchor_comparators", 32))))
            score: dict[int, float] = {}
            for i in candidates:
                # The probe has to be smaller than the frames it is matched against, so
                # whole-frame versus whole-frame would give a degenerate 1x1 response and
                # every pair would be skipped, leaving the first image as the anchor.
                probe = np.ascontiguousarray(central_crop(gradient_at(i, factor), ratio))
                total = 0.0
                used = 0
                for j in comparators:
                    if i == j:
                        continue
                    other = gradient_at(j, factor)
                    if probe.shape[0] >= other.shape[0] or probe.shape[1] >= other.shape[1]:
                        continue
                    _, peak, _, _ = cv2.minMaxLoc(
                        cv2.matchTemplate(other, probe, cv2.TM_CCOEFF_NORMED))
                    total += float(peak)
                    used += 1
                score[i] = total / used if used else 0.0
            shortlist = sorted(score, key=lambda i: -score[i])[:refine]

            best, best_quality = shortlist[0], -1.0
            for i in shortlist:
                quality = measured(i)
                if quality > best_quality:
                    best, best_quality = i, quality
            anchor = best
            self.anchor_quality = best_quality

        base = gray_at(anchor)
        return teach(base, self.cfg, _coarse_box(base.shape, self.cfg)), anchor

    def fit(self, views: list[str] | None = None, detector_names: list[str] | None = None) -> dict[str, Any]:
        grouped = by_view(discover(self.data_dir))
        target_views = views or sorted(grouped)
        names = detector_names or self.cfg.enabled_detectors
        warn: list[str] = []
        notes: list[str] = []
        for view in target_views:
            goods = grouped.get(view, {}).get(GOOD, [])
            if not goods:
                warn.append(f"{view}: no good images - skipped")
                continue
            min_good = int(self.cfg.get("calibration.min_good_per_view", 30))
            if len(goods) < min_good:
                warn.append(f"{view}: only {len(goods)} good images (< {min_good}); model will be brittle")
            bundle, gate = self._build_bundle(view, goods, names, warn, notes)
            self.bundles[view] = bundle
            self._commit(bundle, gate)
        self._write_profile(target_views, names, warn, notes)
        return {"bundles": self.bundles, "warnings": warn, "notes": notes}

    def _build_bundle(self, view: str, goods: list[Sample], names: list[str],
                      warn: list[str], notes: list[str] | None = None,
                      ) -> tuple[ViewBundle, dict[str, Any]]:
        """Fit one view from its good images, without touching anything on disk.

        `warn` is for things that make the model wrong to deploy; `notes` is for what the
        fit measured about itself. They are separate because 'this model cannot be trusted
        on unseen parts yet' is true of every model at four good images and would otherwise
        drown the messages that are not.
        """
        notes = notes if notes is not None else []
        model, anchor = self._teach_model(goods, view)
        first_pass = [self.transform(s.path, view_hint=view, label=GOOD, model=model) for s in goods]
        reference = first_pass[anchor].image
        images = [self.transform(s.path, view_hint=view, reference=reference, label=GOOD,
                                 model=model) for s in goods]
        arrs = [p.image for p in images]
        masks = [p.mask for p in images]

        min_ncc = float(self.cfg.get("preprocess.alignment.min_ncc", 0.95))
        alignment = [_ncc(im, reference) for im in arrs]
        self.alignment[view] = alignment
        if getattr(self, "anchor_quality", None) is not None:
            # The anchor is not a detail: every decision line is set from how well the goods
            # register against it, so this number is the reason a threshold moved.
            notes.append(
                f"{view}: anchor {goods[anchor].path.name}, median good-to-reference ncc "
                f"{self.anchor_quality:.3f} over the sample"
            )
        # Orientation is the other thing an operator has to be told about, because the symptom
        # looks like a lighting problem. The turn is still chosen per image - it repairs the
        # corner ordering of a near-square bounding rectangle, which is a real job - but when
        # the marking band cannot confirm it, the part is being aligned to the reference rather
        # than to its own geometry, and that is only safe while the fixture holds the rotation.
        undecided = sum(1 for im in images
                        if (im.geometry.meta or {}).get("orientation_ambiguous"))
        if images and undecided >= max(2, 0.5 * len(images)):
            warn.append(
                f"{view}: {undecided}/{len(images)} good images have no orientation evidence "
                "of their own, so each one is aligned to the reference by trial turn. That "
                "holds only while the fixture keeps the part at a fixed rotation; if this "
                "product can arrive turned, it needs an asymmetric mark - more good images "
                "will not help"
            )
        values = np.asarray(alignment, dtype=np.float64)
        # The anchor is compared against itself, so it always scores 1.0; leaving it in
        # the population widens the spread and hides the images that genuinely differ.
        others = np.delete(values, anchor) if len(values) > 3 else values
        best = float(others.max())
        median = float(np.median(others))
        mad = float(np.median(np.abs(others - median))) * 1.4826
        # Two different faults, and only one can be claimed at a time. If the view can
        # align at all - something reaches min_ncc - then an image far below it is a
        # second face mixed in, which is actionable. If nothing reaches it, the whole
        # view is limited by the imaging, and calling that a mixed view would send the
        # operator to split data that is not mixed.
        gap = max(4.0 * mad, float(self.cfg.get("preprocess.alignment.min_gap", 0.15)))
        if best >= min_ncc:
            apart = sorted(v for v in values if v < best - gap)
            if apart:
                warn.append(
                    f"{view}: {len(apart)}/{len(values)} good images sit apart from the "
                    f"rest (worst {apart[0]:.3f}, the best reaches {best:.3f}). These are "
                    "probably a different face of the part - split them into their own "
                    "view or every classical detector will read the misalignment as a defect"
                )
        else:
            warn.append(
                f"{view}: good images only repeat at ncc {median:.3f} (best {best:.3f}, "
                f"below {min_ncc}). This is the imaging, not the model: defects whose "
                "signal is smaller than this shot-to-shot spread cannot be detected at this "
                "exposure, and more good images will not help - fix the lighting or the "
                "fixture first"
            )

        detectors = []
        for name in names:
            dcfg = dict(self.cfg.get(f"detectors.{name}", {}) or {})
            detector = build_detector(name, dcfg)
            if name in ("patchcore", "stefann"):
                detector.backbone = self._backbone(dcfg)
            detector.fit(arrs, masks)
            detectors.append(detector)

        template_mask = np.stack(masks).mean(axis=0) >= 0.5
        calibration = calibrate(detectors, arrs, masks, self.cfg, template_mask)
        # Say what the chosen quantile costs, in images, on this batch. A line set at p99 means
        # roughly one good part in a hundred crosses it, and the operator needs that stated:
        # the alternative - a maximum - reports zero false alarms and hides the recall it is
        # spending to get there.
        over = int(calibration.notes.get("goods_over_pixel_cut", 0))
        if arrs and over:
            owners = calibration.notes.get("crossing_by_detector") or {}
            who = ", ".join(f"{name} x{count}" for name, count in
                            sorted(owners.items(), key=lambda kv: -kv[1]))
            notes.append(
                f"{view}: decision lines set at p{calibration.notes.get('reference_percentile')}; "
                f"{over}/{len(arrs)} of these good images already cross them "
                f"(pixel line {calibration.pixel_cut:.2f}, extent line "
                f"{calibration.extent_cut:.5f}; batch maxima were "
                f"{calibration.notes.get('region_ref')} / "
                f"{calibration.notes.get('extent_ref')})"
            )
            if who:
                notes.append(
                    f"{view}: the evidence that puts those good images over the line comes from "
                    f"{who} - name the channel, because the remedy differs: a geometry channel "
                    "arguing with its own median outline is a framing problem, a learned one is "
                    "lighting or a second face"
                )
        bundle = ViewBundle(view, detectors, calibration, reference, template_mask,
                            str(self.cfg.get("fusion.method", "max")), model)
        gate = self._gate_metrics(bundle, [s.path for s in goods], self._ng_paths(view))
        gate.update({"median_ncc": median, "best_ncc": best})
        # The configured margin is kept deliberately. Estimating it by cross-validation was
        # tried and does not work at this data volume: leave-one-out saw almost no gap because
        # 13 of 14 images were still present, and a margin read off half-size folds absorbs
        # the small-model penalty - on the crack channel that variant took known defects from
        # 12/16 caught to 1/16 while unseen goods stayed clean, so it was only ever making the
        # cut blunter. What the folds *are* good for is reporting, so they report: the counts
        # below are the rates this fit would actually deliver on the next part off the line.
        hold = self._holdout_metrics(view, names, arrs, masks, reference, model,
                                      template_mask, detectors, calibration)
        if hold is None:
            notes.append(
                f"{view}: too few good images to hold any out, so every number in the gate is "
                f"in-sample and means nothing yet - it needs at least "
                f"{2 * int(self.cfg.get('calibration.min_images_per_fold', 4))} to be testable"
            )
        else:
            gate["holdout"] = hold
            # The held-out numbers are reported and deliberately not used to set the floors.
            # Taking the worse of "what the fit produced" and "what a fold's held-out parts
            # produced" was tried, and the second is inflated by the fold having fewer images:
            # on the benchmark it moved the peak floor from 1.80 to 3.00 and the extent floor
            # from 0.00816 to 0.029, against unseen parts that actually reach 1.24 and 0.0082.
            # On the production cans that cost two of the sixteen known defects (16/16 ->
            # 14/16). The fit's own leave-one-out statistics are the honest floor at this
            # sample size; the fold pass exists to say how well they hold.
            if hold["false_alarms"]:
                who = ", ".join(f"{k} x{v}" for k, v in
                                sorted(hold["by_detector"].items(), key=lambda kv: -kv[1]))
                ref_pct = float(self.cfg.get("calibration.reference_percentile", 99.0))
                notes.append(
                    f"{view}: {hold['false_alarms']}/{hold['n_scored']} good images the fold "
                    f"never saw were judged defective (votes: {who}). In sample the same number "
                    f"is {gate['false_alarms']}/{gate['n_good']}, which is not a contradiction - "
                    f"the line is the p{ref_pct} of the goods' own evidence, so about "
                    f"{round(100.0 - float(ref_pct), 1)}% of good parts are expected to cross it "
                    "and the held-out count is what an unseen part actually hits. Either raise "
                    "the margin, which costs recall, or add good images: more data sharpens a "
                    "quantile instead of pushing the line up."
                )
            if hold["ng_known"] and hold["ng_caught"] < hold["ng_known"]:
                notes.append(
                    f"{view}: {hold['ng_caught']}/{hold['ng_known']} known defects were caught "
                    "by a fold-sized model - a defect this system only catches because it "
                    "trained on similar parts is not detected, it is remembered"
                )
            elif not hold["ng_known"]:
                notes.append(
                    f"{view}: no NG images under data/{view}/ng, so out-of-sample recall is "
                    "untested and cannot be claimed"
                )
        return bundle, gate

    def _ng_paths(self, view: str) -> list[Path]:
        return [s.path for s in by_view(discover(self.data_dir)).get(view, {}).get(NG, [])]

    def _gate_metrics(self, bundle: ViewBundle, good_paths: list[Path],
                      ng_paths: list[Path]) -> dict[str, Any]:
        """What an online update is allowed to be judged on.

        False alarms on the goods alone cannot catch a poisoned update: feeding a defective
        image into the good set makes the model learn the defect as normal, and the symptom
        is that it starts missing defects, not that it starts flagging good ones. So the
        recall on the known NG images is checked in the same breath, and it is the term that
        has to hold.

        Both numbers here are in-sample - `good_paths` are exactly what the bundle trained
        on, so the false-alarm count measures the percentile, not the detector. They stay
        because they are the only pair the anchor and the candidate share a population on;
        the honest rates come from `_holdout_metrics` next to them.
        """
        return {
            "paths": [str(p) for p in good_paths],
            "n_good": len(good_paths),
            "false_alarms": sum(1 for p in good_paths if self._path_has_defect(bundle, p)),
            "ng_paths": [str(p) for p in ng_paths],
            "ng_caught": sum(1 for p in ng_paths if self._path_has_defect(bundle, p)),
        }

    def _holdout_metrics(self, view: str, names: list[str], arrs: list[np.ndarray],
                         masks: list[np.ndarray], reference: np.ndarray,
                         model: ShapeModel | None, template_mask: np.ndarray,
                         detectors: list[Detector], calibration: Calibration,
                         ) -> dict[str, Any] | None:
        """Judge the decision rule on images the fitted detectors never saw.

        Every other number in a gate is computed on the fit's own inputs, and that is worth
        nothing here: the image-level cut is a percentile of the maxima the training images
        produced, so with ten images it is approximately their maximum - and an unseen part
        exceeds a maximum of ten about as often as it does not. Measured on the production
        set: in-sample 0/14 false alarms, out-of-sample 7/14 at a deployment-sized fit, and
        the seven come from patchcore/stefann while the crack channel stays at 0/14. So the
        fold models score the held-out goods *and* the known NG images, and those counts are
        the ones the gate is allowed to act on.

        They are still never used to set a threshold - see the note in `_build_bundle`. The
        registration here is deliberately the deployed one: a fold cannot re-pick the
        reference, because production predicts with the committed reference.
        """
        per_fold = int(self.cfg.get("calibration.min_images_per_fold", 4))
        if len(arrs) < 2 * per_fold:
            return None
        # Fold count follows the data, so the model doing the scoring stays close to the
        # size it is deployed at. A fixed two folds on fourteen goods fits on seven and
        # reports the small-model penalty as if it were the generalisation gap.
        folds = max(2, min(4, len(arrs) // per_fold))
        ng: list[tuple[np.ndarray, np.ndarray]] = []
        for path in self._ng_paths(view):
            try:
                probe = self.transform(path, view_hint=view, reference=reference, model=model)
            except Exception:  # noqa: BLE001 - one unreadable NG image must not fail the fit
                continue
            ng.append((probe.image, probe.mask))
        out: dict[str, Any] = {"n_scored": 0, "false_alarms": 0,
                               "ng_known": len(ng), "ng_caught": 0, "folds": folds,
                               "by_detector": {}, "region_peak": 0.0, "extent_peak": 0.0,
                               "method": "folds"}
        probe = float(self.cfg.get("calibration.region_probe", 1.0))
        # k refits is the stronger estimate, but its price is `folds` full fits: at the 200
        # good images this project now also holds, that is twenty minutes for a number the fit
        # already implies. Past the budget the deployed model answers the same question in one
        # pass, scored self-excluded.
        if len(arrs) > int(self.cfg.get("calibration.max_fold_images", 60)):
            out["method"] = "leave-one-out"
            out["folds"] = 0
            return self._self_excluded_pass(detectors, calibration, arrs, masks, ng,
                                            template_mask, out, probe)
        for fold in range(folds):
            held = list(range(fold, len(arrs), folds))
            rest = [i for i in range(len(arrs)) if i not in held]
            fold_detectors = []
            for name in names:
                dcfg = dict(self.cfg.get(f"detectors.{name}", {}) or {})
                detector = build_detector(name, dcfg)
                if name in ("patchcore", "stefann"):
                    detector.backbone = self._backbone(dcfg)
                detector.fit([arrs[i] for i in rest], [masks[i] for i in rest])
                fold_detectors.append(detector)
            fold_calib = calibrate(fold_detectors, [arrs[i] for i in rest],
                                   [masks[i] for i in rest], self.cfg, template_mask)
            # An NG image is out-of-sample for every fold, so the first fold scores them
            # all; repeating it would multiply the recall denominator by the fold count.
            cases = ([(arrs[i], masks[i], False) for i in held]
                     + ([(im, m, True) for im, m in ng] if fold == 0 else []))
            for image, mask, is_defective in cases:
                fused = fuse({d.name: d.score_map(image, mask) for d in fold_detectors},
                             fold_calib, method=str(self.cfg.get("fusion.method", "max")),
                             weights=self.cfg.get("fusion.weights"))
                verdict = bool(fused.is_defect or classify(
                    fused, self.cfg, pixel_threshold=fold_calib.pixel_cut,
                    template_mask=template_mask, extent_cut=fold_calib.extent_cut))
                if is_defective:
                    out["ng_caught"] += int(verdict)
                    continue
                out["n_scored"] += 1
                # How much evidence a part nobody trained on actually produces, read at the
                # loose probe so that faint-but-broad defects are counted and not just loud
                # ones. These two numbers become the deployed cuts in `_build_bundle`.
                regions = classify(fused, self.cfg, pixel_threshold=probe,
                                   template_mask=template_mask, extent_cut=1e-9)
                out["region_peak"] = max(out["region_peak"],
                                         max((r.score for r in regions), default=0.0))
                out["extent_peak"] = max(out["extent_peak"],
                                         max((r.mass for r in regions), default=0.0))
                if verdict:
                    out["false_alarms"] += 1
                    for name, ratio in fused.per_detector.items():
                        if ratio > fold_calib.image_margin:
                            out["by_detector"][name] = out["by_detector"].get(name, 0) + 1
        return out

    def _self_excluded_pass(self, detectors: list[Detector], calibration: Calibration,
                            arrs: list[np.ndarray], masks: list[np.ndarray],
                            ng: list[tuple[np.ndarray, np.ndarray]],
                            template_mask: np.ndarray, out: dict[str, Any],
                            probe: float) -> dict[str, Any]:
        """Gate numbers from the deployed model, each good image scored without itself.

        `calibration_map` was introduced to keep the query out of the statistics that set the
        reference, and it answers the gate question in the same breath: scoring image i with
        its own contributions removed is the closest thing to an unseen part that costs no
        refit. It errs in the direction worth erring in - measured on the bottle benchmark the
        self-excluded goods peaked at 2.11 where genuinely unseen parts reached 1.24, so a
        rate it flags is not one the deployed model commits more often.
        """
        for i, (image, mask) in enumerate(zip(arrs, masks)):
            fused = fuse({d.name: d.calibration_map(image, mask, i) for d in detectors},
                         calibration, method=str(self.cfg.get("fusion.method", "max")),
                         weights=self.cfg.get("fusion.weights"))
            self._tally(fused, calibration, template_mask, out, probe, False)
        for image, mask in ng:
            fused = fuse({d.name: d.score_map(image, mask) for d in detectors}, calibration,
                         method=str(self.cfg.get("fusion.method", "max")),
                         weights=self.cfg.get("fusion.weights"))
            self._tally(fused, calibration, template_mask, out, probe, True)
        return out

    def _tally(self, fused, calibration, template_mask, out, probe,
               is_defective: bool) -> None:
        """One scored image into the holdout counters, for either estimator."""
        verdict = bool(fused.is_defect or classify(
            fused, self.cfg, pixel_threshold=calibration.pixel_cut,
            template_mask=template_mask, extent_cut=calibration.extent_cut))
        if is_defective:
            out["ng_caught"] += int(verdict)
            return
        out["n_scored"] += 1
        regions = classify(fused, self.cfg, pixel_threshold=probe,
                           template_mask=template_mask, extent_cut=1e-9)
        out["region_peak"] = max(out["region_peak"], max((r.score for r in regions), default=0.0))
        out["extent_peak"] = max(out["extent_peak"], max((r.mass for r in regions), default=0.0))
        if verdict:
            out["false_alarms"] += 1
            for name, ratio in fused.per_detector.items():
                if ratio > calibration.image_margin:
                    out["by_detector"][name] = out["by_detector"].get(name, 0) + 1

    def _has_defect(self, bundle: ViewBundle, image: np.ndarray, mask: np.ndarray) -> bool:
        maps = {d.name: d.score_map(image, mask) for d in bundle.detectors}
        fused = fuse(maps, bundle.calibration, method=bundle.fusion_method,
                     weights=self.cfg.get("fusion.weights"))
        return bool(classify(fused, self.cfg, pixel_threshold=bundle.calibration.pixel_cut,
                             template_mask=bundle.template_mask,
                             extent_cut=bundle.calibration.extent_cut))

    # ------------------------------------------------------------------ versions
    def _versions_dir(self, view: str) -> Path:
        return self.runs_dir / view / "versions"

    def _current_version(self, view: str) -> str | None:
        pointer = self.runs_dir / view / "current.json"
        if not pointer.exists():
            return None
        try:
            return str(json.loads(pointer.read_text(encoding="utf-8"))["version"])
        except (ValueError, KeyError):
            return None

    def _view_dir(self, view: str) -> Path:
        version = self._current_version(view)
        if version is None:
            return self.runs_dir / view
        return self._versions_dir(view) / version

    def _commit(self, bundle: ViewBundle, gate: dict[str, Any]) -> Path:
        """Write a new version and only then move the pointer onto it.

        The pointer is written last so a crash mid-fit leaves the previous version current
        rather than pointing at a half-written bundle.
        """
        root = self.runs_dir / bundle.view
        existing = [p.name for p in self._versions_dir(bundle.view).glob("v*")] \
            if self._versions_dir(bundle.view).exists() else []
        numbers = [int(name[1:]) for name in existing if name[1:].isdigit()]
        version = f"v{max(numbers, default=0) + 1}"
        view_dir = self._versions_dir(bundle.view) / version
        self._write_bundle(bundle, view_dir)
        # The gate belongs to the version, not to the pointer: it names the anchor images and
        # the rates measured against them. Without a per-copy record, a rollback moves the
        # model and leaves the anchor describing whatever was published last - a can model
        # gated against 209 bottle parts, whose median ncc differs enough to make every
        # later update decision meaningless.
        (view_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        (root / "anchor.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        (root / "current.json").write_text(
            json.dumps({"version": version, "written": _now(), "gate": gate}, indent=2),
            encoding="utf-8")
        return view_dir

    def _write_bundle(self, bundle: ViewBundle, view_dir: Path) -> None:
        view_dir.mkdir(parents=True, exist_ok=True)
        for detector in bundle.detectors:
            detector.save(view_dir / "detectors")
        bundle.calibration.save(view_dir / "calibration.json")
        np.save(view_dir / "reference.npy", bundle.reference.astype(np.float32))
        np.save(view_dir / "template_mask.npy", bundle.template_mask.astype(np.uint8))
        if bundle.model is not None:
            model = bundle.model
            np.savez_compressed(view_dir / "shape_model.npz",
                                template=model.template.astype(np.float32),
                                centre=np.array(model.centre, np.float32),
                                angles=np.array(model.angles),
                                grey=model.grey.astype(np.float32))
        (view_dir / "meta.json").write_text(
            json.dumps({"view": bundle.view, "detectors": [d.name for d in bundle.detectors],
                        "fusion": bundle.fusion_method,
                        "alignment": self.alignment.get(bundle.view, [])}, indent=2), encoding="utf-8"
        )

    def versions(self, view: str) -> list[dict[str, Any]]:
        root = self._versions_dir(view)
        if not root.exists():
            return []
        current = self._current_version(view)
        out = []
        for directory in sorted(root.iterdir()):
            if not (directory / "meta.json").exists():
                continue
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
            gate = {}
            if (directory / "gate.json").exists():
                gate = json.loads((directory / "gate.json").read_text(encoding="utf-8"))
            hold = gate.get("holdout") or {}
            out.append({"version": directory.name, "current": directory.name == current,
                        "detectors": meta.get("detectors", []),
                        "n_good": gate.get("n_good"),
                        "holdout_fa": hold.get("false_alarms"),
                        "holdout_n": hold.get("n_scored")})
        return out

    def rollback(self, view: str, version: str) -> bool:
        root = self._versions_dir(view) / version
        if not (root / "meta.json").exists():
            return False
        gate_path = root / "gate.json"
        (self.runs_dir / view / "current.json").write_text(
            json.dumps({"version": version, "written": _now(), "rolled_back": True,
                        "gate": json.loads(gate_path.read_text(encoding="utf-8"))
                        if gate_path.exists() else None}, indent=2), encoding="utf-8")
        anchor = self.runs_dir / view / "anchor.json"
        if gate_path.exists():
            anchor.write_text(gate_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # A version published before gates were stored per version. Keeping the previous
            # anchor would be worse than having none: `update` would compare a new fit against
            # another model's population. With none, it refuses and says run fit.
            anchor.unlink(missing_ok=True)
        self.bundles.pop(view, None)
        self.load(view)
        return True

    # ------------------------------------------------------------------ online update
    def update(self, views: list[str] | None = None,
               detector_names: list[str] | None = None) -> dict[str, Any]:
        """Re-fit from the current data, but only publish it if the anchor set agrees.

        The anchor set is the good images the accepted version was fitted on and it never
        grows, because a gate measured against a moving population proves nothing. Feeding
        back a wrong confirmation moves the boundary toward the defect, and without this
        check that drift is invisible until a customer reports it.
        """
        grouped = by_view(discover(self.data_dir))
        target_views = views or sorted(grouped)
        names = detector_names or self.cfg.enabled_detectors
        tolerance = float(self.cfg.get("calibration.update_ncc_tolerance", 0.02))
        report: dict[str, Any] = {"accepted": [], "rejected": [], "warnings": [], "notes": []}

        for view in target_views:
            goods = grouped.get(view, {}).get(GOOD, [])
            if not goods:
                report["warnings"].append(f"{view}: no good images - nothing to update")
                continue
            baseline = self._anchor_gate(view)
            if baseline is None:
                report["warnings"].append(f"{view}: no fitted version to gate against - run fit")
                continue

            warn: list[str] = []
            notes: list[str] = []
            bundle, gate = self._build_bundle(view, goods, names, warn, notes)
            report["warnings"] += warn
            report["notes"] += notes

            worse_alarms = gate["false_alarms"] > baseline["false_alarms"]
            lost_recall = gate["ng_caught"] < baseline.get("ng_caught", 0)
            worse_align = gate["median_ncc"] < baseline["median_ncc"] - tolerance
            worse_holdout = self._worse_holdout(gate, baseline)
            if worse_alarms or lost_recall or worse_align or worse_holdout:
                if lost_recall:
                    reason = (f"known defects it used to catch are now missed "
                              f"({baseline['ng_caught']} -> {gate['ng_caught']} of "
                              f"{len(gate['ng_paths'])}) - the good set has absorbed a defect")
                elif worse_holdout:
                    reason = worse_holdout
                elif worse_alarms:
                    reason = (f"more good images now flagged defective "
                              f"({baseline['false_alarms']} -> {gate['false_alarms']})")
                else:
                    reason = (f"anchor alignment dropped {baseline['median_ncc']:.3f} -> "
                              f"{gate['median_ncc']:.3f}")
                report["rejected"].append({"view": view, "reason": reason, "gate": gate})
                continue

            self.bundles[view] = bundle
            self._commit(bundle, gate)
            report["accepted"].append({"view": view, "n_good": gate["n_good"], "gate": gate})
            if not gate["ng_paths"]:
                # Say it out loud: with no known defects the strongest guard is not running.
                report["warnings"].append(
                    f"{view}: update accepted, but no NG images exist for this view so the "
                    "recall guard was vacuous - a poisoned good set would have passed")
        return report

    @staticmethod
    def _worse_holdout(gate: dict[str, Any], baseline: dict[str, Any]) -> str | None:
        """Reject an update that alarms more often on parts it never trained on.

        The in-sample term cannot see this failure: an extra good image tightens the
        percentile it is measured against, so a model that has started to flag ordinary
        parts can still score a perfect 0 on its own inputs. Held-out counts are computed
        by the same fold protocol on both sides, so they are comparable even though the
        populations differ - which is also why only a strict increase is treated as worse.
        """
        now, was = gate.get("holdout"), baseline.get("holdout")
        if not now or not was:
            return None
        if now["false_alarms"] * was["n_scored"] > was["false_alarms"] * now["n_scored"]:
            return (f"more of the parts it never trained on are now flagged defective "
                    f"({was['false_alarms']}/{was['n_scored']} -> "
                    f"{now['false_alarms']}/{now['n_scored']})")
        if (now["ng_known"] and was["ng_known"]
                and now["ng_caught"] * was["ng_known"] < was["ng_caught"] * now["ng_known"]):
            return (f"out-of-sample recall on the known defects dropped "
                    f"({was['ng_caught']}/{was['ng_known']} -> {now['ng_caught']}/{now['ng_known']})")
        return None

    def _anchor_gate(self, view: str) -> dict[str, Any] | None:
        path = self.runs_dir / view / "anchor.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["paths"] = [p for p in payload.get("paths", []) if Path(p).exists()]
        return payload or None

    def _path_has_defect(self, bundle: ViewBundle, path: Path) -> bool:
        prepared = self.transform(path, view_hint=bundle.view, reference=bundle.reference,
                                  model=bundle.model)
        return self._has_defect(bundle, prepared.image, prepared.mask)

    def load(self, view: str | None = None) -> dict[str, ViewBundle]:
        if view:
            views = [view]
        elif self.runs_dir.exists():
            views = [p.name for p in sorted(self.runs_dir.iterdir())
                     if (self._versions_dir(p.name) / "v1" / "meta.json").exists()]
        else:
            views = []
        for name in views:
            view_dir = self._view_dir(name)
            meta = json.loads((view_dir / "meta.json").read_text(encoding="utf-8"))
            self.alignment[name] = [float(v) for v in meta.get("alignment", [])]
            detectors = []
            for det_name in meta["detectors"]:
                dcfg = dict(self.cfg.get(f"detectors.{det_name}", {}) or {})
                detector = build_detector(det_name, dcfg)
                if det_name in ("patchcore", "stefann"):
                    detector.backbone = self._backbone(dcfg)
                detector.load(view_dir / "detectors")
                detectors.append(detector)
            self.bundles[name] = ViewBundle(
                view=name,
                detectors=detectors,
                calibration=Calibration.load(view_dir / "calibration.json"),
                reference=np.load(view_dir / "reference.npy"),
                template_mask=np.load(view_dir / "template_mask.npy").astype(bool),
                fusion_method=meta.get("fusion", "max"),
                model=self._load_model(view_dir),
            )
        return self.bundles

    def _load_model(self, view_dir: Path) -> ShapeModel | None:
        path = view_dir / "shape_model.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            keys = set(data.files)
            template = np.array(data["template"], dtype=np.float32)
            centre = tuple(float(v) for v in data["centre"])
            angles = tuple(int(a) for a in data["angles"])
            grey = np.array(data["grey"], np.float32) if "grey" in keys else None
        scfg = self.cfg.get("preprocess.shape") or {}
        return ShapeModel(template=template, centre=centre, angles=angles, grey=grey,
                          min_points=int(scfg.get("min_points", 24)))

    def _write_profile(self, views: list[str], names: list[str], warn: list[str],
                       notes: list[str] | None = None) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "views": views,
            "detectors": names,
            "canonical_size": list(self.cfg.get("preprocess.canonical.size")),
            "warnings": warn,
            "validation_notes": list(notes or []),
            "n_good": {v: len(self._goods(v)) for v in views if v in self.bundles},
            "n_ng": {v: len(by_view(discover(self.data_dir)).get(v, {}).get(NG, [])) for v in views},
        }
        (self.runs_dir / "profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ inference
    def select_view(self, image: np.ndarray) -> str:
        if not self.bundles:
            raise RuntimeError("no fitted views - run `adkit fit` first")
        return max(self.bundles, key=lambda v: self.bundles[v].ncc(image))

    def predict(self, path: str | Path, view: str | None = None) -> Result:
        path = Path(path)
        if not self.bundles:
            self.load()
        if view is None:
            view = self.select_view(self.transform(path, model=_first_model(self.bundles)).image)
        bundle = self.bundles[view]
        prepared = self.transform(path, view_hint=view, reference=bundle.reference,
                                  model=bundle.model)

        score_maps = {d.name: d.score_map(prepared.image, prepared.mask) for d in bundle.detectors}
        fused = fuse(
            score_maps, bundle.calibration, method=bundle.fusion_method,
            weights=self.cfg.get("fusion.weights"),
        )
        defects = classify(fused, self.cfg, pixel_threshold=bundle.calibration.pixel_cut,
                           template_mask=bundle.template_mask,
                           extent_cut=bundle.calibration.extent_cut)
        # A region the goods never produce is evidence on its own. Without this the verdict
        # rested only on each detector's single image-level peak, which a stray noise blob on
        # a good part can beat - that is how a detected crack still came out OK.
        verdict = fused.is_defect or bool(defects)
        shape = prepared.geometry.source_shape if prepared.geometry.source_shape else             (int(prepared.image.shape[0]), int(prepared.image.shape[1]))
        full = map_back(fused.map, prepared.geometry, shape)
        return Result(
            path=path, view=view, is_defect=verdict, image_score=fused.image_score,
            per_detector=fused.per_detector, defects=defects, canonical=prepared.image,
            fused_map=fused.map, full_map=full, geometry=prepared.geometry, inert=fused.inert,
        )

    def predict_dir(self, src: str | Path, view: str | None = None) -> list[Result]:
        return [self.predict(p, view) for p in list_images(src)]
