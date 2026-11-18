from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adkit.data import GOOD, NG, Sample, discover
from adkit.detect import build_detector
from adkit.pipeline import Pipeline


@pytest.fixture
def dataset(make, tmp_path):
    good_dir = tmp_path / "data" / "synth" / GOOD
    ng_dir = tmp_path / "data" / "synth" / NG / "crack"
    good_dir.mkdir(parents=True)
    ng_dir.mkdir(parents=True)
    angles = [0.0, 90.0, 180.0, 270.0, 4.0, -5.0, 93.0, 182.0]
    for i in range(8):
        make(good_dir / f"g{i:02d}.png", angle=angles[i], seed=i)
    for i in range(3):
        make(ng_dir / f"crack{i}.png", crack=True, angle=[0.0, 90.0, 182.0][i], seed=100 + i)
    return tmp_path


def test_prepare_reports_views_and_writes_manifest(make, tmp_path, config):
    src = tmp_path / "raw"
    src.mkdir()
    for i in range(5):
        make(src / f"r{i}.png", seed=i)
    pipe = Pipeline(config, root=tmp_path)
    rows = pipe.prepare(src)
    assert len(rows) == 5
    assert all(row["status"] == "ok" for row in rows)
    assert (pipe.runs_dir / "prepare_manifest.csv").exists()
    assert (pipe.data_dir / "synth" / "_review" / "contact_sheet.png").exists()


def test_fit_predict_reload_roundtrip(dataset, config):
    pipe = Pipeline(config, root=dataset)
    result = pipe.fit(detector_names=["crack", "missing", "stain"])
    assert not result["warnings"]
    bundle = pipe.bundles["synth"]
    assert {d.name for d in bundle.detectors} == {"crack", "missing", "stain"}
    assert bundle.calibration.image_threshold > 0

    cracked = next((dataset / "data" / "synth" / NG / "crack").glob("*.png"))
    first = pipe.predict(cracked)
    assert first.view == "synth"
    assert first.fused_map.shape == bundle.reference.shape
    assert first.full_map.shape[:2] == (512, 640)

    reloaded = Pipeline(config, root=dataset)
    reloaded.load()
    second = reloaded.predict(cracked)
    assert second.image_score == pytest.approx(first.image_score, rel=1e-5)
    assert [d.label for d in second.defects] == [d.label for d in first.defects]


def test_cracked_parts_score_above_clean_ones(dataset, config):
    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack", "missing", "stain"])
    goods = sorted((dataset / "data" / "synth" / GOOD).glob("*.png"))
    bads = sorted((dataset / "data" / "synth" / NG / "crack").glob("*.png"))
    clean_scores = [pipe.predict(p).image_score for p in goods]
    crack_scores = [pipe.predict(p).image_score for p in bads]
    assert min(crack_scores) > max(clean_scores), (crack_scores, clean_scores)


def test_evaluate_produces_metrics(dataset, config):
    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack", "missing", "stain"])

    from adkit.evaluate import evaluate

    metrics = evaluate(pipe, out_dir=dataset / "eval")
    view = metrics["per_view"]["synth"]
    assert view["n_good"] == 8 and view["n_ng"] == 3
    assert (dataset / "eval" / "report.json").exists()
    assert (dataset / "eval" / "predictions.csv").exists()
    assert len(metrics["rows"]) == 11


def test_profile_records_counts(dataset, config):
    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack"])
    profile = json.loads((pipe.runs_dir / "profile.json").read_text(encoding="utf-8"))
    assert profile["n_good"]["synth"] == 8
    assert profile["n_ng"]["synth"] == 3
    assert profile["detectors"] == ["crack"]


def test_fit_warns_on_too_few_goods(dataset, config):
    config.raw["calibration"]["min_good_per_view"] = 50
    pipe = Pipeline(config, root=dataset)
    result = pipe.fit(detector_names=["crack"])
    assert any("brittle" in w for w in result["warnings"])


def test_defect_masks_follow_the_region_not_the_box(dataset, config):
    """A crack is a line: its mask must be far thinner than the rectangle around it."""
    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack"])
    bads = sorted((dataset / "data" / "synth" / NG / "crack").glob("*.png"))
    defects = [d for p in bads for d in pipe.predict(p).defects]
    assert defects, "no defect regions on the cracked images"

    thinnest = min(defects, key=lambda d: d.fill)
    assert thinnest.mask is not None and thinnest.mask.any()
    assert thinnest.mask.shape[:2] == (thinnest.bbox[3], thinnest.bbox[2])
    assert thinnest.mask_area == float(thinnest.mask.sum())
    assert 0.0 < thinnest.fill < 0.95, (thinnest.fill, thinnest.bbox)
    assert set(thinnest.source_masks) <= set(thinnest.sources)
    for name, sub in thinnest.source_masks.items():
        assert sub.shape == thinnest.mask.shape and sub.any(), name


def test_online_update_refuses_a_poisoned_good_set(dataset, config, make):
    """Feeding a defect into the good set must be caught, not silently published."""
    import shutil

    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack"])
    assert pipe._current_version("synth") == "v1"

    cracked = sorted((dataset / "data" / "synth" / NG / "crack").glob("*.png"))
    assert cracked
    shutil.copy2(cracked[0], dataset / "data" / "synth" / GOOD / "poison.png")

    result = pipe.update(detector_names=["crack"])
    assert result["rejected"], result
    assert "absorbed a defect" in result["rejected"][0]["reason"], result["rejected"]
    assert result["accepted"] == []
    assert pipe._current_version("synth") == "v1", "a rejected update must not be published"


def test_versions_are_kept_and_can_be_rolled_back(dataset, config, make):
    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack"])
    first = pipe.predict(sorted((dataset / "data" / "synth" / NG / "crack").glob("*.png"))[0])

    make(dataset / "data" / "synth" / GOOD / "extra.png", angle=90.0, seed=42)
    second = pipe.update(detector_names=["crack"])
    assert second["accepted"], second
    assert pipe._current_version("synth") == "v2"

    assert pipe.rollback("synth", "v1")
    assert pipe._current_version("synth") == "v1"
    back = pipe.predict(sorted((dataset / "data" / "synth" / NG / "crack").glob("*.png"))[0])
    assert back.image_score == pytest.approx(first.image_score, abs=1e-6)


def test_unseen_good_images_are_not_flagged(config, make, tmp_path):
    """The calibration must hold on parts the fit never saw.

    Every earlier acceptance number for this pipeline was measured on the training images,
    where a cut taken from their own distribution cannot be exceeded - and the same build
    flagged 4/4 held-out production parts.
    """
    import shutil

    angles = [0.0, 90.0, 180.0, 270.0, 4.0, -5.0, 93.0, 182.0, 2.0, -3.0, 88.0, 177.0]
    all_goods = [make(tmp_path / "pool" / f"g{i:02d}.png", angle=a, seed=i)
                 for i, a in enumerate(angles)]
    train, held_out = all_goods[:8], all_goods[8:]

    root = tmp_path / "holdout"
    (root / "data/synth" / GOOD).mkdir(parents=True)
    for path in train:
        shutil.copy2(path, root / "data/synth/good" / path.name)

    pipe = Pipeline(config, root=root)
    pipe.fit(detector_names=["crack", "missing", "stain"])
    flagged = [p.name for p in held_out if pipe.predict(p).is_defect]
    assert not flagged, f"{len(flagged)}/{len(held_out)} unseen good parts flagged: {flagged}"


def test_a_channel_silent_on_goods_does_not_vote_ng_on_everything(dataset, config):
    """A reference that collapses to a floor turns silence into an automatic defect vote."""
    import shutil

    goods = sorted((dataset / "data" / "synth" / GOOD).glob("*.png"))
    root = dataset / "silent"
    (root / "data/synth" / GOOD).mkdir(parents=True)
    for path in goods[:4]:
        shutil.copy2(path, root / "data/synth/good" / path.name)

    pipe = Pipeline(config, root=root)
    pipe.fit(detector_names=["missing"])
    bundle = pipe.bundles["synth"]
    ref = bundle.calibration.detectors["missing"].image_ref
    assert ref > 1e-3, "a silent channel needs its own absolute threshold as the line"
    for path in goods[4:]:
        result = pipe.predict(path)
        assert result.per_detector.get("missing", 0.0) < 900.0, "ratio cap means collapse"


def test_the_gate_is_also_measured_on_parts_the_model_never_saw(dataset, config):
    """The in-sample false-alarm count is 0 for every fit, so it cannot be the gate.

    The cut is a percentile of the maxima the training images produced, therefore nothing
    in the training set exceeds it; the same build flagged 7/14 unseen parts on the
    production set. The gate has to carry the held-out counts beside the in-sample ones,
    and they have to cover each good image exactly once - a protocol that scored the goods
    twice would look stricter while measuring nothing new.
    """
    pipe = Pipeline(config, root=dataset)
    result = pipe.fit(detector_names=["crack", "missing", "stain"])
    gate = pipe._anchor_gate("synth")
    hold = gate["holdout"]
    goods = len(list((dataset / "data" / "synth" / GOOD).glob("*.png")))
    assert hold["n_scored"] == goods, "each good image must be scored by exactly one fold"
    assert hold["ng_known"] == 3, "the known defects must be scored out-of-sample too"
    assert hold["false_alarms"] >= gate["false_alarms"], "holdout cannot be more lenient"
    if hold["false_alarms"]:
        assert any("never saw were judged defective" in w for w in result["notes"]), result["notes"]


def test_an_update_that_alarms_more_off_sample_is_rejected():
    """Cross-multiplying the rates keeps the comparison honest across fold counts."""
    was = {"holdout": {"false_alarms": 1, "n_scored": 8, "ng_known": 3, "ng_caught": 3}}
    same = {"holdout": {"false_alarms": 2, "n_scored": 16, "ng_known": 3, "ng_caught": 3}}
    worse = {"holdout": {"false_alarms": 3, "n_scored": 8, "ng_known": 3, "ng_caught": 2}}
    lost = {"holdout": {"false_alarms": 1, "n_scored": 8, "ng_known": 4, "ng_caught": 2}}
    assert Pipeline._worse_holdout(same, was) is None
    assert "flagged defective" in Pipeline._worse_holdout(worse, was)
    assert "recall on the known defects dropped" in Pipeline._worse_holdout(lost, was)
    assert Pipeline._worse_holdout({"holdout": None}, was) is None


def test_one_bad_good_image_must_not_move_the_decision_line(dataset, config, make):
    """The lines are quantiles, because a maximum is a function of batch size.

    Set on the worst good image in the batch, the line can only climb as data arrives, and
    the system gets blunter with the very images meant to sharpen it: measured on a 209-good
    benchmark fit, the pixel line reached 2.66 and the extent line 0.0551 while unseen good
    parts topped out at 1.24, so a real chip scoring 1.68 was judged good and got no box.
    """
    from adkit.postprocess.threshold import calibrate

    goods = sorted((dataset / "data" / "synth" / GOOD).glob("*.png"))
    pipe = Pipeline(config, root=dataset)

    def floors(paths):
        prepared = [pipe.transform(x, view_hint="synth") for x in paths]
        reference = prepared[0].image
        prepared = [pipe.transform(x, view_hint="synth", reference=reference) for x in paths]
        arrs, masks = [q.image for q in prepared], [q.mask for q in prepared]
        detectors = []
        for name in ("crack", "missing", "stain"):
            det = build_detector(name, dict(config.get(f"detectors.{name}", {}) or {}))
            det.fit(arrs, masks)
            detectors.append(det)
        return calibrate(detectors, arrs, masks, config, np.stack(masks).mean(axis=0) >= 0.5)

    base = floors(goods)
    assert base.notes["extent_ref"] > 0.0, "the goods must produce some extent evidence"
    assert base.extent_cut > 0.0, "an uncalibrated extent rule is switched off in inference"

    # one genuinely bad good image joins the batch: it must raise the recorded maximum and
    # leave the line the decision actually uses close to where it was
    loud = make(dataset / "loud.png", crack=True, seed=91)
    after = floors(goods + [loud])
    assert after.notes["region_ref"] > base.notes["region_ref"],         f"the poisoned image should show in the batch maximum: {base.notes['region_ref']} -> "         f"{after.notes['region_ref']}"
    assert after.notes["region_p"] <= base.notes["region_p"] * 1.6,         f"a single image moved the decision line too much: {base.notes['region_p']} -> "         f"{after.notes['region_p']}"
    assert after.notes["reference_percentile"] == 99.0


def test_crossing_good_images_are_attributed_to_a_channel(dataset, config, make):
    """The cost of a quantile line has to name its source, or the operator cannot act on it.

    "5/209 of your own good parts cross the line" says only that something is loose. Whether
    the remedy is the framing, the lighting or a second face depends on which channel produced
    the evidence, and on the real bottle fit the answer was different per channel - so the
    count is recorded next to the channels that owned it, and a note that could not name one
    would be decoration.
    """
    from adkit.postprocess.threshold import calibrate

    goods = sorted((dataset / "data" / "synth" / GOOD).glob("*.png"))
    pipe = Pipeline(config, root=dataset)
    prepared = [pipe.transform(x, view_hint="synth") for x in goods]
    reference = prepared[0].image
    prepared = [pipe.transform(x, view_hint="synth", reference=reference) for x in goods]
    arrs, masks = [q.image for q in prepared], [q.mask for q in prepared]
    detectors = []
    for name in ("crack", "missing", "stain"):
        det = build_detector(name, dict(config.get(f"detectors.{name}", {}) or {}))
        det.fit(arrs, masks)
        detectors.append(det)

    quiet = calibrate(detectors, arrs, masks, config, np.stack(masks).mean(axis=0) >= 0.5)
    owners = quiet.notes["crossing_by_detector"]
    assert set(owners) <= {"crack", "missing", "stain"}, owners
    assert int(quiet.notes["goods_over_pixel_cut"]) and owners,         (
        f"the fixture stopped producing a crossing good image, so this test is measuring "
        f"nothing: {dict(quiet.notes)}")

    loud = [pipe.transform(make(dataset / "loud.png", missing=True, seed=92),
                           view_hint="synth", reference=reference)]
    arrs2 = arrs + [p.image for p in loud]
    masks2 = masks + [p.mask for p in loud]
    detectors2 = []
    for name in ("crack", "missing", "stain"):
        det = build_detector(name, dict(config.get(f"detectors.{name}", {}) or {}))
        det.fit(arrs2, masks2)
        detectors2.append(det)
    noisy = calibrate(detectors2, arrs2, masks2, config, np.stack(masks2).mean(axis=0) >= 0.5)
    assert noisy.notes["crossing_by_detector"], "a part with material removed crossed the line "  \
        "but no channel was named"


def test_a_large_good_set_is_gated_without_refitting(dataset, config):
    """200 goods must not cost 200 minutes, so the big-data estimator is a single pass.

    The fold estimator is the better number, but its price is one full fit per fold - at the
    209-good set this project now holds, that turned `fit` into twenty-five minutes. Past
    `max_fold_images` the deployed model answers the same question with each image excluded
    from its own statistics, which is conservative in the measured direction (self-excluded
    goods peak higher than genuinely unseen parts do).
    """
    pipe = Pipeline(config, root=dataset)
    goods = [Sample(p, "synth", GOOD) for p in
             sorted((dataset / "data" / "synth" / GOOD).glob("*.png"))]
    config.raw["calibration"]["max_fold_images"] = 4
    _, gate = pipe._build_bundle("synth", goods, ["crack", "missing", "stain"], [])
    hold = gate["holdout"]
    assert hold["method"] == "leave-one-out"
    assert hold["folds"] == 0 and hold["n_scored"] == len(goods)

    config.raw["calibration"]["max_fold_images"] = 10 ** 6
    _, gated = pipe._build_bundle("synth", goods, ["crack", "missing", "stain"], [])
    assert gated["holdout"]["method"] == "folds"
    assert gated["holdout"]["folds"] >= 2


def test_the_anchor_is_chosen_on_measured_registration_not_on_the_vote(config, make, tmp_path):
    """A frame that nothing else can register to must not become the reference.

    The cheap proxy (match a central crop against other frames) ranks candidates by edge
    similarity, which does not track what the anchor is actually used for. Here one image is
    scaled differently: it can look like a fine representative while every other frame
    registers badly against it.
    """
    from adkit.data import GOOD, Sample

    root = tmp_path / "proj"
    good = root / "data/synth/good"
    good.mkdir(parents=True)
    angles = [0.0, 2.0, -3.0, 1.5, -1.0, 3.0, 0.5]
    paths = [make(good / f"g{i}.png", angle=a, seed=i) for i, a in enumerate(angles)]
    odd = make(good / "odd.png", angle=1.0, scale=1.25, seed=99)     # the poor anchor
    goods = [Sample(p, "synth", GOOD) for p in paths + [odd]]

    pipe = Pipeline(config, root=root)
    model, anchor = pipe._teach_model(goods, "synth")
    assert goods[anchor].path != odd, "the outlier was picked as the reference"
    assert pipe.anchor_quality is not None and pipe.anchor_quality > 0.9, pipe.anchor_quality


def test_the_pinned_anchor_overrides_the_search(config, make, tmp_path):
    """A fixed reference frame is what makes two fits comparable at all.

    Without it every change to the front end also moves the anchor, and nothing that comes
    out of the fit can be attributed to the change rather than to the new reference.
    """
    from adkit.data import GOOD, Sample

    root = tmp_path / "proj"
    good = root / "data/synth/good"
    good.mkdir(parents=True)
    angles = [0.0, 2.0, -3.0, 1.5, -1.0, 3.0, 0.5]
    paths = [make(good / f"g{i}.png", angle=a, seed=i) for i, a in enumerate(angles)]
    goods = [Sample(p, "synth", GOOD) for p in paths]

    config.raw["preprocess"]["roi"]["anchor_image"] = "g3.png"
    pipe = Pipeline(config, root=root)
    _, anchor = pipe._teach_model(goods, "synth")
    assert goods[anchor].path.name == "g3.png", goods[anchor].path.name
    assert pipe.anchor_quality is not None, "the pinned anchor was never measured"

    config.raw["preprocess"]["roi"]["anchor_image"] = "not_in_this_view.png"
    with pytest.raises(ValueError, match="not among"):
        pipe._teach_model(goods, "synth")


def test_predict_carries_the_source_shape_without_a_second_pass(config, dataset):
    """The map-back target size comes from the geometry, not from re-running the front end."""
    from adkit.imaging import read_frame

    pipe = Pipeline(config, root=dataset)
    pipe.fit(detector_names=["crack", "stain", "missing"])
    goods = sorted((dataset / "data" / "synth" / GOOD).glob("*.png"))
    result = pipe.predict(goods[0], "synth")
    assert result.geometry.source_shape == read_frame(goods[0]).gray.shape
    assert result.full_map.shape[:2] == result.geometry.source_shape
