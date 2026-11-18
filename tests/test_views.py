from __future__ import annotations

from pathlib import Path

import pytest

from adkit.data import GOOD
from adkit.pipeline import Pipeline

ANGLES = [0.0, 90.0, 180.0, 270.0, 3.0, -4.0]


def _populate(root, views: dict[str, int], make):
    for view, count in views.items():
        good_dir = root / "data" / view / GOOD
        good_dir.mkdir(parents=True)
        for i in range(count):
            make(good_dir / f"g{i}.png", angle=ANGLES[i % len(ANGLES)], variant=view, seed=i)
    return root


@pytest.fixture
def mixed_dataset(make, tmp_path):
    """One folder holding two genuinely different faces of the part - the classic mistake."""
    return _populate(tmp_path, {"mixed": 10}, make)


@pytest.fixture
def two_view_dataset(make, tmp_path):
    return _populate(tmp_path, {"side": 6, "face": 6}, make)


def test_poorly_aligned_goods_are_flagged(mixed_dataset, make, config):
    # terminations on the "side" shots, plain body on the "face" shots
    for i in range(4):
        make(mixed_dataset / "data" / "mixed" / GOOD / f"other{i}.png",
             angle=0.0, variant="face", seed=50 + i)
    config.raw["preprocess"]["alignment"]["min_ncc"] = 0.95
    pipe = Pipeline(config, root=mixed_dataset)
    result = pipe.fit(detector_names=["crack"])
    assert any("sit apart from the rest" in w for w in result["warnings"]), result["warnings"]


def test_alignment_scores_are_recorded(mixed_dataset, config):
    pipe = Pipeline(config, root=mixed_dataset)
    pipe.fit(detector_names=["crack"])
    assert len(pipe.alignment["mixed"]) == 10
    assert all(0.0 < v <= 1.0 for v in pipe.alignment["mixed"])


def test_separate_views_align_cleanly(two_view_dataset, config):
    config.raw["preprocess"]["alignment"]["min_ncc"] = 0.88
    pipe = Pipeline(config, root=two_view_dataset)
    result = pipe.fit(detector_names=["crack"])
    assert not [w for w in result["warnings"]
                if "sit apart" in w or "only repeat at ncc" in w], result["warnings"]
    assert set(result["bundles"]) == {"side", "face"}


def test_a_rig_with_poor_repeatability_is_an_imaging_limit_not_a_mixed_view(two_view_dataset,
                                                                            config):
    """Uniformly low alignment means the optics, and must not be reported as a second face."""
    config.raw["preprocess"]["alignment"]["min_ncc"] = 1.0
    pipe = Pipeline(config, root=two_view_dataset)
    result = pipe.fit(detector_names=["crack"])
    assert any("only repeat at ncc" in w for w in result["warnings"]), result["warnings"]
    assert not [w for w in result["warnings"] if "sit apart" in w], result["warnings"]


def test_rotationally_symmetric_part_is_reported_ambiguous(make, tmp_path, config):
    """A bare rectangle has no measurable orientation; the pipeline must say so."""
    import cv2
    import numpy as np

    from adkit.config import Config
    from adkit.imaging import read_frame
    from adkit.pipeline import Pipeline
    from adkit.preprocess import canonicalize, find_roi

    canvas = np.full((512, 640), 250, np.uint8)
    cv2.rectangle(canvas, (245, 206), (395, 306), 60, -1)
    path = tmp_path / "symmetric.png"
    cv2.imwrite(str(path), canvas)

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    cfg.raw["preprocess"]["view"]["assignment"] = "manual"
    cfg.raw["preprocess"]["view"]["default"] = "synth"
    pipe = Pipeline(cfg, root=tmp_path)
    model, _ = pipe._teach_model([type("S", (), {"path": path})()])

    gray = read_frame(path).gray
    roi = find_roi(gray, cfg, model)
    # Position is still decidable - the box is where the edge points are - but the heading
    # is not, and guessing one would silently rotate half the dataset onto its wrong side.
    assert roi.peak > 0.25, "a plain rectangle is still locatable by shape"

    canon = canonicalize(gray, roi, cfg, None)
    flipped = canonicalize(cv2.rotate(gray, cv2.ROTATE_180), roi, cfg, canon.image)
    assert flipped.geometry.meta["orientation_ambiguous"], flipped.geometry.meta


def test_view_selection_picks_the_closest_bundle(two_view_dataset, make, config):
    pipe = Pipeline(config, root=two_view_dataset)
    pipe.fit(detector_names=["crack"])

    assert pipe.predict(make(two_view_dataset / "p_face.png", angle=12.0, variant="face", seed=9)).view == "face"
    assert pipe.predict(make(two_view_dataset / "p_side.png", angle=-9.0, variant="side", seed=8)).view == "side"
