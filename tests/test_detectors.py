from __future__ import annotations

import numpy as np
import pytest

from adkit.detect import build_detector
from adkit.detect.base import Detector
from adkit.postprocess.threshold import DetectorCalib
from adkit.detect.crack import CrackDetector, dark_line_response
from adkit.detect.features import Backbone
from adkit.detect.missing import MissingDetector
from adkit.detect.patchcore import PatchCore, farthest_point_sample
from adkit.detect.stain import StainDetector
from adkit.detect.stefann import STEFANN
from adkit.imaging import read_frame
from adkit.preprocess import canonicalize, find_roi


@pytest.fixture
def goods(make, tmp_path):
    angles = [0.0, 90.0, 180.0, 270.0, 4.0, -5.0, 93.0, 182.0]
    return [make(tmp_path / f"g{i:02d}.png", angle=angles[i], seed=i) for i in range(8)]


@pytest.fixture
def scene(goods, config, canonicaliser, make, tmp_path):
    """Fitted-on goods plus a way to produce a defect/clean pair in canonical space."""
    prepared = canonicaliser(goods, tmp_path)
    images = [p.image for p in prepared]
    masks = [p.mask for p in prepared]
    reference = images[0]

    def canon(path):
        frame = read_frame(path)
        return canonicalize(frame.gray, find_roi(frame.gray, config), config, reference=reference)

    def pair(**defect):
        flawed = canon(make(tmp_path / f"def_{abs(hash(tuple(defect.items())))}.png", seed=97, **defect))
        clean = canon(goods[1])
        return flawed.image, flawed.mask, clean.image, clean.mask

    return images, masks, pair


def _run(detector: Detector, scene):
    images, masks, pair = scene
    detector.fit(images, masks)
    flawed_image, flawed_mask, clean_image, clean_mask = pair()
    return detector.score_map(flawed_image, flawed_mask), detector.score_map(clean_image, clean_mask)


def test_dark_line_response_is_finite_and_localised(scene, make, tmp_path, config):
    images, _, pair = scene
    flawed, _, _, _ = pair(crack=True)
    resp = dark_line_response(flawed, [1.0, 2.0, 3.0])
    assert resp.shape == flawed.shape
    assert np.isfinite(resp).all()
    assert resp.max() > 0


def test_crack_detector_separates_cracked_from_clean(config, scene):
    detector = CrackDetector(config.get("detectors.crack"))
    images, masks, pair = scene
    detector.fit(images, masks)
    flawed_image, flawed_mask, clean_image, clean_mask = pair(crack=True)
    bad = detector.score_map(flawed_image, flawed_mask)
    good = detector.score_map(clean_image, clean_mask)
    assert bad.image_score > good.image_score * 1.5
    assert bad.mask.any()


def test_missing_detector_catches_cut_corner(config, scene):
    detector = MissingDetector(config.get("detectors.missing"))
    images, masks, pair = scene
    detector.fit(images, masks)
    flawed_image, flawed_mask, clean_image, clean_mask = pair(missing=True)
    bad = detector.score_map(flawed_image, flawed_mask)
    good = detector.score_map(clean_image, clean_mask)
    assert bad.meta["lost_px"] > good.meta["lost_px"]
    assert bad.image_score > good.image_score


def test_stain_detector_catches_blob(config, scene):
    detector = StainDetector(config.get("detectors.stain"))
    images, masks, pair = scene
    detector.fit(images, masks)
    flawed_image, flawed_mask, clean_image, clean_mask = pair(stain=True)
    assert detector.score_map(flawed_image, flawed_mask).image_score > \
        detector.score_map(clean_image, clean_mask).image_score


@pytest.mark.needs_torch
def test_patchcore_shape_and_persistence(config, scene, tmp_path):
    detector = PatchCore(config.get("detectors.patchcore"))
    detector.backbone = Backbone("resnet18", ("layer2", "layer3"), False)
    images, masks, _ = scene
    detector.fit(images, masks)

    smap = detector.score_map(images[0], masks[0])
    assert smap.score.shape == tuple(config.get("detectors.patchcore.grid"))
    assert np.isfinite(smap.score).all()

    store = tmp_path / "store"
    detector.save(store)
    reloaded = PatchCore(config.get("detectors.patchcore"))
    reloaded.backbone = detector.backbone
    reloaded.load(store)
    assert np.allclose(reloaded.score_map(images[0], masks[0]).score, smap.score, atol=1e-5)


@pytest.mark.needs_torch
def test_stefann_scores_are_finite(config, scene):
    detector = STEFANN(config.get("detectors.stefann"))
    detector.backbone = Backbone("resnet18", ("layer2", "layer3"), False)
    images, masks, pair = scene
    detector.fit(images, masks)
    flawed_image, flawed_mask, clean_image, clean_mask = pair(crack=True)
    bad = detector.score_map(flawed_image, flawed_mask)
    good = detector.score_map(clean_image, clean_mask)
    assert np.isfinite(bad.score).all()
    assert bad.score.shape == tuple(config.get("detectors.stefann.grid"))
    assert bad.image_score > 0 and good.image_score > 0


@pytest.mark.needs_torch
def test_stefann_requires_enough_goods(config, tmp_path, make, canonicaliser):
    detector = STEFANN(config.get("detectors.stefann"))
    detector.backbone = Backbone("resnet18", ("layer2", "layer3"), False)
    few = canonicaliser([make(tmp_path / "only.png", seed=7)], tmp_path)
    with pytest.raises(ValueError, match="good images"):
        detector.fit([p.image for p in few], [p.mask for p in few])


def test_registry_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown detector"):
        build_detector("nope")


def test_normalised_falls_back_on_the_detectors_own_cut():
    """Silence on the goods must not erase the detector from the pixel map.

    A defect-specific detector has no good-image distribution to divide by - goods have no
    cracks - and zeroing the map there is what hid every crack behind a stain label.
    """
    from adkit.detect.base import ScoreMap

    score = np.zeros((8, 8), np.float32)
    score[3, 3] = 0.2
    smap = ScoreMap(source="crack", score=score, image_score=0.2)

    assert smap.normalized(0.0, 0.0).max() == 0.0, "no band and no reference stays silent"
    via_ref = smap.normalized(0.0, 0.0, 0.1)
    assert via_ref[3, 3] == pytest.approx(2.0), "1.0 means the detector's own decision point"

    with_band = smap.normalized(0.0, 0.05)
    assert with_band[3, 3] == pytest.approx(4.0), "a real goods band always wins"


def test_crack_appears_as_a_defect_source(config, scene):
    detector = CrackDetector(config.get("detectors.crack"))
    images, masks, pair = scene
    detector.fit(images, masks)
    assert detector.pixel_reference() > 0.0

    flawed_image, flawed_mask, _, _ = pair(crack=True)
    smap = detector.score_map(flawed_image, flawed_mask)
    calib = DetectorCalib(lo=0.0, hi=0.0, image_max=0.0, pixel_ref=float(detector.pixel_reference()))
    assert smap.normalized(calib.lo, calib.hi, calib.pixel_ref or None).max() > 1.0


def test_the_leave_one_out_update_matches_a_refit():
    """The rank-1 downdate STEFANN uses for calibration must be the refit, not a guess."""
    detector = STEFANN({"pca_dim": 3, "shrinkage": 0.3, "grid": [1, 1]})
    rng = np.random.default_rng(11)
    reduced = rng.normal(0.0, 1.0, (9, 1, 3)) + np.array([2.0, -1.0, 0.5])
    detector._n = reduced.shape[0]
    detector._sum1 = reduced.sum(axis=0)
    detector._sum2 = np.einsum("rni,rnj->nij", reduced, reduced)  # per position, over images

    for i in range(reduced.shape[0]):
        keep = np.delete(reduced[:, 0, :], i, axis=0)
        centre, precision = detector._leave_one_out(reduced[i])
        # undo the shrinkage the way `fit` applies it. Its trace-scaled term preserves the
        # trace, so the shrunk matrix's own trace is the scale that was used.
        shrunk = np.linalg.inv(precision[0]) - 1e-6 * np.eye(3)
        tau = np.trace(shrunk) / 3.0
        raw = (shrunk - detector.shrinkage * tau * np.eye(3)) / (1.0 - detector.shrinkage)
        assert np.allclose(centre[0], keep.mean(0), atol=1e-9)
        assert np.allclose(raw, np.cov(keep, rowvar=False), atol=1e-8), (raw, np.cov(keep, rowvar=False))


def test_calibration_uses_the_held_out_score_of_each_image(config):
    """`calibrate` must ask each detector for the score it could not have memorised."""
    from adkit.detect.base import ScoreMap
    from adkit.postprocess.threshold import calibrate

    seen: list[tuple[str, int | None]] = []

    class Recording(Detector):
        name = "recording"

        def fit(self, images, masks):
            self.mark_fitted()

        def score_map(self, image, mask):
            return ScoreMap(source=self.name, score=image.copy(), image_score=float(image.max()))

        def calibration_map(self, image, mask, index=None):
            seen.append((self.name, index))
            return ScoreMap(source=self.name, score=image * 2.0, image_score=float(image.max()) * 2)

    images = [np.full((4, 4), 0.1, np.float32), np.full((4, 4), 0.2, np.float32)]
    masks = [np.ones((4, 4), bool), np.ones((4, 4), bool)]
    calib = calibrate([Recording()], images, masks, config)
    # calibration_map is asked twice per image - once for the pixel distribution, once for the
    # region probe - and both passes must hold the image out, or the cut is back to being
    # measured on scores the detector memorised.
    assert [i for _, i in seen] == [0, 1] * 2, seen
    assert calib.detectors["recording"].image_max == pytest.approx(0.4), \
        "the reference has to come from the held-out scores, not the memorised ones"


@pytest.mark.needs_torch
def test_a_memorising_detector_scores_its_own_images_lower(config, scene):
    """The bias this fixes has to be measurable, or the exclusion is decoration."""
    images, masks, _ = scene
    detector = PatchCore(config.get("detectors.patchcore"))
    detector.backbone = Backbone("resnet18", ("layer2", "layer3"), False)
    detector.fit(images, masks)

    inside = [detector.score_map(im, m).image_score for im, m in zip(images, masks)]
    held = [detector.calibration_map(im, m, i).image_score for i, (im, m) in enumerate(zip(images, masks))]
    assert all(h >= s - 1e-6 for h, s in zip(held, inside)), (inside, held)
    assert np.mean(held) > np.mean(inside) * 1.05, \
        f"excluding the query's own patches barely moved the score: {inside} vs {held}"


def test_the_coreset_stays_fps_when_candidates_exceed_the_cap():
    """Above the cap the pool is random, the *selection* must still be FPS.

    It used to stop at the pool, which silently replaced the coreset with subsampling on any
    view with more than ~20 good images - and left the bank at the full pool size, which is
    what the per-image cost is proportional to.
    """
    rng = np.random.default_rng(4)
    features = rng.normal(size=(5000, 32)).astype(np.float32)
    target = 400
    picked = farthest_point_sample(features, target, seed=1, candidate_cap=1000)
    assert len(picked) == target
    sel = features[picked]
    # k-center radius: how far the worst unselected point is from the chosen set
    def radius(idx):
        d = ((features[idx][:, None, :] - features[idx][None, :, :]) ** 2).sum(-1)
        return float(np.sqrt(d.min(axis=1)).max())
    random_subset = rng.choice(features.shape[0], size=target, replace=False)
    assert radius(picked) <= radius(random_subset), \
        f"FPS coverage is worse than random: {radius(picked)} vs {radius(random_subset)}"


def test_the_memory_bank_is_bounded_by_max_memory():
    """Bank size, not the dot product, is what each prediction pays for."""
    class FakeBackbone:
        def patches(self, image, grid, pool=3, proj_dim=128, seed=0):
            n = grid[0] * grid[1]
            out = np.zeros((n, proj_dim), np.float32)
            for i in range(n):
                out[i, i % proj_dim] = 1.0
            return out

    det = PatchCore({"grid": [16, 16], "coreset_ratio": 0.15, "max_memory": 300,
                     "proj_dim": 32, "k": 3})
    det.backbone = FakeBackbone()
    images = [np.zeros((32, 32), np.float32) for _ in range(40)]   # 40*256 = 10240 patches
    det.fit(images, [np.ones((32, 32), bool) for _ in images])
    assert det.memory.shape[0] <= 300, det.memory.shape
    assert det.score_map(images[0], np.ones((32, 32), bool)).meta["n_memory"] == det.memory.shape[0]
