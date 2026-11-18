from __future__ import annotations

import numpy as np
import pytest

from adkit.imaging import read_frame
from adkit.preprocess import canonicalize, classify_view, find_roi


def test_roi_finds_the_part(good_set, config):
    frame = read_frame(good_set[0])
    roi = find_roi(frame.gray, config)
    assert 0.005 < roi.area_ratio < 0.5
    assert roi.mask.any()
    assert roi.rect[1][0] > 0 and roi.rect[1][1] > 0


def test_canonicalisation_cancels_rotation(good_set, config):
    a = read_frame(good_set[0]).gray
    b = read_frame(good_set[1]).gray
    ca = canonicalize(a, find_roi(a, config), config)
    cb = canonicalize(b, find_roi(b, config), config, reference=ca.image)
    assert ca.image.shape == cb.image.shape == (128, 128)
    ref = ca.image
    alt = np.rot90(cb.image, 2)
    best = max(_ncc(cb.image, ref), _ncc(alt, ref))
    assert best > 0.8, f"aligned images still differ (ncc={best:.3f})"


def test_view_assignment_manual_then_auto(good_set, config):
    frame = read_frame(good_set[0])
    roi = find_roi(frame.gray, config)
    assert classify_view(roi, config, folder_hint="side") == "side"

    config.raw["preprocess"]["view"]["assignment"] = "auto"
    config.raw["preprocess"]["view"]["auto"]["rules"] = [{"id": "wide", "aspect_gt": 1.05}]
    assert classify_view(roi, config) in {"wide", "unknown"}


def test_rejects_annotated_screenshot(tmp_path, config):
    import cv2
    from adkit.pipeline import Pipeline

    image = np.full((300, 300, 3), 250, np.uint8)
    cv2.rectangle(image, (40, 40), (200, 200), (0, 0, 255), 4)
    cv2.rectangle(image, (90, 90), (160, 160), (60, 60, 60), -1)
    path = tmp_path / "marked.png"
    cv2.imwrite(str(path), image)

    pipe = Pipeline(config, root=tmp_path)
    with pytest.raises(ValueError, match="annotated"):
        pipe.transform(path)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a - a.mean(), b - b.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def test_an_ambiguous_band_still_picks_the_turn_that_registers(config, make, tmp_path):
    """Do not force the base orientation just because the marking band is undecided.

    That was tried, on the argument that a zero-margin winner is noise: it is wrong, because
    the quarter-turn search is also repairing the corner ordering of a near-square bounding
    rectangle. Measured on a benchmark of identically-mounted bottles, pinning the turn to the
    base orientation dropped the goods' registration against the reference from ncc 0.851 to
    0.793 and one defect class from 6/8 caught to 4/8. The ambiguity flag is reported; it does
    not override.
    """
    import numpy as np
    from adkit.preprocess import canonicalize, find_roi
    from adkit.imaging import read_frame
    from adkit.pipeline import _ncc

    path = make(tmp_path / "sym.png", variant="face", seed=3)
    frame = read_frame(path)
    roi = find_roi(frame.gray, config, None)
    base = canonicalize(frame.gray, roi, config)

    # A reference with no asymmetric marking: the same part averaged with its own half-turn.
    symmetric = ((base.image + np.rot90(base.image, 2)) / 2).astype(np.float32)
    chosen = canonicalize(frame.gray, roi, config, reference=np.rot90(symmetric, 2))
    assert chosen.geometry.meta["orientation_ambiguous"] is True

    forced = np.rot90(base.image, 0)
    assert _ncc(chosen.image, np.rot90(symmetric, 2)) >= _ncc(forced, np.rot90(symmetric, 2)),         "the turn the search picked registers worse than the base orientation"


def test_ambiguous_orientation_is_reported_at_fit_time(config, tmp_path):
    """The operator has to be told the product needs a fiducial, or they blame the camera.

    A part that looks the same at 0 and 180 degrees gives the marking band nothing to vote
    on, and the symptom downstream is not a warning but a heatmap that lights up along the
    whole outline of a perfect part. So the fit says it out loud.
    """
    import cv2
    import numpy as np

    from adkit.pipeline import Pipeline

    def symmetric_part(path, dx, dy):
        img = np.full((300, 400), 40, np.uint8)
        cv2.rectangle(img, (140 + dx, 80 + dy), (260 + dx, 220 + dy), 200, -1)
        # centred hole: invariant under the half-turn the band would have to decide
        cv2.circle(img, (200 + dx, 150 + dy), 24, 40, -1)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return path

    root = tmp_path / "proj"
    good = root / "data/synth/good"
    for i, (dx, dy) in enumerate([(0, 0), (2, -1), (-3, 2), (1, 3), (-2, -2), (3, 1)]):
        symmetric_part(good / f"s{i:02d}.png", dx, dy)
    pipe = Pipeline(config, root=root)
    result = pipe.fit(detector_names=["crack", "stain"])
    assert any("orientation" in w for w in result["warnings"]), result["warnings"]


def test_a_decidable_marking_does_not_warn_about_orientation(config, make, tmp_path):
    """The warning must not become noise: parts with a real mark stay silent."""
    import shutil

    from adkit.pipeline import Pipeline

    root = tmp_path / "proj"
    (root / "data/synth/good").mkdir(parents=True)
    angles = [0.0, 90.0, 180.0, 270.0, 4.0, -5.0, 93.0, 182.0]
    for i, a in enumerate(angles):
        src = make(tmp_path / f"g{i}.png", angle=a, variant="face", seed=i)
        shutil.copy2(src, root / "data/synth/good" / src.name)
    pipe = Pipeline(config, root=root)
    result = pipe.fit(detector_names=["crack", "stain"])
    assert not any("orientation" in w for w in result["warnings"]), result["warnings"]


def _round_part_at(angle):
    """A near-round part with one off-centre mark, and its bounding rectangle fitted at
    `angle` - which for a round outline is a heading noise chose, not a heading the part has."""
    import cv2
    import numpy as np

    from adkit.preprocess.roi import _roi_from_rect

    gray = np.full((300, 400), 20.0, np.float32)
    cv2.circle(gray, (200, 150), 88, 0.75, -1)
    # Big and low-contrast enough to be found by centroid: the disc outline is
    # rotation-invariant, so anything smaller is invisible to a correlation on this part, which
    # is how the first version of this test managed to assert nothing while still passing.
    cv2.circle(gray, (245, 150), 40, 0.25, -1)
    rect = ((200.0, 150.0), (170.0, 176.0), angle)
    return gray, _roi_from_rect(gray, rect, "test", 0.0)


def test_the_canvas_frame_still_holds_the_part_it_was_fitted_at(config):
    """The framing quad has to contain the part at any heading the rectangle was fitted at.

    This is why the fitted angle is load-bearing rather than incidental, and why removing it
    for round parts - which looks like an obvious cleanup, since the angle is noise - was
    measured twice and rejected both times. `roi.mask` is that quad filled, so an axis-aligned
    frame cannot hold a quad fitted at 45 degrees: over 105 good bottle frames the part then
    occupied 0.756 of the canvas against the rotated quad's 0.815-0.821, which puts the outline
    2.5 px off the median template, doubles the pixel line that keeps the goods under (2.06 to
    4.34) and costs 4 of 21 contamination images. Keeping the area by taking the geometric mean
    of the sides crops it just the same, so it is the axis-aligned box that fails, not its size.

    An angle-free framing is only available together with a pose-free extent (the radial
    statistics of the edge points, say, instead of the minimum-area rectangle). Nothing in the
    current products needs it: registration is already at the imaging limit and the outline is
    near rotation-invariant, which is why the crop costs area and the rotation costs almost
    nothing - the last assertion here measures the rotation directly, because a correlation on a
    round part is too weak to show it.
    """
    fractions, headings, radii = [], [], []
    for angle in (0.0, 25.0, 45.0, 70.0):
        gray, roi = _round_part_at(angle)
        canon = canonicalize(gray, roi, config)
        fractions.append(float(canon.mask.mean()))
        # where the mark sits in the canvas, relative to the part's own centre
        dark = (canon.image < 0.45) & canon.mask
        ys, xs = np.nonzero(dark)
        cx, cy = float(canon.mask.shape[1] / 2), float(canon.mask.shape[0] / 2)
        dx, dy = xs.mean() - cx, ys.mean() - cy
        radii.append(float(np.hypot(dx, dy)))
        headings.append(float(np.degrees(np.arctan2(dy, dx))))

    # The frame keeps holding the part, which is the thing an angle-free framing loses.
    spread = max(fractions) - min(fractions)
    assert spread < 0.005, f"the frame stopped containing the part equally: {fractions}"
    assert max(radii) - min(radii) < 2.0, f"the part changed size in the canvas: {radii}"
    # ... and the angle does reach the content, which is what it costs. Asserted rather than
    # assumed, because a correlation metric on a round part is too weak to show it.
    assert max(headings) - min(headings) > 15.0, (
        f"the fitted heading no longer moves the content, so the trade-off this test records "
        f"has changed: {headings}")

