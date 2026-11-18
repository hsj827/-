from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import Config
from .roi import ROI


@dataclass
class Geometry:
    matrix: np.ndarray
    inverse: np.ndarray
    rect: tuple
    canvas: tuple[int, int]
    flipped: bool = False
    turns: int = 0
    shift: tuple[float, float] = (0.0, 0.0)
    # The size of the frame that was canonicalised. `map_back` needs it, and without it here
    # the caller had to run a second whole locate-and-warp pass just to read the shape off a
    # mask - 27-37% of every prediction, measured.
    source_shape: tuple[int, int] | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Canonical:
    image: np.ndarray
    mask: np.ndarray
    geometry: Geometry


def _shoelace(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ordered_corners(rect: tuple, long_side_horizontal: bool) -> np.ndarray:
    """Corners ordered so that edge 0 is the one mapped to the canvas x-axis.

    boxPoints' winding follows OpenCV's angle convention, which flips between the
    "width is long" and "height is long" cases; without normalising it the resulting
    homography mirrors the part, and no 180-degree flip can undo that afterwards.

    For a near-round part the fitted angle is noise, and the quarter-turn search downstream
    then spends itself repairing a 90-degree component of it: measured over 20 identically
    mounted good bottle frames, edge 0 sat at 0.2, 1.5, 2.5, 13.5, 51.3, 73.9 and 88.9 degrees -
    the minimum-area rectangle of a round silhouette can sit at any heading, so up to 45 degrees
    of arbitrary rotation does reach the canvas, and the marking band has no margin to argue
    about it with (49 of 54 sampled frames scored under the ambiguity threshold).

    Removing the angle was tried twice and both times measured worse, because the part's extent
    is defined *by* that angle: `roi.mask` is this quad filled, so an axis-aligned frame cannot
    contain a quad fitted at 45 degrees and cuts its corners off. Over 105 good frames the part
    then occupied 0.756 of the canvas where the rotated quad holds it at 0.815-0.821, i.e. an
    outline 2.5 px off the median template, and the pixel line that keeps the goods under had
    to double (2.06 to 4.34) while contamination recall fell from 17/21 to 14/21 and an unseen
    good started alarming. Taking the geometric mean of the sides - which keeps the quad's area
    - crops it identically, so it is the axis-aligned box itself that is wrong, not its scale.

    An angle-free framing is therefore only available together with a pose-free extent (the
    radial statistics of the edge points, say, instead of the minimum-area rectangle). Nothing
    in the current products needs it: registration is already at the imaging limit and the
    outline is near rotation-invariant, which is why the crop costs area and the rotation costs
    almost nothing.
    """
    pts = cv2.boxPoints(rect).astype(np.float32)
    if _shoelace(pts) < 0:
        pts = pts[::-1]
    if long_side_horizontal:
        lengths = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(2)]
        pts = np.roll(pts, -int(np.argmax(lengths)), axis=0)
    return pts


def _expand(pts: np.ndarray, margin_ratio: float) -> np.ndarray:
    centre = pts.mean(axis=0, keepdims=True)
    return centre + (pts - centre) * (1.0 + margin_ratio)


def _warp(image: np.ndarray, matrix: np.ndarray, canvas: tuple[int, int], fill: float = 1.0) -> np.ndarray:
    return cv2.warpPerspective(
        image.astype(np.float32), matrix, canvas,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
    )


def canonicalize(gray: np.ndarray, roi: ROI, cfg: Config, reference: np.ndarray | None = None,
                 turns: int | None = None) -> Canonical:
    """Map the part into a fixed canvas.

    `turns` forces a quarter-turn. Leave it None and the heading is decided here, against
    the reference: the canvas is the only frame where "rotate the part" and "rotate the
    picture" mean the same thing, and the decision is made on the marking band because
    everything else about the part looks the same whichever way it faces.
    """
    ccfg = cfg.get("preprocess.canonical")
    rcfg = cfg.get("preprocess.roi")
    cw, ch = (int(v) for v in ccfg.get("size", [256, 256]))
    canvas = (cw, ch)

    rect_w, rect_h = roi.rect[1]
    rect_aspect = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1e-6)
    elongated = rect_aspect >= float(ccfg.get("elongated_aspect", 1.15))

    # `roi.mask` is this same quad, filled - which is what makes the fitted angle load-bearing
    # rather than incidental, and what makes the obvious "drop the angle for round parts"
    # cleanup a regression. See the note on `_ordered_corners`.
    src = _expand(_ordered_corners(roi.rect, elongated and bool(ccfg.get("long_side_horizontal", True))),
                  float(rcfg.get("margin_ratio", 0.1)))
    dst = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)

    image = _warp(gray, matrix, canvas)
    mask = _warp(roi.mask.astype(np.float32), matrix, canvas, fill=0.0) > 0.5
    flipped = False
    turn = 0
    shift = (0.0, 0.0)

    if turns is not None:
        step = (int(turns) // 90) % 4
        if step:
            rot = _rot_matrix(canvas, step)
            matrix = rot @ matrix
            image = np.ascontiguousarray(np.rot90(image, step))
            mask = np.ascontiguousarray(np.rot90(mask, step))
        turn = step * 90
        flipped = step % 2 == 1
        margin, ambiguous = 1.0, False
        if reference is not None:
            image, mask, matrix, shift = _register(image, mask, matrix, reference, cfg, canvas)
    elif reference is not None:
        (turn, image, mask, matrix, shift, margin,
         ambiguous) = _best_orientation(image, mask, matrix, reference, cfg, canvas, elongated)
        flipped = turn % 2 == 1
    else:
        margin, ambiguous = 0.0, False

    return Canonical(
        image=image,
        mask=mask,
        geometry=Geometry(matrix=matrix, inverse=np.linalg.inv(matrix), rect=roi.rect,
                          canvas=canvas, flipped=flipped, turns=turn, shift=shift,
                          source_shape=(int(gray.shape[0]), int(gray.shape[1])),
                          meta={"orientation_margin": margin, "orientation_ambiguous": ambiguous}),
    )


def _rot_matrix(canvas: tuple[int, int], turns: int) -> np.ndarray:
    cw, ch = canvas
    m = cv2.getRotationMatrix2D(((cw - 1) / 2.0, (ch - 1) / 2.0), 90.0 * turns, 1.0)
    return np.vstack([m, np.float32([0.0, 0.0, 1.0])]).astype(np.float32)


def _band_weight(reference: np.ndarray, turns: list[int], blur: float,
                 mask: np.ndarray | None, inset: float) -> np.ndarray:
    """Where the reference actually differs between the competing quarter-turns.

    Most of a component looks the same at every quarter-turn: the lid outline is 180-degree
    symmetric and the body is featureless. A whole-canvas similarity therefore scores the
    background vignette (which does not move when the part does) and ends up voting "no
    rotation" whichever way the component faces - which is the wrong answer half the time.
    Conversely, summing the asymmetry over turns that are not on offer puts the weight on
    the part boundary, which differs enormously between 0 and 90 degrees and identically
    between the 0 and 180 being decided here.

    So the weight is built from exactly the candidates being compared, and then restricted
    to the part's own interior: the outline and whatever background the crop caught are the
    two things that cannot tell which way the part faces.
    """
    weight = np.zeros(reference.shape, np.float32)
    for turn in turns[1:]:
        weight += np.abs(reference - np.rot90(reference, turn)).astype(np.float32)
    if mask is not None and inset > 0:
        k = max(1, int(inset * min(reference.shape)) * 2 + 1)
        interior = cv2.erode(mask.astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        weight *= interior
    k = max(1, int(blur) * 2 + 1)
    weight = cv2.GaussianBlur(weight, (k, k), blur)
    total = float(weight.sum())
    return weight / total if total > 1e-9 else weight


def _best_orientation(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray,
                      reference: np.ndarray, cfg: Config, canvas: tuple[int, int],
                      elongated: bool) -> tuple:
    """Choose the canonical quarter-turn on the marking band, registering each candidate first.

    An elongated part tells us its own long axis, so only the 180-degree ambiguity is left.
    A near-round part cannot: its bounding rectangle has no meaningful heading, so all four
    turns compete - and the search is repairing more than which face is up, because the fitted
    angle it works on is itself noise. See the note on `_ordered_corners` for why that is the
    accepted state rather than something to remove here.
    """
    ccfg = cfg.get("preprocess.canonical")
    if reference.shape != image.shape:
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
    resolve = str(ccfg.get("flip_resolve", "template")) == "template"
    square = canvas[0] == canvas[1]
    if not resolve:
        turns = [0]
    elif elongated or not square:
        turns = [0, 2]
    else:
        turns = [0, 1, 2, 3]

    weight = _band_weight(reference, turns, float(ccfg.get("band_blur", 2.0)),
                          mask, float(ccfg.get("band_inset", 0.10)))
    blur = float(ccfg.get("band_blur", 2.0))
    k = max(1, int(blur) * 2 + 1)
    ranked: list[list] = []
    for turn in turns:
        candidate = np.rot90(image, turn).copy() if turn else image
        candidate_mask = np.rot90(mask, turn).copy() if turn else mask
        candidate_matrix = _rot_matrix(canvas, turn) @ matrix if turn else matrix
        registered, registered_mask, registered_matrix, shift = _register(
            candidate, candidate_mask, candidate_matrix, reference, cfg, canvas)
        # Smoothing the difference before weighting it is what makes the vote tolerate the
        # few pixels of residual misalignment a placement tilt always leaves behind; on the
        # raw pixels a mark one pixel off scores as a total mismatch.
        diff = cv2.GaussianBlur(np.abs(registered - reference).astype(np.float32), (k, k), blur)
        # Higher is better: the negative weighted mean absolute difference over the band.
        score = -float((weight * diff).sum())
        ranked.append([score, turn, registered, registered_mask, registered_matrix, shift])

    ranked.sort(key=lambda item: -item[0])
    best = ranked[0]
    margin = float(best[0] - ranked[1][0]) if len(ranked) > 1 else 1.0
    threshold = float(ccfg.get("ambiguous_margin", 0.02))
    # The margin says the marking band cannot decide the orientation, and it would be a
    # reasonable conclusion that the winner is therefore noise worth overriding. It is not:
    # forcing the base turn made things measurably worse - trained goods' registration against
    # the reference fell from ncc 0.851 to 0.793 and a benchmark class went from 6/8 caught to
    # 4/8. The turn is not only choosing between faces of the part; on a near-round part it is
    # also the only thing standing between the canvas and the heading the bounding rectangle
    # happened to be fitted at, so a large turn can be the correct answer even when the band
    # gives no margin. Ambiguity is reported, and acted on by nothing.
    return best[1], best[2], best[3], best[4], best[5], margin, margin < threshold


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom > 1e-9 else 0.0


def _register(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray, reference: np.ndarray,
              cfg: Config, canvas: tuple[int, int]):
    """Apply a phase-correlation shift only when it actually improves the match.

    On a vignetted field the correlation peak is not the part: measured over the production
    set, the shift it reports lowers the correlation against the reference for most images
    (0.65 to 0.35, 0.50 to -0.16), because the bright ring and the specular blobs are the
    strongest features and they are fixed to the lens, not to the component. Restricting to
    the part's own footprint removes most of that pull; the improvement test removes the
    rest, and costs nothing when the estimate is right.
    """
    if cfg.get("preprocess.register.method") != "phase_correlation":
        return image, mask, matrix, (0.0, 0.0)
    part = mask.astype(np.float32)
    shift = _phase_shift(image * part, reference * part,
                         float(cfg.get("preprocess.register.max_shift", 16)))
    if abs(shift[0]) <= 0.5 and abs(shift[1]) <= 0.5:
        return image, mask, matrix, shift
    t = np.float32([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    warped = _warp(image, t, canvas)
    if _ncc(warped, reference) <= _ncc(image, reference):
        return image, mask, matrix, (0.0, 0.0)
    return (warped,
            _warp(mask.astype(np.float32), t, canvas, fill=0.0) > 0.5,
            t @ matrix, shift)


def _phase_shift(image: np.ndarray, reference: np.ndarray, max_shift: float) -> tuple[float, float]:
    if image.shape != reference.shape:
        reference = cv2.resize(reference, (image.shape[1], image.shape[0]))
    win = np.hanning(image.shape[0]).astype(np.float32)[:, None] * np.hanning(image.shape[1]).astype(np.float32)[None, :]
    (dx, dy), _ = cv2.phaseCorrelate(image * win, reference * win)
    if abs(dx) > max_shift or abs(dy) > max_shift:
        return 0.0, 0.0
    return float(dx), float(dy)


def map_back(score_map: np.ndarray, geometry: Geometry, out_shape: tuple[int, int]) -> np.ndarray:
    """Project a canonical-space map into original image coordinates."""
    canvas = geometry.canvas
    scaled = cv2.resize(score_map.astype(np.float32), canvas, interpolation=cv2.INTER_LINEAR)
    return cv2.warpPerspective(
        scaled, geometry.inverse, (out_shape[1], out_shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
