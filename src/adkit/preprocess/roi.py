from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Config
from .shapemodel import ShapeModel, gradient_map


@dataclass
class ROI:
    rect: tuple
    bbox: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    area_ratio: float
    bbox_aspect: float
    fill: float
    mean: float
    edge_density: float
    method: str = "otsu"
    peak: float = 0.0
    margin: float = 0.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.bbox


def background_is_light(gray: np.ndarray, mode: str = "auto") -> bool:
    if mode == "light":
        return True
    if mode == "dark":
        return False
    border = np.concatenate(
        [gray[:4].ravel(), gray[-4:].ravel(), gray[:, :4].ravel(), gray[:, -4:].ravel()]
    )
    return float(np.median(border)) > 0.5


def part_crop(gray: np.ndarray, cfg: Config) -> np.ndarray | None:
    """Crop covering the component's own extent, measured from where its edges are.

    A fixed fraction of the frame silently breaks the moment a different part fills a
    different share of it - the template then locks onto the marking in the middle and the
    canonical crop becomes a zoom of the text. Edge energy is present on the component and
    almost absent on the smooth fixture, so it gives the size directly.
    """
    rcfg = cfg.get("preprocess.roi")
    h, w = gray.shape
    reach = float(rcfg.get("search_reach", 0.8))
    x0, y0 = int(w * (1 - reach) / 2), int(h * (1 - reach) / 2)
    window = gray[y0:h - y0, x0:w - x0]

    img8 = np.clip(window * 255.0, 0, 255).astype(np.uint8)
    mag = cv2.magnitude(cv2.Sobel(img8, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(img8, cv2.CV_32F, 0, 1, ksize=3))
    threshold = float(np.percentile(mag, float(rcfg.get("edge_percentile", 99.0))))
    strong = (mag > threshold).astype(np.uint8)
    k = int(rcfg.get("edge_dilate", 15))
    strong = cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    ys, xs = np.nonzero(strong)
    if xs.size < 16:
        return None
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    centre, (side_w, side_h), _ = cv2.minAreaRect(points)
    if min(side_w, side_h) < 8:
        return None

    # minAreaRect's sides are pose-invariant, unlike a bounding box that inflates as the
    # part turns. Keeping the component's own aspect ratio matters: a square crop around a
    # rectangular part encodes mostly background, and that background moves when the part
    # rotates, so matching degrades exactly for the rotated images.
    margin = 1.0 + float(rcfg.get("crop_margin", 0.12))
    crop_w = int(max(side_w, side_h) * margin)
    crop_h = int(min(side_w, side_h) * margin)
    cx, cy = int(x0 + centre[0]), int(y0 + centre[1])
    gx0, gy0 = max(0, cx - crop_w // 2), max(0, cy - crop_h // 2)
    gx1, gy1 = min(w, cx + crop_w // 2), min(h, cy + crop_h // 2)
    crop = gray[gy0:gy1, gx0:gx1]
    if min(crop.shape) < 32 or crop.shape[0] * crop.shape[1] > 0.9 * h * w:
        return None
    return crop


def central_crop(gray: np.ndarray, fraction: float = 0.26) -> np.ndarray:
    """The part is fixtured near the frame centre, so the first good image's centre is a template."""
    h, w = gray.shape
    tw, th = int(min(h, w) * fraction * (w / h) if w >= h else w * fraction), int(min(h, w) * fraction)
    tw, th = max(16, tw), max(16, th)
    cx, cy = w // 2, h // 2
    return gray[cy - th // 2:cy + th // 2, cx - tw // 2:cx + tw // 2]


def localizer_crop(gray: np.ndarray, cfg: Config) -> np.ndarray:
    return part_crop(gray, cfg) if str(cfg.get("preprocess.roi.template_source", "edges")) == "edges" \
        else central_crop(gray, float(cfg.get("preprocess.roi.template_crop", 0.26)))


def _roi_from_contour(gray: np.ndarray, contour: np.ndarray, method: str, peak: float) -> ROI:
    h, w = gray.shape
    area = float(cv2.contourArea(contour))
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    rect = cv2.minAreaRect(contour)
    bx, by, bw, bh = cv2.boundingRect(contour)

    img8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    mag = cv2.magnitude(cv2.Sobel(img8, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(img8, cv2.CV_32F, 0, 1, ksize=3))
    inside = mask > 0
    edge_density = float((mag[inside] > 40).mean()) if inside.any() else 0.0

    return ROI(
        rect=rect,
        bbox=(int(bx), int(by), int(bw), int(bh)),
        contour=contour,
        mask=inside,
        area_ratio=area / float(h * w),
        bbox_aspect=float(bw) / float(max(bh, 1)),
        fill=float(area / max(1, bw * bh)),
        mean=float(gray[inside].mean()) if inside.any() else float(gray.mean()),
        edge_density=edge_density,
        method=method,
        peak=peak,
    )


def _outline_in_window(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                       expect_size: tuple[float, float], cfg: Config) -> np.ndarray | None:
    """The part's own oriented outline, found inside a window already known to hold it.

    Thresholding cannot separate these parts from their background over the whole frame -
    the can and the vignetted field sit within a few grey levels of each other - but inside
    a window that is mostly part it is bimodal again. The oriented rect matters: a box that
    only bounds the part axis-aligned leaves its placement tilt in the canvas, and the
    detectors and the marking-band comparison then have to work through that skew.

    `expect_size` is the located box's own width and height, not the window's - the window
    is deliberately looser, and comparing the contour to it would reject the part for being
    the size it actually is.
    """
    rcfg = cfg.get("preprocess.roi")
    window = gray[y0:y1, x0:x1]
    light = background_is_light(window, cfg.get("preprocess.background", "auto"))
    img8 = np.clip(window * 255.0, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img8, (3, 3), 0)
    flags = cv2.THRESH_BINARY_INV if light else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, 0, 255, flags + cv2.THRESH_OTSU)
    k = int(rcfg.get("close_kernel", 5))
    if k > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if not 0.15 <= area / float(window.shape[0] * window.shape[1]) <= 0.95:
        return None
    # The refinement exists to recover the part's pose, not to re-find it at another scale.
    # Area alone is not enough: a crescent of the fixed vignette has a plausible area but a
    # far larger extent, and accepting it replaces the part's box with the lens's.
    expect = float(max(expect_size))
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    if not 0.7 <= max(rw, rh) / max(expect, 1.0) <= 1.3:
        return None
    if not 0.5 <= min(rw, rh) / max(float(min(expect_size)), 1.0) <= 1.5:
        return None
    return contour + np.array([[[x0, y0]]], np.int32)


def part_rect_from_edges(gray: np.ndarray, box: tuple[int, int, int, int],
                         cfg: Config) -> tuple | None:
    """The part's own oriented rect, fitted to the strong edge points inside a box.

    Grey thresholding cannot segment these parts even inside a tight window - the lid seam
    is the only thing bright enough to threshold cleanly, so Otsu returns the whole
    vignetted field. Edge points behave better: measured over the production set, 74-86% of
    a frame's strongest gradient points belong to the can, and the fixture's low-frequency
    ring accounts for the rest. Three rounds of "fit, keep the points inside the fit, refit"
    drop that ring, leaving the can's own outline.

    This is what recovers the placement tilt. The located box is axis-aligned, so without
    it a 2-degree part enters the canvas still tilted by 2 degrees and every pixel
    comparison has to work through that skew.
    """
    scfg = cfg.get("preprocess.shape") or {}
    x, y, w, h = box
    map_ = gradient_map(np.clip(gray * 255.0, 0, 255).astype(np.uint8).astype(np.float32))
    sub = map_[max(0, y):min(map_.shape[0], y + h), max(0, x):min(map_.shape[1], x + w)]
    if sub.size == 0:
        return None
    # The percentile is taken inside the window, not over the frame: a frame that is mostly
    # background has its 99.5th percentile down in the noise, and the fit then runs on
    # random speckle instead of the part.
    threshold = float(np.percentile(sub, float(scfg.get("fit_percentile", 99.0))))
    ys, xs = np.nonzero(sub > threshold)
    if xs.size < int(scfg.get("min_points", 24)):
        return None
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    all_points = points

    def inside_rect(pts, rect, scale):
        hull = cv2.boxPoints((rect[0], (rect[1][0] * scale, rect[1][1] * scale),
                              rect[2])).astype(np.int32)
        cover = np.zeros(sub.shape, np.uint8)
        cv2.fillPoly(cover, [hull], 1)
        return cover[np.clip(pts[:, 1].astype(int), 0, sub.shape[0] - 1),
                     np.clip(pts[:, 0].astype(int), 0, sub.shape[1] - 1)] > 0

    rect = None
    for _ in range(3):
        rect = cv2.minAreaRect(points)
        keep = inside_rect(points, rect, 1.05)
        if keep.sum() < int(scfg.get("min_points", 24)):
            break
        points = points[keep]
    if rect is None:
        return None

    # The rejection loop only ever removes points, so on a part whose edges are a hollow
    # ring it shaves the corners a little more each round and returns a rect smaller than
    # the part. Re-collecting from the original set inside the converged rect puts the
    # shaved boundary back while still excluding what the rejection was for.
    rect = cv2.minAreaRect(all_points[inside_rect(all_points, rect, 1.10)])
    (cx, cy), (rw, rh), angle = rect
    # Plausibility against the box that was located: the fit is meant to recover the pose,
    # not to re-find the part. A cluster far from the located size is the vignette ring.
    expect = float(max(w, h))
    if not 0.7 <= max(rw, rh) / max(expect, 1.0) <= 1.3:
        return None
    if not 0.5 <= min(rw, rh) / max(float(min(w, h)), 1.0) <= 1.5:
        return None
    reach = 0.15 * (w * w + h * h) ** 0.5
    if abs(cx + x - (x + w / 2)) > reach or abs(cy + y - (y + h / 2)) > reach:
        return None
    return (float(cx + x), float(cy + y)), (rw, rh), float(angle)


def _roi_from_rect(gray: np.ndarray, rect: tuple, method: str, peak: float) -> ROI:
    """ROI from an oriented rect, so the canvas can undo the part's own rotation."""
    contour = cv2.boxPoints(rect).astype(np.int32)
    return _roi_from_contour(gray, contour, method, peak)


def _roi_from_template(gray: np.ndarray, template: np.ndarray, min_peak: float,
                       cfg: Config) -> ROI:
    """Locate by template, then recover the true outline inside that window.

    The matched box is axis-aligned, so using it directly would throw away the part's
    rotation; the threshold contour inside the window restores it.
    """
    rcfg = cfg.get("preprocess.roi")
    scales = [float(s) for s in rcfg.get("match_scales", [0.85, 1.0, 1.15])]
    best = None
    for scale in scales:
        if scale == 1.0:
            resized = template
        else:
            resized = cv2.resize(template, (max(16, int(template.shape[1] * scale)),
                                            max(16, int(template.shape[0] * scale))),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if resized.shape[0] >= gray.shape[0] or resized.shape[1] >= gray.shape[1]:
            continue
        _, peak, _, loc = cv2.minMaxLoc(
            cv2.matchTemplate(gray.astype(np.float32), resized.astype(np.float32),
                              cv2.TM_CCOEFF_NORMED))
        if best is None or peak > best[0]:
            best = (float(peak), loc, resized.shape)
    if best is None:
        raise ValueError("template larger than the image")
    peak, loc, (th, tw) = best
    if peak < min_peak:
        raise ValueError(f"template match too weak: peak={peak:.3f}")

    slack = float(rcfg.get("window_slack", 0.5))
    sx, sy = int(tw * slack), int(th * slack)
    x0 = max(0, int(loc[0]) - sx)
    y0 = max(0, int(loc[1]) - sy)
    x1 = min(gray.shape[1], int(loc[0]) + tw + sx)
    y1 = min(gray.shape[0], int(loc[1]) + th + sy)

    contour = _outline_in_window(gray, x0, y0, x1, y1, (tw, th), cfg)
    if contour is not None:
        return _roi_from_contour(gray, contour, "template+contour", peak)

    box = np.array([[[loc[0], loc[1]]], [[loc[0] + tw, loc[1]]],
                    [[loc[0] + tw, loc[1] + th]], [[loc[0], loc[1] + th]]], np.int32)
    return _roi_from_contour(gray, box, "template+box", float(peak))


def _roi_from_threshold(gray: np.ndarray, cfg: Config) -> ROI:
    rcfg = cfg.get("preprocess.roi")
    light = background_is_light(gray, cfg.get("preprocess.background", "auto"))
    img8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img8, (3, 3), 0)
    flags = cv2.THRESH_BINARY_INV if light else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blur, 0, 255, flags + cv2.THRESH_OTSU)

    k = int(rcfg.get("close_kernel", 5))
    if k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, k // 2),) * 2))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no contour found - check preprocess.background / roi settings")
    return _roi_from_contour(gray, max(contours, key=cv2.contourArea), "otsu", 0.0)


def find_roi(gray: np.ndarray, cfg: Config, model: "ShapeModel | np.ndarray | None" = None) -> ROI:
    """Locate the part.

    Grey-level segmentation cannot separate these parts from their background at all - the
    can and the vignetted field sit within a few grey levels of each other - so a gradient
    shape model is preferred and thresholding is only the fallback. The model answers where
    the part is; which way it faces is a separate question with separate evidence, and is
    settled in canonical space against the reference image.
    """
    rcfg = cfg.get("preprocess.roi")
    method = str(rcfg.get("method", "auto"))

    if isinstance(model, ShapeModel) and method in ("auto", "shape"):
        try:
            pose = model.locate(gray)
        except ValueError:
            pose = None
        if pose is not None and pose.peak >= float(cfg.get("preprocess.shape.min_peak", 0.15)):
            cx, cy = pose.centre
            pw, ph = pose.size
            grow = 1.0 + float(rcfg.get("fit_margin", 0.10))
            fw, fh = int(pw * grow), int(ph * grow)
            fx = max(0, int(cx - fw / 2))
            fy = max(0, int(cy - fh / 2))
            # The located box is axis-aligned, so it carries the part's placement tilt as
            # if it were 0 degrees; the canvas then keeps that skew and every pixel
            # comparison works through it. The edge cluster recovers the real rect.
            rect = part_rect_from_edges(gray, (fx, fy, fw, fh), cfg)
            if rect is None:
                contour = np.array([[[int(cx - pw / 2), int(cy - ph / 2)]],
                                    [[int(cx + pw / 2), int(cy - ph / 2)]],
                                    [[int(cx + pw / 2), int(cy + ph / 2)]],
                                    [[int(cx - pw / 2), int(cy + ph / 2)]]], np.int32)
                roi = _roi_from_contour(gray, contour, "shape+box", pose.peak)
            else:
                roi = _roi_from_rect(gray, rect, "shape+edges", pose.peak)
            roi.margin = pose.margin
            return roi
        if method == "shape":
            detail = (f"peak={pose.peak:.3f}" if pose is not None
                      else "model larger than the frame")
            raise ValueError(f"shape match failed: {detail}")
        model = model.grey

    if isinstance(model, np.ndarray) and method in ("auto", "template"):
        try:
            return _roi_from_template(gray, model, float(rcfg.get("min_peak", 0.35)), cfg)
        except ValueError:
            if method == "template":
                raise

    roi = _roi_from_threshold(gray, cfg)
    if roi.area_ratio > float(rcfg.get("max_area_ratio", 0.5)):
        # Almost always means the background was grabbed instead of the part.
        roi.method = f"{roi.method}?"
    return roi
