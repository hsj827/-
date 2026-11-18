from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..imaging import read_frame, write_image
from ..pipeline import Result
from ..postprocess import viz
from ..preprocess import map_back

MAX_ORIGINAL_EDGE = 720
ARTIFACT_NAMES = ("original.png", "canonical.png", "overlay.png", "boxes.png")


def _canvas_mask(defects: list[Any], shape: tuple[int, int]) -> np.ndarray | None:
    """Union of the per-defect masks, placed back onto the canonical canvas.

    The canvas size must be the one the geometry was built for: map_back resizes its input
    to the canvas before warping, so a tightly cropped array would land at the wrong scale.
    """
    canvas = np.zeros(shape, np.uint8)
    placed = False
    for defect in defects:
        if getattr(defect, "mask", None) is None:
            continue
        x, y, w, h = defect.bbox
        x1, y1 = min(x + w, shape[1]), min(y + h, shape[0])
        canvas[y:y1, x:x1] = np.maximum(canvas[y:y1, x:x1],
                                        defect.mask[:y1 - y, :x1 - x].astype(np.uint8))
        placed = True
    return canvas.astype(bool) if placed else None


def token_for(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256(f"{path}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()
    return digest[:16]


def _downscale(image: np.ndarray, max_edge: int = MAX_ORIGINAL_EDGE) -> np.ndarray:
    scale = max_edge / max(image.shape[:2])
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)


def render_result(result: Result, artifact_root: Path) -> dict[str, Any]:
    """Write the PNGs a reviewer needs and return their artifact URLs."""
    token = token_for(result.path)
    directory = artifact_root / token
    directory.mkdir(parents=True, exist_ok=True)

    frame = read_frame(result.path, strip_annotation=True)
    original = _downscale(cv2.cvtColor(frame.u8, cv2.COLOR_GRAY2BGR))
    write_image(directory / "original.png", original)

    canonical = cv2.cvtColor((np.clip(result.canonical, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    write_image(directory / "canonical.png", canonical)

    canvas_mask = _canvas_mask(result.defects, result.canonical.shape[:2])
    full_mask = None
    if canvas_mask is not None and result.geometry is not None:
        full_mask = map_back(canvas_mask.astype(np.float32), result.geometry,
                             frame.u8.shape[:2]) > 0.5

    # The judgement pages are drawn on the ORIGINAL frame, not the canonical one. A reviewer
    # compares what they see on the glass with the picture on the screen; if the canonicaliser
    # rotated the part - which it does whenever it guesses the orientation - a canonical-space
    # overlay puts the box on the opposite side of the part and the first conclusion is that
    # the detector missed an obvious break. Measured on a labelled bottle: the box covered 57%
    # of the ground-truth mask while appearing to sit on clean glass.
    if result.full_map is not None and result.geometry is not None:
        heat = viz.heatmap(result.full_map, frame.u8.shape[:2], gray_base=frame.u8)
        if full_mask is not None:
            contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(heat, contours, -1, (0, 255, 255), 2)
        heat = _downscale(heat)
        boxes = _original_boxes(result.defects, result.geometry, frame.u8.shape[:2],
                               (heat.shape[1], heat.shape[0]))
        write_image(directory / "overlay.png", heat)
        write_image(directory / "boxes.png", viz.annotate(heat, result.defects, boxes=boxes))
    else:
        heat = viz.heatmap(result.fused_map, canonical.shape[:2], gray_base=result.canonical)
        write_image(directory / "overlay.png", heat)
        write_image(directory / "boxes.png", viz.annotate(heat, result.defects))

    urls = {name[:-4]: f"/api/artifact/{token}/{name}" for name in ARTIFACT_NAMES}
    urls["crops"] = _write_crops(directory, token, frame.u8, result.defects, result.geometry,
                                 mask=full_mask)

    return {
        "token": token,
        "artifacts": urls,
        "original_size": list(original.shape[:2]),
    }


def _original_boxes(defects: list[Any], geometry: Any, frame_shape: tuple[int, int],
                    out_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Where each defect sits in the picture the reviewer is looking at.

    Taken from the defect's own mask rather than by projecting its canonical bbox corners:
    the corners of a rotated rectangle are not the rectangle, and the mask is the claim.
    """
    scale_x = out_size[0] / max(1, frame_shape[1])
    scale_y = out_size[1] / max(1, frame_shape[0])
    canvas_h, canvas_w = (geometry.canvas[1], geometry.canvas[0]) if geometry.canvas else \
        (int(frame_shape[0] * scale_y), int(frame_shape[1] * scale_x))
    boxes: list[tuple[int, int, int, int]] = []
    for defect in defects:
        mask = getattr(defect, "mask", None)
        box = None
        if mask is not None and geometry is not None:
            canvas = np.zeros((canvas_h, canvas_w), np.uint8)
            x, y, w, h = defect.bbox
            x1, y1 = min(x + w, canvas_w), min(y + h, canvas_h)
            canvas[y:y1, x:x1] = mask[:y1 - y, :x1 - x].astype(np.uint8)
            placed = map_back(canvas.astype(np.float32), geometry, frame_shape) > 0.5
            if placed.any():
                bx, by, bw, bh = cv2.boundingRect(placed.astype(np.uint8))
                box = (int(bx * scale_x), int(by * scale_y),
                       max(1, int(bw * scale_x)), max(1, int(bh * scale_y)))
        if box is None:
            x, y, w, h = _bbox_in_original(defect.bbox, geometry, frame_shape)
            box = (int(x * scale_x), int(y * scale_y),
                   max(1, int(w * scale_x)), max(1, int(h * scale_y)))
        boxes.append(box)
    return boxes


def _bbox_in_original(bbox: tuple[int, int, int, int], geometry: Any, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Detection boxes live in canonical space; crops must come from the full-resolution original."""
    if geometry is None:
        return bbox
    x, y, w, h = bbox
    corners = np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]])
    points = cv2.perspectiveTransform(corners, geometry.inverse)[0]
    x0 = max(0, int(points[:, 0].min()))
    x1 = min(shape[1] - 1, int(points[:, 0].max()))
    y0 = max(0, int(points[:, 1].min()))
    y1 = min(shape[0] - 1, int(points[:, 1].max()))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _write_crops(directory: Path, token: str, frame_u8: np.ndarray, defects: list[Any],
                 geometry: Any = None, size: int = 220,
                 mask: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Tight crops around each detection so a reviewer can judge it without zooming."""
    entries = []
    for index, defect in enumerate(defects[:6]):
        x, y, w, h = _bbox_in_original(defect.bbox, geometry, frame_u8.shape[:2])
        pad = max(24, int(max(w, h) * 0.25))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame_u8.shape[1], x + w + pad), min(frame_u8.shape[0], y + h + pad)
        crop = frame_u8[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = size / max(crop.shape[:2])
        resized = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
        crop = cv2.resize(crop, resized, interpolation=cv2.INTER_NEAREST)

        entry: dict[str, Any] = {
            "index": index,
            "label": defect.label,
            "score": round(defect.score, 4),
            "bbox": [x, y, w, h],
            "url": "",
        }
        region = None
        if mask is not None:
            window = mask[y0:y1, x0:x1]
            # Counted at full resolution: the crop below is upscaled for display, and
            # reporting its pixel count would overstate how much material the mask covers.
            entry["mask_px"] = int(window.sum())
            region = cv2.resize(window.astype(np.uint8), resized,
                                interpolation=cv2.INTER_NEAREST) > 0
            write_image(directory / f"mask{index}.png", (region.astype(np.uint8) * 255))
            entry["mask_url"] = f"/api/artifact/{token}/mask{index}.png"

        shown = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if region is not None and region.any():
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(shown, contours, -1, (0, 255, 0), 1)
        write_image(directory / f"crop{index}.png", shown)
        entry["url"] = f"/api/artifact/{token}/crop{index}.png"
        entries.append(entry)
    return entries
