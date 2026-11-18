from __future__ import annotations

import cv2
import numpy as np

from .classify import Defect

_COLORS = {
    "crack": (0, 0, 255),
    "missing": (0, 165, 255),
    "stain": (255, 0, 255),
    "unknown": (255, 255, 255),
}


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    image = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def heatmap(score_map: np.ndarray, reference_shape: tuple[int, int], alpha: float = 0.5,
            gray_base: np.ndarray | None = None) -> np.ndarray:
    if score_map.shape != (reference_shape[0], reference_shape[1]):
        score_map = cv2.resize(score_map, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = np.clip(score_map, 0.0, 1.0)
    coloured = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = _to_bgr(gray_base) if gray_base is not None else np.zeros_like(coloured)
    blend = cv2.addWeighted(base, 1.0 - alpha, coloured, alpha, 0)
    blend[norm <= 0.01] = base[norm <= 0.01]
    return blend


def annotate(bgr: np.ndarray, defects: list[Defect], score_map: np.ndarray | None = None,
             label_scale: float = 0.5, boxes: list[tuple[int, int, int, int]] | None = None) -> np.ndarray:
    """Draw one rectangle per defect.

    `boxes` overrides `defect.bbox`, which is what the reviewer page needs: the defects are
    found in the canonical frame, but the picture being judged is the original one, and a box
    from the other frame lands on the wrong side of the part whenever the canonicaliser
    rotated it.
    """
    out = bgr.copy()
    for index, defect in enumerate(defects):
        x, y, w, h = boxes[index] if boxes is not None else defect.bbox
        colour = _COLORS.get(defect.label, _COLORS["unknown"])
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
        text = f"{defect.label} {defect.score:.2f}"
        cv2.putText(out, text, (x, max(12, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, label_scale, colour, 1, cv2.LINE_AA)
    return out


def side_by_side(*panels: np.ndarray, gap: int = 8) -> np.ndarray:
    height = max(p.shape[0] for p in panels)
    prepared = []
    for panel in panels:
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        if panel.shape[0] != height:
            scale = height / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), height))
        prepared.append(panel)
    separator = np.full((height, gap, 3), 40, np.uint8)
    row = [prepared[0]]
    for panel in prepared[1:]:
        row.append(separator)
        row.append(panel)
    return np.hstack(row)
