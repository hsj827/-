from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_SUFFIXES = {".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg"}


@dataclass
class Frame:
    path: Path
    gray: np.ndarray
    color: np.ndarray | None = None
    annotation: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def annotation_ratio(self) -> float:
        return 0.0 if self.annotation is None else float(self.annotation.mean())

    @property
    def u8(self) -> np.ndarray:
        return np.clip(self.gray * 255.0, 0, 255).astype(np.uint8)


def _annotation_overlay(color: np.ndarray) -> np.ndarray | None:
    """Human-drawn red boxes are overlays, not product features."""
    b, g, r = (color[..., i].astype(np.int16) for i in range(3))
    mask = (r > 130) & (r - g > 55) & (r - b > 55)
    return mask if mask.any() else None


def _imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread goes through the ANSI API and fails on non-ASCII Windows paths."""
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)


def read_frame(path: str | Path, strip_annotation: bool = True) -> Frame:
    path = Path(path)
    raw = _imread_unicode(path)
    if raw is None:
        raise OSError(f"cannot decode image: {path}")

    if raw.ndim == 2:
        gray, color = raw, None
    else:
        if raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        elif raw.shape[2] == 1:
            raw = raw[..., 0]
        color = raw
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw

    annotation = _annotation_overlay(color) if color is not None else None
    if annotation is not None and strip_annotation:
        gray = cv2.inpaint(gray, annotation.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return Frame(path=path, gray=gray.astype(np.float32) / 255.0, color=color, annotation=annotation)


def write_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    suffix = path.suffix.lower() if path.suffix.lower() in SUPPORTED_SUFFIXES else ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"cannot encode image: {path}")
    buffer.tofile(str(path))
    return path


def list_images(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
