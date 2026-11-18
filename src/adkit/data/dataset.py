from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..imaging import SUPPORTED_SUFFIXES, read_frame, write_image

GOOD, NG, INBOX = "good", "ng", "_review"


@dataclass
class Sample:
    path: Path
    view: str
    label: str

    @property
    def is_good(self) -> bool:
        return self.label == GOOD


def discover(data_dir: str | Path) -> list[Sample]:
    """Layout is <data>/<view>/{good,ng}/<images>."""
    root = Path(data_dir)
    samples: list[Sample] = []
    if not root.exists():
        return samples
    for view_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for label in (GOOD, NG):
            label_dir = view_dir / label
            if not label_dir.is_dir():
                continue
            # ng/ may be split into defect-type subfolders: ng/crack, ng/stain, ...
            files = [p for p in label_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]
            for kind_dir in sorted(d for d in label_dir.iterdir() if d.is_dir()):
                files += [p for p in kind_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES]
            for path in sorted(files):
                samples.append(Sample(path, view_dir.name, label))
    return samples


def by_view(samples: list[Sample]) -> dict[str, dict[str, list[Sample]]]:
    grouped: dict[str, dict[str, list[Sample]]] = {}
    for sample in samples:
        grouped.setdefault(sample.view, {GOOD: [], NG: []}).setdefault(sample.label, []).append(sample)
    return grouped


def ensure_layout(data_dir: str | Path, views: list[str]) -> Path:
    root = Path(data_dir)
    for view in views:
        for label in (GOOD, NG, INBOX):
            (root / view / label).mkdir(parents=True, exist_ok=True)
    return root


def thumbnail(path: str | Path, size: int = 220) -> np.ndarray:
    frame = read_frame(path, strip_annotation=True)
    gray = frame.u8
    scale = size / max(gray.shape)
    resized = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)


def contact_sheet(items: list[tuple[str, Path]], out_path: str | Path, cols: int = 5,
                  cell: int = 240, label_height: int = 22) -> Path | None:
    if not items:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = int(np.ceil(len(items) / cols))
    sheet = np.full((rows * (cell + label_height), cols * cell, 3), 245, np.uint8)
    for index, (caption, path) in enumerate(items):
        thumb = thumbnail(path, cell)
        thumb = cv2.resize(thumb, (cell, cell), interpolation=cv2.INTER_AREA)
        r, c = divmod(index, cols)
        y, x = r * (cell + label_height), c * cell
        sheet[y:y + cell, x:x + cell] = thumb
        cv2.putText(sheet, caption[:34], (x + 3, y + cell + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    write_image(out_path, sheet)
    return out_path


def write_manifest(rows: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def move_into(sample_dir: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    return Path(shutil.move(str(sample_dir), str(target / sample_dir.name)))
