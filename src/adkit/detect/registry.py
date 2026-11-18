from __future__ import annotations

from typing import Any

from .base import Detector
from .crack import CrackDetector
from .missing import MissingDetector
from .patchcore import PatchCore
from .stefann import STEFANN
from .stain import StainDetector

BUILDERS: dict[str, type[Detector]] = {
    PatchCore.name: PatchCore,
    STEFANN.name: STEFANN,
    CrackDetector.name: CrackDetector,
    MissingDetector.name: MissingDetector,
    StainDetector.name: StainDetector,
}


def build_detector(name: str, cfg: dict[str, Any] | None = None) -> Detector:
    try:
        cls = BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown detector {name!r}; available: {sorted(BUILDERS)}") from exc
    return cls(cfg or {})
