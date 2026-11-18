from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_DEFAULTS: dict[str, Any] = {
    "paths": {"data": "data", "runs": "runs"},
    "ui": {"roots": ["."], "allow_system_drives": True, "max_entries": 400},
    "preprocess": {
        "background": "auto",
        "roi": {"method": "auto", "min_area_ratio": 0.0002, "margin_ratio": 0.10,
                "close_kernel": 5, "max_area_ratio": 0.5,
                "template_source": "edges", "template_crop": 0.26, "min_peak": 0.35,
                "anchor_candidates": 0, "anchor_comparators": 32, "anchor_scale": 1.0,
                "anchor_refine": 10, "anchor_probe_images": 48, "anchor_image": "",
                "search_reach": 0.8, "edge_percentile": 99.0, "edge_dilate": 15,
                "crop_margin": 0.12, "window_slack": 0.5, "fit_margin": 0.10},
        "canonical": {"size": [256, 256], "long_side_horizontal": True,
                      "flip_resolve": "template", "elongated_aspect": 1.15,
                     
                      "band_blur": 2.0, "band_inset": 0.10, "ambiguous_margin": 0.02},
        "register": {"method": "phase_correlation", "max_shift": 16},
        "alignment": {"min_ncc": 0.90, "min_gap": 0.15},
        "shape": {"angles": [0, 90, 180, 270], "edge_percentile": 99.5,
                  "crop_margin": 0.08, "min_peak": 0.15, "min_points": 24,
                  "fit_percentile": 99.0},
        "view": {
            "assignment": "manual",
            "default": "top",
            "auto": {"fallback": "unknown", "rules": []},
        },
        "annotation": {"strip": True, "reject_area_ratio": 0.02},
    },
    "detectors": {
        "crack": {"jitter_px": 2, "floor_ratio": 0.5, "cut_margin": 1.15},
    },
    "fusion": {"method": "max", "pixel_cut": 1.5},
    "classify": {"rules": {}, "default": "stain", "min_area": 12, "border_px": 2,
                 "extent_base": 1.0, "extent_cut": 0.0, "extent_evidence": True, "merge_px": 3},
    "calibration": {"pixel_percentile": 99.9, "reference_percentile": 99.0, "image_margin": 1.15, "min_good_per_view": 30,
                    "region_probe": 1.0, "calibration_folds": 2, "min_images_per_fold": 4,
                    "max_fold_images": 60},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULTS))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def enabled_detectors(self) -> list[str]:
        return [name for name, cfg in self.get("detectors", {}).items() if cfg.get("enabled", True)]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            here = Path(__file__).resolve().parents[2]
            path = here / "config" / "default.yaml"
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(_merge(_DEFAULTS, loaded))
