from __future__ import annotations

import numpy as np

from ..config import Config
from .roi import ROI

_FEATURES = {
    "aspect": "bbox_aspect",
    "mean": "mean",
    "fill": "fill",
    "edge": "edge_density",
}


def _matches(roi: ROI, rule: dict) -> bool:
    for key, attr in _FEATURES.items():
        value = getattr(roi, attr)
        for suffix, test in (
            ("gt", lambda t, v: v > t),
            ("lt", lambda t, v: v < t),
            ("ge", lambda t, v: v >= t),
            ("le", lambda t, v: v <= t),
        ):
            bound = rule.get(f"{key}_{suffix}")
            if bound is not None and not test(bound, value):
                return False
        span = rule.get(f"{key}_between")
        if span is not None and not (float(span[0]) <= value <= float(span[1])):
            return False
    return True


def classify_view(roi: ROI, cfg: Config, folder_hint: str | None = None) -> str:
    vcfg = cfg.get("preprocess.view")
    if vcfg.get("assignment", "manual") == "manual":
        return folder_hint or str(vcfg.get("default", "default"))
    for rule in vcfg.get("auto", {}).get("rules", []) or []:
        if _matches(roi, rule):
            return str(rule["id"])
    return str(vcfg.get("auto", {}).get("fallback", "unknown"))
