from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import GOOD, NG, Sample, by_view, contact_sheet, discover
from .pipeline import Pipeline


@dataclass
class Metrics:
    view: str
    n_good: int
    n_ng: int
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0
    auc_roc: float | None = None
    recall: float | None = None
    false_positive_rate: float | None = None
    type_accuracy: float | None = None
    type_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kind_of(sample: Sample) -> str | None:
    parent = sample.path.parent.name
    return parent if parent not in (GOOD, NG) else None


def evaluate(pipeline: Pipeline, samples: list[Sample] | None = None,
             out_dir: str | Path | None = None) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    samples = samples if samples is not None else discover(pipeline.data_dir)
    if not samples:
        raise FileNotFoundError(f"no samples under {pipeline.data_dir}")

    grouped = by_view(samples)
    metrics: dict[str, Metrics] = {}
    rows: list[dict[str, Any]] = []
    missed: list[tuple[str, Path]] = []
    false_alarms: list[tuple[str, Path]] = []

    for view, group in grouped.items():
        m = Metrics(view=view, n_good=len(group[GOOD]), n_ng=len(group[NG]))
        scores, truths, type_hits, type_total = [], [], 0, 0
        for sample in group[GOOD] + group[NG]:
            result = pipeline.predict(sample.path, view=view)
            truth = sample.label == NG
            scores.append(result.image_score)
            truths.append(int(truth))
            predicted_labels = {d.label for d in result.defects}
            kind = _kind_of(sample)
            if truth and kind:
                type_total += 1
                type_hits += int(kind in predicted_labels)

            if truth and result.is_defect:
                m.true_positive += 1
            elif truth:
                m.false_negative += 1
                missed.append((f"{sample.path.name} score={result.image_score:.2f}", sample.path))
            elif result.is_defect:
                m.false_positive += 1
                false_alarms.append((f"{sample.path.name} score={result.image_score:.2f}", sample.path))
            else:
                m.true_negative += 1

            rows.append({
                "file": sample.path.name, "view": view, "truth": "ng" if truth else "good",
                "predicted": "ng" if result.is_defect else "good",
                "image_score": round(result.image_score, 4),
                "labels": ",".join(sorted(predicted_labels)) or "-",
                "n_defects": len(result.defects),
            })

        if len(set(truths)) == 2:
            m.auc_roc = float(roc_auc_score(truths, scores))
        if m.true_positive + m.false_negative:
            m.recall = m.true_positive / (m.true_positive + m.false_negative)
        if m.false_positive + m.true_negative:
            m.false_positive_rate = m.false_positive / (m.false_positive + m.true_negative)
        if type_total:
            m.type_accuracy = type_hits / type_total
            m.type_total = type_total
        metrics[view] = m

    report: dict[str, Any] = {
        "per_view": {v: m.to_dict() for v, m in metrics.items()},
        "totals": {
            "ng": sum(m.n_ng for m in metrics.values()),
            "good": sum(m.n_good for m in metrics.values()),
            "recall": _micro(metrics, "true_positive", "false_negative"),
            "false_positive_rate": _micro(metrics, "false_positive", "true_negative"),
        },
        "rows": rows,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(rows, out / "predictions.csv")
        contact_sheet(missed, out / "missed_defects.png")
        contact_sheet(false_alarms[:40], out / "false_alarms.png")
        report["artifacts"] = str(out)
    return report


def _micro(metrics: dict[str, Metrics], hit: str, miss: str) -> float | None:
    total = sum(getattr(m, hit) + getattr(m, miss) for m in metrics.values())
    return round(sum(getattr(m, hit) for m in metrics.values()) / total, 4) if total else None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
