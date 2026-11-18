from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

from .config import Config
from .data import GOOD, NG, discover
from .evaluate import evaluate as run_evaluation
from .imaging import list_images, write_image
from .pipeline import Pipeline
from .postprocess import viz


def _pipeline(config_path: str | None, root: str | None) -> Pipeline:
    cfg = Config.load(config_path) if config_path else Config.load()
    return Pipeline(cfg, root or Path.cwd())


@click.group()
def main() -> None:
    """adkit - detect defects by learning only what a good part looks like."""


@main.command()
@click.option("--config", "config_path", default=None, help="path to a yaml config override")
@click.option("--root", default=None, help="project root (default: cwd)")
def status(config_path: str | None, root: str | None) -> None:
    """Report environment, dataset counts, and whether each view has enough goods."""
    pipe = _pipeline(config_path, root)
    click.echo(f"python {sys.version.split()[0]}  cwd={pipe.root}")
    try:
        import torch

        click.echo(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as exc:
        click.echo(f"torch UNAVAILABLE: {exc}")
        click.echo("  PatchCore/STEFANN need torch; classical detectors still work")

    samples = discover(pipe.data_dir)
    if not samples:
        click.echo(f"\nno dataset at {pipe.data_dir} - run `adkit prepare <src>` first")
    else:
        min_good = int(pipe.cfg.get("calibration.min_good_per_view", 30))
        click.echo(f"\ndataset {pipe.data_dir}")
        views: dict[str, dict[str, int]] = {}
        for sample in samples:
            views.setdefault(sample.view, {GOOD: 0, NG: 0})[sample.label] += 1
        for view, counts in sorted(views.items()):
            flag = "OK" if counts[GOOD] >= min_good else f"NEED >= {min_good}"
            click.echo(f"  {view:<18} good={counts[GOOD]:<4} ng={counts[NG]:<4} [{flag}]")

    fitted = [p.name for p in sorted(pipe.runs_dir.iterdir())
              if pipe.runs_dir.exists() and (p / "meta.json").exists()] if pipe.runs_dir.exists() else []
    click.echo(f"\nfitted views: {fitted or '-'}")
    click.echo(f"enabled detectors: {pipe.cfg.enabled_detectors}")


@main.command()
@click.argument("src", type=click.Path(exists=True))
@click.option("--config", "config_path", default=None)
@click.option("--root", default=None)
def prepare(src: str, config_path: str | None, root: str | None) -> None:
    """Canonicalise every image, guess its view, and emit thumbnails for good/ng sorting."""
    pipe = _pipeline(config_path, root)
    rows = pipe.prepare(src)
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]
    click.echo(f"{len(ok)} usable, {len(bad)} rejected")
    for row in bad:
        click.echo(f"  SKIP {row['file']}: {row['status']}")
    seen: dict[str, int] = {}
    for row in ok:
        seen[row["view"]] = seen.get(row["view"], 0) + 1
    click.echo(f"views detected: {seen}")
    click.echo(f"manifest: {pipe.runs_dir / 'prepare_manifest.csv'}")
    click.echo(f"thumbnails: {pipe.data_dir}/<view>/_review/contact_sheet.png")
    click.echo("next: move each image into <view>/good or <view>/ng, then `adkit fit`")


@main.command()
@click.option("--view", multiple=True, help="fit only these views (default: all found)")
@click.option("--detectors", default=None, help="comma list, e.g. patchcore,crack")
@click.option("--config", "config_path", default=None)
@click.option("--root", default=None)
def fit(view: tuple[str, ...], detectors: str | None, config_path: str | None, root: str | None) -> None:
    """Build the good-only models and calibrate thresholds from the good distribution."""
    pipe = _pipeline(config_path, root)
    names = [d.strip() for d in detectors.split(",")] if detectors else None
    result = pipe.fit(list(view) or None, names)
    for name, bundle in result["bundles"].items():
        click.echo(f"\n[{name}] detectors={[d.name for d in bundle.detectors]}")
        click.echo(f"  image threshold = {bundle.calibration.image_threshold:.4f}")
        for det, calib in bundle.calibration.detectors.items():
            click.echo(f"  {det:<10} lo={calib.lo:.4f} hi={calib.hi:.4f} image_max={calib.image_max:.4f}")
    for warning in result["warnings"]:
        click.echo(f"WARNING {warning}", err=True)
    for note in result["notes"]:
        click.echo(f"NOTE  {note}")
    click.echo(f"\nsaved to {pipe.runs_dir}")


@main.command()
@click.argument("src", type=click.Path(exists=True))
@click.option("--view", default=None, help="force a view instead of auto-matching")
@click.option("--out", default=None, help="write overlays + report here")
@click.option("--config", "config_path", default=None)
@click.option("--root", default=None)
def predict(src: str, view: str | None, out: str | None, config_path: str | None, root: str | None) -> None:
    """Score one image or a whole directory."""
    pipe = _pipeline(config_path, root)
    pipe.load()
    out_dir = Path(out) if out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    targets = list_images(src) if Path(src).is_dir() else [Path(src)]
    results = []
    for path in targets:
        result = pipe.predict(path, view)
        results.append(result.to_dict())
        summary = (f"{result.image_score:.3f} {'NG  ' if result.is_defect else 'GOOD'} "
                   f"view={result.view} " + ",".join(sorted({d.label for d in result.defects}) or "-"))
        click.echo(f"{path.name:<44} {summary}")
        click.echo(f"{path.name:<44} {summary}")
        if out_dir:
            base = viz._to_bgr(result.canonical)
            heat = viz.heatmap(result.fused_map, base.shape[:2], gray_base=result.canonical)
            write_image(out_dir / f"{path.stem}_overlay.png", viz.side_by_side(base, viz.annotate(heat, result.defects)))

    if out_dir:
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"overlays + results.json -> {out_dir}")


@main.command()
@click.option("--out", default="runs/evaluation", help="report directory")
@click.option("--config", "config_path", default=None)
@click.option("--root", default=None)
def evaluate(out: str, config_path: str | None, root: str | None) -> None:
    """Recall / false-alarm rate / AUROC against the held-out ng set."""
    pipe = _pipeline(config_path, root)
    pipe.load()
    out_dir = Path(out) if Path(out).is_absolute() else pipe.runs_dir / out
    report = run_evaluation(pipe, out_dir)
    click.echo(json.dumps(report["per_view"], indent=2, ensure_ascii=False))
    click.echo(f"totals: {report['totals']}")
    click.echo(f"artifacts: {report.get('artifacts')}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, type=int, show_default=True)
@click.option("--root", default=None)
def serve(host: str, port: int, root: str | None) -> None:
    """Start the local review UI in a browser."""
    import uvicorn

    from .ui.app import create_app

    project_root = Path(root) if root else Path.cwd()
    click.echo(f"adkit UI -> http://{host}:{port}  (root={project_root})")
    uvicorn.run(create_app(project_root), host=host, port=port, log_level="warning")


@main.command()
@click.option("--config", "config_path", default=None)
@click.option("--root", default=None)
def thresholds(config_path: str | None, root: str | None) -> None:
    """Show the calibrated operating point per view."""
    pipe = _pipeline(config_path, root)
    for path in sorted(pipe.runs_dir.glob("*/calibration.json")) if pipe.runs_dir.exists() else []:
        click.echo(f"[{path.parent.name}]")
        click.echo(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
