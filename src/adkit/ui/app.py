from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import Config
from ..data import GOOD, INBOX, NG, discover
from ..imaging import SUPPORTED_SUFFIXES, list_images
from ..pipeline import Pipeline
from .render import render_result

_TOKEN = re.compile(r"^[a-f0-9]{16}$")
_ARTIFACT = re.compile(r"^(?:original|canonical|overlay|boxes|crop\d+|mask\d+)\.png$")
# A view or defect-type name becomes a path segment under data/, and the reviewer now types
# it instead of picking from a closed list. Anything with a separator or a leading dot is a
# traversal attempt, not a product name - `../..` used to be reachable straight from the UI.
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,31}$")
_STATIC = Path(__file__).parent / "static"


class PrepareRequest(BaseModel):
    src: str


class FitRequest(BaseModel):
    views: list[str] | None = None
    detectors: list[str] | None = None


class PredictRequest(BaseModel):
    path: str
    view: str | None = None


class SweepRequest(BaseModel):
    src: str
    limit: int = 60


class ReviewRequest(BaseModel):
    path: str
    decision: str
    view: str | None = None
    kind: str | None = None


class UpdateRequest(BaseModel):
    views: list[str] | None = None
    detectors: list[str] | None = None


class RollbackRequest(BaseModel):
    view: str
    version: str


def _gate_summary(gate: dict | None) -> dict[str, Any]:
    """What the online-update gate will be judged against, in the numbers it uses.

    `false_alarms` is measured on the images the model trained on, so on its own it reads
    0 for any fit and proves nothing; `holdout_*` is the same protocol applied to images the
    fold never saw, which is the rate a real shift would produce. Both are shown because the
    gap between them is the honest answer to 'how much data do we have'.
    """
    if not gate:
        return {}
    hold = gate.get("holdout") or {}
    return {"n_good": gate.get("n_good"), "false_alarms": gate.get("false_alarms"),
            "ng_known": len(gate.get("ng_paths", [])), "ng_caught": gate.get("ng_caught"),
            "holdout_fa": hold.get("false_alarms"), "holdout_n": hold.get("n_scored"),
            "holdout_ng_caught": hold.get("ng_caught"),
            "holdout_ng_known": hold.get("ng_known"),
            "holdout_by": hold.get("by_detector"),
            "median_ncc": round(gate["median_ncc"], 3) if gate.get("median_ncc") is not None else None}


def _fitted_views(pipeline: Pipeline) -> list[str]:
    """Views that have a published version on disk, not just fitted-in-this-session ones."""
    if not pipeline.runs_dir.exists():
        return []
    return [p.name for p in sorted(pipeline.runs_dir.iterdir())
            if p.is_dir() and (p / "current.json").exists()]


class UIState:
    def __init__(self, root: Path):
        self.root = root
        self.config_path = root / "config" / "default.yaml"
        self.cfg = Config.load(self.config_path)
        self.pipeline = Pipeline(self.cfg, root=root)
        self.cache: dict[str, dict[str, Any]] = {}
        try:
            self.pipeline.load()
        except Exception:
            pass  # nothing fitted yet - the UI still works for prepare/review

    def reload(self) -> None:
        self.cfg = Config.load(self.config_path)
        self.pipeline = Pipeline(self.cfg, root=self.root)
        self.pipeline.load()


def _drive_roots() -> list[Path]:
    import os
    import string

    if os.name == "nt":
        return [Path(f"{letter}:/").resolve() for letter in string.ascii_uppercase
                if Path(f"{letter}:").exists()]
    return [Path("/")]


def allowed_roots(cfg: Config, project_root: Path) -> list[Path]:
    """Directories the UI may read. The project root always is one; drive roots are
    opt-in so this stays safe if anyone ever binds it beyond localhost."""
    roots = []
    for entry in cfg.get("ui.roots", ["."]) or ["."]:
        candidate = Path(entry)
        roots.append((project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    if cfg.get("ui.allow_system_drives", True):
        roots += _drive_roots()
    return roots


def resolve_input(raw: str, roots: list[Path], project_root: Path) -> Path:
    candidate = Path(raw)
    candidate = (project_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise HTTPException(403, f"{raw} 不在允许浏览的目录内；改 config 的 ui.roots 或 ui.allow_system_drives")
    if not candidate.exists():
        raise HTTPException(404, f"路径不存在：{raw}")
    return candidate


def reference(path: Path, project_root: Path) -> str:
    """Short label the client sends back: project-relative when possible, else absolute."""
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def create_app(root: str | Path | None = None) -> FastAPI:
    project_root = Path(root or Path.cwd()).resolve()
    state = UIState(project_root)
    artifact_root = project_root / "runs" / "ui"
    app = FastAPI(title="adkit", docs_url="/api/docs", openapi_url="/api/openapi.json")

    def resolve(raw: str) -> Path:
        return resolve_input(raw, allowed_roots(state.cfg, project_root), project_root)

    @app.get("/")
    def index() -> FileResponse:
        # The whole interface is one inline HTML file with no build hash, so a cached copy
        # runs an old version of the page silently - every panel renders, nothing errors, and
        # the behaviour nobody asked for is what the reviewer sees. Verified the hard way: a
        # fixed overlay kept looking broken because the browser served the previous page.
        return FileResponse(_STATIC / "index.html", media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/browse")
    def browse(path: str = ".") -> dict[str, Any]:
        """One level of the file tree, for the folder picker."""
        directory = resolve(path)
        if directory.is_file():
            directory = directory.parent
        limit = int(state.cfg.get("ui.max_entries", 400))
        try:
            entries = sorted((p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")),
                             key=lambda p: p.name.lower())
            image_count = sum(1 for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        except OSError as exc:
            raise HTTPException(400, f"读不了该目录：{exc}")
        parent = directory.parent
        return {
            "path": str(directory),
            "ref": reference(directory, project_root),
            "parent": reference(parent, project_root) if parent != directory else None,
            "at_root": parent == directory,
            "image_count": image_count,
            "dirs": [{"name": p.name, "ref": reference(p, project_root)} for p in entries[:limit]],
            "truncated": len(entries) > limit,
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        samples = discover(state.pipeline.data_dir)
        grouped: dict[str, dict[str, int]] = {}
        for sample in samples:
            counts = grouped.setdefault(sample.view, {GOOD: 0, NG: 0})
            counts[sample.label] += 1

        torch_info: dict[str, Any] = {"available": False}
        try:
            import torch
            torch_info = {"available": True, "version": torch.__version__, "cuda": torch.cuda.is_available()}
        except Exception as exc:
            torch_info = {"available": False, "error": str(exc)[:200]}

        views = {}
        for name, bundle in state.pipeline.bundles.items():
            views[name] = {
                "detectors": [d.name for d in bundle.detectors],
                "image_margin": bundle.calibration.image_margin,
                "per_detector": {k: {"lo": round(v.lo, 4), "hi": round(v.hi, 4),
                                     "image_ref": round(v.image_ref, 4)}
                                 for k, v in bundle.calibration.detectors.items()},
                # What is actually deciding, not what the config says: the cut is calibrated
                # from the goods at fit time.
                "version": state.pipeline._current_version(name),
                "pixel_cut": round(bundle.calibration.pixel_cut, 4),
                # both region lines, or a box that a peak cannot explain looks like a bug
                "extent_cut": round(bundle.calibration.extent_cut, 5),
                "gate": _gate_summary(state.pipeline._anchor_gate(name)),
            }

        # Every view name the operator could route an image to: the directories under data/
        # that are laid out as views - the ones discovered with samples, and the fitted ones.
        # Not "every directory": someone dropped a raw benchmark tree into data/ (train/test/
        # ground_truth) and a stray `runs` lives there too, and offering both as a view to
        # sort into is how a product ends up trained on the wrong folder.
        def is_view(directory: Path) -> bool:
            return directory.is_dir() and any(
                (directory / label).is_dir() for label in (GOOD, NG, INBOX))

        on_disk = {p.name for p in state.pipeline.data_dir.iterdir() if is_view(p)} \
            if state.pipeline.data_dir.exists() else set()
        view_names = sorted(on_disk | set(grouped) | set(views))

        return {
            "root": str(project_root),
            "data_dir": str(state.pipeline.data_dir),
            "view_names": view_names,
            "python": sys.version.split()[0],
            "torch": torch_info,
            "detectors": state.cfg.enabled_detectors,
            "dataset": grouped,
            "views": views,
            "min_good_per_view": state.cfg.get("calibration.min_good_per_view"),
            # the floor only - the value actually used is calibrated per view and shown above
            "pixel_cut_floor": state.cfg.get("fusion.pixel_cut"),
            "alignment_min_ncc": state.cfg.get("preprocess.alignment.min_ncc"),
            "alignment": {k: [round(v, 3) for v in vals] for k, vals in state.pipeline.alignment.items()},
        }

    @app.get("/api/images")
    def images(src: str = "image") -> dict[str, Any]:
        directory = resolve(src)
        paths = list_images(directory)
        return {"src": src, "count": len(paths),
                "items": [{"path": reference(p, project_root), "name": p.name} for p in paths]}

    @app.get("/api/dataset")
    def dataset() -> dict[str, Any]:
        items = []
        for sample in discover(state.pipeline.data_dir):
            kind = sample.path.parent.name if sample.path.parent.name not in (GOOD, NG) else None
            items.append({"path": reference(sample.path, project_root), "name": sample.path.name,
                          "view": sample.view, "label": sample.label, "kind": kind})
        return {"count": len(items), "items": items}

    @app.post("/api/prepare")
    def prepare(request: PrepareRequest) -> dict[str, Any]:
        source = resolve(request.src)
        rows = state.pipeline.prepare(source)
        ok = [r for r in rows if r.get("status") == "ok"]
        return {"total": len(rows), "usable": len(ok),
                "rejected": [r for r in rows if r.get("status") != "ok"],
                "views": sorted({r["view"] for r in ok}),
                "manifest": "runs/prepare_manifest.csv",
                "rows": ok}

    @app.post("/api/fit")
    def fit(request: FitRequest) -> dict[str, Any]:
        state.pipeline.backbones.clear()
        result = state.pipeline.fit(request.views, request.detectors)
        state.cache.clear()
        return {
            "views": {name: {
                "detectors": [d.name for d in bundle.detectors],
                "image_margin": bundle.calibration.image_margin,
                "per_detector": {k: {"lo": round(v.lo, 4), "hi": round(v.hi, 4),
                                     "image_ref": round(v.image_ref, 4)}
                                 for k, v in bundle.calibration.detectors.items()},
            } for name, bundle in result["bundles"].items()},
            "warnings": result["warnings"],
            # what the fit measured about itself. Separate from warnings because at four good
            # images "not validated on unseen parts yet" is true of every model there is.
            "notes": result["notes"],
            "alignment": {k: [round(v, 3) for v in vals] for k, vals in state.pipeline.alignment.items()},
        }

    def _score(result) -> dict[str, Any]:
        payload = result.to_dict()
        payload["url"] = reference(result.path, project_root)
        payload.update(render_result(result, artifact_root))
        return payload

    @app.post("/api/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        path = resolve(request.path)
        if path.is_dir():
            raise HTTPException(400, "expected a file, got a directory")
        try:
            result = state.pipeline.predict(path, request.view)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        state.cache[str(path)] = result
        return _score(result)

    @app.post("/api/sweep")
    def sweep(request: SweepRequest) -> dict[str, Any]:
        directory = resolve(request.src)
        rows = []
        for path in list_images(directory)[: max(1, request.limit)]:
            try:
                result = state.pipeline.predict(path)
            except Exception as exc:
                rows.append({"path": reference(path, project_root), "error": str(exc)[:180]})
                continue
            state.cache[str(path)] = result
            payload = result.to_dict()
            payload["url"] = reference(path, project_root)
            rows.append(payload)
        rows.sort(key=lambda r: -r.get("image_score", -1))
        return {"count": len(rows), "rows": rows,
                "ng": sum(1 for r in rows if r.get("is_defect"))}

    @app.post("/api/review")
    def review(request: ReviewRequest) -> dict[str, Any]:
        if request.decision not in (GOOD, NG):
            raise HTTPException(400, "decision must be 'good' or 'ng'")
        source = resolve(request.path)
        view = request.view or "default"
        kind = request.kind or "unsorted"
        for label, value in (("view", view), ("kind", kind)):
            if not _SEGMENT.match(value):
                raise HTTPException(
                    400, f"{label} {value!r} 不是合法名字：只能用字母数字开头，"
                         "包含 . _ -，不超过 32 个字符，不能含路径分隔符")
        if request.decision == GOOD:
            target_dir = state.pipeline.data_dir / view / GOOD
        else:
            target_dir = state.pipeline.data_dir / view / NG / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = target_dir / source.name
        shutil.copy2(source, destination)
        return {"copied_to": reference(destination, project_root),
                "view": view, "decision": request.decision, "kind": request.kind}

    @app.post("/api/update")
    def update(request: UpdateRequest) -> dict[str, Any]:
        """Refit from the current data, but only publish what passes the anchor gate."""
        state.pipeline.backbones.clear()
        result = state.pipeline.update(request.views, request.detectors)
        state.cache.clear()
        return {"accepted": result["accepted"], "rejected": result["rejected"],
                "warnings": result["warnings"], "notes": result["notes"],
                "versions": {view: state.pipeline.versions(view)
                             for view in (request.views or _fitted_views(state.pipeline))}}

    @app.get("/api/versions")
    def versions() -> dict[str, Any]:
        return {"views": {view: {
            "current": state.pipeline._current_version(view),
            "gate": state.pipeline._anchor_gate(view),
            "versions": state.pipeline.versions(view),
        } for view in _fitted_views(state.pipeline)}}

    @app.post("/api/rollback")
    def rollback(request: RollbackRequest) -> dict[str, Any]:
        if not state.pipeline.rollback(request.view, request.version):
            raise HTTPException(404, f"no version {request.version} for view {request.view}")
        state.cache.clear()
        return {"view": request.view, "current": state.pipeline._current_version(request.view)}

    @app.get("/api/artifact/{token}/{name}")
    def artifact(token: str, name: str) -> FileResponse:
        if not _TOKEN.match(token) or not _ARTIFACT.match(name):
            raise HTTPException(400, "bad artifact reference")
        path = artifact_root / token / name
        if not path.is_file():
            raise HTTPException(404, "artifact not rendered yet - run predict first")
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "max-age=3600"})

    return app
