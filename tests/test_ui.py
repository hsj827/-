from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def client(make, tmp_path, config):
    from fastapi.testclient import TestClient

    from adkit.ui.app import create_app

    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "image").mkdir()
    angles = [0.0, 90.0, 180.0, 270.0, 4.0]
    for i, a in enumerate(angles):
        make(root / "image" / f"g{i}.png", angle=a, seed=i)
    make(root / "image" / "cracked.png", angle=0.0, crack=True, seed=77)
    (root / "config" / "default.yaml").write_text(
        Path("config/default.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with TestClient(create_app(root)) as c:
        yield c, root


def test_index_serves_the_app(client):
    c, _ = client
    response = c.get("/")
    assert response.status_code == 200
    assert "复核台" in response.text
    assert "text/html" in response.headers["content-type"]


def test_status_reports_environment(client):
    c, _ = client
    payload = c.get("/api/status").json()
    assert payload["torch"]["available"] is True
    assert set(payload["detectors"]) >= {"crack", "missing", "stain"}
    assert payload["dataset"] == {}


def test_images_lists_relative_paths(client):
    c, root = client
    payload = c.get("/api/images", params={"src": "image"}).json()
    assert payload["count"] == 6
    assert all(not Path(item["path"]).is_absolute() for item in payload["items"])


def test_prepare_then_fit_then_predict(client):
    c, root = client
    prepared = c.post("/api/prepare", json={"src": "image"}).json()
    assert prepared["usable"] == 6

    for i in range(5):
        c.post("/api/review", json={"path": f"image/g{i}.png", "decision": "good", "view": "top"})
    assert (root / "data" / "top" / "good").is_dir()

    fitted = c.post("/api/fit", json={"detectors": ["crack", "missing", "stain"]}).json()
    assert "top" in fitted["views"]
    assert fitted["views"]["top"]["image_margin"] > 0

    result = c.post("/api/predict", json={"path": "image/cracked.png"}).json()
    assert result["view"] == "top"
    assert result["image_score"] > 0
    assert set(result["artifacts"]) >= {"original", "canonical", "overlay", "boxes"}

    for url in [result["artifacts"][k] for k in ("original", "canonical", "overlay", "boxes")]:
        response = c.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 500


def test_sweep_sorts_by_score(client):
    c, _ = client
    c.post("/api/prepare", json={"src": "image"})
    for i in range(5):
        c.post("/api/review", json={"path": f"image/g{i}.png", "decision": "good", "view": "top"})
    c.post("/api/fit", json={"detectors": ["crack", "missing", "stain"]})
    swept = c.post("/api/sweep", json={"src": "image", "limit": 20}).json()
    scores = [r["image_score"] for r in swept["rows"]]
    assert scores == sorted(scores, reverse=True)
    assert swept["count"] == 6


def test_browse_lists_directories_one_level(client):
    c, root = client
    payload = c.get("/api/browse", params={"path": "."}).json()
    names = {d["name"] for d in payload["dirs"]}
    assert {"image", "config"} <= names
    assert "data" not in names              # only created once prepare runs
    assert payload["image_count"] == 0      # the project root itself holds no images
    assert payload["ref"] == "."
    assert not payload["at_root"]

    c.post("/api/prepare", json={"src": "image"})
    assert "data" in {d["name"] for d in c.get("/api/browse", params={"path": "."}).json()["dirs"]}


def test_browse_reports_image_counts_for_a_folder(client):
    c, _ = client
    payload = c.get("/api/browse", params={"path": "image"}).json()
    assert payload["image_count"] == 6


def test_a_folder_outside_the_project_is_usable(client, tmp_path):
    c, root = client
    external = root.parent / "external_shots"
    external.mkdir(exist_ok=True)
    import cv2
    import numpy as np

    for i in range(2):
        img = np.full((300, 300), 250, np.uint8)
        cv2.rectangle(img, (90, 90), (210, 210), 60, -1)
        cv2.imwrite(str(external / f"e{i}.png"), img)

    payload = c.get("/api/images", params={"src": str(external)}).json()
    assert payload["count"] == 2
    assert all(Path(item["path"]).is_absolute() for item in payload["items"])

    prepared = c.post("/api/prepare", json={"src": str(external)}).json()
    assert prepared["usable"] == 2

    c.post("/api/review", json={"path": str(external / "e0.png"), "decision": "good", "view": "ext"})
    assert (root / "data" / "ext" / "good" / "e0.png").exists()


def test_browsing_is_refused_when_system_drives_are_disabled(client):
    c, root = client
    from fastapi import HTTPException

    from adkit.config import Config
    from adkit.ui.app import allowed_roots, resolve_input

    cfg = Config.load(Path("config/default.yaml"))
    cfg.raw["ui"]["allow_system_drives"] = False
    outside = root.parent / "somewhere_else"
    outside.mkdir(exist_ok=True)
    with pytest.raises(HTTPException, match="不在允许浏览的目录内"):
        resolve_input(str(outside), allowed_roots(cfg, root), root)


def test_review_rejects_unknown_decision(client):
    c, _ = client
    assert c.post("/api/review", json={"path": "image/g0.png", "decision": "maybe"}).status_code == 400


@pytest.mark.parametrize("target", [
    "/api/images?src=../config",
    "/api/predict",
    "/api/artifact/deadbeefdeadbeef/original.png",
])
def test_paths_outside_the_project_are_refused(client, target):
    c, root = client
    if target == "/api/predict":
        response = c.post("/api/predict", json={"path": "../secret.txt"})
        assert response.status_code in (403, 404)
    else:
        assert c.get(target).status_code in (400, 403, 404)


def test_artifact_names_are_whitelisted(client):
    c, _ = client
    assert c.get("/api/artifact/a1b2c3d4e5f60718/../../etc/passwd").status_code in (400, 404)
    assert c.get("/api/artifact/NOT_A_TOKEN/original.png").status_code in (400, 404)


def test_predict_serves_defect_masks(client):
    c, _ = client
    c.post("/api/prepare", json={"src": "image"})
    for i in range(5):
        c.post("/api/review", json={"path": f"image/g{i}.png", "decision": "good", "view": "top"})
    c.post("/api/fit", json={"detectors": ["crack", "missing", "stain"]})
    result = c.post("/api/predict", json={"path": "image/cracked.png"}).json()

    crops = result["artifacts"]["crops"]
    assert crops, "no defect crops rendered"
    assert all("mask_url" in entry and entry["mask_px"] > 0 for entry in crops), crops
    response = c.get(crops[0]["mask_url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_review_can_open_a_view_that_has_never_been_modelled(client):
    """Onboarding a product must not require creating its folder by hand."""
    c, root = client
    images = c.get("/api/images", params={"src": "image"}).json()["items"]
    response = c.post("/api/review", json={"path": images[0]["path"], "decision": "good",
                                           "view": "widget_x"})
    assert response.status_code == 200, response.text
    landed = root / "data" / "widget_x" / "good" / Path(images[0]["path"]).name
    assert landed.exists(), response.json()

    # an empty folder is still a view the operator can route images to
    (root / "data" / "notyet" / "good").mkdir(parents=True)
    names = c.get("/api/status").json()["view_names"]
    assert {"widget_x", "notyet"} <= set(names), names


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", ".hidden", "x" * 33])
def test_review_refuses_names_that_are_not_single_path_segments(client, bad):
    """The view box is free text now, so the name is client input in a path join."""
    c, root = client
    images = c.get("/api/images", params={"src": "image"}).json()["items"]
    for field in ("view", "kind"):
        payload = {"path": images[0]["path"], "decision": "ng", "view": "safe",
                   "kind": "crack", field: bad}
        payload[field] = bad          # the other field stays a legal name
        response = c.post("/api/review", json=payload)
        assert response.status_code == 400, (field, bad, response.text)
    assert not (root.parent / "escape").exists()
    assert list((root / "data").rglob("safe*")) == []


def test_defect_boxes_are_reported_in_the_picture_the_reviewer_sees():
    """A canonical bbox is the wrong rectangle once the part has been rotated.

    The judgement pages are drawn on the original frame, so the box has to be the mapped-back
    mask, not the numbers the detector worked in.
    """
    import numpy as np
    from adkit.preprocess import Geometry
    from adkit.ui.render import _original_boxes

    frame_shape = (400, 400)          # square, so a half-turn is unambiguous
    matrix = np.float32([[-1, 0, 399], [0, -1, 399], [0, 0, 1]])
    geometry = Geometry(matrix=matrix, inverse=matrix, rect=((0, 0), (400, 400), 0.0),
                        canvas=(400, 400), flipped=False, turns=2)

    class D:
        bbox = (10, 10, 40, 40)       # top-left in canonical space
        label = "crack"
        mask = np.ones((40, 40), bool)

    (x, y, w, h), = _original_boxes([D()], geometry, frame_shape, (400, 400))
    assert abs(x - 350) <= 2 and abs(y - 350) <= 2, (x, y, w, h)
    assert (w, h) == (40, 40)


def test_overlay_is_written_on_the_original_frame(client):
    """The page a reviewer judges must not be rotated relative to the photo."""
    from PIL import Image
    import io

    c, root = client
    items = c.get("/api/images", params={"src": "image"}).json()["items"]
    goods = [i for i in items if "crack" not in Path(i["path"]).name][:5]
    for g in goods:
        assert c.post("/api/review", json={"path": g["path"], "decision": "good",
                                           "view": "synth"}).status_code == 200
    fit = c.post("/api/fit", json={"views": ["synth"],
                                   "detectors": ["crack", "stain", "missing"]}).json()
    assert "synth" in fit["views"], fit

    target = next(i for i in items if "crack" in Path(i["path"]).name)
    result = c.post("/api/predict", json={"path": target["path"], "view": "synth"}).json()
    token = result["token"]
    size_of = lambda name: Image.open(io.BytesIO(
        c.get(f"/api/artifact/{token}/{name}.png").content)).size
    sizes = {name: size_of(name) for name in ("original", "canonical", "overlay", "boxes")}
    # the canonical canvas is 128x128; if the judgement pages came back at that size they are
    # still drawn in the model's frame instead of the photo the reviewer is looking at
    assert sizes["overlay"] == sizes["original"] == sizes["boxes"], sizes
    assert sizes["canonical"] != sizes["original"], sizes
