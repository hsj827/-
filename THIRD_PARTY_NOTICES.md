# Third-party notices

This file records the third-party software adkit depends on, the licence each part is under,
and the notices we are obliged to reproduce. The package table below was generated from the
installed environment on 2026-11-18, so the names and versions are what this project
actually ran against; see [Regenerating](#regenerating).

adkit itself is licensed under the Apache License 2.0 - see [LICENSE](LICENSE).

Nothing in the dependency closure is under the GPL, AGPL or LGPL **as its own licence**. The
copyleft text that does turn up in these packages is the notice they carry for components they
redistribute, and every such component is handled explicitly below.

## Runtime dependencies

| Package | Version | Licence (as declared in the installed distribution) |
|---|---|---|
| `annotated-doc` | 0.0.6 | MIT |
| `annotated-types` | 0.8.0 | MIT |
| `anyio` | 4.16.0 | MIT |
| `certifi` | 2026.10.23 | MPL-2.0 |
| `click` | 8.5.1 | BSD-3-Clause |
| `cloudpickle` | 3.1.2 | BSD-3-Clause |
| `fastapi` | 0.162.5 | MIT |
| `filelock` | 4.2.4 | MIT |
| `fsspec` | 2026.9.0 | BSD-3-Clause |
| `h11` | 0.16.0 | MIT |
| `httpcore` | 1.0.9 | BSD-3-Clause |
| `httpx` | 0.28.1 | BSD-3-Clause |
| `idna` | 3.20 | BSD-3-Clause |
| `imageio` | 2.38.1 | BSD-2-Clause |
| `jinja2` | 3.1.6 | (see licenses/) |
| `joblib` | 1.6.0 | BSD-3-Clause |
| `lazy-loader` | 0.6 | BSD-3-Clause |
| `markupsafe` | 3.0.3 | BSD-3-Clause |
| `mpmath` | 1.3.0 | BSD |
| `narwhals` | 2.28.0 | MIT |
| `networkx` | 3.6.1 | BSD-3-Clause |
| `numpy` | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `opencv-python-headless` | 5.0.0.93 | Apache 2.0 |
| `opentelemetry-api` | 1.45.2 | Apache-2.0 |
| `packaging` | 26.3 | Apache-2.0 OR BSD-2-Clause |
| `pillow` | 13.0.0 | MIT-CMU |
| `pydantic` | 2.14.3 | MIT |
| `pydantic-core` | 2.50.4 | MIT |
| `python-multipart` | 0.0.33 | Apache-2.0 |
| `pyyaml` | 6.0.3 | MIT |
| `scikit-image` | 0.26.0 | Files: * Copyright: 2009-2022 the scikit-image team License: BSD-3-Claus |
| `scikit-learn` | 1.9.1 | BSD-3-Clause |
| `scipy` | 1.17.1 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers. All righ |
| `setuptools` | 78.1.0 | (see licenses/) |
| `starlette` | 1.10.0 | BSD-3-Clause |
| `sympy` | 1.14.0 | BSD |
| `threadpoolctl` | 3.8.0 | BSD-3-Clause |
| `tifffile` | 2026.3.3 | BSD-3-Clause |
| `torch` | 2.15.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3 |
| `torchvision` | 0.29.2+cpu | BSD |
| `typing-extensions` | 4.16.0 | PSF-2.0 |
| `typing-inspection` | 0.4.5 | MIT |
| `uvicorn` | 0.54.0 | BSD-3-Clause |

## Development-only

Used for tests and evaluation; not shipped with a product.

| Package | Version | Licence (as declared in the installed distribution) |
|---|---|---|
| `colorama` | 0.4.6 | (see licenses/) |
| `iniconfig` | 2.3.1 | MIT |
| `pluggy` | 1.7.0 | MIT |
| `pygments` | 2.21.0 | BSD-2-Clause |
| `pytest` | 9.2.0 | MIT |

## Components redistributed inside those packages

The permissive licences above ask one thing of us: keep the copyright notices. These
components ask more, so each gets its own entry.

### FFmpeg, redistributed by the OpenCV wheel - LGPL-2.1

The installed OpenCV wheel ships `cv2/opencv_videoio_ffmpeg500_64.dll`, and its
`cv2/LICENSE-3RD-PARTY.txt` states that FFmpeg is redistributed within all `opencv-python`
packages under the GNU Lesser General Public License, Version 2.1 (February 1999).

How we meet it:

- FFmpeg remains a **separate, dynamically loaded DLL**. Nothing is statically linked into our
  code, so anyone can replace or relink that library - which is the operative LGPL-2.1
  requirement for using a library rather than a derivative work.
- The licence text and this notice accompany the distribution.
- adkit reads still images (PNG/JPEG). It never opens a video stream, so this DLL is loaded
  lazily and is unused in practice. A product build that omits
  `cv2/opencv_videoio_ffmpeg500_64.dll` removes the component altogether; that is the
  recommended packaging, and it is one file to exclude.

The other copyleft block in the same OpenCV file covers **Qt 5**, redistributed only in the
non-headless Linux and macOS packages. This project uses `opencv-python-headless` on Windows,
so Qt is not part of what we distribute.

### FreeType 2.14.3, via Pillow - we rely on the FreeType License (FTL), not the GPL

FreeType is distributed under two mutually exclusive licences: the FTL, or the GNU GPL. This
product uses the **FTL** branch, which is the one suited to software that is not itself under
the GPL. The FTL carries an attribution clause; the credit is reproduced in the form FreeType
recommends:

> Portions of this software are copyright © 2026 The FreeType Project
> (https://freetype.org). All rights reserved.

The year tracks the FreeType release bundled here (reported by Pillow as `2.14.3`); update it
when the Pillow version changes. This concerns the library only - no font files are part of
this repository, and font files carry their own licences.

### zlib-ng 1.3.1, via Pillow - zlib licence

Reported by `PIL.features.version("zlib")` as `1.3.1.zlib-ng`. Permissive; requires retaining
its notice.

### GPL text inside NumPy's and SciPy's licence files - no obligation attaches

Both ship the full text of the GNU General Public License, version 3, **with the GCC Runtime
Library Exception 3.1**. That exception exists precisely so that non-GPL software, including
proprietary software, may link against the runtime in this way, so it imposes nothing on adkit.
The same reading applies to the GPL wording in Pillow's licence file (it is FreeType's
*alternative* licence, which we are not using) and in `typing_extensions` (Python's licence
history table, describing what "GPL-compatible" meant for old Python releases).

`packaging` declares Apache-2.0 **or** BSD-2-Clause and either may be followed. `certifi` is
MPL-2.0, which is copyleft at file level only: we use it unmodified and reproduce its notice.

## Non-code assets

Licences on code are not the only terms in play, and two of these are not settled by this
repository. They are stated here rather than left implicit.

### ImageNet-pretrained ResNet-18 weights (the feature backbone)

The detectors build their backbone with `pretrained=True` by default
(`src/adkit/detect/features.py`), which downloads `resnet18-f37072fd.pth` from
`download.pytorch.org`. The torchvision **code** is BSD-3-Clause. The **checkpoint** is a model
trained on ImageNet, and torchvision's documentation says plainly that models may carry their
own licences or terms derived from the dataset used to train them, and that it is the user's
responsibility to determine whether the intended use is permitted. ImageNet's terms of access
restrict the database to non-commercial research and educational purposes.

**This is an open question for any commercial deployment, and it is ours to answer.** Options,
in the order to evaluate them:

1. obtain a written position from counsel before the weights ship inside a product;
2. retrain or fine-tune the feature extractor on imagery we are entitled to use;
3. move to a backbone whose published weights state a commercial-use licence;
4. set `detectors.*.pretrained: false`, which removes the download entirely. This is supported,
   but every measurement in the README was taken with pretrained weights, so it is a change of
   capability, not a paperwork fix.

### MVTec-AD style bottle benchmark - CC BY-NC-SA 4.0

Used for local validation only. It is not in this repository, is not distributed with the
product, and `.gitignore` excludes it. The licence rules out commercial use and applies
share-alike to the dataset, so product parameters should be set from data we hold rights over.
The benchmark's role is to answer questions our own imagery cannot - a fixed camera, hundreds
of good parts, pixel-level ground truth - and nothing beyond that.

### Product imagery

Images of the parts this system inspects are never committed here. `data/`, `runs/`, `image/`
and `image1/` are gitignored; the repository carries source, tests and evaluation harnesses.

## Where the full licence texts live

Each installed distribution ships its own texts, usually under `<dist>.dist-info/licenses/` or
beside the package:

- OpenCV, including the FFmpeg and Qt blocks: `cv2/LICENSE-3RD-PARTY.txt`
- Pillow, including the FTL text and the zlib-ng notice: `pillow-<version>.dist-info/licenses/LICENSE`
- NumPy and SciPy: `<pkg>-<version>.dist-info/LICENSE.txt`
- Torch: `torch-<version>.dist-info/LICENSE` together with the component notices it lists

## Regenerating

The package table comes from the environment; the sections above it are judgement and should be
re-read by a person, never rewritten by a script.

```bash
python - <<'PY'
import importlib.metadata as im, re
NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
roots = ["numpy", "opencv-python-headless", "scipy", "scikit-image", "scikit-learn",
         "pyyaml", "click", "pillow", "torch", "torchvision", "fastapi", "uvicorn", "pytest"]
seen, stack = set(), roots
while stack:
    req = stack.pop()
    m = NAME.match(req)
    if not m:
        continue
    n = m.group(1).lower().replace("_", "-")
    if n in seen:
        continue
    try:
        d = im.distribution(n)
    except im.PackageNotFoundError:
        continue
    seen.add(n)
    for r in (d.requires or []):
        if "extra ==" in r and 'extra == "standard"' not in r:
            continue
        stack.append(r)
print(len(seen), "distributions")
for n in sorted(seen):
    d = im.distribution(n)
    lic = d.metadata.get("License-Expression") or d.metadata.get("License") or "(see licenses/)"
    print(f"| `{n}` | {d.version} | {' '.join(lic.split())[:72]} |")
PY
```
