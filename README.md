# RePath Model

Python-first repository for training, evaluating, and releasing RePath object-detection artifacts.

## Goals
- Keep model R&D separate from mobile app code.
- Prefer Python workflows and DS tooling.
- Publish semantically versioned model releases for `repath-mobile` consumption.

## Layout
- `src/repath_model/`: Python package code.
- `configs/`: training/eval/release config files.
- `scripts/`: convenience command wrappers.
- `notebooks/`: exploratory notebooks.
- `dist/releases/`: generated release bundles (local).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# training deps if needed:
pip install -e .[train]
```

## DS Helpers

Resolve model/label artifacts from a run directory:
```bash
repath-model-resolve-artifacts --run-dir /path/to/run-or-candidate
```

Build a YOLO line from pixel coordinates:
```bash
repath-model-yolo-box \
  --image-width 1920 --image-height 1080 \
  --class-id 3 --x1 120 --y1 80 --x2 960 --y2 720
```

## Build A Versioned Release Bundle

Creates semver-tagged artifacts and a manifest with SHA256 checksums.

```bash
repath-model-build-release \
  --version 1.0.0 \
  --model /path/to/yolo-repath.tflite \
  --labels /path/to/yolo-repath.labels.json
```

Outputs:
- `dist/releases/v1.0.0/yolo-repath-v1.0.0.tflite`
- `dist/releases/v1.0.0/yolo-repath-v1.0.0.labels.json`
- `dist/releases/v1.0.0/release-manifest-v1.0.0.json`
- `dist/releases/v1.0.0/SHA256SUMS`

Verify bundle integrity:
```bash
repath-model-verify-release --manifest dist/releases/v1.0.0/release-manifest-v1.0.0.json
```

## GitHub Releases

1. Create release bundle locally (above).
2. Push tag `vX.Y.Z`.
3. Attach bundle files to GitHub release (or automate with CI).

`repath-mobile` pulls release assets by version from:
- `https://github.com/krispeterson/repath-model/releases`

## Current Migration Status
- Python release tooling: in this repo.
- Existing training/eval pipeline migration: in progress from `repath-mobile/ml/`.
