# RePath Model

Python-first repository for training, evaluating, and releasing RePath object-detection artifacts.

## Goals
- Keep model R&D separate from mobile app code.
- Prefer Python workflows and data-science tooling.
- Publish semantically versioned model releases for `repath-mobile` consumption.

## Layout
- `src/repath_model/`: Python package code.
- `configs/`: training/eval/release config files.
- `scripts/`: executable workflow scripts.
- `notebooks/`: exploratory notebooks.
- `dist/releases/`: generated release bundles (local, ignored in git).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
# training + benchmark dependencies
pip install -e .[train,eval]
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

## Migrated Python Workflows

The following scripts were migrated from `repath-mobile/ml` into this repo:
- `scripts/fetch_yolov8n_tflite.py`
- `scripts/benchmark_model.py`
- `scripts/seed_annotation_boxes.py`
- `scripts/train_detector_from_annotation.py`
- `scripts/export_candidate_from_retraining.py`

Today these workflows still read/write datasets and artifacts that live in `repath-mobile` (`assets/models`, `ml/artifacts`, `test/benchmarks`).

Run from `repath-model` root:
```bash
# model fetch/export helper
python3 scripts/fetch_yolov8n_tflite.py \
  --out-dir ../repath-mobile/assets/models

# benchmark current model
python3 scripts/benchmark_model.py \
  --model ../repath-mobile/assets/models/yolo-repath.tflite \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest.resolved.json \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/latest-results.json

# seed annotation boxes
python3 scripts/seed_annotation_boxes.py \
  --bundle-root ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --model ../repath-mobile/assets/models/yolo-repath.tflite \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json

# train detector candidate
python3 scripts/train_detector_from_annotation.py \
  --bundle-root ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --candidate-root ../repath-mobile/ml/artifacts/models/candidates \
  --project ../repath-mobile/ml/artifacts/training-runs \
  --nms

# export candidate from retraining manifest
python3 scripts/export_candidate_from_retraining.py \
  --retraining-manifest ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json \
  --base-labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --out-root ../repath-mobile/ml/artifacts/models/candidates \
  --nms
```

## Build A Versioned Release Bundle

Create semver-tagged artifacts and a manifest with SHA256 checksums.

```bash
python3 scripts/build_release.py \
  --version 1.0.0 \
  --model /path/to/yolo-repath.tflite \
  --labels /path/to/yolo-repath.labels.json \
  --source-run-id 20260217-001
```

Outputs:
- `dist/releases/v1.0.0/yolo-repath-v1.0.0.tflite`
- `dist/releases/v1.0.0/yolo-repath-v1.0.0.labels.json`
- `dist/releases/v1.0.0/release-manifest-v1.0.0.json`
- `dist/releases/v1.0.0/SHA256SUMS`

Verify bundle integrity:
```bash
python3 scripts/verify_release.py \
  --manifest dist/releases/v1.0.0/release-manifest-v1.0.0.json
```

## GitHub Releases

Prerequisites:
```bash
gh auth status
```

Build + verify local bundle:
```bash
VERSION=1.0.0

python3 scripts/build_release.py \
  --version "$VERSION" \
  --model /path/to/yolo-repath.tflite \
  --labels /path/to/yolo-repath.labels.json \
  --source-run-id 20260217-001 \
  --notes-file /path/to/release-notes.md

python3 scripts/verify_release.py \
  --manifest "dist/releases/v${VERSION}/release-manifest-v${VERSION}.json"
```

Create/push git tag in `repath-model`:
```bash
git tag "v${VERSION}"
git push origin "v${VERSION}"
```

Create GitHub release and upload artifacts:
```bash
gh release create "v${VERSION}" \
  "dist/releases/v${VERSION}/yolo-repath-v${VERSION}.tflite" \
  "dist/releases/v${VERSION}/yolo-repath-v${VERSION}.labels.json" \
  "dist/releases/v${VERSION}/release-manifest-v${VERSION}.json" \
  "dist/releases/v${VERSION}/SHA256SUMS" \
  --repo krispeterson/repath-model \
  --title "RePath Model v${VERSION}" \
  --notes-file /path/to/release-notes.md
```

Validate release exists:
```bash
gh release view "v${VERSION}" --repo krispeterson/repath-model
```

Smoke test release download and integrity:
```bash
mkdir -p /tmp/repath-model-release-smoke
cd /tmp/repath-model-release-smoke

gh release download "v${VERSION}" \
  --repo krispeterson/repath-model \
  --pattern "yolo-repath-v${VERSION}.tflite" \
  --pattern "yolo-repath-v${VERSION}.labels.json" \
  --pattern "release-manifest-v${VERSION}.json"

python3 /path/to/repath-model/scripts/verify_release.py \
  --manifest "release-manifest-v${VERSION}.json"
```

## Current Migration Status
- Release packaging + verification: in `repath-model`.
- Core Python training/eval scripts: migrated to `repath-model/scripts`.
- Node-based data prep/eval orchestration: still in `repath-mobile/ml`.
- Full data-science refactor (Python-native pipeline + notebooks): next phase.
