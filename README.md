# RePath Model

Python-first repository for training, evaluating, and releasing RePath object-detection artifacts.

## Goals
- Keep model R&D separate from mobile app code.
- Prefer Python workflows and data-science tooling.
- Publish semantically versioned model releases for `repath-mobile` consumption.

## Layout
- `src/repath_model/`: Python package code.
- `configs/`: training/eval/release config files.
- `scripts/`: executable workflow scripts, organized by domain:
  - `scripts/annotation/`: annotation utilities and box seeding.
  - `scripts/training/`: model fetch/train/export flows.
  - `scripts/evaluation/`: benchmark, audit, and planning workflows.
  - `scripts/data/`: taxonomy and benchmark CSV curation tools.
  - `scripts/release/`: release build/verify/publish scripts.
  - `scripts/utilities/`: standalone helper utilities.
- `notebooks/`: exploratory notebooks.
- `release-notes.md`: template notes file for GitHub releases.
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
- `scripts/training/fetch_yolov8n_tflite.py`
- `scripts/evaluation/benchmark_model.py`
- `scripts/annotation/seed_annotation_boxes.py`
- `scripts/training/train_detector_from_annotation.py`
- `scripts/training/export_candidate_from_retraining.py`
- `scripts/training/build_retraining_manifest.py`
- `scripts/training/build_retraining_image_inventory.py`
- `scripts/training/build_retraining_source_issues.py`
- `scripts/evaluation/benchmark_candidate_model.py`
- `scripts/evaluation/run_benchmark_pipeline.py`
- `scripts/evaluation/analyze_benchmark_results.py`
- `scripts/evaluation/compare_benchmark_results.py`
- `scripts/evaluation/sync_benchmark_progress.py`
- `scripts/evaluation/check_benchmark_coverage.py`
- `scripts/evaluation/build_resolved_benchmark_manifest.py`
- `scripts/evaluation/audit_benchmark_dataset.py`
- `scripts/evaluation/build_supported_holdout_manifest.py`
- `scripts/evaluation/plan_benchmark_priority.py`
- `scripts/evaluation/plan_benchmark_coverage_expansion.py`
- `scripts/evaluation/build_benchmark_batches.py`
- `scripts/evaluation/build_benchmark_completion_template.py`
- `scripts/evaluation/build_benchmark_manifest.py`
- `scripts/evaluation/build_retraining_queue.py`
- `scripts/evaluation/dedupe_benchmark_manifest.py`
- `scripts/evaluation/export_unresolved_benchmark_rows.py`
- `scripts/evaluation/seed_benchmark_depth_variants.py`
- `scripts/evaluation/seed_negative_benchmark_entries.py`
- `scripts/data/build_taxonomy.py`
- `scripts/data/sync_labeled_from_manifest.py`
- `scripts/data/dedupe_benchmark_labeled.py`
- `scripts/data/normalize_benchmark_labeled_urls.py`
- `scripts/data/merge_coverage_expansion_template.py`
- `scripts/data/merge_retraining_queue.py`
- `scripts/data/fill_retraining_negatives.py`
- `scripts/data/bootstrap_kaggle_dataset.py`
- `scripts/data/suggest_benchmark_from_kaggle.py`
- `scripts/data/suggest_benchmark_online.py`
- `scripts/data/suggest_benchmark_online_bulk.py`
- `scripts/data/suggest_negative_online.py`

Today these workflows still read/write datasets and artifacts that live in `repath-mobile` (`assets/models`, `ml/artifacts`, `test/benchmarks`).

Run from `repath-model` root:
```bash
# model fetch/export helper
python3 scripts/training/fetch_yolov8n_tflite.py \
  --out-dir ../repath-mobile/assets/models

# benchmark current model
python3 scripts/evaluation/benchmark_model.py \
  --model ../repath-mobile/assets/models/yolo-repath.tflite \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest.resolved.json \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/latest-results.json

# seed annotation boxes
python3 scripts/annotation/seed_annotation_boxes.py \
  --bundle-root ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --model ../repath-mobile/assets/models/yolo-repath.tflite \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json

# train detector candidate
python3 scripts/training/train_detector_from_annotation.py \
  --bundle-root ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --candidate-root ../repath-mobile/ml/artifacts/models/candidates \
  --project ../repath-mobile/ml/artifacts/training-runs \
  --nms

# export candidate from retraining manifest
python3 scripts/training/export_candidate_from_retraining.py \
  --retraining-manifest ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json \
  --base-labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --out-root ../repath-mobile/ml/artifacts/models/candidates \
  --nms

# build retraining helper artifacts from labeled CSV + issue seed
python3 scripts/training/build_retraining_manifest.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json
python3 scripts/training/build_retraining_image_inventory.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/retraining-positive-image-inventory.json \
  --local-prefix test/benchmarks/images/retraining-positives/
python3 scripts/training/build_retraining_source_issues.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --seed ../repath-mobile/test/benchmarks/retraining-positive-source-issues.seed.json \
  --out ../repath-mobile/test/benchmarks/retraining-positive-source-issues.json

# analyze benchmark errors into JSON + priority CSV
python3 scripts/evaluation/analyze_benchmark_results.py \
  --input ../repath-mobile/test/benchmarks/latest-results.json \
  --out ../repath-mobile/test/benchmarks/benchmark-error-analysis.json \
  --template-out ../repath-mobile/test/benchmarks/benchmark-retraining-priority.csv

# compare baseline and candidate benchmark runs
python3 scripts/evaluation/compare_benchmark_results.py \
  --baseline ../repath-mobile/test/benchmarks/latest-results.json \
  --candidate ../repath-mobile/test/benchmarks/latest-results.candidate.json \
  --out ../repath-mobile/test/benchmarks/latest-results.compare.json

# sync labeled progress into benchmark manifest statuses
python3 scripts/evaluation/sync_benchmark_progress.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --completed ../repath-mobile/test/benchmarks/benchmark-labeled.csv

# evaluate taxonomy coverage across benchmark manifest labels
python3 scripts/evaluation/check_benchmark_coverage.py \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json

# build supported holdout manifest for model-supported labels
python3 scripts/evaluation/build_supported_holdout_manifest.py \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --input-csv ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --retraining-manifest ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json \
  --cache-dir ../repath-mobile/test/benchmarks/images/supported-holdout \
  --out ../repath-mobile/test/benchmarks/benchmark-manifest.supported-holdout.json

# materialize a resolved benchmark manifest with local/cache image paths
python3 scripts/evaluation/build_resolved_benchmark_manifest.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --completed ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/municipal-benchmark-manifest.resolved.json

# audit benchmark quality and class-balance signals
python3 scripts/evaluation/audit_benchmark_dataset.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest.resolved.json \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --out ../repath-mobile/test/benchmarks/benchmark-dataset-audit.resolved.json

# plan high-value benchmark labeling queue
python3 scripts/evaluation/plan_benchmark_priority.py \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --out ../repath-mobile/test/benchmarks/benchmark-priority-report.json

# plan coverage expansion rows and template
python3 scripts/evaluation/plan_benchmark_coverage_expansion.py \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --out ../repath-mobile/test/benchmarks/benchmark-coverage-expansion-report.json \
  --template-out ../repath-mobile/test/benchmarks/benchmark-coverage-expansion-template.csv

# build labeling batches + completion template
python3 scripts/evaluation/build_benchmark_batches.py \
  --priority ../repath-mobile/test/benchmarks/benchmark-priority-report.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --out-dir ../repath-mobile/test/benchmarks
python3 scripts/evaluation/build_benchmark_completion_template.py \
  --batches ../repath-mobile/test/benchmarks/benchmark-labeling-batches.json \
  --out ../repath-mobile/test/benchmarks/benchmark-completion-template.csv
python3 scripts/evaluation/build_benchmark_manifest.py \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --seed ../repath-mobile/test/benchmarks/benchmark-manifest.seed.json \
  --out ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json
python3 scripts/evaluation/build_retraining_queue.py \
  --priority-csv ../repath-mobile/test/benchmarks/benchmark-retraining-priority.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-retraining-queue.csv
python3 scripts/evaluation/dedupe_benchmark_manifest.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --report ../repath-mobile/test/benchmarks/benchmark-dedupe-report.json
python3 scripts/evaluation/export_unresolved_benchmark_rows.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-unresolved.csv
python3 scripts/evaluation/seed_benchmark_depth_variants.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --target-ready 3 --max-new 200
python3 scripts/evaluation/seed_negative_benchmark_entries.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --count 20

# local benchmark CSV curation helpers
python3 scripts/data/sync_labeled_from_manifest.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.csv
python3 scripts/data/merge_coverage_expansion_template.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --template ../repath-mobile/test/benchmarks/benchmark-coverage-expansion-template.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.csv
python3 scripts/data/dedupe_benchmark_labeled.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.csv
python3 scripts/data/normalize_benchmark_labeled_urls.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.csv
python3 scripts/data/bootstrap_kaggle_dataset.py \
  --target ../repath-mobile/ml/artifacts/datasets/kaggle-household-waste/images/images \
  --mode symlink
python3 scripts/data/suggest_benchmark_from_kaggle.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.kaggle.csv \
  --merge-into ../repath-mobile/test/benchmarks/benchmark-labeled.csv
python3 scripts/data/suggest_benchmark_online.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.online.csv \
  --merge-into ../repath-mobile/test/benchmarks/benchmark-labeled.csv
python3 scripts/data/suggest_benchmark_online_bulk.py \
  --passes 4 --limit 30 --start-offset 0
python3 scripts/data/suggest_negative_online.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.negatives.csv \
  --merge-into ../repath-mobile/test/benchmarks/benchmark-labeled.csv
```

## Notebooks
- `notebooks/release_workflow.ipynb`: guided release bundle + GitHub release publishing flow.
- `notebooks/retraining_workflow.ipynb`: guided benchmark prep, retraining, and candidate evaluation flow.

## Build A Versioned Release Bundle

Create semver-tagged artifacts and a manifest with SHA256 checksums.

```bash
python3 scripts/release/build_release.py \
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
python3 scripts/release/verify_release.py \
  --manifest dist/releases/v1.0.0/release-manifest-v1.0.0.json
```

## GitHub Releases

Prerequisites:
```bash
gh auth status
```
Before building a release, update the checked-in `release-notes.md` template for the target version.

Build + verify local bundle:
```bash
VERSION=1.0.0

python3 scripts/release/build_release.py \
  --version "$VERSION" \
  --model /path/to/yolo-repath.tflite \
  --labels /path/to/yolo-repath.labels.json \
  --source-run-id 20260217-001 \
  --notes-file release-notes.md

python3 scripts/release/verify_release.py \
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
  --notes-file release-notes.md
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

python3 /path/to/repath-model/scripts/release/verify_release.py \
  --manifest "release-manifest-v${VERSION}.json"
```

## Current Migration Status
- Release packaging + verification: in `repath-model`.
- Core Python training/eval scripts: migrated to `repath-model/scripts`.
- Benchmark orchestration entrypoints: migrated to Python in `repath-model/scripts`.
- Benchmark analysis/comparison scripts: migrated to Python in `repath-model/scripts`.
- Benchmark progress/coverage/resolved-manifest scripts: migrated to Python in `repath-model/scripts`.
- Benchmark audit + supported-holdout scripts: migrated to Python in `repath-model/scripts`.
- Benchmark planning and labeling-queue scripts: migrated to Python in `repath-model/scripts`.
- Local benchmark CSV curation scripts: migrated to Python in `repath-model/scripts`.
- Benchmark scaffold/seed/dedupe/export utility scripts: migrated to Python in `repath-model/scripts`.
- Kaggle bootstrap and online suggestion scripts: migrated to Python in `repath-model/scripts`.
- Retraining manifest/image-inventory/source-issues scripts: migrated to Python in `repath-model/scripts`.
- Annotation bundle generation/validation, positive expansion helpers, and candidate promotion scripts: still in `repath-mobile/ml/training`.
