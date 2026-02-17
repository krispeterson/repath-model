# Training And Benchmark Playbook

This playbook was migrated from `repath-mobile/ml/README.md`.

Use this workflow when you need to improve model quality, expand label coverage, generate retraining data, or publish a new model release.

## Scope

Current data/artifact roots are still in `repath-mobile`:
- Benchmarks: `../repath-mobile/test/benchmarks`
- Runtime model assets: `../repath-mobile/assets/models`
- Training artifacts: `../repath-mobile/ml/artifacts`

## When to use this

- Detection quality regresses after a model change.
- A municipality taxonomy changes and model label coverage must be refreshed.
- You need targeted retraining for low-performing classes.
- You need benchmark-backed before/after model comparisons.

## Core lifecycle

1. Build or refresh benchmark scaffolding.
2. Ingest/curate benchmark data.
3. Generate retraining manifest and annotation bundle.
4. Validate annotations and train/export candidate.
5. Benchmark candidate and compare to baseline.
6. Promote candidate and build semver release bundle.

## Core commands

Run from `repath-model` root.

### 1) Benchmark scaffold and coverage
```bash
python3 scripts/data/build_taxonomy.py \
  --pack ../repath-core/packages/packs/municipal_curbside_v1/pack.json \
  --out ../repath-mobile/assets/models/municipal-taxonomy-v1.json

python3 scripts/evaluation/build_benchmark_manifest.py \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --seed ../repath-mobile/test/benchmarks/benchmark-manifest.seed.json \
  --out ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json

python3 scripts/evaluation/check_benchmark_coverage.py \
  --taxonomy ../repath-mobile/assets/models/municipal-taxonomy-v1.json \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json
```

### 2) Suggest/curate benchmark rows
```bash
python3 scripts/data/suggest_benchmark_from_kaggle.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --merge-into ../repath-mobile/test/benchmarks/benchmark-labeled.csv

python3 scripts/data/suggest_benchmark_online.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --merge-into ../repath-mobile/test/benchmarks/benchmark-labeled.csv

python3 scripts/data/normalize_benchmark_labeled_urls.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.csv

python3 scripts/data/dedupe_benchmark_labeled.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/test/benchmarks/benchmark-labeled.csv
```

### 3) Build retraining bundle
```bash
python3 scripts/training/build_retraining_manifest.py \
  --input ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --out ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json

python3 scripts/training/build_annotation_bundle.py \
  --manifest ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json \
  --out-dir ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --no-download

python3 scripts/training/validate_annotation_bundle.py \
  --bundle-root ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --strict
```

### 4) Train and export candidate
```bash
python3 scripts/training/train_detector_from_annotation.py \
  --bundle-root ../repath-mobile/ml/artifacts/retraining/annotation-bundle \
  --candidate-root ../repath-mobile/ml/artifacts/models/candidates \
  --project ../repath-mobile/ml/artifacts/training-runs \
  --nms
```

### 5) Benchmark candidate and compare
```bash
python3 scripts/evaluation/build_supported_holdout_manifest.py \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --input-csv ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --retraining-manifest ../repath-mobile/ml/artifacts/retraining/retraining-manifest.json \
  --cache-dir ../repath-mobile/test/benchmarks/images/supported-holdout \
  --out ../repath-mobile/test/benchmarks/benchmark-manifest.supported-holdout.json

python3 scripts/evaluation/build_resolved_benchmark_manifest.py \
  --manifest ../repath-mobile/test/benchmarks/municipal-benchmark-manifest-v2.json \
  --completed ../repath-mobile/test/benchmarks/benchmark-labeled.csv \
  --cache-dir ../repath-mobile/test/benchmarks/images \
  --out ../repath-mobile/test/benchmarks/municipal-benchmark-manifest.resolved.json \
  --no-download

python3 scripts/evaluation/benchmark_candidate_model.py --supported-only
python3 scripts/evaluation/compare_benchmark_results.py \
  --baseline ../repath-mobile/test/benchmarks/latest-results.json \
  --candidate ../repath-mobile/test/benchmarks/latest-results.candidate.json \
  --out ../repath-mobile/test/benchmarks/latest-results.compare.json
```

### 6) Promote and release
```bash
python3 scripts/training/promote_candidate_model.py \
  --candidates-root ../repath-mobile/ml/artifacts/models/candidates \
  --assets-dir ../repath-mobile/assets/models \
  --prefix yolo-repath \
  --write-metadata \
  --metadata-path ../repath-mobile/ml/artifacts/models/active-model.json

python3 scripts/release/build_release.py \
  --version 0.1.0 \
  --model ../repath-mobile/assets/models/yolo-repath.tflite \
  --labels ../repath-mobile/assets/models/yolo-repath.labels.json \
  --source-run-id 20260217-000001 \
  --notes-file release-notes.md

python3 scripts/release/verify_release.py \
  --manifest dist/releases/v0.1.0/release-manifest-v0.1.0.json
```

## Manual checkpoints

- Curate `benchmark-labeled.csv` quality before retraining.
- Confirm negatives remain negative after normalization/suggestions.
- Validate annotation boxes before real training.
- Review candidate-vs-baseline comparison on overlap metrics.
- Update `release-notes.md` before creating GitHub releases.

## Related runbooks

- `README.md` for full script list.
- `notebooks/retraining_workflow.ipynb` for step-by-step interactive workflow.
- `notebooks/release_workflow.ipynb` for release creation and publishing.
