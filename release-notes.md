# RePath Model Release Notes

## Unreleased Working Changes
- 2026-02-17: Added `AGENTS.md` with a requirement for agents to keep this file updated when code changes.
- 2026-02-17: Migrated benchmark error analysis and benchmark comparison workflows from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring to call the new Python analysis/comparison scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated benchmark progress sync, benchmark coverage reporting, and resolved benchmark manifest build workflows from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring and pipeline entrypoints to call the new Python progress/coverage/resolved-manifest scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated benchmark dataset audit and supported holdout manifest workflows from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring and benchmark pipeline to call new Python audit/holdout scripts via `scripts/run-python.js`.
- 2026-02-17: Added `notebooks/release_workflow.ipynb` and `notebooks/retraining_workflow.ipynb` as guided, command-oriented runbooks.

## Version
- `vX.Y.Z`

## Summary
- Short description of what changed in this model release.

## Inputs
- Training/annotation bundle: `../repath-mobile/ml/artifacts/retraining/annotation-bundle/<run-id>`
- Candidate run id: `<run-id>`
- Base model: `<model-name-or-path>`

## Benchmark Snapshot
- Manifest: `../repath-mobile/test/benchmarks/municipal-benchmark-manifest.resolved.json`
- Micro precision: `<value>`
- Micro recall: `<value>`
- Any-hit rate: `<value>`
- Negative clean rate: `<value>`

## Notable Improvements
- Label or scenario improvements.

## Known Limitations
- Known misclassifications or low-confidence classes.

## Mobile Compatibility
- Model filename: `yolo-repath-vX.Y.Z.tflite`
- Labels filename: `yolo-repath-vX.Y.Z.labels.json`
- Target app integration: `repath-mobile`

## Validation Checklist
- [ ] Bundle built with `python3 scripts/build_release.py ...`
- [ ] Bundle verified with `python3 scripts/verify_release.py ...`
- [ ] GitHub release created with required artifacts
- [ ] Release pull smoke test completed in `repath-mobile`
