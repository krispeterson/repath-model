# RePath Model Release Notes

## Unreleased Working Changes
- 2026-02-17: Migrated retraining positive lifecycle scripts to Python modules in `repath-model/src/repath_model/training` (`expand_retraining_positives_from_kaggle`, `materialize_retraining_positives`, `fill_missing_positive_boxes`, `promote_candidate_model`) with thin wrappers in `repath-model/scripts/training`.
- 2026-02-17: Updated `repath-mobile/ml/training` wrappers so those four commands now delegate to `repath-model` Python scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated annotation bundle build and validation workflows to Python modules in `repath-model/src/repath_model/training` with thin wrappers in `repath-model/scripts/training`.
- 2026-02-17: Updated `repath-mobile/ml/training` wrappers so `build-annotation-bundle` and `validate-annotation-bundle` now delegate to `repath-model` Python scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated retraining helper scripts (`build retraining manifest`, `build retraining image inventory`, `build retraining source issues`) to Python modules in `repath-model/src/repath_model/training` with thin wrappers in `repath-model/scripts/training`.
- 2026-02-17: Updated `repath-mobile/ml/training` wrappers so those three commands now delegate to `repath-model` Python scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated remaining `ml/data` Kaggle/bootstrap and online suggestion scripts to Python in `repath-model/scripts/data` (`bootstrap dataset`, `suggest from kaggle`, `suggest online`, `bulk online suggest`, `suggest negative online`).
- 2026-02-17: Updated `repath-mobile/ml/data` wrappers so those commands now delegate to `repath-model` Python scripts via `scripts/run-python.js`.
- 2026-02-17: Updated `notebooks/release_workflow.ipynb` default `VERSION` value to `0.1.0`.
- 2026-02-17: Migrated remaining local benchmark eval utility scripts from `repath-mobile/ml/eval` to Python in `repath-model/scripts/evaluation` (`build manifest`, `build retraining queue`, `dedupe manifest`, `export unresolved rows`, `seed depth variants`, `seed negative entries`).
- 2026-02-17: Updated `repath-mobile/ml/eval` wrappers so those commands now delegate to `repath-model` Python scripts via `scripts/run-python.js`.
- 2026-02-17: Updated notebooks to resolve data roots via env overrides and prefer local `repath-model` paths before falling back to `repath-mobile` paths.
- 2026-02-17: Updated `notebooks/release_workflow.ipynb` to auto-generate `SOURCE_RUN_ID` as `YYYYMMDD-HHMMSS` (with optional `SOURCE_RUN_ID` env override).
- 2026-02-17: Refactored Python workflow scripts into domain subdirectories under `scripts/` (`annotation`, `training`, `evaluation`, `data`, `release`, `utilities`) for findability.
- 2026-02-17: Updated `repath-mobile` command wiring and JS forwarders to use new `repath-model/scripts/<domain>/...` script paths.
- 2026-02-17: Fixed `notebooks/release_workflow.ipynb` and `notebooks/retraining_workflow.ipynb` cell sources to use real line breaks instead of visible `\\n` sequences.
- 2026-02-17: Added `AGENTS.md` with a requirement for agents to keep this file updated when code changes.
- 2026-02-17: Migrated benchmark error analysis and benchmark comparison workflows from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring to call the new Python analysis/comparison scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated benchmark progress sync, benchmark coverage reporting, and resolved benchmark manifest build workflows from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring and pipeline entrypoints to call the new Python progress/coverage/resolved-manifest scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated benchmark dataset audit and supported holdout manifest workflows from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring and benchmark pipeline to call new Python audit/holdout scripts via `scripts/run-python.js`.
- 2026-02-17: Added `notebooks/release_workflow.ipynb` and `notebooks/retraining_workflow.ipynb` as guided, command-oriented runbooks.
- 2026-02-17: Migrated benchmark planning and labeling queue scripts (`plan priority`, `coverage expansion`, `batch builder`, and `completion template`) from `repath-mobile/ml/eval` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring so queue/planning commands now invoke `repath-model` Python scripts via `scripts/run-python.js`.
- 2026-02-17: Migrated local benchmark data curation scripts (`build taxonomy`, `sync/merge/dedupe/normalize labeled CSV`, `merge retraining queue`, and `fill retraining negatives`) from `repath-mobile/ml/data` JavaScript scripts to Python scripts in `repath-model/scripts`.
- 2026-02-17: Updated `repath-mobile` script wiring so those data curation commands now invoke `repath-model` Python scripts via `scripts/run-python.js`.

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
- [ ] Bundle built with `python3 scripts/release/build_release.py ...`
- [ ] Bundle verified with `python3 scripts/release/verify_release.py ...`
- [ ] GitHub release created with required artifacts
- [ ] Release pull smoke test completed in `repath-mobile`
