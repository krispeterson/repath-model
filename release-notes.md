# RePath Model Release Notes

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
