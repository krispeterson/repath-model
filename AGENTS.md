# Agents

This repository is Python-first and focused on model development, evaluation, and release packaging.

## Guardrails
- Keep tooling and workflows in Python where practical.
- Prefer small, readable scripts in `scripts/` over adding heavy dependencies.
- Keep package logic in `src/repath_model/` and script entrypoints in `scripts/`.

## Release Notes Requirement
- When an agent changes code in this repository, update `release-notes.md` in the same working pass.
- Add or revise notes for user-visible behavior changes, migration changes, tooling changes, and release process updates.
- Before creating a tagged release, replace placeholder values in `release-notes.md` with real version/run/benchmark details.

## Common commands
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train,eval]
python3 scripts/build_release.py --help
python3 scripts/verify_release.py --help
```
