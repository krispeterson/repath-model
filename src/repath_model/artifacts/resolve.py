#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

MODEL_CANDIDATES = ["yolo-repath.tflite", "yolov8.tflite"]
LABEL_CANDIDATES = ["yolo-repath.labels.json", "yolov8.labels.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve model + labels files from a run/candidate directory.")
    parser.add_argument("--run-dir", required=True, help="Directory containing model artifacts")
    return parser.parse_args()


def first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    model = first_existing(run_dir, MODEL_CANDIDATES)
    labels = first_existing(run_dir, LABEL_CANDIDATES)

    if not model:
        raise SystemExit(f"No model file found in {run_dir} (checked: {', '.join(MODEL_CANDIDATES)})")
    if not labels:
        raise SystemExit(f"No labels file found in {run_dir} (checked: {', '.join(LABEL_CANDIDATES)})")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "model": str(model),
                "labels": str(labels),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
