#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark latest/selected candidate model and analyze results.")
    parser.add_argument("--candidate-dir", default="", help="Path to candidate directory.")
    parser.add_argument(
        "--candidates-root",
        default=str(Path("ml") / "artifacts" / "models" / "candidates"),
        help="Root containing candidate run directories.",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest.resolved.json"),
        help="Benchmark manifest path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "latest-results.candidate.json"),
        help="Output benchmark results path.",
    )
    parser.add_argument("--supported-only", action="store_true", help="Evaluate supported labels only.")
    return parser.parse_args()


def resolve_latest_candidate_dir(root_dir: Path) -> Path | None:
    if not root_dir.exists() or not root_dir.is_dir():
        return None
    dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def resolve_candidate_artifact(candidate_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        path = candidate_dir / name
        if path.exists():
            return path
    return None


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n== {label} ==")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed")


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    benchmark_script = script_dir / "benchmark_model.py"

    candidate_dir = Path(args.candidate_dir).resolve() if args.candidate_dir else resolve_latest_candidate_dir(Path(args.candidates_root).resolve())

    if not candidate_dir or not candidate_dir.exists():
        raise SystemExit("Candidate directory not found. Run export candidate model first.")

    model_path = resolve_candidate_artifact(candidate_dir, ["yolo-repath.tflite", "yolov8.tflite"])
    labels_path = resolve_candidate_artifact(candidate_dir, ["yolo-repath.labels.json", "yolov8.labels.json"])
    out_path = Path(args.out).resolve()
    analysis_out = Path(str(out_path).replace(".json", ".analysis.json"))
    priority_csv_out = Path(str(out_path).replace(".json", ".priority.csv"))

    if not model_path:
        raise SystemExit(f"Candidate model not found in {candidate_dir} (expected yolo-repath.tflite or yolov8.tflite).")
    if not labels_path:
        raise SystemExit(
            f"Candidate labels not found in {candidate_dir} (expected yolo-repath.labels.json or yolov8.labels.json)."
        )

    benchmark_cmd = [
        sys.executable,
        str(benchmark_script),
        "--manifest",
        args.manifest,
        "--model",
        rel_or_abs(model_path, cwd),
        "--labels",
        rel_or_abs(labels_path, cwd),
        "--out",
        rel_or_abs(out_path, cwd),
    ]
    if args.supported_only:
        benchmark_cmd.append("--supported-only")

    run_step("Benchmark Candidate", benchmark_cmd)
    run_step(
        "Analyze Candidate Results",
        [
            "node",
            str(Path("ml") / "eval" / "analyze-benchmark-results.js"),
            "--input",
            rel_or_abs(out_path, cwd),
            "--out",
            rel_or_abs(analysis_out, cwd),
            "--template-out",
            rel_or_abs(priority_csv_out, cwd),
        ],
    )

    print("\nCandidate benchmark complete")
    print(
        json.dumps(
            {
                "candidate_dir": rel_or_abs(candidate_dir, cwd),
                "model": rel_or_abs(model_path, cwd),
                "labels": rel_or_abs(labels_path, cwd),
                "results": rel_or_abs(out_path, cwd),
                "analysis": rel_or_abs(analysis_out, cwd),
                "priority_csv": rel_or_abs(priority_csv_out, cwd),
                "supported_only": args.supported_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
