#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark data/build/eval pipeline.")
    parser.add_argument("--skip-kaggle", action="store_true", help="Skip Kaggle suggestions step.")
    parser.add_argument("--skip-online", action="store_true", help="Skip online suggestions step.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark execution step.")
    parser.add_argument("--strict-network", action="store_true", help="Fail on network-dependent step failures.")
    parser.add_argument("--online-limit", type=int, default=40, help="Online suggestion limit.")
    parser.add_argument("--kaggle-dir", default="", help="Kaggle image dataset directory.")
    return parser.parse_args()


def run_step(label: str, cmd: list[str], allow_failure: bool = False) -> None:
    print(f"\n== {label} ==")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        if allow_failure:
            print(f"{label} failed with exit code {result.returncode}; continuing.")
            return
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    args = parse_args()
    online_limit = args.online_limit if args.online_limit and args.online_limit > 0 else 40
    allow_network_failure = not args.strict_network
    benchmark_script = Path(__file__).resolve().parent / "benchmark_model.py"

    if not args.skip_kaggle:
        kaggle_cmd = [
            "node",
            str(Path("ml") / "data" / "suggest-benchmark-from-kaggle.js"),
            "--merge-into",
            str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        ]
        if args.kaggle_dir:
            kaggle_cmd.extend(["--kaggle-dir", args.kaggle_dir])
        run_step("Suggest Kaggle", kaggle_cmd, allow_failure=allow_network_failure)

    if not args.skip_online:
        run_step(
            "Suggest Online",
            [
                "node",
                str(Path("ml") / "data" / "suggest-benchmark-online.js"),
                "--merge-into",
                str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
                "--limit",
                str(online_limit),
            ],
            allow_failure=allow_network_failure,
        )

    run_step("Normalize URLs", ["node", str(Path("ml") / "data" / "normalize-benchmark-labeled-urls.js")])
    run_step(
        "Sync Progress",
        [
            "node",
            str(Path("ml") / "eval" / "sync-benchmark-progress.js"),
            "--completed",
            str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        ],
    )
    run_step("Build Resolved Manifest", ["node", str(Path("ml") / "eval" / "build-resolved-benchmark-manifest.js")])
    run_step("Coverage (Canonical)", ["node", str(Path("ml") / "eval" / "check-benchmark-coverage.js")])
    run_step(
        "Coverage (Resolved)",
        [
            "node",
            str(Path("ml") / "eval" / "check-benchmark-coverage.js"),
            "--manifest",
            str(Path("test") / "benchmarks" / "municipal-benchmark-manifest.resolved.json"),
            "--out",
            str(Path("test") / "benchmarks" / "benchmark-coverage-report.resolved.json"),
        ],
    )
    run_step(
        "Audit (Resolved)",
        [
            "node",
            str(Path("ml") / "eval" / "audit-benchmark-dataset.js"),
            "--manifest",
            str(Path("test") / "benchmarks" / "municipal-benchmark-manifest.resolved.json"),
            "--out",
            str(Path("test") / "benchmarks" / "benchmark-dataset-audit.resolved.json"),
        ],
    )

    if not args.skip_benchmark:
        run_step(
            "Benchmark (Resolved)",
            [
                sys.executable,
                str(benchmark_script),
                "--manifest",
                str(Path("test") / "benchmarks" / "municipal-benchmark-manifest.resolved.json"),
                "--out",
                "/tmp/repath-benchmark-resolved-results.json",
            ],
        )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
