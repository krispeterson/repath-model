#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple online suggestion passes.",
        usage=(
            "python3 scripts/data/suggest_benchmark_online_bulk.py "
            "[--passes 4] [--limit 30] [--start-offset 0] [--timeout-ms 15000] [--max-retries 3] [--disable-adaptive]"
        ),
    )
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--disable-adaptive", action="store_true")
    args = parser.parse_args()

    args.passes = args.passes if args.passes and args.passes > 0 else 4
    args.limit = args.limit if args.limit and args.limit > 0 else 30
    args.start_offset = args.start_offset if args.start_offset and args.start_offset >= 0 else 0
    args.timeout_ms = args.timeout_ms if args.timeout_ms and args.timeout_ms >= 1000 else 15000
    args.max_retries = args.max_retries if args.max_retries and args.max_retries >= 1 else 3
    return args


def run_pass(offset: int, args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parent / "suggest_benchmark_online.py"
    cmd = [
        sys.executable,
        str(script),
        "--merge-into",
        "test/benchmarks/benchmark-labeled.csv",
        "--out",
        "test/benchmarks/benchmark-labeled.online.csv",
        "--offset",
        str(offset),
        "--limit",
        str(args.limit),
        "--timeout-ms",
        str(args.timeout_ms),
        "--max-retries",
        str(args.max_retries),
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"online suggestion pass failed at offset={offset}")


def main() -> None:
    args = parse_args()
    adaptive = not args.disable_adaptive

    offsets = []
    for index in range(args.passes):
        if adaptive:
            offsets.append(args.start_offset)
        else:
            offsets.append(args.start_offset + index * args.limit)

    for idx, offset in enumerate(offsets, start=1):
        print(f"\n== Online bulk pass {idx}/{len(offsets)} (offset={offset}, limit={args.limit}) ==")
        run_pass(offset, args)

    print("\nOnline bulk suggestion complete.")
    print(
        json.dumps(
            {
                "passes": args.passes,
                "limit": args.limit,
                "offsets": offsets,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
