#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge benchmark retraining queue CSV into benchmark-labeled CSV by row name.",
        usage=(
            "python3 scripts/merge_retraining_queue.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--queue test/benchmarks/benchmark-retraining-queue.csv] "
            "[--out test/benchmarks/benchmark-labeled.csv] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input labeled CSV path.",
    )
    parser.add_argument(
        "--queue",
        default=str(Path("test") / "benchmarks" / "benchmark-retraining-queue.csv"),
        help="Retraining queue CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Output labeled CSV path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write output file.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def read_rows(file_path: Path) -> list[dict]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    if len(lines) < 2:
        return []

    rows = []
    for line in lines[1:]:
        cols = next(csv.reader([line]))
        rows.append(
            {
                "name": str(cols[0] if len(cols) > 0 else "").strip(),
                "url": str(cols[1] if len(cols) > 1 else "").strip(),
                "item_id": str(cols[2] if len(cols) > 2 else "").strip(),
                "canonical_label": str(cols[3] if len(cols) > 3 else "").strip(),
                "source": str(cols[4] if len(cols) > 4 else "").strip(),
                "notes": str(cols[5] if len(cols) > 5 else "").strip(),
            }
        )
    return rows


def write_rows(file_path: Path, rows: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    input_path = Path(args.input).resolve()
    queue_path = Path(args.queue).resolve()
    out_path = Path(args.out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    if not queue_path.exists():
        raise SystemExit(f"Queue file not found: {queue_path}")

    base_rows = read_rows(input_path)
    queue_rows = read_rows(queue_path)
    seen = {row["name"] for row in base_rows if row.get("name")}

    additions = [row for row in queue_rows if row.get("name") and row["name"] not in seen]
    merged = sorted([*base_rows, *additions], key=lambda row: str(row.get("name") or ""))

    if not args.dry_run:
        write_rows(out_path, merged)

    print("Retraining queue merge summary")
    print(
        json.dumps(
            {
                "base_rows": len(base_rows),
                "queue_rows": len(queue_rows),
                "rows_added": len(additions),
                "rows_total": len(merged),
                "output": rel_or_abs(out_path, cwd),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
