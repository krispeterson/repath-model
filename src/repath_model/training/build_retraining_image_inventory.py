#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retraining positive image inventory from benchmark-labeled CSV.",
        usage=(
            "python3 scripts/training/build_retraining_image_inventory.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out test/benchmarks/retraining-positive-image-inventory.json] "
            "[--local-prefix test/benchmarks/images/retraining-positives/]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input benchmark-labeled CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "retraining-positive-image-inventory.json"),
        help="Output inventory JSON path.",
    )
    parser.add_argument(
        "--local-prefix",
        default="test/benchmarks/images/retraining-positives/",
        help="Only include rows whose URL starts with this prefix.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = {column: str((row or {}).get(column, "")).strip() for column in CSV_COLUMNS}
            rows.append(normalized)
        return rows


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    local_prefix = str(args.local_prefix or "").strip()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = []
    for row in read_rows(input_path):
        name = str(row.get("name") or "")
        url = str(row.get("url") or "")
        if not name.startswith("retrain_positive_"):
            continue
        if not local_prefix or not url.startswith(local_prefix):
            continue
        rows.append(row)

    rows.sort(key=lambda row: str(row.get("name") or "").lower())

    by_label: dict[str, int] = {}
    for row in rows:
        key = str(row.get("canonical_label") or "") or "unknown"
        by_label[key] = by_label.get(key, 0) + 1

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": rel_or_abs(input_path, cwd),
        "local_prefix": local_prefix,
        "count": len(rows),
        "labels": by_label,
        "rows": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("Retraining image inventory generated")
    print(
        json.dumps(
            {
                "source_csv": out["source_csv"],
                "output": rel_or_abs(out_path, cwd),
                "count": out["count"],
                "labels": out["labels"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
