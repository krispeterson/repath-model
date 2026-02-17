#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retraining-manifest JSON from benchmark-labeled CSV rows.",
        usage=(
            "python3 scripts/training/build_retraining_manifest.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out ml/artifacts/retraining/retraining-manifest.json]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input benchmark-labeled CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("ml") / "artifacts" / "retraining" / "retraining-manifest.json"),
        help="Output retraining-manifest JSON path.",
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


def build_samples(rows: list[dict]) -> list[dict]:
    retrain_rows = [
        row for row in rows if str(row.get("name") or "").startswith("retrain_") and str(row.get("url") or "")
    ]

    samples = []
    for row in retrain_rows:
        name = str(row.get("name") or "")
        canonical_label = str(row.get("canonical_label") or "")
        is_negative = name.startswith("retrain_negative_") or not canonical_label
        samples.append(
            {
                "id": name,
                "image": str(row.get("url") or ""),
                "label": None if is_negative else canonical_label,
                "is_negative": is_negative,
                "source": str(row.get("source") or "") or "retraining_queue",
                "notes": str(row.get("notes") or ""),
            }
        )

    return samples


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = read_rows(input_path)
    samples = build_samples(rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": rel_or_abs(input_path, cwd),
        "sample_count": len(samples),
        "positive_count": sum(1 for sample in samples if not sample.get("is_negative")),
        "negative_count": sum(1 for sample in samples if sample.get("is_negative")),
        "samples": samples,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Retraining manifest generated")
    print(
        json.dumps(
            {
                "input": rel_or_abs(input_path, cwd),
                "output": rel_or_abs(out_path, cwd),
                "samples": payload["sample_count"],
                "positives": payload["positive_count"],
                "negatives": payload["negative_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
