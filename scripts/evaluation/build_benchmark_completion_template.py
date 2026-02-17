#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark completion CSV template from selected batches.")
    parser.add_argument(
        "--batches",
        default=str(Path("test") / "benchmarks" / "benchmark-labeling-batches.json"),
        help="Batch JSON path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-completion-template.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--bands", default="urgent,high", help="Comma-separated band list.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def unique_by_name(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    bands = [value.strip().lower() for value in str(args.bands or "").split(",") if value.strip()]
    if not bands:
        bands = ["urgent", "high"]

    cwd = Path.cwd()
    batches_path = Path(args.batches).resolve()
    out_path = Path(args.out).resolve()

    if not batches_path.exists():
        raise SystemExit(f"Batches file not found: {batches_path}")

    batches = load_json(batches_path)
    bucket = batches.get("batches") if isinstance(batches, dict) else {}
    bucket = bucket if isinstance(bucket, dict) else {}

    selected = []
    for band in bands:
        rows = bucket.get(band)
        rows = rows if isinstance(rows, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            selected.append(
                {
                    "name": row.get("name") or "",
                    "url": "",
                    "item_id": row.get("item_id") or "",
                    "canonical_label": row.get("canonical_label") or "",
                    "priority_band": row.get("priority_band") or band,
                    "priority_score": float(row.get("priority_score") or 0),
                    "notes": "",
                }
            )

    deduped = unique_by_name(selected)
    deduped.sort(key=lambda row: (-row.get("priority_score", 0), str(row.get("name") or "")))

    header = ["name", "url", "item_id", "canonical_label", "priority_band", "priority_score", "notes"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in deduped:
            writer.writerow({column: row.get(column, "") for column in header})

    print("Benchmark completion template generated")
    print(
        json.dumps(
            {
                "selected_bands": bands,
                "row_count": len(deduped),
                "output": rel_or_abs(out_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
