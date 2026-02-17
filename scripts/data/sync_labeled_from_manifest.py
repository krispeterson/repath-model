#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync benchmark-labeled CSV rows from benchmark manifest entries.",
        usage=(
            "python3 scripts/data/sync_labeled_from_manifest.py "
            "[--manifest test/benchmarks/municipal-benchmark-manifest-v2.json] "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out test/benchmarks/benchmark-labeled.csv] [--include-ready] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Manifest path.",
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--include-ready", action="store_true", help="Include ready manifest entries, not just todo.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output CSV.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    if len(lines) <= 1:
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


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HEADER})


def first_label(entry: dict) -> str:
    expected_any = entry.get("expected_any") if isinstance(entry, dict) else []
    expected_all = entry.get("expected_all") if isinstance(entry, dict) else []
    expected_any = expected_any if isinstance(expected_any, list) else []
    expected_all = expected_all if isinstance(expected_all, list) else []
    if expected_any:
        return str(expected_any[0] or "").strip()
    if expected_all:
        return str(expected_all[0] or "").strip()
    return ""


def merge_notes(existing: str, add: str) -> str:
    e = str(existing or "").strip()
    a = str(add or "").strip()
    if not e:
        return a
    if not a:
        return e
    if a in e:
        return e
    return f"{e}; {a}"


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest_path = Path(args.manifest).resolve()
    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []
    rows = read_csv_rows(input_path)

    row_map = {row["name"]: row for row in rows if row.get("name")}

    added = 0
    enriched = 0

    for entry in images:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue

        status = str(entry.get("status") or "").lower()
        if (not args.include_ready) and status != "todo":
            continue

        canonical_label = first_label(entry)
        item_id = str(entry.get("item_id") or "").strip()
        note = str(entry.get("notes") or "").strip()

        existing = row_map.get(name)
        if not existing:
            row_map[name] = {
                "name": name,
                "url": str(entry.get("url") or "").strip(),
                "item_id": item_id,
                "canonical_label": canonical_label,
                "source": "manifest_todo_queue",
                "notes": merge_notes(note, "synced_from_manifest"),
            }
            added += 1
            continue

        changed = False
        if not existing.get("item_id") and item_id:
            existing["item_id"] = item_id
            changed = True
        if not existing.get("canonical_label") and canonical_label:
            existing["canonical_label"] = canonical_label
            changed = True

        merged_notes = merge_notes(existing.get("notes", ""), "synced_from_manifest")
        if merged_notes != existing.get("notes", ""):
            existing["notes"] = merged_notes
            changed = True

        if changed:
            enriched += 1

    merged = sorted(row_map.values(), key=lambda row: str(row.get("name") or ""))

    if not args.dry_run:
        write_csv_rows(out_path, merged)

    print("Manifest-to-labeled sync summary")
    print(
        json.dumps(
            {
                "rows_before": len(rows),
                "rows_after": len(merged),
                "added": added,
                "enriched": enriched,
                "include_ready": bool(args.include_ready),
                "output": rel_or_abs(out_path, cwd),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
