#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dedupe benchmark-labeled CSV by URL by clearing duplicate URLs.",
        usage=(
            "python3 scripts/data/dedupe_benchmark_labeled.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out test/benchmarks/benchmark-labeled.csv] [--keep-first|--keep-last] [--dry-run]"
        ),
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
    parser.add_argument("--keep-first", action="store_true", help="Keep first row URL in duplicate groups.")
    parser.add_argument("--keep-last", action="store_true", help="Keep last row URL in duplicate groups.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output CSV.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict]:
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


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def append_note(notes: str, marker: str) -> str:
    base = str(notes or "").strip()
    if not base:
        return marker
    if marker in base:
        return base
    return f"{base}; {marker}"


def main() -> None:
    args = parse_args()
    keep_first = True
    if args.keep_last:
        keep_first = False
    elif args.keep_first:
        keep_first = True

    cwd = Path.cwd()
    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = read_rows(input_path)

    by_url: dict[str, list[tuple[dict, int]]] = {}
    for index, row in enumerate(rows):
        url = str(row.get("url") or "").strip()
        if not url:
            continue
        by_url.setdefault(url, []).append((row, index))

    changed = 0
    groups = 0
    for entries in by_url.values():
        if len(entries) < 2:
            continue
        groups += 1
        keep_pos = 0 if keep_first else len(entries) - 1
        for pos, (row, _) in enumerate(entries):
            if pos == keep_pos:
                continue
            if not row.get("url"):
                continue
            row["url"] = ""
            row["notes"] = append_note(row.get("notes", ""), "Needs unique URL (csv dedupe).")
            changed += 1

    if not args.dry_run:
        write_rows(out_path, rows)

    print("Labeled CSV dedupe summary")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "duplicate_url_groups": groups,
                "rows_cleared": changed,
                "keep_first": keep_first,
                "dry_run": bool(args.dry_run),
                "output": rel_or_abs(out_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
