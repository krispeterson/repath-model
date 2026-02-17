#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge coverage expansion template rows into benchmark-labeled CSV.",
        usage=(
            "python3 scripts/data/merge_coverage_expansion_template.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--template test/benchmarks/benchmark-coverage-expansion-template.csv] "
            "[--out test/benchmarks/benchmark-labeled.csv] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input labeled CSV path.",
    )
    parser.add_argument(
        "--template",
        default=str(Path("test") / "benchmarks" / "benchmark-coverage-expansion-template.csv"),
        help="Coverage template CSV path.",
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


def read_csv_rows(file_path: Path) -> list[dict]:
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines:
        return []

    has_header = "name" in lines[0].lower() and "," in lines[0].lower()
    start = 1 if has_header else 0
    rows = []
    for line in lines[start:]:
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


def read_template_rows(file_path: Path) -> list[dict]:
    if not file_path.exists():
        return []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines:
        return []

    has_header = "name" in lines[0].lower() and "canonical_label" in lines[0].lower()
    start = 1 if has_header else 0

    rows = []
    for line in lines[start:]:
        cols = next(csv.reader([line]))
        row = {
            "name": str(cols[0] if len(cols) > 0 else "").strip(),
            "url": str(cols[1] if len(cols) > 1 else "").strip(),
            "item_id": str(cols[2] if len(cols) > 2 else "").strip(),
            "canonical_label": str(cols[3] if len(cols) > 3 else "").strip(),
            "current_ready_count": str(cols[4] if len(cols) > 4 else "").strip(),
            "target_ready_count": str(cols[5] if len(cols) > 5 else "").strip(),
            "needed_for_target": str(cols[6] if len(cols) > 6 else "").strip(),
            "notes": str(cols[7] if len(cols) > 7 else "").strip(),
        }
        if row["name"]:
            rows.append(row)
    return rows


def write_rows(file_path: Path, rows: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def merge_notes(existing: str, addition: str) -> str:
    e = str(existing or "").strip()
    a = str(addition or "").strip()
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

    input_path = Path(args.input).resolve()
    template_path = Path(args.template).resolve()
    out_path = Path(args.out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    if not template_path.exists():
        raise SystemExit(f"Coverage template not found: {template_path}")

    input_rows = read_csv_rows(input_path)
    template_rows = read_template_rows(template_path)

    by_name = {row["name"]: row for row in input_rows if row.get("name")}

    added = 0
    enriched = 0
    unchanged = 0

    for row in template_rows:
        existing = by_name.get(row["name"])
        template_note = f"coverage-expansion target={row['target_ready_count']} needed={row['needed_for_target']}"

        if not existing:
            by_name[row["name"]] = {
                "name": row["name"],
                "url": row["url"] or "",
                "item_id": row["item_id"] or "",
                "canonical_label": row["canonical_label"] or "",
                "source": "coverage_expansion_queue",
                "notes": merge_notes(row["notes"], template_note),
            }
            added += 1
            continue

        changed = False
        if not existing.get("item_id") and row.get("item_id"):
            existing["item_id"] = row["item_id"]
            changed = True
        if not existing.get("canonical_label") and row.get("canonical_label"):
            existing["canonical_label"] = row["canonical_label"]
            changed = True
        if not existing.get("source"):
            existing["source"] = "coverage_expansion_queue"
            changed = True

        merged_notes = merge_notes(existing.get("notes", ""), template_note)
        if merged_notes != existing.get("notes", ""):
            existing["notes"] = merged_notes
            changed = True

        if changed:
            enriched += 1
        else:
            unchanged += 1

    merged_rows = sorted(by_name.values(), key=lambda row: str(row.get("name") or ""))

    if not args.dry_run:
        write_rows(out_path, merged_rows)

    print("Coverage expansion template merge summary")
    print(
        json.dumps(
            {
                "template_rows": len(template_rows),
                "existing_rows": len(input_rows),
                "merged_rows": len(merged_rows),
                "added": added,
                "enriched": enriched,
                "unchanged": unchanged,
                "output": rel_or_abs(out_path, cwd),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
