#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote_plus


HEADER = [
    "name",
    "item_id",
    "canonical_label",
    "current_source",
    "current_notes",
    "wikimedia_search_url",
    "google_images_search_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unresolved benchmark-labeled rows with search helper URLs.",
        usage=(
            "python3 scripts/evaluation/export_unresolved_benchmark_rows.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out test/benchmarks/benchmark-unresolved.csv]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input benchmark-labeled CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-unresolved.csv"),
        help="Output unresolved CSV path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header_seen = False
        for cols in reader:
            if not cols:
                continue
            if not header_seen:
                header_seen = True
                continue
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


def build_search_urls(label: str) -> dict[str, str]:
    query = f"{str(label or '').strip()} recycling".strip()
    encoded = quote_plus(query)
    return {
        "wikimedia_search_url": (
            "https://commons.wikimedia.org/w/index.php?search="
            f"{encoded}&title=Special:MediaSearch&type=image"
        ),
        "google_images_search_url": f"https://www.google.com/search?tbm=isch&q={encoded}",
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()

    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    unresolved = []
    for row in read_rows(in_path):
        if row.get("url"):
            continue
        unresolved.append(
            {
                "name": row.get("name", ""),
                "item_id": row.get("item_id", ""),
                "canonical_label": row.get("canonical_label", ""),
                "current_source": row.get("source", ""),
                "current_notes": row.get("notes", ""),
                **build_search_urls(str(row.get("canonical_label") or "")),
            }
        )

    unresolved.sort(key=lambda row: (str(row.get("canonical_label") or "").lower(), str(row.get("name") or "").lower()))
    write_rows(out_path, unresolved)

    print("Exported unresolved benchmark rows")
    print(
        json.dumps(
            {
                "unresolved_rows": len(unresolved),
                "output": rel_or_abs(out_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
