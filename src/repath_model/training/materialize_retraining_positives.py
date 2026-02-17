#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

CSV_COLUMNS = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def usage() -> str:
    return (
        "python3 scripts/training/materialize_retraining_positives.py "
        "[--input test/benchmarks/benchmark-labeled.csv] "
        "[--out test/benchmarks/benchmark-labeled.csv] "
        "[--cache-dir test/benchmarks/images/retraining-positives] "
        "[--labels \"Aluminum Foil,Pizza Box\"] [--dry-run]"
    )


def parse_labels(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input benchmark-labeled CSV.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Output benchmark-labeled CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images" / "retraining-positives"),
        help="Cache directory for downloaded retraining positives.",
    )
    parser.add_argument("--labels", type=parse_labels, default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")


def rel_or_abs(path: Path) -> str:
    try:
        return rel(path)
    except Exception:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({column: str((row or {}).get(column, "")).strip() for column in CSV_COLUMNS})
        return rows


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: str((row or {}).get(column, "")) for column in CSV_COLUMNS})


def is_http_url(value: str) -> bool:
    return bool(re.match(r"^https?://", str(value or ""), flags=re.IGNORECASE))


def extension_from_url(value: str) -> str:
    match = re.search(r"\.([a-zA-Z0-9]{2,6})(?:[?#].*)?$", str(value or ""))
    if not match:
        return ".jpg"
    return f".{str(match.group(1) or '').lower()}"


def append_source_url_note(existing: str, source_url: str) -> str:
    notes = str(existing or "").strip()
    if not source_url:
        return notes

    escaped = re.escape(source_url)
    pattern = re.compile(rf"(?:^|;\s*)source_url={escaped}(?:;|$)", flags=re.IGNORECASE)
    if pattern.search(notes):
        return notes

    if not notes:
        return f"source_url={source_url}"

    return f"{notes}; source_url={source_url}"


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--max-time",
            "90",
            "--fail",
            url,
            "-o",
            str(out_path),
            "-sS",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    label_filter = set(args.labels)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = read_csv_rows(input_path)
    candidates = [
        row
        for row in rows
        if str((row or {}).get("name", "")).startswith("retrain_positive_")
        and is_http_url(str((row or {}).get("url", "")))
    ]

    if label_filter:
        candidates = [row for row in candidates if str((row or {}).get("canonical_label", "")) in label_filter]

    downloaded = 0
    reused_local = 0
    failures: list[dict] = []

    for row in candidates:
        source_url = str((row or {}).get("url", ""))
        ext = extension_from_url(source_url)
        out_file = cache_dir / f"{str((row or {}).get('name', '')).strip()}{ext}"

        try:
            if not out_file.exists():
                if not args.dry_run:
                    download(source_url, out_file)
                downloaded += 1
            else:
                reused_local += 1

            row["url"] = rel(out_file)
            row["notes"] = append_source_url_note(str((row or {}).get("notes", "")), source_url)
        except Exception as error:
            failures.append(
                {
                    "name": str((row or {}).get("name", "")),
                    "label": str((row or {}).get("canonical_label", "")),
                    "source_url": source_url,
                    "error": str(error),
                }
            )

    if not args.dry_run:
        write_csv_rows(out_path, rows)

    print("Retraining positive materialization summary")
    print(
        json.dumps(
            {
                "input": rel_or_abs(input_path),
                "output": rel_or_abs(out_path),
                "cache_dir": rel_or_abs(cache_dir),
                "labels_filter": args.labels,
                "remote_candidates": len(candidates),
                "downloaded": downloaded,
                "reused_local": reused_local,
                "failed": len(failures),
                "failures": failures[:20],
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
