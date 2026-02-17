#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing retraining negative URLs from existing baseline negative pool.",
        usage=(
            "python3 scripts/fill_retraining_negatives.py "
            "[--input test/benchmarks/benchmark-labeled.csv] [--out test/benchmarks/benchmark-labeled.csv] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input labeled CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Output labeled CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images" / "retraining-negatives"),
        help="Cache directory for materialized negatives.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not materialize files or write output.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


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


def is_retraining_negative(row: dict) -> bool:
    return str(row.get("source") or "").strip() == "retraining_queue_negative"


def is_baseline_negative(row: dict) -> bool:
    return (not str(row.get("canonical_label") or "").strip()) and bool(str(row.get("url") or "").strip()) and (
        "negative" in str(row.get("source") or "")
    )


def extension_from_url(value: str) -> str:
    match = re.search(r"\.([a-zA-Z0-9]{2,6})(?:[?#].*)?$", str(value or ""))
    if not match:
        return ".jpg"
    return f".{match.group(1).lower()}"


def resolve_local_path(raw_url: str) -> Path | None:
    value = str(raw_url or "").strip()
    if not value:
        return None
    if value.startswith("file://"):
        try:
            parsed = urlparse(value)
            return Path(unquote(parsed.path))
        except Exception:  # noqa: BLE001
            return None
    if re.match(r"^https?://", value, flags=re.IGNORECASE):
        return None
    return Path(value).resolve()


def copy_or_download_to(target_path: Path, source_url: str) -> str:
    local_path = resolve_local_path(source_url)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path and local_path.exists():
        shutil.copyfile(local_path, target_path)
        return "copied_local"

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
            source_url,
            "-o",
            str(target_path),
            "-sS",
        ],
        check=True,
    )
    return "downloaded"


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    rows = read_rows(input_path)
    candidate_pool = [row.get("url") for row in rows if is_baseline_negative(row)]
    candidate_pool = [url for url in candidate_pool if url]

    unresolved_targets = [row for row in rows if is_retraining_negative(row) and not str(row.get("url") or "").strip()]

    filled = 0
    copied_local = 0
    downloaded = 0
    failed = 0

    for idx, row in enumerate(unresolved_targets):
        candidate = candidate_pool[idx % len(candidate_pool)] if candidate_pool else None
        if not candidate:
            failed += 1
            continue

        ext = extension_from_url(candidate)
        fallback_name = f"retraining-negative-{idx + 1}"
        out_file = cache_dir / f"{(row.get('name') or fallback_name)}{ext}"
        out_relative = rel_or_abs(out_file, cwd)

        if not args.dry_run:
            try:
                mode = copy_or_download_to(out_file, candidate)
                if mode == "copied_local":
                    copied_local += 1
                if mode == "downloaded":
                    downloaded += 1
            except Exception:  # noqa: BLE001
                failed += 1
                continue

        row["url"] = out_relative
        row["notes"] = f"{row.get('notes', '')}; seeded_from=baseline_negative_pool; materialized_local={out_relative}"
        filled += 1

    if not args.dry_run:
        write_rows(out_path, rows)

    print("Retraining negative fill summary")
    print(
        json.dumps(
            {
                "total_rows": len(rows),
                "unresolved_retraining_negatives": len(unresolved_targets),
                "baseline_negative_candidates": len(candidate_pool),
                "filled": filled,
                "copied_local": copied_local,
                "downloaded": downloaded,
                "failed": failed,
                "cache_dir": rel_or_abs(cache_dir, cwd),
                "output": rel_or_abs(out_path, cwd),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
