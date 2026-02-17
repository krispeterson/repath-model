#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse, unquote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize benchmark-labeled URL values to repo-relative paths where possible.",
        usage=(
            "python3 scripts/data/normalize_benchmark_labeled_urls.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--cache-dir test/benchmarks/images] [--out test/benchmarks/benchmark-labeled.csv]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images"),
        help="Local cache directory for external file:// assets.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines:
        return [], []

    header = next(csv.reader([lines[0]]))
    rows = []
    for line in lines[1:]:
        cols = next(csv.reader([line]))
        row = {}
        for i, key in enumerate(header):
            row[key] = str(cols[i] if i < len(cols) else "")
        rows.append(row)
    return header, rows


def write_rows(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def sanitize_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    text = re.sub(r"^-+|-+$", "", text)
    return text[:120]


def extension_for_url(url_value: str) -> str:
    match = re.search(r"\.([a-zA-Z0-9]{2,6})(?:[?#].*)?$", str(url_value or ""))
    if match:
        return f".{match.group(1).lower()}"
    return ".jpg"


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    input_path = Path(args.input).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_path = Path(args.out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    header, rows = read_rows(input_path)
    if "url" not in header or "name" not in header:
        raise SystemExit("CSV must include at least name,url columns.")

    cache_dir.mkdir(parents=True, exist_ok=True)

    normalized_count = 0
    copied_count = 0

    for row in rows:
        url = str(row.get("url") or "").strip()
        if not url:
            continue

        if re.match(r"^https?://", url, flags=re.IGNORECASE):
            continue

        if not url.startswith("file://"):
            absolute = Path(url).resolve()
            row["url"] = rel_or_abs(absolute, cwd)
            normalized_count += 1
            continue

        try:
            parsed = urlparse(url)
            local_path = Path(unquote(parsed.path))
        except Exception:  # noqa: BLE001
            continue

        if not local_path.exists():
            continue

        local_resolved = local_path.resolve()
        inside_repo = str(local_resolved).startswith(str(cwd.resolve()) + str(Path('/')))
        if inside_repo:
            row["url"] = rel_or_abs(local_resolved, cwd)
            normalized_count += 1
            continue

        ext = extension_for_url(str(local_path))
        out_file = cache_dir / f"{sanitize_name(str(row.get('name') or 'sample'))}{ext}"
        if not out_file.exists():
            shutil.copyfile(local_resolved, out_file)
            copied_count += 1

        row["url"] = rel_or_abs(out_file, cwd)
        normalized_count += 1

    write_rows(out_path, header, rows)

    print("Normalized benchmark labeled URLs")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "normalized": normalized_count,
                "copied_into_cache": copied_count,
                "output": rel_or_abs(out_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
