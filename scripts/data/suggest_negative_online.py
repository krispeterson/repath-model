#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]
DEFAULT_UA = "repath-mobile-negative-bot/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest negative benchmark examples from Wikimedia Commons.",
        usage=(
            "python3 scripts/data/suggest_negative_online.py "
            "[--manifest test/benchmarks/municipal-benchmark-manifest-v2.json] "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out test/benchmarks/benchmark-labeled.negatives.csv] "
            "[--merge-into test/benchmarks/benchmark-labeled.csv] [--limit 20]"
        ),
    )
    parser.add_argument("--manifest", default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"))
    parser.add_argument("--input", default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"))
    parser.add_argument("--out", default=str(Path("test") / "benchmarks" / "benchmark-labeled.negatives.csv"))
    parser.add_argument("--merge-into", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    args.limit = args.limit if args.limit and args.limit > 0 else 20
    args.timeout_ms = args.timeout_ms if args.timeout_ms and args.timeout_ms >= 1000 else 15000
    args.max_retries = args.max_retries if args.max_retries and args.max_retries >= 1 else 3
    return args


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{column: str((row or {}).get(column, "")).strip() for column in HEADER} for row in reader]


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def merge_rows(existing_rows: list[dict], updates: list[dict]) -> list[dict]:
    merged = {}
    for row in existing_rows:
        name = str(row.get("name") or "").strip()
        if name:
            merged[name] = row
    for row in updates:
        name = str(row.get("name") or "").strip()
        if name:
            merged[name] = row
    return [merged[key] for key in sorted(merged.keys(), key=lambda value: value.lower())]


def fetch_json(url: str, timeout_ms: int):
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_UA}, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_ms / 1000.0) as response:
        status = getattr(response, "status", 200)
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return json.loads(response.read().decode("utf-8"))


def search_commons(query: str, timeout_ms: int, max_retries: int) -> list[str]:
    base = "https://commons.wikimedia.org/w/api.php"
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srnamespace": "6",
            "srlimit": "5",
            "srsearch": f"{query} filetype:bitmap",
        }
    )
    url = f"{base}?{params}"

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            payload = fetch_json(url, timeout_ms)
            rows = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
            titles = []
            for row in rows if isinstance(rows, list) else []:
                title = str((row or {}).get("title") or "").strip()
                if title:
                    titles.append(title)
            return titles
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
            last_error = error
            if attempt < max_retries:
                time.sleep(0.3 * attempt)

    if last_error:
        raise last_error
    raise RuntimeError("request_failed")


def to_commons_file_path_url(title: str) -> str:
    normalized = title[5:] if title.startswith("File:") else title
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(normalized)}"


def is_negative(entry: dict) -> bool:
    expected_any = entry.get("expected_any") if isinstance(entry, dict) else []
    expected_all = entry.get("expected_all") if isinstance(entry, dict) else []
    expected_any = expected_any if isinstance(expected_any, list) else []
    expected_all = expected_all if isinstance(expected_all, list) else []
    return len(expected_any) == 0 and len(expected_all) == 0


def query_hint(entry: dict) -> str:
    notes = str((entry or {}).get("notes") or "")
    match = re.search(r"query_hint=([^;]+)", notes, flags=re.IGNORECASE)
    if match and match.group(1):
        return match.group(1).strip()

    name = str((entry or {}).get("name") or "")
    name = re.sub(r"^todo_negative_", "", name)
    name = re.sub(r"_[0-9]+$", "", name)
    name = re.sub(r"[-_]+", " ", name).strip()
    return name or "street scene"


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest = load_json(Path(args.manifest).resolve())
    in_rows = read_csv_rows(Path(args.input).resolve())
    used_urls = {str(row.get("url") or "").strip() for row in in_rows if str(row.get("url") or "").strip()}

    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []
    negatives = [
        entry
        for entry in images
        if isinstance(entry, dict)
        and is_negative(entry)
        and str(entry.get("status") or "").lower() == "todo"
    ][: args.limit]

    updates = []

    for entry in negatives:
        name = str(entry.get("name") or "").strip()
        hint = query_hint(entry)

        try:
            titles = search_commons(hint, timeout_ms=args.timeout_ms, max_retries=args.max_retries)
        except Exception:
            continue

        picked_url = ""
        picked_title = ""
        for title in titles:
            url = to_commons_file_path_url(title)
            if url in used_urls:
                continue
            picked_url = url
            picked_title = title
            used_urls.add(url)
            break

        if not picked_url:
            continue

        updates.append(
            {
                "name": name,
                "url": picked_url,
                "item_id": "",
                "canonical_label": "",
                "source": "wikimedia_commons_negative_search",
                "notes": f"title={picked_title}; query={hint}",
            }
        )

    out_path = Path(args.out).resolve()
    write_csv_rows(out_path, updates)

    merged_count = None
    merged_into = None
    if args.merge_into:
        merge_path = Path(args.merge_into).resolve()
        merged = merge_rows(read_csv_rows(merge_path), updates)
        write_csv_rows(merge_path, merged)
        merged_count = len(merged)
        merged_into = rel_or_abs(merge_path, cwd)

    print("Negative online suggestions generated")
    print(
        json.dumps(
            {
                "attempted": len(negatives),
                "matched_rows": len(updates),
                "output": rel_or_abs(out_path, cwd),
                "merged_into": merged_into,
                "merged_row_count": merged_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
