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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest benchmark URLs from Wikimedia Commons for unresolved labeled rows.",
        usage=(
            "python3 scripts/data/suggest_benchmark_online.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--out test/benchmarks/benchmark-labeled.online.csv] "
            "[--merge-into test/benchmarks/benchmark-labeled.csv] "
            "[--limit 30] [--offset 0] [--timeout-ms 15000] [--max-retries 3] "
            "[--include-previous-failures]"
        ),
    )
    parser.add_argument("--input", default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"))
    parser.add_argument("--out", default=str(Path("test") / "benchmarks" / "benchmark-labeled.online.csv"))
    parser.add_argument("--merge-into", default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--include-previous-failures", action="store_true")
    args = parser.parse_args()

    args.limit = args.limit if args.limit and args.limit > 0 else 30
    args.offset = args.offset if args.offset and args.offset >= 0 else 0
    args.timeout_ms = args.timeout_ms if args.timeout_ms and args.timeout_ms >= 1000 else 15000
    args.max_retries = args.max_retries if args.max_retries and args.max_retries >= 1 else 3
    return args


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({column: str((row or {}).get(column, "")).strip() for column in HEADER})
        return rows


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def fetch_json(url: str, timeout_ms: int):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "repath-mobile-benchmark-bot/1.0 (local dev)"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_ms / 1000.0) as response:
        status = getattr(response, "status", 200)
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        payload = response.read().decode("utf-8")
        return json.loads(payload)


def find_commons_file_titles(label: str, timeout_ms: int, max_retries: int) -> list[str]:
    base = "https://commons.wikimedia.org/w/api.php"
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srnamespace": "6",
            "srlimit": "10",
            "srsearch": f"{label} filetype:bitmap",
        }
    )
    url = f"{base}?{query}"

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


def to_commons_file_path_url(file_title: str) -> str:
    normalized = file_title[5:] if file_title.startswith("File:") else file_title
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(normalized)}"


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def title_case_words(text: str) -> str:
    words = [word for word in str(text or "").split() if word]
    out = []
    for word in words:
        if len(word) <= 2:
            out.append(word.lower())
        else:
            out.append(word[0].upper() + word[1:].lower())
    return " ".join(out)


def tokenize_label(label: str) -> str:
    value = str(label or "")
    value = re.sub(r"[()]", " ", value)
    value = re.sub(r"[,&]", " ", value)
    value = re.sub(r"\bother than\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bmeal kit\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\bsingle-use\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


LABEL_ALIAS_MAP = {
    "automotive fluids other than motor oil antifreeze": ["automotive fluid bottle", "car fluid container", "vehicle fluid bottle"],
    "bulky rigid plastics": ["large plastic item", "rigid plastic container", "plastic storage tote"],
    "cereal liner bag": ["cereal box liner bag", "plastic cereal liner", "mylar cereal bag"],
    "clear plastic berry and salad container": ["clear clamshell container", "plastic produce container", "clear salad container"],
    "corrugated plastic election sign": ["corrugated plastic sign", "yard sign plastic", "political yard sign"],
    "envelopes with bubble wrap inside": ["bubble mailer envelope", "padded envelope", "bubble lined envelope"],
    "kitty litter bucket": ["cat litter bucket", "plastic litter pail", "litter tub container"],
    "plastic disinfectant wipes container": ["disinfecting wipes canister", "clorox wipes container", "wipes plastic canister"],
    "plastic grocery bags and plastic film": ["plastic grocery bag", "plastic film wrap", "polyethylene bag"],
    "plastic lawn furniture": ["plastic patio chair", "outdoor plastic furniture", "resin lawn chair"],
    "reflective bubble wrap and foil bubble mailers": ["foil bubble mailer", "metalized bubble wrap", "reflective bubble insulation"],
    "rubbermaid storage bin": ["plastic storage bin", "rubbermaid tote", "storage tote container"],
    "shredded brown crinkle paper": ["shredded kraft paper", "brown crinkle paper filler", "packaging paper shred"],
    "sun basket paper insulation": ["paper insulation packaging", "meal kit paper insulation", "recycled paper insulation liner"],
    "woven plastic feed bag": ["woven polypropylene bag", "feed sack bag", "grain feed bag woven"],
    "foil coffee bags": ["foil coffee bag", "coffee bean bag foil", "laminated coffee pouch"],
    "antifreeze bottle": ["antifreeze jug", "coolant bottle", "automotive coolant container"],
}


def normalize_label_key(label: str) -> str:
    value = tokenize_label(label).lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def build_query_variants(row: dict) -> list[str]:
    label = str(row.get("canonical_label") or "").strip()
    item_id = str(row.get("item_id") or "").strip()
    item_from_id = re.sub(r"[-_]+", " ", item_id).strip()
    item_from_id = re.sub(r"\bdepth\b", "", item_from_id, flags=re.IGNORECASE).strip()

    cleaned_label = tokenize_label(label)
    cleaned_label = re.sub(r"[/]+", " ", cleaned_label)
    cleaned_label = re.sub(r"\s+", " ", cleaned_label).strip()

    split_parts = []
    for part in re.split(r"/|\bor\b", label, flags=re.IGNORECASE):
        part = tokenize_label(part)
        part = re.sub(r"\bplastic film election sign\b", "plastic sign", part, flags=re.IGNORECASE).strip()
        part = re.sub(r"\bpaperboard election sign\b", "cardboard sign", part, flags=re.IGNORECASE).strip()
        if part:
            split_parts.append(part)

    singular_parts = [
        re.sub(r"\bbags\b", "bag", re.sub(r"\bcontainers\b", "container", part, flags=re.IGNORECASE), flags=re.IGNORECASE)
        for part in split_parts
    ]
    generic_hints = [f"{part} object" for part in split_parts]
    aliases = LABEL_ALIAS_MAP.get(normalize_label_key(label), [])
    safe_title = title_case_words(cleaned_label)

    return unique([label, item_from_id, cleaned_label, safe_title, *aliases, *split_parts, *singular_parts, *generic_hints])


def is_previous_no_match(row: dict) -> bool:
    source = str(row.get("source") or "").strip().lower()
    notes = str(row.get("notes") or "").lower()
    return source == "wikimedia_commons_search" and "no_match" in notes


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


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    in_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    rows = read_csv_rows(in_path)
    existing_rows = read_csv_rows(Path(args.merge_into).resolve()) if args.merge_into else rows

    used_urls = {str(row.get("url") or "").strip() for row in existing_rows if str(row.get("url") or "").strip()}
    unresolved = [row for row in rows if not str(row.get("url") or "").strip() and str(row.get("canonical_label") or "").strip()]

    skipped_previous = sum(1 for row in unresolved if is_previous_no_match(row))
    pool = unresolved if args.include_previous_failures else [row for row in unresolved if not is_previous_no_match(row)]

    targets = pool[args.offset : args.offset + args.limit]

    updates = []
    no_match_count = 0

    for row in targets:
        variants = build_query_variants(row)
        title = None
        matched_query = ""

        for query_text in variants:
            try:
                titles = find_commons_file_titles(query_text, timeout_ms=args.timeout_ms, max_retries=args.max_retries)
            except Exception:
                continue

            for candidate_title in titles:
                candidate_url = to_commons_file_path_url(candidate_title)
                if candidate_url in used_urls:
                    continue
                title = candidate_title
                used_urls.add(candidate_url)
                matched_query = query_text
                break

            if title:
                break

        if title:
            updates.append(
                {
                    **row,
                    "url": to_commons_file_path_url(title),
                    "source": "wikimedia_commons_search",
                    "notes": f"title={title}; query={matched_query}",
                }
            )
        else:
            no_match_count += 1
            updates.append({**row, "source": "wikimedia_commons_search", "notes": "no_match"})

    write_csv_rows(out_path, updates)

    merged_count = None
    merged_into = None
    if args.merge_into:
        merge_path = Path(args.merge_into).resolve()
        merged = merge_rows(read_csv_rows(merge_path), updates)
        write_csv_rows(merge_path, merged)
        merged_count = len(merged)
        merged_into = rel_or_abs(merge_path, cwd)

    print("Online benchmark suggestions generated")
    print(
        json.dumps(
            {
                "attempted": len(targets),
                "offset": args.offset,
                "skipped_previously_attempted": skipped_previous,
                "unresolved_pool": len(pool),
                "matched_rows": sum(1 for row in updates if str(row.get("url") or "").strip()),
                "no_match_rows": no_match_count,
                "output": rel_or_abs(out_path, cwd),
                "merged_into": merged_into,
                "merged_row_count": merged_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
