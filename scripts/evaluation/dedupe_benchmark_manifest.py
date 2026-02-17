#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dedupe benchmark manifest by URL by marking duplicate entries as todo.",
        usage=(
            "python3 scripts/evaluation/dedupe_benchmark_manifest.py "
            "[--manifest test/benchmarks/municipal-benchmark-manifest-v2.json] "
            "[--report test/benchmarks/benchmark-dedupe-report.json] "
            "[--keep-first|--keep-last] [--clear-url|--keep-url] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument(
        "--report",
        default=str(Path("test") / "benchmarks" / "benchmark-dedupe-report.json"),
        help="Output report JSON path.",
    )
    parser.add_argument("--keep-first", action="store_true", help="Keep first URL occurrence.")
    parser.add_argument("--keep-last", action="store_true", help="Keep last URL occurrence.")
    parser.add_argument("--clear-url", action="store_true", help="Clear URL on duplicate rows.")
    parser.add_argument("--keep-url", action="store_true", help="Preserve URL on duplicate rows.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write manifest changes.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_url_groups(images: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for index, entry in enumerate(images):
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        if not url:
            continue
        groups.setdefault(url, []).append({"index": index, "entry": entry})
    return groups


def ensure_todo_notes(entry: dict) -> str:
    base = str(entry.get("notes") or "").strip() if isinstance(entry, dict) else ""
    marker = "Needs unique URL (deduped)."
    if not base:
        return marker
    if marker in base:
        return base
    return f"{base} {marker}"


def apply_dedup(images: list[dict], keep_first: bool, clear_url: bool) -> tuple[list[dict], list[dict]]:
    groups = build_url_groups(images)
    duplicate_groups: list[dict] = []
    changed: list[dict] = []

    for url, rows in groups.items():
        if len(rows) < 2:
            continue

        keeper_pos = 0 if keep_first else len(rows) - 1
        keeper = rows[keeper_pos]
        dupes = [row for pos, row in enumerate(rows) if pos != keeper_pos]

        duplicate_groups.append(
            {
                "url": url,
                "keep_name": str((keeper.get("entry") or {}).get("name") or ""),
                "duplicate_names": [str((row.get("entry") or {}).get("name") or "") for row in dupes],
            }
        )

        for row in dupes:
            entry = row.get("entry")
            if not isinstance(entry, dict):
                continue

            prev = {
                "name": str(entry.get("name") or ""),
                "status": str(entry.get("status") or ""),
                "url": str(entry.get("url") or ""),
                "notes": str(entry.get("notes") or ""),
            }

            entry["status"] = "todo"
            if clear_url:
                entry["url"] = ""
            entry["notes"] = ensure_todo_notes(entry)

            changed.append(
                {
                    "name": prev["name"],
                    "from": prev,
                    "to": {
                        "status": str(entry.get("status") or ""),
                        "url": str(entry.get("url") or ""),
                        "notes": str(entry.get("notes") or ""),
                    },
                }
            )

    return duplicate_groups, changed


def count_summary(images: list[dict]) -> dict:
    return {
        "total": len(images),
        "ready": sum(1 for entry in images if str((entry or {}).get("status") or "").lower() == "ready"),
        "todo": sum(1 for entry in images if str((entry or {}).get("status") or "").lower() == "todo"),
        "with_url": sum(1 for entry in images if str((entry or {}).get("url") or "").strip()),
        "missing_url": sum(1 for entry in images if not str((entry or {}).get("url") or "").strip()),
    }


def main() -> None:
    args = parse_args()
    keep_first = True
    clear_url = True
    if args.keep_last:
        keep_first = False
    elif args.keep_first:
        keep_first = True
    if args.keep_url:
        clear_url = False
    elif args.clear_url:
        clear_url = True

    cwd = Path.cwd()
    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    manifest = load_json(manifest_path)
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    before = count_summary(images)
    duplicate_groups, changed = apply_dedup(images, keep_first=keep_first, clear_url=clear_url)
    after = count_summary(images)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": rel_or_abs(manifest_path, cwd),
            "dry_run": bool(args.dry_run),
            "keep_first": keep_first,
            "clear_url": clear_url,
        },
        "summary": {
            "duplicate_url_groups": len(duplicate_groups),
            "entries_changed": len(changed),
            "before": before,
            "after": after,
        },
        "duplicate_groups": duplicate_groups,
        "changes": changed,
    }

    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Benchmark dedupe summary")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report to {rel_or_abs(report_path, cwd)}")
    if not args.dry_run:
        print(f"Updated manifest: {rel_or_abs(manifest_path, cwd)}")


if __name__ == "__main__":
    main()
