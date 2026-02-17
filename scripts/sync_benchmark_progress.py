#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync benchmark progress into manifest status/url fields.")
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest path.",
    )
    parser.add_argument("--completed", default=None, help="Completed rows file path (CSV or JSON).")
    parser.add_argument(
        "--report",
        default=str(Path("test") / "benchmarks" / "benchmark-progress-report.json"),
        help="Progress report output JSON path.",
    )
    parser.add_argument("--clear-empty-url", action="store_true", help="Clear existing URL when completed row URL is empty.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write updated manifest.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(file_path: Path):
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_completed_entries(file_path: str | None) -> list[dict]:
    if not file_path:
        return []

    full_path = Path(file_path).resolve()
    if not full_path.exists():
        raise SystemExit(f"Completed file not found: {full_path}")

    if full_path.suffix.lower() == ".json":
        data = load_json(full_path)
        rows = data if isinstance(data, list) else []
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "url": str(row.get("url") or "").strip(),
                    "notes": str(row.get("notes") or "").strip(),
                }
            )
        return out

    text = full_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    has_header = "name" in lines[0].lower()
    start = 1 if has_header else 0

    out = []
    for line in lines[start:]:
        if not line:
            continue

        if "," in line:
            cols = next(csv.reader([line]))
            cols = [col.strip() for col in cols]
            name = str(cols[0] if len(cols) > 0 else "").strip()
            url = str(cols[1] if len(cols) > 1 else "").strip()
            notes = str(cols[5] if len(cols) > 5 else "").strip()
            if name:
                out.append({"name": name, "url": url, "notes": notes})
        else:
            out.append({"name": line, "url": "", "notes": ""})

    return out


def index_by_name(images: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for entry in images:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        out.setdefault(name, []).append(entry)
    return out


def should_clear_url_from_row(row: dict) -> bool:
    notes = str(row.get("notes") or "").lower()
    return "needs unique url" in notes


def maybe_promote_from_url(entry: dict, changes: list[dict], locked_ready_names: set[str]) -> None:
    name = str(entry.get("name") or "").strip()
    current_status = str(entry.get("status") or "").lower()
    has_url = bool(str(entry.get("url") or "").strip())

    if has_url and current_status != "ready":
        entry["status"] = "ready"
        changes.append({"type": "status", "name": entry.get("name"), "from": current_status or "", "to": "ready"})

    if (not has_url) and current_status == "ready" and name not in locked_ready_names:
        entry["status"] = "todo"
        changes.append({"type": "status", "name": entry.get("name"), "from": "ready", "to": "todo"})


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    manifest = load_json(manifest_path)
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    by_name = index_by_name(images)
    completed_rows = load_completed_entries(args.completed)

    changes: list[dict] = []
    unknown_completed_names: list[str] = []
    skipped_missing_url: list[str] = []
    locked_ready_names: set[str] = set()

    for row in completed_rows:
        slots = by_name.get(row["name"])
        if not slots:
            unknown_completed_names.append(row["name"])
            continue

        for entry in slots:
            prev_status = str(entry.get("status") or "").lower()
            next_url = str(row.get("url") or "").strip()
            current_url = str(entry.get("url") or "").strip()

            if next_url and current_url != next_url:
                entry["url"] = next_url
                changes.append({"type": "url", "name": row["name"], "from": current_url, "to": next_url})
            elif (not next_url) and current_url and (args.clear_empty_url or should_clear_url_from_row(row)):
                entry["url"] = ""
                changes.append({"type": "url", "name": row["name"], "from": current_url, "to": ""})

            effective_url = str(entry.get("url") or "").strip()
            if not effective_url:
                skipped_missing_url.append(row["name"])
                continue

            if prev_status != "ready":
                entry["status"] = "ready"
                changes.append({"type": "status", "name": row["name"], "from": prev_status or "", "to": "ready"})

            locked_ready_names.add(row["name"])

            note = str(entry.get("notes") or "").strip()
            if "completed" not in note.lower():
                entry["notes"] = f"{note} Completed." if note else "Completed."

    for entry in images:
        if isinstance(entry, dict):
            maybe_promote_from_url(entry, changes, locked_ready_names)

    counts = {
        "total": len(images),
        "ready": sum(1 for entry in images if isinstance(entry, dict) and str(entry.get("status") or "").lower() == "ready"),
        "todo": sum(1 for entry in images if isinstance(entry, dict) and str(entry.get("status") or "").lower() == "todo"),
        "missing_url": sum(1 for entry in images if isinstance(entry, dict) and not str(entry.get("url") or "").strip()),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": rel_or_abs(manifest_path, cwd),
            "completed": rel_or_abs(Path(args.completed).resolve(), cwd) if args.completed else None,
            "dry_run": bool(args.dry_run),
        },
        "summary": {
            "completed_rows_applied": len(completed_rows),
            "unknown_completed_names": len(unknown_completed_names),
            "skipped_missing_url": len(skipped_missing_url),
            "change_count": len(changes),
            "counts": counts,
        },
        "unknown_completed_names": unknown_completed_names,
        "skipped_missing_url_names": skipped_missing_url,
        "changes": changes,
    }

    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Benchmark progress sync summary")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report to {rel_or_abs(report_path, cwd)}")
    if not args.dry_run:
        print(f"Updated manifest: {rel_or_abs(manifest_path, cwd)}")


if __name__ == "__main__":
    main()
