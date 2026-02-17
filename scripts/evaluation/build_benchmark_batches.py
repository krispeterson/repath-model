#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark labeling batches from priority report.")
    parser.add_argument(
        "--priority",
        default=str(Path("test") / "benchmarks" / "benchmark-priority-report.json"),
        help="Priority report JSON path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("test") / "benchmarks"),
        help="Output directory for batch files.",
    )
    parser.add_argument("--urgent", type=int, default=30, help="Urgent batch limit.")
    parser.add_argument("--high", type=int, default=50, help="High batch limit.")
    parser.add_argument("--medium", type=int, default=80, help="Medium batch limit.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_manifest_by_name(manifest: dict) -> dict[str, dict]:
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []
    out = {}
    for entry in images:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if name:
            out[name] = entry
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    headers = [
        "batch",
        "name",
        "item_id",
        "canonical_label",
        "primary_outcome",
        "priority_score",
        "priority_band",
        "url",
        "status",
        "required",
        "reasons",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["reasons"] = " | ".join(row.get("reasons") or []) if isinstance(row.get("reasons"), list) else ""
            writer.writerow({header: csv_row.get(header, "") for header in headers})


def select_batch_rows(candidates: list[dict], manifest_by_name: dict[str, dict], band: str, limit: int) -> list[dict]:
    rows = []
    for candidate in candidates:
        if len(rows) >= limit:
            break
        if str(candidate.get("priority_band") or "") != band:
            continue

        entry = manifest_by_name.get(str(candidate.get("name") or ""), {})
        rows.append(
            {
                "batch": band,
                "name": candidate.get("name"),
                "item_id": candidate.get("item_id") or "",
                "canonical_label": candidate.get("canonical_label") or "",
                "primary_outcome": candidate.get("primary_outcome") or "",
                "priority_score": float(candidate.get("priority_score") or 0),
                "priority_band": candidate.get("priority_band") or "",
                "url": entry.get("url", "") if isinstance(entry, dict) else "",
                "status": entry.get("status", "") if isinstance(entry, dict) else "",
                "required": entry.get("required", "") if isinstance(entry, dict) else "",
                "reasons": candidate.get("reasons") or [],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    for key in ("urgent", "high", "medium"):
        value = getattr(args, key)
        if value is None or value < 0:
            setattr(args, key, 0)

    cwd = Path.cwd()
    priority_path = Path(args.priority).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not priority_path.exists():
        raise SystemExit(f"Priority report not found: {priority_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    priority = load_json(priority_path)
    manifest = load_json(manifest_path)

    candidates = priority.get("all_candidates") if isinstance(priority, dict) else []
    candidates = candidates if isinstance(candidates, list) else []
    manifest_by_name = index_manifest_by_name(manifest)

    urgent_rows = select_batch_rows(candidates, manifest_by_name, "urgent", args.urgent)
    high_rows = select_batch_rows(candidates, manifest_by_name, "high", args.high)
    medium_rows = select_batch_rows(candidates, manifest_by_name, "medium", args.medium)
    combined_rows = [*urgent_rows, *high_rows, *medium_rows]

    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "priority": rel_or_abs(priority_path, cwd),
            "manifest": rel_or_abs(manifest_path, cwd),
        },
        "config": {
            "urgent_limit": args.urgent,
            "high_limit": args.high,
            "medium_limit": args.medium,
        },
        "summary": {
            "urgent_count": len(urgent_rows),
            "high_count": len(high_rows),
            "medium_count": len(medium_rows),
            "total_selected": len(combined_rows),
        },
        "batches": {
            "urgent": urgent_rows,
            "high": high_rows,
            "medium": medium_rows,
            "combined": combined_rows,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark-labeling-batches.json"
    urgent_csv = out_dir / "benchmark-labeling-batch-urgent.csv"
    high_csv = out_dir / "benchmark-labeling-batch-high.csv"
    medium_csv = out_dir / "benchmark-labeling-batch-medium.csv"
    combined_csv = out_dir / "benchmark-labeling-batch-combined.csv"

    json_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    write_csv(urgent_csv, urgent_rows)
    write_csv(high_csv, high_rows)
    write_csv(medium_csv, medium_rows)
    write_csv(combined_csv, combined_rows)

    print("Benchmark labeling batches generated")
    print(
        json.dumps(
            {
                "summary": plan["summary"],
                "files": {
                    "json": rel_or_abs(json_path, cwd),
                    "urgent_csv": rel_or_abs(urgent_csv, cwd),
                    "high_csv": rel_or_abs(high_csv, cwd),
                    "medium_csv": rel_or_abs(medium_csv, cwd),
                    "combined_csv": rel_or_abs(combined_csv, cwd),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
