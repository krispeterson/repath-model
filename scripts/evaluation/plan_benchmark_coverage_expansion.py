#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan benchmark coverage expansion to hit target ready count per label.")
    parser.add_argument(
        "--taxonomy",
        default=str(Path("assets") / "models" / "municipal-taxonomy-v1.json"),
        help="Taxonomy JSON path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest path.",
    )
    parser.add_argument("--target-ready", type=int, default=3, help="Target ready rows per taxonomy label.")
    parser.add_argument("--max-rows", type=int, default=200, help="Maximum rows to emit into expansion template.")
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-coverage-expansion-report.json"),
        help="Output report path.",
    )
    parser.add_argument(
        "--template-out",
        default=str(Path("test") / "benchmarks" / "benchmark-coverage-expansion-template.csv"),
        help="Output template CSV path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_label(entry: dict) -> str:
    expected_any = entry.get("expected_any") if isinstance(entry, dict) else []
    expected_all = entry.get("expected_all") if isinstance(entry, dict) else []
    expected_any = expected_any if isinstance(expected_any, list) else []
    expected_all = expected_all if isinstance(expected_all, list) else []
    if expected_any:
        return str(expected_any[0] or "").strip()
    if expected_all:
        return str(expected_all[0] or "").strip()
    return ""


def write_csv(path: Path, rows: list[dict]) -> None:
    header = [
        "name",
        "url",
        "item_id",
        "canonical_label",
        "current_ready_count",
        "target_ready_count",
        "needed_for_target",
        "notes",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def ensure_list_entry(mapping: dict[str, list], key: str) -> list:
    if key not in mapping:
        mapping[key] = []
    return mapping[key]


def main() -> None:
    args = parse_args()
    if args.target_ready < 1:
        args.target_ready = 3
    if args.max_rows < 1:
        args.max_rows = 200

    cwd = Path.cwd()
    taxonomy_path = Path(args.taxonomy).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()
    template_path = Path(args.template_out).resolve()

    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)

    classes = taxonomy.get("vision_classes") if isinstance(taxonomy, dict) else []
    classes = classes if isinstance(classes, list) else []
    class_labels = sorted(
        [
            str(entry.get("canonical_label") or "").strip()
            for entry in classes
            if isinstance(entry, dict) and str(entry.get("canonical_label") or "").strip()
        ]
    )

    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    class_label_set = set(class_labels)
    ready_counts: dict[str, int] = {}
    todo_by_label: dict[str, list] = {}
    unknown_ready_labels: dict[str, int] = {}
    unknown_todo_labels: dict[str, int] = {}

    for entry in images:
        if not isinstance(entry, dict):
            continue
        label = first_label(entry)
        if not label:
            continue

        status = str(entry.get("status") or "").lower()
        if status == "ready":
            increment(ready_counts, label)
            if label not in class_label_set:
                increment(unknown_ready_labels, label)
            continue

        if status == "todo":
            ensure_list_entry(todo_by_label, label).append(entry)
            if label not in class_label_set:
                increment(unknown_todo_labels, label)

    under_target = []
    for label in class_labels:
        ready = ready_counts.get(label, 0)
        deficit = max(0, args.target_ready - ready)
        todo = len(todo_by_label.get(label, []))
        row = {
            "canonical_label": label,
            "ready_count": ready,
            "target_ready_count": args.target_ready,
            "deficit": deficit,
            "todo_candidates": todo,
        }
        if deficit > 0:
            under_target.append(row)

    under_target.sort(key=lambda row: (-row["deficit"], row["ready_count"], row["canonical_label"]))

    template_rows = []
    for row in under_target:
        candidates = (todo_by_label.get(row["canonical_label"], []))[: row["deficit"]]
        for entry in candidates:
            if len(template_rows) >= args.max_rows:
                break
            template_rows.append(
                {
                    "name": str(entry.get("name") or "").strip(),
                    "url": "",
                    "item_id": str(entry.get("item_id") or "").strip(),
                    "canonical_label": row["canonical_label"],
                    "current_ready_count": row["ready_count"],
                    "target_ready_count": row["target_ready_count"],
                    "needed_for_target": row["deficit"],
                    "notes": "coverage-expansion",
                }
            )
        if len(template_rows) >= args.max_rows:
            break

    without_candidates = [
        {
            "canonical_label": row["canonical_label"],
            "ready_count": row["ready_count"],
            "deficit": row["deficit"],
        }
        for row in under_target
        if row["todo_candidates"] == 0
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "taxonomy": rel_or_abs(taxonomy_path, cwd),
            "manifest": rel_or_abs(manifest_path, cwd),
            "target_ready_per_label": args.target_ready,
            "max_rows": args.max_rows,
        },
        "summary": {
            "taxonomy_label_count": len(class_labels),
            "labels_under_target": len(under_target),
            "labels_with_zero_ready": len([row for row in under_target if row["ready_count"] == 0]),
            "labels_without_todo_candidates": len(without_candidates),
            "template_rows": len(template_rows),
        },
        "labels_under_target": under_target,
        "labels_without_todo_candidates": without_candidates,
        "unknown_labels": {
            "ready": sorted(unknown_ready_labels.keys()),
            "todo": sorted(unknown_todo_labels.keys()),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(template_path, template_rows)

    print("Benchmark coverage expansion plan generated")
    print(
        json.dumps(
            {
                "labels_under_target": report["summary"]["labels_under_target"],
                "labels_with_zero_ready": report["summary"]["labels_with_zero_ready"],
                "labels_without_todo_candidates": report["summary"]["labels_without_todo_candidates"],
                "template_rows": report["summary"]["template_rows"],
                "report": rel_or_abs(out_path, cwd),
                "template": rel_or_abs(template_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
