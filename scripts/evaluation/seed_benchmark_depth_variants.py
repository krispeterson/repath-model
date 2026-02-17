#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed benchmark depth variants for labels below target ready depth.",
        usage=(
            "python3 scripts/evaluation/seed_benchmark_depth_variants.py "
            "[--manifest test/benchmarks/municipal-benchmark-manifest-v2.json] "
            "[--target-ready 3] [--max-new 200] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument("--target-ready", type=int, default=3, help="Target ready+todo entries per label.")
    parser.add_argument("--max-new", type=int, default=200, help="Maximum new rows to generate.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write manifest changes.")
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


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    return normalized.strip("-")[:100]


def increment(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def main() -> None:
    args = parse_args()
    args.target_ready = max(2, int(args.target_ready or 3))
    args.max_new = max(1, int(args.max_new or 200))

    cwd = Path.cwd()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    manifest = load_json(manifest_path)
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    ready_counts: dict[str, int] = {}
    todo_counts: dict[str, int] = {}
    labels = set()
    item_id_by_label: dict[str, str] = {}
    existing_names = set()

    for entry in images:
        if not isinstance(entry, dict):
            continue

        label = first_label(entry)
        name = str(entry.get("name") or "").strip()
        item_id = str(entry.get("item_id") or "").strip()
        status = str(entry.get("status") or "").strip().lower()

        if name:
            existing_names.add(name)
        if not label:
            continue

        labels.add(label)
        if item_id and label not in item_id_by_label:
            item_id_by_label[label] = item_id

        if status == "ready":
            increment(ready_counts, label)
        if status == "todo":
            increment(todo_counts, label)

    candidates = []
    for label in labels:
        ready = ready_counts.get(label, 0)
        todo = todo_counts.get(label, 0)
        deficit = max(0, args.target_ready - (ready + todo))
        if deficit > 0:
            candidates.append({"label": label, "ready": ready, "todo": todo, "deficit": deficit})

    candidates.sort(key=lambda row: (-row["deficit"], row["ready"], row["label"].lower()))

    generated = []
    for row in candidates:
        for offset in range(row["deficit"]):
            if len(generated) >= args.max_new:
                break
            base = f"todo_depth_{slugify(row['label'])}"
            suffix = row["todo"] + offset + 1
            name = f"{base}_v{suffix}"
            while name in existing_names:
                suffix += 1
                name = f"{base}_v{suffix}"
            existing_names.add(name)
            generated.append(
                {
                    "name": name,
                    "url": "",
                    "expected_any": [row["label"]],
                    "expected_all": [],
                    "item_id": item_id_by_label.get(row["label"], ""),
                    "required": False,
                    "status": "todo",
                    "notes": "Auto-generated depth expansion placeholder.",
                }
            )
        if len(generated) >= args.max_new:
            break

    if not args.dry_run and generated:
        manifest["images"] = [*images, *generated]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Benchmark depth variant seed summary")
    print(
        json.dumps(
            {
                "target_ready_per_label": args.target_ready,
                "labels_considered": len(labels),
                "labels_below_target": len(candidates),
                "generated_entries": len(generated),
                "max_new": args.max_new,
                "dry_run": bool(args.dry_run),
                "manifest": rel_or_abs(manifest_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
