#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build benchmark v2 manifest from taxonomy + seed manifest.",
        usage=(
            "python3 scripts/evaluation/build_benchmark_manifest.py "
            "[--taxonomy assets/models/municipal-taxonomy-v1.json] "
            "[--seed test/benchmarks/benchmark-manifest.seed.json] "
            "[--out test/benchmarks/municipal-benchmark-manifest-v2.json]"
        ),
    )
    parser.add_argument(
        "--taxonomy",
        default=str(Path("assets") / "models" / "municipal-taxonomy-v1.json"),
        help="Taxonomy JSON path.",
    )
    parser.add_argument(
        "--seed",
        default=str(Path("test") / "benchmarks" / "benchmark-manifest.seed.json"),
        help="Seed benchmark manifest JSON path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Output benchmark manifest JSON path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    return normalized.strip("-")[:80]


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def index_seed_by_label(seed_manifest: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    images = as_list(seed_manifest.get("images") if isinstance(seed_manifest, dict) else [])

    for entry in images:
        if not isinstance(entry, dict):
            continue

        labels: list[str] = []
        for label in as_list(entry.get("expected_any")) + as_list(entry.get("expected_all")):
            text = str(label or "").strip()
            if text:
                labels.append(text)

        for label in labels:
            out.setdefault(label, []).append(entry)

    return out


def make_placeholder_entry(canonical_label: str, item_id: str) -> dict:
    return {
        "name": f"todo_{slugify(canonical_label) or item_id}",
        "url": "",
        "expected_any": [canonical_label],
        "expected_all": [],
        "item_id": item_id,
        "required": True,
        "status": "todo",
        "notes": "Add at least one representative photo for this item variation.",
    }


def make_negative_entry(name: str, notes: str) -> dict:
    return {
        "name": name,
        "url": "",
        "expected_any": [],
        "expected_all": [],
        "required": True,
        "status": "todo",
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    taxonomy_path = Path(args.taxonomy).resolve()
    seed_path = Path(args.seed).resolve()
    out_path = Path(args.out).resolve()

    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")
    if not seed_path.exists():
        raise SystemExit(f"Seed manifest not found: {seed_path}")

    taxonomy = load_json(taxonomy_path)
    seed_manifest = load_json(seed_path)

    seed_by_label = index_seed_by_label(seed_manifest)
    classes = as_list(taxonomy.get("vision_classes") if isinstance(taxonomy, dict) else [])
    images = []

    for record in classes:
        if not isinstance(record, dict):
            continue

        label = str(record.get("canonical_label") or "").strip()
        item_id = str(record.get("item_id") or "").strip()
        if not label or not item_id:
            continue

        seed_entries = seed_by_label.get(label, [])
        if seed_entries:
            for index, entry in enumerate(seed_entries):
                name = str(entry.get("name") or "").strip() or f"seed_{slugify(label)}"
                images.append(
                    {
                        "name": name if index == 0 else f"{name}_{index + 1}",
                        "url": str(entry.get("url") or ""),
                        "expected_any": as_list(entry.get("expected_any")) or [label],
                        "expected_all": as_list(entry.get("expected_all")),
                        "item_id": item_id,
                        "required": True,
                        "status": "ready" if str(entry.get("url") or "").strip() else "todo",
                        "notes": (
                            "Seed image from v1 benchmark manifest."
                            if str(entry.get("url") or "").strip()
                            else "Missing URL in seed entry."
                        ),
                    }
                )
            continue

        images.append(make_placeholder_entry(label, item_id))

    images.extend(
        [
            make_negative_entry(
                "todo_negative_people_scene",
                "Add cluttered scenes that should produce no detections.",
            ),
            make_negative_entry(
                "todo_negative_street_scene",
                "Add outdoor photos with no recyclable target item.",
            ),
            make_negative_entry(
                "todo_negative_pet_or_toy_scene",
                "Add household objects likely to cause false positives.",
            ),
        ]
    )

    output = {
        "name": "municipal-full-taxonomy-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "taxonomy": rel_or_abs(taxonomy_path, cwd),
            "seed_manifest": rel_or_abs(seed_path, cwd),
        },
        "images": images,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    ready_count = sum(1 for entry in images if str(entry.get("status") or "") == "ready")
    todo_count = len(images) - ready_count
    print(f"Generated benchmark manifest with {len(images)} entries ({ready_count} ready, {todo_count} todo).")
    print(f"- {rel_or_abs(out_path, cwd)}")


if __name__ == "__main__":
    main()
