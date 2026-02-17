#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


DEFAULT_QUERIES = [
    "forest trail",
    "mountain landscape",
    "city skyline",
    "street traffic",
    "living room interior",
    "kitchen interior",
    "office workspace",
    "dog park",
    "cat indoor",
    "soccer field",
    "playground",
    "beach coastline",
    "snowy road",
    "night city",
    "flowers garden",
    "river water",
    "airplane sky",
    "train station",
    "people crowd",
    "bird wildlife",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed negative benchmark manifest entries.",
        usage=(
            "python3 scripts/evaluation/seed_negative_benchmark_entries.py "
            "[--manifest test/benchmarks/municipal-benchmark-manifest-v2.json] "
            "[--count 20] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument("--count", type=int, default=20, help="Target number of negative entries.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write manifest changes.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_negative(entry: dict) -> bool:
    expected_any = entry.get("expected_any") if isinstance(entry, dict) else []
    expected_all = entry.get("expected_all") if isinstance(entry, dict) else []
    expected_any = expected_any if isinstance(expected_any, list) else []
    expected_all = expected_all if isinstance(expected_all, list) else []
    return len(expected_any) == 0 and len(expected_all) == 0


def to_name(query: str, index: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", str(query or "").lower()).strip("-")[:60]
    return f"todo_negative_{base}_{index}"


def main() -> None:
    args = parse_args()
    args.count = max(1, int(args.count or 20))

    cwd = Path.cwd()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    existing_names = {
        str((entry or {}).get("name") or "").strip()
        for entry in images
        if isinstance(entry, dict) and str((entry or {}).get("name") or "").strip()
    }

    current_negatives = sum(1 for entry in images if isinstance(entry, dict) and is_negative(entry))
    needed = max(0, args.count - current_negatives)

    generated = []
    for query in DEFAULT_QUERIES:
        if len(generated) >= needed:
            break
        suffix = 1
        name = to_name(query, suffix)
        while name in existing_names:
            suffix += 1
            name = to_name(query, suffix)
        existing_names.add(name)
        generated.append(
            {
                "name": name,
                "url": "",
                "expected_any": [],
                "expected_all": [],
                "required": False,
                "status": "todo",
                "notes": f"Negative benchmark seed. query_hint={query}",
            }
        )

    if not args.dry_run and generated:
        manifest["images"] = [*images, *generated]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Negative benchmark seed summary")
    print(
        json.dumps(
            {
                "current_negative_entries": current_negatives,
                "target_negative_entries": args.count,
                "generated_entries": len(generated),
                "dry_run": bool(args.dry_run),
                "manifest": rel_or_abs(manifest_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
