#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check benchmark manifest taxonomy coverage.")
    parser.add_argument(
        "--taxonomy",
        default=str(Path("assets") / "models" / "municipal-taxonomy-v1.json"),
        help="Taxonomy JSON path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-coverage-report.json"),
        help="Coverage report output path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(file_path: Path):
    return json.loads(file_path.read_text(encoding="utf-8"))


def to_set(values) -> set[str]:
    if not isinstance(values, list):
        return set()
    out = set()
    for value in values:
        text = str(value or "").strip()
        if text:
            out.add(text)
    return out


def set_diff(left: set[str], right: set[str]) -> list[str]:
    return sorted([value for value in left if value not in right])


def get_manifest_labels(images: list[dict], only_ready: bool) -> set[str]:
    labels: set[str] = set()
    for entry in images:
        if not isinstance(entry, dict):
            continue
        is_ready = str(entry.get("status") or "").lower() == "ready"
        if only_ready and not is_ready:
            continue
        for field in ("expected_any", "expected_all"):
            for label in entry.get(field) if isinstance(entry.get(field), list) else []:
                text = str(label or "").strip()
                if text:
                    labels.add(text)
    return labels


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    taxonomy_path = Path(args.taxonomy).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()

    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)

    classes = taxonomy.get("vision_classes") if isinstance(taxonomy, dict) else []
    images = manifest.get("images") if isinstance(manifest, dict) else []
    classes = classes if isinstance(classes, list) else []
    images = images if isinstance(images, list) else []

    taxonomy_labels = to_set([record.get("canonical_label") for record in classes if isinstance(record, dict)])
    manifest_labels_all = get_manifest_labels(images, only_ready=False)
    manifest_labels_ready = get_manifest_labels(images, only_ready=True)

    missing_in_manifest = set_diff(taxonomy_labels, manifest_labels_all)
    missing_in_ready_only = set_diff(taxonomy_labels, manifest_labels_ready)
    unknown_manifest_labels = set_diff(manifest_labels_all, taxonomy_labels)

    total_entries = len(images)
    ready_entries = sum(1 for entry in images if isinstance(entry, dict) and str(entry.get("status") or "").lower() == "ready")
    todo_entries = sum(1 for entry in images if isinstance(entry, dict) and str(entry.get("status") or "").lower() == "todo")
    missing_url_entries = sum(1 for entry in images if isinstance(entry, dict) and not str(entry.get("url") or "").strip())

    coverage = {
        "taxonomy_label_count": len(taxonomy_labels),
        "manifest_label_count_all": len(manifest_labels_all),
        "manifest_label_count_ready": len(manifest_labels_ready),
        "coverage_all": round(len(manifest_labels_all) / len(taxonomy_labels), 4) if taxonomy_labels else 0,
        "coverage_ready": round(len(manifest_labels_ready) / len(taxonomy_labels), 4) if taxonomy_labels else 0,
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "taxonomy": rel_or_abs(taxonomy_path, cwd),
            "manifest": rel_or_abs(manifest_path, cwd),
        },
        "entries": {
            "total": total_entries,
            "ready": ready_entries,
            "todo": todo_entries,
            "missing_url": missing_url_entries,
        },
        "coverage": coverage,
        "gaps": {
            "taxonomy_labels_missing_in_manifest": missing_in_manifest,
            "taxonomy_labels_missing_in_ready_entries": missing_in_ready_only,
            "manifest_labels_not_in_taxonomy": unknown_manifest_labels,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Benchmark coverage summary")
    print(json.dumps({"entries": report["entries"], "coverage": report["coverage"]}, indent=2))
    print(f"Missing labels in ready entries: {len(missing_in_ready_only)}")
    print(f"Unknown manifest labels: {len(unknown_manifest_labels)}")
    print(f"Saved report to {rel_or_abs(out_path, cwd)}")


if __name__ == "__main__":
    main()
