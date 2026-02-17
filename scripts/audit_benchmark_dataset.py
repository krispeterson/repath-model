#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit benchmark dataset quality, balance, and coverage signals.")
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest path.",
    )
    parser.add_argument(
        "--taxonomy",
        default=str(Path("assets") / "models" / "municipal-taxonomy-v1.json"),
        help="Taxonomy path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-dataset-audit.json"),
        help="Audit report output path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def median(values: list[int]) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return float(ordered[mid])


def build_taxonomy_index(taxonomy: dict) -> dict[str, dict]:
    rows = taxonomy.get("vision_classes") if isinstance(taxonomy, dict) else []
    rows = rows if isinstance(rows, list) else []

    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("canonical_label") or "").strip()
        if label:
            out[label] = row
    return out


def label_for_entry(entry: dict) -> str:
    expected_any = entry.get("expected_any") if isinstance(entry, dict) else []
    expected_all = entry.get("expected_all") if isinstance(entry, dict) else []
    expected_any = expected_any if isinstance(expected_any, list) else []
    expected_all = expected_all if isinstance(expected_all, list) else []

    if expected_any:
        return str(expected_any[0] or "").strip()
    if expected_all:
        return str(expected_all[0] or "").strip()
    return ""


def increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def to_sorted_entries(counter: dict[str, int]) -> list[dict]:
    keys = sorted(counter.keys(), key=lambda key: (-counter[key], key))
    return [{"key": key, "count": counter[key]} for key in keys]


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest_path = Path(args.manifest).resolve()
    taxonomy_path = Path(args.taxonomy).resolve()
    out_path = Path(args.out).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")
    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")

    manifest = load_json(manifest_path)
    taxonomy = load_json(taxonomy_path)

    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    taxonomy_by_label = build_taxonomy_index(taxonomy)

    ready_entries = [
        entry
        for entry in images
        if isinstance(entry, dict) and str(entry.get("status") or "").lower() == "ready"
    ]
    todo_entries = [
        entry
        for entry in images
        if isinstance(entry, dict) and str(entry.get("status") or "").lower() == "todo"
    ]
    negative_entries = [
        entry
        for entry in images
        if isinstance(entry, dict)
        and len(entry.get("expected_any") if isinstance(entry.get("expected_any"), list) else []) == 0
        and len(entry.get("expected_all") if isinstance(entry.get("expected_all"), list) else []) == 0
    ]

    seen_names: dict[str, int] = {}
    seen_urls: dict[str, int] = {}

    for entry in images:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        url = str(entry.get("url") or "").strip()
        if name:
            seen_names[name] = seen_names.get(name, 0) + 1
        if url:
            seen_urls[url] = seen_urls.get(url, 0) + 1

    duplicate_name_map = {name: count for name, count in seen_names.items() if count > 1}
    duplicate_url_map = {url: count for url, count in seen_urls.items() if count > 1}

    missing_url_ready = [entry for entry in ready_entries if not str(entry.get("url") or "").strip()]
    missing_url_total = [entry for entry in images if isinstance(entry, dict) and not str(entry.get("url") or "").strip()]

    class_counts_ready: dict[str, int] = {}
    class_counts_total: dict[str, int] = {}
    outcome_counts_ready: dict[str, int] = {}
    outcome_counts_total: dict[str, int] = {}
    unknown_labels: list[str] = []

    for entry in images:
        if not isinstance(entry, dict):
            continue
        label = label_for_entry(entry)
        if not label:
            continue

        increment(class_counts_total, label)
        row = taxonomy_by_label.get(label)
        if row:
            primary = str(row.get("primary_outcome") or "").strip()
            if primary:
                increment(outcome_counts_total, primary)
        else:
            unknown_labels.append(label)

        if str(entry.get("status") or "").lower() == "ready":
            increment(class_counts_ready, label)
            if row:
                primary = str(row.get("primary_outcome") or "").strip()
                if primary:
                    increment(outcome_counts_ready, primary)

    ready_values = list(class_counts_ready.values())
    total_values = list(class_counts_total.values())

    balance = {
        "ready": {
            "class_count": len(ready_values),
            "min_samples_per_class": min(ready_values) if ready_values else 0,
            "median_samples_per_class": median(ready_values),
            "max_samples_per_class": max(ready_values) if ready_values else 0,
        },
        "total": {
            "class_count": len(total_values),
            "min_samples_per_class": min(total_values) if total_values else 0,
            "median_samples_per_class": median(total_values),
            "max_samples_per_class": max(total_values) if total_values else 0,
        },
    }

    recommendations: list[str] = []
    if len(ready_entries) < 100:
        recommendations.append("Increase ready image count to at least 100 before first training round.")
    if balance["ready"]["median_samples_per_class"] < 3:
        recommendations.append("Raise median ready samples per class to >=3 to reduce collapse on rare labels.")
    if len(missing_url_ready) > 0:
        recommendations.append("Fix ready entries with empty URLs before training/evaluation.")
    if len(negative_entries) < 20:
        recommendations.append("Add more negative/no-target images to control false positives.")
    if len(duplicate_url_map) > 0:
        recommendations.append("De-duplicate repeated image URLs to reduce overfitting to identical scenes.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": rel_or_abs(manifest_path, cwd),
            "taxonomy": rel_or_abs(taxonomy_path, cwd),
        },
        "counts": {
            "total_entries": len(images),
            "ready_entries": len(ready_entries),
            "todo_entries": len(todo_entries),
            "negative_entries": len(negative_entries),
            "missing_url_total": len(missing_url_total),
            "missing_url_ready": len(missing_url_ready),
        },
        "quality_checks": {
            "duplicate_name_count": len(duplicate_name_map),
            "duplicate_url_count": len(duplicate_url_map),
            "unknown_label_count": len(unique(unknown_labels)),
        },
        "class_balance": balance,
        "distributions": {
            "ready_outcomes": to_sorted_entries(outcome_counts_ready),
            "total_outcomes": to_sorted_entries(outcome_counts_total),
            "ready_classes_top25": to_sorted_entries(class_counts_ready)[:25],
            "total_classes_top25": to_sorted_entries(class_counts_total)[:25],
        },
        "duplicates": {
            "names": duplicate_name_map,
            "urls": duplicate_url_map,
        },
        "unknown_labels": sorted(unique(unknown_labels)),
        "recommendations": recommendations,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Benchmark dataset audit summary")
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "class_balance": report["class_balance"],
                "quality_checks": report["quality_checks"],
                "recommendations": report["recommendations"],
            },
            indent=2,
        )
    )
    print(f"Saved report to {rel_or_abs(out_path, cwd)}")


if __name__ == "__main__":
    main()
