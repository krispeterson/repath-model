#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path

CSV_COLUMNS = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def usage() -> str:
    return (
        "python3 scripts/training/expand_retraining_positives_from_kaggle.py "
        "[--input test/benchmarks/benchmark-labeled.csv] "
        "[--priority-csv test/benchmarks/latest-results.candidate.priority.csv] "
        "[--kaggle-dir /path/to/kaggle/images/images] "
        "[--cache-dir test/benchmarks/images/retraining-positives] "
        "[--per-label 2] [--top-labels 6] [--labels \"Tin Can,Cardboard\"] "
        "[--holdout-manifest test/benchmarks/benchmark-manifest.supported-holdout.json] "
        "[--out test/benchmarks/benchmark-labeled.csv] [--dry-run]"
    )


def parse_labels(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
    )
    parser.add_argument(
        "--priority-csv",
        default=str(Path("test") / "benchmarks" / "latest-results.candidate.priority.csv"),
    )
    parser.add_argument("--kaggle-dir", default=str(Path("")))
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images" / "retraining-positives"),
    )
    parser.add_argument("--per-label", type=int, default=2)
    parser.add_argument("--top-labels", type=int, default=6)
    parser.add_argument("--labels", type=parse_labels, default=[])
    parser.add_argument(
        "--holdout-manifest",
        default=str(Path("test") / "benchmarks" / "benchmark-manifest.supported-holdout.json"),
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not isinstance(args.per_label, int) or args.per_label < 1:
        args.per_label = 2
    if not isinstance(args.top_labels, int) or args.top_labels < 1:
        args.top_labels = 6

    return args


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")


def rel_or_abs(path: Path) -> str:
    try:
        return rel(path)
    except Exception:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({column: str((row or {}).get(column, "")).strip() for column in CSV_COLUMNS})
        return rows


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: str((row or {}).get(column, "")) for column in CSV_COLUMNS})


def read_priority_labels(priority_path: Path, top_labels: int) -> list[str]:
    if not priority_path.exists():
        return []

    with priority_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            label = str((row or {}).get("label", "")).strip()
            if not label:
                continue
            rows.append(
                {
                    "rank": to_num((row or {}).get("rank")),
                    "label": label,
                    "priority_score": to_num((row or {}).get("priority_score")),
                    "miss_count": to_num((row or {}).get("miss_count")),
                    "recommended_action": str((row or {}).get("recommended_action", "")).strip(),
                }
            )

    rows = [
        row
        for row in rows
        if row["recommended_action"] == "collect_more_positives" and row["miss_count"] > 0
    ]

    def sort_key(row: dict) -> tuple:
        rank = row.get("rank") or 0
        if rank > 0:
            return (0, rank, 0)
        return (1, 0, -(row.get("priority_score") or 0))

    rows.sort(key=sort_key)
    return [row["label"] for row in rows[:top_labels]]


def to_num(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug[:80]


def kaggle_folder_mapping() -> dict[str, list[str]]:
    return {
        "Aluminum Can": ["aluminum_soda_cans", "aluminum_food_cans"],
        "Tin Can": ["steel_food_cans", "aluminum_food_cans"],
        "Cardboard": ["cardboard_boxes", "cardboard_packaging"],
        "Paperboard": ["cardboard_packaging", "cardboard_boxes"],
        "Vitamin or Prescription Bottle": [
            "plastic_water_bottles",
            "plastic_soda_bottles",
            "plastic_detergent_bottles",
        ],
        "Aluminum Foil": ["aluminum_foil"],
        "Paper Egg Carton": ["egg_cartons"],
        "Pizza Box": ["pizza_boxes"],
    }


def resolve_kaggle_dir(raw: str) -> Path:
    if raw:
        return Path(raw).resolve()

    env_value = str(os.environ.get("KAGGLE_WASTE_DIR", "")).strip()
    if env_value:
        return Path(env_value).resolve()

    candidates = [
        Path("ml") / "artifacts" / "datasets" / "kaggle-household-waste" / "images" / "images",
        Path("..") / "Kaggle Household Waste Images" / "images" / "images",
    ]
    for candidate in candidates:
        full = candidate.resolve()
        if full.exists():
            return full
    return Path("")


def list_images(root_dir: Path) -> list[Path]:
    if not root_dir.exists():
        return []

    matches: list[Path] = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            matches.append(path)

    matches.sort(key=lambda p: str(p))
    return matches


def read_used_source_keys(rows: list[dict]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        source = str((row or {}).get("source", "")).strip()
        if source != "kaggle_household_waste_images":
            continue
        notes = str((row or {}).get("notes", ""))
        folder_match = re.search(r"(?:^|;\s*)folder=([^;]+)", notes, flags=re.IGNORECASE)
        image_match = re.search(r"(?:^|;\s*)source_image=([^;]+)", notes, flags=re.IGNORECASE)
        if not folder_match or not image_match:
            continue
        folder = str(folder_match.group(1) or "").strip()
        image = str(image_match.group(1) or "").strip()
        if folder and image:
            keys.add(f"{folder}/{image}")
    return keys


def read_holdout_keys(manifest_path: Path) -> set[str]:
    keys: set[str] = set()
    if not manifest_path.exists():
        return keys

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else None
    for row in images if isinstance(images, list) else []:
        notes = str((row or {}).get("notes", ""))
        folder_match = re.search(r"(?:^|;\s*)folder=([^;]+)", notes, flags=re.IGNORECASE)
        image_match = re.search(r"(?:^|;\s*)source_image=([^;]+)", notes, flags=re.IGNORECASE)
        if not folder_match or not image_match:
            continue
        folder = str(folder_match.group(1) or "").strip()
        image = str(image_match.group(1) or "").strip()
        if folder and image:
            keys.add(f"{folder}/{image}")

    return keys


def extract_max_variant(rows: list[dict], slug: str) -> int:
    pattern = re.compile(rf"^retrain_positive_{re.escape(slug)}_v(\d+)$")
    max_variant = 0
    for row in rows:
        name = str((row or {}).get("name", ""))
        match = pattern.match(name)
        if not match:
            continue
        try:
            value = int(match.group(1))
        except ValueError:
            continue
        if value > max_variant:
            max_variant = value
    return max_variant


def extension_for(path: Path) -> str:
    ext = str(path.suffix or "").lower()
    return ext if ext else ".jpg"


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    priority_path = Path(args.priority_csv).resolve()
    kaggle_dir = resolve_kaggle_dir(args.kaggle_dir)
    cache_dir = Path(args.cache_dir).resolve()
    holdout_path = Path(args.holdout_manifest).resolve()
    out_path = Path(args.out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    if not kaggle_dir.exists():
        raise SystemExit("Kaggle image dir not found. Set KAGGLE_WASTE_DIR or pass --kaggle-dir.")

    rows = read_csv_rows(input_path)
    mapping = kaggle_folder_mapping()
    holdout_keys = read_holdout_keys(holdout_path)
    used_source_keys = read_used_source_keys(rows)
    labels = args.labels if args.labels else read_priority_labels(priority_path, args.top_labels)

    added_rows: list[dict] = []
    label_summary: list[dict] = []
    skipped_labels: list[dict] = []

    for label in labels:
        folders = mapping.get(label) if isinstance(mapping.get(label), list) else []
        if not folders:
            skipped_labels.append({"label": label, "reason": "no_kaggle_mapping"})
            continue

        slug = slugify(label)
        if not slug:
            skipped_labels.append({"label": label, "reason": "invalid_slug"})
            continue

        candidates: list[dict] = []
        for folder in folders:
            folder_path = kaggle_dir / folder
            for image_path in list_images(folder_path):
                image_name = image_path.name
                key = f"{folder}/{image_name}"
                candidates.append(
                    {
                        "folder": folder,
                        "image_name": image_name,
                        "image_path": image_path,
                        "key": key,
                    }
                )

        next_variant = extract_max_variant(rows, slug) + 1
        added = 0

        for candidate in candidates:
            if added >= args.per_label:
                break
            if candidate["key"] in used_source_keys:
                continue
            if candidate["key"] in holdout_keys:
                continue

            name = f"retrain_positive_{slug}_v{next_variant}"
            ext = extension_for(candidate["image_path"])
            target_path = cache_dir / f"{name}{ext}"
            url = rel(target_path)

            if not args.dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate["image_path"], target_path)

            row = {
                "name": name,
                "url": url,
                "item_id": f"retrain-{slug}-v{next_variant}",
                "canonical_label": label,
                "source": "kaggle_household_waste_images",
                "notes": (
                    f"folder={candidate['folder']}; source_image={candidate['image_name']}; "
                    "selected_from=retraining_priority"
                ),
            }

            rows.append(row)
            added_rows.append(row)
            used_source_keys.add(candidate["key"])
            added += 1
            next_variant += 1

        label_summary.append(
            {
                "label": label,
                "requested": args.per_label,
                "added": added,
                "available_candidates": len(candidates),
            }
        )

    rows.sort(key=lambda row: str((row or {}).get("name", "")))

    if not args.dry_run:
        write_csv_rows(out_path, rows)

    print("Retraining positive expansion summary")
    print(
        json.dumps(
            {
                "input": rel_or_abs(input_path),
                "priority_csv": rel_or_abs(priority_path),
                "kaggle_dir": rel_or_abs(kaggle_dir),
                "holdout_manifest_exists": holdout_path.exists(),
                "labels_selected": labels,
                "per_label_requested": args.per_label,
                "rows_added": len(added_rows),
                "label_summary": label_summary,
                "skipped_labels": skipped_labels,
                "output": rel_or_abs(out_path),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
