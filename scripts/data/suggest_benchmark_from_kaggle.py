#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest benchmark rows by mapping todo manifest labels to local Kaggle image folders.",
        usage=(
            "python3 scripts/data/suggest_benchmark_from_kaggle.py "
            "[--manifest test/benchmarks/municipal-benchmark-manifest-v2.json] "
            "[--kaggle-dir /path/to/kaggle/images/images] "
            "[--cache-dir test/benchmarks/images] "
            "[--out test/benchmarks/benchmark-labeled.kaggle.csv] "
            "[--merge-into test/benchmarks/benchmark-labeled.csv]"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument("--kaggle-dir", default=os.environ.get("KAGGLE_WASTE_DIR", ""), help="Kaggle images root dir.")
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images"),
        help="Cache destination for copied images.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.kaggle.csv"),
        help="Output CSV path for generated suggestions.",
    )
    parser.add_argument("--merge-into", default=None, help="Optional benchmark CSV to merge suggestions into.")
    return parser.parse_args()


def resolve_kaggle_dir(input_value: str) -> Path | None:
    if input_value:
        return Path(input_value).expanduser().resolve()

    for candidate in [
        Path("ml") / "artifacts" / "datasets" / "kaggle-household-waste" / "images" / "images",
        Path("..") / "Kaggle Household Waste Images" / "images" / "images",
    ]:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    return None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve())).replace(os.sep, "/")
    except ValueError:
        return str(path.resolve())


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def label_for_entry(entry: dict) -> str:
    expected_any = as_list(entry.get("expected_any") if isinstance(entry, dict) else [])
    if expected_any:
        return str(expected_any[0] or "").strip()
    expected_all = as_list(entry.get("expected_all") if isinstance(entry, dict) else [])
    if expected_all:
        return str(expected_all[0] or "").strip()
    return ""


def list_images(folder_path: Path) -> list[Path]:
    if not folder_path.exists():
        return []
    out = []
    for path in folder_path.rglob("*"):
        if path.is_file() and re.search(r"\.(jpg|jpeg|png|webp)$", path.name, flags=re.IGNORECASE):
            out.append(path)
    out.sort(key=lambda item: str(item).lower())
    return out


def build_image_pool(kaggle_dir: Path) -> dict[str, list[Path]]:
    pool = {}
    if not kaggle_dir.exists():
        return pool
    for folder in sorted(kaggle_dir.iterdir(), key=lambda item: item.name.lower()):
        if folder.is_dir():
            pool[folder.name] = list_images(folder)
    return pool


def get_mapping() -> dict[str, list[str]]:
    return {
        "Aluminum Can": ["aluminum_soda_cans", "aluminum_food_cans"],
        "Tin Can": ["steel_food_cans", "aluminum_food_cans"],
        "Empty Aerosol Can": ["aerosol_cans"],
        "Cardboard": ["cardboard_boxes", "cardboard_packaging"],
        "Waxed Cardboard": ["cardboard_packaging", "cardboard_boxes"],
        "Glass Bottle or Jar": ["glass_food_jars", "glass_beverage_bottles"],
        "Plastic Jug": ["plastic_detergent_bottles", "plastic_soda_bottles"],
        "Plastic Container": ["plastic_food_containers"],
        "Plastic Caps & Lids": ["plastic_cup_lids"],
        "Take Out Food Container": ["styrofoam_food_containers", "plastic_food_containers"],
        "Food Grade Styrofoam": ["styrofoam_food_containers", "styrofoam_cups"],
        "Packaging Styrofoam or Polystyrene Foam": ["styrofoam_food_containers", "styrofoam_cups"],
        "Coffee Grounds": ["coffee_grounds"],
        "Clothing and Fabric": ["clothing"],
        "Magazine": ["magazines"],
        "White Office Paper": ["office_paper"],
        "Paper Cup": ["paper_cups"],
        "Newspaper": ["newspaper"],
        "Plastic Shopping Bags": ["plastic_shopping_bags"],
        "Plastic Wrap": ["plastic_trash_bags"],
        "Tea Bags": ["tea_bags"],
    }


def sanitize_name(value: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()))[:120]


def extension_from_path(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    return ext if ext else ".jpg"


def pick_next_image(folders: list[str], pool: dict[str, list[Path]], used: set[str]):
    for folder in folders:
        images = pool.get(folder, [])
        for candidate in images:
            key = str(candidate)
            if key not in used:
                used.add(key)
                return {"image_path": candidate, "folder": folder}

    for folder in folders:
        images = pool.get(folder, [])
        if images:
            return {"image_path": images[0], "folder": folder}

    return None


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({column: str((row or {}).get(column, "")).strip() for column in HEADER})
        return rows


def merge_rows(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    by_name = {}
    for row in existing_rows:
        name = str(row.get("name") or "").strip()
        if name:
            by_name[name] = row
    for row in new_rows:
        name = str(row.get("name") or "").strip()
        if name:
            by_name[name] = row
    return [by_name[key] for key in sorted(by_name.keys(), key=lambda value: value.lower())]


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest_path = Path(args.manifest).resolve()
    kaggle_dir = resolve_kaggle_dir(args.kaggle_dir)
    cache_dir = Path(args.cache_dir).resolve()
    out_path = Path(args.out).resolve()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if kaggle_dir is None or not kaggle_dir.exists():
        raise SystemExit("Kaggle image dir not found. Set KAGGLE_WASTE_DIR or pass --kaggle-dir.")

    manifest = load_json(manifest_path)
    images = as_list(manifest.get("images") if isinstance(manifest, dict) else [])
    mapping = get_mapping()
    pool = build_image_pool(kaggle_dir)
    used_paths: set[str] = set()
    rows: list[dict] = []

    cache_dir.mkdir(parents=True, exist_ok=True)

    for entry in images:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").lower()
        if status != "todo":
            continue

        label = label_for_entry(entry)
        folders = mapping.get(label)
        if not folders:
            continue

        picked = pick_next_image(folders, pool, used_paths)
        if not picked:
            continue

        image_path = Path(picked["image_path"])
        ext = extension_from_path(image_path)
        out_file = cache_dir / f"{sanitize_name(str(entry.get('name') or 'sample')) or 'sample'}{ext}"
        if not out_file.exists():
            shutil.copyfile(image_path, out_file)

        rows.append(
            {
                "name": str(entry.get("name") or "").strip(),
                "url": rel_or_abs(out_file, cwd),
                "item_id": str(entry.get("item_id") or "").strip(),
                "canonical_label": label,
                "source": "kaggle_household_waste_images",
                "notes": f"folder={picked['folder']}",
            }
        )

    write_rows(out_path, rows)

    merged_count = None
    merged_into = None
    if args.merge_into:
        merge_path = Path(args.merge_into).resolve()
        merged = merge_rows(read_rows(merge_path), rows)
        write_rows(merge_path, merged)
        merged_count = len(merged)
        merged_into = rel_or_abs(merge_path, cwd)

    print("Kaggle benchmark suggestions generated")
    print(
        json.dumps(
            {
                "matched_rows": len(rows),
                "output": rel_or_abs(out_path, cwd),
                "merged_into": merged_into,
                "merged_row_count": merged_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
