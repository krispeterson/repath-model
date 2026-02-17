#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supported holdout manifest from model labels + source pools.")
    parser.add_argument("--labels", default="", help="Labels JSON path (defaults to latest candidate labels).")
    parser.add_argument(
        "--candidates-root",
        default=str(Path("ml") / "artifacts" / "models" / "candidates"),
        help="Candidate root directory.",
    )
    parser.add_argument("--kaggle-dir", default=os.environ.get("KAGGLE_WASTE_DIR", ""), help="Kaggle dataset root.")
    parser.add_argument(
        "--input-csv",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Benchmark labeled CSV for exclusions.",
    )
    parser.add_argument(
        "--manual-seed",
        default=str(Path("test") / "benchmarks" / "benchmark-supported-holdout-overrides.seed.json"),
        help="Manual overrides seed path.",
    )
    parser.add_argument(
        "--retraining-manifest",
        default=str(Path("ml") / "artifacts" / "retraining" / "retraining-manifest.json"),
        help="Retraining manifest used to exclude training URLs.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images" / "supported-holdout"),
        help="Cache directory for selected holdout images.",
    )
    parser.add_argument("--per-label", type=int, default=3, help="Requested holdout samples per label.")
    parser.add_argument("--no-download", action="store_true", help="Disable manual HTTP URL download.")
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-manifest.supported-holdout.json"),
        help="Output holdout manifest path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_kaggle_dir(input_value: str) -> Path | None:
    if input_value:
        path = Path(input_value).resolve()
        return path

    candidates = [
        Path("ml") / "artifacts" / "datasets" / "kaggle-household-waste" / "images" / "images",
        Path("..") / "Kaggle Household Waste Images" / "images" / "images",
    ]
    for candidate in candidates:
        full = candidate.resolve()
        if full.exists():
            return full
    return None


def resolve_latest_candidate_labels(candidates_root: str) -> Path | None:
    root = Path(candidates_root).resolve()
    if not root.exists() or not root.is_dir():
        return None

    dirs = [path for path in root.iterdir() if path.is_dir()]
    dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for directory in dirs:
        preferred = directory / "yolo-repath.labels.json"
        legacy = directory / "yolov8.labels.json"
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
    return None


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def extension_for_file(file_path: str | Path) -> str:
    ext = Path(str(file_path)).suffix.lower()
    return ext if ext else ".jpg"


def read_excluded_kaggle_keys(csv_path: Path) -> set[str]:
    keys: set[str] = set()
    if not csv_path.exists():
        return keys

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    if len(rows) < 2:
        return keys

    for line in rows[1:]:
        if not line.strip():
            continue
        cols = next(csv.reader([line]))
        source = str(cols[4] if len(cols) > 4 else "").strip()
        notes = str(cols[5] if len(cols) > 5 else "").strip()
        if source != "kaggle_household_waste_images":
            continue

        folder_match = re.search(r"(?:^|;\s*)folder=([^;]+)", notes, flags=re.IGNORECASE)
        image_match = re.search(r"(?:^|;\s*)source_image=([^;]+)", notes, flags=re.IGNORECASE)
        if not folder_match or not image_match:
            continue

        folder = str(folder_match.group(1) or "").strip()
        image = str(image_match.group(1) or "").strip()
        if folder and image:
            keys.add(f"{folder}/{image}")

    return keys


def load_manual_seed(seed_path: Path) -> dict:
    if not seed_path.exists():
        return {"seed_path": seed_path, "labels": {}}

    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    labels = payload.get("labels") if isinstance(payload, dict) else {}
    labels = labels if isinstance(labels, dict) else {}
    return {"seed_path": seed_path, "labels": labels}


def normalize_url_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def load_excluded_training_urls(manifest_path: Path) -> dict:
    urls: set[str] = set()
    if not manifest_path.exists():
        return {"manifest_path": manifest_path, "urls": urls}

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else []
    samples = samples if isinstance(samples, list) else []

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        image = normalize_url_for_compare(str(sample.get("image") or ""))
        if image:
            urls.add(image)

        notes = str(sample.get("notes") or "")
        source_match = re.search(r"(?:^|;\s*)source_url=([^;]+)", notes, flags=re.IGNORECASE)
        if source_match:
            source_url = normalize_url_for_compare(source_match.group(1))
            if source_url:
                urls.add(source_url)

    return {"manifest_path": manifest_path, "urls": urls}


def list_images(root_dir: Path) -> list[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []

    out = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if not re.search(r"\.(jpg|jpeg|png|webp)$", path.name, flags=re.IGNORECASE):
            continue
        out.append(path)
    out.sort(key=lambda item: str(item))
    return out


def get_kaggle_folder_mapping() -> dict[str, list[str]]:
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
        "Aluminum Foil": [],
        "Paper Egg Carton": [],
        "Pizza Box": [],
    }


def build_image_pool(kaggle_dir: Path, labels: list[str], mapping: dict[str, list[str]]) -> dict[str, list[dict]]:
    pool: dict[str, list[dict]] = {}

    for label in labels:
        folders = mapping.get(label, [])
        images: list[dict] = []
        for folder in folders:
            folder_path = kaggle_dir / folder
            for full_path in list_images(folder_path):
                rel = full_path.relative_to(kaggle_dir).as_posix()
                parts = rel.split("/")
                folder_key = f"{parts[0]}/{parts[1]}" if len(parts) >= 3 else parts[0]
                source_image = parts[-1]
                images.append(
                    {
                        "full_path": full_path,
                        "folder": folder_key,
                        "source_image": source_image,
                        "key": f"{folder_key}/{source_image}",
                    }
                )
        pool[label] = images

    return pool


def extension_from_url(value: str) -> str:
    match = re.search(r"\.([a-zA-Z0-9]{2,6})(?:[?#].*)?$", str(value or ""))
    if not match:
        return ".jpg"
    return f".{match.group(1).lower()}"


def is_http_url(value: str) -> bool:
    return bool(re.match(r"^https?://", str(value or ""), flags=re.IGNORECASE))


def download_url(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--max-time",
            "90",
            "--fail",
            url,
            "-o",
            str(out_file),
            "-sS",
        ],
        check=True,
    )


def copy_local(local_path: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, out_file)


def main() -> None:
    args = parse_args()
    if not args.per_label or args.per_label < 1:
        args.per_label = 3

    cwd = Path.cwd()

    labels_path = Path(args.labels).resolve() if args.labels else resolve_latest_candidate_labels(args.candidates_root)
    kaggle_dir = resolve_kaggle_dir(args.kaggle_dir)
    csv_path = Path(args.input_csv).resolve()
    manual_seed = load_manual_seed(Path(args.manual_seed).resolve())
    training_exclusions = load_excluded_training_urls(Path(args.retraining_manifest).resolve())
    cache_dir = Path(args.cache_dir).resolve()
    out_path = Path(args.out).resolve()

    if not labels_path or not labels_path.exists():
        raise SystemExit("Labels file not found. Pass --labels or create candidate labels first.")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise SystemExit("Labels file must be a JSON array.")
    labels = [str(label or "").strip() for label in labels if str(label or "").strip()]

    mapping = get_kaggle_folder_mapping()
    excluded = read_excluded_kaggle_keys(csv_path)
    excluded_training_urls = training_exclusions["urls"]
    selected_keys: set[str] = set()
    selected_manual_urls: set[str] = set()
    unsupported: list[dict] = []
    rows: list[dict] = []

    if not kaggle_dir or not kaggle_dir.exists():
        unsupported.append(
            {
                "reason": "kaggle_dir_not_found",
                "detail": "Set KAGGLE_WASTE_DIR or pass --kaggle-dir to build supported holdout entries.",
            }
        )

    pool = build_image_pool(kaggle_dir, labels, mapping) if kaggle_dir and kaggle_dir.exists() else {}

    cache_dir.mkdir(parents=True, exist_ok=True)

    for label in labels:
        candidates = pool.get(label, [])
        manual_urls = manual_seed["labels"].get(label) if isinstance(manual_seed["labels"], dict) else []
        manual_urls = manual_urls if isinstance(manual_urls, list) else []

        selected_count = 0
        for candidate in candidates:
            if selected_count >= args.per_label:
                break
            if candidate["key"] in excluded:
                continue
            if candidate["key"] in selected_keys:
                continue

            idx = selected_count + 1
            slug = slugify(label)
            entry_name = f"holdout_{slug}_kaggle_v{idx}"
            out_file = cache_dir / f"{entry_name}{extension_for_file(candidate['full_path'])}"
            if not out_file.exists():
                shutil.copyfile(candidate["full_path"], out_file)

            rows.append(
                {
                    "name": entry_name,
                    "url": rel_or_abs(out_file, cwd),
                    "expected_any": [label],
                    "expected_all": [],
                    "item_id": f"holdout-{slug}-v{idx}",
                    "required": False,
                    "status": "ready",
                    "notes": (
                        "supported-holdout; source=kaggle_household_waste_images; "
                        f"folder={candidate['folder']}; source_image={candidate['source_image']}"
                    ),
                }
            )

            selected_keys.add(candidate["key"])
            selected_count += 1

        if selected_count < args.per_label and manual_urls:
            manual_index = 1
            for raw_url in manual_urls:
                if selected_count >= args.per_label:
                    break
                url = str(raw_url or "").strip()
                if not url:
                    continue
                if url in selected_manual_urls:
                    continue
                if normalize_url_for_compare(url) in excluded_training_urls:
                    continue

                slug = slugify(label)
                entry_name = f"holdout_{slug}_manual_v{manual_index}"
                ext = extension_from_url(url) if is_http_url(url) else extension_for_file(url)
                out_file = cache_dir / f"{entry_name}{ext}"

                try:
                    if not out_file.exists():
                        if is_http_url(url):
                            if args.no_download:
                                continue
                            download_url(url, out_file)
                        else:
                            local_path = Path(url).resolve()
                            if not local_path.exists():
                                continue
                            copy_local(local_path, out_file)

                    rows.append(
                        {
                            "name": entry_name,
                            "url": rel_or_abs(out_file, cwd),
                            "expected_any": [label],
                            "expected_all": [],
                            "item_id": f"holdout-{slug}-manual-v{manual_index}",
                            "required": False,
                            "status": "ready",
                            "notes": f"supported-holdout; source=manual_seed; source_url={url}",
                        }
                    )

                    selected_manual_urls.add(url)
                    selected_count += 1
                    manual_index += 1
                except Exception:  # noqa: BLE001
                    continue

        if selected_count < args.per_label:
            unsupported.append(
                {
                    "label": label,
                    "reason": "insufficient_unique_images",
                    "selected": selected_count,
                    "requested": args.per_label,
                }
            )

    rows.sort(key=lambda row: str(row.get("name") or ""))

    labels_with_rows = {row.get("expected_any", [None])[0] for row in rows if isinstance(row.get("expected_any"), list)}

    out = {
        "name": "municipal-supported-holdout-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "labels": rel_or_abs(labels_path, cwd),
            "kaggle_dir": rel_or_abs(kaggle_dir, cwd) if kaggle_dir else None,
            "excluded_csv": rel_or_abs(csv_path, cwd) if csv_path.exists() else None,
            "manual_seed": rel_or_abs(manual_seed["seed_path"], cwd) if manual_seed["seed_path"].exists() else None,
            "retraining_manifest": (
                rel_or_abs(training_exclusions["manifest_path"], cwd)
                if training_exclusions["manifest_path"].exists()
                else None
            ),
            "download_enabled": not args.no_download,
            "per_label": args.per_label,
        },
        "summary": {
            "rows": len(rows),
            "labels_requested": len(labels),
            "labels_with_rows": len(labels_with_rows),
            "unsupported_labels": len(unsupported),
        },
        "unsupported": unsupported,
        "images": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("Supported holdout manifest generated")
    print(
        json.dumps(
            {
                "labels": rel_or_abs(labels_path, cwd),
                "kaggle_dir": rel_or_abs(kaggle_dir, cwd) if kaggle_dir else None,
                "output": rel_or_abs(out_path, cwd),
                "rows": out["summary"]["rows"],
                "labels_with_rows": out["summary"]["labels_with_rows"],
                "unsupported_labels": out["summary"]["unsupported_labels"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
