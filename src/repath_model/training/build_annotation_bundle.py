#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build annotation bundle from retraining manifest samples.",
        usage=(
            "python3 scripts/training/build_annotation_bundle.py "
            "[--manifest ml/artifacts/retraining/retraining-manifest.json] "
            "[--out-dir ml/artifacts/retraining/annotation-bundle] [--run-id <id>] "
            "[--refresh] [--no-download] [--dry-run]"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("ml") / "artifacts" / "retraining" / "retraining-manifest.json"),
        help="Retraining manifest JSON path.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("ml") / "artifacts" / "retraining" / "annotation-bundle"),
        help="Annotation bundle root directory.",
    )
    parser.add_argument("--run-id", default="", help="Optional bundle run id directory name.")
    parser.add_argument("--refresh", action="store_true", help="Delete existing bundle dir before building.")
    parser.add_argument("--no-download", action="store_true", help="Disable HTTP image downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Compute summary only, no filesystem writes.")
    return parser.parse_args()


def sanitize(value: str) -> str:
    return re.sub(r"^-+|-+$", "", re.sub(r"[^a-z0-9._-]+", "-", str(value or "").lower()))[:100]


def extension_from_source(source: str) -> str:
    text = str(source or "")
    match = re.search(r"\.([a-zA-Z0-9]{2,6})(?:$|[?#])", text)
    if not match:
        return ".jpg"
    return f".{match.group(1).lower()}"


def resolve_local_path(raw: str) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    if value.lower().startswith("file://"):
        try:
            parsed = urllib.parse.urlparse(value)
            return Path(urllib.request.url2pathname(parsed.path)).resolve()
        except Exception:
            return None
    if value.lower().startswith("http://") or value.lower().startswith("https://"):
        return None
    return Path(value).expanduser().resolve()


def is_http_url(raw: str) -> bool:
    text = str(raw or "").lower()
    return text.startswith("http://") or text.startswith("https://")


def download_to(source: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(source, headers={"User-Agent": "repath-model-bundle/1.0"}, method="GET")

    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if getattr(response, "status", 200) >= 400:
                    raise RuntimeError(f"HTTP {response.status}")
                out_file.write_bytes(response.read())
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
    if last_error:
        raise last_error


def copy_or_download(source: str, out_file: Path, allow_download: bool) -> str:
    local_path = resolve_local_path(source)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if local_path and local_path.exists():
        shutil.copyfile(local_path, out_file)
        return "copied_local"

    if is_http_url(source):
        if not allow_download:
            return "skipped_remote"
        download_to(source, out_file)
        return "downloaded"

    raise RuntimeError(f"Unsupported source image path: {source}")


def write_json(file_path: Path, value) -> None:
    file_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_dataset_yaml(bundle_root: Path, classes: list[str]) -> str:
    normalized = str(bundle_root.resolve()).replace(os.sep, "/")
    lines = [
        f"path: {normalized}",
        "train: images",
        "val: images",
        "test: images",
        "",
        f"nc: {len(classes)}",
        f"names: [{', '.join(json.dumps(name) for name in classes)}]",
        "",
    ]
    return "\n".join(lines)


def build_instructions() -> str:
    return "\n".join(
        [
            "# Annotation Instructions",
            "",
            "This bundle is generated from `retrain_*` samples in `benchmark-labeled.csv`.",
            "",
            "## Goal",
            "- Draw YOLO bounding boxes for **positive** images.",
            "- Keep **negative** images with empty label files.",
            "",
            "## Label Format",
            "- Each row in `labels/<image>.txt` uses: `class_id x_center y_center width height`.",
            "- Coordinates are normalized to `[0, 1]`.",
            "",
            "## Files",
            "- `annotations-template.csv`: assignment sheet with class IDs and hints.",
            "- `classes.json`: class list and numeric IDs.",
            "- `dataset.yaml`: YOLO dataset config.",
            "- `images/`: local annotation images.",
            "- `labels/`: YOLO label files (empty placeholders generated).",
            "",
            "## Completion Criteria",
            "- All non-negative rows in `annotations-template.csv` have at least one box in corresponding label files.",
            "- Negative rows remain empty.",
            "- Run `npm run validate:annotation:bundle -- --bundle-dir <bundle-dir>` to verify before training.",
            "",
        ]
    )


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest_path = Path(args.manifest).resolve()
    root_out_dir = Path(args.out_dir).resolve()
    run_id = str(args.run_id or "").strip() or datetime.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")
    bundle_dir = root_out_dir / run_id
    images_dir = bundle_dir / "images"
    labels_dir = bundle_dir / "labels"

    if not manifest_path.exists():
        raise SystemExit(f"Retraining manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples") if isinstance(manifest, dict) else []
    samples = samples if isinstance(samples, list) else []
    if not samples:
        raise SystemExit("No samples found in retraining manifest.")

    positives = [sample for sample in samples if not bool((sample or {}).get("is_negative")) and (sample or {}).get("label")]
    classes = sorted({str((sample or {}).get("label") or "").strip() for sample in positives if str((sample or {}).get("label") or "").strip()})
    class_to_id = {label: idx for idx, label in enumerate(classes)}

    copied_local = 0
    downloaded = 0
    skipped_existing = 0
    skipped_remote = 0
    skipped_invalid_source = 0
    tasks = []

    if not args.dry_run:
        if args.refresh and bundle_dir.exists():
            shutil.rmtree(bundle_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        sample = sample if isinstance(sample, dict) else {}
        sample_id = sanitize(str(sample.get("id") or f"sample-{index + 1}"))
        source_image = str(sample.get("image") or "").strip()
        if not source_image:
            continue

        ext = extension_from_source(source_image)
        image_file = f"{sample_id}{ext}"
        image_out_path = images_dir / image_file
        label_out_path = labels_dir / f"{sample_id}.txt"

        is_negative = bool(sample.get("is_negative") or not sample.get("label"))
        class_label = "" if is_negative else str(sample.get("label") or "").strip()
        class_id = class_to_id.get(class_label, "") if class_label else ""

        if not args.dry_run:
            try:
                if image_out_path.exists() and not args.refresh:
                    skipped_existing += 1
                else:
                    mode = copy_or_download(source_image, image_out_path, allow_download=not args.no_download)
                    if mode == "copied_local":
                        copied_local += 1
                    elif mode == "downloaded":
                        downloaded += 1
                    elif mode == "skipped_remote":
                        skipped_remote += 1
                        continue
            except Exception:  # noqa: BLE001
                skipped_invalid_source += 1
                continue

            if not label_out_path.exists():
                label_out_path.write_text("", encoding="utf-8")

        tasks.append(
            {
                "id": sample_id,
                "image_file": str(Path("images") / image_file).replace(os.sep, "/"),
                "label_file": str(Path("labels") / f"{sample_id}.txt").replace(os.sep, "/"),
                "is_negative": is_negative,
                "class_label": class_label,
                "class_id": class_id,
                "source": str(sample.get("source") or ""),
                "notes": str(sample.get("notes") or ""),
            }
        )

    if not args.dry_run:
        template_path = bundle_dir / "annotations-template.csv"
        with template_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "image_file", "label_file", "is_negative", "class_label", "class_id", "status", "annotator", "notes"])
            for row in tasks:
                writer.writerow(
                    [
                        row["id"],
                        row["image_file"],
                        row["label_file"],
                        "true" if row["is_negative"] else "false",
                        row["class_label"],
                        row["class_id"],
                        "todo",
                        "",
                        row["notes"],
                    ]
                )

        write_json(
            bundle_dir / "classes.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "classes": [{"id": class_to_id[label], "label": label} for label in classes],
            },
        )
        (bundle_dir / "dataset.yaml").write_text(build_dataset_yaml(bundle_dir, classes), encoding="utf-8")
        (bundle_dir / "INSTRUCTIONS.md").write_text(build_instructions(), encoding="utf-8")
        write_json(
            bundle_dir / "bundle-metadata.json",
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_manifest": rel_or_abs(manifest_path, cwd),
                "classes": len(classes),
                "total_samples": len(tasks),
                "positives": sum(1 for row in tasks if not row["is_negative"]),
                "negatives": sum(1 for row in tasks if row["is_negative"]),
                "refresh": bool(args.refresh),
            },
        )

    print("Annotation bundle prepared")
    print(
        json.dumps(
            {
                "manifest": rel_or_abs(manifest_path, cwd),
                "bundle_dir": rel_or_abs(bundle_dir, cwd),
                "samples": len(tasks),
                "classes": len(classes),
                "positives": sum(1 for row in tasks if not row["is_negative"]),
                "negatives": sum(1 for row in tasks if row["is_negative"]),
                "copied_local": copied_local,
                "downloaded": downloaded,
                "skipped_remote": skipped_remote,
                "skipped_invalid_source": skipped_invalid_source,
                "skipped_existing": skipped_existing,
                "download_enabled": not args.no_download,
                "dry_run": bool(args.dry_run),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
