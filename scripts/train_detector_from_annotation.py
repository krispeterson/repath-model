#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime

from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector from an annotation bundle and export candidate TFLite artifacts."
    )
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Path to annotation bundle directory. Defaults to latest bundle under --bundle-root.",
    )
    parser.add_argument(
        "--bundle-root",
        default=os.path.join("ml", "artifacts", "retraining", "annotation-bundle"),
        help="Root directory containing annotation bundle runs.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base Ultralytics model name or .pt path for training initialization.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training/export image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--device", default=None, help="Training device (cpu, mps, cuda:0).")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--project", default=os.path.join("ml", "artifacts", "training-runs"), help="Ultralytics project output root.")
    parser.add_argument("--name", default="", help="Ultralytics run name. Defaults to run-id.")
    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NMS during TFLite export (default: true). Use --no-nms to disable.",
    )
    parser.add_argument("--half", action="store_true", help="Enable FP16 during TFLite export.")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 during TFLite export.")
    parser.add_argument("--fraction", type=float, default=None, help="Optional dataset fraction for INT8 calibration.")
    parser.add_argument(
        "--candidate-root",
        default=os.path.join("ml", "artifacts", "models", "candidates"),
        help="Where candidate model artifacts are written.",
    )
    parser.add_argument("--run-id", default="", help="Candidate run id. Defaults to UTC timestamp.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip strict annotation validation check.")
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=8_000_000,
        help="Downscale bundle images larger than this many pixels before training (0 disables).",
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=2048,
        help="Downscale bundle images whose width/height exceed this size before training (0 disables).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Emit metadata only; do not train/export.")
    return parser.parse_args()


def ensure_ultralytics():
    if YOLO is None:
        raise SystemExit("Ultralytics is required. Install with: pip install ultralytics")


def resolve_latest_bundle(bundle_root):
    if not os.path.isdir(bundle_root):
        return None
    dirs = [
        os.path.join(bundle_root, name)
        for name in os.listdir(bundle_root)
        if os.path.isdir(os.path.join(bundle_root, name))
    ]
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[0]


def load_classes(bundle_dir):
    classes_path = os.path.join(bundle_dir, "classes.json")
    if not os.path.exists(classes_path):
        raise SystemExit(f"classes.json not found in bundle: {bundle_dir}")
    with open(classes_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("classes", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("classes.json is empty or invalid.")
    labels = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        if label:
            labels.append(label)
    if not labels:
        raise SystemExit("No class labels found in classes.json.")
    return labels


def run_validation(bundle_dir):
    cmd = ["node", "ml/training/validate-annotation-bundle.js", "--bundle-dir", bundle_dir, "--strict"]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def downscale_bundle_images(bundle_dir, max_pixels, max_dim):
    images_dir = os.path.join(bundle_dir, "images")
    summary = {
        "enabled": bool((max_pixels and max_pixels > 0) or (max_dim and max_dim > 0)),
        "max_image_pixels": max_pixels,
        "max_image_dim": max_dim,
        "images_scanned": 0,
        "images_downscaled": 0,
        "images_failed": 0,
        "downscaled_examples": [],
    }
    if not summary["enabled"]:
        return summary
    if not os.path.isdir(images_dir):
        return summary

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    for name in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, name)
        if not os.path.isfile(image_path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in valid_ext:
            continue
        summary["images_scanned"] += 1
        try:
            with Image.open(image_path) as src:
                width, height = src.size
                pixels = width * height

                scale = 1.0
                if max_pixels and max_pixels > 0 and pixels > max_pixels:
                    scale = min(scale, math.sqrt(float(max_pixels) / float(pixels)))
                if max_dim and max_dim > 0:
                    max_current_dim = max(width, height)
                    if max_current_dim > max_dim:
                        scale = min(scale, float(max_dim) / float(max_current_dim))

                if scale >= 0.999:
                    continue

                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                resample_lanczos = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                resized = src.resize((new_w, new_h), resample_lanczos)

                save_kwargs = {}
                if ext in {".jpg", ".jpeg"}:
                    save_kwargs = {"quality": 90, "optimize": True}
                elif ext == ".png":
                    save_kwargs = {"optimize": True}

                resized.save(image_path, **save_kwargs)
                summary["images_downscaled"] += 1
                if len(summary["downscaled_examples"]) < 10:
                    summary["downscaled_examples"].append(
                        {
                            "file": os.path.relpath(image_path, os.getcwd()),
                            "before": [width, height],
                            "after": [new_w, new_h],
                        }
                    )
        except Exception:
            summary["images_failed"] += 1

    return summary


def export_tflite(best_pt_path, args, dataset_yaml):
    export_model = YOLO(best_pt_path)
    export_kwargs = {
        "format": "tflite",
        "imgsz": args.imgsz,
        "half": bool(args.half),
        "int8": bool(args.int8),
        "nms": bool(args.nms),
    }
    if args.int8:
        export_kwargs["data"] = dataset_yaml
    if args.fraction is not None:
        export_kwargs["fraction"] = args.fraction
    if args.device:
        export_kwargs["device"] = args.device
    exported = export_model.export(**export_kwargs)
    export_path = exported[0] if isinstance(exported, (list, tuple)) else exported
    if not export_path or not os.path.exists(export_path):
        raise SystemExit("Export failed or output not found.")
    return export_path


def main():
    args = parse_args()

    bundle_dir = os.path.abspath(args.bundle_dir) if args.bundle_dir else resolve_latest_bundle(os.path.abspath(args.bundle_root))
    if not bundle_dir or not os.path.isdir(bundle_dir):
        raise SystemExit("Annotation bundle directory not found.")

    dataset_yaml = os.path.join(bundle_dir, "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise SystemExit(f"dataset.yaml not found in bundle: {bundle_dir}")

    if not args.skip_validate:
        code = run_validation(bundle_dir)
        if code != 0:
            raise SystemExit("Strict annotation validation failed. Complete labels before training or pass --skip-validate.")

    preprocess_summary = downscale_bundle_images(bundle_dir, args.max_image_pixels, args.max_image_dim)
    if preprocess_summary["enabled"]:
        print("Bundle image preprocessing complete")
        print(
            json.dumps(
                {
                    "images_scanned": preprocess_summary["images_scanned"],
                    "images_downscaled": preprocess_summary["images_downscaled"],
                    "images_failed": preprocess_summary["images_failed"],
                },
                indent=2,
            )
        )

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    candidate_dir = os.path.abspath(os.path.join(args.candidate_root, run_id))
    labels = load_classes(bundle_dir)
    labels_path = os.path.join(candidate_dir, "yolo-repath.labels.json")
    metadata_path = os.path.join(candidate_dir, "metadata.json")

    os.makedirs(candidate_dir, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)
        handle.write("\n")

    trained_best_pt = None
    trained_last_pt = None
    tflite_path = None
    run_name = args.name or run_id

    if not args.dry_run:
        ensure_ultralytics()
        train_model = YOLO(args.model)
        train_kwargs = {
            "data": dataset_yaml,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "patience": args.patience,
            "project": args.project,
            "name": run_name,
            "exist_ok": True,
        }
        if args.device:
            train_kwargs["device"] = args.device

        results = train_model.train(**train_kwargs)
        save_dir = getattr(results, "save_dir", None)
        if not save_dir:
            save_dir = os.path.join(args.project, run_name)
        weights_dir = os.path.join(save_dir, "weights")
        trained_best_pt = os.path.join(weights_dir, "best.pt")
        trained_last_pt = os.path.join(weights_dir, "last.pt")
        if not os.path.exists(trained_best_pt):
            if os.path.exists(trained_last_pt):
                trained_best_pt = trained_last_pt
            else:
                raise SystemExit("Training finished but no weights found.")

        exported = export_tflite(trained_best_pt, args, dataset_yaml)
        tflite_path = os.path.join(candidate_dir, "yolo-repath.tflite")
        copy_file(exported, tflite_path)
        copy_file(trained_best_pt, os.path.join(candidate_dir, "model.best.pt"))
        if os.path.exists(trained_last_pt):
            copy_file(trained_last_pt, os.path.join(candidate_dir, "model.last.pt"))

    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "bundle": {
            "dir": os.path.relpath(bundle_dir, os.getcwd()),
            "dataset_yaml": os.path.relpath(dataset_yaml, os.getcwd()),
            "class_count": len(labels),
            "preprocessing": preprocess_summary,
        },
        "training": {
            "base_model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "patience": args.patience,
            "device": args.device,
            "project": args.project,
            "name": run_name,
        },
        "export": {
            "nms": bool(args.nms),
            "half": bool(args.half),
            "int8": bool(args.int8),
            "fraction": args.fraction,
        },
        "artifacts": {
            "candidate_dir": os.path.relpath(candidate_dir, os.getcwd()),
            "labels": os.path.relpath(labels_path, os.getcwd()),
            "model": os.path.relpath(tflite_path, os.getcwd()) if tflite_path else None,
            "best_pt": os.path.relpath(trained_best_pt, os.getcwd()) if trained_best_pt else None,
            "last_pt": os.path.relpath(trained_last_pt, os.getcwd()) if trained_last_pt and os.path.exists(trained_last_pt) else None,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print("Training candidate prepared")
    print(json.dumps(metadata["artifacts"], indent=2))
    if args.dry_run:
        print("Dry run: training/export skipped.")


if __name__ == "__main__":
    main()
