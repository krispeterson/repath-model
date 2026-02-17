#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a candidate YOLO-World TFLite model from retraining manifest label priorities."
    )
    parser.add_argument(
        "--retraining-manifest",
        default=os.path.join("ml", "artifacts", "retraining", "retraining-manifest.json"),
        help="Path to retraining manifest JSON",
    )
    parser.add_argument(
        "--base-labels",
        default=os.path.join("assets", "models", "yolo-repath.labels.json"),
        help="Existing model labels to merge with retraining labels",
    )
    parser.add_argument(
        "--label-mode",
        choices=["merged", "retraining-only"],
        default="merged",
        help="How to build candidate class list",
    )
    parser.add_argument(
        "--model",
        default="yolov8s-worldv2.pt",
        help="Ultralytics model name or path to .pt for export",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input size for export")
    parser.add_argument("--half", action="store_true", help="Enable FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument(
        "--nms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable NMS in export (default: true). Use --no-nms to disable.",
    )
    parser.add_argument("--data", default=None, help="Dataset yaml (required for int8)")
    parser.add_argument("--fraction", type=float, default=None, help="Dataset fraction for int8")
    parser.add_argument("--device", default=None, help="Export device (e.g. cpu, mps)")
    parser.add_argument(
        "--out-root",
        default=os.path.join("ml", "artifacts", "models", "candidates"),
        help="Output root directory for candidate runs",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id folder name. Default uses UTC timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write candidate metadata and labels without running model export.",
    )
    return parser.parse_args()


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_json_list(file_path):
    if not file_path or not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        return []
    return unique_in_order(data)


def load_retraining_labels(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        return []
    labels = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if sample.get("is_negative"):
            continue
        labels.append(sample.get("label"))
    return unique_in_order(labels)


def build_candidate_labels(args, retraining_labels):
    if args.label_mode == "retraining-only":
        return retraining_labels
    base_labels_path = args.base_labels
    if (not base_labels_path or not os.path.exists(base_labels_path)) and os.path.exists(
        os.path.join("assets", "models", "yolov8.labels.json")
    ):
        base_labels_path = os.path.join("assets", "models", "yolov8.labels.json")
    base_labels = load_json_list(base_labels_path)
    return unique_in_order(base_labels + retraining_labels)


def ensure_ultralytics_available():
    if YOLO is None:
        raise SystemExit("Ultralytics is required. Install with: pip install ultralytics")


def export_model(args, class_list, out_dir):
    ensure_ultralytics_available()

    if args.int8 and not args.data:
        raise SystemExit("--data is required when using --int8")

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="repath-candidate-") as tmpdir:
        os.chdir(tmpdir)
        try:
            model = YOLO(args.model)
            if hasattr(model, "set_classes"):
                model.set_classes(class_list)
            else:
                raise SystemExit("Selected model does not support set_classes. Use a YOLO-World variant.")

            export_kwargs = {
                "format": "tflite",
                "imgsz": args.imgsz,
                "half": bool(args.half),
                "int8": bool(args.int8),
                "nms": bool(args.nms),
            }
            if args.data:
                export_kwargs["data"] = args.data
            if args.fraction is not None:
                export_kwargs["fraction"] = args.fraction
            if args.device:
                export_kwargs["device"] = args.device

            exported = model.export(**export_kwargs)
            export_path = exported[0] if isinstance(exported, (list, tuple)) else exported
            if not export_path or not os.path.exists(export_path):
                raise SystemExit("Export failed or output not found.")

            os.makedirs(out_dir, exist_ok=True)
            out_model_path = os.path.join(out_dir, "yolo-repath.tflite")
            shutil.copy2(export_path, out_model_path)
            return out_model_path
        finally:
            os.chdir(cwd)


def main():
    args = parse_args()
    manifest_path = os.path.abspath(args.retraining_manifest)
    out_root = os.path.abspath(args.out_root)
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(out_root, run_id)

    if not os.path.exists(manifest_path):
        raise SystemExit(f"Retraining manifest not found: {manifest_path}")

    retraining_labels = load_retraining_labels(manifest_path)
    if not retraining_labels:
        raise SystemExit("No positive labels found in retraining manifest.")

    class_list = build_candidate_labels(args, retraining_labels)
    if not class_list:
        raise SystemExit("Candidate class list is empty.")

    os.makedirs(out_dir, exist_ok=True)
    labels_out = os.path.join(out_dir, "yolo-repath.labels.json")
    with open(labels_out, "w", encoding="utf-8") as handle:
        json.dump(class_list, handle, indent=2)
        handle.write("\n")

    model_path = None
    if not args.dry_run:
        model_path = export_model(args, class_list, out_dir)

    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "source": {
            "retraining_manifest": os.path.relpath(manifest_path, os.getcwd()),
            "base_labels": os.path.relpath(os.path.abspath(args.base_labels), os.getcwd())
            if args.base_labels
            else None,
        },
        "export": {
            "model": args.model,
            "imgsz": args.imgsz,
            "half": bool(args.half),
            "int8": bool(args.int8),
            "nms": bool(args.nms),
            "data": args.data,
            "fraction": args.fraction,
            "device": args.device,
            "label_mode": args.label_mode,
        },
        "labels": {
            "count": len(class_list),
            "retraining_label_count": len(retraining_labels),
            "class_list_preview": class_list[:20],
        },
        "artifacts": {
            "candidate_dir": os.path.relpath(out_dir, os.getcwd()),
            "labels": os.path.relpath(labels_out, os.getcwd()),
            "model": os.path.relpath(model_path, os.getcwd()) if model_path else None,
        },
        "notes": [
            "This workflow updates model vocabulary using retraining priorities.",
            "True detector retraining still requires box-level annotations."
        ],
    }

    metadata_out = os.path.join(out_dir, "metadata.json")
    with open(metadata_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print("Candidate model export prepared")
    print(json.dumps(metadata["artifacts"], indent=2))
    if args.dry_run:
        print("Dry run: export skipped (labels + metadata generated).")


if __name__ == "__main__":
    main()
