#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed YOLO annotation label files in a bundle using model predictions."
    )
    parser.add_argument(
        "--bundle-dir",
        default="",
        help="Path to annotation bundle. Defaults to latest bundle under --bundle-root.",
    )
    parser.add_argument(
        "--bundle-root",
        default=os.path.join("ml", "artifacts", "retraining", "annotation-bundle"),
        help="Root folder containing bundle runs.",
    )
    parser.add_argument(
        "--model",
        default=os.path.join("assets", "models", "yolo-repath.tflite"),
        help="TFLite model path.",
    )
    parser.add_argument(
        "--labels",
        default=os.path.join("assets", "models", "yolo-repath.labels.json"),
        help="Model label JSON array.",
    )
    parser.add_argument("--threshold", type=float, default=0.30, help="Detection confidence threshold.")
    parser.add_argument("--topk", type=int, default=10, help="Top detections to inspect per image.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing positive label files.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="If no class-match detection exists, use top detection box with expected class id.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute seeds without writing files.")
    parser.add_argument(
        "--report-out",
        default="",
        help="Optional report path. Defaults to <bundle>/seed-report.json",
    )
    return parser.parse_args()


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


def read_classes(bundle_dir):
    classes_path = os.path.join(bundle_dir, "classes.json")
    if not os.path.exists(classes_path):
        raise SystemExit(f"classes.json not found in bundle: {bundle_dir}")
    with open(classes_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("classes", [])
    label_to_id = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        class_id = row.get("id")
        if not label:
            continue
        if not isinstance(class_id, int):
            continue
        label_to_id[label] = class_id
    if not label_to_id:
        raise SystemExit("No classes found in classes.json")
    return label_to_id


def load_model_labels(path_):
    with open(path_, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit("Model labels file must be a JSON array")
    return [str(label or "").strip() for label in data]


def read_template_rows(bundle_dir):
    template_path = os.path.join(bundle_dir, "annotations-template.csv")
    if not os.path.exists(template_path):
        raise SystemExit(f"annotations-template.csv not found in bundle: {bundle_dir}")
    with open(template_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_has_boxes(path_):
    if not os.path.exists(path_):
        return False
    with open(path_, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return True
    return False


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def row_to_bbox(row):
    x1, y1, x2, y2 = [float(row[i]) for i in range(4)]
    scale_normalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5
    if not scale_normalized:
        x1 /= 640.0
        y1 /= 640.0
        x2 /= 640.0
        y2 /= 640.0
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    w = clamp01(xb - xa)
    h = clamp01(yb - ya)
    xc = clamp01(xa + w / 2.0)
    yc = clamp01(ya + h / 2.0)
    if w <= 0.0 or h <= 0.0:
        return None
    return (xc, yc, w, h)


def infer(interpreter, input_info, output_info, labels, image_path, threshold, topk):
    image = Image.open(image_path).convert("RGB").resize((640, 640))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    interpreter.set_tensor(input_info["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_info["index"])[0]
    keep = output[output[:, 4] >= threshold]
    if keep.size == 0:
        return []
    keep = keep[np.argsort(-keep[:, 4])]

    detections = []
    for row in keep[:topk]:
        class_id = int(round(float(row[5])))
        if class_id < 0 or class_id >= len(labels):
            continue
        bbox = row_to_bbox(row)
        if not bbox:
            continue
        detections.append(
            {
                "label": labels[class_id],
                "score": float(row[4]),
                "bbox": bbox,
            }
        )
    return detections


def main():
    args = parse_args()
    bundle_dir = os.path.abspath(args.bundle_dir) if args.bundle_dir else resolve_latest_bundle(os.path.abspath(args.bundle_root))
    if not bundle_dir or not os.path.isdir(bundle_dir):
        raise SystemExit("Bundle directory not found.")

    model_path = os.path.abspath(args.model)
    labels_path = os.path.abspath(args.labels)
    if not os.path.exists(model_path) and model_path.endswith("yolo-repath.tflite"):
        legacy_model = model_path.replace("yolo-repath.tflite", "yolov8.tflite")
        if os.path.exists(legacy_model):
            model_path = legacy_model
    if not os.path.exists(labels_path) and labels_path.endswith("yolo-repath.labels.json"):
        legacy_labels = labels_path.replace("yolo-repath.labels.json", "yolov8.labels.json")
        if os.path.exists(legacy_labels):
            labels_path = legacy_labels
    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.exists(labels_path):
        raise SystemExit(f"Labels not found: {labels_path}")

    label_to_id = read_classes(bundle_dir)
    model_labels = load_model_labels(labels_path)
    rows = read_template_rows(bundle_dir)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    positives_total = 0
    positives_seeded = 0
    positives_existing = 0
    skipped_unknown_class = 0
    skipped_no_detection = 0
    fallback_used = 0
    errors = []

    for row in rows:
        is_negative = str(row.get("is_negative") or "").strip().lower() == "true"
        if is_negative:
            continue

        positives_total += 1
        class_label = str(row.get("class_label") or "").strip()
        class_id = label_to_id.get(class_label)
        image_file = str(row.get("image_file") or "").strip()
        label_file = str(row.get("label_file") or "").strip()
        image_path = os.path.join(bundle_dir, image_file)
        label_path = os.path.join(bundle_dir, label_file)

        if class_id is None:
            skipped_unknown_class += 1
            errors.append({"id": row.get("id"), "issue": "unknown_class_label", "class_label": class_label})
            continue

        if not os.path.exists(image_path):
            skipped_no_detection += 1
            errors.append({"id": row.get("id"), "issue": "missing_image", "image_file": image_file})
            continue

        if file_has_boxes(label_path) and not args.overwrite:
            positives_existing += 1
            continue

        detections = infer(
            interpreter=interpreter,
            input_info=input_info,
            output_info=output_info,
            labels=model_labels,
            image_path=image_path,
            threshold=args.threshold,
            topk=args.topk,
        )

        selected = None
        for det in detections:
            if det["label"] == class_label:
                selected = det
                break

        if selected is None and args.allow_fallback and detections:
            selected = detections[0]
            fallback_used += 1

        if selected is None:
            skipped_no_detection += 1
            continue

        xc, yc, w, h = selected["bbox"]
        line = f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
        if not args.dry_run:
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w", encoding="utf-8") as handle:
                handle.write(line)
        positives_seeded += 1

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "bundle_dir": os.path.relpath(bundle_dir, os.getcwd()),
        "model": os.path.relpath(model_path, os.getcwd()),
        "labels": os.path.relpath(labels_path, os.getcwd()),
        "settings": {
            "threshold": args.threshold,
            "topk": args.topk,
            "overwrite": bool(args.overwrite),
            "allow_fallback": bool(args.allow_fallback),
            "dry_run": bool(args.dry_run),
        },
        "summary": {
            "positives_total": positives_total,
            "positives_seeded": positives_seeded,
            "positives_existing": positives_existing,
            "skipped_unknown_class": skipped_unknown_class,
            "skipped_no_detection": skipped_no_detection,
            "fallback_used": fallback_used,
            "error_count": len(errors),
        },
        "errors": errors[:200],
    }

    report_out = os.path.abspath(args.report_out) if args.report_out else os.path.join(bundle_dir, "seed-report.json")
    with open(report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print("Annotation seed complete")
    print(json.dumps({"report": os.path.relpath(report_out, os.getcwd()), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
