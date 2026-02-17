#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

import numpy as np
from PIL import Image
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a TFLite detection model against a fixed image manifest.")
    parser.add_argument("--model", default="assets/models/yolo-repath.tflite", help="Path to .tflite model")
    parser.add_argument("--labels", default="assets/models/yolo-repath.labels.json", help="Path to label JSON array")
    parser.add_argument("--manifest", default="test/benchmarks/benchmark-manifest.seed.json", help="Benchmark manifest JSON")
    parser.add_argument("--cache-dir", default="test/benchmarks/images", help="Where benchmark images are cached")
    parser.add_argument("--out", default="test/benchmarks/latest-results.json", help="Where to write benchmark results")
    parser.add_argument("--threshold", type=float, default=0.35, help="Minimum score threshold")
    parser.add_argument("--topk", type=int, default=5, help="Max labels per image")
    parser.add_argument(
        "--supported-only",
        action="store_true",
        help="Evaluate only entries whose expected labels are present in the model labels file.",
    )
    return parser.parse_args()


def run_curl_download(url, out_file):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--retry", "3", "--retry-all-errors", "--connect-timeout", "20", url, "-o", str(out_file), "-sS"]
    subprocess.run(cmd, check=True)


def resolve_image_path(url, cache_file):
    value = str(url or "").strip()
    if not value:
        return None

    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.scheme != "file":
            return None
        local_path = unquote(parsed.path)
        return Path(local_path)

    if value.startswith("http://") or value.startswith("https://"):
        if not cache_file.exists():
            run_curl_download(value, cache_file)
        return cache_file

    local = Path(value)
    return local


def load_manifest(manifest_path):
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("Manifest must be an object with an images array.")
    return data


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def infer_labels(interpreter, input_info, output_info, labels, image_path, threshold, topk):
    image = Image.open(image_path).convert("RGB").resize((640, 640))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    interpreter.set_tensor(input_info["index"], arr)
    interpreter.invoke()
    output_tensor = np.asarray(interpreter.get_tensor(output_info["index"]))
    if output_tensor.ndim == 3 and output_tensor.shape[0] == 1:
        output = output_tensor[0]
    elif output_tensor.ndim == 2:
        output = output_tensor
    elif output_tensor.ndim == 1 and output_tensor.size % 6 == 0:
        output = output_tensor.reshape((-1, 6))
    else:
        raise ValueError(
            f"Unsupported output tensor shape {tuple(output_tensor.shape)}. "
            "Expected NMS detections with shape [1, N, 6] or [N, 6]."
        )

    if output.ndim != 2:
        raise ValueError(
            f"Unsupported output tensor rank after reshape: {output.ndim}. "
            "Expected NMS detections with shape [N, 6]."
        )

    if output.shape[1] == 6:
        detections_matrix = output
    elif output.shape[0] == 6:
        detections_matrix = output.T
    elif output.shape[0] == 4 + len(labels):
        raise ValueError(
            f"Model appears to output raw YOLO head tensor {tuple(output_tensor.shape)} "
            "(no integrated NMS). Re-export model with --nms."
        )
    else:
        raise ValueError(
            f"Unsupported detection layout {tuple(output_tensor.shape)}. "
            "Expected NMS detections with 6 values per row: x1,y1,x2,y2,score,class_id."
        )

    keep = detections_matrix[detections_matrix[:, 4] >= threshold]
    keep = keep[np.argsort(-keep[:, 4])]

    detections = []
    predicted_labels = []
    for row in keep:
        class_id = int(round(float(row[5])))
        if class_id < 0 or class_id >= len(labels):
            continue
        score = float(row[4])
        label = labels[class_id]
        detections.append({"label": label, "score": round(score, 4)})
        predicted_labels.append(label)

    predicted_labels = unique_in_order(predicted_labels)[:topk]
    return detections[:topk], predicted_labels


def pr_stats(predicted_labels, expected_labels):
    pred = set(predicted_labels)
    exp = set(expected_labels)
    if not pred and not exp:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0}
    tp = len(pred & exp)
    fp = len(pred - exp)
    fn = len(exp - pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def main():
    args = parse_args()
    if not Path(args.model).exists() and str(args.model).endswith("yolo-repath.tflite"):
        legacy_model = str(args.model).replace("yolo-repath.tflite", "yolov8.tflite")
        if Path(legacy_model).exists():
            args.model = legacy_model
    if not Path(args.labels).exists() and str(args.labels).endswith("yolo-repath.labels.json"):
        legacy_labels = str(args.labels).replace("yolo-repath.labels.json", "yolov8.labels.json")
        if Path(legacy_labels).exists():
            args.labels = legacy_labels

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if not isinstance(labels, list):
        raise ValueError("Labels file must be a JSON array.")

    manifest = load_manifest(args.manifest)
    image_entries = manifest.get("images", [])

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    results = []
    micro_tp = micro_fp = micro_fn = 0
    any_cases = 0
    any_hits = 0
    negative_cases = 0
    negative_clean = 0
    skipped_unsupported = 0
    model_label_set = set(str(label).strip() for label in labels if str(label).strip())

    for entry in image_entries:
        name = str(entry.get("name") or "").strip()
        url = str(entry.get("url") or "").strip()
        raw_expected_any = [str(v).strip() for v in entry.get("expected_any", []) if str(v).strip()]
        raw_expected_all = [str(v).strip() for v in entry.get("expected_all", []) if str(v).strip()]
        expected_any = list(raw_expected_any)
        expected_all = list(raw_expected_all)
        if not name or not url:
            continue

        if args.supported_only:
            expected_any = [label for label in expected_any if label in model_label_set]
            expected_all = [label for label in expected_all if label in model_label_set]
            had_expected = bool(raw_expected_any or raw_expected_all)
            if had_expected and not expected_any and not expected_all:
                skipped_unsupported += 1
                continue

        out_file = Path(args.cache_dir) / f"{name}.jpg"
        image_path = resolve_image_path(url, out_file)
        if not image_path or not image_path.exists():
            continue

        detections, predicted_labels = infer_labels(
            interpreter=interpreter,
            input_info=input_info,
            output_info=output_info,
            labels=labels,
            image_path=image_path,
            threshold=args.threshold,
            topk=args.topk,
        )

        expected_set = expected_all if expected_all else expected_any
        stats = pr_stats(predicted_labels, expected_set)
        micro_tp += stats["tp"]
        micro_fp += stats["fp"]
        micro_fn += stats["fn"]

        any_hit = None
        if expected_any:
            any_cases += 1
            any_hit = bool(set(predicted_labels) & set(expected_any))
            if any_hit:
                any_hits += 1
        else:
            negative_cases += 1
            if not predicted_labels:
                negative_clean += 1

        results.append(
            {
                "name": name,
                "url": url,
                "expected_any": expected_any,
                "expected_all": expected_all,
                "predicted_labels": predicted_labels,
                "detections": detections,
                "precision": round(stats["precision"], 4),
                "recall": round(stats["recall"], 4),
                "any_hit": any_hit,
            }
        )

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    any_hit_rate = any_hits / any_cases if any_cases > 0 else 0.0
    negative_clean_rate = negative_clean / negative_cases if negative_cases > 0 else 0.0

    summary = {
        "model": args.model,
        "labels": args.labels,
        "manifest": args.manifest,
        "threshold": args.threshold,
        "topk": args.topk,
        "images_evaluated": len(results),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "any_hit_rate": round(any_hit_rate, 4),
        "negative_clean_rate": round(negative_clean_rate, 4),
        "tp": micro_tp,
        "fp": micro_fp,
        "fn": micro_fn,
        "supported_only": bool(args.supported_only),
        "skipped_unsupported_entries": skipped_unsupported,
    }

    payload = {"summary": summary, "results": results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Benchmark summary")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
