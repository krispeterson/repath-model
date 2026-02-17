#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import tempfile

try:
    from ultralytics import YOLO
except Exception as exc:
    raise SystemExit(
        "Ultralytics is required. Install with: pip install ultralytics"
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a YOLOv8 .pt via Ultralytics and export to TFLite."
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics model name or path to a .pt file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--half", action="store_true", help="Enable FP16 quantization")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--nms", action="store_true", help="Enable NMS in export")
    parser.add_argument("--data", default=None, help="Dataset yaml (required for int8)")
    parser.add_argument("--fraction", type=float, default=None, help="Dataset fraction for int8")
    parser.add_argument("--device", default=None, help="Export device (e.g. cpu, mps)")
    parser.add_argument(
        "--classes",
        default=None,
        help="Path to a JSON or newline-delimited text file of class labels",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join("assets", "models"),
        help="Output directory for yolo-repath.tflite and labels",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.int8 and not args.data:
        raise SystemExit("--data is required when using --int8")

    cwd = os.getcwd()
    out_dir = os.path.abspath(args.out_dir)
    classes_path = os.path.abspath(args.classes) if args.classes else None
    with tempfile.TemporaryDirectory(prefix="repath-yolo-") as tmpdir:
        os.chdir(tmpdir)
        try:
            class_list = None
            if classes_path:
                with open(classes_path, "r", encoding="utf-8") as handle:
                    raw = handle.read().strip()
                if raw:
                    if args.classes.endswith(".json"):
                        data = json.loads(raw)
                        if isinstance(data, list):
                            class_list = [str(item).strip() for item in data if str(item).strip()]
                        elif isinstance(data, dict) and isinstance(data.get("classes"), list):
                            class_list = [str(item).strip() for item in data["classes"] if str(item).strip()]
                        else:
                            raise SystemExit("--classes JSON must be an array or {\"classes\": [...]}")
                    else:
                        class_list = [line.strip() for line in raw.splitlines() if line.strip()]

            model = YOLO(args.model)
            if class_list:
                if hasattr(model, "set_classes"):
                    model.set_classes(class_list)
                else:
                    print("Warning: model does not support set_classes; ignoring --classes.")

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

            if isinstance(exported, (list, tuple)):
                export_path = exported[0]
            else:
                export_path = exported

            if not export_path or not os.path.exists(export_path):
                raise SystemExit("Export failed or output not found.")

            os.makedirs(out_dir, exist_ok=True)
            tflite_out = os.path.join(out_dir, "yolo-repath.tflite")
            shutil.copy2(export_path, tflite_out)

            if class_list:
                label_list = class_list
            else:
                labels = model.names if hasattr(model, "names") else {}
                if isinstance(labels, dict):
                    label_list = [labels[i] for i in sorted(labels.keys())]
                else:
                    label_list = list(labels)

            labels_out = os.path.join(out_dir, "yolo-repath.labels.json")
            with open(labels_out, "w", encoding="utf-8") as handle:
                json.dump(label_list, handle, indent=2)
                handle.write("\n")

            print("Exported:", tflite_out)
            print("Labels:", labels_out)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    main()
