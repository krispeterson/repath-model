#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO annotation bundle consistency and label quality.",
        usage=(
            "python3 scripts/training/validate_annotation_bundle.py "
            "[--bundle-dir ml/artifacts/retraining/annotation-bundle/<run-id>] "
            "[--bundle-root ml/artifacts/retraining/annotation-bundle] "
            "[--out ml/artifacts/retraining/annotation-bundle/<run-id>/validation-report.json] [--strict]"
        ),
    )
    parser.add_argument("--bundle-dir", default="", help="Specific bundle directory.")
    parser.add_argument(
        "--bundle-root",
        default=str(Path("ml") / "artifacts" / "retraining" / "annotation-bundle"),
        help="Bundle root; latest child dir is used when --bundle-dir is unset.",
    )
    parser.add_argument("--out", default="", help="Validation report output path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when issues exist.")
    return parser.parse_args()


def parse_yolo_line(line: str):
    parts = [part for part in str(line or "").strip().split() if part]
    if len(parts) != 5:
        return None
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    return {"classId": values[0], "x": values[1], "y": values[2], "w": values[3], "h": values[4]}


def read_classes(bundle_dir: Path) -> list[float]:
    file_path = bundle_dir / "classes.json"
    if not file_path.exists():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        classes = payload.get("classes") if isinstance(payload, dict) else []
        classes = classes if isinstance(classes, list) else []
        ids = []
        for row in classes:
            try:
                ids.append(float((row or {}).get("id")))
            except (TypeError, ValueError):
                continue
        return sorted(ids)
    except Exception:  # noqa: BLE001
        return []


def resolve_latest_bundle(bundle_root: Path) -> Path | None:
    if not bundle_root.exists():
        return None
    dirs = [entry for entry in bundle_root.iterdir() if entry.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return dirs[0]


def read_template_rows(bundle_dir: Path) -> list[dict]:
    template_path = bundle_dir / "annotations-template.csv"
    if not template_path.exists():
        raise SystemExit(f"annotations-template.csv not found in bundle: {bundle_dir}")

    with template_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = row or {}
            rows.append(
                {
                    "id": str(row.get("id") or "").strip(),
                    "imageFile": str(row.get("image_file") or "").strip(),
                    "labelFile": str(row.get("label_file") or "").strip(),
                    "isNegative": str(row.get("is_negative") or "").strip().lower() == "true",
                    "classId": str(row.get("class_id") or "").strip(),
                    "classLabel": str(row.get("class_label") or "").strip(),
                }
            )
        return rows


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else resolve_latest_bundle(Path(args.bundle_root).resolve())
    if bundle_dir is None or not bundle_dir.exists():
        raise SystemExit("Bundle directory not found. Build an annotation bundle first.")

    rows = read_template_rows(bundle_dir)
    valid_class_ids = set(read_classes(bundle_dir))
    max_class_id = max(valid_class_ids) if valid_class_ids else -1

    issues = []
    positives = 0
    negatives = 0
    positives_with_boxes = 0
    negatives_with_boxes = 0
    total_boxes = 0

    for row in rows:
        image_path = bundle_dir / str(row.get("imageFile") or "")
        label_path = bundle_dir / str(row.get("labelFile") or "")

        if not image_path.exists():
            issues.append({"id": row.get("id"), "issue": "missing_image_file", "path": row.get("imageFile")})
            continue
        if not label_path.exists():
            issues.append({"id": row.get("id"), "issue": "missing_label_file", "path": row.get("labelFile")})
            continue

        raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        parsed = []
        for raw in raw_lines:
            entry = parse_yolo_line(raw)
            if not entry:
                issues.append({"id": row.get("id"), "issue": "invalid_yolo_line", "line": raw, "path": row.get("labelFile")})
                continue

            in_range = (
                entry["x"] >= 0
                and entry["x"] <= 1
                and entry["y"] >= 0
                and entry["y"] <= 1
                and entry["w"] > 0
                and entry["w"] <= 1
                and entry["h"] > 0
                and entry["h"] <= 1
            )
            if not in_range:
                issues.append({"id": row.get("id"), "issue": "bbox_out_of_range", "line": raw, "path": row.get("labelFile")})

            if valid_class_ids and entry["classId"] not in valid_class_ids:
                issues.append(
                    {
                        "id": row.get("id"),
                        "issue": "unknown_class_id",
                        "class_id": entry["classId"],
                        "max_class_id": max_class_id,
                        "path": row.get("labelFile"),
                    }
                )

            parsed.append(entry)

        total_boxes += len(parsed)
        if bool(row.get("isNegative")):
            negatives += 1
            if parsed:
                negatives_with_boxes += 1
                issues.append(
                    {
                        "id": row.get("id"),
                        "issue": "negative_has_boxes",
                        "path": row.get("labelFile"),
                        "boxes": len(parsed),
                    }
                )
        else:
            positives += 1
            if not parsed:
                issues.append({"id": row.get("id"), "issue": "positive_missing_boxes", "path": row.get("labelFile")})
            else:
                positives_with_boxes += 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": rel_or_abs(bundle_dir, cwd),
        "summary": {
            "rows": len(rows),
            "positives": positives,
            "negatives": negatives,
            "positives_with_boxes": positives_with_boxes,
            "negatives_with_boxes": negatives_with_boxes,
            "total_boxes": total_boxes,
            "issue_count": len(issues),
        },
        "issues": issues,
    }

    out_path = Path(args.out).resolve() if args.out else bundle_dir / "validation-report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Annotation bundle validation complete")
    print(
        json.dumps(
            {
                "bundle_dir": report["bundle_dir"],
                "output": rel_or_abs(out_path, cwd),
                "summary": report["summary"],
            },
            indent=2,
        )
    )

    if args.strict and issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
