#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def usage() -> str:
    return (
        "python3 scripts/training/fill_missing_positive_boxes.py "
        "[--bundle-dir ml/artifacts/retraining/annotation-bundle/<run-id>] "
        "[--bundle-root ml/artifacts/retraining/annotation-bundle] "
        "[--x-center 0.5] [--y-center 0.5] [--width 1] [--height 1] "
        "[--overwrite] [--dry-run]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument(
        "--bundle-root",
        default=str(Path("ml") / "artifacts" / "retraining" / "annotation-bundle"),
    )
    parser.add_argument("--x-center", type=float, default=0.5)
    parser.add_argument("--y-center", type=float, default=0.5)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() == "true"


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def resolve_latest_bundle(bundle_root: Path) -> Path | None:
    if not bundle_root.exists():
        return None

    dirs = [path for path in bundle_root.iterdir() if path.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def read_template_rows(bundle_dir: Path) -> list[dict]:
    template_path = bundle_dir / "annotations-template.csv"
    if not template_path.exists():
        raise SystemExit(f"annotations-template.csv not found in bundle: {bundle_dir}")

    with template_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                {
                    "id": str((row or {}).get("id", "")).strip(),
                    "is_negative": parse_bool(str((row or {}).get("is_negative", ""))),
                    "class_id": str((row or {}).get("class_id", "")).strip(),
                    "label_file": str((row or {}).get("label_file", "")).strip(),
                }
            )
        return rows


def validate_box_args(args: argparse.Namespace) -> None:
    values = [args.x_center, args.y_center, args.width, args.height]
    if not all(isinstance(v, (float, int)) and 0 <= float(v) <= 1 for v in values):
        raise SystemExit("--x-center, --y-center, --width, and --height must all be in [0, 1].")
    if float(args.width) <= 0 or float(args.height) <= 0:
        raise SystemExit("--width and --height must be greater than 0.")


def main() -> None:
    args = parse_args()
    validate_box_args(args)

    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else resolve_latest_bundle(Path(args.bundle_root).resolve())

    if bundle_dir is None or not bundle_dir.exists():
        raise SystemExit("Bundle directory not found. Build an annotation bundle first.")

    rows = read_template_rows(bundle_dir)
    box = f"{args.x_center} {args.y_center} {args.width} {args.height}"

    positives_total = 0
    positives_already_labeled = 0
    positives_filled = 0
    positives_skipped_missing_class = 0

    for row in rows:
        if row["is_negative"]:
            continue

        positives_total += 1
        class_id = str(row["class_id"] or "").strip()
        if not class_id:
            positives_skipped_missing_class += 1
            continue

        label_path = bundle_dir / str(row["label_file"])
        current = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        if current and not args.overwrite:
            positives_already_labeled += 1
            continue

        if not args.dry_run:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(f"{class_id} {box}\n", encoding="utf-8")

        positives_filled += 1

    summary = {
        "bundle_dir": rel_or_abs(bundle_dir),
        "box": {
            "x_center": args.x_center,
            "y_center": args.y_center,
            "width": args.width,
            "height": args.height,
        },
        "positives_total": positives_total,
        "positives_already_labeled": positives_already_labeled,
        "positives_filled": positives_filled,
        "positives_skipped_missing_class": positives_skipped_missing_class,
        "overwrite": bool(args.overwrite),
        "dry_run": bool(args.dry_run),
    }

    print("Fallback fill complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
