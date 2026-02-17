#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


HEADER = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retraining queue CSV from benchmark retraining priority CSV.",
        usage=(
            "python3 scripts/evaluation/build_retraining_queue.py "
            "[--priority-csv test/benchmarks/benchmark-retraining-priority.csv] "
            "[--out test/benchmarks/benchmark-retraining-queue.csv] "
            "[--positive-top 8] [--negative-top 4] [--variants 3]"
        ),
    )
    parser.add_argument(
        "--priority-csv",
        default=str(Path("test") / "benchmarks" / "benchmark-retraining-priority.csv"),
        help="Input priority CSV path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-retraining-queue.csv"),
        help="Output retraining queue CSV path.",
    )
    parser.add_argument("--positive-top", type=int, default=8, help="Top positive labels to include.")
    parser.add_argument("--negative-top", type=int, default=4, help="Top negative labels to include.")
    parser.add_argument("--variants", type=int, default=3, help="Rows per label.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    return normalized.strip("-")[:80]


def to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_priority_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            label = str((row or {}).get("label") or "").strip()
            if not label:
                continue
            rows.append(
                {
                    "label": label,
                    "priority_score": to_float((row or {}).get("priority_score"), 0.0),
                    "recommended_action": str((row or {}).get("recommended_action") or "").strip(),
                    "miss_count": to_int((row or {}).get("miss_count"), 0),
                    "false_positive_count": to_int((row or {}).get("false_positive_count"), 0),
                }
            )
        return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HEADER})


def main() -> None:
    args = parse_args()
    args.positive_top = max(1, int(args.positive_top or 8))
    args.negative_top = max(1, int(args.negative_top or 4))
    args.variants = max(1, int(args.variants or 3))

    cwd = Path.cwd()
    priority_path = Path(args.priority_csv).resolve()
    out_path = Path(args.out).resolve()

    if not priority_path.exists():
        raise SystemExit(f"Priority CSV not found: {priority_path}")

    priority_rows = read_priority_rows(priority_path)

    positives = sorted(
        [row for row in priority_rows if row["recommended_action"] == "collect_more_positives"],
        key=lambda row: (-row["priority_score"], -row["miss_count"]),
    )[: args.positive_top]

    negatives = sorted(
        [row for row in priority_rows if row["recommended_action"] == "add_hard_negatives"],
        key=lambda row: (-row["priority_score"], -row["false_positive_count"]),
    )[: args.negative_top]

    out_rows = []

    for row in positives:
        slug = slugify(row["label"]) or "label"
        for index in range(1, args.variants + 1):
            out_rows.append(
                {
                    "name": f"retrain_positive_{slug}_v{index}",
                    "url": "",
                    "item_id": f"retrain-{slug}-v{index}",
                    "canonical_label": row["label"],
                    "source": "retraining_queue",
                    "notes": f"action=collect_more_positives; priority_score={row['priority_score']}; variant={index}",
                }
            )

    for row in negatives:
        slug = slugify(row["label"]) or "label"
        for index in range(1, args.variants + 1):
            out_rows.append(
                {
                    "name": f"retrain_negative_{slug}_v{index}",
                    "url": "",
                    "item_id": f"retrain-negative-{slug}-v{index}",
                    "canonical_label": "",
                    "source": "retraining_queue_negative",
                    "notes": (
                        f"target_false_positive_label={row['label']}; action=add_hard_negatives; "
                        f"priority_score={row['priority_score']}; variant={index}"
                    ),
                }
            )

    write_rows(out_path, out_rows)

    print("Retraining queue generated")
    print(
        json.dumps(
            {
                "input": rel_or_abs(priority_path, cwd),
                "output": rel_or_abs(out_path, cwd),
                "positive_labels_selected": len(positives),
                "negative_labels_selected": len(negatives),
                "variants_per_label": args.variants,
                "queue_rows": len(out_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
