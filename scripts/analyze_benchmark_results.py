#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark results and build retraining priority outputs.")
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "latest-results.json"),
        help="Input benchmark results JSON.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-error-analysis.json"),
        help="Output analysis JSON path.",
    )
    parser.add_argument(
        "--template-out",
        default=str(Path("test") / "benchmarks" / "benchmark-retraining-priority.csv"),
        help="Output CSV template path.",
    )
    parser.add_argument("--top", type=int, default=25, help="Top-N labels/pairs to emit.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def sorted_entries(counter: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda pair: pair[1], reverse=True)


def write_priority_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "rank",
        "label",
        "priority_score",
        "expected_count",
        "miss_count",
        "hit_count",
        "false_positive_count",
        "hit_rate",
        "recommended_action",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    idx,
                    row.get("label", ""),
                    row.get("priority_score", 0),
                    row.get("expected_count", 0),
                    row.get("miss_count", 0),
                    row.get("hit_count", 0),
                    row.get("false_positive_count", 0),
                    row.get("hit_rate", 0),
                    row.get("recommended_action", ""),
                    "",
                ]
            )


def main() -> None:
    args = parse_args()
    if args.top < 1:
        args.top = 25

    cwd = Path.cwd()
    input_path = Path(args.input).resolve()
    out_path = Path(args.out).resolve()
    template_path = Path(args.template_out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input results file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else []
    if not isinstance(results, list):
        results = []

    expected_count: dict[str, int] = {}
    hit_count: dict[str, int] = {}
    miss_count: dict[str, int] = {}
    fp_count: dict[str, int] = {}
    pair_fp: dict[str, int] = {}

    for row in results:
        if not isinstance(row, dict):
            continue
        expected_any = row.get("expected_any") or []
        predicted_labels = row.get("predicted_labels") or []

        expected = {
            str(v).strip() for v in expected_any if str(v).strip()
        }
        predicted = {
            str(v).strip() for v in predicted_labels if str(v).strip()
        }

        for label in expected:
            expected_count[label] = expected_count.get(label, 0) + 1
            if label in predicted:
                hit_count[label] = hit_count.get(label, 0) + 1
            else:
                miss_count[label] = miss_count.get(label, 0) + 1

        for pred_label in predicted:
            if pred_label in expected:
                continue
            fp_count[pred_label] = fp_count.get(pred_label, 0) + 1
            for exp_label in expected:
                key = f"{exp_label} -> {pred_label}"
                pair_fp[key] = pair_fp.get(key, 0) + 1

    labels = set(expected_count.keys()) | set(fp_count.keys())
    rows: list[dict] = []
    for label in labels:
        expected = expected_count.get(label, 0)
        miss = miss_count.get(label, 0)
        hit = hit_count.get(label, 0)
        fp = fp_count.get(label, 0)
        hit_rate = (hit / expected) if expected > 0 else 0
        priority = miss * 2 + fp + ((1 - hit_rate) * expected if expected > 0 else 0)
        action = "collect_more_positives" if miss >= fp else "add_hard_negatives"

        rows.append(
            {
                "label": label,
                "priority_score": round(priority, 2),
                "expected_count": expected,
                "miss_count": miss,
                "hit_count": hit,
                "false_positive_count": fp,
                "hit_rate": round(hit_rate, 4),
                "recommended_action": action,
            }
        )

    rows.sort(
        key=lambda r: (
            -float(r.get("priority_score", 0)),
            -int(r.get("miss_count", 0)),
            -int(r.get("false_positive_count", 0)),
        )
    )
    top_rows = rows[: args.top]

    out_payload = {
        "source": rel_or_abs(input_path, cwd),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": payload.get("summary", {}) if isinstance(payload, dict) else {},
        "counts": {
            "result_rows": len(results),
            "expected_labels": len(expected_count),
            "false_positive_labels": len(fp_count),
        },
        "top_missed_labels": [
            {"label": label, "count": count}
            for label, count in sorted_entries(miss_count)[: args.top]
        ],
        "top_false_positive_labels": [
            {"label": label, "count": count}
            for label, count in sorted_entries(fp_count)[: args.top]
        ],
        "top_confusion_pairs": [
            {"pair": pair, "count": count}
            for pair, count in sorted_entries(pair_fp)[: args.top]
        ],
        "priority_table": top_rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    write_priority_csv(template_path, top_rows)

    print("Benchmark error analysis generated")
    print(
        json.dumps(
            {
                "input": rel_or_abs(input_path, cwd),
                "output": rel_or_abs(out_path, cwd),
                "template_output": rel_or_abs(template_path, cwd),
                "rows_analyzed": len(results),
                "priority_rows": len(top_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
