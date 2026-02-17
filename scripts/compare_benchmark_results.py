#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate benchmark result files.")
    parser.add_argument(
        "--baseline",
        default=str(Path("test") / "benchmarks" / "latest-results.json"),
        help="Baseline benchmark results path.",
    )
    parser.add_argument(
        "--candidate",
        default=str(Path("test") / "benchmarks" / "latest-results.candidate.json"),
        help="Candidate benchmark results path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "latest-results.compare.json"),
        help="Comparison output path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def to_number(value):
    try:
        n = float(value)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def delta(a, b):
    if a is None or b is None:
        return None
    return round(b - a, 4)


def normalize_labels(value) -> list[str]:
    if not isinstance(value, list):
        return []
    labels = [str(entry or "").strip() for entry in value]
    labels = [label for label in labels if label]
    labels.sort()
    return labels


def row_key(row: dict) -> str:
    payload = {
        "name": str((row or {}).get("name") or "").strip(),
        "url": str((row or {}).get("url") or "").strip(),
        "expected_any": normalize_labels((row or {}).get("expected_any")),
        "expected_all": normalize_labels((row or {}).get("expected_all")),
    }
    return json.dumps(payload, sort_keys=True)


def pr_stats(predicted_labels, expected_labels) -> tuple[int, int, int]:
    pred = set(predicted_labels if isinstance(predicted_labels, list) else [])
    exp = set(expected_labels if isinstance(expected_labels, list) else [])

    tp = sum(1 for label in exp if label in pred)
    fp = len(pred) - tp
    fn = len(exp) - tp
    return tp, fp, fn


def summarize_rows(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    any_cases = any_hits = 0
    negative_cases = negative_clean = 0

    for row in rows:
        expected_any = normalize_labels(row.get("expected_any"))
        expected_all = normalize_labels(row.get("expected_all"))
        predicted = normalize_labels(row.get("predicted_labels"))
        expected_set = expected_all if expected_all else expected_any

        row_tp, row_fp, row_fn = pr_stats(predicted, expected_set)
        tp += row_tp
        fp += row_fp
        fn += row_fn

        if expected_any:
            any_cases += 1
            pred_set = set(predicted)
            if any(label in pred_set for label in expected_any):
                any_hits += 1
        else:
            negative_cases += 1
            if not predicted:
                negative_clean += 1

    micro_precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    micro_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    any_hit_rate = (any_hits / any_cases) if any_cases > 0 else 0
    negative_clean_rate = (negative_clean / negative_cases) if negative_cases > 0 else 0

    return {
        "images_evaluated": len(rows),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "any_hit_rate": round(any_hit_rate, 4),
        "negative_clean_rate": round(negative_clean_rate, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def build_comparison(before_summary: dict, after_summary: dict) -> dict:
    fields = [
        "images_evaluated",
        "micro_precision",
        "micro_recall",
        "any_hit_rate",
        "negative_clean_rate",
        "tp",
        "fp",
        "fn",
        "skipped_unsupported_entries",
    ]

    out: dict[str, dict] = {}
    for field in fields:
        before = to_number(before_summary.get(field))
        after = to_number(after_summary.get(field))
        out[field] = {
            "baseline": before,
            "candidate": after,
            "delta": delta(before, after),
        }
    return out


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    out_path = Path(args.out).resolve()

    if not baseline_path.exists():
        raise SystemExit(f"Baseline results not found: {baseline_path}")
    if not candidate_path.exists():
        raise SystemExit(f"Candidate results not found: {candidate_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    baseline_summary = baseline.get("summary", {}) if isinstance(baseline, dict) else {}
    candidate_summary = candidate.get("summary", {}) if isinstance(candidate, dict) else {}
    comparison = build_comparison(baseline_summary, candidate_summary)

    baseline_results = baseline.get("results", []) if isinstance(baseline, dict) else []
    candidate_results = candidate.get("results", []) if isinstance(candidate, dict) else []
    baseline_results = baseline_results if isinstance(baseline_results, list) else []
    candidate_results = candidate_results if isinstance(candidate_results, list) else []

    baseline_by_key = {row_key(row): row for row in baseline_results if isinstance(row, dict)}
    candidate_by_key = {row_key(row): row for row in candidate_results if isinstance(row, dict)}
    overlap_keys = [key for key in baseline_by_key.keys() if key in candidate_by_key]

    baseline_overlap_rows = [baseline_by_key[key] for key in overlap_keys]
    candidate_overlap_rows = [candidate_by_key[key] for key in overlap_keys]

    baseline_overlap_summary = summarize_rows(baseline_overlap_rows)
    candidate_overlap_summary = summarize_rows(candidate_overlap_rows)
    overlap_comparison = build_comparison(baseline_overlap_summary, candidate_overlap_summary)

    out_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": rel_or_abs(baseline_path, cwd),
        "candidate": rel_or_abs(candidate_path, cwd),
        "comparison": comparison,
        "overlap": {
            "rows": len(overlap_keys),
            "baseline_summary": baseline_overlap_summary,
            "candidate_summary": candidate_overlap_summary,
            "comparison": overlap_comparison,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

    print("Benchmark comparison generated")
    print(
        json.dumps(
            {
                "baseline": out_payload["baseline"],
                "candidate": out_payload["candidate"],
                "output": rel_or_abs(out_path, cwd),
                "metrics": comparison,
                "overlap_rows": len(overlap_keys),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
