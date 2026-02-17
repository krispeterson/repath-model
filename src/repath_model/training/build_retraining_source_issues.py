#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = ["name", "url", "item_id", "canonical_label", "source", "notes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build retraining source issue log by resolving seeded issue names against benchmark-labeled CSV.",
        usage=(
            "python3 scripts/training/build_retraining_source_issues.py "
            "[--input test/benchmarks/benchmark-labeled.csv] "
            "[--seed test/benchmarks/retraining-positive-source-issues.seed.json] "
            "[--out test/benchmarks/retraining-positive-source-issues.json]"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Input benchmark-labeled CSV path.",
    )
    parser.add_argument(
        "--seed",
        default=str(Path("test") / "benchmarks" / "retraining-positive-source-issues.seed.json"),
        help="Seed issue JSON path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "retraining-positive-source-issues.json"),
        help="Output issue JSON path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized = {column: str((row or {}).get(column, "")).strip() for column in CSV_COLUMNS}
            rows.append(normalized)
        return rows


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    input_path = Path(args.input).resolve()
    seed_path = Path(args.seed).resolve()
    out_path = Path(args.out).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}")

    rows = read_csv_rows(input_path)
    by_name = {str(row.get("name") or ""): row for row in rows}
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    issues = seed.get("issues") if isinstance(seed, dict) else []
    issues = issues if isinstance(issues, list) else []

    resolved = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue_name = str(issue.get("name") or "").strip()
        row = by_name.get(issue_name)
        resolved.append(
            {
                "name": issue_name,
                "status": "mapped_to_replacement" if row else "not_found_in_csv",
                "issue_type": str(issue.get("issue_type") or "source_content_mismatch"),
                "expected_label": str(issue.get("expected_label") or ""),
                "original_problem_summary": str(issue.get("original_problem_summary") or ""),
                "current": (
                    {
                        "url": str(row.get("url") or ""),
                        "source": str(row.get("source") or ""),
                        "notes": str(row.get("notes") or ""),
                    }
                    if row
                    else None
                ),
            }
        )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": rel_or_abs(seed_path, cwd),
        "source_csv": rel_or_abs(input_path, cwd),
        "summary": {
            "issue_count": len(resolved),
            "mapped_to_replacement": sum(1 for issue in resolved if issue.get("status") == "mapped_to_replacement"),
            "not_found_in_csv": sum(1 for issue in resolved if issue.get("status") == "not_found_in_csv"),
        },
        "issues": resolved,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("Retraining source issue log generated")
    print(
        json.dumps(
            {
                "seed": output["seed"],
                "source_csv": output["source_csv"],
                "output": rel_or_abs(out_path, cwd),
                "summary": output["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
