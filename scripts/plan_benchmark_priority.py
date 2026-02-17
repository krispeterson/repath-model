#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan benchmark labeling priorities based on taxonomy risk heuristics.")
    parser.add_argument(
        "--taxonomy",
        default=str(Path("assets") / "models" / "municipal-taxonomy-v1.json"),
        help="Taxonomy JSON path.",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Benchmark manifest JSON path.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "benchmark-priority-report.json"),
        help="Priority report output path.",
    )
    parser.add_argument("--top", type=int, default=50, help="Top-N candidate rows in summary output.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tokens(value: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return [token for token in normalized.split() if token]


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_token_frequency(records: list[dict]) -> dict[str, int]:
    stopwords = {"and", "or", "the", "with", "for", "other", "than", "to", "of", "in", "on", "a", "an"}
    freq: dict[str, int] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        tokens = unique(
            [
                token
                for token in normalize_tokens(str(record.get("canonical_label") or ""))
                if len(token) > 2 and token not in stopwords
            ]
        )
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

    return freq


def band_for_score(score: float) -> str:
    if score >= 70:
        return "urgent"
    if score >= 50:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def first_label(entry: dict) -> str:
    expected_any = entry.get("expected_any") if isinstance(entry, dict) else []
    expected_all = entry.get("expected_all") if isinstance(entry, dict) else []
    expected_any = expected_any if isinstance(expected_any, list) else []
    expected_all = expected_all if isinstance(expected_all, list) else []

    if expected_any:
        return str(expected_any[0] or "").strip()
    if expected_all:
        return str(expected_all[0] or "").strip()
    return ""


def score_candidate(record: dict, token_freq: dict[str, int]) -> dict:
    reasons: list[str] = []
    score = 0

    label = str(record.get("canonical_label") or "")
    label_lower = label.lower()
    outcomes = record.get("outcomes") if isinstance(record.get("outcomes"), list) else []
    primary = str(record.get("primary_outcome") or "").strip()

    outcome_weights = {
        "dropoff_hhw": 36,
        "dropoff_other": 20,
        "trash": 14,
        "compost": 10,
        "dropoff_recycle": 12,
        "curbside_recycle": 8,
        "reuse": 6,
    }

    if primary and primary in outcome_weights:
        points = outcome_weights[primary]
        score += points
        reasons.append(f"{primary} outcome (+{points})")

    if len(outcomes) > 1:
        multi_points = 6 + min(10, (len(outcomes) - 1) * 2)
        score += multi_points
        reasons.append(f"multiple disposal options ({len(outcomes)}) (+{multi_points})")

    hazard_regex = re.compile(
        r"(battery|paint|oil|antifreeze|ammunition|explosive|flammable|propane|pesticide|chemical|"
        r"solvent|fire extinguisher|electronics|mercury|medication|pharmaceutical|syringe|needle|"
        r"sharps|engine coolant)",
        flags=re.IGNORECASE,
    )
    if hazard_regex.search(label_lower):
        score += 28
        reasons.append("hazard-adjacent item (+28)")

    ambiguous_terms = [
        "container",
        "bottle",
        "can",
        "box",
        "bag",
        "tray",
        "tub",
        "carton",
        "cup",
        "jar",
        "lid",
        "cap",
        "wrapper",
        "packaging",
    ]
    ambiguous_hits = sum(1 for term in ambiguous_terms if term in label_lower)
    if ambiguous_hits > 0:
        points = min(18, ambiguous_hits * 4)
        score += points
        reasons.append(f"visually ambiguous shape terms ({ambiguous_hits}) (+{points})")

    family_matches = {
        "plastic": bool(re.search(r"(plastic|styrofoam|polystyrene|foam|bubble wrap|blister)", label_lower)),
        "paper": bool(re.search(r"(paper|cardboard|carton|book|magazine|box|envelope|wrapping)", label_lower)),
        "metal": bool(re.search(r"(aluminum|tin|metal|steel|foil|aerosol)", label_lower)),
        "glass": "glass" in label_lower,
    }
    family_count = sum(1 for match in family_matches.values() if match)
    if family_count > 1:
        points = family_count * 4
        score += points
        reasons.append(f"cross-material ambiguity ({family_count} families) (+{points})")

    if "other" in label_lower or "mixed" in label_lower:
        score += 7
        reasons.append("broad/other category naming (+7)")

    tokens = unique([token for token in normalize_tokens(label) if len(token) > 2])
    if tokens:
        crowding_raw = sum(max(0, token_freq.get(token, 1) - 1) for token in tokens) / len(tokens)
        crowding_points = min(20, round(crowding_raw * 2))
        if crowding_points > 0:
            score += crowding_points
            reasons.append(f"label token crowding (+{crowding_points})")

    return {
        "score": score,
        "priority_band": band_for_score(score),
        "reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    if args.top < 1:
        args.top = 50

    cwd = Path.cwd()
    taxonomy_path = Path(args.taxonomy).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()

    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")

    taxonomy = load_json(taxonomy_path)
    manifest = load_json(manifest_path)

    classes = taxonomy.get("vision_classes") if isinstance(taxonomy, dict) else []
    classes = classes if isinstance(classes, list) else []
    images = manifest.get("images") if isinstance(manifest, dict) else []
    images = images if isinstance(images, list) else []

    by_label = {}
    for record in classes:
        if not isinstance(record, dict):
            continue
        label = str(record.get("canonical_label") or "").strip()
        if label:
            by_label[label] = record

    token_freq = build_token_frequency(classes)

    todo_candidates = []
    for entry in images:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("status") or "").lower() != "todo":
            continue

        label = first_label(entry)
        record = by_label.get(label)
        if not record:
            todo_candidates.append(
                {
                    "name": entry.get("name"),
                    "item_id": entry.get("item_id"),
                    "canonical_label": label or None,
                    "status": "unmapped",
                    "priority_score": 0,
                    "priority_band": "low",
                    "reasons": ["label not mapped to taxonomy"],
                }
            )
            continue

        scored = score_candidate(record, token_freq)
        todo_candidates.append(
            {
                "name": entry.get("name"),
                "item_id": record.get("item_id"),
                "canonical_label": record.get("canonical_label"),
                "primary_outcome": record.get("primary_outcome"),
                "outcomes": record.get("outcomes"),
                "required": bool(entry.get("required")),
                "priority_score": scored["score"],
                "priority_band": scored["priority_band"],
                "reasons": scored["reasons"],
            }
        )

    todo_candidates.sort(
        key=lambda row: (
            -float(row.get("priority_score") or 0),
            str(row.get("canonical_label") or ""),
        )
    )

    top_n = todo_candidates[: args.top]
    band_counts: dict[str, int] = {}
    for row in todo_candidates:
        band = str(row.get("priority_band") or "")
        band_counts[band] = band_counts.get(band, 0) + 1

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "taxonomy": rel_or_abs(taxonomy_path, cwd),
            "manifest": rel_or_abs(manifest_path, cwd),
        },
        "summary": {
            "todo_candidates": len(todo_candidates),
            "requested_top_n": args.top,
            "priority_band_counts": band_counts,
        },
        "top_candidates": top_n,
        "all_candidates": todo_candidates,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Benchmark labeling priority summary")
    print(
        json.dumps(
            {
                "todo_candidates": report["summary"]["todo_candidates"],
                "priority_band_counts": report["summary"]["priority_band_counts"],
                "top_5": [
                    {
                        "canonical_label": row.get("canonical_label"),
                        "score": row.get("priority_score"),
                        "band": row.get("priority_band"),
                    }
                    for row in top_n[:5]
                ],
            },
            indent=2,
        )
    )
    print(f"Saved report to {rel_or_abs(out_path, cwd)}")


if __name__ == "__main__":
    main()
