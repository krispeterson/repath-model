#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def usage() -> str:
    return (
        "python3 scripts/training/promote_candidate_model.py "
        "[--candidate-dir ml/artifacts/models/candidates/<run-id>] "
        "[--candidate-id <run-id>] "
        "[--from-analysis test/benchmarks/latest-results.candidate.analysis.json] "
        "[--candidates-root ml/artifacts/models/candidates] "
        "[--assets-dir assets/models] [--prefix yolo-repath] "
        "[--write-metadata] [--metadata-path ml/artifacts/models/active-model.json] [--dry-run]"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--candidate-dir", default="")
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--from-analysis", nargs="?", const="")
    parser.add_argument(
        "--candidates-root",
        default=str(Path("ml") / "artifacts" / "models" / "candidates"),
    )
    parser.add_argument("--assets-dir", default=str(Path("assets") / "models"))
    parser.add_argument("--prefix", default="yolo-repath")
    parser.add_argument("--write-metadata", action="store_true")
    parser.add_argument(
        "--metadata-path",
        default=str(Path("ml") / "artifacts" / "models" / "active-model.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def resolve_latest_candidate_dir(root_dir: Path) -> Path | None:
    if not root_dir.exists():
        return None

    dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def resolve_candidate_from_analysis(analysis_path: Path) -> Path | None:
    if not analysis_path.exists():
        return None

    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    model_ref = ""
    if isinstance(payload, dict):
        summary = payload.get("summary")
        if isinstance(summary, dict):
            model_ref = str(summary.get("model") or "")

    if not model_ref.endswith(".tflite"):
        return None

    return Path(model_ref).resolve().parent


def copy_file_if_exists(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        raise SystemExit(f"Required file not found: {src}")
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def file_size_if_exists(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except Exception:
        return None


def resolve_candidate_artifact(candidate_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        artifact = candidate_dir / name
        if artifact.exists():
            return artifact
    return None


def main() -> None:
    args = parse_args()
    candidates_root = Path(args.candidates_root).resolve()

    candidate_dir: Path | None = None
    if args.candidate_dir:
        candidate_dir = Path(args.candidate_dir).resolve()
    elif args.candidate_id:
        candidate_dir = (candidates_root / args.candidate_id).resolve()
    else:
        analysis_value = args.from_analysis
        if analysis_value is None:
            analysis_path = Path("test") / "benchmarks" / "latest-results.candidate.analysis.json"
            candidate_dir = resolve_candidate_from_analysis(analysis_path.resolve())
        elif analysis_value == "":
            analysis_path = Path("test") / "benchmarks" / "latest-results.candidate.analysis.json"
            candidate_dir = resolve_candidate_from_analysis(analysis_path.resolve())
        else:
            candidate_dir = resolve_candidate_from_analysis(Path(analysis_value).resolve())

    if candidate_dir is None:
        candidate_dir = resolve_latest_candidate_dir(candidates_root)

    if candidate_dir is None or not candidate_dir.exists():
        raise SystemExit("Candidate directory not found. Provide --candidate-dir or create candidate artifacts first.")

    model_src = resolve_candidate_artifact(candidate_dir, ["yolo-repath.tflite", "yolov8.tflite"])
    labels_src = resolve_candidate_artifact(candidate_dir, ["yolo-repath.labels.json", "yolov8.labels.json"])

    if model_src is None:
        raise SystemExit(
            f"Candidate model not found in {candidate_dir} (expected yolo-repath.tflite or yolov8.tflite)."
        )
    if labels_src is None:
        raise SystemExit(
            f"Candidate labels not found in {candidate_dir} (expected yolo-repath.labels.json or yolov8.labels.json)."
        )

    assets_dir = Path(args.assets_dir).resolve()
    model_dst = assets_dir / f"{args.prefix}.tflite"
    labels_dst = assets_dir / f"{args.prefix}.labels.json"

    copy_file_if_exists(model_src, model_dst, args.dry_run)
    copy_file_if_exists(labels_src, labels_dst, args.dry_run)

    metadata: dict[str, object] = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": rel_or_abs(candidate_dir),
        "model_source": rel_or_abs(model_src),
        "labels_source": rel_or_abs(labels_src),
        "model_target": rel_or_abs(model_dst),
        "labels_target": rel_or_abs(labels_dst),
        "model_bytes": file_size_if_exists(model_src),
        "labels_bytes": file_size_if_exists(labels_src),
    }

    if args.write_metadata:
        metadata_path = Path(args.metadata_path).resolve()
        if not args.dry_run:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        metadata["metadata_path"] = rel_or_abs(metadata_path)

    print("Candidate promotion summary")
    print(json.dumps({**metadata, "dry_run": bool(args.dry_run)}, indent=2))


if __name__ == "__main__":
    main()
