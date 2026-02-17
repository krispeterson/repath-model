#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap Kaggle household waste dataset into the repo by symlink or copy.",
        usage=(
            "python3 scripts/data/bootstrap_kaggle_dataset.py "
            "[--source /path/to/kaggle/images/images] "
            "[--target ml/artifacts/datasets/kaggle-household-waste/images/images] "
            "[--mode symlink|copy] [--force]"
        ),
    )
    parser.add_argument("--source", default=os.environ.get("KAGGLE_WASTE_DIR", ""), help="Source Kaggle images root.")
    parser.add_argument(
        "--target",
        default=str(Path("ml") / "artifacts" / "datasets" / "kaggle-household-waste" / "images" / "images"),
        help="Target path inside repo.",
    )
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink", help="Bootstrap mode.")
    parser.add_argument("--force", action="store_true", help="Replace existing target if present.")
    return parser.parse_args()


def resolve_source_path(input_source: str) -> Path | None:
    if input_source:
        return Path(input_source).expanduser().resolve()

    for candidate in [
        Path("..") / "Kaggle Household Waste Images" / "images" / "images",
        Path("Kaggle Household Waste Images") / "images" / "images",
    ]:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    return None


def validate_source(source_path: Path) -> None:
    if not source_path.exists():
        raise SystemExit("Kaggle source directory not found. Pass --source or set KAGGLE_WASTE_DIR.")
    if not source_path.is_dir():
        raise SystemExit(f"Kaggle source must be a directory: {source_path}")

    subdirs = [entry for entry in source_path.iterdir() if entry.is_dir()]
    if not subdirs:
        raise SystemExit(f"Kaggle source appears empty or unexpected (no class subfolders): {source_path}")


def is_path_inside_repo(repo_root: Path, target_path: Path) -> bool:
    try:
        target_path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def ensure_clean_target(target_path: Path, force: bool) -> None:
    if not target_path.exists() and not target_path.is_symlink():
        return

    if not force:
        raise SystemExit(f"Target already exists: {target_path}. Re-run with --force to replace it.")

    if target_path.is_symlink() or target_path.is_file():
        target_path.unlink(missing_ok=True)
        return

    shutil.rmtree(target_path, ignore_errors=True)


def copy_recursive(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)


def link_dataset(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.symlink_to(source_path, target_is_directory=True)


def rel_to_cwd(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace(os.sep, "/")
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    source_path = resolve_source_path(args.source)
    if source_path is None:
        raise SystemExit("Kaggle source directory not found. Pass --source or set KAGGLE_WASTE_DIR.")

    target_path = Path(args.target).expanduser().resolve()

    validate_source(source_path)

    if not is_path_inside_repo(repo_root, target_path):
        raise SystemExit(f"Target must be inside repo: {target_path}")

    ensure_clean_target(target_path, args.force)

    if args.mode == "copy":
        copy_recursive(source_path, target_path)
    else:
        try:
            link_dataset(source_path, target_path)
        except OSError:
            # Symlink permissions can fail on some systems; fallback to copy.
            copy_recursive(source_path, target_path)
            args.mode = "copy"

    print("Kaggle dataset bootstrap complete")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "source": str(source_path),
                "target": rel_to_cwd(target_path),
                "reminder": "You can now run `npm run ml:labeling:ingest` without passing --kaggle-dir.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
