#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a release bundle using release-manifest JSON.")
    parser.add_argument("--manifest", required=True, help="Path to release-manifest-vX.Y.Z.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("Manifest has no artifacts list.")

    release_dir = manifest_path.parent
    failures: list[str] = []

    for artifact in artifacts:
        filename = artifact.get("filename")
        expected_hash = artifact.get("sha256")
        if not filename or not expected_hash:
            failures.append("Artifact entry missing filename or sha256")
            continue
        file_path = release_dir / filename
        if not file_path.exists():
            failures.append(f"Missing file: {file_path}")
            continue
        actual_hash = sha256_of(file_path)
        if actual_hash != expected_hash:
            failures.append(f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}")

    if failures:
        print("Verification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("Verification passed")
    print(json.dumps({
        "manifest": str(manifest_path),
        "artifacts_checked": len(artifacts),
    }, indent=2))


if __name__ == "__main__":
    main()
