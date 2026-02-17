#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from repath_model.constants import MODEL_BASENAME

SEMVER_RE = re.compile(r"^(?:v)?(\d+)\.(\d+)\.(\d+)$")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_version(text: str) -> str:
    match = SEMVER_RE.match(text.strip())
    if not match:
        raise SystemExit("--version must be semver, e.g. 1.2.3 or v1.2.3")
    return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semver release bundle for yolo-repath model artifacts.")
    parser.add_argument("--version", required=True, help="Semver version, e.g. 1.2.3")
    parser.add_argument("--model", required=True, help="Path to source .tflite model")
    parser.add_argument("--labels", required=True, help="Path to source .labels.json")
    parser.add_argument("--out-root", default="dist/releases", help="Output release root directory")
    parser.add_argument("--source-run-id", default="", help="Optional run/candidate id")
    parser.add_argument("--notes-file", default="", help="Optional markdown/text notes file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = normalize_version(args.version)
    tag = f"v{version}"

    model_src = Path(args.model).resolve()
    labels_src = Path(args.labels).resolve()
    if not model_src.exists():
        raise SystemExit(f"Model file not found: {model_src}")
    if not labels_src.exists():
        raise SystemExit(f"Labels file not found: {labels_src}")

    out_dir = Path(args.out_root).resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{MODEL_BASENAME}-{tag}.tflite"
    labels_name = f"{MODEL_BASENAME}-{tag}.labels.json"
    manifest_name = f"release-manifest-{tag}.json"

    model_out = out_dir / model_name
    labels_out = out_dir / labels_name
    shutil.copy2(model_src, model_out)
    shutil.copy2(labels_src, labels_out)

    notes_text = ""
    if args.notes_file:
        notes_path = Path(args.notes_file).resolve()
        if notes_path.exists():
            notes_text = notes_path.read_text(encoding="utf-8")

    model_hash = sha256_of(model_out)
    labels_hash = sha256_of(labels_out)

    manifest = {
        "name": MODEL_BASENAME,
        "version": version,
        "tag": tag,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "model": str(model_src),
            "labels": str(labels_src),
            "run_id": args.source_run_id or None,
        },
        "artifacts": [
            {
                "filename": model_name,
                "sha256": model_hash,
                "bytes": model_out.stat().st_size,
                "content_type": "application/octet-stream",
            },
            {
                "filename": labels_name,
                "sha256": labels_hash,
                "bytes": labels_out.stat().st_size,
                "content_type": "application/json",
            },
        ],
        "notes": notes_text,
    }

    manifest_path = out_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text(
        f"{model_hash}  {model_name}\n{labels_hash}  {labels_name}\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "version": version,
        "tag": tag,
        "release_dir": str(out_dir),
        "model": str(model_out),
        "labels": str(labels_out),
        "manifest": str(manifest_path),
        "sha256sums": str(sums_path),
    }, indent=2))


if __name__ == "__main__":
    main()
