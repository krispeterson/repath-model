#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resolved benchmark manifest with local cache paths.")
    parser.add_argument(
        "--manifest",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest-v2.json"),
        help="Base manifest path.",
    )
    parser.add_argument(
        "--append-manifest",
        action="append",
        default=[],
        help="Append manifest path (repeatable).",
    )
    parser.add_argument(
        "--completed",
        default=str(Path("test") / "benchmarks" / "benchmark-labeled.csv"),
        help="Completed rows CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("test") / "benchmarks" / "images"),
        help="Cache directory for downloaded/copied images.",
    )
    parser.add_argument(
        "--out",
        default=str(Path("test") / "benchmarks" / "municipal-benchmark-manifest.resolved.json"),
        help="Output manifest path.",
    )
    parser.add_argument("--no-download", action="store_true", help="Disable HTTP downloads.")
    parser.add_argument("--no-copy-local", action="store_true", help="Disable local file copy into cache.")
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_json(file_path: Path):
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_append_images(paths: list[str], cwd: Path) -> dict:
    images = []
    loaded = []
    missing = []

    for manifest_path in paths:
        full_path = Path(manifest_path).resolve()
        if not full_path.exists():
            missing.append(rel_or_abs(full_path, cwd))
            continue
        payload = load_json(full_path)
        rows = payload.get("images") if isinstance(payload, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    images.append(dict(row))
        loaded.append(rel_or_abs(full_path, cwd))

    return {"images": images, "loaded": loaded, "missing": missing}


def load_completed_map(file_path: Path) -> dict[str, str]:
    if not file_path.exists():
        return {}

    rows = file_path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in rows if line.strip()]
    if not rows:
        return {}

    out: dict[str, str] = {}
    # Preserve JS behavior: skip first row as header.
    for line in rows[1:]:
        cols = next(csv.reader([line]))
        name = str(cols[0] if len(cols) > 0 else "").strip()
        url = str(cols[1] if len(cols) > 1 else "").strip()
        if name and url:
            out[name] = url
    return out


def is_http_url(value: str) -> bool:
    return bool(re.match(r"^https?://", str(value or ""), flags=re.IGNORECASE))


def is_file_url(value: str) -> bool:
    return bool(re.match(r"^file://", str(value or ""), flags=re.IGNORECASE))


def sanitize_name(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"^-+|-+$", "", text)
    return text[:120]


def extension_from_url(value: str) -> str:
    match = re.search(r"\.([a-zA-Z0-9]{2,6})(?:[?#].*)?$", str(value or ""))
    if not match:
        return ".jpg"
    return f".{match.group(1).lower()}"


def build_cache_path(cache_dir: Path, image_name: str, source_url: str) -> Path:
    base = sanitize_name(image_name) or "sample"
    ext = extension_from_url(source_url)
    return cache_dir / f"{base}{ext}"


def download_to_cache(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-L",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "20",
            "--max-time",
            "90",
            "--fail",
            url,
            "-o",
            str(out_file),
            "-sS",
        ],
        check=True,
    )


def copy_local_to_cache(local_path: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local_path, out_file)


def resolve_local_path(raw_url: str) -> Path | None:
    value = str(raw_url or "").strip()
    if not value:
        return None
    if is_file_url(value):
        parsed = urlparse(value)
        return Path(unquote(parsed.path))
    if is_http_url(value):
        return None
    return Path(value)


def normalize_label_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    out = [str(entry or "").strip() for entry in value]
    out = [item for item in out if item]
    out.sort()
    return out


def dedupe_exact_rows(rows: list[dict]) -> dict:
    seen = set()
    deduped = []
    removed = 0

    for entry in rows:
        key = json.dumps(
            {
                "name": str(entry.get("name") or "").strip(),
                "url": str(entry.get("url") or "").strip(),
                "status": str(entry.get("status") or "").strip(),
                "expected_any": normalize_label_list(entry.get("expected_any")),
                "expected_all": normalize_label_list(entry.get("expected_all")),
            },
            sort_keys=True,
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(entry)

    return {"rows": deduped, "removed": removed}


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    manifest_path = Path(args.manifest).resolve()
    completed_path = Path(args.completed).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_path = Path(args.out).resolve()

    append_input = list(args.append_manifest)
    default_append = Path("test") / "benchmarks" / "benchmark-manifest.supported-holdout.json"
    if not append_input and default_append.resolve().exists():
        append_input.append(str(default_append))

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    completed_map = load_completed_map(completed_path)

    base_images = manifest.get("images") if isinstance(manifest, dict) else []
    base_images = base_images if isinstance(base_images, list) else []
    append = load_append_images(append_input, cwd)
    images = [entry for entry in base_images if isinstance(entry, dict)] + append["images"]

    updated = []
    downloaded_count = 0
    copied_count = 0

    for entry in images:
        next_entry = dict(entry)
        name = str(next_entry.get("name") or "").strip()
        override = completed_map.get(name)
        source_url = str(override or next_entry.get("url") or "").strip()

        next_entry["source_url"] = source_url or None

        if not source_url:
            next_entry["url"] = ""
            next_entry["status"] = "todo"
            updated.append(next_entry)
            continue

        if is_http_url(source_url):
            out_file = build_cache_path(cache_dir, name, source_url)
            if not out_file.exists():
                if not args.no_download:
                    try:
                        download_to_cache(source_url, out_file)
                        downloaded_count += 1
                    except Exception as error:  # noqa: BLE001
                        next_entry["url"] = ""
                        next_entry["status"] = "todo"
                        next_entry["resolve_error"] = f"download_failed: {error}"
                        updated.append(next_entry)
                        continue
                else:
                    next_entry["url"] = ""
                    next_entry["status"] = "todo"
                    next_entry["resolve_error"] = "download_disabled"
                    updated.append(next_entry)
                    continue

            next_entry["url"] = rel_or_abs(out_file, cwd)
            next_entry["status"] = "ready"
            updated.append(next_entry)
            continue

        local_path = resolve_local_path(source_url)
        if not local_path:
            next_entry["url"] = ""
            next_entry["status"] = "todo"
            next_entry["resolve_error"] = "local_not_found"
            updated.append(next_entry)
            continue

        local_path = local_path.resolve()
        if not local_path.exists():
            next_entry["url"] = ""
            next_entry["status"] = "todo"
            next_entry["resolve_error"] = "local_not_found"
            updated.append(next_entry)
            continue

        if not args.no_copy_local:
            out_file = build_cache_path(cache_dir, name, str(local_path))
            if not out_file.exists():
                copy_local_to_cache(local_path, out_file)
                copied_count += 1
            next_entry["url"] = rel_or_abs(out_file, cwd)
        else:
            next_entry["url"] = rel_or_abs(local_path, cwd)

        next_entry["status"] = "ready"
        updated.append(next_entry)

    dedupe = dedupe_exact_rows(updated)
    images_out = dedupe["rows"]

    output = dict(manifest) if isinstance(manifest, dict) else {}
    output.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "manifest": rel_or_abs(manifest_path, cwd),
                "append_manifests": append["loaded"],
                "missing_append_manifests": append["missing"],
                "completed": rel_or_abs(completed_path, cwd) if completed_path.exists() else None,
                "cache_dir": rel_or_abs(cache_dir, cwd),
            },
            "images": images_out,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    resolved = sum(1 for entry in images_out if str(entry.get("status") or "").lower() == "ready")
    unresolved = sum(1 for entry in images_out if str(entry.get("status") or "").lower() != "ready")

    print("Resolved benchmark manifest generated")
    print(
        json.dumps(
            {
                "resolved": resolved,
                "unresolved": unresolved,
                "downloaded": downloaded_count,
                "copied_local": copied_count,
                "deduped_exact_rows": dedupe["removed"],
                "appended_images": len(append["images"]),
                "append_manifests": append["loaded"],
                "missing_append_manifests": append["missing"],
                "output": rel_or_abs(out_path, cwd),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
