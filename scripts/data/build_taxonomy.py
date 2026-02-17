#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build municipal taxonomy JSON from pack JSON data.",
        usage="python3 scripts/data/build_taxonomy.py [--pack assets/packs/<pack-id>.pack.json] [--out assets/models/municipal-taxonomy-v1.json]",
    )
    parser.add_argument("--pack", default=None, help="Pack JSON path.")
    parser.add_argument(
        "--out",
        default=str(Path("assets") / "models" / "municipal-taxonomy-v1.json"),
        help="Output taxonomy JSON path.",
    )
    return parser.parse_args()


def rel_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def find_default_pack_path() -> Path | None:
    packs_dir = (Path("assets") / "packs").resolve()
    if not packs_dir.exists() or not packs_dir.is_dir():
        return None
    entries = sorted([entry for entry in packs_dir.iterdir() if entry.name.endswith(".pack.json")])
    return entries[0] if entries else None


def normalize_alias(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def to_slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower())
    text = re.sub(r"^-+|-+$", "", text)
    return text[:80]


def unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def get_primary_outcome(option_cards: list[dict]) -> str | None:
    cards = option_cards[:] if isinstance(option_cards, list) else []
    if not cards:
        return None

    def priority_value(card: dict) -> int:
        value = card.get("priority") if isinstance(card, dict) else None
        return int(value) if isinstance(value, (int, float)) else 999

    cards.sort(key=priority_value)
    first = cards[0] if cards else {}
    kind = first.get("kind") if isinstance(first, dict) else None
    return kind if isinstance(kind, str) and kind.strip() else None


def get_item_record(item: dict) -> dict:
    item_id = str(item.get("id") or "").strip()
    name = str(item.get("name") or "").strip()
    keywords = [str(k or "").strip() for k in (item.get("keywords") if isinstance(item.get("keywords"), list) else [])]
    keywords = [k for k in keywords if k]
    aliases = unique([name, *keywords])

    option_cards = item.get("option_cards") if isinstance(item.get("option_cards"), list) else []
    outcomes = unique([str(card.get("kind") or "").strip() for card in option_cards if isinstance(card, dict)])
    outcomes = [outcome for outcome in outcomes if outcome]
    primary_outcome = get_primary_outcome(option_cards)

    return {
        "item_id": item_id,
        "canonical_label": name,
        "class_id": to_slug(name or item_id),
        "aliases": aliases,
        "normalized_aliases": unique([normalize_alias(alias) for alias in aliases]),
        "outcomes": outcomes,
        "primary_outcome": primary_outcome,
        "option_card_ids": [
            str(card.get("id") or "").strip()
            for card in option_cards
            if isinstance(card, dict) and str(card.get("id") or "").strip()
        ],
    }


def create_alias_index(item_records: list[dict]) -> dict[str, list[str]]:
    alias_map: dict[str, set[str]] = {}
    for record in item_records:
        item_id = str(record.get("item_id") or "").strip()
        for alias in (record.get("normalized_aliases") if isinstance(record.get("normalized_aliases"), list) else []):
            alias = str(alias or "").strip()
            if not alias:
                continue
            alias_map.setdefault(alias, set()).add(item_id)

    out: dict[str, list[str]] = {}
    for alias in sorted(alias_map.keys()):
        out[alias] = sorted(alias_map[alias])
    return out


def create_outcome_counts(item_records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in item_records:
        for kind in (record.get("outcomes") if isinstance(record.get("outcomes"), list) else []):
            text = str(kind or "").strip()
            if not text:
                continue
            counts[text] = counts.get(text, 0) + 1
    return {key: counts[key] for key in sorted(counts.keys())}


def main() -> None:
    args = parse_args()
    cwd = Path.cwd()

    pack_path = Path(args.pack).resolve() if args.pack else find_default_pack_path()
    out_path = Path(args.out).resolve()

    if not pack_path or not pack_path.exists():
        raise SystemExit(f"Pack file not found: {pack_path}")

    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    items = pack.get("items") if isinstance(pack, dict) else []
    items = items if isinstance(items, list) else []

    item_records = [get_item_record(item) for item in items if isinstance(item, dict)]
    item_records = [record for record in item_records if record["item_id"] and record["canonical_label"]]
    item_records.sort(key=lambda record: str(record.get("canonical_label") or ""))

    taxonomy = {
        "taxonomy_id": "municipal-taxonomy-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "pack_id": pack.get("pack_id") if isinstance(pack, dict) else None,
            "pack_version": pack.get("pack_version") if isinstance(pack, dict) else None,
            "retrieved_at": pack.get("retrieved_at") if isinstance(pack, dict) else None,
            "municipality": pack.get("municipality") if isinstance(pack, dict) else None,
            "pack_path": rel_or_abs(pack_path, cwd),
        },
        "summary": {
            "item_count": len(item_records),
            "alias_count": sum(len(record.get("aliases") or []) for record in item_records),
            "normalized_alias_count": sum(len(record.get("normalized_aliases") or []) for record in item_records),
            "outcome_counts": create_outcome_counts(item_records),
        },
        "vision_classes": item_records,
        "alias_index": create_alias_index(item_records),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(taxonomy, indent=2) + "\n", encoding="utf-8")

    print(f"Generated taxonomy for {len(item_records)} items.")
    print(f"- {rel_or_abs(out_path, cwd)}")


if __name__ == "__main__":
    main()
