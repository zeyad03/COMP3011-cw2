"""Migrate an existing v1.0/v1.1 index to v1.2 in place.

The v1.2 schema (Phase E) carries per-field positions (title vs body).
For an existing v1.1 index — built before the schema change — we can
derive those fields without re-crawling because:

* The DocMeta carries the full title string already.
* :func:`src.indexer.build_index` always emits title tokens *first*,
  then body tokens. So given the title token count L_t for a doc, the
  positions in the combined ``positions`` list with value < L_t are
  title positions and the rest are body positions.

This script reads ``data/index.json``, derives the per-field positions
and lengths in memory, and writes the same path back as a v1.2 index.

Usage::

    python scripts/migrate_index_to_v12.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.storage import load, save  # noqa: E402
from src.tokenizer import tokenize  # noqa: E402

DEFAULT_INDEX = ROOT / "data" / "index.json"


def migrate(path: Path) -> None:
    index = load(path)
    title_lengths: dict[str, int] = {}
    for doc_id, doc in index["docs"].items():
        title_tokens = tokenize(doc["title"])
        title_lengths[doc_id] = len(title_tokens)
        doc["title_length"] = len(title_tokens)
        doc["body_length"] = max(0, doc["length"] - len(title_tokens))

    for entry in index["terms"].values():
        for doc_id, posting in entry["postings"].items():
            split = title_lengths.get(doc_id, 0)
            positions = posting["positions"]
            posting["title_positions"] = [p for p in positions if p < split]
            posting["body_positions"] = [
                p - split for p in positions if p >= split
            ]

    save(index, path)
    print(f"Migrated {path} to v1.2 with per-field positions.")


if __name__ == "__main__":
    migrate(DEFAULT_INDEX)
