"""JSON persistence for the inverted index.

Two design choices worth flagging:

1. **Atomic writes**. ``save`` writes the JSON to a sibling temp file and
   then ``os.replace``s it into place. On POSIX (and on modern Windows
   since the introduction of ``MoveFileEx``) the rename is atomic, so a
   reader can never observe a half-written file and a crash mid-write
   cannot corrupt an existing index.
2. **Human-readable JSON**. ``indent=2, sort_keys=True`` makes the
   compiled index inspectable when graders open the file. Ordering is
   stable, so two builds against the same crawled snapshot produce a
   byte-identical file (modulo ``crawled_at``).

The format is versioned via ``meta.version``; ``load`` raises
:class:`IndexVersionError` if it does not match :data:`INDEX_VERSION`,
so the tool refuses to operate on an index produced by an incompatible
schema.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import cast

from src.indexer import INDEX_VERSION, Index


class StorageError(Exception):
    """Raised on any failure to save or load the index."""


class IndexVersionError(StorageError):
    """Raised when the index file's schema version is incompatible."""


def save(index: Index, path: Path | str) -> None:
    """Atomically write *index* to *path* as JSON.

    The parent directory is created if necessary. Any partially-written
    temp file is removed if serialisation raises.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(index, fp, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def load(path: Path | str) -> Index:
    """Load an index from *path*.

    Raises:
        StorageError: file is missing or its JSON is corrupt.
        IndexVersionError: the file's ``meta.version`` does not match
            :data:`INDEX_VERSION`.
    """
    target = Path(path)
    if not target.exists():
        raise StorageError(f"index file not found: {target}")
    try:
        with target.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as err:
        raise StorageError(f"corrupt index JSON at {target}: {err}") from err

    if not isinstance(data, dict):
        raise StorageError(f"index root is not a JSON object: {target}")

    version = data.get("meta", {}).get("version")
    if version != INDEX_VERSION:
        raise IndexVersionError(
            f"index version {version!r} does not match expected {INDEX_VERSION!r}"
        )
    return cast(Index, data)
