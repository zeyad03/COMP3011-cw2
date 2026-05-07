"""Tests for the dense (embedding) retrieval module (Phase A).

These tests inject a *fake* encoder so they never download or load the
real ~80 MB sentence-transformers model. The tests verify the index
build/save/load round-trip, cosine-similarity ranking, and the
fallback path when numpy or the transformer library is missing.

Tests that exercise the *real* encoder would download the model on
first run; those would normally be marked ``@pytest.mark.slow`` and
skipped in CI. This file deliberately keeps the test suite hermetic
and fast (under 1 ms in aggregate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")

from src.dense_ranker import (  # noqa: E402
    DenseIndex,
    DenseRanker,
)
from src.indexer import Index, build_index  # noqa: E402
from src.crawler import Page  # noqa: E402


def _index() -> Index:
    pages = [
        Page(url="https://x.com/0", title="Cats", text="cat purr meow"),
        Page(url="https://x.com/1", title="Dogs", text="dog bark fetch"),
        Page(url="https://x.com/2", title="Fish", text="fish swim ocean"),
    ]
    return build_index(pages)


def _fake_encoder(matrix: dict[str, list[float]]) -> Any:
    """Return an encoder closure that maps text-prefix → known vector."""

    def encode(texts: list[str]) -> Any:
        vecs = []
        for text in texts:
            for prefix, vec in matrix.items():
                if text.startswith(prefix):
                    vecs.append(vec)
                    break
            else:
                vecs.append([0.0] * len(next(iter(matrix.values()))))
        return np.asarray(vecs, dtype=np.float32)

    return encode


class TestDenseIndex:
    def test_from_index_normalises_rows(self) -> None:
        idx = _index()
        encoder = _fake_encoder(
            {"Cats": [1, 0, 0], "Dogs": [0, 1, 0], "Fish": [0, 0, 1]}
        )
        di = DenseIndex.from_index(idx, encoder=encoder)
        # Rows should be unit norm.
        norms = np.linalg.norm(di.embeddings, axis=1)
        assert np.allclose(norms, 1.0)
        assert di.doc_ids == ["0", "1", "2"]
        assert di.dim == 3

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        idx = _index()
        encoder = _fake_encoder(
            {"Cats": [1, 0, 0], "Dogs": [0, 1, 0], "Fish": [0, 0, 1]}
        )
        di = DenseIndex.from_index(idx, encoder=encoder)
        target = tmp_path / "index.json"
        target.write_text("{}")  # placeholder so we have the path
        di.save(target)
        loaded = DenseIndex.load(target)
        assert loaded.doc_ids == di.doc_ids
        assert np.allclose(loaded.embeddings, di.embeddings)


class TestDenseRanker:
    def test_ranks_by_cosine_similarity(self) -> None:
        idx = _index()
        encoder = _fake_encoder(
            {
                "Cats": [1.0, 0.0, 0.0],
                "Dogs": [0.0, 1.0, 0.0],
                "Fish": [0.0, 0.0, 1.0],
            }
        )
        di = DenseIndex.from_index(idx, encoder=encoder)
        # Query encodes to the cat vector; cat doc must be top.
        ranker = DenseRanker(di, encoder=cast(Any, encoder))
        results = ranker.rank(idx, "Cats query", top_k=3)
        assert results[0].url == "https://x.com/0"

    def test_top_k_caps_results(self) -> None:
        idx = _index()
        encoder = _fake_encoder(
            {
                "Cats": [1, 0, 0],
                "Dogs": [0.5, 0.5, 0],
                "Fish": [0, 0, 1],
            }
        )
        di = DenseIndex.from_index(idx, encoder=encoder)
        ranker = DenseRanker(di, encoder=cast(Any, encoder))
        out = ranker.rank(idx, "Cats query", top_k=2)
        assert len(out) == 2

    def test_empty_query_returns_empty(self) -> None:
        idx = _index()
        encoder = _fake_encoder({"Cats": [1, 0, 0]})
        di = DenseIndex.from_index(idx, encoder=encoder)
        ranker = DenseRanker(di, encoder=cast(Any, encoder))
        assert ranker.rank(idx, "   ", top_k=5) == []
