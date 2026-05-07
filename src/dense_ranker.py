"""Dense (transformer-embedding) retrieval.

This is the modern semantic-search complement to the sparse rankers in
:mod:`src.ranker`. It encodes documents and queries into a shared
embedding space with a small open model
(``sentence-transformers/all-MiniLM-L6-v2``, 384-dim, ~80 MB) and ranks
documents by cosine similarity.

Sparse rankers (TF-IDF, BM25) reward exact term overlap; dense
retrieval rewards meaning. The two are complementary:

- *einstein imagination* — dense excels because the page about Einstein
  often phrases the famous quote in surrounding language ("creativity",
  "thinking") rather than the literal query terms.
- *deep thoughts* — BM25 excels because the matching tag is a literal
  exact phrase the embedding model has no privileged knowledge about.

The fusion ranker in :mod:`src.ranker` (RRF) combines the two.

Module design notes
-------------------

* The heavy dependency (``sentence-transformers``) is imported lazily
  inside the encoder so the rest of the test suite (and the cold start
  of the CLI) doesn't pay the import cost when no dense ranker is in
  use. Tests that don't exercise the encoder run fine without the
  package installed.
* :class:`DenseIndex` stores per-doc embeddings as a ``numpy.ndarray``
  saved alongside ``data/index.json`` as ``data/index.embeddings.npy``.
  The matrix is laid out as ``(num_docs, dim)`` with row order matching
  the index's ``docs`` insertion order. A separate ``doc_ids`` list
  preserves the mapping from row index → doc_id string.
* Cosine similarity is computed as a single matrix-vector multiply
  against L2-normalised rows; with 200 documents and a 384-dim
  embedding this is roughly 76,800 multiplies per query, well under a
  millisecond on any modern CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.indexer import Index
from src.search import SearchResult

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIDECAR_SUFFIX = ".embeddings.npy"
DOCID_SIDECAR_SUFFIX = ".embeddings.docids.json"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as err:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "numpy is required for the dense ranker; install with `pip install numpy`"
        ) from err
    return np


def _require_sentence_transformers() -> Any:
    try:
        from sentence_transformers import (  # type: ignore[import-not-found]
            SentenceTransformer,
        )
    except ImportError as err:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "sentence-transformers is required for dense retrieval; install "
            "with `pip install sentence-transformers`"
        ) from err
    return SentenceTransformer


@dataclass
class DenseIndex:
    """Per-document embedding matrix and the doc-id row order.

    ``embeddings[i]`` is the L2-normalised embedding of the document
    whose id is ``doc_ids[i]``. Storage is a NumPy ``.npy`` file plus a
    sibling JSON file holding the doc-id list, both written next to
    ``data/index.json``.
    """

    embeddings: Any  # numpy.ndarray; typed loosely to avoid hard dep at import
    doc_ids: list[str]
    model_name: str = DEFAULT_MODEL

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def save(self, json_path: Path | str) -> None:
        np = _require_numpy()
        target = Path(json_path)
        np.save(target.with_name(target.name + EMBEDDING_SIDECAR_SUFFIX),
                self.embeddings)
        target.with_name(target.name + DOCID_SIDECAR_SUFFIX).write_text(
            json.dumps({"model": self.model_name, "doc_ids": self.doc_ids})
        )

    @classmethod
    def load(cls, json_path: Path | str) -> "DenseIndex":
        np = _require_numpy()
        target = Path(json_path)
        emb_path = target.with_name(target.name + EMBEDDING_SIDECAR_SUFFIX)
        ids_path = target.with_name(target.name + DOCID_SIDECAR_SUFFIX)
        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"dense sidecar files not found next to {target}"
            )
        meta = json.loads(ids_path.read_text())
        return cls(
            embeddings=np.load(emb_path),
            doc_ids=list(meta["doc_ids"]),
            model_name=str(meta.get("model", DEFAULT_MODEL)),
        )

    @classmethod
    def from_index(
        cls,
        index: Index,
        *,
        model_name: str = DEFAULT_MODEL,
        encoder: Any = None,
    ) -> "DenseIndex":
        """Encode every document in *index* (title + body) into the matrix.

        For tests, an *encoder* function ``(list[str]) -> np.ndarray`` may
        be supplied to avoid downloading the real model. When omitted,
        :class:`sentence_transformers.SentenceTransformer` is loaded.
        """
        np = _require_numpy()
        doc_ids = list(index["docs"].keys())
        if encoder is None:
            SentenceTransformer = _require_sentence_transformers()
            model = SentenceTransformer(model_name)
            encoder = lambda texts: model.encode(  # noqa: E731
                texts, show_progress_bar=False, convert_to_numpy=True
            )
        texts = [
            f"{index['docs'][d]['title']} {_doc_body(index, d)}".strip()
            for d in doc_ids
        ]
        raw = encoder(texts)
        # L2-normalise rows once at index time so cosine similarity is
        # a plain dot product at query time.
        embeddings = _l2_normalise(np, np.asarray(raw, dtype=np.float32))
        return cls(
            embeddings=embeddings, doc_ids=doc_ids, model_name=model_name
        )


def _doc_body(index: Index, doc_id: str) -> str:
    """Reconstruct the body text of a document for re-encoding.

    The indexer no longer keeps the raw body around (only positions in
    the inverted index), so we reuse the snippet helper to walk the
    index and recover the token order. This is only ever called at
    dense-index build time, not at query time.
    """
    from src.snippet import reconstruct_tokens

    return " ".join(reconstruct_tokens(index, doc_id))


def _l2_normalise(np: Any, matrix: Any) -> Any:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class DenseRanker:
    """Cosine-similarity ranker over a pre-built :class:`DenseIndex`."""

    name = "dense"

    def __init__(
        self,
        dense_index: DenseIndex,
        *,
        encoder: Any = None,
    ) -> None:
        self._dense = dense_index
        self._encoder = encoder

    def _encode_query(self, query: str) -> Any:
        np = _require_numpy()
        if self._encoder is None:
            SentenceTransformer = _require_sentence_transformers()
            self._encoder = SentenceTransformer(self._dense.model_name)
        if hasattr(self._encoder, "encode"):
            raw = self._encoder.encode(
                [query], show_progress_bar=False, convert_to_numpy=True
            )
        else:
            raw = self._encoder([query])
        return _l2_normalise(np, np.asarray(raw, dtype=np.float32))[0]

    def rank(
        self,
        index: Index,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return documents ranked by cosine similarity to *query*."""
        np = _require_numpy()
        if not query.strip():
            return []
        q = self._encode_query(query)
        sims = self._dense.embeddings @ q  # (num_docs,)
        order = np.argsort(-sims)
        out: list[SearchResult] = []
        for row in order:
            row_int = int(row)
            doc_id = self._dense.doc_ids[row_int]
            doc = index["docs"].get(doc_id)
            if doc is None:
                continue
            out.append(
                SearchResult(
                    url=doc["url"],
                    title=doc["title"],
                    score=round(float(sims[row_int]), 6),
                    matched_terms=(),
                )
            )
            if top_k is not None and len(out) >= top_k:
                break
        return out


__all__ = [
    "DEFAULT_MODEL",
    "DenseIndex",
    "DenseRanker",
    "EMBEDDING_SIDECAR_SUFFIX",
    "DOCID_SIDECAR_SUFFIX",
]
