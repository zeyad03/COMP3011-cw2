"""Pluggable scoring functions ("rankers") for the search module.

The :class:`Ranker` protocol decouples the search procedure (look up
posting lists, intersect, score, sort) from the scoring formula. Three
implementations ship with the project:

* :class:`FrequencyRanker` — naive baseline; ranks by raw term frequency
  alone. Used to show empirically *why* a smarter scoring function is
  needed (long documents and very-frequent terms dominate).
* :class:`TFIDFRanker` — sub-linear TF and smoothed IDF. The default,
  derived in ``docs/design.md`` §5.
* :class:`BM25Ranker` — Robertson/Sparck Jones probabilistic model with
  tunable ``k1`` and ``b``. Saturates TF non-linearly and normalises
  by document length. The de-facto industry default for keyword search.

All three accept the same keyword arguments so the search procedure can
swap between them without branching. The :data:`RANKERS` registry maps
names to ready-to-use instances and is the source of truth for the
``--ranker`` CLI flag.
"""

from __future__ import annotations

import math
from typing import Any, Protocol


class Ranker(Protocol):
    """Score a single (term, document) pair.

    Implementations return a float that grows with relevance; the
    search module sums these contributions across query terms and sorts
    by the total. Return ``0.0`` for the degenerate case where
    ``tf == 0`` or ``df == 0``.
    """

    name: str

    def score(
        self,
        *,
        tf: int,
        df: int,
        num_docs: int,
        doc_length: int,
        avg_doc_length: float,
    ) -> float: ...


class FrequencyRanker:
    """Rank by raw term frequency. Baseline."""

    name = "frequency"

    def score(
        self,
        *,
        tf: int,
        df: int,
        num_docs: int,
        doc_length: int,
        avg_doc_length: float,
    ) -> float:
        del df, num_docs, doc_length, avg_doc_length  # unused
        return float(tf) if tf > 0 else 0.0


class TFIDFRanker:
    """Sub-linear TF * smoothed IDF.

    ``score = (1 + log10(tf)) * log10(1 + N / df)``

    Same formula as the original ``_term_score`` in :mod:`search` —
    preserved verbatim so existing tests and the v1.0 index continue to
    rank identically.
    """

    name = "tfidf"

    def score(
        self,
        *,
        tf: int,
        df: int,
        num_docs: int,
        doc_length: int,
        avg_doc_length: float,
    ) -> float:
        del doc_length, avg_doc_length  # unused
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        tf_weight = 1.0 + math.log10(tf)
        idf_weight = math.log10(1.0 + num_docs / df)
        return tf_weight * idf_weight


class BM25Ranker:
    """Okapi BM25 with smoothed IDF.

    ``score = IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))``

    where ``IDF(t) = log10(1 + (N - df + 0.5) / (df + 0.5))``.

    Args:
        k1: Term-frequency saturation. Higher values make the score
            grow more linearly with TF; ``k1 = 0`` collapses BM25 to
            pure IDF (binary term presence). Industry default 1.2–2.0.
        b: Document-length normalisation. ``b = 0`` disables length
            normalisation; ``b = 1`` fully normalises. Industry default
            0.75.
    """

    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be in [0, 1]")
        self.k1 = k1
        self.b = b

    def score(
        self,
        *,
        tf: int,
        df: int,
        num_docs: int,
        doc_length: int,
        avg_doc_length: float,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        idf = math.log10(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        avgdl = max(avg_doc_length, 1.0)
        length_norm = 1 - self.b + self.b * (doc_length / avgdl)
        tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        return idf * tf_component


class BM25FRanker:
    """BM25F: per-field BM25 with title boosted relative to body.

    Standard reference: Robertson & Zaragoza, *The Probabilistic
    Relevance Framework: BM25 and Beyond* (FnTIR, 2009).

    The score for a single (term, doc) pair is::

        IDF(t) * tf' * (k1 + 1) / (tf' + k1)

    where ``tf'`` is the *weighted, length-normalised* sum of per-field
    occurrences. For a document with title length |dt| and body length
    |db|, average title length avgdt and average body length avgdb::

        tf' = w_title * tf_t / B_t  +  w_body * tf_b / B_b
        B_t = 1 - b_t + b_t * |dt| / avgdt      (and analogously for body)

    A title hit contributes ``w_title``× more than a body hit, controlled
    by ``weights``. ``b_field`` per-field length-normalisation defaults to
    0.75 for body (BM25's industry default) and 0.0 for title (titles
    are short and uniform; normalising them tends to hurt). The IDF
    formula matches :class:`BM25Ranker` for fair head-to-head MAP comparison.

    BM25F needs per-field statistics that don't fit the simple
    :class:`Ranker` protocol, so it implements an additional
    :meth:`score_fielded` method that the search loop dispatches on
    when the index carries v1.2+ field-level positions.
    """

    name = "bm25f"

    def __init__(
        self,
        *,
        k1: float = 1.5,
        weights: dict[str, float] | None = None,
        b_field: dict[str, float] | None = None,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        self.k1 = k1
        self.weights = weights or {"title": 5.0, "body": 1.0}
        self.b_field = b_field or {"title": 0.0, "body": 0.75}
        for field, b in self.b_field.items():
            if not 0 <= b <= 1:
                raise ValueError(f"b_{field} must be in [0, 1]")

    # The Ranker protocol's per-pair score interface — BM25F falls back
    # to plain BM25 when called from a v1.0/v1.1 index that has no
    # field-level positions. This keeps the ranker selectable from the
    # CLI even on older indexes (it just collapses to BM25 silently).
    def score(
        self,
        *,
        tf: int,
        df: int,
        num_docs: int,
        doc_length: int,
        avg_doc_length: float,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        b_body = self.b_field.get("body", 0.75)
        idf = math.log10(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        avgdl = max(avg_doc_length, 1.0)
        length_norm = 1 - b_body + b_body * (doc_length / avgdl)
        tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)
        return idf * tf_component

    def score_fielded(
        self,
        *,
        tf_title: int,
        tf_body: int,
        df: int,
        num_docs: int,
        title_length: int,
        body_length: int,
        avg_title_length: float,
        avg_body_length: float,
    ) -> float:
        """BM25F score using per-field tf and length statistics."""
        if df <= 0 or num_docs <= 0 or (tf_title == 0 and tf_body == 0):
            return 0.0
        avgdt = max(avg_title_length, 1.0)
        avgdb = max(avg_body_length, 1.0)
        bt = self.b_field.get("title", 0.0)
        bb = self.b_field.get("body", 0.75)
        b_t_norm = 1 - bt + bt * (title_length / avgdt)
        b_b_norm = 1 - bb + bb * (body_length / avgdb)
        weighted_tf = (
            self.weights.get("title", 1.0) * tf_title / max(b_t_norm, 1e-9)
            + self.weights.get("body", 1.0) * tf_body / max(b_b_norm, 1e-9)
        )
        idf = math.log10(1.0 + (num_docs - df + 0.5) / (df + 0.5))
        return idf * weighted_tf * (self.k1 + 1) / (weighted_tf + self.k1)


# A run-time helper used by :func:`src.search.find` to detect whether a
# ranker can take advantage of per-field positions.
def supports_fields(ranker: Any) -> bool:
    return callable(getattr(ranker, "score_fielded", None))


class HybridRanker:
    """Reciprocal-rank fusion of two list-level rankers.

    Reciprocal-rank fusion (Cormack et al., 2009) is the simplest robust
    way to combine heterogeneously-scored rankings: every doc gets a
    fused score of ``1 / (k + rank)`` from each input, summed across
    inputs. The constant ``k`` (default 60) damps the contribution of
    very low ranks so a single poor ranker can't dominate.

    Unlike the per-pair :class:`Ranker` protocol, RRF works at the
    *list* level: it needs each input ranker's full top-k. The
    consumer code is therefore in :func:`src.search.find_hybrid` rather
    than :func:`src.search.find`.
    """

    name = "hybrid"

    def __init__(
        self,
        a: Any,
        b: Any,
        *,
        k: int = 60,
        a_results: list[Any] | None = None,
        b_results: list[Any] | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("RRF k must be positive")
        self.a = a
        self.b = b
        self.k = k
        # a_results / b_results are an *injection point* used by tests
        # that don't want to call the underlying rankers themselves.
        self._a_results = a_results
        self._b_results = b_results

    def fuse(self, a_results: list[Any], b_results: list[Any]) -> list[Any]:
        from src.search import SearchResult  # local import to avoid cycle

        scores: dict[str, float] = {}
        meta: dict[str, SearchResult] = {}
        for rank, r in enumerate(a_results, 1):
            scores[r.url] = scores.get(r.url, 0.0) + 1.0 / (self.k + rank)
            meta.setdefault(r.url, r)
        for rank, r in enumerate(b_results, 1):
            scores[r.url] = scores.get(r.url, 0.0) + 1.0 / (self.k + rank)
            meta.setdefault(r.url, r)
        # Sort by fused score desc, URL asc for determinism.
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out: list[SearchResult] = []
        for url, fused in ordered:
            base = meta[url]
            out.append(
                SearchResult(
                    url=base.url,
                    title=base.title,
                    score=round(fused, 6),
                    matched_terms=base.matched_terms,
                )
            )
        return out


RANKERS: dict[str, Ranker] = {
    FrequencyRanker.name: FrequencyRanker(),
    TFIDFRanker.name: TFIDFRanker(),
    BM25Ranker.name: BM25Ranker(),
    BM25FRanker.name: BM25FRanker(),
}
"""Registry of named ranker instances, used by the ``--ranker`` CLI flag.

Note: ``DenseRanker`` and ``HybridRanker`` are *not* in this registry
because they require either a pre-built embedding sidecar
(:class:`~src.dense_ranker.DenseIndex`) or an injected pair of input
rankers. The CLI handles them as special cases.
"""
