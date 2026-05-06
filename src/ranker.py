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
from typing import Protocol


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


RANKERS: dict[str, Ranker] = {
    FrequencyRanker.name: FrequencyRanker(),
    TFIDFRanker.name: TFIDFRanker(),
    BM25Ranker.name: BM25Ranker(),
}
"""Registry of named ranker instances, used by the ``--ranker`` CLI flag."""
