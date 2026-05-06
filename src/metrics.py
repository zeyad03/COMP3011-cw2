"""Information-retrieval evaluation metrics.

All functions take *retrieved* (a ranked list of items) and *relevant*
(an unranked set of relevant items) and return a float in ``[0, 1]``.
The relevant set is treated as binary — an item is either relevant or
not — which is the standard simplification for small evaluations.

References
----------
Manning, Raghavan & Schütze, *Introduction to Information Retrieval*,
chapter 8 ("Evaluation in information retrieval"), CUP 2008.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import fmean


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-*k* retrieved items that are relevant.

    ``P@k = |relevant ∩ retrieved[:k]| / k``.

    Returns ``0.0`` when ``k <= 0``.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items captured within the top-*k*.

    ``R@k = |relevant ∩ retrieved[:k]| / |relevant|``.

    Returns ``0.0`` when *relevant* is empty (no information).
    """
    if not relevant:
        return 0.0
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item.

    ``RR = 1 / rank_of_first_relevant``, or ``0.0`` if no relevant
    item is retrieved.
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Mean of precision values taken at the rank of each relevant item.

    ``AP = (1/|relevant|) * Σ over relevant items of P@(rank_of_item)``.

    The denominator is the *total* number of relevant items, not the
    number retrieved — so unretrieved relevant items implicitly
    contribute ``0`` to the sum, which is the standard definition.

    Returns ``0.0`` when *relevant* is empty.
    """
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / rank
    return sum_precision / len(relevant)


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of *values*; returns ``0.0`` for an empty input."""
    seq = list(values)
    if not seq:
        return 0.0
    return fmean(seq)
