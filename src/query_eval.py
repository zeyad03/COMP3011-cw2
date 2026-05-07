"""Evaluate a boolean-query AST against an inverted index.

Given an :class:`~src.query_parser.BoolExpr` and an :class:`Index`, this
module returns the set of document IDs that satisfy the expression.

The walk is straightforward set-algebra:

* ``Term(t)``   → posting list keys for ``t`` (empty set if unknown).
* ``Not(x)``    → universe ∖ eval(x).
* ``And(x, y)`` → eval(x) ∩ eval(y).
* ``Or(x, y)``  → eval(x) ∪ eval(y).

Scoring is *not* done here — the caller passes the candidate set into
the existing per-ranker scoring pipeline in :mod:`src.search`.
"""

from __future__ import annotations

from src.indexer import Index
from src.query_parser import And, BoolExpr, Not, Or, Term


def evaluate(expr: BoolExpr, index: Index) -> set[str]:
    """Return the set of doc IDs satisfying *expr* against *index*."""
    universe = set(index["docs"].keys())
    return _eval(expr, index, universe)


def _eval(expr: BoolExpr, index: Index, universe: set[str]) -> set[str]:
    if isinstance(expr, Term):
        entry = index["terms"].get(expr.text)
        if entry is None:
            return set()
        return set(entry["postings"].keys())
    if isinstance(expr, Not):
        return universe - _eval(expr.operand, index, universe)
    if isinstance(expr, And):
        return _eval(expr.left, index, universe) & _eval(
            expr.right, index, universe
        )
    if isinstance(expr, Or):
        return _eval(expr.left, index, universe) | _eval(
            expr.right, index, universe
        )
    raise TypeError(f"unknown AST node: {type(expr).__name__}")


def collect_terms(expr: BoolExpr) -> list[str]:
    """Return every distinct ``Term`` text appearing in *expr*, in source order."""
    seen: dict[str, None] = {}
    _collect(expr, seen)
    return list(seen.keys())


def _collect(expr: BoolExpr, seen: dict[str, None]) -> None:
    if isinstance(expr, Term):
        seen.setdefault(expr.text, None)
    elif isinstance(expr, Not):
        _collect(expr.operand, seen)
    elif isinstance(expr, (And, Or)):
        _collect(expr.left, seen)
        _collect(expr.right, seen)


__all__ = ["collect_terms", "evaluate"]
