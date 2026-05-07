"""Tests for SymSpell-based spell correction (Phase D)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.search import suggest
from src.symspell import SymSpell, build_symspell


def _index(terms: list[str]) -> Any:
    return cast(
        Any,
        {
            "meta": {
                "crawled_at": "",
                "num_docs": 0,
                "num_terms": len(terms),
                "total_tokens": 0,
                "version": "1.1",
            },
            "docs": {},
            "terms": {
                t: {"df": 1, "cf": 1, "postings": {}} for t in terms
            },
        },
    )


class TestSymSpellLookup:
    def test_exact_match_returns_empty(self) -> None:
        sym = SymSpell(max_distance=2)
        sym.add("apple")
        assert sym.lookup("apple") == []

    def test_finds_one_edit_typo(self) -> None:
        sym = SymSpell(max_distance=2)
        for term in ["apple", "ample", "apply", "amber"]:
            sym.add(term)
        out = sym.lookup("aplle")
        assert "apple" in out

    def test_respects_max_distance(self) -> None:
        sym = SymSpell(max_distance=2)
        for term in ["dog", "cat", "fish"]:
            sym.add(term)
        # 'xxx' is more than 2 edits from any term.
        assert sym.lookup("xxx") == []

    def test_orders_by_distance_then_alpha(self) -> None:
        sym = SymSpell(max_distance=2)
        for term in ["cat", "bat", "rat", "cab"]:
            sym.add(term)
        out = sym.lookup("zat")
        # cat, bat, rat all distance 1 from zat — should appear before cab (d=2).
        assert set(out[:3]) == {"cat", "bat", "rat"}
        assert out[:3] == sorted(out[:3])  # alpha tie-break

    def test_max_distance_zero_only_exact(self) -> None:
        sym = SymSpell(max_distance=0)
        sym.add("hello")
        assert sym.lookup("hello") == []
        assert sym.lookup("hellp") == []

    def test_lookup_above_built_max_raises(self) -> None:
        sym = SymSpell(max_distance=1)
        sym.add("a")
        with pytest.raises(ValueError):
            sym.lookup("a", max_distance=3)

    def test_negative_max_distance_rejected(self) -> None:
        with pytest.raises(ValueError):
            SymSpell(max_distance=-1)


class TestSymSpellAgreesWithLevenshtein:
    """The deletion-lookup must produce the same set as the linear scan."""

    @pytest.mark.parametrize(
        "vocab,query",
        [
            (["apple", "apply", "amber", "ample", "ripple"], "aplle"),
            (["happy", "happen", "appy"], "hapy"),
            (["color", "colour"], "colur"),
            (["a", "b", "c"], "z"),
        ],
    )
    def test_same_results_as_linear(
        self, vocab: list[str], query: str
    ) -> None:
        idx = _index(vocab)
        sym = build_symspell(idx, max_distance=2)
        slow = suggest(idx, query)
        fast = suggest(idx, query, symspell=sym)
        assert sorted(slow) == sorted(fast)
