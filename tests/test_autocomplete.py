"""Tests for the prefix-trie autocomplete (Phase H)."""

from __future__ import annotations

from typing import Any, cast

from src.autocomplete import TermTrie, build_trie


def _index(terms: dict[str, int]) -> Any:
    return cast(
        Any,
        {
            "meta": {
                "crawled_at": "",
                "num_docs": 0,
                "num_terms": len(terms),
                "total_tokens": sum(terms.values()),
                "version": "1.1",
            },
            "docs": {},
            "terms": {
                t: {"df": 1, "cf": cf, "postings": {}} for t, cf in terms.items()
            },
        },
    )


class TestTermTrie:
    def test_empty_prefix_returns_empty(self) -> None:
        trie = build_trie(_index({"a": 1, "b": 2}))
        assert trie.suggest("") == []

    def test_unmatched_prefix_returns_empty(self) -> None:
        trie = build_trie(_index({"apple": 5}))
        assert trie.suggest("z") == []

    def test_returns_terms_in_subtree(self) -> None:
        trie = build_trie(
            _index({"compute": 1, "computer": 1, "computing": 1, "completion": 1})
        )
        out = trie.suggest("comp", k=10)
        assert set(out) == {"compute", "computer", "computing", "completion"}

    def test_orders_by_cf_desc_then_alpha(self) -> None:
        trie = build_trie(
            _index({"cab": 1, "car": 5, "cat": 5, "card": 2})
        )
        out = trie.suggest("ca", k=10)
        # cf desc: car(5), cat(5) tied alpha; then card(2); then cab(1).
        assert out == ["car", "cat", "card", "cab"]

    def test_k_caps_results(self) -> None:
        trie = build_trie(_index({f"x{i}": i for i in range(20)}))
        assert len(trie.suggest("x", k=3)) == 3

    def test_k_zero_returns_empty(self) -> None:
        trie = build_trie(_index({"a": 1}))
        assert trie.suggest("a", k=0) == []

    def test_exact_match_is_returned(self) -> None:
        trie = build_trie(_index({"go": 1, "good": 1}))
        out = trie.suggest("go", k=10)
        assert "go" in out and "good" in out


class TestTrieDirect:
    def test_insert_and_suggest_no_index(self) -> None:
        trie = TermTrie()
        trie.insert("hello", 4)
        trie.insert("help", 9)
        assert trie.suggest("he", k=2) == ["help", "hello"]
