"""Tests for the reciprocal-rank-fusion HybridRanker (Phase A)."""

from __future__ import annotations

import pytest

from src.ranker import HybridRanker
from src.search import SearchResult


def _result(url: str, score: float = 1.0) -> SearchResult:
    return SearchResult(url=url, title=url, score=score, matched_terms=())


class TestRRFFusion:
    def test_doc_in_both_outranks_doc_in_one(self) -> None:
        a = [_result("u1"), _result("u2")]
        b = [_result("u2"), _result("u3")]
        fused = HybridRanker(None, None).fuse(a, b)
        assert fused[0].url == "u2"

    def test_lower_rank_inputs_decrease_score(self) -> None:
        a = [_result(f"u{i}") for i in range(3)]
        b = [_result(f"u{i}") for i in range(3)]
        fused = HybridRanker(None, None, k=60).fuse(a, b)
        # Same input => order matches input order.
        assert [r.url for r in fused] == ["u0", "u1", "u2"]
        # Scores strictly descending.
        scores = [r.score for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_ties_broken_alphabetically_by_url(self) -> None:
        # Both URLs at rank 1 in one ranker only — same fused score.
        a = [_result("zebra")]
        b = [_result("apple")]
        fused = HybridRanker(None, None).fuse(a, b)
        # Same fused score, alphabetical order.
        assert fused[0].url == "apple"
        assert fused[1].url == "zebra"

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            HybridRanker(None, None, k=0)

    def test_only_in_a_appears(self) -> None:
        a = [_result("u_only_in_a")]
        b: list[SearchResult] = []
        fused = HybridRanker(None, None).fuse(a, b)
        assert any(r.url == "u_only_in_a" for r in fused)

    def test_rrf_score_formula(self) -> None:
        a = [_result("u1"), _result("u2")]
        b = [_result("u1"), _result("u3")]
        ranker = HybridRanker(None, None, k=60)
        fused = ranker.fuse(a, b)
        scores = {r.url: r.score for r in fused}
        # u1 in both at rank 1 => 2 / 61
        # u2 in a at rank 2     => 1 / 62
        # u3 in b at rank 2     => 1 / 62
        assert scores["u1"] == pytest.approx(2 / 61, rel=1e-4)
        assert scores["u2"] == pytest.approx(1 / 62, rel=1e-4)
        assert scores["u3"] == pytest.approx(1 / 62, rel=1e-4)
