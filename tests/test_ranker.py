"""Tests for the pluggable ranker abstraction.

Each ranker is exercised in isolation (numerical sanity, monotonicity,
parameter handling) and via the search procedure in
``test_search.py`` (different ranker → different ordering on the same
index).
"""

from __future__ import annotations

import math

import pytest

from src.crawler import Page
from src.indexer import build_index
from src.ranker import (
    RANKERS,
    BM25Ranker,
    FrequencyRanker,
    Ranker,
    TFIDFRanker,
)
from src.search import find


# Args shared across many parametrised tests.
_BASE: dict[str, float | int] = {
    "doc_length": 100,
    "avg_doc_length": 100.0,
}


# --------------------------------------------------------------- registry


class TestRegistry:
    def test_registry_has_three_rankers_by_name(self) -> None:
        assert set(RANKERS) == {"frequency", "tfidf", "bm25"}

    def test_registry_entries_have_matching_names(self) -> None:
        for name, ranker in RANKERS.items():
            assert ranker.name == name


# ------------------------------------------------------------ degenerate


@pytest.mark.parametrize("ranker", list(RANKERS.values()))
class TestDegenerateInputs:
    def test_zero_tf_yields_zero(self, ranker: Ranker) -> None:
        assert (
            ranker.score(tf=0, df=1, num_docs=10, doc_length=10, avg_doc_length=10.0)
            == 0.0
        )

    def test_zero_df_yields_zero(self, ranker: Ranker) -> None:
        # FrequencyRanker ignores df and would happily score; document
        # the difference and skip.
        if isinstance(ranker, FrequencyRanker):
            pytest.skip("FrequencyRanker ignores df by design")
        assert (
            ranker.score(tf=5, df=0, num_docs=10, doc_length=10, avg_doc_length=10.0)
            == 0.0
        )


# ----------------------------------------------------------- monotonicity


@pytest.mark.parametrize("ranker", list(RANKERS.values()))
class TestMonotonicity:
    def test_score_grows_with_tf(self, ranker: Ranker) -> None:
        prev = -1.0
        for tf in (1, 2, 5, 10, 50):
            score = ranker.score(
                tf=tf, df=5, num_docs=100, doc_length=100, avg_doc_length=100.0
            )
            assert score >= prev
            prev = score


class TestRareTermBoost:
    """IDF-aware rankers should reward rarer terms over common ones."""

    @pytest.mark.parametrize("ranker", [TFIDFRanker(), BM25Ranker()])
    def test_lower_df_gives_higher_score(self, ranker: Ranker) -> None:
        rare = ranker.score(
            tf=1, df=1, num_docs=100, doc_length=100, avg_doc_length=100.0
        )
        common = ranker.score(
            tf=1, df=80, num_docs=100, doc_length=100, avg_doc_length=100.0
        )
        assert rare > common

    def test_frequency_ranker_does_not_penalise_common_terms(self) -> None:
        # Same TF -> same score regardless of df.
        f = FrequencyRanker()
        assert (
            f.score(tf=3, df=1, num_docs=100, doc_length=100, avg_doc_length=100.0)
            == f.score(tf=3, df=80, num_docs=100, doc_length=100, avg_doc_length=100.0)
        )


# ------------------------------------------------------------------ BM25


class TestBM25Specific:
    def test_b_zero_disables_length_normalisation(self) -> None:
        # With b=0, doc_length/avgdl term vanishes from the formula.
        ranker = BM25Ranker(b=0.0)
        short_doc = ranker.score(
            tf=2, df=5, num_docs=100, doc_length=50, avg_doc_length=200.0
        )
        long_doc = ranker.score(
            tf=2, df=5, num_docs=100, doc_length=200, avg_doc_length=200.0
        )
        assert short_doc == pytest.approx(long_doc)

    def test_b_one_fully_normalises_length(self) -> None:
        # With b=1, identical TF in a *long* doc scores lower than in a short one.
        ranker = BM25Ranker(b=1.0)
        short = ranker.score(
            tf=3, df=5, num_docs=100, doc_length=50, avg_doc_length=100.0
        )
        long_doc = ranker.score(
            tf=3, df=5, num_docs=100, doc_length=200, avg_doc_length=100.0
        )
        assert short > long_doc

    def test_invalid_k1_rejected(self) -> None:
        with pytest.raises(ValueError):
            BM25Ranker(k1=-1.0)

    def test_invalid_b_rejected(self) -> None:
        with pytest.raises(ValueError):
            BM25Ranker(b=2.0)
        with pytest.raises(ValueError):
            BM25Ranker(b=-0.1)

    def test_score_matches_textbook_example(self) -> None:
        """Hand-computed reference against the BM25 formula.

        Inputs:
            tf = 2, df = 3, N = 10, doc_length = 100, avgdl = 100,
            k1 = 1.5, b = 0.75.

        IDF = log10(1 + (10 - 3 + 0.5) / (3 + 0.5))
            = log10(1 + 7.5 / 3.5)
            = log10(3.142857...) ≈ 0.4972

        length_norm = 1 - 0.75 + 0.75 * (100/100) = 1.0.
        tf_component = (2 * 2.5) / (2 + 1.5 * 1.0) = 5 / 3.5 ≈ 1.4286.
        score = 0.4972 * 1.4286 ≈ 0.7103.
        """
        ranker = BM25Ranker(k1=1.5, b=0.75)
        score = ranker.score(
            tf=2, df=3, num_docs=10, doc_length=100, avg_doc_length=100.0
        )
        idf = math.log10(1 + 7.5 / 3.5)
        expected = idf * 5.0 / 3.5
        assert score == pytest.approx(expected)


# ------------------------------------------------------------ via search


class TestRankerViaFind:
    """End-to-end: same index, different rankers → different orderings."""

    @pytest.fixture
    def index(self) -> object:
        # Three docs. 'foo' appears in all three at different rates and
        # in documents of different lengths.
        pages = [
            Page(url="https://x.com/a", title="", text="foo bar"),
            Page(url="https://x.com/b", title="", text="foo foo bar baz"),
            Page(url="https://x.com/c", title="", text="foo " + " ".join(f"w{i}" for i in range(50))),
        ]
        return build_index(pages)

    def test_three_rankers_return_same_doc_set(self, index: object) -> None:
        sets = {
            ranker.name: {r.url for r in find(index, "foo", ranker=ranker)}  # type: ignore[arg-type]
            for ranker in RANKERS.values()
        }
        assert sets["frequency"] == sets["tfidf"] == sets["bm25"]

    def test_frequency_ranker_orders_by_raw_tf(self, index: object) -> None:
        results = find(index, "foo", ranker=FrequencyRanker())  # type: ignore[arg-type]
        # Doc B has tf=2, A and C have tf=1. B must be first.
        assert results[0].url == "https://x.com/b"

    def test_bm25_penalises_long_doc_with_same_tf(self, index: object) -> None:
        results = find(index, "foo", ranker=BM25Ranker())  # type: ignore[arg-type]
        # A and C both have tf=1, but C is much longer and BM25 should
        # rank A above C.
        positions = {r.url: i for i, r in enumerate(results)}
        assert positions["https://x.com/a"] < positions["https://x.com/c"]
