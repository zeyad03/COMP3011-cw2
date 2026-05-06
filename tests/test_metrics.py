"""Tests for IR evaluation metrics."""

from __future__ import annotations

import pytest

from src.metrics import (
    average_precision,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ----------------------------------------------------------- precision@k


class TestPrecisionAtK:
    def test_all_relevant_in_top_k(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_none_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0

    def test_partial_match(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self) -> None:
        # 1 hit out of k=5 slots = 0.2.
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(1 / 5)

    def test_zero_k_yields_zero(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0

    def test_negative_k_yields_zero(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, -1) == 0.0

    def test_empty_retrieved_yields_zero(self) -> None:
        assert precision_at_k([], {"a"}, 5) == 0.0


# -------------------------------------------------------------- recall@k


class TestRecallAtK:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_partial_recall(self) -> None:
        # Half of the relevant set retrieved.
        assert recall_at_k(["a"], {"a", "b"}, 5) == 0.5

    def test_empty_relevant_yields_zero(self) -> None:
        assert recall_at_k(["a"], set(), 5) == 0.0

    def test_zero_k_yields_zero(self) -> None:
        assert recall_at_k(["a"], {"a"}, 0) == 0.0


# -------------------------------------------------------------------- MRR


class TestReciprocalRank:
    def test_first_item_relevant_yields_one(self) -> None:
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_item_relevant(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_relevant_retrieved(self) -> None:
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_first_match_wins(self) -> None:
        # Only the *first* match counts in RR.
        assert reciprocal_rank(["x", "a", "a"], {"a"}) == pytest.approx(1 / 2)


# ---------------------------------------------------------- average precision


class TestAveragePrecision:
    def test_textbook_example(self) -> None:
        """Manning §8.4 worked example.

        Retrieved: [a, b, c, d, e].
        Relevant:  {a, c}.
        P@1 = 1, P@3 = 2/3.
        AP = (1.0 + 2/3) / 2 = 5/6 ≈ 0.8333.
        """
        assert average_precision(
            ["a", "b", "c", "d", "e"], {"a", "c"}
        ) == pytest.approx(5 / 6)

    def test_perfect_ranking_yields_one(self) -> None:
        assert average_precision(["a", "b", "c"], {"a", "b", "c"}) == 1.0

    def test_no_relevant_retrieved_yields_zero(self) -> None:
        assert average_precision(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant_yields_zero(self) -> None:
        assert average_precision(["a"], set()) == 0.0

    def test_unretrieved_relevant_lowers_ap(self) -> None:
        # Two relevant but only one retrieved at the front:
        # (1/1) / 2 = 0.5.
        assert average_precision(["a", "x", "y"], {"a", "z"}) == pytest.approx(0.5)


# -------------------------------------------------------------------- mean


class TestMean:
    def test_arithmetic_mean(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_yields_zero(self) -> None:
        assert mean([]) == 0.0

    def test_single_value(self) -> None:
        assert mean([42.0]) == 42.0

    def test_accepts_generator(self) -> None:
        assert mean(x * 2 for x in [1.0, 2.0]) == 3.0
