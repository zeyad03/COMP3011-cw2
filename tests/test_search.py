"""Tests for the search module."""

from __future__ import annotations

import pytest

from src.crawler import Page
from src.indexer import Index, build_index
from src.ranker import TFIDFRanker
from src.search import (
    SearchResult,
    TermContribution,
    _is_adjacent_in_doc,
    _levenshtein,
    find,
    print_term,
    suggest,
)


_score = TFIDFRanker().score


def _index(*texts: str) -> Index:
    pages = [
        Page(url=f"https://x.com/p{i}", title="", text=t)
        for i, t in enumerate(texts)
    ]
    return build_index(pages)


# ---------------------------------------------------------------- print_term


class TestPrintTerm:
    def test_known_term_includes_postings(self) -> None:
        idx = _index("good day", "good news")
        out = print_term(idx, "good")
        assert "good" in out
        assert "df=2" in out
        assert "https://x.com/p0" in out
        assert "https://x.com/p1" in out

    def test_unknown_term_returns_not_found_message(self) -> None:
        idx = _index("hello")
        out = print_term(idx, "xyzzy")
        assert "xyzzy" in out
        assert "not found" in out

    def test_empty_word_returns_message(self) -> None:
        idx = _index("hello")
        assert print_term(idx, "") == "empty term"

    def test_case_insensitive(self) -> None:
        idx = _index("hello world")
        assert print_term(idx, "Hello") == print_term(idx, "hello")

    def test_punctuation_stripped(self) -> None:
        idx = _index("hello world")
        assert print_term(idx, "hello!") == print_term(idx, "hello")


# ---------------------------------------------------------------------- find


class TestFindEdgeCases:
    def test_empty_query_returns_empty(self) -> None:
        idx = _index("hello")
        assert find(idx, "") == []

    def test_whitespace_only_query_returns_empty(self) -> None:
        idx = _index("hello")
        assert find(idx, "   \t  ") == []

    def test_unknown_term_returns_empty(self) -> None:
        idx = _index("hello world")
        assert find(idx, "xyzzy") == []

    def test_partly_unknown_query_returns_empty(self) -> None:
        idx = _index("hello world")
        # AND semantics: any unknown token kills the result set.
        assert find(idx, "hello xyzzy") == []


class TestFindSingleTerm:
    def test_returns_matching_pages(self) -> None:
        idx = _index("hello world", "hello sun")
        results = find(idx, "hello")
        urls = {r.url for r in results}
        assert urls == {"https://x.com/p0", "https://x.com/p1"}

    def test_case_insensitive(self) -> None:
        idx = _index("hello")
        upper = find(idx, "HELLO")
        lower = find(idx, "hello")
        assert [r.url for r in upper] == [r.url for r in lower]

    def test_returns_search_result_dataclass(self) -> None:
        idx = _index("hello world")
        results = find(idx, "hello")
        assert isinstance(results[0], SearchResult)
        assert results[0].url == "https://x.com/p0"
        assert results[0].score > 0


class TestFindAndSemantics:
    def test_multi_word_requires_all_terms_in_same_doc(self) -> None:
        idx = _index("good day", "good news", "happy days")
        results = find(idx, "good day")
        urls = {r.url for r in results}
        assert urls == {"https://x.com/p0"}

    def test_disjoint_terms_return_empty(self) -> None:
        idx = _index("good day", "happy news")
        # Both terms exist but never co-occur.
        assert find(idx, "good news") == []


class TestRanking:
    def test_higher_tf_ranks_higher(self) -> None:
        # Three docs so df < num_docs and idf is non-zero.
        idx = _index("foo bar", "foo foo bar", "baz qux")
        results = find(idx, "foo")
        assert results[0].url == "https://x.com/p1"
        assert results[1].url == "https://x.com/p0"

    def test_results_are_sorted_by_score_descending(self) -> None:
        idx = _index("foo bar baz", "foo foo bar", "qux quux")
        results = find(idx, "foo")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_caps_results(self) -> None:
        idx = _index("foo", "foo", "foo", "foo")
        results = find(idx, "foo", top_k=2)
        assert len(results) == 2

    def test_secondary_sort_is_url_alphabetical(self) -> None:
        # Equal scores should tie-break by URL ascending for determinism.
        idx = _index("foo", "foo")
        results = find(idx, "foo")
        urls = [r.url for r in results]
        assert urls == sorted(urls)


class TestPhraseMode:
    def test_phrase_matches_when_terms_adjacent(self) -> None:
        idx = _index("hello world goodbye", "hello world")
        results = find(idx, "hello world", mode="phrase")
        urls = {r.url for r in results}
        assert urls == {"https://x.com/p0", "https://x.com/p1"}

    def test_phrase_rejects_non_adjacent_terms(self) -> None:
        # "good ... friends" — co-occurring but not adjacent.
        idx = _index("good morning friends", "good day for old friends")
        assert find(idx, "good friends", mode="phrase") == []

    def test_phrase_single_term_behaves_like_and(self) -> None:
        idx = _index("hello world")
        and_results = find(idx, "hello", mode="and")
        phrase_results = find(idx, "hello", mode="phrase")
        assert [r.url for r in and_results] == [r.url for r in phrase_results]


class TestAdjacencyHelper:
    def test_adjacent_positions_detected(self) -> None:
        postings = [
            {"0": {"tf": 1, "positions": [3]}},
            {"0": {"tf": 1, "positions": [4]}},
        ]
        assert _is_adjacent_in_doc(postings, "0") is True

    def test_non_adjacent_returns_false(self) -> None:
        postings = [
            {"0": {"tf": 1, "positions": [3]}},
            {"0": {"tf": 1, "positions": [9]}},
        ]
        assert _is_adjacent_in_doc(postings, "0") is False

    def test_picks_any_matching_start_position(self) -> None:
        # First posting list has two starts; only one is followed by adjacent.
        postings = [
            {"0": {"tf": 2, "positions": [1, 5]}},
            {"0": {"tf": 1, "positions": [6]}},
        ]
        assert _is_adjacent_in_doc(postings, "0") is True


class TestTermScore:
    """Sanity tests for the default TF-IDF scorer.

    These assertions also verify that the previously inlined
    ``_term_score`` behaviour is preserved by ``TFIDFRanker``.
    """

    _kwargs = {"doc_length": 100, "avg_doc_length": 100.0}

    def test_zero_inputs_yield_zero(self) -> None:
        assert _score(tf=0, df=1, num_docs=10, **self._kwargs) == 0.0
        assert _score(tf=1, df=0, num_docs=10, **self._kwargs) == 0.0
        assert _score(tf=1, df=1, num_docs=0, **self._kwargs) == 0.0

    def test_score_is_positive_for_normal_inputs(self) -> None:
        assert _score(tf=1, df=1, num_docs=10, **self._kwargs) > 0

    def test_rare_term_outscores_common_term(self) -> None:
        rare = _score(tf=1, df=1, num_docs=100, **self._kwargs)
        common = _score(tf=1, df=50, num_docs=100, **self._kwargs)
        assert rare > common


# ------------------------------------------------------------------- suggest


class TestExplain:
    def test_explain_omits_breakdown_by_default(self) -> None:
        idx = _index("hello world")
        results = find(idx, "hello")
        assert results[0].breakdown == ()

    def test_explain_attaches_breakdown_per_term(self) -> None:
        idx = _index("good day for friends", "good news for everyone")
        results = find(idx, "good friends", explain=True)
        # Only doc 0 has both terms.
        assert len(results) == 1
        terms = {c.term for c in results[0].breakdown}
        assert terms == {"good", "friends"}

    def test_breakdown_contributions_sum_to_score(self) -> None:
        idx = _index(
            "alpha beta gamma",
            "alpha alpha beta",
            "beta gamma delta",
        )
        results = find(idx, "alpha beta", explain=True)
        for r in results:
            total = sum(c.contribution for c in r.breakdown)
            assert r.score == pytest.approx(total, abs=1e-5)

    def test_breakdown_records_tf_and_df(self) -> None:
        idx = _index("foo foo bar")
        results = find(idx, "foo", explain=True)
        c = results[0].breakdown[0]
        assert c.term == "foo"
        assert c.tf == 2
        assert c.df == 1


class TestSuggest:
    def test_known_term_returns_no_suggestions(self) -> None:
        idx = _index("hello world")
        assert suggest(idx, "hello") == []

    def test_close_misspelling_suggested(self) -> None:
        idx = _index("hello world")
        suggestions = suggest(idx, "helo")  # one deletion
        assert "hello" in suggestions

    def test_far_query_returns_empty(self) -> None:
        idx = _index("hello world")
        assert suggest(idx, "qwerty", max_distance=2) == []

    def test_empty_query_returns_empty(self) -> None:
        idx = _index("hello")
        assert suggest(idx, "") == []

    def test_results_sorted_by_distance(self) -> None:
        idx = _index("cat bat car")
        suggestions = suggest(idx, "rat", max_distance=2)
        # 'cat' and 'bat' are distance 1; 'car' is distance 2.
        assert suggestions[:2] == ["bat", "cat"]
        if len(suggestions) > 2:
            assert suggestions[2] == "car"

    def test_max_suggestions_caps_results(self) -> None:
        idx = _index("cat bat hat rat mat fat sat")
        suggestions = suggest(idx, "pat", max_suggestions=3)
        assert len(suggestions) == 3


class TestLevenshtein:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("", "abc", 3),
            ("abc", "", 3),
            ("abc", "abd", 1),  # substitution
            ("abc", "ab", 1),  # deletion
            ("abc", "abcd", 1),  # insertion
            ("kitten", "sitting", 3),  # classic example
        ],
    )
    def test_distance_matches_known_values(
        self, a: str, b: str, expected: int
    ) -> None:
        assert _levenshtein(a, b, max_distance=10) == expected

    def test_early_exit_when_over_budget(self) -> None:
        # 'abcdef' vs 'zzzzzz' is distance 6; with budget 2 we expect 3 (cap+1).
        result = _levenshtein("abcdef", "zzzzzz", max_distance=2)
        assert result > 2
