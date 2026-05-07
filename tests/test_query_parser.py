"""Tests for the boolean query parser and evaluator (Phase C)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.query_eval import collect_terms, evaluate
from src.query_parser import (
    And,
    Not,
    Or,
    ParseError,
    Term,
    has_boolean_operators,
    parse,
)


# ---------------------------- parser ----------------------------


class TestHasBooleanOperators:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("good friends", False),
            ("a AND b", True),
            ("a OR b", True),
            ("NOT a", True),
            ("(a OR b)", True),
            ("apple", False),
            ("apple AND", True),
            ("Bandwidth", False),  # 'AND' must be standalone
        ],
    )
    def test_detection(self, query: str, expected: bool) -> None:
        assert has_boolean_operators(query) == expected


class TestParse:
    def test_term(self) -> None:
        assert parse("apple") == Term("apple")

    def test_and(self) -> None:
        assert parse("a AND b") == And(Term("a"), Term("b"))

    def test_or(self) -> None:
        assert parse("a OR b") == Or(Term("a"), Term("b"))

    def test_not(self) -> None:
        assert parse("NOT a") == Not(Term("a"))

    def test_double_not(self) -> None:
        assert parse("NOT NOT a") == Not(Not(Term("a")))

    def test_parentheses(self) -> None:
        assert parse("(a OR b) AND c") == And(
            Or(Term("a"), Term("b")), Term("c")
        )

    def test_precedence_and_binds_tighter_than_or(self) -> None:
        # a AND b OR c parses as (a AND b) OR c
        assert parse("a AND b OR c") == Or(
            And(Term("a"), Term("b")), Term("c")
        )

    def test_precedence_not_binds_tighter_than_and(self) -> None:
        assert parse("NOT a AND b") == And(Not(Term("a")), Term("b"))

    def test_left_associative_and(self) -> None:
        assert parse("a AND b AND c") == And(
            And(Term("a"), Term("b")), Term("c")
        )

    def test_case_insensitive_operators(self) -> None:
        assert parse("a and b OR c") == Or(
            And(Term("a"), Term("b")), Term("c")
        )

    def test_terms_normalised_through_tokenizer(self) -> None:
        # Apostrophes collapse, case folds.
        assert parse("Don't AND BoOK") == And(Term("dont"), Term("book"))

    def test_unmatched_paren_raises(self) -> None:
        with pytest.raises(ParseError):
            parse("(a OR b")

    def test_trailing_operator_raises(self) -> None:
        with pytest.raises(ParseError):
            parse("a AND")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ParseError):
            parse("")


class TestCollectTerms:
    def test_unique_in_source_order(self) -> None:
        ast = parse("(a OR b) AND a AND NOT c")
        assert collect_terms(ast) == ["a", "b", "c"]


# --------------------------- evaluator --------------------------


def _index(docs: dict[str, list[str]]) -> Any:
    """Build a tiny synthetic index from a doc_id -> token-list map."""
    terms: dict[str, dict[str, Any]] = {}
    for doc_id, tokens in docs.items():
        for pos, token in enumerate(tokens):
            entry = terms.setdefault(
                token, {"df": 0, "cf": 0, "postings": {}}
            )
            posting = entry["postings"].get(doc_id)
            if posting is None:
                entry["df"] += 1
                entry["postings"][doc_id] = {"tf": 0, "positions": []}
            entry["cf"] += 1
            entry["postings"][doc_id]["tf"] += 1
            entry["postings"][doc_id]["positions"].append(pos)

    return cast(
        Any,
        {
            "meta": {
                "crawled_at": "",
                "num_docs": len(docs),
                "num_terms": len(terms),
                "total_tokens": sum(len(t) for t in docs.values()),
                "version": "1.1",
            },
            "docs": {
                d: {
                    "url": f"https://example.com/{d}",
                    "title": d,
                    "length": len(tokens),
                }
                for d, tokens in docs.items()
            },
            "terms": terms,
        },
    )


class TestEvaluate:
    def setup_method(self) -> None:
        self.idx = _index(
            {
                "0": ["a", "b"],
                "1": ["a", "c"],
                "2": ["b", "c"],
                "3": ["d"],
            }
        )

    def test_term(self) -> None:
        assert evaluate(parse("a"), self.idx) == {"0", "1"}

    def test_and(self) -> None:
        assert evaluate(parse("a AND b"), self.idx) == {"0"}

    def test_or(self) -> None:
        assert evaluate(parse("a OR b"), self.idx) == {"0", "1", "2"}

    def test_not(self) -> None:
        # universe is {0,1,2,3}; NOT a = {2,3}
        assert evaluate(parse("NOT a"), self.idx) == {"2", "3"}

    def test_compound(self) -> None:
        # (a OR b) AND NOT c
        assert evaluate(parse("(a OR b) AND NOT c"), self.idx) == {"0"}

    def test_unknown_term_is_empty_set(self) -> None:
        assert evaluate(parse("zzz"), self.idx) == set()
        # AND with unknown term collapses to empty
        assert evaluate(parse("a AND zzz"), self.idx) == set()
        # OR with unknown term ignores it
        assert evaluate(parse("a OR zzz"), self.idx) == {"0", "1"}
