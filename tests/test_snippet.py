"""Tests for snippet generation."""

from __future__ import annotations

import pytest

from src.crawler import Page
from src.indexer import build_index
from src.snippet import (
    DEFAULT_WINDOW,
    generate_snippet,
    reconstruct_tokens,
)


def _index(*texts: str) -> object:
    return build_index(
        [Page(url=f"https://x.com/p{i}", title="", text=t) for i, t in enumerate(texts)]
    )


# ----------------------------------------------------- reconstruct_tokens


class TestReconstructTokens:
    def test_returns_tokens_in_original_order(self) -> None:
        idx = _index("alpha beta gamma delta")
        assert reconstruct_tokens(idx, "0") == [
            "alpha", "beta", "gamma", "delta"
        ]

    def test_unknown_doc_id_yields_empty(self) -> None:
        idx = _index("hello world")
        assert reconstruct_tokens(idx, "999") == []

    def test_handles_repeated_tokens(self) -> None:
        idx = _index("foo bar foo bar foo")
        assert reconstruct_tokens(idx, "0") == [
            "foo", "bar", "foo", "bar", "foo"
        ]


# ------------------------------------------------------- generate_snippet


class TestGenerateSnippet:
    def test_marks_query_terms(self) -> None:
        idx = _index("the quick brown fox")
        snippet = generate_snippet(idx, "0", ["fox"])
        assert "**fox**" in snippet
        # Non-matches stay unmarked.
        assert "the" in snippet
        assert "**the**" not in snippet

    def test_short_doc_returned_in_full(self) -> None:
        idx = _index("foo bar")
        snippet = generate_snippet(idx, "0", ["foo"], window=10)
        assert snippet == "**foo** bar"

    def test_window_caps_length(self) -> None:
        idx = _index(" ".join(f"word{i}" for i in range(50)))
        snippet = generate_snippet(idx, "0", ["word25"], window=5)
        # Up to 5 tokens between the optional ellipses.
        body = snippet.replace("... ", "").replace(" ...", "")
        assert len(body.split()) <= 5
        assert "**word25**" in body

    def test_picks_window_with_most_hits(self) -> None:
        # Place query hits in the latter half of the doc; the chosen
        # window should land there.
        idx = _index(
            " ".join(["filler"] * 20)
            + " alpha beta alpha beta alpha "
            + " ".join(["tail"] * 20)
        )
        snippet = generate_snippet(
            idx, "0", ["alpha", "beta"], window=5
        )
        # The dense region has 5 hits in 5 slots.
        assert snippet.count("**alpha**") + snippet.count("**beta**") >= 5

    def test_unknown_doc_id_yields_empty_string(self) -> None:
        idx = _index("hello world")
        assert generate_snippet(idx, "999", ["hello"]) == ""

    def test_zero_or_negative_window_yields_empty(self) -> None:
        idx = _index("hello world")
        assert generate_snippet(idx, "0", ["hello"], window=0) == ""
        assert generate_snippet(idx, "0", ["hello"], window=-1) == ""

    def test_default_window_is_reasonable(self) -> None:
        # Sanity-check the public default.
        assert 5 <= DEFAULT_WINDOW <= 50

    def test_no_matches_still_returns_an_excerpt(self) -> None:
        # A word not in the doc still produces a valid snippet (just no
        # marked terms).
        idx = _index("alpha beta gamma")
        snippet = generate_snippet(idx, "0", ["xyzzy"])
        assert "alpha" in snippet
        assert "**" not in snippet  # nothing to mark
