"""Property-based tests for module invariants.

Where the example-based tests in ``test_*.py`` check specific inputs,
these tests use :mod:`hypothesis` to fuzz a property over many random
inputs. The properties below capture invariants that should hold for
*every* well-formed input; a failing case is a real bug.

Properties exercised:

* **Tokeniser**: every emitted token is non-empty, lowercase, and free
  of whitespace; the tokeniser is idempotent on its own joined output.
* **Indexer**: ``df`` equals the number of postings; ``cf`` equals the
  sum of per-doc ``tf``; ``len(positions) == tf``; ``meta.num_docs``
  is consistent with ``docs``.
* **Search**: results are always a list; adding query terms can only
  shrink the result set (AND monotonicity); ``top_k`` is an upper
  bound on the result count.
* **Storage**: ``save`` then ``load`` round-trips an index unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.crawler import Page
from src.indexer import build_index
from src.search import find
from src.storage import load, save
from src.tokenizer import tokenize


# Cap example count so the suite stays fast and deterministic enough
# for CI; broaden locally with HYPOTHESIS_PROFILE=full if needed.
SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------- strategies


# Restrict text to printable BMP characters; tokenisation should still
# handle arbitrary Unicode but the strategy is more useful when
# generated text occasionally contains real words.
_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs", "Po", "Pd"),
        max_codepoint=0x2FF,
    ),
    max_size=120,
)


@st.composite
def _pages(draw, min_size: int = 0, max_size: int = 5) -> list[Page]:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    return [
        Page(
            url=f"https://example.com/p{i}",
            title=draw(_text),
            text=draw(_text),
        )
        for i in range(n)
    ]


# --------------------------------------------------------- tokeniser


class TestTokeniserProperties:
    @SETTINGS
    @given(text=_text)
    def test_every_token_is_non_empty_lowercase_no_whitespace(
        self, text: str
    ) -> None:
        for token in tokenize(text):
            assert token, "tokenizer must not emit empty tokens"
            assert token == token.lower()
            assert not any(c.isspace() for c in token)

    @SETTINGS
    @given(text=_text)
    def test_idempotent_on_joined_output(self, text: str) -> None:
        # Re-tokenising "joined tokens" yields the same list.
        once = tokenize(text)
        twice = tokenize(" ".join(once))
        assert once == twice


# --------------------------------------------------------- indexer


class TestIndexerProperties:
    @SETTINGS
    @given(pages=_pages())
    def test_df_equals_posting_count(self, pages: list[Page]) -> None:
        index = build_index(pages)
        for term, entry in index["terms"].items():
            assert entry["df"] == len(entry["postings"]), term

    @SETTINGS
    @given(pages=_pages())
    def test_cf_equals_sum_of_tf(self, pages: list[Page]) -> None:
        index = build_index(pages)
        for term, entry in index["terms"].items():
            total_tf = sum(p["tf"] for p in entry["postings"].values())
            assert entry["cf"] == total_tf, term

    @SETTINGS
    @given(pages=_pages())
    def test_tf_equals_position_count(self, pages: list[Page]) -> None:
        index = build_index(pages)
        for entry in index["terms"].values():
            for posting in entry["postings"].values():
                assert posting["tf"] == len(posting["positions"])

    @SETTINGS
    @given(pages=_pages())
    def test_meta_num_docs_matches_docs(self, pages: list[Page]) -> None:
        index = build_index(pages)
        assert index["meta"]["num_docs"] == len(index["docs"])

    @SETTINGS
    @given(pages=_pages(min_size=1))
    def test_every_token_in_doc_has_a_posting(
        self, pages: list[Page]
    ) -> None:
        index = build_index(pages)
        for doc_id, doc_meta in index["docs"].items():
            page = pages[int(doc_id)]
            for token in set(tokenize(f"{page.title} {page.text}")):
                assert doc_id in index["terms"][token]["postings"]


# --------------------------------------------------------- search


_query_words = st.lists(
    st.from_regex(r"\A[a-z]{1,8}\Z"), min_size=1, max_size=3
)


class TestSearchProperties:
    @SETTINGS
    @given(pages=_pages(), words=_query_words)
    def test_results_are_always_a_list(
        self, pages: list[Page], words: list[str]
    ) -> None:
        index = build_index(pages)
        results = find(index, " ".join(words))
        assert isinstance(results, list)

    @SETTINGS
    @given(pages=_pages(min_size=1), words=_query_words)
    def test_adding_query_term_shrinks_or_preserves_results(
        self, pages: list[Page], words: list[str]
    ) -> None:
        # AND monotonicity: find(q1 ∪ q2) ⊆ find(q1).
        index = build_index(pages)
        if not words:
            return
        base_query = " ".join(words)
        extended_query = f"{base_query} aaaaaaaa"  # extra likely-unknown term

        base = {r.url for r in find(index, base_query)}
        extended = {r.url for r in find(index, extended_query)}
        assert extended.issubset(base)

    @SETTINGS
    @given(pages=_pages(), words=_query_words, k=st.integers(1, 10))
    def test_top_k_is_an_upper_bound(
        self, pages: list[Page], words: list[str], k: int
    ) -> None:
        index = build_index(pages)
        results = find(index, " ".join(words), top_k=k)
        assert len(results) <= k


# --------------------------------------------------------- storage


class TestStorageProperties:
    @SETTINGS
    @given(pages=_pages())
    def test_save_load_round_trips(
        self, pages: list[Page], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        index = build_index(pages)
        target = tmp_path_factory.mktemp("idx") / "idx.json"
        save(index, target)
        loaded = load(target)
        assert loaded == index
