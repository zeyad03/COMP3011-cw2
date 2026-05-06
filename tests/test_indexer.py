"""Tests for the inverted indexer."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.crawler import Page
from src.indexer import INDEX_VERSION, build_index


def _page(text: str, *, url: str = "https://x.com/a", title: str = "") -> Page:
    return Page(url=url, title=title, text=text)


class TestEmptyAndSingleDoc:
    def test_empty_pages_list_produces_empty_index(self) -> None:
        idx = build_index([])
        assert idx["meta"]["num_docs"] == 0
        assert idx["meta"]["num_terms"] == 0
        assert idx["meta"]["total_tokens"] == 0
        assert idx["docs"] == {}
        assert idx["terms"] == {}

    def test_single_word_page(self) -> None:
        idx = build_index([_page("hello")])
        assert idx["meta"]["num_docs"] == 1
        assert idx["meta"]["num_terms"] == 1
        assert "hello" in idx["terms"]

        entry = idx["terms"]["hello"]
        assert entry["df"] == 1
        assert entry["cf"] == 1
        assert entry["postings"]["0"]["tf"] == 1
        assert entry["postings"]["0"]["positions"] == [0]


class TestPositionsAndFrequency:
    def test_repeated_word_records_positions(self) -> None:
        idx = build_index([_page("the cat sat on the mat")])
        the_postings = idx["terms"]["the"]["postings"]["0"]
        assert the_postings["tf"] == 2
        assert the_postings["positions"] == [0, 4]

    def test_collection_frequency_sums_across_docs(self) -> None:
        pages = [
            _page("foo foo bar", url="https://x.com/a"),
            _page("bar bar foo", url="https://x.com/b"),
        ]
        idx = build_index(pages)
        assert idx["terms"]["foo"]["cf"] == 3
        assert idx["terms"]["bar"]["cf"] == 3

    def test_document_frequency_counts_distinct_pages(self) -> None:
        pages = [
            _page("foo foo", url="https://x.com/a"),
            _page("bar", url="https://x.com/b"),
            _page("foo bar", url="https://x.com/c"),
        ]
        idx = build_index(pages)
        assert idx["terms"]["foo"]["df"] == 2
        assert idx["terms"]["bar"]["df"] == 2

    def test_postings_per_doc_are_present(self) -> None:
        pages = [
            _page("foo bar", url="https://x.com/a"),
            _page("foo baz", url="https://x.com/b"),
        ]
        idx = build_index(pages)
        assert set(idx["terms"]["foo"]["postings"].keys()) == {"0", "1"}


class TestDocsEntries:
    def test_doc_entry_records_url_title_length(self) -> None:
        # Title is concatenated with body text, so length includes both.
        pages = [_page("hello world", url="https://x.com/a", title="Greeting")]
        idx = build_index(pages)
        doc = idx["docs"]["0"]
        assert doc["url"] == "https://x.com/a"
        assert doc["title"] == "Greeting"
        assert doc["length"] == 3  # ["greeting", "hello", "world"]

    def test_doc_ids_are_sequential_strings(self) -> None:
        pages = [
            _page("a", url="https://x.com/1"),
            _page("b", url="https://x.com/2"),
            _page("c", url="https://x.com/3"),
        ]
        idx = build_index(pages)
        assert list(idx["docs"].keys()) == ["0", "1", "2"]


class TestTitleSearchability:
    def test_title_words_are_indexed(self) -> None:
        idx = build_index([_page("body content", title="Special")])
        assert "special" in idx["terms"]

    def test_title_lowercased_via_tokenizer(self) -> None:
        idx = build_index([_page("body", title="UPPER")])
        assert "upper" in idx["terms"]
        assert "UPPER" not in idx["terms"]


class TestMeta:
    def test_meta_includes_version(self) -> None:
        idx = build_index([_page("hi")])
        assert idx["meta"]["version"] == INDEX_VERSION

    def test_meta_total_tokens_is_sum_of_doc_lengths(self) -> None:
        idx = build_index([_page("a b c"), _page("d e")])
        assert idx["meta"]["total_tokens"] == 5

    def test_meta_num_docs_and_terms(self) -> None:
        idx = build_index([_page("a b"), _page("b c")])
        assert idx["meta"]["num_docs"] == 2
        assert idx["meta"]["num_terms"] == 3  # a, b, c

    def test_meta_crawled_at_is_iso8601(self) -> None:
        idx = build_index([_page("hi")])
        # Should parse without error.
        datetime.fromisoformat(idx["meta"]["crawled_at"])


class TestEmptyPageHandling:
    def test_empty_text_produces_doc_with_zero_length(self) -> None:
        idx = build_index([_page("")])
        assert idx["meta"]["num_docs"] == 1
        assert idx["docs"]["0"]["length"] == 0
        assert idx["terms"] == {}

    def test_punctuation_only_page_yields_no_terms(self) -> None:
        idx = build_index([_page("...!?", title="!@#")])
        assert idx["docs"]["0"]["length"] == 0
        assert idx["terms"] == {}


class TestDeterminism:
    def test_same_input_produces_same_index_excluding_timestamp(self) -> None:
        pages = [_page("hello world"), _page("foo bar")]
        idx_a = build_index(pages)
        idx_b = build_index(pages)
        for key in ("docs", "terms"):
            assert idx_a[key] == idx_b[key]
        for meta_key in ("num_docs", "num_terms", "total_tokens", "version"):
            assert idx_a["meta"][meta_key] == idx_b["meta"][meta_key]
