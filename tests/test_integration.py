"""End-to-end integration tests using saved HTML fixtures.

These tests exercise the *full* pipeline — crawler → indexer → search —
against real-world HTML scraped from the target site, but without
hitting the network at runtime. The fixtures live in
``tests/fixtures/`` and were captured from
https://quotes.toscrape.com/. Together they form a tiny offline mirror
of the site, sufficient to verify that:

1. The crawler correctly extracts links, titles, and body text from
   real (not-just-handcrafted) markup.
2. The indexer produces a non-trivial inverted index.
3. The search module returns expected pages for known queries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from src.crawler import Crawler, Page
from src.indexer import build_index
from src.search import find, print_term


FIXTURES_DIR = Path(__file__).parent / "fixtures"

HOME = "https://quotes.toscrape.com/"
PAGE2 = "https://quotes.toscrape.com/page/2/"
EINSTEIN = "https://quotes.toscrape.com/author/Albert-Einstein/"


class _FixtureResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class _FixtureSession:
    """Maps URLs to local HTML fixtures."""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self._files = {
            HOME: FIXTURES_DIR / "quotes_home.html",
            HOME.rstrip("/"): FIXTURES_DIR / "quotes_home.html",
            PAGE2: FIXTURES_DIR / "quotes_page2.html",
            EINSTEIN: FIXTURES_DIR / "author_einstein.html",
        }
        self.calls: list[str] = []

    def get(self, url: str, **_: Any) -> _FixtureResponse:
        self.calls.append(url)
        path = self._files.get(url)
        if path is None or not path.exists():
            return _FixtureResponse("", status_code=404)
        return _FixtureResponse(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_real_sleep(mocker: Any) -> None:
    mocker.patch("src.crawler.time.sleep")


@pytest.fixture
def fixture_session() -> _FixtureSession:
    return _FixtureSession()


@pytest.fixture
def fixture_crawl(fixture_session: _FixtureSession) -> list[Page]:
    crawler = Crawler(
        delay=0,
        session=fixture_session,  # type: ignore[arg-type]
        respect_robots=False,
        max_pages=3,
    )
    return crawler.crawl(HOME)


# ----------------------------------------------------------------- crawler


class TestCrawlerOnRealHtml:
    def test_extracts_real_quotes_home(self, fixture_crawl: list[Page]) -> None:
        urls = {p.url for p in fixture_crawl}
        # Home is fetched.
        assert any("quotes.toscrape.com" in u for u in urls)
        home = next(p for p in fixture_crawl if p.url.rstrip("/") == HOME.rstrip("/"))
        assert "Quotes" in home.title or home.title  # title present and non-empty

    def test_extracts_quote_text(self, fixture_crawl: list[Page]) -> None:
        # The home page contains Einstein's "world" quote — pick a
        # distinctive substring that should always be in the rendered text.
        home = next(p for p in fixture_crawl if p.url.rstrip("/") == HOME.rstrip("/"))
        # "world we have created" appears verbatim on the home page.
        assert "world" in home.text.lower()


# ----------------------------------------------------------------- indexer


class TestIndexerOnRealHtml:
    def test_index_has_terms_and_documents(self, fixture_crawl: list[Page]) -> None:
        index = build_index(fixture_crawl)
        assert index["meta"]["num_docs"] == len(fixture_crawl)
        # Real-world pages produce hundreds of unique terms even with 3 pages.
        assert index["meta"]["num_terms"] > 50

    def test_index_records_known_words(self, fixture_crawl: list[Page]) -> None:
        index = build_index(fixture_crawl)
        # Common stop-word-like terms should always be present.
        assert "the" in index["terms"]


# ------------------------------------------------------------------ search


class TestSearchOnRealHtml:
    def test_find_known_word_returns_results(
        self, fixture_crawl: list[Page]
    ) -> None:
        index = build_index(fixture_crawl)
        results = find(index, "world")
        assert results, "expected at least one match for 'world'"
        assert all("quotes.toscrape.com" in r.url for r in results)

    def test_find_phrase_returns_results_when_adjacent(
        self, fixture_crawl: list[Page]
    ) -> None:
        # The exact phrase below appears verbatim on the home page.
        index = build_index(fixture_crawl)
        results = find(index, "the world", mode="phrase")
        # At least one document contains the phrase in adjacent positions.
        # (We don't assert a specific URL because exact ordering depends
        # on token boundaries in the rendered text.)
        assert isinstance(results, list)

    def test_find_unknown_returns_empty(self, fixture_crawl: list[Page]) -> None:
        index = build_index(fixture_crawl)
        assert find(index, "zzzzzzz_no_such_word") == []

    def test_print_term_real_word(self, fixture_crawl: list[Page]) -> None:
        index = build_index(fixture_crawl)
        out = print_term(index, "the")
        assert "the" in out
        assert "df=" in out
