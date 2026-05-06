"""Tests for the polite BFS crawler.

All tests run **offline**: HTTP is replaced with :class:`FakeSession` and
:func:`time.sleep` is patched to a no-op via ``mocker``. Tests therefore
finish in milliseconds rather than waiting on the real politeness window.
"""

from __future__ import annotations

from typing import Any

import pytest
import requests

from src.crawler import CrawlError, Crawler, Page


# --------------------------------------------------------------------- helpers


class FakeResponse:
    def __init__(self, text: str = "", status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """In-memory stand-in for :class:`requests.Session`.

    ``pages`` maps a URL to either a string (200 OK with that body) or a
    :class:`FakeResponse` (for status / failure scenarios). Unmapped URLs
    return 404. ``failures`` may map a URL to a list of consecutive
    responses; each ``get`` pops the next one, exercising retry paths.
    """

    def __init__(
        self,
        pages: dict[str, str | FakeResponse],
        *,
        sequences: dict[str, list[FakeResponse]] | None = None,
    ) -> None:
        self.pages = pages
        self.sequences = sequences or {}
        self.calls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, **_kwargs: Any) -> FakeResponse:
        self.calls.append(url)
        if url in self.sequences and self.sequences[url]:
            return self.sequences[url].pop(0)
        if url in self.pages:
            value = self.pages[url]
            return value if isinstance(value, FakeResponse) else FakeResponse(value)
        return FakeResponse("", status_code=404)


HOME = "https://example.com"


def _link(href: str) -> str:
    return f'<a href="{href}">link</a>'


def _page(title: str, body: str = "") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _no_real_sleep(mocker: Any) -> None:
    """Patch ``time.sleep`` in the crawler module to a no-op for every test."""
    mocker.patch("src.crawler.time.sleep")


@pytest.fixture
def two_page_session() -> FakeSession:
    return FakeSession(
        {
            f"{HOME}/": _page("Home", _link("/about") + _link("https://other.com/x")),
            f"{HOME}/about": _page("About", "some content"),
        }
    )


# ---------------------------------------------------------------------- tests


class TestNormalise:
    def test_strips_fragment(self) -> None:
        assert Crawler._normalise("https://x.com/p#frag") == "https://x.com/p"

    def test_keeps_trailing_slash(self) -> None:
        # Trailing slashes are kept so visited-set keys match the requested URL.
        assert Crawler._normalise("https://x.com/p/") == "https://x.com/p/"

    def test_idempotent(self) -> None:
        url = "https://x.com/p"
        assert Crawler._normalise(Crawler._normalise(url)) == Crawler._normalise(url)


class TestExtractLinks:
    def test_returns_absolute_urls(self) -> None:
        html = _page("h", _link("/a") + _link("https://other.com/b"))
        links = Crawler._extract_links(html, base_url=f"{HOME}/")
        assert f"{HOME}/a" in links
        assert "https://other.com/b" in links

    def test_filters_pseudo_protocols(self) -> None:
        html = _page(
            "h",
            _link("javascript:void(0)") + _link("mailto:x@y") + _link("tel:123"),
        )
        assert Crawler._extract_links(html, base_url=HOME) == []

    def test_filters_pure_fragment(self) -> None:
        html = _page("h", _link("#section"))
        assert Crawler._extract_links(html, base_url=HOME) == []


class TestExtractText:
    def test_removes_script_and_style(self) -> None:
        html = (
            "<html><head><title>T</title>"
            "<style>x{}</style></head>"
            "<body><script>alert(1)</script>hello world</body></html>"
        )
        title, text = Crawler._extract_text(html)
        assert title == "T"
        assert "alert" not in text
        assert "hello world" in text

    def test_handles_missing_title(self) -> None:
        title, text = Crawler._extract_text("<html><body>just text</body></html>")
        assert title == ""
        assert "just text" in text


class TestCrawlBasics:
    def test_returns_pages_with_url_title_text(
        self, two_page_session: FakeSession
    ) -> None:
        crawler = Crawler(delay=0, session=two_page_session, respect_robots=False)
        pages = crawler.crawl(f"{HOME}/")
        assert len(pages) == 2
        urls = {p.url for p in pages}
        assert f"{HOME}" in urls or f"{HOME}/" in urls
        about = next(p for p in pages if p.url.endswith("/about"))
        assert about.title == "About"
        assert "some content" in about.text

    def test_skips_external_domains(self, two_page_session: FakeSession) -> None:
        crawler = Crawler(delay=0, session=two_page_session, respect_robots=False)
        pages = crawler.crawl(f"{HOME}/")
        assert all(p.url.startswith(HOME) for p in pages)
        assert all("other.com" not in c for c in two_page_session.calls)

    def test_avoids_loops_with_visited_set(self) -> None:
        # Two pages link to each other; crawler must visit each once.
        session = FakeSession(
            {
                f"{HOME}/a": _page("A", _link("/b") + _link("/a")),
                f"{HOME}/b": _page("B", _link("/a")),
            }
        )
        crawler = Crawler(delay=0, session=session, respect_robots=False)
        pages = crawler.crawl(f"{HOME}/a")
        assert len(pages) == 2
        # /a fetched once, /b fetched once (not counting any 200 retries).
        assert session.calls.count(f"{HOME}/a") == 1
        assert session.calls.count(f"{HOME}/b") == 1

    def test_max_pages_caps_results(self) -> None:
        session = FakeSession(
            {
                f"{HOME}/p{i}": _page(f"P{i}", _link(f"/p{i + 1}"))
                for i in range(10)
            }
        )
        crawler = Crawler(
            delay=0, session=session, respect_robots=False, max_pages=3
        )
        pages = crawler.crawl(f"{HOME}/p0")
        assert len(pages) == 3


class TestPoliteness:
    def test_sleeps_between_successive_requests(
        self, two_page_session: FakeSession, mocker: Any
    ) -> None:
        # Replace time.sleep with a recording stub. monotonic increases by 0
        # so _wait_for_politeness sees zero elapsed and sleeps the full delay.
        sleep = mocker.patch("src.crawler.time.sleep")
        mocker.patch("src.crawler.time.monotonic", return_value=0.0)

        crawler = Crawler(delay=6.0, session=two_page_session, respect_robots=False)
        crawler.crawl(f"{HOME}/")

        sleep_args = [call.args[0] for call in sleep.call_args_list]
        # At least one sleep should have been ≥ the configured delay.
        assert any(arg >= 6.0 for arg in sleep_args), sleep_args

    def test_first_request_does_not_block(self, mocker: Any) -> None:
        sleep = mocker.patch("src.crawler.time.sleep")
        session = FakeSession({f"{HOME}/": _page("Home", "")})

        Crawler(delay=6.0, session=session, respect_robots=False).crawl(f"{HOME}/")

        # No sleep should have a delay >= 6 (only a single request was made).
        large_sleeps = [c for c in sleep.call_args_list if c.args and c.args[0] >= 6]
        assert large_sleeps == []


class TestRetries:
    def test_retries_on_5xx(self) -> None:
        session = FakeSession(
            {f"{HOME}/": _page("Home", "")},
            sequences={
                f"{HOME}/": [
                    FakeResponse("", status_code=503),
                    FakeResponse(_page("Home", ""), status_code=200),
                ]
            },
        )
        crawler = Crawler(
            delay=0, session=session, respect_robots=False, max_retries=3
        )
        pages = crawler.crawl(f"{HOME}/")
        assert len(pages) == 1
        assert session.calls.count(f"{HOME}/") == 2

    def test_gives_up_after_max_retries_without_crashing(self, caplog: Any) -> None:
        session = FakeSession(
            {f"{HOME}/start": _page("Start", _link("/dead"))},
            sequences={
                f"{HOME}/dead": [FakeResponse("", status_code=503)] * 5,
            },
        )
        crawler = Crawler(
            delay=0, session=session, respect_robots=False, max_retries=2
        )
        pages = crawler.crawl(f"{HOME}/start")

        # Start succeeded; dead URL exhausted retries and was skipped.
        assert {p.url for p in pages} == {f"{HOME}/start"}
        assert session.calls.count(f"{HOME}/dead") == 2

    def test_invalid_delay_rejected(self) -> None:
        with pytest.raises(ValueError):
            Crawler(delay=-1.0)

    def test_invalid_max_retries_rejected(self) -> None:
        with pytest.raises(ValueError):
            Crawler(max_retries=0)

    def test_start_url_without_host_rejected(self) -> None:
        with pytest.raises(ValueError):
            Crawler(respect_robots=False).crawl("not-a-url")


class TestRobots:
    def test_disallowed_url_is_skipped(self) -> None:
        robots_txt = "User-agent: *\nDisallow: /private"
        session = FakeSession(
            {
                f"{HOME}/robots.txt": robots_txt,
                f"{HOME}/": _page("Home", _link("/private") + _link("/public")),
                f"{HOME}/public": _page("Public", "ok"),
                f"{HOME}/private": _page("Private", "secret"),
            }
        )
        crawler = Crawler(delay=0, session=session, respect_robots=True)
        pages = crawler.crawl(f"{HOME}/")
        urls = {p.url for p in pages}
        assert any(u.endswith("/public") for u in urls)
        assert not any(u.endswith("/private") for u in urls)

    def test_robots_404_falls_back_to_no_restrictions(self) -> None:
        # robots.txt returns 404 — crawler should proceed with no rules.
        session = FakeSession(
            {
                f"{HOME}/robots.txt": FakeResponse("", status_code=404),
                f"{HOME}/": _page("Home", _link("/private")),
                f"{HOME}/private": _page("Private", "ok"),
            }
        )
        crawler = Crawler(delay=0, session=session, respect_robots=True)
        pages = crawler.crawl(f"{HOME}/")
        urls = {p.url for p in pages}
        # /private should now be reachable (no robots restriction applied).
        assert any(u.endswith("/private") for u in urls)

    def test_robots_request_exception_falls_back(self, mocker: Any) -> None:
        # A network error during robots.txt fetch must not block the crawl.
        session = FakeSession({f"{HOME}/": _page("Home", "")})

        def _flaky_get(url: str, **_kw: Any) -> FakeResponse:
            if url.endswith("/robots.txt"):
                raise requests.ConnectionError("dns failed")
            return FakeResponse(_page("Home", ""))

        mocker.patch.object(session, "get", side_effect=_flaky_get)
        crawler = Crawler(delay=0, session=session, respect_robots=True)
        pages = crawler.crawl(f"{HOME}/")
        # The crawl proceeded despite the robots.txt failure.
        assert len(pages) == 1


class TestModuleLevelHelper:
    def test_crawl_function_delegates_to_class(self) -> None:
        session = FakeSession({f"{HOME}/": _page("Home", "")})
        from src.crawler import crawl

        pages = crawl(
            f"{HOME}/", delay=0, session=session, respect_robots=False
        )
        assert isinstance(pages[0], Page)


class TestCrawlErrorRaising:
    def test_crawl_error_message(self) -> None:
        err = CrawlError("boom")
        assert str(err) == "boom"
