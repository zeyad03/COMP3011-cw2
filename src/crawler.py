"""Polite BFS web crawler.

Crawls a single host starting from a seed URL. Honours a configurable
politeness delay between successive HTTP requests, retries transient
failures with exponential backoff, and respects ``robots.txt``.

The crawler is decoupled from the indexer: it returns a list of
:class:`Page` records (URL, title, plain text). The indexer is then
responsible for tokenisation and inverted-index construction.

The default politeness window is six seconds, matching the requirement in
the COMP3011 CW2 brief. ``Crawler`` exposes the delay as a constructor
argument so tests can use a smaller value (and patch :func:`time.sleep`).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_DELAY: float = 6.0
DEFAULT_USER_AGENT: str = (
    "COMP3011-CW2-SearchEngine/1.0 (+educational; University of Leeds)"
)
DEFAULT_TIMEOUT: float = 10.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_BACKOFF_BASE: float = 2.0


@dataclass(frozen=True)
class Page:
    """A single crawled page."""

    url: str
    title: str
    text: str


class CrawlError(Exception):
    """Raised when a single page could not be fetched after retries."""


class Crawler:
    """Polite BFS crawler restricted to the seed URL's host.

    Args:
        delay: Minimum seconds between successive requests to the host.
        user_agent: ``User-Agent`` header sent on every request.
        timeout: Per-request timeout in seconds.
        max_pages: Optional hard cap on pages crawled. ``None`` = unlimited.
        max_retries: Number of attempts per URL before giving up.
        backoff_base: Exponent base for retry backoff (``base ** attempt``).
        respect_robots: If true, fetch and honour ``/robots.txt``.
        session: Optional pre-configured :class:`requests.Session`.
    """

    def __init__(
        self,
        *,
        delay: float = DEFAULT_DELAY,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        max_pages: int | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        respect_robots: bool = True,
        session: requests.Session | None = None,
        verbose: bool = False,
    ) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self._delay = delay
        self._user_agent = user_agent
        self._timeout = timeout
        self._max_pages = max_pages
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._respect_robots = respect_robots
        self._session = session if session is not None else requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._verbose = verbose
        self._last_request_at: float | None = None

    def crawl(self, start_url: str) -> list[Page]:
        """Crawl the host of *start_url* breadth-first and return pages."""
        host = urlparse(start_url).netloc
        if not host:
            raise ValueError(f"start_url has no host: {start_url!r}")

        robots = self._load_robots(start_url) if self._respect_robots else None

        visited: set[str] = set()
        pages: list[Page] = []
        frontier: deque[str] = deque([self._normalise(start_url)])

        while frontier:
            if self._max_pages is not None and len(pages) >= self._max_pages:
                break
            url = frontier.popleft()
            if url in visited:
                continue
            visited.add(url)

            if robots is not None and not robots.can_fetch(self._user_agent, url):
                logger.info("robots.txt disallows %s; skipping", url)
                continue

            try:
                html = self._fetch(url)
            except CrawlError as err:
                logger.warning("giving up on %s: %s", url, err)
                continue

            title, text = self._extract_text(html)
            pages.append(Page(url=url, title=title, text=text))
            if self._verbose:
                print(f"  [{len(pages):>3}] {url}", flush=True)

            for link in self._extract_links(html, base_url=url):
                if urlparse(link).netloc != host:
                    continue
                norm = self._normalise(link)
                if norm not in visited:
                    frontier.append(norm)

        return pages

    def _fetch(self, url: str) -> str:
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            self._wait_for_politeness()
            try:
                response = self._session.get(url, timeout=self._timeout)
                self._last_request_at = time.monotonic()
                if response.status_code >= 500:
                    last_err = CrawlError(f"HTTP {response.status_code}")
                    self._backoff(attempt)
                    continue
                response.raise_for_status()
                return str(response.text)
            except requests.RequestException as err:
                last_err = err
                self._backoff(attempt)
        raise CrawlError(
            f"failed to fetch {url} after {self._max_retries} attempts: {last_err}"
        )

    def _wait_for_politeness(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

    def _backoff(self, attempt: int) -> None:
        time.sleep(self._backoff_base**attempt)

    def _load_robots(self, start_url: str) -> RobotFileParser | None:
        parsed = urlparse(start_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            self._wait_for_politeness()
            response = self._session.get(robots_url, timeout=self._timeout)
            self._last_request_at = time.monotonic()
        except requests.RequestException as err:
            logger.warning("could not load robots.txt for %s: %s", parsed.netloc, err)
            return None
        if response.status_code != 200:
            return None
        rp.parse(response.text.splitlines())
        return rp

    @staticmethod
    def _normalise(url: str) -> str:
        """Strip the URL fragment for stable comparison.

        We deliberately keep the trailing slash: the URL we record in the
        ``visited`` set has to match the URL we issue HTTP requests against,
        and a server may treat ``/foo`` and ``/foo/`` as different resources.
        """
        defragged, _ = urldefrag(url)
        return defragged

    @staticmethod
    def _extract_links(html: str, *, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            # bs4 may return AttributeValueList for repeated attributes;
            # coerce to a plain string.
            href = str(tag.get("href", "")).strip()
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue
            absolute = urljoin(base_url, href)
            absolute, _ = urldefrag(absolute)
            links.append(absolute)
        return links

    @staticmethod
    def _extract_text(html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return title, text


def crawl(start_url: str, **kwargs: object) -> list[Page]:
    """Convenience wrapper around :class:`Crawler`."""
    return Crawler(**kwargs).crawl(start_url)  # type: ignore[arg-type]
